"""Implementation of FLARE server for federated learning."""

import math
import torch
import copy

from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from logging import INFO
from backfed.datasets import FL_DataLoader
from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log

def bypass_last_layer(model):
    """Hacky way of separating features and classification head for many models."""
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        layer_cake = list(model.module.children())
    else:
        layer_cake = list(model.children())
        
    last_layer = layer_cake[-1]
    headless_model = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten()).eval()
    return headless_model, last_layer

class FlareServer(RobustAggregationServer):
    """
    FLARE server implementation that uses Maximum Mean Discrepancy (MMD)
    to detect and filter malicious updates.

    This is a hybrid defense that combines anomaly detection (MMD-based detection)
    with robust aggregation (weighted aggregation based on trust scores).
    """

    def __init__(
        self,
        server_config,
        server_type: str = "flare",
        voting_threshold: float = 0.5,
        temperature: float = 1.0,
        eta: float = 0.1,
        m: int = 10, # Number of auxiliary data samples
        aux_class: int = 5, # The class used as auxiliary data
    ):
        self.voting_threshold = voting_threshold
        self.temperature = max(float(temperature), 1e-6)
        self.m = m
        self.aux_class = aux_class
        
        super().__init__(server_config, server_type, eta) # Setup datasets and so on
        
        log(
            INFO,
            "Initialized FLARE server with voting_threshold=%s, temperature=%s",
            voting_threshold,
            self.temperature,
        )
        
    def _prepare_dataset(self):
        """Very hacky. We override the _prepare_dataset function to load auxiliary clean data for the defense."""
        
        self.fl_dataloader = FL_DataLoader(config=self.config)
        if self.config.dataset.upper() in ["REDDIT", "FEMNIST", "SENTIMENT140"]:
            raise NotImplementedError(f"FLARE not implemented for {self.config.dataset} dataset")
        else:
            self.trainset, self.client_data_indices, self.secret_dataset_indices, self.testset = self.fl_dataloader.prepare_dataset() 
        
        self.test_loader = DataLoader(self.testset, 
                            batch_size=self.config.test_batch_size, 
                            num_workers=self.config.num_workers,
                            pin_memory=self.config.pin_memory,
                            shuffle=False
        )
                                    
        # Sample m indices of the auxiliary class from the training set
        chosen_indices = []
        targets = getattr(self.trainset, 'targets', None)
        if targets is None:
            targets = getattr(self.trainset, 'labels', None)
        if targets is not None:
            for idx, label in enumerate(targets):
                if label == self.aux_class:
                    chosen_indices.append(idx)
                    if len(chosen_indices) >= self.m:
                        break
        else:
            # Fallback: try to access label from dataset[idx][1]
            idx = 0
            while len(chosen_indices) < self.m and idx < len(self.trainset):
                sample = self.trainset[idx]
                label = sample[1] if isinstance(sample, (tuple, list)) and len(sample) > 1 else None
                if label == self.aux_class:
                    chosen_indices.append(idx)
                idx += 1
        if len(chosen_indices) < self.m:
            raise ValueError(f"Not enough samples of class {self.aux_class} in the training set.")
        
        if self.normalization:
            self.aux_inputs = self.normalization(torch.stack([self.trainset[i][0] for i in chosen_indices]).to(self.device))
        else:
            self.aux_inputs = torch.stack([self.trainset[i][0] for i in chosen_indices]).to(self.device)
            
    def _kernel_function(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between two sets of vectors."""
        sigma = 1.0
        return torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2 * sigma**2))

    def _compute_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy between two sets of features."""
        m, n = x.size(0), y.size(0)

        if m == 0 or n == 0:
            return torch.tensor(0.0, device=x.device if m else y.device)

        xx_kernel = self._kernel_function(x, x)
        yy_kernel = self._kernel_function(y, y)
        xy_kernel = self._kernel_function(x, y)

        if m > 1:
            xx_sum = (xx_kernel.sum() - torch.diagonal(xx_kernel).sum()) / (m * (m - 1))
        else:
            xx_sum = torch.tensor(0.0, device=xx_kernel.device)

        if n > 1:
            yy_sum = (yy_kernel.sum() - torch.diagonal(yy_kernel).sum()) / (n * (n - 1))
        else:
            yy_sum = torch.tensor(0.0, device=yy_kernel.device)

        xy_sum = xy_kernel.sum() / (m * n)

        return xx_sum + yy_sum - 2 * xy_sum

    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates using FLARE mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False

        client_features = []
        for client_id, _, model_update in client_updates:
            # Load client model update into a temporary model
            temp_model = copy.deepcopy(self.global_model)
            temp_model.load_state_dict(model_update)
            temp_model.eval()
            
            # get feature_extractor
            feature_extractor, _ = bypass_last_layer(temp_model)
            feature_extractor.to(self.device).eval()
            
            with torch.no_grad():
                features = feature_extractor(self.aux_inputs)
                client_features.append(features.cpu())
        
        num_clients = len(client_updates)
        distance_matrix = torch.zeros((num_clients, num_clients), dtype=torch.float32)
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                mmd_score = self._compute_mmd(client_features[i], client_features[j]).item()
                distance_matrix[i, j] = distance_matrix[j, i] = mmd_score
        
        if self.verbose:
            log(INFO, "FLARE distances: %s", distance_matrix.tolist())

        neighbor_count = max(1, int(math.ceil(self.voting_threshold * (num_clients - 1))))
        vote_counter = torch.zeros(num_clients, dtype=torch.float32)

        for i in range(num_clients):
            distances = distance_matrix[i]
            sorted_indices = torch.argsort(distances)
            neighbor_indices = [idx.item() for idx in sorted_indices if idx != i][:neighbor_count]
            for neighbor in neighbor_indices:
                vote_counter[neighbor] += 1

        trust_scores = torch.softmax(vote_counter / self.temperature, dim=0)
        
        if self.verbose:
            log(INFO, "FLARE trust scores: %s", trust_scores.tolist())

        global_state_dict = self.global_model.state_dict()
        weight_accumulator: Dict[str, torch.Tensor] = {}

        for name, param in global_state_dict.items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            weight_accumulator[name] = torch.zeros_like(
                param, device=self.device, dtype=torch.float32
            )

        for weight, (_, _, client_state) in zip(trust_scores.tolist(), client_updates):
            trust_weight = float(weight)
            for name, param in client_state.items():
                if any(pattern in name for pattern in self.ignore_weights):
                    continue
                client_param = param.to(device=self.device, dtype=torch.float32)
                global_param = global_state_dict[name].to(device=self.device, dtype=torch.float32)
                diff = client_param - global_param
                weight_accumulator[name].add_(diff * trust_weight)

        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)

        return True
