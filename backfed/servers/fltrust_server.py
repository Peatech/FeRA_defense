"""
Implementation of FLTrust server for federated learning.
"""

import torch
import torch.nn.functional as F
import copy

from typing import Dict, List, Tuple
from logging import INFO
from torch.utils.data import DataLoader, TensorDataset
from backfed.datasets import FL_DataLoader
from backfed.servers.defense_categories import RobustAggregationServer
from backfed.utils.logging_utils import log
from hydra.utils import instantiate

class FLTrustServer(RobustAggregationServer):
    """
    FLTrust server implementation that uses cosine similarity with trusted data
    to assign trust scores to client updates.
    """

    def __init__(self, 
        server_config, 
        server_type = "fltrust", 
        eta: float = 0.1,
        m: int = 100, # Number of samples in server's root dataset
    ):
        self.m = m
        
        super().__init__(server_config, server_type, eta) # Setup datasets and so on
        
        self.global_lr = self.config.client_config.lr
        self.global_epochs = 1 # Follow original paper
        self.server_optimizer = instantiate(self.config.client_config.optimizer, params=self.global_model.parameters())

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
                                    
        if self.m > len(self.trainset):
            raise ValueError(f"FLTrust: m ({self.m}) is larger than training set size ({len(self.trainset)}), reducing m to {len(self.trainset)}")

        random_indices = torch.randperm(len(self.trainset))[:self.m]

        self.server_root_data = TensorDataset(torch.stack([self.normalization(self.trainset[i][0]) for i in random_indices]),
                                                torch.tensor([self.trainset[i][1] for i in random_indices]))
        self.server_dataloader = DataLoader(self.server_root_data, 
                                    batch_size=self.config.client_config.batch_size, # Follo
                                    shuffle=False, 
                                    num_workers=self.config.num_workers,
                                    pin_memory=self.config.pin_memory,
                                )

    def _central_update(self):
        """Perform update on the server's root dataset to obtain the central update."""
        ref_model = copy.deepcopy(self.global_model)
        ref_model.to(self.device)
        ref_model.train()
        
        loss_func = torch.nn.CrossEntropyLoss()
        for epoch in range(self.global_epochs):
            for data, label in self.server_dataloader:
                data, label = data.to(self.device), label.to(self.device)
                self.server_optimizer.zero_grad()
                preds = ref_model(data)
                loss = loss_func(preds, label)
                loss.backward()
                self.server_optimizer.step()
        
        return self._parameters_dict_to_vector(ref_model.state_dict()) - self._parameters_dict_to_vector(self.global_model.state_dict())
    
    def _parameters_dict_to_vector(self, net_dict: Dict) -> torch.Tensor:
        """Convert parameters dictionary to flat vector, excluding batch norm parameters."""
        vec = []
        for key, param in net_dict.items():
            if any(x in key for x in ['num_batches_tracked', 'running_mean', 'running_var']):
                continue
            vec.append(param.reshape(-1))
        return torch.cat(vec)
    
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates using FLTrust mechanism.

        Args:
            client_updates: List of (client_id, num_examples, model_update)
        Returns:
            True if aggregation was successful, False otherwise
        """
        if len(client_updates) == 0:
            return False
        
        central_update = self._central_update()
        central_norm = torch.linalg.norm(central_update)

        score_list = []
        total_score = 0
        sum_parameters = {}

        global_vector = self._parameters_dict_to_vector(self.global_model.state_dict())
        for _, _, local_update in client_updates:
            # Convert local update to vector
            local_vector = self._parameters_dict_to_vector(local_update) - global_vector

            # Calculate cosine similarity and trust score
            client_cos = F.cosine_similarity(central_update, local_vector, dim=0)
            client_cos = max(client_cos.item(), 0) # ReLU
            local_norm = torch.linalg.norm(local_vector)
            client_norm_ratio = central_norm / (local_norm + 1e-12)

            score_list.append(client_cos)
            total_score += client_cos

            # Accumulate weighted updates
            for key, param in local_update.items():
                if key not in sum_parameters:
                    sum_parameters[key] = client_cos * client_norm_ratio * param.clone().to(self.device)
                else:
                    sum_parameters[key].add_(client_cos * client_norm_ratio * param.to(self.device))

        if self.verbose:
            log(INFO, f"FLTrust scores: {score_list}")

        # If all scores are 0, return current global model
        if total_score == 0:
            log(INFO, "FLTrust: All trust scores are 0, keeping current model")
            return False

        # Update global model parameters in-place
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            if name in sum_parameters:
                update = (sum_parameters[name] / total_score)
                param.data.add_(update * self.eta)

        return True
    
