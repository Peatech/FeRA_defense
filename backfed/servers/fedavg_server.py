"""
FedAvg server implementation for FL.
"""
import torch

from backfed.servers.base_server import BaseServer
from backfed.utils.logging_utils import log
from logging import INFO, WARNING
from typing import List, Tuple
from backfed.const import StateDict, client_id, num_examples

class UnweightedFedAvgServer(BaseServer):
    """
    FedAvg server with equal client weights, following standard FedAvg algorithm.

    Formula: G^{t+1} = (1/m) * sum_{i=1}^{m} L_i^{t+1}
    where G^t: global model, m: num clients, L_i: client model
    """

    def __init__(self, server_config, server_type = "unweighted_fedavg", eta=1.0, **kwargs):
        super(UnweightedFedAvgServer, self).__init__(server_config, server_type, **kwargs)
        self.eta = eta
        log(INFO, f"Initialized UnweightedFedAvg server with eta={eta}")

    def _compute_client_distance(self, client_state: StateDict) -> float:
        """
        Compute L2 distance between client model and global model for differentiable parameters only.
        """
        flatten_weights = []
        
        for name, global_param in self.global_model.named_parameters():
            diff = client_state[name].to(device=self.device, dtype=global_param.dtype) - global_param
            flatten_weights.append(diff.view(-1))

        if not flatten_weights:
            return 0.0
        
        flatten_weights = torch.cat(flatten_weights)
        weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2)
        return weight_diff_norm.item()

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]):
        """
        Aggregate client updates using FedAvg with equal weights.
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False

        num_clients = len(client_updates)

        # Report client-global model distances
        if self.verbose:
            for client_id_val, _, client_state in client_updates:
                distance = self._compute_client_distance(client_state)
                log(INFO, f"Client {client_id_val} has weight diff norm {distance:.4f}")

        # Cumulative model updates with equal weights
        weight = 1 / num_clients
        weight_accumulator = {}
        global_state_dict = self.global_model.state_dict()
        
        # Initialize float accumulators for all parameters (including integer buffers)
        for name, param in global_state_dict.items():
            weight_accumulator[name] = torch.zeros_like(
                param, device=self.device, dtype=torch.float32
            )
        
        for _, _, client_state in client_updates:
            for name, param in client_state.items():
                # Only process parameters that exist in the model
                if any(pattern in name for pattern in self.ignore_weights):
                    continue

                # Convert to float for accumulation
                client_param = param.to(device=self.device, dtype=torch.float32)
                global_param = global_state_dict[name].to(device=self.device, dtype=torch.float32)
                diff = client_param - global_param
                weight_accumulator[name].add_(diff * weight)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True

class WeightedFedAvgServer(BaseServer):
    """
    FedAvg server with client weights proportional to their number of samples.
    """

    def __init__(self, server_config, server_type="weighted_fedavg", eta=1.0, **kwargs):
        super(WeightedFedAvgServer, self).__init__(server_config, server_type, **kwargs)
        self.eta = eta
        log(INFO, f"Initialized Weighted FedAvg server with eta={eta}")

    def _compute_client_distance(self, client_state: StateDict) -> float:
        """
        Compute L2 distance between client model and global model for differentiable parameters only.
        """
        flatten_weights = []
        
        for name, global_param in self.global_model.named_parameters():
            diff = client_state[name].to(device=self.device, dtype=global_param.dtype) - global_param
            flatten_weights.append(diff.view(-1))

        if not flatten_weights:
            return 0.0
        
        flatten_weights = torch.cat(flatten_weights)
        weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2)
        return weight_diff_norm.item()

    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]):
        """
        Aggregate client updates using FedAvg with weights proportional to number of samples.
        """
        if not client_updates:
            return False

        # Report client-global model distances
        if self.verbose:
            for client_id_val, _, client_state in client_updates:
                distance = self._compute_client_distance(client_state)
                log(INFO, f"Client {client_id_val} has weight diff norm {distance:.4f}")

        # Cumulative model updates with weights proportional to number of samples
        weight_accumulator = {
            name: torch.zeros_like(param, device=self.device, dtype=torch.float32)
            for name, param in self.global_model.state_dict().items()
        }

        total_samples = sum(num_samples for _, num_samples, _ in client_updates)
        global_state_dict = self.global_model.state_dict()
        
        # Initialize float accumulators for all parameters (including integer buffers)
        for name, param in global_state_dict.items():
            weight_accumulator[name] = torch.zeros_like(
                param, device=self.device, dtype=torch.float32
            )
        
        for _, num_samples, client_state in client_updates:
            weight = (num_samples / total_samples)
            for name, param in client_state.items():
                # Only process parameters that exist in the model
                if any(pattern in name for pattern in self.ignore_weights):
                    continue

                # Convert to float for accumulation
                client_param = param.to(device=self.device, dtype=torch.float32)
                global_param = global_state_dict[name].to(device=self.device, dtype=torch.float32)
                diff = client_param - global_param
                weight_accumulator[name].add_(diff * weight)

        # Update global model with learning rate
        for name, param in self.global_model.state_dict().items():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            param.data.add_(weight_accumulator[name] * self.eta)
        return True
