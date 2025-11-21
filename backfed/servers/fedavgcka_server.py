"""
FedAvgCKA Server Implementation for BackFed.

Based on "Exploiting Layerwise Feature Representation Similarity For Backdoor 
Defence in Federated Learning" by Walter et al. (ESORICS 2024).

This defense uses Centered Kernel Alignment (CKA) to detect malicious clients
by comparing their penultimate layer representations.
"""

import copy
import torch
import numpy as np
from typing import List, Tuple, Dict
from logging import INFO
from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.utils import log
from backfed.const import StateDict, client_id, num_examples
from backfed.servers.fedspectre_utils import (
    linear_cka, 
    get_penultimate_layer_name,
    extract_layer_activations,
    create_root_dataset_loader,
    load_ood_dataset
)


class FedAvgCKAServer(AnomalyDetectionServer):
    """
    FedAvgCKA defense server.
    
    Uses CKA similarity between client models' representations to detect
    backdoor attacks. Clients with low average CKA similarity to other
    clients are considered malicious.
    """
    defense_categories = ["anomaly_detection"]
    
    def __init__(self, 
                 server_config,
                 server_type: str = "fedavgcka",
                 eta: float = 0.5,
                 root_size: int = 16,
                 cka_threshold: float = 0.85,
                 layer_comparison: str = "penultimate",
                 use_ood_root_dataset: bool = False,
                 trim_fraction: float = 0.2,
                 **kwargs):
        """
        Initialize FedAvgCKA server.
        
        Args:
            server_config: Server configuration
            server_type: Type of server
            eta: Learning rate for server update
            root_size: Number of samples in root dataset
            cka_threshold: Threshold for anomaly detection (not used in ranking mode)
            layer_comparison: Which layer to compare ("penultimate", "layer2", "layer3")
            use_ood_root_dataset: Whether to use OOD dataset for root
            trim_fraction: Fraction of clients to exclude (default 0.2 = bottom 20%)
        """
        super().__init__(server_config, server_type, eta, **kwargs)
        
        self.root_size = root_size
        self.cka_threshold = cka_threshold
        self.layer_comparison = layer_comparison
        self.use_ood_root_dataset = use_ood_root_dataset
        self.trim_fraction = trim_fraction
        
        # Create root dataset loader
        self.root_loader = self._create_root_loader()
        
        log(INFO, f"Initialized FedAvgCKA server with root_size={root_size}, trim_fraction={trim_fraction}")
        log(INFO, f"Layer comparison mode: {layer_comparison}")
        if use_ood_root_dataset:
            log(INFO, "Using OOD dataset for root dataset")
    
    def _create_root_loader(self):
        """Create root dataset loader from test set or OOD dataset."""
        ood_dataset = None
        
        if self.use_ood_root_dataset:
            ood_dataset = load_ood_dataset(self.config.dataset, self.config)
        
        return create_root_dataset_loader(
            testset=self.testset,
            root_size=self.root_size,
            batch_size=self.root_size,
            num_workers=self.config.num_workers,
            device=self.device,
            ood_dataset=ood_dataset
        )
    
    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> Tuple[List[int], List[int]]:
        """
        Detect anomalies using CKA similarity.
        
        Args:
            client_updates: List of (client_id, num_examples, model_state_dict)
            
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        if len(client_updates) < 2:
            log(INFO, "Too few clients for CKA comparison, accepting all")
            benign_ids = [cid for cid, _, _ in client_updates]
            return [], benign_ids
        
        try:
            # Reconstruct client models from state dicts
            client_models = {}
            for cid, _, state_dict in client_updates:
                model = copy.deepcopy(self.global_model)
                model.load_state_dict(state_dict, strict=True)
                client_models[cid] = model
            
            # Determine layer to extract
            sample_model = next(iter(client_models.values()))
            layer_name = self._get_layer_name(sample_model)
            
            log(INFO, f"Extracting activations from layer: {layer_name}")
            
            # Extract activations for all clients
            activations = {}
            failed_clients = []
            
            for cid, model in client_models.items():
                try:
                    acts, _ = extract_layer_activations(
                        model=model,
                        data_loader=self.root_loader,
                        layer_name=layer_name,
                        device=self.device
                    )
                    # Convert to torch tensor for CKA computation
                    activations[cid] = torch.from_numpy(acts).float()
                except Exception as e:
                    log(INFO, f"Failed to extract activations for client {cid}: {e}")
                    failed_clients.append(cid)
            
            if len(activations) < 2:
                log(INFO, "Too few clients with valid activations, accepting all")
                benign_ids = [cid for cid, _, _ in client_updates]
                return [], benign_ids
            
            # Compute pairwise CKA scores and rank clients
            selected_clients, excluded_clients, cka_scores = self._rank_clients_by_cka(activations)
            
            # Log results
            log(INFO, f"FedAvgCKA filtering: selected {len(selected_clients)}, excluded {len(excluded_clients)}")
            for cid in excluded_clients:
                if cid in cka_scores:
                    log(INFO, f"  Excluded client {cid} with avg CKA={cka_scores[cid]:.4f}")
            
            return excluded_clients, selected_clients
            
        except Exception as e:
            log(INFO, f"FedAvgCKA detection failed: {e}. Accepting all clients.")
            benign_ids = [cid for cid, _, _ in client_updates]
            return [], benign_ids
    
    def _get_layer_name(self, model):
        """Get layer name based on comparison mode."""
        if self.layer_comparison == "penultimate":
            return get_penultimate_layer_name(model)
        elif self.layer_comparison == "layer2":
            if hasattr(model, 'layer2'):
                return 'layer2'
            elif hasattr(model, 'conv2'):
                return 'conv2'
            else:
                log(INFO, "layer2 not found, falling back to penultimate")
                return get_penultimate_layer_name(model)
        elif self.layer_comparison == "layer3":
            if hasattr(model, 'layer3'):
                return 'layer3'
            elif hasattr(model, 'fc1'):
                return 'fc1'
            else:
                log(INFO, "layer3 not found, falling back to penultimate")
                return get_penultimate_layer_name(model)
        else:
            return get_penultimate_layer_name(model)
    
    def _rank_clients_by_cka(self, activations: Dict[int, torch.Tensor]) -> Tuple[List[int], List[int], Dict[int, float]]:
        """
        Rank clients by average CKA similarity.
        
        Args:
            activations: Dict mapping client_id to activation tensor
            
        Returns:
            (selected_clients, excluded_clients, avg_cka_scores)
        """
        client_ids = list(activations.keys())
        n_clients = len(client_ids)
        
        if n_clients == 1:
            return client_ids, [], {client_ids[0]: 1.0}
        
        # Compute pairwise CKA scores
        cka_matrix = torch.zeros(n_clients, n_clients)
        
        log(INFO, f"Computing pairwise CKA scores for {n_clients} clients...")
        
        for i in range(n_clients):
            for j in range(i, n_clients):
                if i == j:
                    cka_score = 1.0
                else:
                    client_i, client_j = client_ids[i], client_ids[j]
                    # Convert to numpy for linear_cka function
                    acts_i = activations[client_i].numpy()
                    acts_j = activations[client_j].numpy()
                    cka_score = linear_cka(acts_i, acts_j)
                
                cka_matrix[i, j] = cka_score
                cka_matrix[j, i] = cka_score
        
        # Calculate average CKA score for each client (excluding self-similarity)
        avg_cka_scores = {}
        for i, client_id in enumerate(client_ids):
            if n_clients > 1:
                mask = torch.ones(n_clients, dtype=torch.bool)
                mask[i] = False
                avg_score = cka_matrix[i, mask].mean().item()
            else:
                avg_score = 1.0
            avg_cka_scores[client_id] = avg_score
        
        # Sort clients by average CKA score (ascending = most dissimilar first)
        sorted_clients = sorted(client_ids, key=lambda cid: avg_cka_scores[cid])
        
        # Exclude bottom trim_fraction clients
        n_exclude = int(self.trim_fraction * n_clients)
        n_exclude = min(n_exclude, n_clients - 1)  # Keep at least 1 client
        
        excluded_clients = sorted_clients[:n_exclude]
        selected_clients = sorted_clients[n_exclude:]
        
        log(INFO, f"CKA ranking complete: selected {len(selected_clients)}, excluded {len(excluded_clients)}")
        
        return selected_clients, excluded_clients, avg_cka_scores
    
    def __repr__(self) -> str:
        return f"FedAvgCKA(root_size={self.root_size}, trim_fraction={self.trim_fraction}, layer={self.layer_comparison})"

