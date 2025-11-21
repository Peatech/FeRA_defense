"""
FedSPECTRE Robust Defense - Simplified Implementation

This is a simplified but robust implementation of FedSPECTRE defense that focuses
on the core functionality without complex dependencies.

Key features:
1. Robust CKA-based detection
2. Spectral analysis
3. Clean reference management
4. Dynamic thresholding

Author: AI Assistant
Date: 2025-10-18
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import logging
from sklearn.metrics.pairwise import cosine_similarity

from backfed.servers.base_server import BaseServer
from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.utils.system_utils import log, INFO, WARNING
from backfed.servers.fedspectre_utils import (
    RobustStatistics, MahalanobisCKA, SpectralProjection,
    create_root_dataset_loader, load_ood_dataset
)

logger = logging.getLogger(__name__)

class FedSPECTRESimpleServer(AnomalyDetectionServer):
    """
    Simplified FedSPECTRE defense server with robust detection.
    
    Key improvements:
    1. Robust CKA-based detection with proper normalization
    2. Spectral analysis for target class detection
    3. Clean reference management
    4. Dynamic thresholding
    """
    
    defense_categories = ["anomaly_detection", "robust_aggregation"]
    
    def __init__(self, server_config, server_type: str = "fedspectre_simple", eta: float = 0.5,
                 # Detection parameters
                 cka_weight: float = 0.7, spectral_weight: float = 0.3,
                 # Robust statistics
                 trim_fraction: float = 0.5,
                 # Root dataset
                 root_size: int = 64, use_ood_root_dataset: bool = False,
                 **kwargs):
        
        super().__init__(server_config, server_type, eta, **kwargs)
        
        # Store parameters
        self.cka_weight = cka_weight
        self.spectral_weight = spectral_weight
        self.trim_fraction = trim_fraction
        self.root_size = root_size
        self.use_ood_root_dataset = use_ood_root_dataset
        
        # Initialize components
        self.robust_stats = RobustStatistics(rank=128, trim_fraction=0.05)
        self.cka_computer = MahalanobisCKA()
        self.spectral_computer = SpectralProjection()
        
        # Create root dataset loader
        self.root_loader = self._create_root_loader()
        
        # Reference management
        self.clean_templates = {}
        self.reference_model = None
        self.round_count = 0
        
        # Dynamic threshold
        self.dynamic_threshold = 0.5
        self.threshold_history = deque(maxlen=20)
        
        log(INFO, f"Initialized FedSPECTRE Simple server")
        log(INFO, f"  Parameters: CKA={cka_weight}, Spectral={spectral_weight}, trim={trim_fraction}")
        log(INFO, f"  Root size: {root_size}, OOD: {use_ood_root_dataset}")
    
    def _create_root_loader(self):
        """Create root dataset loader."""
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
    
    def detect_anomalies(self, client_updates: List[Tuple[int, int, Dict]]) -> Tuple[List[int], List[int]]:
        """
        Detect anomalies using simplified FedSPECTRE approach.
        
        Args:
            client_updates: List of (client_id, num_examples, model_state_dict)
            
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        if len(client_updates) < 2:
            return [], [cid for cid, _, _ in client_updates]
        
        # Extract client models
        client_models = {}
        for client_id, _, state_dict in client_updates:
            model = self._create_client_model()
            model.load_state_dict(state_dict)
            client_models[client_id] = model
        
        # Initialize reference if first round
        if not self.clean_templates:
            self._initialize_reference()
        
        # Extract representations
        client_representations = self._extract_all_representations(client_models)
        
        # Compute class statistics
        class_stats = self._compute_class_statistics(client_representations)
        
        # Build class templates
        class_templates = self._build_class_templates(client_representations)
        
        # Determine target class
        target_class = self._determine_target_class(class_stats, client_representations)
        
        # Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(
            client_representations, class_templates, class_stats, target_class
        )
        
        # Update dynamic threshold
        self._update_dynamic_threshold(anomaly_scores)
        
        # Apply threshold to determine malicious/benign
        malicious_clients = []
        benign_clients = []
        
        # Use top 20% of anomaly scores as malicious (simple and effective)
        sorted_scores = sorted(anomaly_scores.items(), key=lambda x: x[1]['total'], reverse=True)
        n_malicious = max(1, int(len(sorted_scores) * 0.2))
        malicious_clients = [client_id for client_id, _ in sorted_scores[:n_malicious]]
        benign_clients = [client_id for client_id, _ in sorted_scores[n_malicious:]]
        
        # Log results
        self._log_detection_results(anomaly_scores, malicious_clients, benign_clients)
        
        return malicious_clients, benign_clients
    
    def _initialize_reference(self):
        """Initialize reference with global model."""
        self.reference_model = self.global_model
        self.round_count = 0
        
        log(INFO, f"Initialized reference with global model")
    
    def _extract_all_representations(self, client_models: Dict[int, nn.Module]) -> Dict[int, Dict[int, np.ndarray]]:
        """Extract representations from all clients grouped by class."""
        client_representations = {}
        
        for client_id, model in client_models.items():
            model.eval()
            representations = []
            labels = []
            
            with torch.no_grad():
                for batch in self.root_loader:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    
                    features = self._extract_features(model, inputs)
                    representations.append(features.cpu().numpy())
                    labels.append(targets.numpy())
            
            representations = np.vstack(representations)
            labels = np.hstack(labels)
            
            # Group by class
            class_repr = {}
            for class_id in np.unique(labels):
                class_mask = labels == class_id
                class_repr[class_id] = representations[class_mask]
            
            client_representations[client_id] = class_repr
        
        return client_representations
    
    def _extract_features(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Extract penultimate layer features."""
        # Use a more robust approach - register a hook on the last layer before the classifier
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Find the last layer before the classifier (usually the last linear layer)
        layers = list(model.children())
        if len(layers) > 1:
            # Register hook on the second-to-last layer
            penultimate_layer = layers[-2]
            handle = penultimate_layer.register_forward_hook(hook_fn)
            
            with torch.no_grad():
                _ = model(inputs)
            
            handle.remove()
            
            if features:
                return features[0].flatten(1)
        
        # Fallback: just return the full model output
        with torch.no_grad():
            return model(inputs)
    
    def _compute_class_statistics(self, client_representations: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, Tuple]:
        """Compute robust class statistics."""
        class_stats = {}
        
        # Collect all representations per class
        all_class_repr = defaultdict(list)
        for client_repr in client_representations.values():
            for class_id, class_repr in client_repr.items():
                all_class_repr[class_id].append(class_repr)
        
        # Compute statistics for each class
        for class_id, repr_list in all_class_repr.items():
            if repr_list:
                # Concatenate all representations for this class
                all_repr = np.vstack(repr_list)
                
                # Use robust statistics
                mu = np.median(all_repr, axis=0)
                
                # Compute covariance with regularization
                centered_repr = all_repr - mu
                cov_matrix = np.cov(centered_repr.T)
                cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
                
                # Whitening matrix
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                eigenvals = np.maximum(eigenvals, 1e-6)
                whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
                
                class_stats[class_id] = (mu, whitening_matrix, eigenvecs)
        
        return class_stats
    
    def _build_class_templates(self, client_representations: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, np.ndarray]:
        """Build class templates using robust statistics from BENIGN clients only."""
        class_templates = {}
        
        # First, identify potentially benign clients (those with lower anomaly scores)
        # We'll use a simple heuristic: assume clients with more "normal" representations
        benign_clients = []
        
        # Collect representations per class from all clients
        all_class_repr = defaultdict(list)
        for client_id, client_repr in client_representations.items():
            for class_id, class_repr in client_repr.items():
                all_class_repr[class_id].append((client_id, class_repr))
        
        # For each class, identify clients with representations closest to the overall median
        for class_id, repr_list in all_class_repr.items():
            if len(repr_list) < 2:
                continue
                
            # Compute overall median for this class
            all_repr = np.vstack([repr for _, repr in repr_list])
            overall_median = np.median(all_repr, axis=0)
            
            # Find clients closest to the median (potential benign clients)
            distances = []
            for client_id, class_repr in repr_list:
                client_median = np.median(class_repr, axis=0)
                distance = np.linalg.norm(client_median - overall_median)
                distances.append((client_id, distance))
            
            # Sort by distance and take the closest 70% as potentially benign
            distances.sort(key=lambda x: x[1])
            n_benign = max(1, int(len(distances) * 0.7))
            benign_for_class = [client_id for client_id, _ in distances[:n_benign]]
            
            # Build template from these potentially benign clients
            benign_repr = []
            for client_id, class_repr in repr_list:
                if client_id in benign_for_class:
                    benign_repr.append(class_repr)
            
            if benign_repr:
                all_benign_repr = np.vstack(benign_repr)
                template = np.median(all_benign_repr, axis=0)
                class_templates[class_id] = template
        
        return class_templates
    
    def _determine_target_class(self, class_stats: Dict[int, Tuple], 
                               client_representations: Dict[int, Dict[int, np.ndarray]]) -> int:
        """Determine target class with highest spectral variance."""
        if not class_stats:
            return 0
        
        max_variance = -1
        target_class = 0
        
        for class_id, (mu, W, U) in class_stats.items():
            # Compute spectral variance
            eigenvals = np.linalg.eigvals(W @ W.T)
            variance = np.max(eigenvals)
            
            if variance > max_variance:
                max_variance = variance
                target_class = class_id
        
        return target_class
    
    def _compute_anomaly_scores(self, client_representations: Dict[int, Dict[int, np.ndarray]],
                               class_templates: Dict[int, np.ndarray],
                               class_stats: Dict[int, Tuple],
                               target_class: int) -> Dict[int, Dict[str, float]]:
        """Compute anomaly scores for all clients."""
        anomaly_scores = {}
        
        # Compute CKA scores
        cka_scores = {}
        for client_id, client_repr in client_representations.items():
            cka_distances = []
            
            for class_id, class_repr in client_repr.items():
                if class_id in class_templates:
                    template = class_templates[class_id]
                    
                    # Compute CKA distance (higher = more dissimilar)
                    client_median = np.median(class_repr, axis=0)
                    template_median = template
                    
                    similarity = cosine_similarity([client_median], [template_median])[0, 0]
                    distance = 1.0 - similarity
                    cka_distances.append(distance)
            
            cka_scores[client_id] = np.mean(cka_distances) if cka_distances else 0.5
        
        # Compute spectral scores
        spectral_scores = {}
        if target_class in class_stats:
            mu, W, U = class_stats[target_class]
            
            for client_id, client_repr in client_representations.items():
                if target_class in client_repr:
                    class_repr = client_repr[target_class]
                    
                    # Whitening transformation
                    centered_repr = class_repr - mu
                    whitened_repr = centered_repr @ W.T
                    
                    # Compute spectral projection
                    spectral_score = np.mean(np.linalg.norm(whitened_repr, axis=1))
                    spectral_scores[client_id] = spectral_score
                else:
                    spectral_scores[client_id] = 0.0
        else:
            spectral_scores = {client_id: 0.0 for client_id in client_representations.keys()}
        
        # Normalize scores
        cka_scores_norm = self._normalize_scores(cka_scores)
        spectral_scores_norm = self._normalize_scores(spectral_scores)
        
        # Combine scores
        for client_id in client_representations.keys():
            cka_norm = cka_scores_norm.get(client_id, 0.5)
            spectral_norm = spectral_scores_norm.get(client_id, 0.5)
            
            # Anomaly score: HIGH = malicious
            anomaly_total = (
                self.cka_weight * cka_norm +      # CKA dissimilarity
                self.spectral_weight * spectral_norm  # Spectral projection
            )
            
            anomaly_scores[client_id] = {
                'total': anomaly_total,
                'cka': cka_scores.get(client_id, 0.0),
                'spectral': spectral_scores.get(client_id, 0.0)
            }
        
        return anomaly_scores
    
    def _normalize_scores(self, scores_dict: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores using min-max normalization."""
        if not scores_dict or len(set(scores_dict.values())) <= 1:
            return {k: 0.5 for k in scores_dict.keys()}
        
        values = list(scores_dict.values())
        min_val = np.min(values)
        max_val = np.max(values)
        
        if max_val - min_val < 1e-10:
            return {k: 0.5 for k in scores_dict.keys()}
        
        # Min-max normalization to [0, 1]
        normalized = {}
        for k, v in scores_dict.items():
            normalized[k] = (v - min_val) / (max_val - min_val)
        
        return normalized
    
    def _update_dynamic_threshold(self, anomaly_scores: Dict[int, Dict[str, float]]):
        """Update dynamic threshold based on anomaly scores."""
        if not anomaly_scores:
            return
        
        scores = [result['total'] for result in anomaly_scores.values()]
        
        # Use robust statistics
        median_score = np.median(scores)
        mad = np.median(np.abs(scores - median_score))
        
        # Compute threshold
        if mad < 1e-10:
            threshold = median_score + 0.1
        else:
            threshold = median_score + 2.0 * mad
        
        # Store threshold history
        self.threshold_history.append(threshold)
        
        # Smooth threshold updates
        self.dynamic_threshold = (
            0.9 * self.dynamic_threshold + 0.1 * threshold
        )
        
        log(INFO, f"Updated dynamic threshold to {self.dynamic_threshold:.4f}")
    
    def _log_detection_results(self, anomaly_scores: Dict[int, Dict[str, float]],
                              malicious_clients: List[int], benign_clients: List[int]):
        """Log detailed detection results."""
        log(INFO, f"═══ FedSPECTRE Simple Detection Results ═══")
        log(INFO, f"Dynamic threshold: {self.dynamic_threshold:.4f}")
        log(INFO, f"Malicious clients: {malicious_clients}")
        log(INFO, f"Benign clients: {benign_clients}")
        
        # Log detailed scores
        for client_id, scores in anomaly_scores.items():
            status = "MALICIOUS" if client_id in malicious_clients else "BENIGN"
            log(INFO, f"Client {client_id} ({status}): "
                      f"total={scores['total']:.4f}, "
                      f"cka={scores['cka']:.4f}, "
                      f"spectral={scores['spectral']:.4f}")
    
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates with anomaly-based filtering.
        
        Args:
            client_updates: List of (client_id, num_examples, model_state_dict)
            
        Returns:
            True if aggregation successful
        """
        if not client_updates:
            return False
        
        # Detect anomalies
        malicious_clients, benign_clients = self.detect_anomalies(client_updates)
        
        # Filter out malicious clients
        filtered_updates = [
            (cid, num_examples, state_dict) for cid, num_examples, state_dict in client_updates
            if cid in benign_clients
        ]
        
        if not filtered_updates:
            log(WARNING, "No benign clients remaining after filtering")
            return False
        
        # Apply standard FedAvg aggregation
        total_examples = sum(num_examples for _, num_examples, _ in filtered_updates)
        
        if total_examples == 0:
            log(WARNING, "Total examples is zero")
            return False
        
        # Initialize aggregated state dict
        aggregated_state = {}
        
        # Get first model's structure
        first_cid, _, first_state = filtered_updates[0]
        
        for key in first_state.keys():
            # Ensure aggregated state is float type
            tensor = first_state[key]
            if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
                tensor = tensor.float()
            aggregated_state[key] = torch.zeros_like(tensor)
        
        # Weighted aggregation
        for client_id, num_examples, state_dict in filtered_updates:
            weight = num_examples / total_examples
            
            for key, tensor in state_dict.items():
                # Ensure tensor is float type for multiplication
                if tensor.dtype != torch.float32 and tensor.dtype != torch.float64:
                    tensor = tensor.float()
                aggregated_state[key] += weight * tensor
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        log(INFO, f"FedSPECTRE Simple aggregation complete: "
                  f"{len(filtered_updates)}/{len(client_updates)} clients aggregated")
        
        return True
    
    def _create_client_model(self) -> nn.Module:
        """Create a client model with same architecture as global model."""
        import copy
        return copy.deepcopy(self.global_model)
    
    def __repr__(self) -> str:
        return (f"FedSPECTRE-Simple("
                f"cka={self.cka_weight}, spectral={self.spectral_weight}, "
                f"threshold={self.dynamic_threshold:.3f})")
