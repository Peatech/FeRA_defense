"""
FedSPECTRE-Hybrid Server Implementation for BackFed.

Enhanced defense that combines:
1. Mahalanobis-CKA: Robust CKA in whitened space
2. Spectral Projection: Analysis of top principal component
3. Robust Statistics: Median location + OAS covariance

Based on the FedSPECTRE-Hybrid defense algorithm.
"""

import copy
import torch
import numpy as np
from typing import List, Tuple, Dict
from logging import INFO
from collections import defaultdict
from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.utils import log
from backfed.const import StateDict, client_id, num_examples
from backfed.servers.fedspectre_utils import (
    get_penultimate_layer_name,
    extract_layer_activations,
    create_root_dataset_loader,
    load_ood_dataset,
    RobustStatistics,
    MahalanobisCKA,
    SpectralProjection
)


class FedSPECTREHybridServer(AnomalyDetectionServer):
    """
    FedSPECTRE-Hybrid defense server.
    
    Uses robust statistics, Mahalanobis-CKA, and spectral projection to
    detect backdoor attacks in federated learning.
    """
    defense_categories = ["anomaly_detection"]
    
    def __init__(self, 
                 server_config,
                 server_type: str = "fedspectre_hybrid",
                 eta: float = 0.5,
                 root_size: int = 16,
                 alpha: float = 0.8,
                 beta: float = 0.0,
                 gamma: float = 0.2,
                 rank: int = 128,
                 trim_fraction: float = 0.5,
                 use_ood_root_dataset: bool = False,
                 num_augmentations: int = 5,
                 # Multi-layer parameters
                 use_multi_layer: bool = False,
                 layer_aggregation: str = 'max',
                 **kwargs):
        """
        Initialize FedSPECTRE-Hybrid server.
        
        Args:
            server_config: Server configuration
            server_type: Type of server
            eta: Learning rate for server update
            root_size: Number of samples in root dataset
            alpha: Weight for CKA component (default 0.8)
            beta: Weight for stability component (default 0.0, disabled)
            gamma: Weight for spectral component (default 0.2)
            rank: Rank for low-rank projection
            trim_fraction: Fraction of clients to exclude (legacy method)
            use_ood_root_dataset: Whether to use OOD dataset
            num_augmentations: Number of augmentations for stability (Paper K parameter)
            use_multi_layer: If True, use multi-layer templates; if False, use single layer (default False)
            layer_aggregation: Method to aggregate scores across layers: 'max', 'mean', or 'top_k' (default 'max')
        """
        super().__init__(server_config, server_type, eta, **kwargs)
        
        self.root_size = root_size
        self.alpha = alpha   # CKA weight (legacy)
        self.beta = beta     # Stability weight
        self.gamma = gamma   # Spectral weight (legacy)
        self.rank = rank
        self.trim_fraction = trim_fraction
        self.use_ood_root_dataset = use_ood_root_dataset
        self.num_augmentations = num_augmentations
        
        # Multi-layer parameters
        self.use_multi_layer = use_multi_layer
        self.layer_aggregation = layer_aggregation
        
        # Initialize components
        self.robust_stats = RobustStatistics(rank=rank, trim_fraction=0.05)
        self.cka_computer = MahalanobisCKA()
        self.spectral_computer = SpectralProjection()
        
        # Create root dataset loader
        self.root_loader = self._create_root_loader()
        
        # Store last scores for stateful extension
        self.last_anomaly_scores = {}
        
        log(INFO, f"Initialized FedSPECTRE-Hybrid server")
        log(INFO, f"  Parameters: alpha={alpha}, beta={beta}, gamma={gamma}, rank={rank}, trim={trim_fraction}")
        log(INFO, f"  Multi-layer: use_multi_layer={use_multi_layer}, aggregation={layer_aggregation}")
        log(INFO, f"  Root size: {root_size}, OOD: {use_ood_root_dataset}, augmentations: {num_augmentations}")
    
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
        Detect anomalies using FedSPECTRE-Hybrid.
        
        Args:
            client_updates: List of (client_id, num_examples, model_state_dict)
            
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        if len(client_updates) < 2:
            log(INFO, "Too few clients for FedSPECTRE-Hybrid, accepting all")
            benign_ids = [cid for cid, _, _ in client_updates]
            return [], benign_ids
        
        try:
            # Reconstruct client models
            client_models = {}
            for cid, _, state_dict in client_updates:
                model = copy.deepcopy(self.global_model)
                model.load_state_dict(state_dict, strict=True)
                client_models[cid] = model
            
            # 1. Extract representations from all clients
            if self.use_multi_layer:
                log(INFO, "Phase 2: Extracting multi-layer representations from all clients...")
                client_representations = self._extract_multi_layer_representations(client_models)
                
                if not client_representations:
                    log(INFO, "Failed to extract multi-layer representations, accepting all clients")
                    benign_ids = [cid for cid, _, _ in client_updates]
                    return [], benign_ids
                
                # 2. Compute robust statistics per class per layer
                log(INFO, "Computing robust statistics per class per layer...")
                layer_stats = self._compute_multi_layer_class_statistics(client_representations)
                
                if not layer_stats:
                    log(INFO, "Failed to compute multi-layer class statistics, accepting all clients")
                    benign_ids = [cid for cid, _, _ in client_updates]
                    return [], benign_ids
                
                # 3. Build multi-layer class templates (with whitening)
                log(INFO, "Building multi-layer class templates...")
                layer_templates = self._build_multi_layer_class_templates(client_representations, layer_stats)
                
                # 4. Determine target class (use first layer for consistency)
                first_layer = list(layer_stats.keys())[0]
                target_class = self._determine_target_class(layer_stats[first_layer], 
                                                          {cid: reps[first_layer] for cid, reps in client_representations.items() 
                                                           if first_layer in reps})
                
                # 5. Compute multi-layer anomaly scores
                log(INFO, f"Computing multi-layer anomaly scores (aggregation={self.layer_aggregation})...")
                anomaly_scores = self._compute_multi_layer_anomaly_scores(
                    client_representations,
                    layer_templates,
                    layer_stats,
                    target_class,
                    aggregation_method=self.layer_aggregation
                )
            else:
                log(INFO, "Phase 1: Extracting single-layer representations from all clients...")
                client_representations = self._extract_all_representations(client_models)
                
                if not client_representations:
                    log(INFO, "Failed to extract representations, accepting all clients")
                    benign_ids = [cid for cid, _, _ in client_updates]
                    return [], benign_ids
                
                # 2. Compute robust statistics per class
                log(INFO, "Computing robust statistics per class...")
                class_stats = self._compute_class_statistics(client_representations)
                
                if not class_stats:
                    log(INFO, "Failed to compute class statistics, accepting all clients")
                    benign_ids = [cid for cid, _, _ in client_updates]
                    return [], benign_ids
                
                # 3. Build class templates (with whitening)
                log(INFO, "Building class templates...")
                class_templates = self._build_class_templates(client_representations, class_stats)
                self.class_templates = class_templates  # Store for spectral computation
                
                # 4. Determine target class
                log(INFO, "Determining target class...")
                target_class = self._determine_target_class(class_stats, client_representations)
                
                # 5. Compute anomaly scores
                log(INFO, "Computing anomaly scores...")
                anomaly_scores = self._compute_anomaly_scores(
                    client_representations, 
                    class_templates, 
                    class_stats,
                    target_class,
                    client_models  # Pass models for stability computation
                )
            
            # Note: Components are already normalized within _compute_anomaly_scores
            # The 'total' score is properly computed from normalized components
            # Redundant second normalization removed to prevent metric imbalance
            
            # Store for stateful extension
            self.last_anomaly_scores = anomaly_scores
            
            # 6. Select clients based on scores (revert to 50% trim)
            log(INFO, "Using legacy trim-fraction method for client selection...")
            selected_clients, excluded_clients = self._select_clients(anomaly_scores)
            
            # Log results
            log(INFO, f"FedSPECTRE-Hybrid: selected {len(selected_clients)}, excluded {len(excluded_clients)}")
            log(INFO, f"  Target class: {target_class}")
            for cid in excluded_clients:
                if cid in anomaly_scores:
                    scores = anomaly_scores[cid]
                    log(INFO, f"  Excluded client {cid}: total={scores['total']:.4f}, cka={scores['cka']:.4f} (raw={scores.get('cka_raw', 0.0):.4f}), spectral={scores['spectral']:.4f} (raw={scores.get('spectral_raw', 0.0):.2f})")
            
            return excluded_clients, selected_clients
            
        except Exception as e:
            log(INFO, f"FedSPECTRE-Hybrid detection failed: {e}. Accepting all clients.")
            import traceback
            log(INFO, traceback.format_exc())
            benign_ids = [cid for cid, _, _ in client_updates]
            return [], benign_ids
    
    def _extract_all_representations(self, client_models: Dict[int, torch.nn.Module]) -> Dict[int, Dict[int, np.ndarray]]:
        """Extract representations from all clients, grouped by class."""
        client_representations = {}
        
        layer_name = get_penultimate_layer_name(self.global_model)
        log(INFO, f"Extracting from layer: {layer_name}")
        
        for cid, model in client_models.items():
            try:
                # Extract representations with labels
                representations, labels = extract_layer_activations(
                    model=model,
                    data_loader=self.root_loader,
                    layer_name=layer_name,
                    device=self.device
                )
                
                # Group by class
                class_reps = {}
                unique_classes = np.unique(labels)
                for class_id in unique_classes:
                    mask = labels == class_id
                    class_reps[int(class_id)] = representations[mask]
                
                client_representations[cid] = class_reps
                
            except Exception as e:
                log(INFO, f"Failed to extract representations for client {cid}: {e}")
                client_representations[cid] = {}
        
        return client_representations
    
    def _extract_multi_layer_representations(self, client_models: Dict[int, torch.nn.Module]) -> Dict[int, Dict[str, Dict[int, np.ndarray]]]:
        """
        Extract representations from multiple layers for each client.
        
        Phase 2: Multi-layer templates to detect sophisticated attacks like Neurotoxin.
        
        Args:
            client_models: Dictionary of client models
            
        Returns:
            {cid: {layer_name: {class_id: representations}}}
        """
        # Define layers to extract (ResNet-specific for now)
        layer_names = self._get_multi_layer_names()
        log(INFO, f"Phase 2: Extracting from {len(layer_names)} layers: {layer_names}")
        
        client_representations = {}
        
        for cid, model in client_models.items():
            try:
                layer_reps = {}
                
                for layer_name in layer_names:
                    try:
                        # Extract representations from this layer
                        representations, labels = extract_layer_activations(
                            model=model,
                            data_loader=self.root_loader,
                            layer_name=layer_name,
                            device=self.device
                        )
                        
                        # Group by class
                        class_reps = {}
                        unique_classes = np.unique(labels)
                        for class_id in unique_classes:
                            mask = labels == class_id
                            class_reps[int(class_id)] = representations[mask]
                        
                        layer_reps[layer_name] = class_reps
                        
                    except Exception as e:
                        log(INFO, f"Failed to extract from layer {layer_name} for client {cid}: {e}")
                        layer_reps[layer_name] = {}
                
                client_representations[cid] = layer_reps
                
            except Exception as e:
                log(INFO, f"Failed to extract multi-layer representations for client {cid}: {e}")
                client_representations[cid] = {}
        
        return client_representations
    
    def _get_multi_layer_names(self) -> List[str]:
        """
        Get list of layer names for multi-layer extraction.
        
        Returns:
            List of layer names to extract representations from
        """
        # ResNet-specific layers (most effective for backdoor detection)
        if hasattr(self.global_model, 'layer2') and hasattr(self.global_model, 'layer3'):
            return ['layer2', 'layer3', 'avgpool']
        
        # VGG-specific layers
        if hasattr(self.global_model, 'features') and hasattr(self.global_model, 'classifier'):
            return ['features.10', 'features.20', 'features.30']  # Example VGG layers
        
        # Generic fallback: use penultimate layer
        penultimate = get_penultimate_layer_name(self.global_model)
        return [penultimate]
    
    def _compute_multi_layer_class_statistics(self, client_representations: Dict[int, Dict[str, Dict[int, np.ndarray]]]) -> Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Compute robust statistics for each class in each layer.
        
        Args:
            client_representations: {cid: {layer_name: {class_id: reps}}}
            
        Returns:
            {layer_name: {class_id: (mu, W, U)}}
        """
        layer_stats = {}
        
        # Process each layer separately
        for layer_name in client_representations[list(client_representations.keys())[0]].keys():
            # Collect all representations for this layer
            layer_data = defaultdict(list)
            for client_reps in client_representations.values():
                if layer_name in client_reps:
                    for class_id, reps in client_reps[layer_name].items():
                        layer_data[class_id].append(reps)
            
            # Compute statistics for this layer
            layer_class_stats = {}
            for class_id, rep_list in layer_data.items():
                if not rep_list:
                    continue
                
                # Stack all representations for this class
                stacked_reps = np.vstack(rep_list)
                
                # Compute robust statistics
                mu, W, U = self.robust_stats.compute_robust_statistics(stacked_reps)
                layer_class_stats[class_id] = (mu, W, U)
            
            layer_stats[layer_name] = layer_class_stats
        
        return layer_stats
    
    def _build_multi_layer_class_templates(self, 
                                         client_representations: Dict[int, Dict[str, Dict[int, np.ndarray]]],
                                         layer_stats: Dict[str, Dict[int, Tuple]]) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Build whitened templates per class per layer.
        
        Phase 2: Multi-layer templates for sophisticated attack detection.
        
        Args:
            client_representations: {cid: {layer_name: {class_id: reps}}}
            layer_stats: {layer_name: {class_id: (mu, W, U)}}
            
        Returns:
            {layer_name: {class_id: template}}
        """
        layer_templates = {}
        
        for layer_name in layer_stats.keys():
            class_templates = {}
            
            for class_id in layer_stats[layer_name].keys():
                # Collect representations from all clients for this class in this layer
                class_reps = []
                for cid, layer_reps in client_representations.items():
                    if layer_name in layer_reps and class_id in layer_reps[layer_name]:
                        class_reps.append(layer_reps[layer_name][class_id])
                
                if not class_reps:
                    continue
                
                # Build median template
                stacked = np.vstack(class_reps)
                median_rep = np.median(stacked, axis=0)
                
                # Whiten template using layer-specific statistics
                mu, W, U = layer_stats[layer_name][class_id]
                template_whitened = (median_rep - mu) @ W.T
                
                class_templates[class_id] = template_whitened
            
            layer_templates[layer_name] = class_templates
        
        return layer_templates
    
    def _compute_multi_layer_anomaly_scores(self,
                                          client_representations: Dict[int, Dict[str, Dict[int, np.ndarray]]],
                                          layer_templates: Dict[str, Dict[int, np.ndarray]],
                                          layer_stats: Dict[str, Dict[int, Tuple]],
                                          target_class: int,
                                          aggregation_method: str = 'max') -> Dict[int, Dict[str, float]]:
        """
        Compute anomaly scores across multiple layers.
        
        Phase 2: Multi-layer scoring to detect sophisticated attacks.
        
        Args:
            client_representations: Multi-layer client reps
            layer_templates: Multi-layer templates
            layer_stats: Multi-layer stats
            target_class: Target class for spectral analysis
            aggregation_method: 'max', 'mean', or 'top_k'
            
        Returns:
            Per-client anomaly scores
        """
        # Compute scores per layer per client
        client_layer_scores = defaultdict(lambda: defaultdict(list))
        
        for layer_name in layer_templates.keys():
            if layer_name not in layer_stats:
                continue
            
            # Compute CKA scores for this layer
            for cid, layer_reps in client_representations.items():
                if layer_name not in layer_reps:
                    continue
                
                layer_cka_scores = []
                for class_id, client_rep in layer_reps[layer_name].items():
                    if class_id in layer_templates[layer_name] and class_id in layer_stats[layer_name]:
                        mu, W, U = layer_stats[layer_name][class_id]
                        template = layer_templates[layer_name][class_id]
                        
                        cka_dist = self.cka_computer.compute_cka_score(
                            client_rep, template, mu, W
                        )
                        layer_cka_scores.append(cka_dist)
                
                if layer_cka_scores:
                    client_layer_scores[cid][layer_name] = np.mean(layer_cka_scores)
            
            # Compute spectral scores for this layer
            if layer_name in layer_stats and target_class in layer_stats[layer_name]:
                layer_spectral_scores = self._compute_spectral_scores_per_client(
                    {cid: reps[layer_name] for cid, reps in client_representations.items() 
                     if layer_name in reps},
                    {target_class: layer_stats[layer_name][target_class]},
                    target_class
                )
                
                for cid, score in layer_spectral_scores.items():
                    if cid in client_layer_scores:
                        client_layer_scores[cid][f"{layer_name}_spectral"] = score
        
        # Aggregate scores across layers
        anomaly_scores = {}
        for cid in client_representations.keys():
            if cid not in client_layer_scores:
                anomaly_scores[cid] = {'total': 0.5, 'cka': 0.5, 'spectral': 0.5}
                continue
            
            # Collect all layer scores for this client
            layer_scores = []
            spectral_scores = []
            
            for layer_name in layer_templates.keys():
                if layer_name in client_layer_scores[cid]:
                    layer_scores.append(client_layer_scores[cid][layer_name])
                if f"{layer_name}_spectral" in client_layer_scores[cid]:
                    spectral_scores.append(client_layer_scores[cid][f"{layer_name}_spectral"])
            
            # Aggregate across layers
            if layer_scores:
                if aggregation_method == 'max':
                    cka_score = max(layer_scores)
                elif aggregation_method == 'mean':
                    cka_score = np.mean(layer_scores)
                elif aggregation_method == 'top_k':
                    k = min(2, len(layer_scores))
                    cka_score = np.mean(sorted(layer_scores, reverse=True)[:k])
                else:
                    cka_score = max(layer_scores)
            else:
                cka_score = 0.5
            
            if spectral_scores:
                if aggregation_method == 'max':
                    spectral_score = max(spectral_scores)
                elif aggregation_method == 'mean':
                    spectral_score = np.mean(spectral_scores)
                elif aggregation_method == 'top_k':
                    k = min(2, len(spectral_scores))
                    spectral_score = np.mean(sorted(spectral_scores, reverse=True)[:k])
                else:
                    spectral_score = max(spectral_scores)
            else:
                spectral_score = 0.5
            
            # Legacy combination (revert from Phase 1 calibration)
            total_score = (
                self.alpha * cka_score +
                self.gamma * spectral_score
            )
            
            anomaly_scores[cid] = {
                'total': total_score,
                'cka': cka_score,
                'spectral': spectral_score,
                'cka_raw': cka_score,
                'spectral_raw': spectral_score
            }
        
        return anomaly_scores
    
    def _compute_class_statistics(self, client_representations: Dict[int, Dict[int, np.ndarray]]) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute robust statistics for each class."""
        class_stats = {}
        
        # Collect all representations per class
        class_data = defaultdict(list)
        for client_reps in client_representations.values():
            for class_id, reps in client_reps.items():
                class_data[class_id].append(reps)
        
        # Compute statistics for each class
        for class_id, class_reps_list in class_data.items():
            if not class_reps_list:
                continue
            
            try:
                # Stack all representations for this class
                all_reps = np.vstack(class_reps_list)
                
                # Compute robust statistics
                mu, W, U = self.robust_stats.compute_robust_stats(all_reps)
                class_stats[class_id] = (mu, W, U)
            except Exception as e:
                log(INFO, f"Failed to compute stats for class {class_id}: {e}")
        
        return class_stats
    
    def _build_class_templates(self, client_representations: Dict[int, Dict[int, np.ndarray]], 
                               class_stats: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[int, np.ndarray]:
        """
        Build class templates as whitened median across clients.
        
        Paper Eq. (Step 1.3): Template T_c = W_c · μ_c
        where μ_c is the robust median and W_c is the whitening matrix.
        
        Args:
            client_representations: Client representations grouped by class
            class_stats: Robust statistics (mu, W, U) for each class
            
        Returns:
            Dict mapping class_id to whitened template in whitened space
        """
        class_templates = {}
        
        # Collect representations per class
        class_data = defaultdict(list)
        for client_reps in client_representations.values():
            for class_id, reps in client_reps.items():
                class_data[class_id].append(reps)
        
        # Build templates with whitening
        for class_id, class_reps_list in class_data.items():
            if not class_reps_list or len(class_reps_list) < 2:
                continue
            
            if class_id not in class_stats:
                log(INFO, f"No class stats for class {class_id}, skipping template")
                continue
            
            try:
                # Stack and take median
                stacked = np.stack(class_reps_list, axis=0)
                median_rep = np.median(stacked, axis=0)
                
                # Apply whitening transformation: T_c = W_c · (median - μ_c)
                mu, W, U = class_stats[class_id]
                template_whitened = (median_rep - mu) @ W.T
                
                class_templates[class_id] = template_whitened
                log(INFO, f"Built whitened template for class {class_id}, shape: {template_whitened.shape}")
            except Exception as e:
                log(INFO, f"Failed to build template for class {class_id}: {e}")
        
        return class_templates
    
    def _get_target_template(self, target_class: int) -> np.ndarray:
        """Get template direction for target class."""
        if hasattr(self, 'class_templates') and target_class in self.class_templates:
            return self.class_templates[target_class]
        return None
    
    def _determine_target_class(self, class_stats: Dict[int, Tuple], client_representations: Dict[int, Dict[int, np.ndarray]]) -> int:
        """Determine target class with highest spike in whitened space."""
        if not class_stats:
            return 0
        
        max_spike = -1
        target_class = 0
        
        for class_id, (mu, W, U) in class_stats.items():
            try:
                # Collect all whitened representations for this class
                whitened_reps = []
                for client_reps in client_representations.values():
                    if class_id in client_reps:
                        rep_white = (client_reps[class_id] - mu) @ W.T
                        whitened_reps.append(rep_white)
                
                if not whitened_reps:
                    continue
                
                # Pool and compute covariance
                pooled_white = np.vstack(whitened_reps)
                cov_white = pooled_white.T @ pooled_white / (len(pooled_white) - 1)
                
                # Largest eigenvalue is the spike
                eigenvals = np.linalg.eigvals(cov_white)
                spike = np.max(np.real(eigenvals))
                
                if spike > max_spike:
                    max_spike = spike
                    target_class = class_id
            except Exception as e:
                log(INFO, f"Failed to compute spike for class {class_id}: {e}")
        
        return target_class
    
    def _compute_spectral_scores_per_client(self, 
                                            client_representations: Dict[int, Dict[int, np.ndarray]],
                                            class_stats: Dict[int, Tuple],
                                            target_class: int) -> Dict[int, float]:
        """
        Compute per-client spectral scores as per paper Eq. 64.
        
        Paper Formula: Spectral_i = Σ_{k=1}^r λ_k^(i) · |v_k^(i)^T · P_target|
        
        This computes eigenvalues and eigenvectors for EACH client separately,
        then projects onto the target class template direction and sums over top-r components.
        """
        spectral_scores = {}
        
        if target_class not in class_stats:
            return {cid: 0.0 for cid in client_representations.keys()}
        
        mu, W, U = class_stats[target_class]
        
        # Get target class template direction (already whitened)
        target_template = self._get_target_template(target_class)
        
        for cid, client_reps in client_representations.items():
            if target_class not in client_reps:
                spectral_scores[cid] = 0.0
                continue
            
            try:
                # Whiten client's representations
                rep_white = (client_reps[target_class] - mu) @ W.T
                
                # Compute per-client covariance in whitened space
                if len(rep_white) < 2:
                    spectral_scores[cid] = 0.0
                    continue
                    
                cov_client = rep_white.T @ rep_white / (len(rep_white) - 1)
                
                # Get eigenvalues/eigenvectors for this client
                eigenvals, eigenvecs = np.linalg.eigh(cov_client)
                
                # Sort descending (largest eigenvalues first)
                idx = np.argsort(eigenvals)[::-1]
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
                # Take top-r components as per paper
                r = min(self.rank, len(eigenvals))
                
                # Paper Eq. 64: Sum over top-r eigenvalues weighted by projection
                if target_template is not None and len(target_template) == len(eigenvecs):
                    # Normalize target template for projection
                    target_proj = target_template / (np.linalg.norm(target_template) + 1e-8)
                    
                    # Sum: Σ_{k=1}^r λ_k · |v_k^T · P_target|
                    score = 0.0
                    for k in range(r):
                        eigenval = eigenvals[k]
                        eigenvec = eigenvecs[:, k]
                        projection = abs(np.dot(eigenvec, target_proj))
                        score += eigenval * projection
                else:
                    # Fallback: sum top-r eigenvalues (no projection available)
                    score = np.sum(eigenvals[:r])
                
                spectral_scores[cid] = float(score)
            except Exception as e:
                log(INFO, f"Failed to compute spectral score for client {cid}: {e}")
                spectral_scores[cid] = 0.0
        
        return spectral_scores
    
    def _compute_augmentation_stability(self, client_models: Dict[int, torch.nn.Module]) -> Dict[int, float]:
        """
        Compute augmentation stability scores for clients (Paper Eq. 70-72).
        
        Formula: Stability_i = (1/K) Σ_{k=1}^K ||f_i^(k) - f_i^(0)||_2
        
        where f_i^(k) is the representation under augmentation k.
        Higher score = more unstable = more suspicious.
        
        Args:
            client_models: Dict mapping client_id to model
            
        Returns:
            Dict mapping client_id to stability score (higher = more unstable)
        """
        if self.beta == 0.0 or self.num_augmentations == 0:
            # Beta=0 means stability not used, skip computation
            return {cid: 0.5 for cid in client_models.keys()}
        
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader, Dataset
        
        stability_scores = {}
        layer_name = get_penultimate_layer_name(self.global_model)
        
        # Get original root dataset
        try:
            # Extract one batch from root loader
            root_batch = next(iter(self.root_loader))
            if isinstance(root_batch, (list, tuple)):
                original_data = root_batch[0]
            else:
                original_data = root_batch
        except Exception as e:
            log(INFO, f"Failed to get root batch for augmentation stability: {e}")
            return {cid: 0.5 for cid in client_models.keys()}
        
        # Define augmentations (similar to standard data augmentation)
        augmentation_transforms = []
        
        # Check if data is image-like
        if len(original_data.shape) == 4 and original_data.shape[1] in [1, 3]:
            # Image data: [B, C, H, W]
            for k in range(self.num_augmentations):
                if k == 0:
                    # No augmentation (identity)
                    augmentation_transforms.append(transforms.Compose([]))
                elif k == 1:
                    # Horizontal flip
                    augmentation_transforms.append(transforms.RandomHorizontalFlip(p=1.0))
                elif k == 2:
                    # Random crop with padding
                    augmentation_transforms.append(transforms.RandomCrop(original_data.shape[-1], padding=4))
                elif k == 3:
                    # Color jitter (if RGB)
                    if original_data.shape[1] == 3:
                        augmentation_transforms.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.1))
                    else:
                        augmentation_transforms.append(transforms.Compose([]))
                else:
                    # Random affine
                    augmentation_transforms.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
        else:
            # Non-image data, use identity transforms
            augmentation_transforms = [transforms.Compose([]) for _ in range(self.num_augmentations)]
        
        # Compute stability for each client
        for cid, model in client_models.items():
            try:
                # Extract original representation (f_i^(0))
                original_rep, _ = extract_layer_activations(
                    model=model,
                    data_loader=self.root_loader,
                    layer_name=layer_name,
                    device=self.device
                )
                
                # Flatten to vector per sample
                if original_rep.ndim > 2:
                    original_rep = original_rep.reshape(original_rep.shape[0], -1)
                
                # Compute representations under augmentations
                distances = []
                for k, aug_transform in enumerate(augmentation_transforms[1:], start=1):  # Skip k=0 (identity)
                    try:
                        # Apply augmentation
                        augmented_data = aug_transform(original_data)
                        
                        # Create temporary loader
                        class SimpleDataset(Dataset):
                            def __init__(self, data):
                                self.data = data
                            def __len__(self):
                                return len(self.data)
                            def __getitem__(self, idx):
                                return self.data[idx], 0  # Dummy label
                        
                        aug_dataset = SimpleDataset(augmented_data)
                        aug_loader = DataLoader(aug_dataset, batch_size=len(augmented_data), shuffle=False)
                        
                        # Extract augmented representation
                        aug_rep, _ = extract_layer_activations(
                            model=model,
                            data_loader=aug_loader,
                            layer_name=layer_name,
                            device=self.device
                        )
                        
                        if aug_rep.ndim > 2:
                            aug_rep = aug_rep.reshape(aug_rep.shape[0], -1)
                        
                        # Compute L2 distance: ||f_i^(k) - f_i^(0)||_2
                        distance = np.linalg.norm(aug_rep - original_rep, axis=1).mean()
                        distances.append(distance)
                    except Exception as e:
                        log(INFO, f"Augmentation {k} failed for client {cid}: {e}")
                        continue
                
                # Average over augmentations: (1/K) Σ ||f_i^(k) - f_i^(0)||_2
                if distances:
                    stability_score = np.mean(distances)
                else:
                    stability_score = 0.0
                
                stability_scores[cid] = float(stability_score)
                
            except Exception as e:
                log(INFO, f"Failed to compute stability for client {cid}: {e}")
                stability_scores[cid] = 0.5
        
        return stability_scores
    
    def _compute_anomaly_scores(self, 
                                client_representations: Dict[int, Dict[int, np.ndarray]],
                                class_templates: Dict[int, np.ndarray],
                                class_stats: Dict[int, Tuple],
                                target_class: int,
                                client_models: Dict[int, torch.nn.Module] = None) -> Dict[int, Dict[str, float]]:
        """
        Compute anomaly scores for all clients (Paper Step 2.2).
        
        Combines three components:
        - CKA: Mahalanobis-CKA distance in whitened space
        - Spectral: Projection onto target class principal components
        - Stability: Augmentation stability (if beta > 0)
        """
        anomaly_scores = {}
        
        # Compute CKA scores for each client
        cka_scores = {}
        for cid, client_reps in client_representations.items():
            cka_list = []
            for class_id, client_rep in client_reps.items():
                if class_id in class_templates and class_id in class_stats:
                    mu, W, U = class_stats[class_id]
                    template = class_templates[class_id]
                    
                    # Compute CKA distance (template already whitened)
                    cka_dist = self.cka_computer.compute_cka_score(client_rep, template, mu, W)
                    cka_list.append(cka_dist)
            
            cka_scores[cid] = np.mean(cka_list) if cka_list else 0.5
        
        # Compute spectral scores per-client (Paper Eq. 64)
        spectral_scores = self._compute_spectral_scores_per_client(
            client_representations, class_stats, target_class
        )
        
        # Compute augmentation stability scores (Paper Eq. 70-72)
        if self.beta > 0 and client_models is not None:
            log(INFO, f"Computing augmentation stability (beta={self.beta}, K={self.num_augmentations})...")
            stability_scores = self._compute_augmentation_stability(client_models)
        else:
            stability_scores = {cid: 0.5 for cid in client_representations.keys()}
        
        # Use legacy IQR normalization (revert from MAD z-scoring)
        log(INFO, "Using IQR normalization...")
        cka_z = self._normalize_scores(cka_scores)
        spectral_z = self._normalize_scores(spectral_scores)
        stability_z = self._normalize_scores(stability_scores)
        
        # Combine scores: HIGH anomaly = malicious
        for cid in client_representations.keys():
            z_cka = cka_z.get(cid, 0.0)
            z_spectral = spectral_z.get(cid, 0.0)
            z_stability = stability_z.get(cid, 0.0)
            
            # Legacy: use old formula with alpha, beta, gamma
            # Note: z_cka is ALREADY dissimilarity (1 - similarity) from compute_cka_score
            anomaly_total = (
                self.alpha * z_cka +                # CKA dissimilarity
                self.beta * (1.0 - z_stability) +   # Instability
                self.gamma * z_spectral             # Spectral projection
            )
            
            # Store normalized scores to ensure proper scaling for downstream consumers
            # All scores normalized using IQR method to [0,1] range
            anomaly_scores[cid] = {
                'total': anomaly_total,
                'cka': z_cka,           # IQR normalized CKA [0,1]
                'spectral': z_spectral, # IQR normalized spectral [0,1]
                'stability': z_stability, # IQR normalized stability [0,1]
                # Keep raw scores for logging/debugging
                'cka_raw': cka_scores.get(cid, 0.0),
                'spectral_raw': spectral_scores.get(cid, 0.0),
                'stability_raw': stability_scores.get(cid, 0.0)
            }
        
        return anomaly_scores
    
    def _normalize_anomaly_scores(self, anomaly_scores: Dict[int, Dict[str, float]]):
        """
        Cross-client normalization of anomaly scores (Paper Step 2.3, Lines 186-194).
        
        Formula: z_i = clip((A_i - median) / (1.5 * IQR) + 0.5, 0, 1)
        
        Modifies anomaly_scores dict in-place to add 'normalized' field.
        """
        if not anomaly_scores:
            return
        
        # Extract total scores
        total_scores = [score_dict['total'] for score_dict in anomaly_scores.values()]
        
        if len(set(total_scores)) <= 1:
            # All scores the same, use neutral value
            for score_dict in anomaly_scores.values():
                score_dict['normalized'] = 0.5
            return
        
        # Robust statistics
        median_score = np.median(total_scores)
        q25, q75 = np.percentile(total_scores, [25, 75])
        iqr = q75 - q25
        
        # Normalize each client's score
        for cid, score_dict in anomaly_scores.items():
            total_score = score_dict['total']
            
            if iqr < 1e-10:
                # Fallback to standard deviation if IQR too small
                std_score = np.std(total_scores)
                if std_score < 1e-10:
                    z_score = 0.5
                else:
                    z_score = np.clip((total_score - median_score) / (3 * std_score) + 0.5, 0.0, 1.0)
            else:
                # Paper formula: z_i = clip((A_i - median) / (1.5 * IQR) + 0.5, 0, 1)
                z_score = np.clip((total_score - median_score) / (1.5 * iqr) + 0.5, 0.0, 1.0)
            
            score_dict['normalized'] = float(z_score)
    
    def _normalize_scores(self, scores_dict: Dict[int, float]) -> Dict[int, float]:
        """Normalize scores using robust median/IQR scaling (legacy, for internal use)."""
        if not scores_dict or len(set(scores_dict.values())) <= 1:
            return {k: 0.5 for k in scores_dict.keys()}
        
        values = list(scores_dict.values())
        median_val = np.median(values)
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        
        if iqr < 1e-10:
            std_val = np.std(values)
            if std_val < 1e-10:
                return {k: 0.5 for k in scores_dict.keys()}
            return {k: np.clip((v - median_val) / (3 * std_val) + 0.5, 0, 1) for k, v in scores_dict.items()}
        
        return {k: np.clip((v - median_val) / (1.5 * iqr) + 0.5, 0, 1) for k, v in scores_dict.items()}
    
    def _compute_robust_z_scores_UNUSED(self, scores_dict: Dict[int, float], z_max: float = None) -> Dict[int, float]:
        """
        Robust z-score normalization using MAD (Median Absolute Deviation).
        
        Formula: z = clip((x - median) / (1.4826 * MAD), 0, z_max)
        
        MAD is more robust than IQR for outlier detection.
        The constant 1.4826 makes MAD consistent with std dev for normal distributions.
        One-sided clipping: negative values (below median) are set to 0 (no suspicion),
        positive values indicate increasing suspicion up to z_max.
        
        Args:
            scores_dict: Raw scores per client
            z_max: Maximum z-score (clip outliers), uses self.z_max if None
            
        Returns:
            Z-scored values clipped to [0, z_max]
        """
        if z_max is None:
            z_max = self.z_max
        
        if not scores_dict or len(scores_dict) < 2:
            return {k: 0.0 for k in scores_dict.keys()}
        
        values = np.array(list(scores_dict.values()))
        
        # Robust location and scale
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        
        # Handle constant values (MAD=0)
        if mad < 1e-10:
            return {k: 0.0 for k in scores_dict.keys()}
        
        # Consistent MAD estimator (matches std dev for normal distribution)
        mad_scaled = 1.4826 * mad
        
        # Compute z-scores and clip to [0, z_max]
        # One-sided: negative z (below median) → 0 (no suspicion)
        #           positive z (above median) → increasing suspicion
        z_scores = {}
        for cid, value in scores_dict.items():
            z = (value - median) / mad_scaled
            z_scores[cid] = float(np.clip(z, 0.0, z_max))
        
        return z_scores
    
    def _select_clients(self, anomaly_scores: Dict[int, Dict[str, float]]) -> Tuple[List[int], List[int]]:
        """Select clients based on anomaly scores."""
        if not anomaly_scores:
            return [], []
        
        # Sort by total score (ascending = most benign first)
        sorted_clients = sorted(anomaly_scores.keys(), key=lambda cid: anomaly_scores[cid]['total'])
        
        # Select top (1 - trim_fraction) clients
        n_clients = len(sorted_clients)
        n_select = max(1, int(n_clients * (1 - self.trim_fraction)))
        
        selected_clients = sorted_clients[:n_select]
        excluded_clients = sorted_clients[n_select:]
        
        return selected_clients, excluded_clients
    
    def _select_clients_quantile_UNUSED(self, 
                                 anomaly_scores: Dict[int, Dict[str, float]],
                                 client_representations: Dict[int, Dict[int, np.ndarray]],
                                 quantile_threshold: float = None) -> Tuple[List[int], List[int]]:
        """
        Select clients using per-class quantile gating (Phase 1, Step 3).
        
        Controls FPR by design: at most (1 - quantile_threshold) of clients per class
        are marked anomalous. This prevents arbitrary exclusions and adapts to per-class
        score distributions.
        
        Strategy: Per-class by dominant class (Option A from clarifications)
        - Dominant class = argmax over client's label counts (number of samples per class)
        - If unavailable, fallback to global quantile for that client
        
        Args:
            anomaly_scores: Anomaly scores with 'total' field
            client_representations: Client reps by class {cid: {class_id: reps}}
            quantile_threshold: Quantile threshold (0.90 = 90th percentile = 10% FPR), uses self.quantile_threshold if None
            
        Returns:
            (selected_clients, excluded_clients)
        """
        if quantile_threshold is None:
            quantile_threshold = self.quantile_threshold
        
        if not anomaly_scores:
            return [], []
        
        # Group clients by their dominant class
        client_classes = {}
        for cid, class_reps in client_representations.items():
            if not class_reps:
                # No representations for this client, use None as marker
                client_classes[cid] = None
                continue
            
            # Dominant class = class with most samples (argmax over sample counts)
            dominant_class = max(class_reps.items(), key=lambda x: x[1].shape[0])[0]
            client_classes[cid] = dominant_class
        
        # Group scores by class
        class_scores = defaultdict(list)
        global_scores = []  # For clients without class info
        
        for cid, scores in anomaly_scores.items():
            class_id = client_classes.get(cid, None)
            total_score = scores['total']
            
            if class_id is not None:
                class_scores[class_id].append((cid, total_score))
            else:
                # Fallback to global for clients without class info
                global_scores.append((cid, total_score))
        
        # Compute per-class thresholds and select clients
        excluded_clients = []
        selected_clients = []
        
        for class_id, client_score_list in class_scores.items():
            if not client_score_list:
                continue
            
            scores = [score for _, score in client_score_list]
            threshold = np.quantile(scores, quantile_threshold)
            
            for cid, score in client_score_list:
                if score > threshold:
                    excluded_clients.append(cid)
                else:
                    selected_clients.append(cid)
        
        # Handle clients without class info using global quantile (fallback)
        if global_scores:
            scores = [score for _, score in global_scores]
            threshold = np.quantile(scores, quantile_threshold)
            
            for cid, score in global_scores:
                if score > threshold:
                    excluded_clients.append(cid)
                else:
                    selected_clients.append(cid)
        
        log(INFO, f"Per-class quantile gating (threshold={quantile_threshold:.2f}): "
                  f"selected {len(selected_clients)}, excluded {len(excluded_clients)}")
        log(INFO, f"  Class distribution: {len(class_scores)} classes, {len(global_scores)} clients without class info")
        
        return selected_clients, excluded_clients
    
    def __repr__(self) -> str:
        return f"FedSPECTRE-Hybrid(alpha={self.alpha}, beta={self.beta}, rank={self.rank}, trim={self.trim_fraction})"

