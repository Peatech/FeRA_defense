"""
FeRA Visualize Server

Multi-component filtering defense for backdoor detection in federated learning.

Computes per-client metrics and applies configurable filters to detect malicious clients.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from logging import INFO, WARNING
from torch.utils.data import DataLoader, Subset
import random
import time

from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.servers.fedspectre_utils import load_ood_dataset, create_root_dataset_loader
from backfed.utils.system_utils import log
from backfed.utils.model_utils import get_model
from backfed.const import client_id, num_examples, StateDict


class FeraVisualizeServer(AnomalyDetectionServer):
    """
    FeRA Visualize: Multi-component filtering defense with switchable variants.
    
    Variant 1 (Default): Multi-component filters (default, collusion, outlier, scaled norm)
    Variant 2 (Simplified): Consistency + collusion + norm-inflation filters with clipping
    
    Configurable via filter_variant parameter ("v1" or "v2").
    Visualizations (CSV outputs, ranked tables) can be enabled via enable_visualizations flag.
    """
    
    defense_categories = ["anomaly_detection"]
    
    def __init__(
        self,
        server_config,
        server_type: str = "fera_visualize",
        eta: float = 0.5,
        root_size: int = 64,
        epsilon: float = 1e-12,
        param_change_threshold: float = 0.0,
        feature_dim: int = None,  # Changed to None - will auto-detect
        spectral_weight: float = 0.6,
        delta_weight: float = 0.4,
        # Layer selection parameters
        extraction_layer: str = 'penultimate',
        extraction_layers: List[str] = None,
        combine_layers_method: str = 'mean',
        # OOD root dataset parameters
        use_ood_root_dataset: bool = False,
        ood_root_dataset_name: str = None,
        # Filter configurations
        default_filter: dict = None,
        collusion_filter: dict = None,
        outlier_filter: dict = None,
        scaled_norm_filter: dict = None,
        # Filter variant selection
        filter_variant: str = "v1",  # "v1" (original) or "v2" (simplified)
        # V2 Clipping and Noise parameters
        enable_clipping: bool = True,  # Enable clipping for flagged malicious clients (V2 only, default: True)
        clip_percentage: float = 0.6,  # Percentage of median to clip malicious updates to (0.6 = 60%, default: 0.6)
        enable_adaptive_noise: bool = False,  # Enable adaptive noise addition (V2 only, default: False)
        noise_mode: str = "adaptive",  # Noise mode: "adaptive" (scales with param std) or "fixed" (constant std)
        noise_lambda: float = 0.001,  # Lambda parameter for adaptive noise (default: 0.001, suggest: 0.01-0.1)
        noise_std: float = 0.025,  # Standard deviation for fixed noise mode (default: 0.025, like WeakDP)
        # Eigenvalue metrics parameters
        top_k_eigenvalues: int = 5,  # Number of top eigenvalues to track (default: 5)
        effective_rank_threshold: float = 0.9,  # Variance percentage for effective rank (default: 0.9 = 90%)
        # Visualization parameters
        enable_visualizations: bool = False,  # Enable CSV outputs and ranked table logs (default: False)
        **kwargs
    ):
        """
        Initialize FeRA Visualize server.
        
        Args:
            server_config: Configuration dictionary
            server_type: Type identifier (default: "fera_visualize")
            eta: Learning rate for aggregation
            root_size: Number of samples for representation extraction (default: 64)
            epsilon: Small constant for numerical stability (default: 1e-12)
            param_change_threshold: Threshold for counting parameter changes (default: 0.0)
            feature_dim: Expected feature dimension (default: None, auto-detect from model)
            spectral_weight: Weight for spectral norm in combined score (default: 0.6)
            delta_weight: Weight for delta norm in combined score (default: 0.4)
            extraction_layer: Layer name for single-layer extraction (default: 'penultimate')
            extraction_layers: List of layers for multi-layer extraction (default: None)
            combine_layers_method: Method to combine multi-layer features (default: 'mean')
            use_ood_root_dataset: Use OOD dataset for root samples (default: False)
            ood_root_dataset_name: Explicit OOD dataset name or None for auto-detect
            default_filter: Default filter configuration (Component 1)
            collusion_filter: Collusion filter configuration (Component 2)
            outlier_filter: Outlier filter configuration (Component 3)
            scaled_norm_filter: Scaled norm filter configuration (Component 4)
            filter_variant: Filter logic variant - "v1" (original) or "v2" (simplified)
            enable_clipping: Enable clipping for flagged malicious clients (V2 only, default: True)
            clip_percentage: Percentage of median to clip malicious updates to (0.6 = 60%, default: 0.6)
            enable_adaptive_noise: Enable noise addition after aggregation (V2 only, default: False)
            noise_mode: Noise mode - "adaptive" (scales with param std and clip_norm) or "fixed" (constant std)
            noise_lambda: Lambda for adaptive noise (default: 0.001, recommend: 0.01-0.1 for stronger noise)
            noise_std: Std dev for fixed noise mode (default: 0.025, recommend: 0.05-0.1 for stronger noise)
            top_k_eigenvalues: Number of top eigenvalues to track for visualization (default: 5)
            effective_rank_threshold: Variance percentage threshold for effective rank computation (default: 0.9 = 90%)
            enable_visualizations: Enable CSV outputs and ranked table console logs (default: False)
        """
        super().__init__(server_config, server_type, eta, **kwargs)
        
        # Store filter variant
        self.filter_variant = filter_variant
        
        # V2 Clipping and Noise parameters
        self.enable_clipping = enable_clipping
        self.clip_percentage = clip_percentage
        self.enable_adaptive_noise = enable_adaptive_noise
        self.noise_mode = noise_mode
        self.noise_lambda = noise_lambda
        self.noise_std = noise_std
        
        # Eigenvalue metrics parameters
        self.top_k_eigenvalues = top_k_eigenvalues
        self.effective_rank_threshold = effective_rank_threshold
        
        # Visualization parameters
        self.enable_visualizations = enable_visualizations
        
        # Store parameters
        self.root_size = root_size
        self.epsilon = epsilon
        self.param_change_threshold = param_change_threshold
        self.spectral_weight = spectral_weight
        self.delta_weight = delta_weight
        
        # Layer selection
        self.extraction_layer = extraction_layer
        self.extraction_layers = extraction_layers if extraction_layers else [extraction_layer]
        self.combine_layers_method = combine_layers_method
        self.use_multi_layer = len(self.extraction_layers) > 1
        
        # OOD root dataset
        self.use_ood_root_dataset = use_ood_root_dataset
        self.ood_root_dataset_name = ood_root_dataset_name
        
        # Store filter configurations with defaults
        self.default_filter = default_filter or {
            'enabled': True,
            'combined_threshold': 0.50,
            'tda_threshold': 0.50,
            'mutual_sim_threshold': 0.70
        }
        self.collusion_filter = collusion_filter or {
            'enabled': False,
            'mutual_sim_top_percent': 0.40,
            'tda_top_percent': 0.40
        }
        self.outlier_filter = outlier_filter or {
            'enabled': False
        }
        self.scaled_norm_filter = scaled_norm_filter or {
            'enabled': False,
            'spectral_ratio_threshold': 100.0
        }
        
        # Auto-detect feature dimension if not provided
        if feature_dim is None:
            self.feature_dim = self._auto_detect_feature_dim()
            log(INFO, f"Auto-detected feature dimension: {self.feature_dim}")
        else:
            self.feature_dim = feature_dim
        
        # Validate weights sum to 1.0
        total_weight = spectral_weight + delta_weight
        if not np.isclose(total_weight, 1.0):
            log(WARNING, f"Weights sum to {total_weight:.4f}, normalizing to 1.0")
            self.spectral_weight = spectral_weight / total_weight
            self.delta_weight = delta_weight / total_weight
        
        # Create root dataset loader
        self.root_loader = self._create_root_loader()
        
        # Create output directory for metrics
        self.metrics_output_dir = Path(self.config.output_dir) / "fera_visualize"
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)
        
        log(INFO, "═══════════════════════════════════════════════")
        log(INFO, f"   FeRA Visualize - Filter Variant: {self.filter_variant.upper()}")
        log(INFO, "═══════════════════════════════════════════════")
        log(INFO, f"Root dataset size: {self.root_size} samples")
        log(INFO, f"Feature dimension: {self.feature_dim}")
        log(INFO, f"Combined score weights: spectral={self.spectral_weight:.2f}, delta={self.delta_weight:.2f}")
        log(INFO, f"Parameter change threshold: {self.param_change_threshold}")
        log(INFO, f"Numerical epsilon: {self.epsilon}")
        log(INFO, f"Eigenvalue metrics: top_k={self.top_k_eigenvalues}, effective_rank_threshold={self.effective_rank_threshold:.1%}")
        log(INFO, f"Extraction layers: {self.extraction_layers}")
        if self.use_multi_layer:
            log(INFO, f"Multi-layer mode: combining with '{self.combine_layers_method}'")
        log(INFO, f"OOD root dataset: {self.use_ood_root_dataset}")
        if self.use_ood_root_dataset:
            log(INFO, f"  OOD dataset name: {self.ood_root_dataset_name or 'auto-detect'}")
        log(INFO, f"Output directory: {self.metrics_output_dir}")
        log(INFO, "")
        
        if self.filter_variant == "v1":
            log(INFO, "Filter Components (Variant 1 - Original):")
        log(INFO, f"  [1] Default Filter: {'ENABLED' if self.default_filter['enabled'] else 'DISABLED'}")
        if self.default_filter['enabled']:
            log(INFO, f"      Combined≤{int(self.default_filter['combined_threshold']*100)}%, "
                      f"TDA≤{int(self.default_filter['tda_threshold']*100)}%, "
                      f"MutualSim≥{int(self.default_filter['mutual_sim_threshold']*100)}%")
        log(INFO, f"  [2] Collusion Filter: {'ENABLED' if self.collusion_filter['enabled'] else 'DISABLED'}")
        if self.collusion_filter['enabled']:
            log(INFO, f"      Top {int(self.collusion_filter['tda_top_percent']*100)}% TDA AND "
                      f"Top {int(self.collusion_filter['mutual_sim_top_percent']*100)}% MutualSim")
        log(INFO, f"  [3] Outlier Filter: {'ENABLED' if self.outlier_filter['enabled'] else 'DISABLED'}")
        log(INFO, f"  [4] Scaled Norm Filter: {'ENABLED' if self.scaled_norm_filter['enabled'] else 'DISABLED'}")
        if self.scaled_norm_filter['enabled']:
            log(INFO, f"      Spectral ratio threshold: {self.scaled_norm_filter['spectral_ratio_threshold']}")
        elif self.filter_variant == "v2":
            log(INFO, "Filter Components (Variant 2 - Simplified):")
            log(INFO, "  [1] Consistency Filter: ALWAYS ON")
            log(INFO, "      Bottom-50% Combined AND Bottom-50% TDA")
            log(INFO, f"  [2] Collusion Filter: {'ENABLED' if self.collusion_filter['enabled'] else 'DISABLED'}")
            if self.collusion_filter['enabled']:
                log(INFO, "      Bottom-50% Combined AND Bottom-50% MutualSim")
            log(INFO, f"  [3] Norm-Inflation Filter: {'ENABLED' if self.scaled_norm_filter['enabled'] else 'DISABLED'}")
            if self.scaled_norm_filter['enabled']:
                log(INFO, "      Spectral Ratio > 100")
            log(INFO, "")
            log(INFO, "V2 Post-Processing (after flagging):")
            log(INFO, f"  Clipping: {'ENABLED' if self.enable_clipping else 'DISABLED'}")
            if self.enable_clipping:
                log(INFO, f"      Two-stage: (1) Median threshold, (2) {int(self.clip_percentage*100)}% of median")
            log(INFO, f"  Adaptive Noise: {'ENABLED' if self.enable_adaptive_noise else 'DISABLED'}")
            if self.enable_adaptive_noise:
                if self.noise_mode == "adaptive":
                    log(INFO, f"      Mode: Adaptive (lambda={self.noise_lambda}, scales with param std)")
                elif self.noise_mode == "fixed":
                    log(INFO, f"      Mode: Fixed (std={self.noise_std}, constant across params)")
                else:
                    log(INFO, f"      Mode: {self.noise_mode} (lambda={self.noise_lambda})")
        
        log(INFO, "═══════════════════════════════════════════════")
    
    def _create_root_loader(self) -> DataLoader:
        """Create root dataset loader for feature extraction."""
        ood_dataset = None
        
        if self.use_ood_root_dataset:
            if self.ood_root_dataset_name:
                # Use explicitly specified OOD dataset
                ood_dataset = load_ood_dataset(self.ood_root_dataset_name, self.config)
                log(INFO, f"Using explicit OOD dataset: {self.ood_root_dataset_name}")
            else:
                # Auto-detect OOD dataset based on main dataset
                ood_dataset = load_ood_dataset(self.config.dataset, self.config)
                log(INFO, f"Using auto-detected OOD dataset for {self.config.dataset}")
        
        return create_root_dataset_loader(
            testset=self.testset,
            root_size=self.root_size,
            batch_size=min(self.root_size, 64),
            num_workers=self.config.num_workers,
            device=self.device,
            ood_dataset=ood_dataset
        )
    
    def _auto_detect_feature_dim(self) -> int:
        """Auto-detect feature dimension via test forward pass."""
        # Get a sample batch from test set
        test_batch = next(iter(self.test_loader))
        if isinstance(test_batch, (list, tuple)):
            test_input = test_batch[0][:1]  # Take first sample
        else:
            test_input = test_batch[:1]
        test_input = test_input.to(self.device)
        
        # Hook to capture feature dimension
        captured_dim = [None]
        
        def capture_dim_hook(module, input, output):
            # Flatten spatial dimensions if needed
            if len(output.shape) > 2:
                output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                output = output.view(output.size(0), -1)
            captured_dim[0] = output.shape[1]
        
        # Register hook on penultimate layer
        hook_handle = self._register_penultimate_hook(self.global_model, capture_dim_hook)
        
        # Forward pass
        self.global_model.eval()
        with torch.no_grad():
            _ = self.global_model(test_input)
        
        # Remove hook
        hook_handle.remove()
        
        if captured_dim[0] is None:
            # Fallback: try to infer from model architecture
            model_name = self.config.model.lower()
            if 'resnet' in model_name:
                feature_dim = 512  # ResNet18 default
            elif 'mnist' in model_name or 'mnistnet' in model_name:
                feature_dim = 500  # MnistNet default
            elif 'vgg' in model_name:
                feature_dim = 512  # VGG default
            else:
                feature_dim = 512  # Default fallback
            log(WARNING, f"Could not auto-detect feature dim, using fallback: {feature_dim}")
            return feature_dim
        
        return captured_dim[0]
    
    def detect_anomalies(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using the configured filter variant.
        
        Routes to either V1 (original multi-component) or V2 (simplified) filter logic
        based on the filter_variant configuration.
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        if self.filter_variant == "v1":
            return self._detect_anomalies_v1(client_updates)
        elif self.filter_variant == "v2":
            return self._detect_anomalies_v2(client_updates)
        else:
            raise ValueError(
                f"Invalid filter_variant: {self.filter_variant}. Must be 'v1' or 'v2'."
            )
    
    def _detect_anomalies_v1(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using multi-component filtering (Variant 1 - Original Logic).
        
        Orchestrates multiple filter components and combines their results using UNION logic.
        Any filter can flag a client as malicious.
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        # Handle edge cases
        if len(client_updates) < 2:
            log(WARNING, "FeRA Visualize: Less than 2 clients, cannot perform detection")
            return [], [cid for cid, _, _ in client_updates]
        
        try:
            # Compute all metrics once (single source of truth)
            all_metrics = self._compute_all_metrics(client_updates)
            
            # Run each enabled filter component
            malicious_sets = []
            
            # Component 1: Default filter
            if self.default_filter['enabled']:
                mal_ids = self._filter_default(all_metrics, client_updates)
                malicious_sets.append(('Default', mal_ids))
            
            # Component 2: Collusion filter
            if self.collusion_filter['enabled']:
                mal_ids = self._filter_collusion(all_metrics, client_updates)
                malicious_sets.append(('Collusion', mal_ids))
            
            # Component 3: Outlier filter
            if self.outlier_filter['enabled']:
                mal_ids = self._filter_outliers(all_metrics, client_updates)
                malicious_sets.append(('Outlier', mal_ids))
            
            # Component 4: Scaled norm filter
            if self.scaled_norm_filter['enabled']:
                mal_ids = self._filter_scaled_norm(all_metrics, client_updates)
                malicious_sets.append(('ScaledNorm', mal_ids))
            
            # Combine results (UNION of all flagged clients)
            malicious_clients = self._combine_filter_results(malicious_sets)
            benign_clients = [cid for cid, _, _ in client_updates if cid not in malicious_clients]
            
            # Log detailed results
            self._log_filter_results(malicious_sets, malicious_clients, benign_clients)
            
            return malicious_clients, benign_clients
        
        except Exception as e:
            log(WARNING, f"FeRA Visualize: Detection failed with error: {str(e)}")
            log(WARNING, "FeRA Visualize: Falling back to no detection (all clients benign)")
            import traceback
            traceback.print_exc()
            return [], [cid for cid, _, _ in client_updates]
    
    def _detect_anomalies_v2(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using simplified filter logic (Variant 2).
        
        V2 uses three simplified filters:
        1. Consistency Filter: Bottom-50% Combined AND Bottom-50% TDA
        2. Collusion Filter: Bottom-50% Combined AND Bottom-50% Mutual Similarity
        3. Norm-Inflation Filter: Spectral Ratio > 100
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        # Handle edge cases
        if len(client_updates) < 2:
            log(WARNING, "FeRA Visualize V2: Less than 2 clients, cannot perform detection")
            return [], [cid for cid, _, _ in client_updates]
        
        try:
            # Compute all metrics once (single source of truth)
            all_metrics = self._compute_all_metrics(client_updates)
            
            # Run V2 filters
            malicious_sets = []
            
            # Component 1: Consistency filter (always on in V2)
            mal_ids = self._filter_consistency_v2(all_metrics, client_updates)
            malicious_sets.append(('Consistency', mal_ids))
            
            # Component 2: Collusion filter (optional)
            if self.collusion_filter['enabled']:
                mal_ids = self._filter_collusion_v2(all_metrics, client_updates)
                malicious_sets.append(('Collusion', mal_ids))
            
            # Component 3: Norm-Inflation filter (optional)
            if self.scaled_norm_filter['enabled']:
                mal_ids = self._filter_norm_inflation_v2(all_metrics, client_updates)
                malicious_sets.append(('NormInflation', mal_ids))
            
            # Combine results (UNION of all flagged clients)
            malicious_clients = self._combine_filter_results(malicious_sets)
            benign_clients = [cid for cid, _, _ in client_updates if cid not in malicious_clients]
            
            # Log detailed results
            self._log_filter_results(malicious_sets, malicious_clients, benign_clients)
            
            return malicious_clients, benign_clients
        
        except Exception as e:
            log(WARNING, f"FeRA Visualize V2: Detection failed with error: {str(e)}")
            log(WARNING, "FeRA Visualize V2: Falling back to no detection (all clients benign)")
            import traceback
            traceback.print_exc()
            return [], [cid for cid, _, _ in client_updates]
    
    def _rank_clients(self, metric_dict: Dict[int, float]) -> Dict[int, int]:
        """
        Rank clients by metric values in ascending order.
        
        Args:
            metric_dict: {client_id: metric_value}
        
        Returns:
            {client_id: rank} where rank 1 = lowest value
        """
        sorted_clients = sorted(metric_dict.items(), key=lambda x: x[1])
        return {cid: rank for rank, (cid, _) in enumerate(sorted_clients, start=1)}
    
    def _filter_default(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Default filter: Combined Score ≤ 50%, TDA ≤ 50%, Mutual Sim ≥ 70%
        ALL conditions must be met (AND logic)
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        combined_scores = all_metrics['combined_score']
        tda_scores = all_metrics['tda']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
        # Rank clients (ascending order: rank 1 = lowest value)
        combined_ranks = self._rank_clients(combined_scores)
        tda_ranks = self._rank_clients(tda_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
        # Calculate thresholds
        n = len(client_updates)
        combined_thresh = int(n * self.default_filter['combined_threshold'])
        tda_thresh = int(n * self.default_filter['tda_threshold'])
        mutual_sim_thresh = int(n * self.default_filter['mutual_sim_threshold']) + 1
        
        # Apply filter (ALL conditions must be met)
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= combined_thresh and
                tda_ranks[cid] <= tda_thresh and
                mutual_sim_ranks[cid] >= mutual_sim_thresh):
                malicious.add(cid)
        
        return malicious
    
    def _filter_collusion(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Collusion filter: Top 40% mutual_sim AND top 40% TDA
        Targets colluding attackers with high similarity
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        tda_scores = all_metrics['tda']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
        # Rank clients (ascending: rank 1 = lowest value)
        tda_ranks = self._rank_clients(tda_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
        # Calculate thresholds for TOP 40%
        # Top 40% means rank > 60th percentile (since rank 1 = lowest)
        n = len(client_updates)
        tda_top_thresh = int(n * (1.0 - self.collusion_filter['tda_top_percent']))
        mutual_sim_top_thresh = int(n * (1.0 - self.collusion_filter['mutual_sim_top_percent']))
        
        # Apply filter (BOTH conditions must be met)
        malicious = set()
        for cid, _, _ in client_updates:
            if (tda_ranks[cid] > tda_top_thresh and
                mutual_sim_ranks[cid] > mutual_sim_top_thresh):
                malicious.add(cid)
        
        return malicious
    
    def _filter_outliers(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Outlier filter: Flags clients at both extremes of metrics
        If client is in {highest OR lowest} mutual_sim 
        AND {highest OR lowest} TDA: Malicious
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        tda_scores = all_metrics['tda']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
        # Sort to get extremes
        tda_sorted = sorted(tda_scores.items(), key=lambda x: x[1])
        mutual_sim_sorted = sorted(mutual_sim_scores.items(), key=lambda x: x[1])
        
        # Get highest and lowest for each metric
        tda_lowest = tda_sorted[0][0]
        tda_highest = tda_sorted[-1][0]
        mutual_sim_lowest = mutual_sim_sorted[0][0]
        mutual_sim_highest = mutual_sim_sorted[-1][0]
        
        # Create sets of extremes
        tda_extremes = {tda_lowest, tda_highest}
        mutual_sim_extremes = {mutual_sim_lowest, mutual_sim_highest}
        
        # Flag clients in BOTH extreme sets
        malicious = tda_extremes & mutual_sim_extremes
        
        return malicious
    
    def _filter_scaled_norm(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Scaled Norm filter: Flags clients with extreme spectral norm ratios
        Detects norm-inflation attacks
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        spectral_norms = all_metrics['spectral_norm']
        
        # Compute median spectral norm for this round
        median_spectral = np.median(list(spectral_norms.values()))
        
        # Avoid division by zero
        if median_spectral < self.epsilon:
            return set()
        
        # Calculate ratios and flag high outliers
        threshold = self.scaled_norm_filter['spectral_ratio_threshold']
        malicious = set()
        
        for cid, spectral_norm in spectral_norms.items():
            spectral_ratio = spectral_norm / median_spectral
            if spectral_ratio > threshold:
                malicious.add(cid)
        
        return malicious
    
    def _filter_consistency_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Variant 2 Consistency Filter:
        Flags clients with bottom-50% Combined AND bottom-50% TDA.
        
        Targets clients with low variance (low combined score) and directional
        deviation (low TDA score).
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        combined_scores = all_metrics['combined_score']
        tda_scores = all_metrics['tda']
        
        # Rank clients (ascending: rank 1 = lowest value)
        combined_ranks = self._rank_clients(combined_scores)
        tda_ranks = self._rank_clients(tda_scores)
        
        # Bottom-50% threshold
        n = len(client_updates)
        thresh_50 = int(n * 0.50)
        
        # Apply filter (BOTH conditions must be met)
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= thresh_50 and 
                tda_ranks[cid] <= thresh_50):
                malicious.add(cid)
        
        return malicious
    
    def _filter_collusion_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Variant 2 Collusion Filter:
        Flags clients with bottom-50% Combined AND bottom-50% Mutual Similarity.
        
        Targets colluding attackers with low variance and low similarity to others.
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        combined_scores = all_metrics['combined_score']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
        # Rank clients (ascending: rank 1 = lowest value)
        combined_ranks = self._rank_clients(combined_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
        # Bottom-50% threshold
        n = len(client_updates)
        thresh_50 = int(n * 0.50)
        
        # Apply filter (BOTH conditions must be met)
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= thresh_50 and 
                mutual_sim_ranks[cid] <= thresh_50):
                malicious.add(cid)
        
        return malicious
    
    def _filter_norm_inflation_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Variant 2 Norm-Inflation Filter:
        Flags clients with spectral_ratio > 100.
        
        Detects norm-inflation and model-replacement style attacks.
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        spectral_norms = all_metrics['spectral_norm']
        
        # Compute median spectral norm
        median_spectral = np.median(list(spectral_norms.values()))
        
        # Avoid division by zero
        if median_spectral < self.epsilon:
            return set()
        
        # Flag clients with ratio > 100
        threshold = 100.0
        malicious = set()
        
        for cid, spectral_norm in spectral_norms.items():
            spectral_ratio = spectral_norm / median_spectral
            if spectral_ratio > threshold:
                malicious.add(cid)
        
        return malicious
    
    def _clip_malicious_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]],
        malicious_clients: List[int]
    ) -> List[Tuple[client_id, num_examples, StateDict]]:
        """
        Clip malicious client updates using two-stage adaptive thresholding.
        
        Stage 1: Clip to median norm, Stage 2: Clip to percentage of median.
        """
        if not malicious_clients:
            return client_updates
        
        malicious_set = set(malicious_clients)
        global_state_dict = dict(self.global_model.named_parameters())
        
        update_norms = []
        client_diffs = {}
        
        for client_id, num_examples, client_params in client_updates:
            diff_dict = {}
            flatten_weights = []
            
            for name, param in client_params.items():
                if name.endswith('num_batches_tracked'):
                    continue
                if name in global_state_dict:
                    diff = param.to(self.device) - global_state_dict[name]
                    diff_dict[name] = diff
                    if 'weight' in name or 'bias' in name:
                        flatten_weights.append(diff.view(-1))
            
            if flatten_weights:
                flatten_weights = torch.cat(flatten_weights)
                weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                update_norms.append(weight_diff_norm)
                client_diffs[client_id] = diff_dict
            else:
                update_norms.append(0.0)
                client_diffs[client_id] = diff_dict
        
        if not update_norms:
            log(WARNING, "FeRA Visualize V2: No update norms computed, skipping clipping")
            return client_updates
        
        clip_norm = np.median(update_norms)
        percentage_clip_norm = clip_norm * self.clip_percentage
        log(INFO, f"FeRA Visualize V2: Clipping thresholds - Median: {clip_norm:.6f}, "
                  f"{int(self.clip_percentage*100)}%: {percentage_clip_norm:.6f}")
        
        stage1_clipped = 0
        stage2_clipped = 0
        
        for client_id, num_examples, client_params in client_updates:
            if client_id not in malicious_set:
                continue
            
            if client_id not in client_diffs:
                continue
            
            diff_dict = client_diffs[client_id]
            flatten_weights = []
            
            for name, diff in diff_dict.items():
                if 'weight' in name or 'bias' in name:
                    flatten_weights.append(diff.view(-1))
            
            if not flatten_weights:
                continue
            
            flatten_weights = torch.cat(flatten_weights)
            current_norm = torch.linalg.norm(flatten_weights, ord=2).item()
            original_norm = current_norm
            
            if current_norm > clip_norm:
                scaling_factor = clip_norm / current_norm
                
                for name, param in client_params.items():
                    if name in diff_dict:
                        param.data.copy_(global_state_dict[name] + diff_dict[name] * scaling_factor)
                
                for name, param in client_params.items():
                    if name in global_state_dict:
                        diff_dict[name] = param.to(self.device) - global_state_dict[name]
                
                flatten_weights = []
                for name, diff in diff_dict.items():
                    if 'weight' in name or 'bias' in name:
                        flatten_weights.append(diff.view(-1))
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    current_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                
                stage1_clipped += 1
                log(INFO, f"FeRA Visualize V2: Stage 1 - Clipped client {client_id} "
                          f"(norm: {original_norm:.6f} -> {current_norm:.6f})")
            
            if current_norm > percentage_clip_norm:
                scaling_factor = percentage_clip_norm / current_norm
                
                for name, param in client_params.items():
                    if name in diff_dict:
                        param.data.copy_(global_state_dict[name] + diff_dict[name] * scaling_factor)
                
                for name, param in client_params.items():
                    if name in global_state_dict:
                        diff_dict[name] = param.to(self.device) - global_state_dict[name]
                
                flatten_weights = []
                for name, diff in diff_dict.items():
                    if 'weight' in name or 'bias' in name:
                        flatten_weights.append(diff.view(-1))
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    final_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                else:
                    final_norm = percentage_clip_norm
                
                stage2_clipped += 1
                log(INFO, f"FeRA Visualize V2: Stage 2 - Clipped client {client_id} "
                          f"(norm: {current_norm:.6f} -> {final_norm:.6f})")
            else:
                stage2_clipped += 1
                log(INFO, f"FeRA Visualize V2: Stage 2 - Client {client_id} already below threshold "
                          f"(norm: {current_norm:.6f} <= {percentage_clip_norm:.6f})")
        
        log(INFO, f"FeRA Visualize V2: Stage 1 clipped {stage1_clipped}/{len(malicious_clients)} malicious clients")
        log(INFO, f"FeRA Visualize V2: Stage 2 clipped {stage2_clipped}/{len(malicious_clients)} malicious clients")
        return client_updates
    
    @torch.no_grad()
    def _add_adaptive_noise(self, clip_norm: float):
        """
        Add noise to global model parameters.
        
        Supports adaptive (scales with param std and clip_norm) and fixed modes.
        """
        noise_added_count = 0
        skipped_ignore = 0
        skipped_buffers = 0
        total_noise_magnitude = 0.0
        
        all_params = list(self.global_model.named_parameters())
        total_params = len(all_params)
        
        for name, param in all_params:
            if any(pattern in name for pattern in self.ignore_weights):
                skipped_ignore += 1
                continue
            
            if "running" in name or "num_batches_tracked" in name:
                skipped_buffers += 1
                continue
            
            if self.noise_mode == "adaptive":
                param_std = torch.std(param).item()
                noise_std_val = self.noise_lambda * clip_norm * param_std
            elif self.noise_mode == "fixed":
                noise_std_val = self.noise_std
            else:
                log(WARNING, f"Invalid noise_mode '{self.noise_mode}', defaulting to fixed")
                noise_std_val = self.noise_std
            
            noise = torch.normal(0, noise_std_val, param.shape, device=param.device)
            param.data.add_(noise)
            noise_added_count += 1
            total_noise_magnitude += torch.abs(noise).sum().item()
        
        # Log with mode-specific details
        if self.noise_mode == "adaptive":
            log(INFO, f"FeRA Visualize V2: Added adaptive noise to {noise_added_count}/{total_params} parameters "
                      f"(lambda={self.noise_lambda}, clip_norm={clip_norm:.6f}, total_noise_mag={total_noise_magnitude:.2f})")
        elif self.noise_mode == "fixed":
            log(INFO, f"FeRA Visualize V2: Added fixed noise to {noise_added_count}/{total_params} parameters "
                      f"(std={self.noise_std}, total_noise_mag={total_noise_magnitude:.2f})")
        
        if skipped_ignore + skipped_buffers > 0:
            log(INFO, f"FeRA Visualize V2: Skipped {skipped_ignore} ignore_weights, {skipped_buffers} buffers")
    
    def _combine_filter_results(
        self,
        malicious_sets: List[Tuple[str, set]]
    ) -> List[int]:
        """
        Combine results from multiple filters using UNION logic
        
        Args:
            malicious_sets: List of (filter_name, set_of_client_ids)
        
        Returns:
            Combined list of malicious client IDs (sorted)
        """
        combined = set()
        for filter_name, mal_ids in malicious_sets:
            combined.update(mal_ids)
        return sorted(list(combined))
    
    def _log_filter_results(
        self,
        malicious_sets: List[Tuple[str, set]],
        malicious_clients: List[int],
        benign_clients: List[int]
    ):
        """
        Log detailed filter results showing contribution of each component
        
        Args:
            malicious_sets: List of (filter_name, set_of_client_ids)
            malicious_clients: Combined list of malicious clients
            benign_clients: List of benign clients
        """
        log(INFO, "")
        log(INFO, "─" * 120)
        log(INFO, "MULTI-COMPONENT FILTER RESULTS:")
        log(INFO, "─" * 120)
        
        # Show per-filter results
        for filter_name, mal_ids in malicious_sets:
            log(INFO, f"{filter_name:12s} Filter: {sorted(list(mal_ids))}")
        
        log(INFO, "─" * 120)
        log(INFO, f"Combined Malicious (UNION): {malicious_clients}")
        log(INFO, f"Predicted Benign:           {benign_clients}")
        
        # Get ground truth for THIS round only
        ground_truth_this_round = self.get_clients_info(self.current_round)["malicious_clients"]
        ground_truth_sorted = sorted(ground_truth_this_round)
        log(INFO, f"Ground Truth Malicious (this round): {ground_truth_sorted}")
        log(INFO, "─" * 120)
        log(INFO, "")
    
    def aggregate_client_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ):
        """
        Override aggregation to compute metrics, visualize, and perform filtering.
        
        Process:
        1. Compute all 6 metrics for each client
        2. Log ranked tables to console
        3. Save ranked tables and master CSV
        4. Call parent class which performs detection and aggregation (filters malicious clients)
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            
        Returns:
            True if aggregation successful, False otherwise
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False
        
        log(INFO, "")
        log(INFO, "═══════════════════════════════════════════════")
        log(INFO, f"  FeRA Visualize Metrics - Round {self.current_round}")
        log(INFO, "═══════════════════════════════════════════════")
        
        # Start defense computation timing
        defense_start_time = time.perf_counter()
        
        try:
            # Compute all metrics
            metrics_start_time = time.perf_counter()
            all_metrics = self._compute_all_metrics(client_updates)
            metrics_time = time.perf_counter() - metrics_start_time
            
            # Log ranked tables and save CSVs only if visualizations enabled
            if self.enable_visualizations:
                self._log_metrics_tables(all_metrics, self.current_round)
                self._save_metrics_csvs(all_metrics, self.current_round)
                log(INFO, "")
                log(INFO, f"✓ Metrics saved to: {self.metrics_output_dir}")
                log(INFO, "═══════════════════════════════════════════════")
                log(INFO, "")
            
        except Exception as e:
            log(WARNING, f"Failed to compute metrics: {str(e)}")
            log(WARNING, "Proceeding with aggregation anyway...")
            metrics_time = 0.0
            all_metrics = None
        
        # Time detection/filtering phase
        detection_start_time = time.perf_counter()
        
        # For V2: Apply clipping and optional noise, then aggregate all clients
        # For V1: Use parent class behavior (filter malicious, aggregate only benign)
        if self.filter_variant == "v2":
            result = self._aggregate_v2_with_clipping(client_updates)
        else:
        # Call parent class which performs detection and filtered aggregation
            result = super().aggregate_client_updates(client_updates)
        
        detection_time = time.perf_counter() - detection_start_time
        total_defense_time = time.perf_counter() - defense_start_time
        
        # Log defense computation timing (metrics + filtering, excluding aggregation)
        log(INFO, f"FeRA Visualize Defense Computation Time:")
        log(INFO, f"  Metrics computation: {metrics_time:.3f} seconds")
        log(INFO, f"  Detection/Filtering: {detection_time:.3f} seconds")
        log(INFO, f"  Total defense time: {total_defense_time:.3f} seconds")
        
        return result
    
    def _aggregate_v2_with_clipping(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> bool:
        """
        V2 aggregation with clipping and optional noise.
        
        Process:
        1. Detect malicious clients using V2 filters
        2. Clip malicious updates (if enabled)
        3. Aggregate ALL clients (benign + clipped malicious)
        4. Add adaptive noise to global model (if enabled)
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            True if aggregation successful, False otherwise
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False
        
        # Step 1: Detect malicious clients using V2 filters
        malicious_clients, benign_clients = self.detect_anomalies(client_updates)
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        detection_metrics = self.evaluate_detection(malicious_clients, true_malicious_clients, len(client_updates))
        
        # Step 2: Clip malicious updates (if enabled)
        clip_norm = None
        if self.enable_clipping and malicious_clients:
            # Compute clip_norm before clipping (needed for noise)
            global_state_dict = dict(self.global_model.named_parameters())
            update_norms = []
            
            for client_id, num_examples, client_params in client_updates:
                flatten_weights = []
                for name, param in client_params.items():
                    if name.endswith('num_batches_tracked'):
                        continue
                    if name in global_state_dict:
                        diff = param.to(self.device) - global_state_dict[name]
                        if 'weight' in name or 'bias' in name:
                            flatten_weights.append(diff.view(-1))
                
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                    update_norms.append(weight_diff_norm)
            
            if update_norms:
                clip_norm = np.median(update_norms)
            
            # Apply clipping
            client_updates = self._clip_malicious_updates(client_updates, malicious_clients)
        
        # Step 3: Aggregate ALL clients (benign + clipped malicious) using FedAvg
        # Use parent's UnweightedFedAvgServer.aggregate_client_updates directly
        from backfed.servers.fedavg_server import UnweightedFedAvgServer
        aggregation_success = UnweightedFedAvgServer.aggregate_client_updates(self, client_updates)
        
        if not aggregation_success:
            return False
        
        # Step 4: Add adaptive noise (if enabled)
        if self.enable_adaptive_noise and clip_norm is not None:
            self._add_adaptive_noise(clip_norm)
        elif self.enable_adaptive_noise:
            # If clipping was disabled but noise is enabled, use a default clip_norm
            # Compute it from current update norms
            global_state_dict = dict(self.global_model.named_parameters())
            update_norms = []
            
            for client_id, num_examples, client_params in client_updates:
                flatten_weights = []
                for name, param in client_params.items():
                    if name.endswith('num_batches_tracked'):
                        continue
                    if name in global_state_dict:
                        diff = param.to(self.device) - global_state_dict[name]
                        if 'weight' in name or 'bias' in name:
                            flatten_weights.append(diff.view(-1))
                
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                    update_norms.append(weight_diff_norm)
            
            if update_norms:
                clip_norm = np.median(update_norms)
                self._add_adaptive_noise(clip_norm)
            else:
                log(WARNING, "FeRA Visualize V2: Cannot compute clip_norm for noise, skipping noise addition")
        
        return True
    
    def _compute_all_metrics(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute all 6 metrics for each client.
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            
        Returns:
            Dictionary mapping metric names to client_id -> value dicts
        """
        all_metrics = {}
        
        # Load client models
        client_models = self._load_client_models(client_updates)
        client_ids = list(client_models.keys())
        
        # Compute representation-based metrics (spectral, delta, combined)
        client_representations = self._extract_all_representations(client_models)
        global_representation = self._extract_global_representation()
        
        # Compute comprehensive eigenvalue metrics
        eigenvalue_metrics = self._compute_eigenvalue_metrics(
            client_representations, global_representation
        )
        
        # Extract spectral_norm for backward compatibility and normalization
        spectral_scores = eigenvalue_metrics['spectral_norm']
        
        # Store all eigenvalue metrics
        for metric_name, metric_dict in eigenvalue_metrics.items():
            all_metrics[metric_name] = metric_dict
        
        # Compute delta norms (separate from eigenvalue metrics)
        delta_scores = self._compute_delta_norms(
            client_representations, global_representation
        )
        
        # Normalize and combine (using spectral_norm for backward compatibility)
        spectral_normed = self._robust_normalize(spectral_scores)
        delta_normed = self._robust_normalize(delta_scores)
        combined_scores = self._compute_combined_scores(spectral_normed, delta_normed)
        
        # Store normalized and combined scores
        all_metrics['spectral_norm_normed'] = spectral_normed
        all_metrics['delta_norm'] = delta_scores
        all_metrics['delta_norm_normed'] = delta_normed
        all_metrics['combined_score'] = combined_scores
        
        # Compute parameter-based metrics (TDA, mutual similarity)
        tda_scores = self._compute_tda_scores(client_updates)
        mutual_similarity = self._compute_mutual_similarity(client_updates)
        
        all_metrics['tda'] = tda_scores
        all_metrics['mutual_similarity'] = mutual_similarity
        
        return all_metrics
    
    def _load_client_models(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, nn.Module]:
        """Load client models from state dictionaries."""
        client_models = {}
        
        for cid, _, state_dict in client_updates:
            model = get_model(
                model_name=self.config.model,
                num_classes=self.config.num_classes,
                dataset_name=self.config.dataset
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            client_models[cid] = model
        
        return client_models
    
    def _extract_all_representations(
        self,
        client_models: Dict[int, nn.Module]
    ) -> Dict[int, torch.Tensor]:
        """
        Extract representations from clients, combining multiple layers if configured.
        
        Args:
            client_models: Dictionary of client_id -> model
            
        Returns:
            Dictionary of client_id -> representation tensor [root_size, feature_dim]
        """
        
        if not self.use_multi_layer:
            # Single layer extraction (existing behavior)
            return {cid: self._extract_representation_single_model(model) 
                    for cid, model in client_models.items()}
        
        # Multi-layer extraction
        all_layer_representations = {}
        for layer_name in self.extraction_layers:
            layer_representations = {}
            for cid, model in client_models.items():
                with torch.no_grad():
                    representations = []
                    for batch in self.root_loader:
                        inputs, _ = batch
                        inputs = inputs.to(self.device)
                        features = self._extract_features_from_layer(model, inputs, layer_name)
                        representations.append(features.cpu())
                    layer_representations[cid] = torch.cat(representations, dim=0)
            all_layer_representations[layer_name] = layer_representations
        
        # Combine representations across layers
        return self._combine_layer_representations(all_layer_representations)
    
    def _combine_layer_representations(
        self,
        all_layer_representations: Dict[str, Dict[int, torch.Tensor]]
    ) -> Dict[int, torch.Tensor]:
        """Combine representations from multiple layers."""
        client_ids = list(next(iter(all_layer_representations.values())).keys())
        combined = {}
        
        for cid in client_ids:
            layer_reps = [all_layer_representations[layer][cid] for layer in self.extraction_layers]
            
            if self.combine_layers_method == 'mean':
                combined[cid] = torch.stack(layer_reps).mean(dim=0)
            elif self.combine_layers_method == 'max':
                combined[cid] = torch.stack(layer_reps).max(dim=0)[0]
            elif self.combine_layers_method == 'min':
                combined[cid] = torch.stack(layer_reps).min(dim=0)[0]
            else:
                log(WARNING, f"Unknown combine method '{self.combine_layers_method}', using mean")
                combined[cid] = torch.stack(layer_reps).mean(dim=0)
        
        return combined
    
    def _extract_representation_single_model(
        self,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Extract penultimate layer features from a single model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tensor of shape [root_size, feature_dim]
        """
        features_list = []
        
        # Hook to capture penultimate layer
        def hook_fn(module, input, output):
            # Flatten spatial dimensions if needed (e.g., conv layers)
            if len(output.shape) > 2:
                output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                output = output.view(output.size(0), -1)
            features_list.append(output.detach())
        
        # Register hook on penultimate layer
        hook_handle = self._register_penultimate_hook(model, hook_fn)
        
        # Forward pass through root dataset
        model.eval()
        with torch.no_grad():
            for batch_data, _ in self.root_loader:
                batch_data = batch_data.to(self.device)
                _ = model(batch_data)
        
        # Remove hook
        hook_handle.remove()
        
        # Concatenate all features
        representations = torch.cat(features_list, dim=0)
        
        return representations
    
    def _register_penultimate_hook(
        self,
        model: nn.Module,
        hook_fn
    ):
        """
        Register hook on the penultimate layer (before final classification).
        
        Args:
            model: PyTorch model
            hook_fn: Hook function to register
            
        Returns:
            Hook handle
        """
        # Get model architecture specific penultimate layer
        model_name = self.config.model.lower()
        
        if 'resnet' in model_name:
            # ResNet: avgpool output
            return model.avgpool.register_forward_hook(hook_fn)
        elif 'vgg' in model_name:
            # VGG: last layer before classifier
            return model.features.register_forward_hook(hook_fn)
        elif 'mnist' in model_name or 'mnistnet' in model_name:
            # MNISTNet: typically has 'features' or penultimate FC
            if hasattr(model, 'features'):
                return model.features.register_forward_hook(hook_fn)
            else:
                # Find last layer before final FC
                layers = list(model.children())
                return layers[-2].register_forward_hook(hook_fn)
        else:
            # Default: try to find avgpool or last conv layer
            if hasattr(model, 'avgpool'):
                return model.avgpool.register_forward_hook(hook_fn)
            else:
                # Fall back to layer before final FC
                layers = list(model.children())
                return layers[-2].register_forward_hook(hook_fn)
    
    def _extract_global_representation(self) -> torch.Tensor:
        """
        Extract penultimate layer representation from global model.
        
        Returns:
            Tensor of shape [root_size, feature_dim]
        """
        return self._extract_representation_single_model(self.global_model)
    
    def _extract_features_from_layer(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Extract features from a specific layer using hooks."""
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        target_layer = self._get_target_layer(model, layer_name)
        
        if target_layer is None:
            log(WARNING, f"Could not find layer '{layer_name}', using penultimate")
            return self._extract_representation_single_model(model)
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            _ = model(inputs)
            if not features:
                raise RuntimeError(f"Hook did not capture features for layer '{layer_name}'")
            
            output_features = features[0]
            if len(output_features.shape) > 2:
                output_features = output_features.flatten(1)
            
            return output_features
        finally:
            handle.remove()
    
    def _get_target_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get target layer from model."""
        if layer_name == 'penultimate':
            layers = list(model.children())
            if len(layers) >= 2:
                return layers[-2]
        elif layer_name in ['layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                return getattr(model, layer_name)
        elif hasattr(model, layer_name):
            return getattr(model, layer_name)
        
        return None
    
    def _compute_eigenvalue_metrics(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute eigenvalue-based metrics from delta covariance.
        
        Returns metrics including spectral_norm, trace, top-k eigenvalues, effective_rank,
        condition_number, eigenvalue_entropy, and eigenvalue_decay_rate.
        """
        # Initialize all metric dictionaries
        metrics = {
            'spectral_norm': {},
            'trace': {},
            'effective_rank': {},
            'condition_number': {},
            'eigenvalue_entropy': {},
            'eigenvalue_decay_rate': {}
        }
        
        # Initialize top-k eigenvalue dictionaries
        for k in range(1, self.top_k_eigenvalues + 1):
            metrics[f'eigenvalue_{k}'] = {}
        
        n = global_representation.shape[0]  # Should be root_size
        
        for cid, client_repr in client_representations.items():
            try:
                # Compute delta
                delta = client_repr - global_representation  # [n, D]
                
                # Center delta
                delta_centered = delta - delta.mean(dim=0, keepdim=True)  # [n, D]
                
                # Compute covariance matrix
                cov_matrix = (delta_centered.T @ delta_centered) / (n - 1)  # [D, D]
                
                # Compute all eigenvalues (sorted ascending)
                eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                
                # Check for empty eigenvalues
                if eigenvalues.numel() == 0:
                    log(WARNING, f"Empty eigenvalues for client {cid}, using defaults")
                    for key in metrics:
                        metrics[key][cid] = 0.0
                    continue
                
                # Sort descending (largest first)
                eigenvalues = torch.sort(eigenvalues, descending=True)[0]
                eigenvalues_np = eigenvalues.cpu().numpy()
                
                # Remove negative eigenvalues (numerical artifacts, should be non-negative for covariance)
                eigenvalues_np = np.maximum(eigenvalues_np, 0.0)
                
                # Spectral norm (λ_max)
                spectral_norm = float(eigenvalues_np[0])
                metrics['spectral_norm'][cid] = spectral_norm
                
                # Trace (sum of all eigenvalues)
                trace = float(np.sum(eigenvalues_np))
                metrics['trace'][cid] = trace
                
                # Top-k eigenvalues
                k = min(self.top_k_eigenvalues, len(eigenvalues_np))
                for i in range(1, self.top_k_eigenvalues + 1):
                    if i <= k:
                        metrics[f'eigenvalue_{i}'][cid] = float(eigenvalues_np[i - 1])
                    else:
                        metrics[f'eigenvalue_{i}'][cid] = 0.0
                
                # Effective rank (number of eigenvalues needed for threshold% variance)
                if trace > self.epsilon:
                    cumsum = np.cumsum(eigenvalues_np)
                    cumsum_normalized = cumsum / trace
                    # Find first index where cumulative variance >= threshold
                    # searchsorted with side='left' returns the first index where value >= threshold
                    effective_rank_idx = np.searchsorted(cumsum_normalized, self.effective_rank_threshold, side='left')
                    # Convert from 0-indexed to 1-indexed rank, ensure it's at least 1
                    effective_rank = min(max(effective_rank_idx + 1, 1), len(eigenvalues_np))
                else:
                    effective_rank = 0
                metrics['effective_rank'][cid] = float(effective_rank)
                
                # Condition number (λ_max / λ_min)
                if len(eigenvalues_np) > 0:
                    lambda_min = eigenvalues_np[-1]
                    condition_num = spectral_norm / (lambda_min + self.epsilon)
                else:
                    condition_num = 0.0
                metrics['condition_number'][cid] = float(condition_num)
                
                # Eigenvalue entropy (Shannon entropy of normalized distribution)
                if trace > self.epsilon:
                    # Normalize eigenvalues to sum to 1
                    eigenvalues_normalized = eigenvalues_np / trace
                    # Compute entropy: -sum(p * log(p + epsilon))
                    # Only consider non-zero eigenvalues
                    non_zero_mask = eigenvalues_normalized > self.epsilon
                    if np.any(non_zero_mask):
                        p = eigenvalues_normalized[non_zero_mask]
                        entropy = -np.sum(p * np.log(p + self.epsilon))
                    else:
                        entropy = 0.0
                else:
                    entropy = 0.0
                metrics['eigenvalue_entropy'][cid] = float(entropy)
                
                # Eigenvalue decay rate (λ_2 / λ_max)
                if len(eigenvalues_np) >= 2 and spectral_norm > self.epsilon:
                    decay_rate = float(eigenvalues_np[1] / (spectral_norm + self.epsilon))
                else:
                    decay_rate = 0.0
                metrics['eigenvalue_decay_rate'][cid] = decay_rate
                
            except Exception as e:
                log(WARNING, f"Eigenvalue metrics computation failed for client {cid}: {e}")
                # Set all metrics to 0.0 on failure
                for key in metrics:
                    metrics[key][cid] = 0.0
        
        return metrics
    
    def _compute_delta_norms(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[int, float]:
        """Compute Frobenius norm of representation difference."""
        delta_norms = {}
        
        for cid, client_repr in client_representations.items():
            # Compute delta
            delta = client_repr - global_representation
            
            # Compute Frobenius norm
            delta_norm = torch.linalg.norm(delta, ord='fro').item()
            delta_norms[cid] = delta_norm
        
        return delta_norms
    
    def _robust_normalize(
        self,
        scores: Dict[int, float]
    ) -> Dict[int, float]:
        """Robust normalization using median and IQR."""
        if not scores:
            return {}
        
        score_values = np.array(list(scores.values()))
        
        # Compute median and IQR
        median = np.median(score_values)
        q1 = np.percentile(score_values, 25)
        q3 = np.percentile(score_values, 75)
        iqr = q3 - q1
        
        # Normalize
        normalized_scores = {}
        for cid, score in scores.items():
            normalized = (score - median) / (iqr + self.epsilon)
            normalized_scores[cid] = float(normalized)
        
        return normalized_scores
    
    def _compute_combined_scores(
        self,
        spectral_normed: Dict[int, float],
        delta_normed: Dict[int, float]
    ) -> Dict[int, float]:
        """Compute weighted combination of normalized spectral and delta norms."""
        combined_scores = {}
        
        for cid in spectral_normed.keys():
            combined = (self.spectral_weight * spectral_normed[cid] + 
                       self.delta_weight * delta_normed[cid])
            combined_scores[cid] = combined
        
        return combined_scores
    
    def _compute_tda_scores(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, float]:
        """
        Compute Temporal Direction Alignment (TDA) scores.
        
        Measures cosine similarity between client and global models.
        Returns scores in [0, 1] range where 1.0 = aligned, 0.0 = opposite.
        """
        tda_scores = {}
        
        # Flatten global model parameters
        global_params = self._flatten_state_dict(self.global_model.state_dict())
        global_norm = torch.linalg.norm(global_params).clamp(min=self.epsilon)
        
        for cid, _, state_dict in client_updates:
            try:
                # Flatten client model parameters
                client_params = self._flatten_state_dict(state_dict)
                client_norm = torch.linalg.norm(client_params).clamp(min=self.epsilon)
                
                # Compute cosine similarity between client model and global model
                dot_product = torch.dot(client_params, global_params)
                cosine_sim = (dot_product / (client_norm * global_norm)).item()
                
                # Normalize to [0, 1] range
                tda_score = (cosine_sim + 1.0) / 2.0
                tda_scores[cid] = tda_score
                
            except Exception as e:
                log(WARNING, f"TDA computation failed for client {cid}: {e}")
                tda_scores[cid] = 0.5  # Neutral score on failure
        
        return tda_scores
    
    def _compute_mutual_similarity(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, float]:
        """
        Compute mutual similarity (maximum pairwise cosine similarity).
        
        Computes the maximum cosine similarity between each client's update
        and all other clients' updates.
        """
        mutual_scores = {}
        
        # Flatten global model
        global_params = self._flatten_state_dict(self.global_model.state_dict())
        
        # Collect all update vectors
        client_ids = []
        update_vectors = []
        
        for cid, _, state_dict in client_updates:
            try:
                client_params = self._flatten_state_dict(state_dict)
                update_vector = client_params - global_params
                
                client_ids.append(cid)
                update_vectors.append(update_vector)
            except Exception as e:
                log(WARNING, f"Failed to compute update for client {cid}: {e}")
                continue
        
        if len(update_vectors) < 2:
            return {cid: 0.0 for cid in client_ids}
        
        U = torch.stack(update_vectors, dim=0)
        U_norms = torch.linalg.norm(U, dim=1, keepdim=True).clamp(min=self.epsilon)
        U_normalized = U / U_norms
        S = U_normalized @ U_normalized.T
        
        N = len(client_ids)
        for i, cid in enumerate(client_ids):
            S_masked = S[i].clone()
            S_masked[i] = -1e9
            mutual_sim = S_masked.max().item()
            mutual_scores[cid] = mutual_sim
        
        return mutual_scores
    
    def _flatten_state_dict(self, state_dict: StateDict) -> torch.Tensor:
        """Flatten all parameters into a single 1D tensor."""
        flat_params = []
        for key in sorted(state_dict.keys()):
            param = state_dict[key]
            if key in self.ignore_weights:
                continue
            if isinstance(param, torch.Tensor):
                param = param.to(self.device)
            flat_params.append(param.flatten())
        
        if not flat_params:
            return torch.tensor([], device=self.device)
        
        return torch.cat(flat_params)
    
    def _log_metrics_tables(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        round_num: int
    ):
        """Log ranked metric tables to console."""
        # Get ground truth labels
        ground_truth = self.client_manager.get_malicious_clients()
        
        # Get all client IDs
        client_ids = list(all_metrics['spectral_norm'].keys())
        
        log(INFO, "")
        log(INFO, "╔═══════════════════════════════════════════════╗")
        log(INFO, f"║  Ranked Metric Tables - Round {round_num:4d}          ║")
        log(INFO, "╚═══════════════════════════════════════════════╝")
        
        # Metrics to display in ranked order (ascending)
        display_metrics = [
            ('spectral_norm', 'Spectral Norm'),
            ('delta_norm', 'Delta Norm'),
            ('combined_score', 'Combined Score'),
            ('tda', 'TDA (Temporal Direction Alignment) [0-1]'),
            ('mutual_similarity', 'Mutual Similarity'),
            ('trace', 'Trace (Total Variance)'),
            ('effective_rank', f'Effective Rank ({self.effective_rank_threshold:.0%} variance)'),
            ('condition_number', 'Condition Number (λ_max/λ_min)'),
            ('eigenvalue_entropy', 'Eigenvalue Entropy'),
            ('eigenvalue_decay_rate', 'Eigenvalue Decay Rate (λ_2/λ_max)')
        ]
        
        # Add top-k eigenvalues to display list
        for k in range(1, self.top_k_eigenvalues + 1):
            display_metrics.append((f'eigenvalue_{k}', f'Eigenvalue {k}'))
        
        for metric_key, metric_name in display_metrics:
            # Skip if metric doesn't exist in all_metrics
            if metric_key not in all_metrics:
                continue
                
            log(INFO, "")
            log(INFO, f"─── {metric_name} (Ascending) ───")
            log(INFO, "Rank  Client    Value        Label")
            log(INFO, "────  ────────  ───────────  ───────")
            
            # Sort by metric value (ascending)
            metric_values = all_metrics[metric_key]
            sorted_clients = sorted(metric_values.items(), key=lambda x: x[1])
            
            # Display top 15 (or all if fewer)
            display_count = min(15, len(sorted_clients))
            for rank, (cid, value) in enumerate(sorted_clients[:display_count], start=1):
                label = "MAL" if cid in ground_truth else "BEN"
                log(INFO, f"{rank:4d}  {cid:8d}  {value:11.6f}  {label:7s}")
            
            if len(sorted_clients) > display_count:
                log(INFO, f"      ... ({len(sorted_clients) - display_count} more clients)")
        
        log(INFO, "")
        log(INFO, "═══════════════════════════════════════════════")
    
    def _save_metrics_csvs(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        round_num: int
    ):
        """Save metrics to cumulative CSV file (all_rounds_metrics.csv)."""
        # Get ground truth labels
        ground_truth = self.client_manager.get_malicious_clients()
        
        # Get all client IDs
        client_ids = list(all_metrics['spectral_norm'].keys())
        
        # Create label mapping
        labels = {cid: 'malicious' if cid in ground_truth else 'benign' 
                 for cid in client_ids}
        
        # Path to cumulative CSV file
        cumulative_csv = self.metrics_output_dir / "all_rounds_metrics.csv"
        
        # Check if this is the first round (file doesn't exist yet)
        is_first_round = not cumulative_csv.exists()
        
        # Open file in append mode
        with open(cumulative_csv, 'a') as f:
            if is_first_round:
                # Write ground truth malicious clients header
                mal_clients = ','.join(map(str, sorted(ground_truth)))
                f.write(f"Ground truth (GT) malicious [{mal_clients}]\n\n")
            
            # Write round header
            f.write(f"Round {round_num}\n")
            
            # Write CSV header
            # Base metrics
            f.write("client_id,label,spectral_norm,spectral_norm_normed,")
            f.write("delta_norm,delta_norm_normed,combined_score,tda,")
            f.write("mutual_similarity,")
            # Eigenvalue metrics
            f.write("trace,")
            # Top-k eigenvalues
            for k in range(1, self.top_k_eigenvalues + 1):
                f.write(f"eigenvalue_{k},")
            # Other eigenvalue metrics
            f.write("effective_rank,condition_number,eigenvalue_entropy,eigenvalue_decay_rate\n")
            
            # Write data for all clients
            for cid in client_ids:
                # Base metrics
                f.write(f"{cid},{labels[cid]},")
                f.write(f"{all_metrics.get('spectral_norm', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('spectral_norm_normed', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('delta_norm', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('delta_norm_normed', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('combined_score', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('tda', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('mutual_similarity', {}).get(cid, 0.0)},")
                # Eigenvalue metrics
                f.write(f"{all_metrics.get('trace', {}).get(cid, 0.0)},")
                # Top-k eigenvalues
                for k in range(1, self.top_k_eigenvalues + 1):
                    f.write(f"{all_metrics.get(f'eigenvalue_{k}', {}).get(cid, 0.0)},")
                # Other eigenvalue metrics
                f.write(f"{all_metrics.get('effective_rank', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('condition_number', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('eigenvalue_entropy', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('eigenvalue_decay_rate', {}).get(cid, 0.0)}\n")
            
            # Add blank line after round data
            f.write("\n")
        
        log(INFO, f"✓ Appended Round {round_num} data to all_rounds_metrics.csv")

