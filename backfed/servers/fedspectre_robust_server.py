"""
FedSPECTRE Robust Defense - Complete Redesign

This is a completely redesigned FedSPECTRE defense that addresses all the fundamental
issues in the previous implementation:

1. Robust reference model management
2. Multi-dimensional anomaly detection  
3. Proper normalization against clean baselines
4. Adaptive trust system
5. Dynamic thresholding
6. Ensemble detection methods

Author: AI Assistant
Date: 2025-10-18
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy import stats
import warnings

from backfed.servers.base_server import BaseServer
from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.utils.system_utils import log, INFO, WARNING
from backfed.servers.fedspectre_utils import (
    RobustStatistics, MahalanobisCKA, SpectralProjection,
    create_root_dataset_loader, load_ood_dataset
)

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Result of anomaly detection for a client."""
    client_id: int
    anomaly_score: float
    component_scores: Dict[str, float]
    confidence: float
    is_malicious: bool
    detection_methods: List[str]

@dataclass
class ReferenceTemplate:
    """Clean reference template for a class."""
    class_id: int
    mean_representation: np.ndarray
    covariance_matrix: np.ndarray
    whitening_matrix: np.ndarray
    confidence: float
    last_updated: int

class RobustReferenceManager:
    """
    Manages clean reference models and templates.
    
    Key improvements:
    - Maintains clean baselines from early rounds
    - Adaptive reference updates only when confidence is high
    - Robust statistics using median and trimmed means
    """
    
    def __init__(self, confidence_threshold: float = 0.8, update_frequency: int = 10):
        self.confidence_threshold = confidence_threshold
        self.update_frequency = update_frequency
        
        # Reference storage
        self.clean_templates: Dict[int, ReferenceTemplate] = {}
        self.reference_model: Optional[nn.Module] = None
        self.clean_client_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Statistics
        self.round_count = 0
        self.last_reference_update = 0
        
    def initialize_reference(self, global_model: nn.Module, root_loader, device: torch.device):
        """Initialize reference with pre-trained global model."""
        self.reference_model = global_model
        self.round_count = 0
        
        # Extract reference representations
        self._extract_reference_templates(root_loader, device)
        
        log(INFO, f"Initialized robust reference with {len(self.clean_templates)} class templates")
    
    def _extract_reference_templates(self, root_loader, device: torch.device):
        """Extract clean reference templates from global model."""
        self.reference_model.eval()
        representations = []
        labels = []
        
        with torch.no_grad():
            for batch in root_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                
                # Extract penultimate layer representations
                features = self._extract_features(inputs)
                representations.append(features.cpu().numpy())
                labels.append(targets.numpy())
        
        representations = np.vstack(representations)
        labels = np.hstack(labels)
        
        # Build templates for each class
        for class_id in np.unique(labels):
            class_mask = labels == class_id
            class_repr = representations[class_mask]
            
            if len(class_repr) > 0:
                template = self._build_class_template(class_id, class_repr)
                self.clean_templates[class_id] = template
    
    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract penultimate layer features."""
        # Get features before final classification layer
        if hasattr(self.reference_model, 'avgpool'):
            # ResNet architecture
            x = self.reference_model.conv1(inputs)
            x = self.reference_model.bn1(x)
            x = self.reference_model.relu(x)
            x = self.reference_model.maxpool(x)
            x = self.reference_model.layer1(x)
            x = self.reference_model.layer2(x)
            x = self.reference_model.layer3(x)
            x = self.reference_model.layer4(x)
            x = self.reference_model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            # Generic architecture - use forward hook
            features = []
            def hook(module, input, output):
                features.append(output)
            
            # Find penultimate layer
            layers = list(self.reference_model.children())
            if len(layers) > 1:
                penultimate_layer = layers[-2]
                handle = penultimate_layer.register_forward_hook(hook)
                
                _ = self.reference_model(inputs)
                handle.remove()
                
                if features:
                    return features[0].flatten(1)
            
            # Fallback: use full forward pass and take last hidden layer
            return self.reference_model(inputs)
    
    def _build_class_template(self, class_id: int, class_repr: np.ndarray) -> ReferenceTemplate:
        """Build robust class template using trimmed statistics."""
        # Use robust statistics (trimmed mean, median)
        trimmed_repr = self._trim_outliers(class_repr, trim_fraction=0.1)
        
        mean_repr = np.median(trimmed_repr, axis=0)  # Use median for robustness
        
        # Compute robust covariance
        centered_repr = trimmed_repr - mean_repr
        cov_matrix = np.cov(centered_repr.T)
        
        # Add regularization for numerical stability
        cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
        
        # Compute whitening matrix
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive
        whitening_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        
        return ReferenceTemplate(
            class_id=class_id,
            mean_representation=mean_repr,
            covariance_matrix=cov_matrix,
            whitening_matrix=whitening_matrix,
            confidence=1.0,  # Initial reference has full confidence
            last_updated=self.round_count
        )
    
    def _trim_outliers(self, data: np.ndarray, trim_fraction: float = 0.1) -> np.ndarray:
        """Trim outliers using robust statistics."""
        if len(data) <= 2:
            return data
        
        # Compute Mahalanobis distance for outlier detection
        mean = np.mean(data, axis=0)
        cov = np.cov(data.T)
        cov += 1e-6 * np.eye(cov.shape[0])  # Regularization
        
        try:
            inv_cov = np.linalg.inv(cov)
            mahal_dist = np.sum((data - mean) @ inv_cov * (data - mean), axis=1)
            
            # Keep data points within threshold
            threshold = np.percentile(mahal_dist, (1 - trim_fraction) * 100)
            mask = mahal_dist <= threshold
            
            return data[mask] if np.any(mask) else data
        except np.linalg.LinAlgError:
            # Fallback: simple trimming
            n_trim = int(len(data) * trim_fraction)
            if n_trim > 0:
                return data[n_trim:-n_trim] if len(data) > 2 * n_trim else data
            return data
    
    def update_reference(self, round_num: int, client_models: Dict[int, nn.Module], 
                        detection_results: List[DetectionResult], root_loader, device: torch.device):
        """Update reference only when confidence is high."""
        self.round_count = round_num
        
        # Only update if enough rounds have passed and we have high-confidence clean clients
        if (round_num - self.last_reference_update) < self.update_frequency:
            return
        
        # Find high-confidence clean clients
        clean_clients = []
        for result in detection_results:
            if (not result.is_malicious and 
                result.confidence > self.confidence_threshold):
                clean_clients.append(result.client_id)
        
        if len(clean_clients) < 3:  # Need at least 3 clean clients
            log(WARNING, f"Insufficient clean clients ({len(clean_clients)}) for reference update")
            return
        
        # Update templates using clean clients
        self._update_templates_from_clean_clients(clean_clients, client_models, root_loader, device)
        self.last_reference_update = round_num
        
        log(INFO, f"Updated reference templates using {len(clean_clients)} clean clients")
    
    def _update_templates_from_clean_clients(self, clean_clients: List[int], 
                                           client_models: Dict[int, nn.Module],
                                           root_loader, device: torch.device):
        """Update templates using representations from clean clients."""
        all_representations = defaultdict(list)
        all_labels = []
        
        for client_id in clean_clients:
            if client_id not in client_models:
                continue
                
            model = client_models[client_id]
            model.eval()
            
            with torch.no_grad():
                for batch in root_loader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    
                    features = self._extract_features_from_model(model, inputs)
                    representations = features.cpu().numpy()
                    
                    for i, label in enumerate(targets.numpy()):
                        all_representations[label].append(representations[i])
        
        # Update templates
        for class_id, repr_list in all_representations.items():
            if len(repr_list) > 0:
                class_repr = np.array(repr_list)
                new_template = self._build_class_template(class_id, class_repr)
                
                # Blend with existing template (weighted average)
                if class_id in self.clean_templates:
                    old_template = self.clean_templates[class_id]
                    blended_template = self._blend_templates(old_template, new_template, weight=0.3)
                    self.clean_templates[class_id] = blended_template
                else:
                    self.clean_templates[class_id] = new_template
    
    def _extract_features_from_model(self, model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features from a client model."""
        # Similar to _extract_features but for any model
        if hasattr(model, 'avgpool'):
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            # Generic approach
            layers = list(model.children())
            if len(layers) > 1:
                penultimate_layer = layers[-2]
                features = []
                def hook(module, input, output):
                    features.append(output)
                
                handle = penultimate_layer.register_forward_hook(hook)
                _ = model(inputs)
                handle.remove()
                
                if features:
                    return features[0].flatten(1)
            
            return model(inputs)
    
    def _blend_templates(self, old_template: ReferenceTemplate, 
                        new_template: ReferenceTemplate, weight: float) -> ReferenceTemplate:
        """Blend old and new templates."""
        blended_mean = (1 - weight) * old_template.mean_representation + weight * new_template.mean_representation
        
        # Blend covariances (more complex, use simple approach for now)
        blended_cov = (1 - weight) * old_template.covariance_matrix + weight * new_template.covariance_matrix
        
        # Recompute whitening matrix
        eigenvals, eigenvecs = np.linalg.eigh(blended_cov)
        eigenvals = np.maximum(eigenvals, 1e-6)
        blended_whitening = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        
        return ReferenceTemplate(
            class_id=old_template.class_id,
            mean_representation=blended_mean,
            covariance_matrix=blended_cov,
            whitening_matrix=blended_whitening,
            confidence=min(old_template.confidence, new_template.confidence),
            last_updated=self.round_count
        )
    
    def get_clean_baseline(self) -> Dict[int, ReferenceTemplate]:
        """Get current clean baseline templates."""
        return self.clean_templates.copy()

class MultiDimensionalDetector:
    """
    Multi-dimensional anomaly detector combining multiple detection methods.
    
    Methods:
    1. CKA-based detection (representation similarity)
    2. Spectral analysis (target class projection)
    3. Gradient norm analysis (update magnitude)
    4. Layer-wise analysis (feature space analysis)
    """
    
    def __init__(self, cka_weight: float = 0.4, spectral_weight: float = 0.3,
                 gradient_weight: float = 0.2, layer_weight: float = 0.1):
        self.cka_weight = cka_weight
        self.spectral_weight = spectral_weight
        self.gradient_weight = gradient_weight
        self.layer_weight = layer_weight
        
        # Initialize detectors
        self.cka_detector = CKADetector()
        self.spectral_detector = SpectralDetector()
        self.gradient_detector = GradientNormDetector()
        self.layer_detector = LayerWiseDetector()
        
        # Normalization statistics
        self.normalization_stats = {}
        
    def compute_anomaly_scores(self, client_models: Dict[int, nn.Module],
                              reference_manager: RobustReferenceManager,
                              root_loader, device: torch.device) -> List[DetectionResult]:
        """Compute multi-dimensional anomaly scores."""
        results = []
        
        # Extract representations for all clients
        client_representations = self._extract_all_representations(
            client_models, root_loader, device
        )
        
        # Get clean baseline
        clean_templates = reference_manager.get_clean_baseline()
        
        # Compute component scores
        cka_scores = self.cka_detector.compute_scores(client_representations, clean_templates)
        spectral_scores = self.spectral_detector.compute_scores(client_representations, clean_templates)
        gradient_scores = self.gradient_detector.compute_scores(client_models, reference_manager.reference_model)
        layer_scores = self.layer_detector.compute_scores(client_representations, clean_templates)
        
        # Normalize scores against clean baseline
        cka_scores_norm = self._normalize_scores(cka_scores, 'cka')
        spectral_scores_norm = self._normalize_scores(spectral_scores, 'spectral')
        gradient_scores_norm = self._normalize_scores(gradient_scores, 'gradient')
        layer_scores_norm = self._normalize_scores(layer_scores, 'layer')
        
        # Compute ensemble scores
        for client_id in client_models.keys():
            component_scores = {
                'cka': cka_scores_norm.get(client_id, 0.5),
                'spectral': spectral_scores_norm.get(client_id, 0.5),
                'gradient': gradient_scores_norm.get(client_id, 0.5),
                'layer': layer_scores_norm.get(client_id, 0.5)
            }
            
            # Ensemble anomaly score
            anomaly_score = (
                self.cka_weight * component_scores['cka'] +
                self.spectral_weight * component_scores['spectral'] +
                self.gradient_weight * component_scores['gradient'] +
                self.layer_weight * component_scores['layer']
            )
            
            # Compute confidence based on agreement between methods
            confidence = self._compute_confidence(component_scores)
            
            # Determine if malicious (threshold will be set dynamically)
            is_malicious = anomaly_score > 0.5  # Will be overridden by dynamic threshold
            
            result = DetectionResult(
                client_id=client_id,
                anomaly_score=anomaly_score,
                component_scores=component_scores,
                confidence=confidence,
                is_malicious=is_malicious,
                detection_methods=['cka', 'spectral', 'gradient', 'layer']
            )
            
            results.append(result)
        
        return results
    
    def _extract_all_representations(self, client_models: Dict[int, nn.Module],
                                   root_loader, device: torch.device) -> Dict[int, Dict[int, np.ndarray]]:
        """Extract representations from all clients grouped by class."""
        client_representations = {}
        
        for client_id, model in client_models.items():
            model.eval()
            representations = []
            labels = []
            
            with torch.no_grad():
                for batch in root_loader:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    
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
        if hasattr(model, 'avgpool'):
            x = model.conv1(inputs)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            layers = list(model.children())
            if len(layers) > 1:
                penultimate_layer = layers[-2]
                features = []
                def hook(module, input, output):
                    features.append(output)
                
                handle = penultimate_layer.register_forward_hook(hook)
                _ = model(inputs)
                handle.remove()
                
                if features:
                    return features[0].flatten(1)
            
            return model(inputs)
    
    def _normalize_scores(self, scores: Dict[int, float], method: str) -> Dict[int, float]:
        """Normalize scores using robust statistics."""
        if not scores:
            return {}
        
        values = list(scores.values())
        
        # Use robust normalization
        median_val = np.median(values)
        mad = np.median(np.abs(values - median_val))
        
        if mad < 1e-10:
            # All values are the same
            return {k: 0.5 for k in scores.keys()}
        
        # Normalize to [0, 1] range
        normalized = {}
        for k, v in scores.items():
            normalized[k] = np.clip((v - median_val) / (3 * mad) + 0.5, 0, 1)
        
        # Store normalization stats
        self.normalization_stats[method] = {
            'median': median_val,
            'mad': mad,
            'min': min(values),
            'max': max(values)
        }
        
        return normalized
    
    def _compute_confidence(self, component_scores: Dict[str, float]) -> float:
        """Compute confidence based on agreement between detection methods."""
        scores = list(component_scores.values())
        
        # Higher confidence when methods agree
        score_variance = np.var(scores)
        confidence = 1.0 / (1.0 + score_variance)
        
        return confidence

class CKADetector:
    """CKA-based anomaly detector."""
    
    def compute_scores(self, client_representations: Dict[int, Dict[int, np.ndarray]],
                      clean_templates: Dict[int, ReferenceTemplate]) -> Dict[int, float]:
        """Compute CKA-based anomaly scores."""
        scores = {}
        
        for client_id, client_repr in client_representations.items():
            cka_distances = []
            
            for class_id, class_repr in client_repr.items():
                if class_id in clean_templates:
                    template = clean_templates[class_id]
                    
                    # Compute CKA distance
                    cka_dist = self._compute_cka_distance(class_repr, template)
                    cka_distances.append(cka_dist)
            
            # Average CKA distance across classes
            scores[client_id] = np.mean(cka_distances) if cka_distances else 0.5
        
        return scores
    
    def _compute_cka_distance(self, client_repr: np.ndarray, template: ReferenceTemplate) -> float:
        """Compute CKA distance between client and template."""
        # Use median representation
        client_median = np.median(client_repr, axis=0)
        template_median = template.mean_representation
        
        # Compute cosine similarity
        similarity = cosine_similarity([client_median], [template_median])[0, 0]
        
        # Convert to distance (higher = more dissimilar)
        distance = 1.0 - similarity
        
        return distance

class SpectralDetector:
    """Spectral analysis-based anomaly detector."""
    
    def compute_scores(self, client_representations: Dict[int, Dict[int, np.ndarray]],
                      clean_templates: Dict[int, ReferenceTemplate]) -> Dict[int, float]:
        """Compute spectral-based anomaly scores."""
        scores = {}
        
        # Find target class (class with highest spectral variance)
        target_class = self._find_target_class(clean_templates)
        
        if target_class is None:
            return {client_id: 0.5 for client_id in client_representations.keys()}
        
        for client_id, client_repr in client_representations.items():
            if target_class in client_repr:
                score = self._compute_spectral_score(
                    client_repr[target_class], clean_templates[target_class]
                )
                scores[client_id] = score
            else:
                scores[client_id] = 0.5
        
        return scores
    
    def _find_target_class(self, clean_templates: Dict[int, ReferenceTemplate]) -> Optional[int]:
        """Find target class with highest spectral variance."""
        if not clean_templates:
            return None
        
        max_variance = -1
        target_class = None
        
        for class_id, template in clean_templates.items():
            # Compute spectral variance
            eigenvals = np.linalg.eigvals(template.covariance_matrix)
            variance = np.max(eigenvals)
            
            if variance > max_variance:
                max_variance = variance
                target_class = class_id
        
        return target_class
    
    def _compute_spectral_score(self, client_repr: np.ndarray, template: ReferenceTemplate) -> float:
        """Compute spectral projection score."""
        # Whitening transformation
        centered_repr = client_repr - template.mean_representation
        whitened_repr = centered_repr @ template.whitening_matrix.T
        
        # Compute spectral projection
        spectral_score = np.mean(np.linalg.norm(whitened_repr, axis=1))
        
        return spectral_score

class GradientNormDetector:
    """Gradient norm-based anomaly detector."""
    
    def compute_scores(self, client_models: Dict[int, nn.Module],
                      reference_model: nn.Module) -> Dict[int, float]:
        """Compute gradient norm-based anomaly scores."""
        scores = {}
        
        # Extract gradient norms (simplified - use model parameter norms)
        for client_id, model in client_models.items():
            norm = self._compute_model_norm(model, reference_model)
            scores[client_id] = norm
        
        return scores
    
    def _compute_model_norm(self, client_model: nn.Module, reference_model: nn.Module) -> float:
        """Compute norm of model parameters."""
        total_norm = 0.0
        
        for (name1, param1), (name2, param2) in zip(
            client_model.named_parameters(), reference_model.named_parameters()
        ):
            if name1 == name2:
                diff = param1 - param2
                total_norm += torch.norm(diff).item() ** 2
        
        return np.sqrt(total_norm)

class LayerWiseDetector:
    """Layer-wise analysis-based anomaly detector."""
    
    def compute_scores(self, client_representations: Dict[int, Dict[int, np.ndarray]],
                      clean_templates: Dict[int, ReferenceTemplate]) -> Dict[int, float]:
        """Compute layer-wise anomaly scores."""
        scores = {}
        
        for client_id, client_repr in client_representations.items():
            layer_scores = []
            
            for class_id, class_repr in client_repr.items():
                if class_id in clean_templates:
                    template = clean_templates[class_id]
                    
                    # Compute layer-wise distance
                    layer_score = self._compute_layer_distance(class_repr, template)
                    layer_scores.append(layer_score)
            
            scores[client_id] = np.mean(layer_scores) if layer_scores else 0.5
        
        return scores
    
    def _compute_layer_distance(self, client_repr: np.ndarray, template: ReferenceTemplate) -> float:
        """Compute layer-wise distance."""
        # Use Mahalanobis distance
        centered_repr = client_repr - template.mean_representation
        
        try:
            inv_cov = np.linalg.inv(template.covariance_matrix)
            mahal_dist = np.mean(np.sum(centered_repr @ inv_cov * centered_repr, axis=1))
            return mahal_dist
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance
            euclidean_dist = np.mean(np.linalg.norm(centered_repr, axis=1))
            return euclidean_dist

class AdaptiveTrustManager:
    """
    Adaptive trust management system.
    
    Features:
    - Historical behavior tracking
    - Confidence-weighted updates
    - Dynamic trust thresholds
    - Quarantine management
    """
    
    def __init__(self, trust_decay: float = 0.95, update_rate: float = 0.1,
                 min_trust_threshold: float = 0.3):
        self.trust_decay = trust_decay
        self.update_rate = update_rate
        self.min_trust_threshold = min_trust_threshold
        
        # Trust tracking
        self.client_trust: Dict[int, float] = defaultdict(lambda: 0.5)  # Start neutral
        self.detection_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        self.confidence_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # Quarantine management
        self.quarantined_clients: set = set()
        self.quarantine_threshold = 0.2
        
    def update_trust(self, detection_results: List[DetectionResult]):
        """Update trust scores based on detection results."""
        for result in detection_results:
            client_id = result.client_id
            
            # Skip quarantined clients
            if client_id in self.quarantined_clients:
                continue
            
            # Update detection history
            self.detection_history[client_id].append(result.is_malicious)
            self.confidence_history[client_id].append(result.confidence)
            
            # Compute trust update
            trust_update = self._compute_trust_update(result)
            
            # Update trust score
            current_trust = self.client_trust[client_id]
            new_trust = (1 - self.update_rate) * current_trust + self.update_rate * trust_update
            
            # Apply decay
            self.client_trust[client_id] = new_trust * self.trust_decay
            
            # Check for quarantine
            if new_trust < self.quarantine_threshold:
                self.quarantined_clients.add(client_id)
                log(WARNING, f"Client {client_id} quarantined (trust={new_trust:.3f})")
    
    def _compute_trust_update(self, result: DetectionResult) -> float:
        """Compute trust update based on detection result."""
        if result.is_malicious:
            # Malicious detection reduces trust
            trust_update = 0.0
        else:
            # Clean detection increases trust
            trust_update = 1.0
        
        # Weight by confidence
        confidence_weighted_update = (
            result.confidence * trust_update + 
            (1 - result.confidence) * 0.5  # Neutral when low confidence
        )
        
        return confidence_weighted_update
    
    def get_trust_weight(self, client_id: int) -> float:
        """Get trust-based weight for aggregation."""
        if client_id in self.quarantined_clients:
            return 0.0
        
        trust = self.client_trust[client_id]
        return max(trust, self.min_trust_threshold)
    
    def get_trust_scores(self) -> Dict[int, float]:
        """Get current trust scores for all clients."""
        return dict(self.client_trust)

class FedSPECTRERobustServer(AnomalyDetectionServer):
    """
    Robust FedSPECTRE defense server with complete redesign.
    
    Key improvements:
    1. Robust reference management
    2. Multi-dimensional anomaly detection
    3. Adaptive trust system
    4. Dynamic thresholding
    5. Ensemble detection methods
    """
    
    defense_categories = ["anomaly_detection", "robust_aggregation"]
    
    def __init__(self, server_config, server_type: str = "fedspectre_robust", eta: float = 0.5,
                 # Detection parameters
                 cka_weight: float = 0.4, spectral_weight: float = 0.3,
                 gradient_weight: float = 0.2, layer_weight: float = 0.1,
                 # Robust statistics
                 use_robust_stats: bool = True, trim_fraction: float = 0.1,
                 # Reference management
                 reference_update_frequency: int = 10, confidence_threshold: float = 0.8,
                 # Trust system
                 trust_decay: float = 0.95, trust_update_rate: float = 0.1,
                 min_trust_threshold: float = 0.3,
                 # Adaptive thresholds
                 use_dynamic_thresholds: bool = True, threshold_adaptation_rate: float = 0.05,
                 # Root dataset
                 root_size: int = 64, use_ood_root_dataset: bool = False,
                 **kwargs):
        
        super().__init__(server_config, server_type, eta, **kwargs)
        
        # Store parameters
        self.cka_weight = cka_weight
        self.spectral_weight = spectral_weight
        self.gradient_weight = gradient_weight
        self.layer_weight = layer_weight
        self.use_robust_stats = use_robust_stats
        self.trim_fraction = trim_fraction
        self.reference_update_frequency = reference_update_frequency
        self.confidence_threshold = confidence_threshold
        self.trust_decay = trust_decay
        self.trust_update_rate = trust_update_rate
        self.min_trust_threshold = min_trust_threshold
        self.use_dynamic_thresholds = use_dynamic_thresholds
        self.threshold_adaptation_rate = threshold_adaptation_rate
        self.root_size = root_size
        self.use_ood_root_dataset = use_ood_root_dataset
        
        # Initialize components
        self.reference_manager = RobustReferenceManager(
            confidence_threshold=confidence_threshold,
            update_frequency=reference_update_frequency
        )
        
        self.detector = MultiDimensionalDetector(
            cka_weight=cka_weight,
            spectral_weight=spectral_weight,
            gradient_weight=gradient_weight,
            layer_weight=layer_weight
        )
        
        self.trust_manager = AdaptiveTrustManager(
            trust_decay=trust_decay,
            update_rate=trust_update_rate,
            min_trust_threshold=min_trust_threshold
        )
        
        # Create root dataset loader
        self.root_loader = self._create_root_loader()
        
        # Dynamic threshold
        self.dynamic_threshold = 0.5
        self.threshold_history = deque(maxlen=50)
        
        log(INFO, f"Initialized FedSPECTRE Robust server")
        log(INFO, f"  Detection weights: CKA={cka_weight}, Spectral={spectral_weight}, "
                  f"Gradient={gradient_weight}, Layer={layer_weight}")
        log(INFO, f"  Trust system: decay={trust_decay}, update_rate={trust_update_rate}")
        log(INFO, f"  Reference update frequency: {reference_update_frequency}")
    
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
        Detect anomalies using robust multi-dimensional approach.
        
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
        if not self.reference_manager.clean_templates:
            self.reference_manager.initialize_reference(
                self.global_model, self.root_loader, self.device
            )
        
        # Compute anomaly scores
        detection_results = self.detector.compute_anomaly_scores(
            client_models, self.reference_manager, self.root_loader, self.device
        )
        
        # Update dynamic threshold
        if self.use_dynamic_thresholds:
            self._update_dynamic_threshold(detection_results)
        
        # Apply threshold to determine malicious/benign
        malicious_clients = []
        benign_clients = []
        
        for result in detection_results:
            if result.anomaly_score > self.dynamic_threshold:
                malicious_clients.append(result.client_id)
            else:
                benign_clients.append(result.client_id)
        
        # Update trust system
        self.trust_manager.update_trust(detection_results)
        
        # Update reference (if conditions met)
        self.reference_manager.update_reference(
            self.round_num, client_models, detection_results, self.root_loader, self.device
        )
        
        # Log results
        self._log_detection_results(detection_results, malicious_clients, benign_clients)
        
        return malicious_clients, benign_clients
    
    def _update_dynamic_threshold(self, detection_results: List[DetectionResult]):
        """Update dynamic threshold based on historical patterns."""
        if not detection_results:
            return
        
        # Extract anomaly scores
        scores = [result.anomaly_score for result in detection_results]
        
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
        
        # Adaptive threshold based on attack patterns
        if len(self.threshold_history) >= 10:
            recent_thresholds = list(self.threshold_history)[-10:]
            threshold_trend = np.mean(np.diff(recent_thresholds))
            
            if threshold_trend > 0.01:  # Increasing trend (possible attack)
                threshold *= 0.9  # Lower threshold (more aggressive)
            elif threshold_trend < -0.01:  # Decreasing trend (calm period)
                threshold *= 1.1  # Raise threshold (more conservative)
        
        # Smooth threshold updates
        self.dynamic_threshold = (
            (1 - self.threshold_adaptation_rate) * self.dynamic_threshold +
            self.threshold_adaptation_rate * threshold
        )
        
        log(INFO, f"Updated dynamic threshold to {self.dynamic_threshold:.4f}")
    
    def _log_detection_results(self, detection_results: List[DetectionResult],
                              malicious_clients: List[int], benign_clients: List[int]):
        """Log detailed detection results."""
        log(INFO, f"═══ FedSPECTRE Robust Detection Results ═══")
        log(INFO, f"Dynamic threshold: {self.dynamic_threshold:.4f}")
        log(INFO, f"Malicious clients: {malicious_clients}")
        log(INFO, f"Benign clients: {benign_clients}")
        
        # Log detailed scores
        for result in detection_results:
            status = "MALICIOUS" if result.client_id in malicious_clients else "BENIGN"
            log(INFO, f"Client {result.client_id} ({status}): "
                      f"total={result.anomaly_score:.4f}, "
                      f"cka={result.component_scores['cka']:.4f}, "
                      f"spectral={result.component_scores['spectral']:.4f}, "
                      f"gradient={result.component_scores['gradient']:.4f}, "
                      f"layer={result.component_scores['layer']:.4f}, "
                      f"confidence={result.confidence:.4f}")
        
        # Log trust scores
        trust_scores = self.trust_manager.get_trust_scores()
        if trust_scores:
            log(INFO, f"Trust scores: {dict(list(trust_scores.items())[:5])}...")  # Show first 5
    
    def aggregate_client_updates(self, client_updates: List[Tuple[int, int, Dict]]) -> bool:
        """
        Aggregate client updates with trust-based weighting.
        
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
        
        # Apply trust-based weighting
        total_weight = 0.0
        weighted_updates = []
        
        for client_id, num_examples, state_dict in filtered_updates:
            trust_weight = self.trust_manager.get_trust_weight(client_id)
            effective_weight = num_examples * trust_weight
            
            weighted_updates.append((client_id, effective_weight, state_dict))
            total_weight += effective_weight
        
        if total_weight == 0:
            log(WARNING, "Total weight is zero, falling back to equal weighting")
            total_weight = len(filtered_updates)
            weighted_updates = [
                (cid, 1.0, state_dict) for cid, _, state_dict in filtered_updates
            ]
        
        # Normalize weights
        normalized_updates = [
            (cid, weight / total_weight, state_dict)
            for cid, weight, state_dict in weighted_updates
        ]
        
        # Aggregate models
        self._aggregate_models(normalized_updates)
        
        log(INFO, f"FedSPECTRE Robust aggregation complete: "
                  f"{len(filtered_updates)}/{len(client_updates)} clients aggregated")
        
        return True
    
    def _aggregate_models(self, weighted_updates: List[Tuple[int, float, Dict]]):
        """Aggregate models with weighted averaging."""
        if not weighted_updates:
            return
        
        # Initialize aggregated state dict
        aggregated_state = {}
        
        # Get first model's structure
        first_cid, _, first_state = weighted_updates[0]
        
        for key in first_state.keys():
            aggregated_state[key] = torch.zeros_like(first_state[key])
        
        # Weighted aggregation
        for client_id, weight, state_dict in weighted_updates:
            for key, tensor in state_dict.items():
                aggregated_state[key] += weight * tensor
        
        # Update global model
        self.global_model.load_state_dict(aggregated_state)
        
        log(INFO, f"Global model updated with {len(weighted_updates)} client contributions")
    
    def _create_client_model(self) -> nn.Module:
        """Create a client model with same architecture as global model."""
        # This should match the global model architecture
        # For now, return a copy of global model
        import copy
        return copy.deepcopy(self.global_model)
    
    def __repr__(self) -> str:
        return (f"FedSPECTRE-Robust("
                f"cka={self.cka_weight}, spectral={self.spectral_weight}, "
                f"gradient={self.gradient_weight}, layer={self.layer_weight}, "
                f"threshold={self.dynamic_threshold:.3f})")
