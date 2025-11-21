"""
Utility functions for FedSPECTRE defense suite.

This module provides shared components used across FedAvgCKA, FedSPECTRE-Hybrid,
and FedSPECTRE-Stateful defenses, adapted to work with BackFed's architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from logging import INFO
from sklearn.covariance import OAS
from backfed.utils import log


def linear_cka(X: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute linear CKA similarity between two activation matrices.
    
    Based on FedAvgCKA Algorithm 1 and Equation 4.
    
    Args:
        X: Activation matrix [k, d1] 
        Y: Activation matrix [k, d2] 
        eps: Small constant for numerical stability
        
    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], f"Batch size mismatch: {X.shape[0]} vs {Y.shape[0]}"
    
    n = X.shape[0]
    if n <= 1:
        log(INFO, f"CKA computation with n={n} samples may be unreliable")
        return 0.0
    
    # Center the matrices (H = I - (1/n)*11^T)
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Apply centering: X_centered = HX, Y_centered = HY  
    X_centered = H @ X
    Y_centered = H @ Y
    
    # Compute HSIC values using linear kernel
    hsic_xy = np.trace(X_centered @ Y_centered.T) / ((n - 1) ** 2)
    hsic_xx = np.trace(X_centered @ X_centered.T) / ((n - 1) ** 2)  
    hsic_yy = np.trace(Y_centered @ Y_centered.T) / ((n - 1) ** 2)
    
    # Compute normalized CKA score
    denominator = np.sqrt(hsic_xx * hsic_yy)
    if denominator < eps:
        log(INFO, f"Small CKA denominator: {denominator}. Returning 0.0")
        return 0.0
        
    cka_score = hsic_xy / denominator
    cka_score = np.clip(cka_score, 0.0, 1.0)
    
    return float(cka_score)


def get_penultimate_layer_name(model: nn.Module) -> str:
    """
    Automatically determine the penultimate layer name for different architectures.
    
    Args:
        model: Neural network model
        
    Returns:
        Name of the penultimate (last feature) layer
    """
    # Check for ResNet architecture
    if hasattr(model, 'fc') and hasattr(model, 'avgpool'):
        return 'avgpool'
    
    # Check for SimpleNet architecture  
    if hasattr(model, 'fc2') and hasattr(model, 'fc1'):
        return 'fc1'
    
    # VGG wrapper specific handling
    if hasattr(model, 'features_module'):
        return 'features_module'
        
    # Check for VGG-like architectures
    if hasattr(model, 'classifier') and hasattr(model, 'features'):
        return 'features'
    
    # Check for VGG model structure (vgg_model.features)
    if hasattr(model, 'vgg_model') and hasattr(model.vgg_model, 'features'):
        return 'vgg_model.features'
    
    # Generic fallback
    layers = list(model.named_modules())
    if len(layers) < 2:
        raise ValueError("Model too simple to determine penultimate layer")
        
    penultimate_name = layers[-2][0]
    log(INFO, f"Auto-detected penultimate layer: '{penultimate_name}'")
    return penultimate_name


def extract_layer_activations(
    model: nn.Module, 
    data_loader: DataLoader,
    layer_name: str,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations from a specific layer of the model.
    
    Args:
        model: Neural network model (will be set to eval mode)
        data_loader: DataLoader containing input data
        layer_name: Name of layer to extract activations from
        device: Device to run computations on
        
    Returns:
        Tuple of (activation_matrix, labels)
        - activation_matrix: [n_samples, activation_dim]
        - labels: [n_samples]
    """
    model.eval()
    model.to(device)
    
    # Freeze BN layers
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()
    
    activations = []
    labels = []
    
    def hook_fn(module, input, output):
        # Flatten spatial dimensions if needed
        if output.dim() > 2:
            output = output.view(output.size(0), -1)
        activations.append(output.detach().cpu())
    
    # Find and register hook
    target_layer = None
    
    # Special handling for VGGWrapper features_module
    if layer_name == 'features_module' and hasattr(model, 'features_module'):
        target_layer = model.features_module
    else:
        # Try to find as named module
        for name, module in model.named_modules():
            if name == layer_name:
                target_layer = module
                break
    
    if target_layer is None:
        available_layers = [name for name, _ in model.named_modules()]
        raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers[:10]}...")
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    inputs = batch_data[0].to(device)
                    batch_labels = batch_data[1]
                    if isinstance(batch_labels, torch.Tensor):
                        batch_labels = batch_labels.cpu().numpy()
                    labels.append(batch_labels)
                else:
                    inputs = batch_data.to(device)
                    # Create dummy labels if not available
                    labels.append(np.zeros(inputs.size(0), dtype=int))
                
                _ = model(inputs)
        
        if not activations:
            raise RuntimeError("No activations captured. Check layer name and data loader.")
            
        activation_matrix = torch.cat(activations, dim=0).numpy()
        labels_array = np.concatenate(labels, axis=0)
        
        # Row-center the activation matrix
        activation_matrix = activation_matrix - activation_matrix.mean(axis=1, keepdims=True)
        
        return activation_matrix, labels_array
        
    finally:
        handle.remove()


def create_root_dataset_loader(
    testset, 
    root_size: int,
    batch_size: int = None,
    num_workers: int = 0,
    device: torch.device = None,
    ood_dataset = None
) -> DataLoader:
    """
    Create root dataset loader from test set or OOD dataset.
    
    Args:
        testset: Test dataset to sample from
        root_size: Number of samples for root dataset
        batch_size: Batch size (defaults to root_size)
        num_workers: Number of worker processes
        device: Device for data loading
        ood_dataset: Optional OOD dataset to use instead of testset
        
    Returns:
        DataLoader for root dataset
    """
    if batch_size is None:
        batch_size = root_size
    
    # Use OOD dataset if provided, otherwise use testset
    source_dataset = ood_dataset if ood_dataset is not None else testset
    
    total_samples = len(source_dataset)
    
    if root_size > total_samples:
        log(INFO, f"Requested root_size {root_size} > available samples {total_samples}. Using all samples.")
        root_size = total_samples
    
    # Random sampling
    indices = torch.randperm(total_samples)[:root_size].tolist()
    root_subset = Subset(source_dataset, indices)
    
    root_loader = DataLoader(
        root_subset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device and device.type == 'cuda' else False
    )
    
    log(INFO, f"Created root dataset with {len(indices)} samples from {'OOD dataset' if ood_dataset else 'test set'}")
    return root_loader


class RobustStatistics:
    """
    Robust statistics computation with median location and OAS covariance.
    
    - Median has 50% breakdown for location
    - OAS provides small-sample stability
    - Proper whitening via eigendecomposition
    """
    
    def __init__(self, rank: int = 128, trim_fraction: float = 0.05):
        """
        Initialize robust statistics estimator.
        
        Args:
            rank: Rank for low-rank projection (default 128)
            trim_fraction: Fraction to trim before covariance (0.05 = top 5%)
        """
        self.rank = rank
        self.trim_fraction = trim_fraction
        self.oas_estimator = OAS(store_precision=True)
        
    def compute_robust_stats(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute robust location and covariance with whitening matrix.
        
        Args:
            X: Data matrix [n_samples, n_features]
            
        Returns:
            mu: Robust location estimate [n_features]
            W: Whitening matrix [rank, n_features] 
            U: Projection matrix [n_features, rank]
        """
        n_samples, n_features = X.shape
        
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")
            
        # 1. Robust location: coordinate-wise median (50% breakdown)
        mu = np.median(X, axis=0)
        
        # 2. Optional trimming before covariance
        if self.trim_fraction > 0 and n_samples > 10:
            # Compute diagonal Mahalanobis distance to median
            X_centered = X - mu
            diag_cov = np.var(X_centered, axis=0)
            diag_cov = np.maximum(diag_cov, 1e-12)  # Avoid division by zero
            
            mahal_dist = np.sum(X_centered**2 / diag_cov, axis=1)
            threshold = np.percentile(mahal_dist, (1 - self.trim_fraction) * 100)
            
            keep_mask = mahal_dist <= threshold
            X_trimmed = X[keep_mask]
            log(INFO, f"Trimmed {np.sum(~keep_mask)}/{n_samples} outliers")
        else:
            X_trimmed = X
            
        # 3. OAS covariance (small-sample stability)
        X_centered = X_trimmed - mu
        cov_matrix = self.oas_estimator.fit(X_centered).covariance_
        
        # 4. Low-rank projection and whitening
        U, W = self._compute_whitening(cov_matrix)
        
        return mu, W, U
    
    def _compute_whitening(self, cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute whitening matrix via eigendecomposition.
        
        Args:
            cov_matrix: Covariance matrix [n_features, n_features]
            
        Returns:
            U: Projection matrix [n_features, rank]
            W: Whitening matrix [rank, n_features]
        """
        # Eigendecomposition: C = U @ S @ U.T
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (ascending)
        idx = np.argsort(eigenvals)
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Select top components (largest eigenvalues)
        rank = min(self.rank, len(eigenvals))
        eigenvals = eigenvals[-rank:]
        eigenvecs = eigenvecs[:, -rank:]
        
        # Avoid numerical issues with small eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        # Whitening matrix: W = S^(-1/2) @ U.T
        W = np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        
        return eigenvecs, W


class MahalanobisCKA:
    """
    Mahalanobis-CKA implementation for FedSPECTRE.
    
    Computes CKA between whitened client representations and class templates.
    """
    
    def __init__(self, eps: float = 1e-12):
        self.eps = eps
        
    def compute_cka_score(self, client_rep: np.ndarray, template_whitened: np.ndarray, 
                         mu: np.ndarray, W: np.ndarray) -> float:
        """
        Compute Mahalanobis-CKA score between client representation and pre-whitened template.
        
        Paper: Templates are pre-whitened (T_c = W_c · μ_c), so we only whiten client rep.
        
        Args:
            client_rep: Client representation in original space [k, d]
            template_whitened: Pre-whitened class template in whitened space [k, d_white]
            mu: Robust mean for whitening client representation
            W: Whitening matrix [d_white, d]
            
        Returns:
            CKA distance (higher = more anomalous, 1 - similarity)
        """
        # Whiten client representation to match template space
        client_white = (client_rep - mu) @ W.T
        
        # Template is already in whitened space, use directly
        # Compute CKA similarity between whitened representations
        cka = linear_cka(client_white, template_whitened, self.eps)
        
        # Return 1 - cka so that higher score = more anomalous
        return 1.0 - cka


class SpectralProjection:
    """
    Spectral projection analysis in whitened space.
    
    Computes projection of client activations onto top principal component.
    """
    
    def compute_spectral_score(self, client_rep: np.ndarray, 
                               mu: np.ndarray, W: np.ndarray,
                               top_pc: np.ndarray) -> float:
        """
        Compute spectral projection score for client.
        
        Args:
            client_rep: Client representation [k, d]
            mu: Robust mean for whitening
            W: Whitening matrix
            top_pc: Top principal component in whitened space
            
        Returns:
            Spectral projection score (normalized by sqrt(k))
        """
        # Whiten representation
        rep_white = (client_rep - mu) @ W.T
        
        # Project onto top PC
        k = rep_white.shape[0]
        score = np.linalg.norm(rep_white @ top_pc) / np.sqrt(k)
        
        return float(score)


def load_ood_dataset(base_dataset: str, config):
    """
    Load out-of-distribution dataset based on the base dataset.
    
    Mappings:
    - CIFAR10 -> CIFAR100
    - EMNIST -> MNIST  
    - FEMNIST -> EMNIST
    
    Args:
        base_dataset: Name of the base dataset (e.g., 'CIFAR10')
        config: Server configuration
        
    Returns:
        OOD dataset or None if not available
    """
    from torchvision import datasets, transforms
    
    base_dataset_upper = base_dataset.upper()
    
    try:
        if base_dataset_upper == 'CIFAR10':
            log(INFO, "Loading CIFAR100 as OOD dataset for CIFAR10")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
            ood_dataset = datasets.CIFAR100(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
            return ood_dataset
            
        elif base_dataset_upper == 'EMNIST':
            log(INFO, "Loading MNIST as OOD dataset for EMNIST")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            ood_dataset = datasets.MNIST(
                root='./data', 
                train=False, 
                download=True, 
                transform=transform
            )
            return ood_dataset
            
        elif base_dataset_upper == 'FEMNIST':
            log(INFO, "Loading EMNIST as OOD dataset for FEMNIST")
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1722,), (0.3309,))
            ])
            ood_dataset = datasets.EMNIST(
                root='./data', 
                split='byclass',
                train=False, 
                download=True, 
                transform=transform
            )
            return ood_dataset
            
        else:
            log(INFO, f"No OOD dataset mapping defined for {base_dataset}")
            return None
            
    except Exception as e:
        log(INFO, f"Failed to load OOD dataset: {e}")
        return None

