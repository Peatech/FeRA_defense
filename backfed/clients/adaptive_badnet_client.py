"""
Adaptive BadNet Attack - Evasion-Aware Backdoor Client

This attack extends the classic BadNets backdoor attack with evasion techniques
designed to bypass norm-based and alignment-based defenses like FeRA.

DEFENSE AWARENESS:
- Knows: Defense monitors update norms (spectral/delta) and directional alignment (TDA/mutual similarity)
- Doesn't know: Exact thresholds, filtering logic, or multi-metric combinations

EVASION TECHNIQUES:
1. Norm Clipping: Limits update magnitude to appear within benign range
2. Direction Alignment: Blends update direction toward global model to increase TDA score
3. Gradual Ramping: Starts subtle, increases attack strength over rounds
4. Adaptive Scaling: Dynamically adjusts update scale based on historical norms

HOW IT WORKS:
The attack operates in two phases:
1. Standard backdoor training (inherited from MaliciousClient)
2. Post-processing to make the malicious update appear benign:
   - Computes update delta (client_params - global_params)
   - Clips norm to target percentile of expected benign norms
   - Blends direction with global model direction (increases alignment)
   - Applies gradual ramping factor (weaker early, stronger later)

CONFIGURATION PARAMETERS:
- norm_clip_percentile (float): Target percentile for norm clipping (default: 75)
  Higher values = more aggressive updates (easier to detect)
  Lower values = more conservative updates (harder to detect, weaker attack)
  
- alignment_weight (float): Weight for direction alignment [0-1] (default: 0.3)
  0.0 = no alignment (pure malicious direction)
  1.0 = full alignment (towards global model, weak attack)
  0.3 = 30% blend with global direction (balanced evasion)
  
- gradual_ramp_rounds (int): Rounds to reach full attack strength (default: 10)
  Attack strength = min(1.0, current_round / gradual_ramp_rounds)
  Early rounds use reduced strength to avoid detection
  
- target_norm_scale (float): Base norm scale for clipping (default: 0.5)
  Multiplied by estimated benign norm to get clip threshold
  Lower values = more conservative clipping

USAGE:
In your config file:
```yaml
attack:
  _target_: backfed.clients.adaptive_badnet_client.AdaptiveBadNetClient
  norm_clip_percentile: 75
  alignment_weight: 0.3
  gradual_ramp_rounds: 10
  target_norm_scale: 0.5
```

Or via command line:
```bash
python main.py attack=adaptive_badnet defense=fera_visualize \
    attack.norm_clip_percentile=75 attack.alignment_weight=0.3
```

LIMITATIONS:
- Assumes defense uses norm-based and alignment-based metrics
- Does not know exact detection thresholds
- Cannot evade defenses that use fundamentally different approaches (e.g., certified defenses)
- Trade-off: stronger evasion → weaker backdoor effectiveness

Author: AI Assistant
Date: 2025-11-09
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from logging import INFO, WARNING
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.utils import log
from backfed.const import Metrics


class AdaptiveBadNetClient(MaliciousClient):
    """
    Adaptive BadNet attack with norm clipping and direction alignment evasion.
    
    Extends MaliciousClient with post-training evasion techniques to bypass
    norm-based and alignment-based defenses.
    """
    
    def __init__(
        self,
        client_id,
        dataset,
        model,
        client_config,
        atk_config,
        poison_module,
        context_actor,
        client_type: str = "adaptive_badnet",
        verbose: bool = True,
        norm_clip_percentile: float = 75.0,
        alignment_weight: float = 0.3,
        gradual_ramp_rounds: int = 10,
        target_norm_scale: float = 0.5,
        **kwargs
    ):
        """
        Initialize Adaptive BadNet client.
        
        Args:
            client_id: Unique client identifier
            dataset: Client dataset
            model: Training model
            client_config: Client configuration
            atk_config: Attack configuration
            poison_module: Poison module (typically BadNets)
            context_actor: Context actor for synchronization
            client_type: Type identifier (default: "adaptive_badnet")
            verbose: Verbose logging flag
            norm_clip_percentile: Target percentile for norm clipping (0-100)
            alignment_weight: Weight for direction alignment (0-1)
            gradual_ramp_rounds: Rounds to reach full attack strength
            target_norm_scale: Base norm scale for clipping
            **kwargs: Additional arguments
        """
        super().__init__(
            client_id=client_id,
            dataset=dataset,
            model=model,
            client_config=client_config,
            atk_config=atk_config,
            poison_module=poison_module,
            context_actor=context_actor,
            client_type=client_type,
            verbose=verbose,
            **kwargs
        )
        
        # Evasion parameters
        self.norm_clip_percentile = norm_clip_percentile
        self.alignment_weight = alignment_weight
        self.gradual_ramp_rounds = gradual_ramp_rounds
        self.target_norm_scale = target_norm_scale
        
        # Historical norm tracking for adaptive clipping
        self.norm_history = []
        
        log(INFO, f"Client [{self.client_id}] Initialized Adaptive BadNet Attack")
        log(INFO, f"  Norm clip percentile: {self.norm_clip_percentile}")
        log(INFO, f"  Alignment weight: {self.alignment_weight}")
        log(INFO, f"  Gradual ramp rounds: {self.gradual_ramp_rounds}")
        log(INFO, f"  Target norm scale: {self.target_norm_scale}")
    
    def train(self, train_package: Dict[str, Any]) -> Tuple[int, Dict[str, torch.Tensor], Metrics]:
        """
        Train with backdoor and apply evasion techniques.
        
        Args:
            train_package: Training package with global model params and metadata
            
        Returns:
            Tuple of (num_examples, evaded_state_dict, training_metrics)
        """
        # Standard backdoor training
        num_examples, state_dict, metrics = super().train(train_package)
        
        # Apply adaptive evasion post-processing
        server_round = train_package.get("server_round", 0)
        evaded_state_dict = self._apply_evasion(
            state_dict=state_dict,
            global_params=train_package["global_model_params"],
            server_round=server_round
        )
        
        return num_examples, evaded_state_dict, metrics
    
    def _apply_evasion(
        self,
        state_dict: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
        server_round: int
    ) -> Dict[str, torch.Tensor]:
        """
        Apply evasion techniques to make malicious update appear benign.
        
        Process:
        1. Compute update delta (malicious_params - global_params)
        2. Apply gradual ramping (reduce strength in early rounds)
        3. Clip norm to appear within benign range
        4. Align direction with global model
        5. Reconstruct final parameters
        
        Args:
            state_dict: Malicious client parameters
            global_params: Global model parameters
            server_round: Current training round
            
        Returns:
            Evaded state dictionary
        """
        # 1. Compute update delta
        delta = self._compute_delta(state_dict, global_params)
        
        # 2. Gradual ramping: reduce attack strength in early rounds
        ramp_factor = min(1.0, server_round / max(1, self.gradual_ramp_rounds))
        delta_ramped = delta * ramp_factor
        
        # 3. Norm clipping: limit update magnitude
        delta_clipped = self._clip_norm(delta_ramped, server_round)
        
        # 4. Direction alignment: blend with global model direction
        delta_aligned = self._align_direction(
            delta_clipped,
            global_params,
            weight=self.alignment_weight
        )
        
        # 5. Reconstruct final parameters
        evaded_state_dict = self._apply_delta(global_params, delta_aligned)
        
        # Log evasion statistics
        original_norm = torch.linalg.norm(delta).item()
        final_norm = torch.linalg.norm(delta_aligned).item()
        
        if self.verbose:
            log(INFO, f"Client [{self.client_id}] Round {server_round} Evasion:")
            log(INFO, f"  Ramp factor: {ramp_factor:.3f}")
            log(INFO, f"  Original norm: {original_norm:.6f}")
            log(INFO, f"  Final norm: {final_norm:.6f} (reduction: {(1 - final_norm/max(original_norm, 1e-9))*100:.1f}%)")
        
        return evaded_state_dict
    
    def _compute_delta(
        self,
        state_dict: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute flattened update delta: client_params - global_params
        
        Args:
            state_dict: Client parameters
            global_params: Global model parameters
            
        Returns:
            Flattened delta tensor
        """
        delta_list = []
        for key in sorted(state_dict.keys()):
            if key in global_params:
                client_param = state_dict[key].to(self.device)
                global_param = global_params[key].to(self.device)
                delta_list.append((client_param - global_param).flatten())
        
        if not delta_list:
            return torch.tensor([], device=self.device)
        
        return torch.cat(delta_list)
    
    def _clip_norm(
        self,
        delta: torch.Tensor,
        server_round: int
    ) -> torch.Tensor:
        """
        Clip update norm to appear within benign range.
        
        Estimates target norm based on historical data or conservative defaults.
        Clips delta to not exceed (target_norm_scale * estimated_benign_norm).
        
        Args:
            delta: Update delta tensor
            server_round: Current round (for adaptive estimation)
            
        Returns:
            Norm-clipped delta tensor
        """
        current_norm = torch.linalg.norm(delta).item()
        
        # Estimate target norm from historical data
        if len(self.norm_history) > 0:
            # Use historical percentile as target
            target_norm = np.percentile(self.norm_history, self.norm_clip_percentile)
        else:
            # Conservative default: scale down current norm
            target_norm = current_norm * self.target_norm_scale
        
        # Apply target_norm_scale
        target_norm = target_norm * self.target_norm_scale
        
        # Clip if exceeds target
        if current_norm > target_norm and target_norm > 1e-9:
            scale_factor = target_norm / current_norm
            delta_clipped = delta * scale_factor
        else:
            delta_clipped = delta
        
        # Track norm for future rounds (even if clipped)
        # This helps adapt to actual benign norms over time
        self.norm_history.append(current_norm)
        
        # Keep history bounded
        if len(self.norm_history) > 50:
            self.norm_history.pop(0)
        
        return delta_clipped
    
    def _align_direction(
        self,
        delta: torch.Tensor,
        global_params: Dict[str, torch.Tensor],
        weight: float
    ) -> torch.Tensor:
        """
        Align update direction with global model direction.
        
        Blends the malicious update direction with the global model's parameter
        direction to increase TDA (Temporal Direction Alignment) score.
        
        Formula:
        aligned_delta = (1 - weight) * delta + weight * global_direction
        
        Where global_direction = global_params (flattened and normalized)
        
        Args:
            delta: Update delta tensor
            global_params: Global model parameters
            weight: Alignment weight [0-1]
            
        Returns:
            Direction-aligned delta tensor
        """
        if weight <= 0.0 or weight >= 1.0:
            # No alignment needed
            if weight >= 1.0:
                log(WARNING, f"Client [{self.client_id}] alignment_weight=1.0 means no attack!")
            return delta
        
        # Flatten global parameters as reference direction
        global_flat = []
        for key in sorted(global_params.keys()):
            global_flat.append(global_params[key].to(self.device).flatten())
        
        if not global_flat:
            return delta
        
        global_direction = torch.cat(global_flat)
        
        # Normalize global direction
        global_norm = torch.linalg.norm(global_direction).clamp(min=1e-9)
        global_direction_normalized = global_direction / global_norm
        
        # Get delta norm for rescaling
        delta_norm = torch.linalg.norm(delta).clamp(min=1e-9)
        
        # Normalize delta direction
        delta_direction = delta / delta_norm
        
        # Blend directions
        blended_direction = (1 - weight) * delta_direction + weight * global_direction_normalized
        
        # Normalize blended direction and restore magnitude
        blended_norm = torch.linalg.norm(blended_direction).clamp(min=1e-9)
        aligned_delta = (blended_direction / blended_norm) * delta_norm
        
        return aligned_delta
    
    def _apply_delta(
        self,
        global_params: Dict[str, torch.Tensor],
        delta: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Reconstruct state dict from global params and delta.
        
        Args:
            global_params: Global model parameters
            delta: Flattened update delta
            
        Returns:
            Reconstructed state dictionary
        """
        state_dict = {}
        offset = 0
        
        for key in sorted(global_params.keys()):
            global_param = global_params[key].to(self.device)
            param_numel = global_param.numel()
            
            # Extract corresponding delta slice
            if offset + param_numel <= delta.numel():
                delta_slice = delta[offset:offset + param_numel]
                delta_reshaped = delta_slice.reshape(global_param.shape)
                
                # Reconstruct parameter (keep on same device as global_param)
                state_dict[key] = global_param + delta_reshaped
            else:
                # Safety: if delta is shorter, use global param
                state_dict[key] = global_param
            
            offset += param_numel
        
        return state_dict

