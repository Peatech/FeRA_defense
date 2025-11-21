"""
FedSPECTRE-Stateful Server Implementation for BackFed.

Extends FedSPECTRE-Hybrid with a stateful trust system that:
1. Maintains per-client trust scores via exponential moving averages
2. Penalizes selected clients' weights based on trust
3. Assigns leak weights to excluded clients
4. Quarantines persistently malicious clients

Based on the FedSPECTRE-Stateful defense algorithm.
"""

import copy
import math
import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from logging import INFO
from collections import defaultdict
from backfed.servers.fedspectre_hybrid_server import FedSPECTREHybridServer
from backfed.utils import log
from backfed.const import StateDict, client_id, num_examples


class StatefulTrustManager:
    """
    Manages persistent trust state for all clients with EMA tracking and quarantine logic.
    
    Implements:
    - Exclusion EMA: e_i^t = μ * e_i^{t-1} + (1-μ) * 1[i∈E_t]
    - Anomaly EMA: a_i^t = μ * a_i^{t-1} + (1-μ) * z_i^t  
    - Risk EMA: r_i^t = λ * r_i^{t-1} + (1-λ) * (α * e_i^t + (1-α) * a_i^t)
    - Trust: τ_i^t = 1 / (1 + exp(k * (r_i^t - m)))
    """
    
    def __init__(self, config):
        """Initialize trust manager with configuration parameters."""
        # Trust system hyperparameters
        self.mu = getattr(config, 'trust_mu', 0.95)
        self.lmbda = getattr(config, 'trust_lmbda', 0.85)
        self.alpha = getattr(config, 'trust_alpha', 0.85)
        self.k = getattr(config, 'trust_k', 16)
        self.m = getattr(config, 'trust_m', 0.52)
        self.theta = getattr(config, 'trust_theta', 0.65)
        self.gamma = getattr(config, 'trust_gamma', 0.5)
        self.q = getattr(config, 'trust_q', 1)
        self.p = getattr(config, 'trust_p', 3)
        self.eta = getattr(config, 'trust_eta', 0.10)
        self.delta = getattr(config, 'trust_delta', 0.02)
        self.leak_cap = getattr(config, 'trust_leak_cap', 0.10)
        
        # Conservative leak parameters
        self.eta_conservative = getattr(config, 'trust_eta_conservative', 0.02)
        self.delta_conservative = getattr(config, 'trust_delta_conservative', 0.01)
        self.leak_cap_conservative = getattr(config, 'trust_leak_cap_conservative', 0.03)
        
        # Quarantine parameters
        self.quarantine_exclusion_threshold = getattr(config, 'trust_quarantine_exclusion_threshold', 0.95)
        self.quarantine_risk_threshold = getattr(config, 'trust_quarantine_risk_threshold', 0.9)
        self.quarantine_exclusion_rounds = getattr(config, 'trust_quarantine_exclusion_rounds', 10)
        self.quarantine_risk_rounds = getattr(config, 'trust_quarantine_risk_rounds', 5)
        self.exit_exclusion_threshold = getattr(config, 'trust_exit_exclusion_threshold', 0.6)
        self.exit_risk_threshold = getattr(config, 'trust_exit_risk_threshold', 0.5)
        self.exit_rounds = getattr(config, 'trust_exit_rounds', 20)
        self.tau_drop = getattr(config, 'trust_tau_drop', 0.4)
        
        # Persistent state
        self.excl_ema = defaultdict(float)
        self.anom_ema = defaultdict(float)
        self.risk = defaultdict(lambda: 0.55)
        self.trust = defaultdict(lambda: 0.55)
        self.streak_low_trust = defaultdict(int)
        self.quarantined = set()
        
        # History tracking
        self.exclusion_history = defaultdict(list)
        self.risk_history = defaultdict(list)
        self.exit_counter = defaultdict(int)
        
        log(INFO, "=== FedSPECTRE Stateful Trust System Initialized ===")
        log(INFO, f"EMA params: μ={self.mu}, λ={self.lmbda}, α={self.alpha}")
        log(INFO, f"Trust mapping: k={self.k}, m={self.m}, θ={self.theta}, γ={self.gamma}, p={self.p}")
        log(INFO, f"Leak params: η={self.eta}, δ={self.delta}, cap={self.leak_cap}")
    
    def update_state(self, sampled_ids: List[int], excluded_ids: List[int], 
                    total_scores: Dict[int, float], anomaly_scores: Dict[int, Dict[str, float]] = None):
        """
        Update trust state for all sampled clients (Paper Step 3.1).
        
        Uses pre-normalized scores from anomaly detection (Step 2.3).
        """
        if not sampled_ids:
            log(INFO, "Empty sampled_ids, skipping trust update")
            return
        
        # Update each client
        for cid in sampled_ids:
            # Use normalized total anomaly score (already properly computed from normalized components)
            if anomaly_scores and cid in anomaly_scores and 'total' in anomaly_scores[cid]:
                z_score = anomaly_scores[cid]['total']
            else:
                # Fallback if score wasn't computed (shouldn't happen with correct flow)
                log(INFO, f"Warning: No anomaly score for client {cid}, using 0.5")
                z_score = 0.5
            
            # Exclusion indicator
            was_excluded = 1.0 if cid in excluded_ids else 0.0
            
            # Update EMAs
            self.excl_ema[cid] = self.mu * self.excl_ema[cid] + (1 - self.mu) * was_excluded
            self.anom_ema[cid] = self.mu * self.anom_ema[cid] + (1 - self.mu) * z_score
            
            # Update risk
            signal = self.alpha * self.excl_ema[cid] + (1 - self.alpha) * self.anom_ema[cid]
            self.risk[cid] = self.lmbda * self.risk[cid] + (1 - self.lmbda) * signal
            
            # Update trust
            self.trust[cid] = 1.0 / (1.0 + math.exp(self.k * (self.risk[cid] - self.m)))
            
            # Update histories
            self.exclusion_history[cid].append(was_excluded)
            self.risk_history[cid].append(self.risk[cid])
            
            # Keep only recent history
            if len(self.exclusion_history[cid]) > 20:
                self.exclusion_history[cid] = self.exclusion_history[cid][-20:]
            if len(self.risk_history[cid]) > 20:
                self.risk_history[cid] = self.risk_history[cid][-20:]
        
        # Update quarantine status
        self._update_quarantine_status(sampled_ids)
    
    def _update_quarantine_status(self, sampled_ids: List[int]):
        """Update quarantine status based on exclusion and risk histories."""
        for cid in sampled_ids:
            should_quarantine = False
            
            # Check exclusion-based quarantine
            if len(self.exclusion_history[cid]) >= self.quarantine_exclusion_rounds:
                recent_exclusions = self.exclusion_history[cid][-self.quarantine_exclusion_rounds:]
                avg_exclusion = np.mean(recent_exclusions)
                if avg_exclusion >= self.quarantine_exclusion_threshold:
                    should_quarantine = True
                    log(INFO, f"Client {cid} quarantined (exclusion rate: {avg_exclusion:.3f})")
            
            # Check risk-based quarantine
            if len(self.risk_history[cid]) >= self.quarantine_risk_rounds:
                recent_risk = self.risk_history[cid][-self.quarantine_risk_rounds:]
                avg_risk = np.mean(recent_risk)
                if avg_risk >= self.quarantine_risk_threshold:
                    should_quarantine = True
                    log(INFO, f"Client {cid} quarantined (risk: {avg_risk:.3f})")
            
            if should_quarantine and cid not in self.quarantined:
                self.quarantined.add(cid)
                self.exit_counter[cid] = 0
                log(INFO, f"⚠️  Client {cid} ENTERED quarantine")
            
            # Check for exit
            if cid in self.quarantined:
                recent_exclusions = self.exclusion_history[cid][-self.exit_rounds:] if len(self.exclusion_history[cid]) >= self.exit_rounds else self.exclusion_history[cid]
                recent_risk = self.risk_history[cid][-self.exit_rounds:] if len(self.risk_history[cid]) >= self.exit_rounds else self.risk_history[cid]
                
                if (len(recent_exclusions) >= self.exit_rounds and 
                    len(recent_risk) >= self.exit_rounds and
                    np.mean(recent_exclusions) <= self.exit_exclusion_threshold and
                    np.mean(recent_risk) <= self.exit_risk_threshold):
                    
                    self.exit_counter[cid] += 1
                    if self.exit_counter[cid] >= self.exit_rounds:
                        self.quarantined.remove(cid)
                        self.exit_counter[cid] = 0
                        log(INFO, f"✓ Client {cid} EXITED quarantine")
                else:
                    self.exit_counter[cid] = 0
    
    def get_trust_scores(self, client_ids: List[int]) -> Dict[int, float]:
        """Get trust scores for specified clients."""
        return {cid: (0.0 if cid in self.quarantined else self.trust[cid]) for cid in client_ids}
    
    def get_risk_scores(self, client_ids: List[int]) -> Dict[int, float]:
        """Get risk scores for specified clients."""
        return {cid: self.risk[cid] for cid in client_ids}


class FedSPECTREStatefulServer(FedSPECTREHybridServer):
    """
    FedSPECTRE-Stateful defense server.
    
    Extends FedSPECTRE-Hybrid with stateful trust system and trust-based aggregation.
    """
    defense_categories = ["anomaly_detection", "robust_aggregation"]
    
    def __init__(self, server_config, server_type: str = "fedspectre_stateful", eta: float = 0.5, **kwargs):
        """Initialize FedSPECTRE-Stateful server."""
        # Initialize parent (FedSPECTRE-Hybrid)
        super().__init__(server_config, server_type, eta, **kwargs)
        
        # Initialize trust system
        self.trust_manager = StatefulTrustManager(server_config)
        
        # Extract trust parameters for aggregation
        self.trust_p = getattr(server_config, 'trust_p', 3)
        self.trust_theta = getattr(server_config, 'trust_theta', 0.65)
        self.trust_gamma = getattr(server_config, 'trust_gamma', 0.5)
        self.trust_q = getattr(server_config, 'trust_q', 1)
        self.trust_eta = getattr(server_config, 'trust_eta', 0.10)
        self.trust_delta = getattr(server_config, 'trust_delta', 0.02)
        self.trust_eta_conservative = getattr(server_config, 'trust_eta_conservative', 0.02)
        self.trust_delta_conservative = getattr(server_config, 'trust_delta_conservative', 0.01)
        self.trust_tau_drop = getattr(server_config, 'trust_tau_drop', 0.4)
        
        log(INFO, "FedSPECTRE-Stateful initialized with trust system enabled")
    
    def aggregate_client_updates(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> bool:
        """
        Override aggregation to apply trust-based weighting.
        
        Args:
            client_updates: List of (client_id, num_examples, model_state_dict)
            
        Returns:
            True if aggregation succeeded
        """
        if not client_updates:
            log(INFO, "No client updates to aggregate")
            return False
        
        # Run anomaly detection (updates trust state)
        malicious_ids, benign_ids = self.detect_anomalies(client_updates)
        
        # Evaluate detection performance
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        detection_metrics = self.evaluate_detection(malicious_ids, true_malicious_clients, len(client_updates))
        
        # Get trust scores
        all_ids = [cid for cid, _, _ in client_updates]
        trust_scores = self.trust_manager.get_trust_scores(all_ids)
        
        log(INFO, "=== Trust-Based Aggregation ===")
        log(INFO, f"Selected: {len(benign_ids)}, Excluded: {len(malicious_ids)}, Quarantined: {len(self.trust_manager.quarantined)}")
        
        # Apply hard gate: remove very low-trust selected clients
        hard_gate_triggered = []
        for cid in benign_ids:
            if trust_scores[cid] < self.trust_tau_drop:
                log(INFO, f"⚠️  Hard gate: Removing client {cid} (τ={trust_scores[cid]:.3f} < {self.trust_tau_drop})")
                hard_gate_triggered.append(cid)
        
        if hard_gate_triggered:
            benign_ids = [cid for cid in benign_ids if cid not in hard_gate_triggered]
            malicious_ids.extend(hard_gate_triggered)
            log(INFO, f"Hard gate removed {len(hard_gate_triggered)} clients")
        
        # Calculate base weights (equal weight for simplicity)
        base_weights = {cid: 1.0 / len(client_updates) for cid in all_ids}
        
        # Calculate penalized weights for selected clients
        selected_weights = {}
        for cid in benign_ids:
            tau = trust_scores[cid]
            base_weight = base_weights[cid]
            
            # Trust penalty
            weight = base_weight * (tau ** self.trust_p)
            
            # Consecutive low-trust dampening
            if tau < self.trust_theta:
                self.trust_manager.streak_low_trust[cid] += 1
                weight *= (self.trust_gamma ** self.trust_manager.streak_low_trust[cid])
                log(INFO, f"  Client {cid}: τ={tau:.3f}, streak={self.trust_manager.streak_low_trust[cid]}, weight={weight:.6f}")
            else:
                self.trust_manager.streak_low_trust[cid] = 0
            
            selected_weights[cid] = weight
        
        # Calculate leak weights for excluded clients
        excluded_weights = {}
        if malicious_ids:
            # Detect attack for conservative leak (Paper Step 4.2)
            attack_detected = any(trust_scores.get(cid, 1.0) < 0.7 for cid in all_ids)
            
            if attack_detected:
                eta = self.trust_eta_conservative
                delta = self.trust_delta_conservative
                leak_cap = self.trust_manager.leak_cap_conservative
                log(INFO, f"Attack detected - using conservative leak (η={eta}, δ={delta}, cap={leak_cap})")
            else:
                eta = self.trust_eta
                delta = self.trust_delta
                leak_cap = self.trust_manager.leak_cap
            
            # Calculate leak weights with cap (Paper Line 252)
            trust_sum = sum(trust_scores[cid] ** self.trust_q for cid in malicious_ids 
                           if cid not in self.trust_manager.quarantined) + 1e-12
            selected_mass = sum(selected_weights.values())
            
            # Paper Eq. 252: LeakMass = min(η * M_sel, cap * M_sel)
            leak_mass = min(eta * selected_mass, leak_cap * selected_mass)
            avg_selected_weight = selected_mass / max(len(benign_ids), 1)
            
            for cid in malicious_ids:
                if cid in self.trust_manager.quarantined:
                    excluded_weights[cid] = 0.0
                else:
                    tau = trust_scores[cid]
                    leak = leak_mass * (tau ** self.trust_q) / trust_sum
                    leak = min(leak, delta * avg_selected_weight)
                    excluded_weights[cid] = leak
            
            log(INFO, f"Leak weights: total={sum(excluded_weights.values()):.6f} ({sum(excluded_weights.values())/selected_mass*100:.1f}% of selected)")
        
        # Paper: Trust penalties directly reduce weights (no renormalization that undoes penalties)
        # We normalize only to ensure weights sum to 1.0 for aggregation, but this preserves
        # the relative penalization from trust scores
        total_weight = sum(selected_weights.values()) + sum(excluded_weights.values())
        
        if total_weight < 1e-12:
            log(INFO, "Warning: Total weight near zero, using uniform weights")
            uniform_weight = 1.0 / len(all_ids)
            for cid in all_ids:
                if cid in benign_ids:
                    selected_weights[cid] = uniform_weight
                else:
                    excluded_weights[cid] = 0.0
            total_weight = sum(selected_weights.values())
        
        # Normalize to sum to 1.0 (preserves trust-based relative penalties)
        for cid in selected_weights:
            selected_weights[cid] /= total_weight
        for cid in excluded_weights:
            excluded_weights[cid] /= total_weight
        
        log(INFO, f"Trust aggregation: selected_mass={sum(selected_weights.values()):.6f}, leak_mass={sum(excluded_weights.values()):.6f}")
        
        # Perform weighted aggregation
        aggregated_state = {}
        for cid, _, state_dict in client_updates:
            weight = selected_weights.get(cid, 0.0) + excluded_weights.get(cid, 0.0)
            
            if weight > 0:
                for key in state_dict:
                    if any(pattern in key for pattern in self.ignore_weights):
                        continue
                    
                    if key not in aggregated_state:
                        aggregated_state[key] = state_dict[key].clone().float() * weight
                    else:
                        aggregated_state[key] += state_dict[key].float() * weight
        
        # Update global model
        global_state = self.global_model.state_dict()
        for key in aggregated_state:
            global_state[key].copy_(aggregated_state[key].to(global_state[key].dtype))
        
        log(INFO, "Trust-based aggregation complete")
        
        return True
    
    def detect_anomalies(self, client_updates: List[Tuple[client_id, num_examples, StateDict]]) -> Tuple[List[int], List[int]]:
        """
        Detect anomalies and update trust state.
        
        Overrides parent to add trust state update.
        """
        # Get base anomaly detection from parent
        malicious_ids, benign_ids = super().detect_anomalies(client_updates)
        
        # Update trust state
        sampled_ids = [cid for cid, _, _ in client_updates]
        total_scores = {cid: self.last_anomaly_scores[cid]['total'] 
                       for cid in self.last_anomaly_scores.keys()}
        
        self.trust_manager.update_state(
            sampled_ids=sampled_ids,
            excluded_ids=malicious_ids,
            total_scores=total_scores,
            anomaly_scores=self.last_anomaly_scores
        )
        
        # Log trust metrics
        trust_scores = self.trust_manager.get_trust_scores(sampled_ids)
        risk_scores = self.trust_manager.get_risk_scores(sampled_ids)
        
        log(INFO, "=== Trust State ===")
        for cid in malicious_ids[:5]:  # Log first 5 excluded
            log(INFO, f"  Excluded {cid}: τ={trust_scores[cid]:.3f}, risk={risk_scores[cid]:.3f}")
        
        if self.trust_manager.quarantined:
            log(INFO, f"⚠️  Quarantined clients: {list(self.trust_manager.quarantined)}")
        
        return malicious_ids, benign_ids
    
    def __repr__(self) -> str:
        return f"FedSPECTRE-Stateful(alpha={self.alpha}, p={self.trust_p}, quarantined={len(self.trust_manager.quarantined)})"

