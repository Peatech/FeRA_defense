"""
Anticipate attack malicious client implementation for FL.
Reference: https://github.com/YuxinWenRick/thinking-two-moves-ahead
"""
import torch
import copy

from torch.func import functional_call
from logging import INFO, WARNING
from typing import Dict, Tuple, Any
from backfed.clients.base_malicious_client import MaliciousClient
from backfed.utils import log
from backfed.const import Metrics, StateDict

DEFAULT_PARAMS = {
    "anticipate_steps": 9,  # Number of future steps to anticipate
    "anticipate_lr": 0.1,  # Learning rate for anticipate optimizer
    "anticipate_momentum": 0.9,  # Momentum for anticipate optimizer
    "anticipate_gamma": 0.998,  # Learning rate decay for anticipate optimizer
    "num_benign_users": 9,  # Estimated number of benign users
}
    
class AnticipateClient(MaliciousClient):
    """
    Anticipate attack client that anticipates future aggregation steps.
    
    This attack simulates future federated learning rounds to craft
    malicious updates that will be effective after aggregation with
    benign clients' updates.
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
        client_type: str = "anticipate",
        verbose: bool = True,
        **kwargs
    ):
        # Merge default parameters with provided params
        params_to_update = DEFAULT_PARAMS.copy()
        params_to_update.update(kwargs)
        
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
    
    def _set_optimizer(self):
        """
        We don't need to create optimizer for AnticipateClient.
        """
        pass
            
    def _create_anticipate_optimizer(self, params, epoch=0):
        """Create optimizer for anticipate training."""
        return torch.optim.SGD(
            params,
            lr=self.atk_config['anticipate_lr'],
            momentum=self.atk_config['anticipate_momentum']
        )
    
    def _train_with_functorch(
        self,
        model: torch.nn.Module,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        train_loader,
        lr: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate training using functorch for differentiation through optimization.
        Based on the reference train_with_functorch implementation.
        
        Args:
            model: The model architecture (for functional_call)
            params: Current model parameters
            buffers: Current model buffers
            train_loader: Training data loader
            lr: Learning rate for this simulation
            num_users: Number of users (not used, kept for compatibility)
            
        Returns:
            Updated parameters after simulated training
        """
        from torch.func import grad
        
        def compute_loss(params_dict, buffers_dict, x, y):
            """Compute loss for a batch using functional_call."""
            logits = functional_call(
                model,
                {**params_dict, **buffers_dict},
                x
            )
            loss = self.criterion(logits, y).mean()
            return loss
        
        # Convert dict params to list for grad computation
        param_names = list(params.keys())
        param_list = [params[name] for name in param_names]
        
        # Simulate training for one batch with local_epochs
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Multiple local epochs on the same batch
            for _ in range(self.client_config["local_epochs"]):
                # Reconstruct params dict for grad computation
                params_dict = {name: param for name, param in zip(param_names, param_list)}
                
                # Compute gradients using functorch.grad
                grads_dict = grad(compute_loss)(params_dict, buffers, inputs, targets)
                
                # Update parameters: p = p - lr * grad
                param_list = [
                    param - grads_dict[name] * lr
                    for param, name in zip(param_list, param_names)
                ]
            
            # Only process first batch as in reference
            break
        
        # Convert back to dictionary
        updated_params = {name: param for name, param in zip(param_names, param_list)}
        return updated_params
    
    def _simulate_benign_update(
        self,
        model: torch.nn.Module,
        curr_params: Dict[str, torch.Tensor],
        curr_buffers: Dict[str, torch.Tensor],
        train_loader,
        server_round: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate benign clients' gradient updates using functorch.
        
        Args:
            model: The model architecture (for functional_call)
            curr_params: Current model parameters
            curr_buffers: Current model buffers
            train_loader: Training data loader
            server_round: Current server round for learning rate decay
            
        Returns:
            Updated parameters after simulated benign updates
        """
        # Calculate learning rate with decay (gamma^round)
        lr = self.client_config.lr * (self.atk_config['anticipate_gamma'] ** server_round)
        
        # Use functorch-based training simulation
        updated_params = self._train_with_functorch(
            model=model,
            params=curr_params,
            buffers=curr_buffers,
            train_loader=train_loader,
            lr=lr,
        )
        
        return updated_params
    
    def _aggregate_params(
        self,
        attack_params: Dict[str, torch.Tensor],
        benign_params: Dict[str, torch.Tensor],
        num_benign: int
    ) -> Dict[str, torch.Tensor]:
        """
        Simulate federated averaging aggregation.
        
        Args:
            attack_params: Malicious client parameters
            benign_params: Aggregated benign parameters
            num_benign: Number of benign clients
            
        Returns:
            Aggregated parameters
        """
        total_users = num_benign + 1  # benign + 1 malicious
        
        aggregated = {}
        for name in attack_params:
            aggregated[name] = (
                attack_params[name] + benign_params[name] * num_benign
            ) / total_users
        
        return aggregated
    
    def train(self, train_package: Dict[str, Any]) -> Tuple[int, StateDict, Metrics]:
        """
        Train the anticipate malicious client with multi-step anticipation.
        
        Args:
            train_package: Training package from server
            
        Returns:
            Tuple of (client_id, model_update, metrics)
        """
        self.model.train()
        self.model.to(self.device)
        
        # Get global model parameters
        global_params = train_package.get("global_params", None)
        if global_params:
            self.model.load_state_dict(global_params)
        
        server_round = train_package.get("server_round", 0)
        
        # Update poison module if needed
        selected_malicious_clients = train_package.get("selected_malicious_clients", [self.client_id])
        normalization = train_package.get("normalization", None)
        self._update_and_sync_poison(selected_malicious_clients, server_round, normalization)
        
        # Create attack model
        attack_model = copy.deepcopy(self.model)
        
        # Get parameters and buffers as dictionaries
        attack_params = {
            name: param.clone().detach().requires_grad_(True)
            for name, param in attack_model.named_parameters()
        }
        attack_buffers = {
            name: buffer.clone().detach()
            for name, buffer in attack_model.named_buffers()
        }
        
        # Determine number of training epochs
        if self.atk_config.poison_until_convergence:
            num_epochs = 100  # Large number for convergence-based training
            log(WARNING, f"Client [{self.client_id}] ({self.client_type}) at round {server_round} "
                "- Training until convergence of backdoor loss")
        else:
            num_epochs = self.atk_config.poison_epochs
            
        # Create optimizer with parameter values
        opt_params = list(attack_params.values())
        optimizer = self._create_anticipate_optimizer(opt_params, epoch=server_round)
        
        total_loss = 0.0
        num_batches = 0
        
        # Anticipatory training
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                # Get current model parameters and buffers
                curr_params = {
                    name: param.clone().detach().requires_grad_(True)
                    for name, param in self.model.named_parameters()
                }
                curr_buffers = {
                    name: buffer.clone().detach()
                    for name, buffer in self.model.named_buffers()
                }
                
                # Prepare poisoned batch
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Apply poison to batch
                poisoned_inputs, poisoned_targets = self.poison_module.poison_batch(batch=(inputs, targets))
                
                optimizer.zero_grad()
                loss = None
                
                # Multi-step anticipation
                for anticipate_step in range(self.atk_config['anticipate_steps']):
                    if anticipate_step == 0:
                        # Step 1: Estimate benign users' updates
                        benign_params = self._simulate_benign_update(
                            self.model,
                            curr_params,
                            curr_buffers,
                            self.train_loader,
                            server_round=server_round + anticipate_step
                        )
                        
                        # Step 2: Simulate aggregation with attack params
                        curr_params = self._aggregate_params(
                            attack_params,
                            benign_params,
                            self.atk_config['num_benign_users'] - 1
                        )
                    else:
                        # Subsequent steps: Simulate normal benign updates
                        curr_params = self._simulate_benign_update(
                            self.model,
                            curr_params,
                            curr_buffers,
                            self.train_loader,
                            server_round=server_round + anticipate_step
                        )
                    
                    # Compute adversarial loss at this anticipation step
                    logits = functional_call(
                        self.model,
                        {**curr_params, **curr_buffers},
                        poisoned_inputs
                    )
                    
                    step_loss = self.criterion(logits, poisoned_targets).mean()
                    
                    if loss is None:
                        loss = step_loss
                    else:
                        loss += step_loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if self.verbose and batch_idx % (len(self.train_loader) // 3) == 0:
                    log(INFO, f"Client [{self.client_id}] ({self.client_type}) - "
                             f"Round {server_round}, Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                             f"Loss: {loss.item():.4f}")
        
        # Copy optimized parameters back to attack model
        with torch.no_grad():
            for name, param in attack_model.named_parameters():
                param.copy_(attack_params[name])
            for name, buffer in attack_model.named_buffers():
                buffer.copy_(attack_buffers[name])
                
        train_loss = total_loss / max(num_batches, 1)
        training_metrics = {
            "train_backdoor_loss": train_loss,
        }

        return len(self.train_dataset), attack_model.state_dict(), training_metrics
