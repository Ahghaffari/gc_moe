"""Simple ensemble baseline: uniform averaging of expert predictions."""

import torch
import torch.nn as nn
from typing import List


class SimpleEnsemble(nn.Module):
    """Averages predictions from multiple frozen experts with uniform weights."""
    
    def __init__(
        self,
        expert_models: List[nn.Module],
        expert_names: List[str],
        node_num: int,
        horizon: int = 12,
        frozen: bool = True
    ):
        super(SimpleEnsemble, self).__init__()
        
        self.num_experts = len(expert_models)
        self.expert_names = expert_names
        self.node_num = node_num
        self.horizon = horizon
        
        self.experts = nn.ModuleList(expert_models)
        
        if frozen:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False
            print(f"[SimpleEnsemble] All {self.num_experts} experts frozen")
        else:
            print(f"[SimpleEnsemble] All {self.num_experts} experts trainable")
        
        self.register_buffer('uniform_weights', torch.ones(self.num_experts) / self.num_experts)
        
        # Tiny learnable scale so backward() works even when experts are frozen.
        # Initialized to 1.0 — has negligible effect but keeps the optimizer happy.
        self.output_scale = nn.Parameter(torch.ones(1))
        
        print(f"[SimpleEnsemble] Initialized with {self.num_experts} experts: {expert_names}")
        print(f"[SimpleEnsemble] Uniform weight per expert: {1.0/self.num_experts:.4f}")
    
    def _call_expert(self, expert, x, label=None, iter_cnt=None):
        """Call an expert with flexible signature handling."""
        try:
            return expert(x, label, iter_cnt)
        except TypeError:
            try:
                return expert(x, label)
            except TypeError:
                try:
                    return expert(x, iter_cnt)
                except TypeError:
                    return expert(x)

    def forward(self, x, label=None, iter_cnt=None, **kwargs):
        """
        Forward pass: Get predictions from all experts and average them.
        
        Args:
            x: Input tensor [B, T, N, F]
            label: Target tensor (passed to experts that need it)
            iter_cnt: Iteration counter (passed to experts that need it)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            averaged_output: [B, horizon, N, output_dim]
        """
        expert_outputs = []
        
        for i, expert in enumerate(self.experts):
            output = self._call_expert(expert, x, label, iter_cnt)

            # Ensure output is [B, horizon, N, output_dim]
            if output.shape[1] > self.horizon:
                output = output[:, -self.horizon:, :, :]
            elif output.shape[1] < self.horizon:
                pad_size = self.horizon - output.shape[1]
                padding = torch.zeros(
                    output.shape[0], pad_size, output.shape[2], output.shape[3],
                    device=output.device, dtype=output.dtype
                )
                output = torch.cat([padding, output], dim=1)
            
            expert_outputs.append(output)
        
        # Stack [E, B, T, N, D] and average over experts dim
        expert_outputs = torch.stack(expert_outputs, dim=0)
        averaged_output = expert_outputs.mean(dim=0)  # [B, T, N, D]
        
        # Scale by learnable param (starts at 1.0) to maintain gradient path
        averaged_output = averaged_output * self.output_scale
        
        return averaged_output
    
    def param_num(self):
        """
        Count trainable parameters.
        For frozen ensemble, this should be 0.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_expert_statistics(self):
        """
        Get statistics about expert usage.
        For simple ensemble, all experts have equal weight.
        """
        stats = {
            'expert_names': self.expert_names,
            'expert_weights': self.uniform_weights.cpu().numpy(),
            'routing_type': 'uniform',
            'num_experts': self.num_experts
        }
        return stats


def create_simple_ensemble_from_config(args, expert_models, expert_names):
    """
    Create a simple ensemble model from configuration.
    
    Args:
        args: Configuration arguments
        expert_models: List of expert models
        expert_names: List of expert names
        
    Returns:
        model: SimpleEnsemble model with optimizer and scheduler
    """
    model = SimpleEnsemble(
        expert_models=expert_models,
        expert_names=expert_names,
        node_num=args.node_num,
        horizon=args.horizon,
        frozen=getattr(args, 'frozen_experts', True)
    )
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if len(trainable_params) > 0:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=getattr(args, 'last_lr', args.lrate),
            weight_decay=getattr(args, 'last_weight_decay', args.wdecay)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.95
        )
    else:
        # Should not happen now (output_scale is always trainable)
        optimizer = torch.optim.Adam([model.output_scale], lr=args.lrate)
        scheduler = None
    
    model.optimizer = optimizer
    model.scheduler = scheduler
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print(f"Simple Ensemble Model Summary:")
    print(f"{'='*60}")
    print(f"Number of experts: {len(expert_models)}")
    print(f"Expert names: {expert_names}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100.0 * trainable_params / max(total_params, 1):.2f}%)")
    print(f"Frozen: {getattr(args, 'frozen_experts', True)}")
    print(f"Averaging strategy: Uniform (1/{len(expert_models)} per expert)")
    print(f"{'='*60}\n")
    
    return model
