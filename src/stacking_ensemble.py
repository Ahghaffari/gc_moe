"""Stacking ensemble baseline: meta-learner MLP on concatenated expert outputs."""

import torch
import torch.nn as nn
from typing import List


class StackingEnsemble(nn.Module):
    """Two-stage stacking: frozen experts produce preds, meta-learner MLP combines them."""

    def __init__(
        self,
        expert_models: List[nn.Module],
        expert_names: List[str],
        node_num: int,
        output_dim: int = 1,
        horizon: int = 12,
        hidden_dim: int = 64,
        meta_layers: int = 2,
        dropout: float = 0.1,
        frozen: bool = True,
    ):
        super().__init__()

        self.num_experts = len(expert_models)
        self.expert_names = expert_names
        self.node_num = node_num
        self.output_dim = output_dim
        self.horizon = horizon

        # --- Experts ---
        self.experts = nn.ModuleList(expert_models)
        if frozen:
            for expert in self.experts:
                for param in expert.parameters():
                    param.requires_grad = False
            print(f"[StackingEnsemble] All {self.num_experts} experts frozen")
        else:
            print(f"[StackingEnsemble] All {self.num_experts} experts trainable")

        # --- Meta-learner ---
        # Input: concatenation of all expert outputs [B, T, N, E * output_dim]
        meta_input_dim = self.num_experts * output_dim
        layers = []
        in_dim = meta_input_dim
        for _ in range(meta_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.meta_learner = nn.Sequential(*layers)

        # Per-node bias (captures node-specific correction)
        self.node_bias = nn.Parameter(torch.zeros(1, 1, node_num, output_dim))

        print(f"[StackingEnsemble] Meta-learner: {meta_input_dim} -> "
              f"{meta_layers}x{hidden_dim} -> {output_dim}")

    # ------------------------------------------------------------------ #
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
        Forward pass:
          1. Get predictions from each expert (detached if frozen).
          2. Concatenate along feature dim.
          3. Feed through meta-learner.

        Returns:
            final_output: [B, horizon, N, output_dim]
        """
        expert_outputs = []

        with torch.no_grad():
            for expert in self.experts:
                out = self._call_expert(expert, x, label, iter_cnt)
                if isinstance(out, tuple):
                    out = out[0]

                # Ensure [B, horizon, N, D]
                if out.shape[1] > self.horizon:
                    out = out[:, -self.horizon:, :, :]
                elif out.shape[1] < self.horizon:
                    pad = self.horizon - out.shape[1]
                    out = torch.cat([
                        torch.zeros(out.shape[0], pad, out.shape[2], out.shape[3],
                                    device=out.device, dtype=out.dtype),
                        out
                    ], dim=1)
                expert_outputs.append(out)

        # Concatenate: [B, T, N, E*D]
        concat = torch.cat(expert_outputs, dim=-1)

        # Meta-learner
        final = self.meta_learner(concat)  # [B, T, N, D]
        final = final + self.node_bias

        return final

    def param_num(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_expert_statistics(self):
        return {
            'expert_names': self.expert_names,
            'routing_type': 'stacking_meta_learner',
            'num_experts': self.num_experts,
        }


# ---------------------------------------------------------------------- #
def create_stacking_ensemble_from_config(args, expert_models, expert_names):
    """
    Factory function to build a StackingEnsemble from CLI args.

    Expected special args:
        --stacking_hidden_dim  (default 64)
        --stacking_meta_layers (default 2)
        --stacking_dropout     (default 0.1)
    """
    model = StackingEnsemble(
        expert_models=expert_models,
        expert_names=expert_names,
        node_num=args.node_num,
        output_dim=args.output_dim,
        horizon=args.horizon,
        hidden_dim=getattr(args, 'stacking_hidden_dim', 64),
        meta_layers=getattr(args, 'stacking_meta_layers', 2),
        dropout=getattr(args, 'stacking_dropout', 0.1),
        frozen=getattr(args, 'frozen_experts', True),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=getattr(args, 'last_lr', args.lrate),
        weight_decay=getattr(args, 'last_weight_decay', args.wdecay),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

    model.optimizer = optimizer
    model.scheduler = scheduler

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)

    print(f"\n{'='*60}")
    print(f"Stacking Ensemble Model Summary:")
    print(f"{'='*60}")
    print(f"  Experts: {expert_names}")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,} ({100*trainable/max(total,1):.2f}%)")
    print(f"  Meta-learner hidden dim: {getattr(args, 'stacking_hidden_dim', 64)}")
    print(f"  Meta-learner layers: {getattr(args, 'stacking_meta_layers', 2)}")
    print(f"{'='*60}\n")

    return model
