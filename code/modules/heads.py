"""Heads for the B-Free backbone ablation ladder.

`ClsPlusPatchHead` keeps the B-Free CLS path at step 0: gamma is initialised to
0, so logits == CLS-head logits exactly. Patch information enters only through
gamma, which is logged during training (if it never moves, give it its own LR).
"""

import torch
import torch.nn as nn


class ClsPlusPatchHead(nn.Module):
    """logits = head(cls_feat) + gamma * head(patch_feat_projected)."""

    def __init__(self, dim: int, head: nn.Module, norm: nn.Module, drop: nn.Module,
                 gamma_init: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        # Borrowed references (already owned by the backbone); not re-registered.
        self.head = [head]
        self.norm = [norm]
        self.drop = [drop]

    def logits_from(self, cls_feat: torch.Tensor, patch_feat: torch.Tensor) -> torch.Tensor:
        head, norm, drop = self.head[0], self.norm[0], self.drop[0]
        base = head(drop(norm(cls_feat)))
        extra = head(drop(norm(self.proj(patch_feat))))
        return base + self.gamma * extra
