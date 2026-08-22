"""GSRAdapter — Global Structural Reasoning cho B-Free (768-dim).

Chuyển đổi từ GlobalForge GlobalStructuralReasoning (REM.py:63-108)
sang dim=768 cho DINOv2 ViT-B/14 của B-Free.

Cơ chế: Single-head self-attention + Chebyshev neighborhood masking.
Mask che mọi token trong vùng Chebyshev distance ≤ window → buộc mỗi
token phải gather evidence từ vùng xa, tập trung vào global structure
thay vì local artifact dễ vỡ.

Thay đổi so với bản gốc:
    - dim mặc định: 1024 → 768
    - Không có thay đổi logic nào khác — GSR dùng single-head attention
      (không phải multi-head), nên không cần tính lại num_heads.

Interface contract (xem plan-nguoi1-modules.md):
    GSRAdapter(dim=768, window=3, mask_prob=1.0)
    forward(tokens (B,N,768), hw=None) -> tokens (B,N,768)
"""

import math
import torch
import torch.nn as nn


class GSRAdapter(nn.Module):
    """Global Structural Reasoning cho B-Free ViT-B/14 (768-dim).

    Single-head self-attention với Chebyshev neighborhood masking.
    Residual connection: output = proj(attn @ V) + input.
    """

    def __init__(self, dim: int = 768, window: int = 3, mask_prob: float = 1.0):
        super().__init__()
        self.dim = dim
        self.window = window
        self.mask_prob = mask_prob

        # Single-head attention: Q, K, V, proj đều là Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, hw: int = None):
        """
        Args:
            tokens: (B, N, C) — patch tokens, C = self.dim
            hw: int — số patch mỗi cạnh

        Returns:
            out: (B, N, C) — tokens sau GSR (có residual)
        """
        B, N, C = tokens.shape

        if hw is None:
            hw = int(math.sqrt(N))

        q = self.q(tokens)
        k = self.k(tokens)
        v = self.v(tokens)

        # Attention scores: (B, N, N)
        attn = (q @ k.transpose(1, 2)) / math.sqrt(C)

        # Chebyshev neighborhood masking
        apply_mask = self.window > 0 and hw * hw == N
        if apply_mask and self.training:
            apply_mask = torch.rand(1).item() < self.mask_prob
        elif apply_mask and not self.training:
            apply_mask = self.mask_prob > 0

        if apply_mask:
            device = tokens.device
            idx = torch.arange(hw, device=device)
            grid_y, grid_x = torch.meshgrid(idx, idx, indexing='ij')
            coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # (N, 2)
            diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
            dist = diff.abs().max(dim=-1).values  # Chebyshev distance (N, N)
            mask = (dist <= self.window).to(attn.dtype)
            mask = mask.unsqueeze(0)  # (1, N, N) — broadcast qua batch
            attn = attn.masked_fill(mask > 0, float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v
        out = self.proj(out) + tokens  # Residual connection

        return out
