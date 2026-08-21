"""Stub cho GSRAdapter — thay thế tạm thời cho modules/gsr_adapter.py của Nguoi 1.

Giữ đúng signature của ``GlobalStructuralReasoning`` (GlobalForge-BE0F/code/models/REM.py:64)
để khi merge bản thật của Nguoi 1, chỉ cần đổi import, không sửa code gọi.

    GlobalStructuralReasoning(dim, window=3, mask_prob=1.0)
        forward(tokens (B,N,C), hw=None) -> tokens (B,N,C)

Bản tạm thời là Identity: trả về tokens nguyên bản.
XÓA file này khi merge với modules/gsr_adapter.py thật của Nguoi 1.
"""

import torch.nn as nn


class GSRAdapter(nn.Module):
    def __init__(self, dim: int = 768, window: int = 3, mask_prob: float = 1.0):
        super().__init__()
        self.dim = dim
        self.identity = nn.Identity()

    def forward(self, tokens, hw=None):
        return self.identity(tokens)
