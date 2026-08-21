"""Stub cho LIBAdapter — thay thế tạm thời cho modules/lib_adapter.py của Nguoi 1.

Giữ đúng signature của ``LocalInformationBottleneck`` (GlobalForge-BE0F/code/models/REM.py:18)
để khi merge bản thật của Nguoi 1, chỉ cần đổi import, không sửa code gọi.

    LocalInformationBottleneck(dim, kernel_size=3, tau=0.5)
        forward(feat_tokens (B,N,C), hw=None) -> (feat_hat (B,N,C), mask (B,N)|None)

Bản tạm thời là Identity: trả về (feat_tokens, None) — không thay đổi token.
XÓA file này khi merge với modules/lib_adapter.py thật của Nguoi 1.
"""

import torch.nn as nn


class LIBAdapter(nn.Module):
    def __init__(self, dim: int = 768, kernel_size: int = 3, tau: float = 0.5):
        super().__init__()
        self.dim = dim
        self.identity = nn.Identity()

    def forward(self, feat_tokens, hw=None):
        return self.identity(feat_tokens), None
