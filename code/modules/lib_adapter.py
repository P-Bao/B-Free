"""LIBAdapter — Local Information Bottleneck cho B-Free (768-dim).

Chuyển đổi từ GlobalForge LocalInformationBottleneck (REM.py:17-60)
sang dim=768 cho DINOv2 ViT-B/14 của B-Free.

Cơ chế: Fixed Gaussian low-pass với 1 learnable mixing scalar:
    x_hat = (1 - alpha) * x + alpha * blur(x),  alpha = sigmoid(beta)

Thay đổi so với bản gốc:
    - dim mặc định: 1024 → 768
    - Không có thay đổi logic nào khác — Conv2d depthwise (groups=dim)
      tự co giãn theo dim.

Interface contract (xem plan-nguoi1-modules.md):
    LIBAdapter(dim=768, kernel_size=3, tau=0.5)
    forward(feat_tokens (B,N,768), hw=None) -> (feat_hat (B,N,768), mask (B,N)|None)
"""

import math
import torch
import torch.nn as nn


class LIBAdapter(nn.Module):
    """Local Information Bottleneck cho B-Free ViT-B/14 (768-dim).

    Fixed Gaussian low-pass + learnable mixing scalar:
        x_hat = (1 - alpha) * x + alpha * blur(x),  alpha = sigmoid(beta)
    """

    def __init__(self, dim: int = 768, kernel_size: int = 3, tau: float = 0.5):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2

        # Depthwise conv — mỗi channel có 1 Gaussian filter riêng, weights cố định
        self.blur = nn.Conv2d(
            dim, dim, kernel_size,
            padding=padding, groups=dim, bias=False
        )

        # Khởi tạo Gaussian kernel cố định
        sigma = kernel_size / 3.0
        coords = torch.arange(kernel_size).float() - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel_2d = g.outer(g)
        kernel_2d = kernel_2d / kernel_2d.sum()

        with torch.no_grad():
            self.blur.weight.copy_(
                kernel_2d.unsqueeze(0).unsqueeze(0).expand(
                    dim, 1, kernel_size, kernel_size
                )
            )
        self.blur.weight.requires_grad = False  # Gaussian kernel KHÔNG học

        # Scalar duy nhất cần học — khởi tạo -0.85 (giống bản gốc)
        self.beta = nn.Parameter(torch.tensor(-0.85))

    def forward(self, feat_tokens: torch.Tensor, hw: int = None):
        """
        Args:
            feat_tokens: (B, N, C) — patch tokens, C = self.dim
            hw: int — số patch mỗi cạnh (36 cho 504×504, 16 cho 224×224)

        Returns:
            feat_hat: (B, N, C) — tokens sau LIB
            mask: (B, N) hoặc None — giá trị alpha broadcast
        """
        B, N, C = feat_tokens.shape

        if hw is None:
            hw = int(math.sqrt(N))

        # Nếu N không phải perfect square → bypass (safety check)
        if hw * hw != N:
            return feat_tokens, None

        # Reshape: (B, N, C) → (B, C, hw, hw) để áp conv
        x = feat_tokens.transpose(1, 2).reshape(B, C, hw, hw)
        x_smooth = self.blur(x)

        # Mixing: x_hat = (1 - alpha) * x + alpha * blur(x)
        alpha = torch.sigmoid(self.beta)
        x_hat = (1 - alpha) * x + alpha * x_smooth

        # Reshape lại: (B, C, hw, hw) → (B, N, C)
        feat_hat = x_hat.flatten(2).transpose(1, 2)
        mask = alpha.expand(B, N)

        return feat_hat, mask
