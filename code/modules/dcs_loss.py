"""L_DCS: degradation-aware contrastive structural loss.

Tái triển khai đúng công thức thật từ GlobalForge-BE0F/code/engine_finetune.py:29-37
(hàm ``info_nce_loss``): symmetric InfoNCE giữa CLS/mean-pooled feature của 2 view
(clean và degraded) cùng một ảnh.

    z_hat      = normalize(z)
    z_aug_hat  = normalize(z_aug)
    S          = z_hat @ z_aug_hat^T / tau          (B, B)
    labels     = arange(B)
    L_DCS      = 0.5 * (CE(S, labels) + CE(S^T, labels))

L_DCS không phụ thuộc cấu trúc dữ liệu riêng của GlobalForge (RealDeg-Bench / VAE)
— chỉ cần cặp feature (z, z_aug) shape (B, C) và nhiệt độ tau.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def info_nce_loss(z: torch.Tensor, z_aug: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE cho cặp CLS/mean-pooled feature shape (B, C).

    Bản sao chính xác của ``GlobalForge-BE0F/code/engine_finetune.py:29-37``.
    """
    z = F.normalize(z, dim=-1)
    z_aug = F.normalize(z_aug, dim=-1)
    logits = torch.matmul(z, z_aug.t()) / tau  # (B, B)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss1 = F.cross_entropy(logits, labels)
    loss2 = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss1 + loss2)


class DCSLoss(nn.Module):
    """Wrapper nn.Module cho L_DCS.

    ``weight`` là lambda_dcs (hệ số nhân khi ghép vào L_total), giữ trong config.
    ``forward`` chỉ trả về giá trị loss thuần (không nhân weight) để caller tự nhân
    — tách bạch cho logging.
    """

    def __init__(self, tau: float = 0.07, weight: float = 0.01):
        super().__init__()
        self.tau = float(tau)
        self.weight = float(weight)

    def forward(self, feat_clean: torch.Tensor, feat_degraded: torch.Tensor) -> torch.Tensor:
        return info_nce_loss(feat_clean, feat_degraded, self.tau)
