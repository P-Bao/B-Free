"""B-Free backbone (timm ViT-B/14 reg4 dinov2) + post-hoc LIB/GSR + L_DCS.

Sửa kiến trúc theo quyết định đã chốt (xem agent.md):
  - LIB/GSR áp dụng POST-HOC trên patch tokens cuối (giống GlobalForge REM.py:251-282),
    KHÔNG chèn giữa block 6/12.
  - Feature cho L_DCS = mean-pool patch tokens (REM.py:279), KHÔNG dùng cls token.
  - L_DCS = symmetric InfoNCE (modules.dcs_loss.info_nce_loss).
  - num_classes = 2 (B-Free gốc dùng num_classes=1, nhưng hợp đồng giao diện plan-nguoi2
    yêu cầu logits (B,2); dùng 2 cho CE multi-class real/fake).

Token order của timm vit_*_reg4: [cls, reg×4, patch×N] -> num_prefix_tokens = 5.
Với ảnh 504×504, patch 14: N = 36×36 = 1296, hw = 36.
Với ảnh 224×224 (overfit nhanh): N = 16×16 = 256, hw = 16.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.vision_transformer import VisionTransformer

from modules.lib_adapter import LIBAdapter
from modules.gsr_adapter import GSRAdapter
from modules.dcs_loss import DCSLoss, info_nce_loss


class BFreeGlobalForgeViT(nn.Module):
    """timm ViT-B/14 reg4 dinov2 + post-hoc LIB/GSR trên patch tokens.

    forward(images) -> dict(logits (B,2), cls (B,C))   # cls = mean-pool patch tokens
    compute_loss(images_clean, images_degraded, labels) -> (total, ce, dcs)
    load_checkpoint(path, strict=False)
    """

    def __init__(
        self,
        arch: str = "vit_base_patch14_reg4_dinov2.lvd142m",
        num_classes: int = 2,
        img_size: int = 504,
        pretrained: bool = False,
        use_lib: bool = True,
        use_gsr: bool = True,
        use_dcs: bool = True,
        lib_kernel: int = 3,
        lib_tau: float = 0.5,
        gsr_window: int = 3,
        gsr_mask_prob: float = 1.0,
        dcs_tau: float = 0.07,
        lambda_dcs: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_lib = use_lib
        self.use_gsr = use_gsr
        self.use_dcs = use_dcs
        self.lambda_dcs = float(lambda_dcs)

        # Backbone timm. patch_embed + pos_embed + 12 blocks + norm + head.
        self.model = timm.create_model(
            arch,
            num_classes=num_classes,
            pretrained=pretrained,
        )
        self.model.set_input_size(img_size)
        self.embed_dim = self.model.embed_dim  # 768
        self.num_prefix = self.model.num_prefix_tokens  # 5 (cls + 4 reg)

        # LIB/GSR post-hoc (theo dim=768 của ViT-B/14). Stub = Identity cho bước này.
        self.lib = LIBAdapter(self.embed_dim, kernel_size=lib_kernel, tau=lib_tau) if use_lib else None
        self.gsr = GSRAdapter(self.embed_dim, window=gsr_window, mask_prob=gsr_mask_prob) if use_gsr else None

        # Loss
        self.dcs = DCSLoss(tau=dcs_tau, weight=lambda_dcs) if use_dcs else None
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # ---- feature extraction -------------------------------------------------

    def _patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Chạy backbone -> strip prefix tokens -> (patch_tokens, feat_cls_mean, hw).

        Trả về:
          patch_tokens: (B, N, C)  — N = hw*hw
          feat_cls: (B, C)         — mean-pool của patch tokens (theo REM.py:279)
          hw: int                  — số patch mỗi cạnh (36 cho 504, 16 cho 224)
        """
        x = self.model.forward_features(x)          # (B, 1+4+N, C)
        patch = x[:, self.num_prefix:, :]           # (B, N, C)
        N = patch.shape[1]
        hw = int(math.isqrt(N))
        assert hw * hw == N, f"patch count {N} not a perfect square (hw={hw})"
        # post-hoc LIB -> GSR trên patch tokens (giống REM.py:272-276)
        if self.lib is not None:
            patch, _ = self.lib(patch, hw=hw)
        if self.gsr is not None:
            patch = self.gsr(patch, hw=hw)
        feat_cls = patch.mean(dim=1)                # (B, C)
        return patch, feat_cls, hw

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        patch, feat_cls, hw = self._patch_tokens(images)
        # head dùng feat_cls (mean-pool patch) thay vì cls token, theo GlobalForge.
        # Gọi trực tiếp self.model.head (bypass forward_head vì pool mặc định lấy cls token).
        feat_cls = self.model.fc_norm(feat_cls)
        feat_cls = self.model.head_drop(feat_cls)
        logits = self.model.head(feat_cls)          # (B, num_classes)
        return {"logits": logits, "cls": feat_cls}

    # ---- loss ---------------------------------------------------------------

    def compute_loss(
        self,
        images_clean: torch.Tensor,
        images_degraded: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (total_loss, ce_loss, dcs_loss) — tách bạch cho logging Phase 8.

        CE tính trên logits của view clean (theo convention GlobalForge engine_finetune.py:130).
        L_DCS tính giữa cls_clean và cls_degraded (symmetric InfoNCE).
        total = CE + lambda_dcs * L_DCS.
        """
        out_clean = self.forward(images_clean)
        ce_loss = self.ce(out_clean["logits"], labels)

        dcs_loss = out_clean["logits"].new_zeros(())
        if self.use_dcs and self.dcs is not None:
            out_deg = self.forward(images_degraded)
            dcs_loss = info_nce_loss(out_clean["cls"], out_deg["cls"], tau=self.dcs.tau)

        total = ce_loss + self.lambda_dcs * dcs_loss
        return total, ce_loss, dcs_loss

    # ---- checkpoint ---------------------------------------------------------

    def load_checkpoint(self, path: str, strict: bool = False) -> Dict:
        """Load checkpoint B-Free (định dạng Wrapper5crops: keys 'patch_embed.*' và 'model.*').

        Vì ta không bọc Wrapper5crops, remap:
          'model.X'      -> 'model.X'        (giữ nguyên — sub-module self.model)
          'patch_embed.X'->'model.patch_embed.X'  (Wrapper5crops đã tách patch_embed ra)
        Mọi key của LIB/GSR/dcs/head mới sẽ missing (OK vì strict=False).
        Trả về báo cáo load_state_dict.
        """
        from torch import load

        dat = load(path, map_location="cpu")
        if isinstance(dat, dict) and "model" in dat and not isinstance(dat["model"], torch.Tensor):
            sd = dat["model"]
        elif isinstance(dat, dict) and "state_dict" in dat:
            sd = dat["state_dict"]
        else:
            sd = dat

        # B-Free checkpoint (từ Wrapper5crops.load_state_dict) có keys 'patch_embed.*' và 'model.*'.
        # self.model là timm VisionTransformer: patch_embed nằm trong self.model.patch_embed.
        # Remap 'patch_embed.X' -> 'model.patch_embed.X' để khớp self.state_dict().
        remapped = {}
        for k, v in sd.items():
            if k.startswith("patch_embed."):
                remapped["model." + k] = v
            else:
                remapped[k] = v

        report = self.load_state_dict(remapped, strict=strict)
        return {
            "missing_keys": list(report.missing_keys)[:20],
            "unexpected_keys": list(report.unexpected_keys)[:20],
            "num_missing": len(report.missing_keys),
            "num_unexpected": len(report.unexpected_keys),
        }
