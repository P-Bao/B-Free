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
from modules.dcs_loss import DCSLoss, info_nce_loss, class_aware_info_nce, multi_severity_dcs
from modules.heads import ClsPlusPatchHead


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
        lib_beta_init: float = -0.85,
        gsr_window: int = 3,
        gsr_mask_prob: float = 1.0,
        gsr_zero_init: bool = False,
        head_mode: str = "meanpool",
        dcs_tau: float = 0.07,
        lambda_dcs: float = 0.1,
        dcs_mode: str = "legacy",
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        if head_mode not in ("meanpool", "cls", "cls_plus_patch"):
            raise ValueError(f"head_mode must be meanpool|cls|cls_plus_patch, got {head_mode!r}")
        if dcs_mode not in ("legacy", "class_aware"):
            raise ValueError(f"dcs_mode must be legacy|class_aware, got {dcs_mode!r}")
        self.num_classes = num_classes
        self.use_lib = use_lib
        self.use_gsr = use_gsr
        self.use_dcs = use_dcs
        self.lambda_dcs = float(lambda_dcs)
        self.head_mode = head_mode
        self.dcs_mode = dcs_mode

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
        self.lib = LIBAdapter(self.embed_dim, kernel_size=lib_kernel, tau=lib_tau,
                              beta_init=lib_beta_init) if use_lib else None
        self.gsr = GSRAdapter(self.embed_dim, window=gsr_window, mask_prob=gsr_mask_prob,
                              zero_init=gsr_zero_init) if use_gsr else None

        # Loss
        self.dcs = DCSLoss(tau=dcs_tau, weight=lambda_dcs) if use_dcs else None
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # cls_plus_patch: gamma=0 nên bước 0 ≡ CLS head (AGENT_PLAN_V5 1.4).
        self.extra_head = (ClsPlusPatchHead(self.embed_dim, self.model.head,
                                            self.model.fc_norm, self.model.head_drop)
                           if head_mode == "cls_plus_patch" else None)
        # Per-view DCS terms of the last compute_loss (multi-severity mode) + CE terms, for logging.
        self.last_dcs_terms: Dict[str, torch.Tensor] = {}
        self.last_terms: Dict[str, torch.Tensor] = {}

    # ---- feature extraction -------------------------------------------------

    def _patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Chạy backbone -> strip prefix tokens -> (patch_tokens, prefix_feats, hw, feat_cls_mean).

        Trả về:
          patch_tokens: (B, N, C)  — N = hw*hw
          prefix_feats: (B, 5, C)  — [cls, reg×4] trước LIB/GSR (dùng cho head_mode=cls)
          hw: int                  — số patch mỗi cạnh (36 cho 504, 16 cho 224)
          feat_cls: (B, C)         — mean-pool của patch tokens sau LIB/GSR (theo REM.py:279)
        """
        x = self.model.forward_features(x)          # (B, 1+4+N, C)
        prefix = x[:, :self.num_prefix, :]
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
        return patch, prefix, hw, feat_cls

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        patch, prefix, hw, patch_mean = self._patch_tokens(images)
        if self.head_mode == "cls":
            # B-Free gốc: token 0 -> fc_norm -> head_drop -> head. DCS dùng chính
            # feature này (nhất quán với classifier), không dùng mean-pool.
            feats = self.model.head_drop(self.model.fc_norm(prefix[:, 0]))
            return {"logits": self.model.head(feats), "cls": feats}
        # meanpool (mặc định, đường cũ) và cls_plus_patch (gamma=0 -> ≡ cls).
        feats = self.model.head_drop(self.model.fc_norm(patch_mean))
        if self.extra_head is not None:
            logits = self.extra_head.logits_from(prefix[:, 0], patch_mean)
        else:
            logits = self.model.head(feats)
        return {"logits": logits, "cls": feats}

    # ---- loss ---------------------------------------------------------------

    def compute_loss(
        self,
        images_clean: torch.Tensor,
        images_degraded,
        labels: torch.Tensor,
        view_weights=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả về (total_loss, ce_loss, dcs_loss) — tách bạch cho logging Phase 8.

        CE tính trên logits của view clean (theo convention GlobalForge engine_finetune.py:130).
        L_DCS tính giữa cls_clean và cls_degraded (symmetric InfoNCE).
        total = CE + lambda_dcs * L_DCS.

        ``images_degraded``: Tensor (đường mặc định, không đổi so với v2) hoặc list K Tensor
        (multi-severity). Với list, dcs_loss = trung bình có trọng số (``view_weights``,
        mặc định đều) của K symmetric InfoNCE; từng term detached ghi vào
        ``self.last_dcs_terms`` = {"dcs_k1": ..., ...}. Hợp đồng 3 scalar giữ nguyên.
        """
        out_clean = self.forward(images_clean)
        ce_clean = self.ce(out_clean["logits"], labels)

        dcs_loss = out_clean["logits"].new_zeros(())
        self.last_dcs_terms = {}
        if self.use_dcs and self.dcs is not None:
            if isinstance(images_degraded, (list, tuple)):
                if self.dcs_mode == "class_aware":
                    raise ValueError("dcs_mode=class_aware chỉ hỗ trợ 1 view (multi_severity tắt)")
                cls_views = [self.forward(v)["cls"] for v in images_degraded]
                dcs_loss, terms = multi_severity_dcs(
                    out_clean["cls"], cls_views, weights=view_weights, tau=self.dcs.tau)
                self.last_dcs_terms = {f"dcs_k{i + 1}": t for i, t in enumerate(terms)}
            else:
                out_deg = self.forward(images_degraded)
                if self.dcs_mode == "class_aware":
                    dcs_loss = class_aware_info_nce(out_clean["cls"], out_deg["cls"],
                                                    labels, tau=self.dcs.tau)
                else:
                    dcs_loss = info_nce_loss(out_clean["cls"], out_deg["cls"], tau=self.dcs.tau)

        total = ce_clean + self.lambda_dcs * dcs_loss
        self.last_terms = {"ce_clean": ce_clean.detach(), "dcs": dcs_loss.detach()}
        return total, ce_clean, dcs_loss

    # ---- checkpoint ---------------------------------------------------------

    def load_checkpoint(self, path: str, strict: bool = False) -> Dict:
        """Load checkpoint B-Free (định dạng lưu bởi `Wrapper5crops.load_state_dict`).

        Checkpoint có keys `patch_embed.*` + các key timm KHÔNG tiền tố `model.`
        (`blocks.0...`, `cls_token`, `pos_embed`, `head.*`). Ta remap:
          'patch_embed.X' -> 'model.patch_embed.X'
          'X'             -> 'model.X'
        Head B-Free 1 logit -> 2 logit sao cho l1-l0 = l (AGENT_PLAN_V5 mục 10):
          W2 = [-w/2; +w/2], b2 = [-b/2; +b/2].
        Key lạ/sai shape bị bỏ qua và báo lại trong report (không im lặng).
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

        ref = self.state_dict()
        remapped, skipped = {}, []
        for k, v in sd.items():
            if k.startswith("patch_embed."):
                k2 = "model." + k
            elif k.startswith("model."):
                k2 = k
            else:
                k2 = "model." + k
            if k2 == "model.head.weight" and v.shape[0] == 1 and ref[k2].shape[0] == 2:
                v = torch.cat([-0.5 * v, 0.5 * v], dim=0)
            elif k2 == "model.head.bias" and v.shape[0] == 1 and ref[k2].shape[0] == 2:
                v = torch.cat([-0.5 * v, 0.5 * v], dim=0)
            if k2 not in ref or ref[k2].shape != v.shape:
                skipped.append(k2)
                continue
            remapped[k2] = v

        report = self.load_state_dict(remapped, strict=strict)
        return {
            "missing_keys": list(report.missing_keys),
            "unexpected_keys": list(report.unexpected_keys),
            "num_missing": len(report.missing_keys),
            "num_unexpected": len(report.unexpected_keys),
            "skipped_shape": skipped,
            "num_skipped_shape": len(skipped),
        }
