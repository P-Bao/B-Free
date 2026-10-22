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
from modules.dcs_loss import DCSLoss, info_nce_loss, multi_severity_dcs


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
        head_mode: str = "meanpool",
        gsr_zero_init: bool = False,
        lib_beta_init: float = -0.85,
        loss_mode: str = "legacy",
        w_deg: float = 1.0,
        lam_cons: float = 0.0,
        cons_T: float = 1.0,
    ):
        super().__init__()
        if head_mode not in ("cls", "meanpool"):
            raise ValueError(f"head_mode must be 'cls' or 'meanpool', got {head_mode!r}")
        if loss_mode not in ("legacy", "dual_view"):
            raise ValueError(f"loss_mode must be 'legacy' or 'dual_view', got {loss_mode!r}")
        if cons_T <= 0:
            raise ValueError(f"cons_T must be > 0, got {cons_T}")
        # head_mode mặc định của CLASS giữ "meanpool" để checkpoint cũ nạp/eval không đổi;
        # mặc định cho training mới ("cls") nằm ở configs/bfree_dcs.yaml (backbone.head_mode).
        self.head_mode = head_mode
        self.loss_mode = loss_mode
        self.w_deg = float(w_deg)
        self.lam_cons = float(lam_cons)
        self.cons_T = float(cons_T)
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
        self.lib = (LIBAdapter(self.embed_dim, kernel_size=lib_kernel, tau=lib_tau,
                               beta_init=lib_beta_init) if use_lib else None)
        self.gsr = (GSRAdapter(self.embed_dim, window=gsr_window, mask_prob=gsr_mask_prob,
                               zero_init_gate=gsr_zero_init) if use_gsr else None)

        # Loss
        self.dcs = DCSLoss(tau=dcs_tau, weight=lambda_dcs) if use_dcs else None
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # Per-view DCS terms of the last compute_loss (multi-severity mode), detached, for logging.
        self.last_dcs_terms: Dict[str, torch.Tensor] = {}
        # Per-term CE/consistency of the last compute_loss (dual_view mode), detached, for logging.
        self.last_terms: Dict[str, torch.Tensor] = {}

    # ---- feature extraction -------------------------------------------------

    def _patch_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Chạy backbone -> strip prefix tokens -> (patch_tokens, feat_cls_mean, hw).

        Trả về:
          patch_tokens: (B, N, C)  — N = hw*hw
          feat_cls: (B, C)         — mean-pool của patch tokens (theo REM.py:279)
          hw: int                  — số patch mỗi cạnh (36 cho 504, 16 cho 224)
        """
        _, patch, feat_cls, hw = self._encode(x)
        return patch, feat_cls, hw

    def _encode(self, x: torch.Tensor):
        """Như _patch_tokens nhưng trả thêm ``feats`` đầy đủ (B, 1+4+N, C) từ forward_features
        (chưa qua LIB/GSR) — cần cho head_mode='cls' (timm forward_head lấy token 0)."""
        feats = self.model.forward_features(x)      # (B, 1+4+N, C)
        x = feats
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
        return feats, patch, feat_cls, hw

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats, patch, feat_cls, hw = self._encode(images)
        if self.head_mode == "cls":
            # P2.B: head đọc CLS token của backbone (như B-Free gốc) + độ lệch residual từ LIB/GSR
            # trên mean-pool patch tokens: cls_in = cls_token + (feat_cls - mean(patch_raw)).
            # - Khi không dùng LIB/GSR (hoặc gate=0, alpha=0): cls_in == feats[:, 0, :] -> trùng
            #   khít self.model.forward_head(feats) của B-Free gốc.
            # - Khi gsr_zero_init=True (gate=0) và lib_beta_init=-6 (alpha0~0.0025): ở bước 0
            #   |Δ logits| ~ 1e-4 < 1e-3 (Gate "bước 0 ≡ B-Free"), đồng thời LIB/GSR vẫn nhận
            #   gradient trực tiếp từ CE trong các run R3 (+LIB) và R4 (+GSR).
            patch_raw_mean = feats[:, self.num_prefix:, :].mean(dim=1)
            cls_in = feats[:, 0, :] + (feat_cls - patch_raw_mean)
            cls_in = self.model.fc_norm(cls_in)
            cls_in = self.model.head_drop(cls_in)
            logits = self.model.head(cls_in)         # (B, num_classes)
            feat_cls = self.model.fc_norm(feat_cls)
            feat_cls = self.model.head_drop(feat_cls)
            return {"logits": logits, "cls": feat_cls}
        # meanpool (legacy): head dùng feat_cls (mean-pool patch) thay vì cls token, theo GlobalForge.
        # Gọi trực tiếp self.model.head (bypass forward_head vì pool mặc định lấy cls token).
        feat_cls = self.model.fc_norm(feat_cls)
        feat_cls = self.model.head_drop(feat_cls)
        logits = self.model.head(feat_cls)          # (B, num_classes)
        return {"logits": logits, "cls": feat_cls}

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

        ``loss_mode == "dual_view"`` (P2.A): xem ``_compute_loss_dual_view``.
        """
        if self.loss_mode == "dual_view":
            return self._compute_loss_dual_view(images_clean, images_degraded, labels)

        out_clean = self.forward(images_clean)
        ce_loss = self.ce(out_clean["logits"], labels)

        dcs_loss = out_clean["logits"].new_zeros(())
        self.last_dcs_terms = {}
        self.last_terms = {}
        if self.use_dcs and self.dcs is not None:
            if isinstance(images_degraded, (list, tuple)):
                cls_views = [self.forward(v)["cls"] for v in images_degraded]
                dcs_loss, terms = multi_severity_dcs(
                    out_clean["cls"], cls_views, weights=view_weights, tau=self.dcs.tau)
                self.last_dcs_terms = {f"dcs_k{i + 1}": t for i, t in enumerate(terms)}
            else:
                out_deg = self.forward(images_degraded)
                dcs_loss = info_nce_loss(out_clean["cls"], out_deg["cls"], tau=self.dcs.tau)

        total = ce_loss + self.lambda_dcs * dcs_loss
        return total, ce_loss, dcs_loss

    def _compute_loss_dual_view(
        self,
        images_clean: torch.Tensor,
        images_degraded,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """P2.A — view suy giảm tham gia phân loại (không tốn thêm forward: view này đã được
        forward cho DCS).

            total = CE(clean) + w_deg * CE(deg) + lam_cons * KL(clean.detach() || deg) + lambda_dcs * DCS

        Trả về (total, ce_clean, dcs) — giữ hợp đồng 3 scalar; ``self.last_terms`` =
        {ce_c, ce_d, cons} (detached) để log. ``w_deg=0, lam_cons=0`` ≡ đường legacy.
        """
        if isinstance(images_degraded, (list, tuple)):
            raise ValueError("loss.mode=dual_view không hỗ trợ multi_severity (list view); "
                             "tắt multi_severity hoặc dùng loss.mode=legacy")
        out_c = self.forward(images_clean)
        out_d = self.forward(images_degraded)
        ce_c = self.ce(out_c["logits"], labels)
        ce_d = self.ce(out_d["logits"], labels)         # nhãn không đổi dưới suy giảm
        T = self.cons_T
        cons = F.kl_div(
            F.log_softmax(out_d["logits"].float() / T, dim=-1),
            F.softmax(out_c["logits"].detach().float() / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)                                      # nhất quán quyết định clean -> deg
        dcs = out_c["logits"].new_zeros(())
        if self.use_dcs and self.dcs is not None:
            dcs = info_nce_loss(out_c["cls"], out_d["cls"], tau=self.dcs.tau)

        total = ce_c + self.w_deg * ce_d + self.lam_cons * cons + self.lambda_dcs * dcs
        self.last_dcs_terms = {}
        self.last_terms = {"ce_c": ce_c.detach(), "ce_d": ce_d.detach(), "cons": cons.detach()}
        return total, ce_c, dcs

    # ---- checkpoint ---------------------------------------------------------

    _OURS_ONLY_PREFIXES = ("lib.", "gsr.", "dcs.")

    def remap_bfree_state_dict(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Đưa state_dict B-Free (hoặc dạng đã có tiền tố) về không gian key của self.state_dict().

        - Bỏ tiền tố ``module.`` (DDP) nếu có.
        - Key của module riêng của ta (``lib.``/``gsr.``/``dcs.``) giữ nguyên.
        - Mọi key còn lại: bỏ ``model.`` nếu có (một lần) rồi thêm ``model.`` đúng một lần. Gồm cả
          ``patch_embed.*`` (Wrapper5crops tách riêng) và các key timm trần (``blocks.0...``,
          ``cls_token``, ``head.*``) — checkpoint B-Free chính thức KHÔNG có tiền tố ``model.``.
        - Head: B-Free 1 logit ``l = w.f + b`` -> 2 logit sao cho ``z = l1 - l0 = l``:
          ``W2 = [-w/2; +w/2]``, ``b2 = [-b/2; +b/2]`` (khi ``num_classes == 2``).
        """
        out: Dict[str, torch.Tensor] = {}
        for k, v in sd.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if k.startswith(self._OURS_ONLY_PREFIXES):
                out[k] = v
                continue
            if k.startswith("model."):
                k = k[len("model."):]
            out["model." + k] = v

        for name in ("weight", "bias"):
            key = f"model.head.{name}"
            if key not in out:
                continue
            t = out[key]
            if self.num_classes == 2 and t.shape[0] == 1:
                out[key] = torch.cat([-t / 2, t / 2], dim=0)
            elif t.shape[0] != self.num_classes:
                raise ValueError(
                    f"{key}: checkpoint có {t.shape[0]} logit, model cần {self.num_classes}")
        return out

    def load_checkpoint(self, path: str, strict: bool = False,
                        require_complete_backbone: bool = False) -> Dict:
        """Load checkpoint B-Free (định dạng Wrapper5crops: key 'patch_embed.*' + key timm trần).

        Xử lý key/head xem ``remap_bfree_state_dict``. pos_embed khác kích thước (img_size khác
        504) được resample. Mọi key của LIB/GSR/dcs sẽ missing (khởi tạo tại __init__).

        ``require_complete_backbone=True``: raise RuntimeError nếu còn key ``model.*`` missing
        hoặc có key unexpected — chặn kiểu lỗi "bỏ qua im lặng" của ``strict=False``.
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

        remapped = self.remap_bfree_state_dict(sd)

        pe_key = "model.pos_embed"
        if pe_key in remapped:
            pe = remapped[pe_key]
            new_grid = tuple(self.model.patch_embed.grid_size)
            n = pe.shape[1]
            prefix = 0 if math.isqrt(n) ** 2 == n else self.num_prefix
            old_hw = math.isqrt(n - prefix)
            if old_hw * old_hw != n - prefix:
                raise ValueError(f"cannot parse pos_embed layout {tuple(pe.shape)}")
            if (old_hw, old_hw) != new_grid:
                from timm.layers.pos_embed import resample_abs_pos_embed
                remapped[pe_key] = resample_abs_pos_embed(
                    pe, new_size=new_grid, old_size=(old_hw, old_hw), num_prefix_tokens=prefix)

        report = self.load_state_dict(remapped, strict=strict)
        bad_missing = [k for k in report.missing_keys if not k.startswith(self._OURS_ONLY_PREFIXES)]
        if require_complete_backbone and (bad_missing or report.unexpected_keys):
            raise RuntimeError(
                f"load_checkpoint: backbone không nạp đầy đủ | missing={bad_missing[:10]} "
                f"({len(bad_missing)}) unexpected={list(report.unexpected_keys)[:10]} "
                f"({len(report.unexpected_keys)})")
        return {
            "missing_keys": list(report.missing_keys)[:20],
            "unexpected_keys": list(report.unexpected_keys)[:20],
            "num_missing": len(report.missing_keys),
            "num_unexpected": len(report.unexpected_keys),
            "num_missing_backbone": len(bad_missing),
        }
