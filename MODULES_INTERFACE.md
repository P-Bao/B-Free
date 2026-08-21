# Modules Interface — B-Free GlobalForge Integration (branch integration/loss-backbone)

Loss and backbone module contract for the team. This branch provides the backbone
with post-hoc LIB/GSR and the L_DCS loss; the real LIB/GSR adapters land on the
`integration/lib-gsr` branch and the training/eval scripts on `integration/train-eval`.

## Model

```python
from networks.bfree_globalforge_vit import BFreeGlobalForgeViT
from configs.loader import load_config, build_model_kwargs

model = BFreeGlobalForgeViT(
    arch="vit_base_patch14_reg4_dinov2.lvd142m",
    num_classes=2,
    img_size=504,
    pretrained=True,          # True = tải dinov2 từ timm; False = random init
    use_lib=True,
    use_gsr=True,
    use_dcs=True,
    lib_kernel=3, lib_tau=0.5,
    gsr_window=3, gsr_mask_prob=1.0,
    dcs_tau=0.07,
    lambda_dcs=0.1,
    label_smoothing=0.1,
)
```

### forward(images) → dict
```python
out = model(images)          # images: (B, 3, 504, 504) hoặc (B, 3, 224, 224) nếu img_size=224
out["logits"]                # (B, 2)
out["cls"]                   # (B, 768) — mean-pool patch tokens (giống GlobalForge REM.py:279)
```

### compute_loss(images_clean, images_degraded, labels) → (total, ce, dcs)
```python
total, ce, dcs = model.compute_loss(images_clean, images_degraded, labels)
# total = ce + lambda_dcs * dcs
# ce  = CrossEntropy(logits_clean, labels) với label_smoothing=0.1
# dcs = info_nce_loss(cls_clean, cls_degraded, tau=0.07)  # symmetric InfoNCE
# Trả về 3 giá trị tách biệt để log riêng (Phase 8).
```

### load_checkpoint(path, strict=False)
```python
rep = model.load_checkpoint("bfree_baseline.pth", strict=False)
# Remap 'patch_embed.X' -> 'model.patch_embed.X' (định dạng Wrapper5crops).
# LIB/GSR/dcs/head mới sẽ missing (OK vì strict=False).
```

## Kiến trúc (quyết định đã chốt, khác plan gốc)

- **LIB/GSR áp dụng POST-HOC** trên patch tokens cuối (sau 12 block + norm), KHÔNG chèn giữa block 6/12.
  Lý do: code thật GlobalForge (`REM.py:251-282`) làm post-hoc, không chèn giữa block.
- Token order: `[cls, reg×4, patch×N]` — `num_prefix=5`, patch tokens = `x[:, 5:, :]`.
- 504×504: N=1296, hw=36. 224×224: N=256, hw=16. Dùng `model.set_input_size()`.
- Feature cho L_DCS = **mean-pool patch tokens** (không dùng cls token), giống `REM.py:279`.

## L_DCS (modules/dcs_loss.py)

```python
from modules.dcs_loss import info_nce_loss, DCSLoss
loss = info_nce_loss(z_clean, z_degraded, tau=0.07)  # (B, C) -> scalar
# = 0.5 * (CE(z @ z_aug.T / tau, arange) + CE(z_aug @ z.T / tau, arange))
# L2 normalize bên trong.
```
Không phụ thuộc dữ liệu RealDeg-Bench/VAE — chỉ cần cặp feature (clean, degraded).

## View suy giảm (cho L_DCS)

Pipeline GlobalForge (`datasets_real.py:378-383`), KHÔNG phải inpainted++ (B-Free không có code):
- RandomJPEG(quality 20-80, p=1.0)
- RandomGaussianBlur(kernel=7, sigma 0.5-1.5, p=0.8)
- ColorDistortion(level=3, p=0.8) = ColorJitter(0.4, 0.4, 0.4, 0.06)
- Bản triển khai: `tests/degradation.py` (`DegradationPipeline`) — chỉ local, không commit.

## Stub LIB/GSR (temporary, replaced by the real adapters on the lib-gsr branch)

```python
# modules/lib_adapter_stub.py — remove after merging modules/lib_adapter.py
class LIBAdapter(nn.Module):
    def __init__(self, dim=768, kernel_size=3, tau=0.5): ...
    def forward(self, feat_tokens, hw=None): return feat_tokens, None  # Identity

# modules/gsr_adapter_stub.py — remove after merging modules/gsr_adapter.py
class GSRAdapter(nn.Module):
    def __init__(self, dim=768, window=3, mask_prob=1.0): ...
    def forward(self, tokens, hw=None): return tokens  # Identity
```

At merge: swap the `*_stub` imports in `networks/bfree_globalforge_vit.py` for the real
adapters. **Re-run the overfit gate** afterwards — the stubs are Identity, so LIB/GSR
have not yet been exercised against the real backbone.

## Config (configs/bfree_dcs.yaml)

```yaml
backbone:
  arch: vit_base_patch14_reg4_dinov2.lvd142m
  num_classes: 2
  img_size: 504
  pretrained: false
  use_lib: true, use_gsr: true
  lib_kernel: 3, lib_tau: 0.5
  gsr_window: 3, gsr_mask_prob: 1.0
loss:
  use_dcs: true
  lambda_dcs: 0.1     # khởi điểm, cần sweep
  dcs_tau: 0.07
  label_smoothing: 0.1
degradation:
  jpeg_quality_min: 20, jpeg_quality_max: 80
  blur_kernel: 7, blur_sigma_min: 0.5, blur_sigma_max: 1.5, blur_prob: 0.8
  color_level: 3, color_prob: 0.8
```

## Gate Phase 4 (overfit test) — ĐÃ PASS

- 4 ảnh, 224×224, 20 bước, lr=5e-5, pretrained=True.
- total: 0.816 → 0.214 (ratio 0.262), CE: 0.683 → 0.203, DCS: 1.327 → 0.111.
- Kiến trúc (backbone + LIB/GSR stub + L_DCS + degradation) không có silent bug.

## Hyperparameter khởi điểm (cần sweep ở Phase 5+)

- `lambda_dcs`: 0.1 (khởi điểm, GlobalForge released dùng 0.01)
- `dcs_tau`: 0.07 (GlobalForge default)
- `label_smoothing`: 0.1 (GlobalForge convention)
- `lr`: 5e-5 (overfit test; training thật có thể khác)
