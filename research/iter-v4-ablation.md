# iter-v4-ablation — P2.F Screening & Confirmation Ablation (`R0` $\rightarrow$ `R5-l`)

## Changed
- **P2.A — Dual-View Loss (`code/networks/bfree_globalforge_vit.py`, `code/configs/bfree_dcs.yaml`, `code/train_lora.py`):**
  - Bổ sung nhánh `loss.mode: dual_view` trong [`BFreeGlobalForgeViT._compute_loss_dual_view`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L168-L199):
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(\text{clean}) + w_{\text{deg}} \cdot \mathcal{L}_{\text{CE}}(\text{deg}) + \lambda_{\text{cons}} \cdot T^2 \cdot \text{KL}\!\left(\sigma(\text{logits}_c^{\text{detach}}/T) \,\|\, \sigma(\text{logits}_d/T)\right) + \lambda_{\text{dcs}} \cdot \mathcal{L}_{\text{DCS}}$$
  - Giữ nguyên hợp đồng trả về 3 scalar `(total, ce_c, dcs)` và ghi `self.last_terms = {"ce_c", "ce_d", "cons"}` (detached) để log riêng vào `train_log.csv` (cả train và val).
  - Thêm `clean_view(img, normalize)` với xác suất `loss.p_aug_on_clean` (khuyến nghị `0.2` ở `dual_view`; bằng `0.0` ở `legacy` và không tiêu thụ RNG).
- **P2.B — Khởi tạo từ B-Free & Zero-Init Adapter (`code/networks/bfree_globalforge_vit.py`, `code/modules/gsr_adapter.py`, `code/modules/lib_adapter.py`):**
  - Thêm `--init_from {dinov2, bfree}` và `--bfree_ckpt` vào [`train_lora.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/train_lora.py#L140-L150).
  - Xử lý triệt để tiền tố `model.` / `module.` và ánh xạ head 1-logit $\rightarrow$ 2-logit trong [`remap_bfree_state_dict`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/networks/bfree_globalforge_vit.py#L204-L236): $W_2 = [-w/2;\, +w/2]$, $b_2 = [-b/2;\, +b/2]$ sao cho $z = l_1 - l_0 \equiv l_{\text{B-Free}}$. Bật `require_complete_backbone=True` để chặn lỗi bỏ qua trọng số im lặng.
  - Thêm `gate = nn.Parameter(torch.zeros(1))` cho [`GSRAdapter`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/modules/gsr_adapter.py#L33-L45) (`backbone.gsr_zero_init`) và `beta_init = -6.0` ($\alpha_0 = \sigma(-6) \approx 0.0025$) cho [`LIBAdapter`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/modules/lib_adapter.py#L31-L61) (`backbone.lib_beta_init`).
  - Đổi `backbone.head_mode: cls` làm mặc định trong `configs/bfree_dcs.yaml`, kết hợp độ lệch residual $(\text{feat\_cls} - \text{mean}(\text{patch\_raw}))$ vào CLS token để vừa thỏa Gate `Bước 0 ≡ B-Free` ($|\Delta\text{logits}| < 10^{-3}$) vừa truyền gradient từ CE về `LIB`/`GSR` trong các run đơn lẻ `R3` (`+LIB`) và `R4` (`+GSR`).
- **P2.C / P2.F — Cờ dữ liệu, CLI Override & Gói chạy Kaggle (`code/train_lora.py`, `code/configs/loader.py`, `code/scripts/run_p2f_ablation_kaggle.py`, `notebooks/version4/xla-ablation-p2f.ipynb`):**
  - Thêm cờ `--use_lib`, `--use_gsr`, `--use_dcs`, `--mix_preset {mix_current,mix_bfree,mix_noaigb}`, `--fmt_match`, `--res_match`, `--holdout_generators`, `--train_subsample_frac` vào `train_lora.py` và `RESUME_CHECK_KEYS`.
  - Tạo script chuẩn hóa [`code/scripts/run_p2f_ablation_kaggle.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/scripts/run_p2f_ablation_kaggle.py) và notebook [`notebooks/version4/xla-ablation-p2f.ipynb`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/notebooks/version4/xla-ablation-p2f.ipynb).
  - Đăng ký trước tiêu chí thành công/bác bỏ P2.G tại [`research/directions/D-recipe-v4/README.md`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/research/directions/D-recipe-v4/README.md).

---

## Verified
- **Unit tests trên CPU local (`pytest tests/`):**
  - [`tests/test_init_bfree.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/tests/test_init_bfree.py) & [`tests/test_dual_view.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/tests/test_dual_view.py): **19 passed, 1 skipped** (test với checkpoint B-Free thật được tích hợp tự động vào Bước 1 của `run_p2f_ablation_kaggle.py` trên GPU Kaggle).
  - Toàn bộ suite `tests/` (`test_backbone_shape.py`, `test_multi_severity.py`, `test_dcs_loss.py`, `overfit_test.py`, `resume_test.py`, `test_init_bfree.py`, `test_dual_view.py`): **31 passed, 1 skipped**.
  - Kiểm chứng bit-identity trong cùng môi trường: nhánh `loss.mode: legacy` + cờ mặc định khớp **bit-for-bit** (`torch.equal == True`) với code gốc trước khi sửa (`total=0.7343366742134094, ce=0.7191473841667175, dcs=1.5189313888549805`) và trùng khít với `dual_view` khi `w_deg=0.0, lam_cons=0.0`.
  - Notebook [`notebooks/version4/xla-ablation-p2f.ipynb`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/notebooks/version4/xla-ablation-p2f.ipynb) và [`code/scripts/run_p2f_ablation_kaggle.py`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/code/scripts/run_p2f_ablation_kaggle.py): đã kiểm tra `json.loads` và `ast.parse` không lỗi cú pháp.

---

## You must do

### Bước 0: Đồng bộ code lên Kaggle (`xla-setup`)
1. Commit & push nhánh hiện tại lên GitHub (hoặc cho phép tôi commit sau khi bạn duyệt).
2. Mở notebook `baopv051/xla-setup` trên Kaggle (chế độ Online), bấm **Save & Run All** để cập nhật thư mục `/kaggle/input/notebooks/baopv051/xla-setup/bfree_src` chứa `train_lora.py` mới và `code/scripts/run_p2f_ablation_kaggle.py`.

### Bước 1: Tạo Notebook Kaggle từ `notebooks/version4/xla-ablation-p2f.ipynb`
1. Upload file [`notebooks/version4/xla-ablation-p2f.ipynb`](file:///C:/Users/ADMIN/OneDrive%20-%20ptit.edu.vn/Documents/GitHub/B-Free/notebooks/version4/xla-ablation-p2f.ipynb) lên Kaggle, bật **GPU Accelerator** và **tắt Internet (Offline mode)**.
2. Attach đầy đủ các Input Datasets / Notebooks vào phiên làm việc (khớp đường dẫn tuyệt đối ở **Cell 2**):
   - Notebook `baopv051/xla-setup` (`/kaggle/input/notebooks/baopv051/xla-setup`)
   - Notebook `baopv051/xla-setup-eval` (`/kaggle/input/notebooks/baopv051/xla-setup-eval`)
   - Model `baopv051/bfree-baseline-weights` (`/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1`)
   - Dataset `baopv051/b-free-training-data` (`/kaggle/input/datasets/baopv051/b-free-training-data`)
   - Dataset `baopv051/ai-genbench-v1` (`/kaggle/input/datasets/baopv051/ai-genbench-v1`)
   - Dataset `ngovanphuong2604/wildrf-dataset` (`/kaggle/input/datasets/ngovanphuong2604/wildrf-dataset`)
   - Dataset `ngovanphuong2604/synthwildx-dataset` (`/kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset`)
   - Dataset `ngovanphuong2604/bfree-viral-images` (`/kaggle/input/datasets/ngovanphuong2604/bfree-viral-images`)

### Bước 2: Chạy các Run Sàng lọc (`R0`, `R1`, `R1b`, `R2`, `R3`, `R4`, `R5` và `R5-d..R5-l`)
Bạn có thể chạy theo **Cách A** (sửa biến `RUNS_TO_EXECUTE` ở **Cell 1** của notebook rồi bấm **Run All**) hoặc **Cách B** (chạy từng dòng lệnh CLI tường minh cho từng run trong một code cell trên Kaggle):

#### Cách A — Qua `run_p2f_ablation_kaggle.py` (Tự động Gate Bước 0 + Train + Eval Yardstick v2 + Xuất Parquet + In bảng Paste-back)
- **Phiên 1 (Tham chiếu `R0` + Recipe thuần `R1` + Đối chứng W1 `R1b`, ~1.5h):**
  Tại **Cell 1**, đặt:
  ```python
  RUNS_TO_EXECUTE = "R0,R1,R1b"
  STAGE = "screening"
  ```
- **Phiên 2 (Tách đóng góp từng module `R2`, `R3`, `R4`, `R5`, ~2.5h):**
  Tại **Cell 1**, đặt:
  ```python
  RUNS_TO_EXECUTE = "R2,R3,R4,R5"
  STAGE = "screening"
  ```
- **Phiên 3 (Các biến thể đối chứng `R5-d`, `R5-a`, `R5-f`, `R5-l`, ~2.5h):**
  Tại **Cell 1**, đặt:
  ```python
  RUNS_TO_EXECUTE = "R5-d,R5-a,R5-f,R5-l"
  STAGE = "screening"
  ```

#### Cách B — Lệnh CLI trực tiếp từng Run (`R1`, `R1b`, `R2`, `R3`, `R4`, `R5`)
Nếu bạn muốn chạy từng lệnh độc lập trong cell bash của Kaggle (sau khi đã chạy **Cell 2** và **Cell 3**):

- **Kiểm tra Gate bắt buộc "Bước 0 ≡ B-Free" trên trọng số thật:**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/tests/test_init_bfree.py" \
      --ckpt "/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1/model_epoch_best.pth" \
      --n 64 --img 504
  ```
- **Run `R1` (Init B-Free, KHÔNG module, Dual-View CE, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R1 --stage screening
  ```
- **Run `R1b` (Init B-Free, KHÔNG module, Legacy Clean CE, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R1b --stage screening
  ```
- **Run `R2` (Init B-Free, `+DCS`, Dual-View, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R2 --stage screening
  ```
- **Run `R3` (Init B-Free, `+LIB` với `beta_init=-6`, Dual-View, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R3 --stage screening
  ```
- **Run `R4` (Init B-Free, `+GSR` với `gsr_zero_init=1`, Dual-View, `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R4 --stage screening
  ```
- **Run `R5` (Mô hình đầy đủ: Init B-Free + `LIB+GSR+DCS` + Dual-View + `mix_bfree`, 25% data, 3 epochs):**
  ```bash
  python "/kaggle/input/notebooks/baopv051/xla-setup/bfree_src/code/scripts/run_p2f_ablation_kaggle.py" --runs R5 --stage screening
  ```

---

## Output schema

Mỗi run `<RUN_ID>` lưu kết quả tại `/kaggle/working/p2f_runs/<RUN_ID>/` với 2 file dữ liệu chuẩn hóa:

### 1. File `/kaggle/working/p2f_runs/<RUN_ID>/train_log.csv`
- **Chế độ `loss.mode = dual_view` (`R1, R2, R3, R4, R5, R5-d, R5-a, R5-f, R5-l`):**
  ```text
  epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc,ce_c,ce_d,cons,val_ce_c,val_ce_d,val_cons
  ```
  - `epoch` (`int`): Chỉ số epoch (`1..3` ở screening, `1..5` ở confirm).
  - `train_loss`, `val_loss` (`float`): Tổng hàm mục tiêu $\mathcal{L}_{\text{total}}$.
  - `train_ce`, `ce_c`, `val_ce_c` (`float`): Cross-Entropy trên view sạch $\mathcal{L}_{\text{CE}}(\text{clean})$.
  - `ce_d`, `val_ce_d` (`float`): Cross-Entropy trên view suy giảm $\mathcal{L}_{\text{CE}}(\text{deg})$.
  - `cons`, `val_cons` (`float`): KL-divergence nhất quán quyết định $T^2 \cdot \text{KL}(\text{clean}^{\text{detach}} \parallel \text{deg})$.
  - `train_dcs` (`float`): Giá trị $\mathcal{L}_{\text{DCS}}$ (InfoNCE đối xứng).
  - `val_auc`, `val_bacc` (`float`): ROC-AUC và Balanced Accuracy tại ngưỡng logit $0$ trên tập val in-distribution.
- **Chế độ `loss.mode = legacy` (`R1b`):**
  ```text
  epoch,train_loss,train_ce,train_dcs,val_loss,val_auc,val_bacc
  ```

### 2. File `/kaggle/working/p2f_runs/<RUN_ID>/scores.parquet` (Theo Quy tắc #8, P0.9, P0.10)
| Tên cột | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `image_id` | `string` | Đường dẫn tương đối của ảnh hoặc định danh hàng Arrow (`aigb_val_<idx>`) |
| `benchmark` | `string` | Tên bộ đánh giá: `wildrf_val`, `wildrf`, `bfree_viral_images`, `ai_genbench`, `realdeg_heldout` |
| `group` | `string` | Tên generator hoặc nền tảng (`facebook`, `reddit`, `twitter`, `viral`, `REAL_POOL`, ...) |
| `is_seen_gen` | `bool` | `True` nếu thuộc 13 `PRIORITY_GENERATORS` đã thấy khi train; `False` nếu unseen/real |
| `origin_dataset` | `string` | Nguồn dữ liệu gốc của mẫu (phục vụ kiểm tra rò rỉ miền) |
| `condition` | `string` | `clean` hoặc tên toán tử suy giảm held-out (`heldout_webp`, `heldout_jpeg_420`, `heldout_resize_lanczos`, `heldout_poisson_iso`, `heldout_sharpen`, `heldout_social_compound`) |
| `label` | `int64` | Nhãn thực tế (`0` = Real, `1` = Fake) |
| `logit` | `float32` | Logit nhị phân tự nhiên $z = l_1 - l_0$ (hoặc $l$ với B-Free 1-logit) |
| `prob` | `float32` | Xác suất dự đoán trước hiệu chuẩn $p = \sigma(z)$ |
| `prob_cal` | `float32` | Xác suất sau khi chia nhiệt độ Temperature Scaling $p_{\text{cal}} = \sigma(z / T_{\text{cal}})$ |
| `width` | `int64` | Chiều rộng gốc của ảnh (px) |
| `height` | `int64` | Chiều cao gốc của ảnh (px) |
| `format` | `string` | Định dạng file ảnh gốc (`JPEG`, `PNG`, `WEBP`, ...) |
| `run_id` | `string` | Mã định danh run (`R0`, `R1`, `R1b`, `R2`, `R3`, `R4`, `R5`, ...) |
| `git_sha` | `string` | Commit SHA của repo `bfree_src` |
| `ckpt_sha256` | `string` | 16 ký tự đầu SHA-256 của file checkpoint đã chấm |
| `config_hash` | `string` | 16 ký tự đầu SHA-256 của cấu hình thực nghiệm |
| `seed` | `int64` | Seed thực nghiệm (`42`) |
| `harness_version` | `int64` | Phiên bản thước đo (`2` — Yardstick v2) |

---

## Paste back

Sau khi chạy xong trên Kaggle, hãy dán:
1. Dòng log `[GATE P2.B] Step 0 vs B-Free official: max |Delta logits| = ...`
2. Nội dung đầu ra của **Cell 5** (hoặc các dòng `PASTE BACK ROW`) vào bảng dưới đây:

| Run | WildRF+Viral AUC | WildRF+Viral bAcc@0.5 | WildRF+Viral ECE_cal | AIGB-Unseen AUC | AIGB-Unseen bAcc@0.5 | AIGB-Unseen ECE_cal | RealDeg-Heldout bAcc@0.5 | AIGB-Seen AUC (Guard) | Temp T | ckpt_sha256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `R0` (B-Free ref) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R1` (Pure recipe) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R1b` (Clean CE) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R2` (+DCS) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R3` (+LIB) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R4` (+GSR) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5` (LIB+GSR+DCS) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-d` (DINOv2 init) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-a` (mix_current) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-f` (fmt+res match) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
| `R5-l` (ls=0) | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run | TODO: pending run |
