# Yardstick v2 — Đặc tả giao thức đánh giá công bằng (`harness_version = 2`)

> **Trạng thái:** **APPROVED** bởi User (2026-10-22) cho toàn bộ mục 🔒 `P0.1–P0.8`, cùng quyết định chốt Mục 10 của `AGENT_PLAN_V4.md`.
> **Nguyên tắc:** Không chạm vào bất kỳ file code train nào (`code/train_lora.py`, `code/networks/*`, `code/modules/*`, `code/configs/*`) cho đến khi hoàn tất P0 và P1.

---

## 0. Quyết định đã chốt từ User (Mục 10 `AGENT_PLAN_V4.md`)

1. **Duyệt toàn bộ `P0.1–P0.8`** về giao thức đánh giá mới (`harness_version = 2`).
2. **Provenance `ours-v2`:** lấy từ checkpoint **phase-2 stage-2** (slug `/kaggle/input/notebooks/baopv051/xla-trainng-stage-2/bfree_globalforge_lora_r16_best.pth` hoặc thư mục checkpoint tương ứng).
3. **Hoãn hoàn toàn MS-DCS (`v3`, K=4)** cho đến khi hoàn thành xong P2 (`multi_severity.enabled=false` giữ nguyên).
4. **Chốt cứng ngưỡng giả thuyết `H-A` đến `H-E` và `Guardrails`** đúng theo bảng ở Mục 5 (`P2.G` trong `AGENT_PLAN_V4.md`).
5. **Phân vai môi trường:** Local của Agent chỉ chạy CPU (`xla-cpu`). Mọi lệnh train/eval nặng đóng gói thành **Kaggle Notebook / script offline** để User chạy thủ công.
6. **Weights B-Free official:** đã có sẵn trong Kaggle input (`/kaggle/input/models/baopv051/bfree-baseline-weights/pytorch/default/1`, kèm `code/weights/BFREE_dino2reg4/config.yaml` + `model_epoch_best.pth`).

---

## 1. Ánh xạ tổng thể `U1–U7` (`research/eval-diff.md`) ↔ `P0.1–P0.12`

| Cờ bất công (`eval-diff.md`) | Bản chất sai lệch trong Harness v1 | Mục P0 giải quyết | Giải pháp kỹ thuật trong Yardstick v2 |
|---|---|---|---|
| **U1** — `ours` train trên AI-GenBench `train` (13 generator + 46.8k real), chấm trên `validation`; baseline zero-shot | Gộp chung 36 generator làm lệch trung bình có lợi cho `ours` trên 13 generator đã thấy | **P0.7, P0.8** | Tách bạch 2 nhóm `AIGB-seen` (13 `PRIORITY_GENERATORS`) và `AIGB-unseen` (23 generator còn lại) + có tuỳ chọn loại `origin_dataset` trùng; `seen` chỉ dùng làm Guardrail, **cấm** dùng để tuyên bố vượt baseline |
| **U2** — `ours` dùng `multi_crop=True` (tối đa 9 crop 504×504 phủ toàn ảnh) trong khi `bfree_official` & `globalforge` dùng `multi_crop=False` (1 center crop 504×504) | (1) `ours` được lợi thế ensemble 9 góc nhìn; (2) `bfree_official` nhận đầu vào đã bị cắt 504×504 nên `Wrapper5crops` bên trong tạo ra **5 crop giống hệt nhau** thay vì 5-crop trên embedding toàn ảnh như paper gốc | **P0.1** | Báo cáo song song 2 giao thức: **(i) `official-per-model`** (mỗi mô hình chạy đúng tiền xử lý/crop của tác giả gốc) và **(ii) `unified`** (mọi mô hình dùng chung chính sách crop: `unified_1crop` center 504 và/hoặc `unified_5crop`) |
| **U3** — `globalforge` chạy off-protocol (crop 504, không resize-to-1296 + JPEG q100, chạy `bf16` thay vì `fp32`) | Điểm dự đoán real bị kéo về `mean_p_fake = 0.4965`, `bAcc@0.5` kẹt ở `76.10%` | **P0.2** | Trong giao thức `official-per-model`: khôi phục đúng pipeline GlobalForge (`short256` / `resize-to-1296`, JPEG `q=100` re-encode nếu theo repo gốc, crop `224×224`, chạy `fp32` không ép `bfloat16`). Trong bảng `bAcc@0.5` của `unified`: ghi chú rõ nếu off-protocol hoặc chỉ so sánh `AUC` / `bAcc@calib` |
| **U4** — Bicubic pre-upscale cạnh ngắn `< 504` lên `504` trong `ModelWrapper.score` | Khớp với `ensure_min_size` khi train `ours`, nhưng phá cơ chế `replicate_wrap` trên embedding của `B-Free` gốc | **P0.1** | Ở `official-per-model`, `bfree_official` đưa nguyên tensor ảnh gốc (đã normalize) vào `Wrapper5crops` để tự `replicate_wrap` ở tầng patch embedding (`grid_size = 36×36`), **bỏ** bước `img.resize(..., Image.BICUBIC)` khi `min(w, h) < 504` |
| **U5** — `CALIB_THRESHOLD` (Youden's $J$) được fit trực tiếp trên toàn bộ điểm test `ai_genbench` (`n=6900`) rồi chấm lại chính nó và đem sang WildRF/viral | Rò rỉ tập test vào việc chọn ngưỡng (optimistic bias cho mọi mô hình) | **P0.7** | Fit `CALIB_THRESHOLD` và nhiệt độ $T$ (Temperature Scaling) trên **split calibration độc lập** (phần giữ riêng từ AIGB hoặc `wildrf/val`), đóng băng tham số rồi chỉ chấm **1 lần** trên các tập `test`. Giữ cột `bacc_calib_legacy` nếu cần đối chiếu lịch sử |
| **U6** — Label smoothing `0.1` khi train `ours` đặt sàn xác suất $[0.05, 0.95]$ ⇒ sàn ECE $\approx 0.05$; hàm `binary_ece` trong harness không cân bằng lớp | Khi `n_real = 1500` và `n_fake = 150` (tỷ lệ 10:1 mỗi generator AIGB), `binary_ece` bị thống trị 91% bởi lớp real | **P0.6, P0.7, P0.12** | Thay bằng `balanced_ece_binary` và `balanced_nll_binary` từ `code/utils/dmetrics.py` (trọng số $1/N_c$ cân bằng hai lớp); kèm báo cáo ECE sau Temperature Scaling trên tập calibration; sửa lỗi `sigmoid(2l)` của B-Free (P0.12) |
| **U7** — Tập toán tử suy giảm lúc train `v2` (`DEGRADE_STRENGTHS` 7 op) trùng khít với bộ toán tử chấm `RealDeg-Bench-style` và `realdeg_bench_style=False` | Chưa có số liệu RealDeg; nếu bật nguyên bản cũ thì chỉ đo *in-distribution robustness*, không đo được tổng quát hóa sang suy giảm chưa thấy | **P0.3** | Bật `realdeg_bench_style = True`, seed cố định theo từng ảnh (`hash(key, seed)`), tách rõ 2 nhóm: **(a) `RealDeg-indist`** (jpeg, blur, resize bicubic, gaussian noise) và **(b) `RealDeg-heldout`** (WebP, JPEG chroma-subsampling 4:2:0, Lanczos/bilinear resize, Poisson/ISO noise, sharpen, chuỗi mạng xã hội `resize→JPEG→resize→JPEG`). Loại bỏ `brightness/contrast/saturation` khỏi tuyên bố độ bền chính |
| *(Bổ sung V4/V5)* — Mẫu nhỏ (150 fake/gen), không CI, lỗi SynthWildX, mất điểm từng ảnh, sai công thức xác suất B-Free | Không kiểm định được ý nghĩa thống kê; mất dữ liệu cho chẩn đoán P1 | **P0.4, P0.5, P0.9, P0.10, P0.12** | Sửa loader SynthWildX (P0.4); nâng $\ge 500$ fake/gen + Bootstrap 1,000 lần CI 95% + DeLong/McNemar (P0.5); lưu `scores_{model}_{protocol}.parquet` (P0.9); thêm cột provenance (P0.10); sửa lỗi `cat([-l, l])` $\Rightarrow \sigma(2l)$ (P0.12) |

---

## 2. Chi tiết kỹ thuật từng hạng mục `P0.1–P0.12` và vị trí sửa trong Harness

> **Đối chiếu file thực tế đã mở:**
> - `C:\Users\ADMIN\Downloads\xla-eval-harness-ours (1).ipynb` (14 cells, chạy ngày `2026-10-03`, `OURS_CKPT` trỏ tới `xla-trainng-stage-2/bfree_globalforge_lora_r16_best.pth`)
> - `C:\Users\ADMIN\Downloads\xla-eval-harness-b-free.ipynb` (14 cells, chạy ngày `2026-09-23`)
> - `C:\Users\ADMIN\Downloads\xla-eval-harness-globalforge-dinov3.ipynb` (15 cells — có thêm cell 6 `!pip uninstall -y torchao`)

### P0.1 — Chính sách Crop & Tiền xử lý (`official-per-model` vs `unified`) [Map: U2, U4]
- **Hiện trạng (Cell 6 trong `ours`/`bfree`, Cell 7 trong `globalforge` — class `ModelWrapper`):**
  - Mọi ảnh có `min(w, h) < 504` đều bị ép `img.resize(..., Image.BICUBIC)` lên cạnh ngắn `504`.
  - `_crops(img)`: với `multi_crop=True` (`ours`) sinh lưới tối đa $3 \times 3 = 9$ crop `504×504`; với `multi_crop=False` (`bfree_official`, `globalforge`) chỉ cắt **1 center crop `504×504`**.
  - Hệ quả nghiêm trọng cho `bfree_official`: vì `ModelWrapper` đã cắt ảnh về đúng `504×504` trước khi gọi `self.forward_fn`, bên trong `Wrapper5crops.forward(x)` (`code/networks/wrapper5crops.py:38-51`), `embeddings` có kích thước đúng bằng `patch_size = (36, 36)`, dẫn đến `hs = 0, ws = 0` và 5 crop (`center, TL, BL, BR, TR`) **trùng khít nhau 100%** (5 bản sao của cùng 1 center crop), đồng thời `replicate_wrap` không bao giờ được kích hoạt!
- **Thiết kế Yardstick v2:**
  - Tham số hóa `PROTOCOL_MODE ∈ {"official", "unified_1crop", "unified_5crop"}`:
    1. **`official` (`official-per-model`):**
       - `bfree_official`: **Không** pre-upscale bicubic, **không** cắt 504×504 ở `ModelWrapper`. Đưa trực tiếp tensor `self.normalize(img).unsqueeze(0)` của toàn ảnh vào `Wrapper5crops` (xử lý từng ảnh vì kích thước khác nhau) để `Wrapper5crops` tự chạy `patch_embed.proj(x)`, `replicate_wrap` nếu `< 504`, và lấy trung bình 5 crop thực sự trên lưới patch embedding.
       - `globalforge_dinov2` / `globalforge_dinov3`: Chạy đúng chuẩn chính thức (xem P0.2).
       - `ours`: Chạy chuẩn 5-crop (hoặc 1-crop / 9-crop được gắn nhãn rõ ràng trong cột `protocol`).
    2. **`unified` (`unified_1crop` / `unified_5crop`):**
       - Mọi mô hình dùng chung một quy tắc tiền xử lý & số lượng crop để cô lập đóng góp của kiến trúc/trọng số khỏi TTA.

### P0.2 — Khôi phục giao thức gốc cho GlobalForge [Map: U3]
- **Hiện trạng (Cell 6/7 `_load_globalforge` & `ModelWrapper.score`):**
  - Đang dùng `IMG_SIZE = 504`, center crop `504`, không nén JPEG `q=100`, và chạy dưới `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.
- **Thiết kế Yardstick v2:**
  - Trong chế độ `official`:
    - Áp dụng pipeline tiền xử lý chuẩn của GlobalForge (resize cạnh ngắn / đưa về kích thước chuẩn repo gốc, JPEG `q=100` round-trip qua `io.BytesIO`, center crop `224×224` hoặc theo cấu hình `REM`).
    - Tắt `bfloat16` autocast (`dtype=torch.float32`) khi chạy `globalforge_*`.
  - Trong bảng tổng hợp: ghi rõ `protocol = official_fp32` vs `unified`; không dùng `bAcc@0.5` off-protocol để kết luận.

### P0.3 — Bật `RealDeg` có hạt giống theo ảnh & tách `in-distribution` vs `held-out` [Map: U7]
- **Hiện trạng (Cell 2 `BENCHMARKS_ENABLED` & Cell 9 `DEGRADE_STRENGTHS` / Cell 12 `run_realdeg`):**
  - `BENCHMARKS_ENABLED["realdeg_bench_style"] = False`.
  - `apply_degradation` dùng `np.random.normal` (global RNG, phụ thuộc thứ tự gọi) cho `noise`, và `run_realdeg` chọn `rng.choice(DEGRADE_STRENGTHS[cond_val])` bằng 1 `rng` tuần tự.
  - 7 toán tử hiện tại (`jpeg, blur, resize, noise, brightness, contrast, saturation`) trùng hoàn toàn với `train_lora.py:182-190`.
- **Thiết kế Yardstick v2:**
  - Bật `BENCHMARKS_ENABLED["realdeg_bench_style"] = True`.
  - **Deterministic per-image seed:** Với mỗi ảnh `key` và điều kiện `cond_name`, khởi tạo `rng_img = np.random.default_rng( zlib.crc32(f"{key}:{cond_name}:{AIGENBENCH_SEED}".encode()) )` để mọi mô hình nhận **cùng một ảnh suy giảm bit-identical**.
  - **Phân tách 2 tập toán tử:**
    - `RealDeg-indist` (4 toán tử tần số/không gian có trong train v2): `jpeg` ($Q \in \{90,80,70,60,40\}$), `blur` ($\sigma \in \{0.5,1,2,3,5\}$), `resize_bicubic` ($s \in \{0.9,0.7,0.5,0.3,0.2\}$), `gaussian_noise` ($v \in \{0.0005,0.001,0.002,0.005,0.01\}$) + `compound_indist` (1–5 bước).
    - `RealDeg-heldout` (**6 toán tử chưa từng xuất hiện khi train**):
      1. `webp`: nén WebP $Q \in \{85, 70, 50, 30\}$.
      2. `jpeg_subsampling`: nén JPEG với `subsampling="4:2:0"` vs `"4:4:4"` ở $Q \in \{85, 65, 45\}$.
      3. `resize_lanczos_bilinear`: thu phóng bằng `Image.LANCZOS` hoặc `Image.BILINEAR` ($s \in \{0.75, 0.5, 0.35\}$).
      4. `poisson_iso_noise`: nhiễu Poisson (shot noise) + Gaussian read noise mô phỏng cảm biến ($\lambda \in \{100, 50, 25\}$).
      5. `sharpen`: `ImageFilter.UnsharpMask(radius=2, percent=p, threshold=3)` với $p \in \{120, 200, 300\}$.
      6. `social_pipeline`: chuỗi mô phỏng mạng xã hội `resize(0.75, BILINEAR) -> JPEG(q=80, 4:2:0) -> resize(0.85, BICUBIC) -> JPEG(q=70, 4:2:0)` và biến thể `compound_heldout` (3–5 bước).
  - **Loại bỏ** `brightness`, `contrast`, `saturation` khỏi chỉ số trung bình độ bền chính (chỉ giữ riêng nếu cần kiểm tra phụ).

### P0.4 — Sửa lỗi Loader `SynthWildX` [Map: U-SynthWildX]
- **Hiện trạng (Cell 11 `build_synthwildx_items` & log lỗi Cell 13):**
  - Lỗi thực tế: `ERROR | [synthwildx] BO QUA: Directory /kaggle/input/datasets/ngovanphuong2604/synthwildx-dataset is neither a Dataset directory nor a DatasetDict directory.`
  - Nguyên nhân: `load_from_disk(str(root))` được gọi trực tiếp trên thư mục gốc của Kaggle dataset, trong khi file `dataset_dict.json` hoặc `state.json` (chuẩn HuggingFace Arrow) hoặc các file `.parquet`/`.arrow`/thư mục ảnh nằm ở thư mục con bên dưới `SYNTHWILDX_DIR`.
- **Thiết kế Yardstick v2 (Cell 11 `build_synthwildx_items`):**
  - Viết hàm dò tìm đệ quy `_resolve_hf_or_image_dataset(root)`:
    1. Tìm thư mục con đầu tiên chứa `dataset_dict.json` hoặc `state.json` $\rightarrow$ gọi `load_from_disk(str(hit))`.
    2. Nếu không có `state.json`, tìm các file `*.parquet` hoặc `*.arrow` $\rightarrow$ nạp bằng `datasets.Dataset.from_parquet(...)` / `from_file(...)`.
    3. Nếu lưu dạng thư mục ảnh rời (`0_real`/`1_fake` hoặc `real`/`fake` hoặc kèm metadata CSV/JSON) $\rightarrow$ fallback sang `list_images` như `wildrf` / `bfree_viral_images`.
  - Đảm bảo có đủ số liệu `synthwildx` cho cả 3 mô hình (`ours`, `bfree_official`, `globalforge`).

### P0.5 — Tăng cỡ mẫu & Kiểm định thống kê (Bootstrap CI 95%, DeLong, McNemar) [Map: W7]
- **Hiện trạng (Cell 3 `CONFIG-PATHS` & Cell 7 `compute_all_metrics`):**
  - `AIGENBENCH_MAX_PER_GEN = 150` (quá nhỏ, sai số chuẩn lớn), không có khoảng tin cậy (CI), không có kiểm định ghép cặp.
- **Thiết kế Yardstick v2:**
  - Tăng `AIGENBENCH_MAX_PER_GEN = 500` (hoặc `None` = toàn bộ fake mỗi generator, tối đa $1,000$/generator trên `validation` $n=72,000$; `AIGENBENCH_MAX_REAL = 3000`).
  - **Bootstrap CI 95% ($B = 1,000$ lần, stratified theo lớp real/fake với seed cố định):** xuất thêm các cột `auc_ci_lo`, `auc_ci_hi`, `bacc_ci_lo`, `bacc_ci_hi` cho từng benchmark/nhóm.
  - **Kiểm định ghép cặp trên điểm từng ảnh (`scores_*.parquet`):**
    - **Kiểm định DeLong** (hoặc paired bootstrap trên hiệu $\Delta\text{AUC}$) cho AUC.
    - **Kiểm định McNemar** cho quyết định phân loại nhị phân (`bAcc@0.5` và `bAcc@calib`).

### P0.6 — Bộ Metric đầy đủ & Chuẩn hóa theo `dmetrics.py` [Map: U6, W6]
- **Hiện trạng (Cell 7 `binary_ece`, `balanced_nll`, `compute_all_metrics`):**
  - Tự viết lại `binary_ece(y_true, y_prob)` không cân bằng lớp; thiếu `Pd@FAR=10%`, `EER`, `FPR@TPR95`, và không tách `TPR`/`FPR` tại ngưỡng `0.5` và `calib`.
- **Thiết kế Yardstick v2:**
  - Gọi trực tiếp (hoặc nhúng nguyên bản) các hàm chuẩn từ `code/utils/dmetrics.py`:
    - `balanced_ece_binary(y_true, y_logit, bins=15)` (dùng trọng số $1/N_c$ cân bằng real/fake).
    - `balanced_nll_binary(y_true, y_logit)` (tính từ `balanced_cllr * ln(2)` trên logit tự nhiên).
    - `pd_at_far(y_true, y_score, 0.10)` ($\text{Pd}@10\%\text{FAR}$).
    - `calculate_eer2(y_true, y_score)` (Equal Error Rate nội suy).
  - Bổ sung thêm:
    - `fpr_at_tpr95`: FPR tại điểm $\text{TPR} = 0.95$.
    - **Tách riêng `tpr_05`, `fpr_05` (tại ngưỡng $p=0.5$ / logit $z=0$) và `tpr_calib`, `fpr_calib` (tại ngưỡng hiệu chuẩn)** kèm `mean_p_real`, `mean_p_fake` — phục vụ trực tiếp chẩn đoán **D3** ở P1 (phân biệt "real bị báo fake" vs "fake bị bỏ sót").

### P0.7 — Hiệu chuẩn ngưỡng & Temperature Scaling trên tập tách riêng [Map: U5, U6]
- **Hiện trạng (Cell 12 `run_ai_genbench`):**
  - Đang gộp toàn bộ điểm test `ai_genbench` (`all_true, all_prob`, $n=6,900$) để chạy `find_optimal_threshold(..., method="youden")` rồi chấm lại chính `ai_genbench` đó và dùng cho `wildrf`/`bfree_viral`.
- **Thiết kế Yardstick v2:**
  - **Tách tập Calibration độc lập (`calib_split`), tuyệt đối không đụng vào tập `test`:**
    1. **Calibration miền chuẩn (`calib_aigb`):** Tách cố định 1 tập con độc lập không trùng với mẫu test (ví dụ: lấy từ `ai_genbench` split `train` phần held-out hoặc tách 20% index cố định theo seed `CALIB_SEED = 2026` trước khi lấy mẫu `test_items`, hoàn toàn tách rời khỏi `test_items`).
    2. **Calibration miền mạng xã hội (`calib_wildrf_val`):** Sử dụng trực tiếp thư mục `WILDRF_DIR / "val"` (WildRF có sẵn split `val` độc lập với `test`).
  - Trên tập calibration, fit **1 lần duy nhất**:
    - Ngưỡng Youden's $J$: $z^* = \arg\max_z (\text{TPR}(z) - \text{FPR}(z))$.
    - Nhiệt độ **Temperature Scaling** $T > 0$: tối thiểu hóa `balanced_nll_binary(y_cal, z_cal / T)` bằng `scipy.optimize.minimize_scalar`.
  - Đánh giá trên tập test chỉ **1 lần** với $(z^*, T)$ đã khóa. Giữ cột `bacc_calib_legacy` (tính kiểu cũ) chỉ để đối chiếu lịch sử với v1.

### P0.8 — Tách bạch nhóm `Seen` vs `Unseen` trên AI-GenBench [Map: U1]
- **Hiện trạng (Cell 12 `run_ai_genbench`):**
  - Chỉ xuất 1 dòng tổng hợp `generator = "ALL"` gộp chung cả 36 generator.
- **Thiết kế Yardstick v2:**
  - Định nghĩa hằng số `SEEN_GENERATORS` trong harness khớp chính xác với `PRIORITY_GENERATORS` trong `code/train_lora.py:52-57` (13 generator):
    ```python
    SEEN_GENERATORS = {
        "ADM", "DDPM", "DeepFloyd IF", "Palette", "Denoising Diffusion GAN",
        "LaMa", "MAT", "SN-PatchGAN", "StyleGAN1", "VQGAN", "GANformer",
        "CIPS", "FaceSynthetics",
    }
    ```
  - Xuất 3 dòng tổng hợp riêng biệt cho `ai_genbench`:
    1. `ALL_SEEN` (13 generator đã thấy khi train `ours`) — **chỉ dùng làm Guardrail** (kiểm tra mô hình có quên kiến thức cũ không), **cấm** dùng để so sánh thắng/thua với baseline.
    2. `ALL_UNSEEN` (23 generator chưa từng thấy khi train, bao gồm DALL-E 3, FLUX 1 Dev/Schnell, Midjourney, SD 1.4/1.5/XL, StyleGAN2/3, ProGAN, ProjectedGAN, GauGAN, StarGAN, Glide, IMLE, VQ-Diffusion, ...) — **thước đo chính** trên AI-GenBench (báo cáo cả макро-average theo generator và pooled).
    3. `ALL_UNSEEN_EXCL_ORIGIN` — loại thêm các `origin_dataset` trùng với tập train (`COCO`, `DRCT` nếu có) để khử rò rỉ nguồn ảnh real.

### P0.9 — Lưu điểm số từng ảnh (`scores_{model}_{protocol}.parquet`) [Phục vụ P0.5 & P1]
- **Hiện trạng:**
  - `score_items` chỉ trả về `y_true, y_prob` trong RAM rồi xóa ngay sau khi tính metric.
- **Thiết kế Yardstick v2:**
  - Mọi lần chấm đều ghi lại bảng chi tiết từng mẫu ra `/kaggle/working/scores_{MODEL_ID}_{PROTOCOL_MODE}.parquet` với schema cố định:
    - `image_id` (`str`: đường dẫn tương đối hoặc `arrow_row_{idx}`)
    - `benchmark` (`str`: `ai_genbench`, `wildrf`, `synthwildx`, `bfree_viral_images`, `realdeg::...`)
    - `group` (`str`: tên generator hoặc platform `facebook`/`reddit`/`twitter`)
    - `is_seen_gen` (`bool`)
    - `origin_dataset` (`str`)
    - `condition` (`str`: `clean`, `single_jpeg`, `heldout_webp`, ...)
    - `label` (`int`: `0`=real, `1`=fake)
    - `logit` (`float32`: logit thực sự $z = l_1 - l_0$ hoặc $l$ trước sigmoid)
    - `prob` (`float32`: $\sigma(z)$)
    - `width`, `height`, `format` (`int`, `int`, `str` — ghi siêu dữ liệu ảnh ngay lúc mở PIL để phục vụ trực tiếp chẩn đoán **D1, D3, D4** ở P1 mà không cần đọc lại 158 GB dữ liệu!).

### P0.10 — Provenance & Quản lý phiên bản Harness (`harness_version = 2`) [Quy tắc #8]
- **Thiết kế Yardstick v2:**
  - Mọi dòng trong `eval_results_v2_{MODEL_ID}.csv` và metadata của `.parquet` bắt buộc có 5 cột định danh theo Quy tắc bất biến #8:
    - `git_sha`: commit SHA của repo `bfree_src`
    - `ckpt_sha256`: SHA-256 (16 ký tự đầu hoặc đầy đủ) của file trọng số đã nạp
    - `config_hash`: SHA-256 của cấu hình model/eval
    - `seed`: `AIGENBENCH_SEED` (`42`)
    - `harness_version`: `2` (kèm `protocol_mode`)
  - Cập nhật `research/eval-changelog.md` ghi nhận nâng cấp lên `harness_version = 2`.

### P0.11 — Chạy lại toàn bộ Baseline trên Yardstick v2 (`USER-KAGGLE`)
- Danh sách chạy trên Kaggle sau khi đóng gói notebook v2:
  1. `bfree_official` (kiểm tra sanity: đối chiếu số paper B-Free trên `bfree_viral_images` / `Synthbuster` subset).
  2. `globalforge_dinov3` (và `globalforge_dinov2` nếu có).
  3. `ours-v2` (checkpoint phase-2 stage-2 đã chốt ở Mục 10.2) + `ours-v1` (checkpoint phase-2 best nếu giữ để so sánh lịch sử).

---

## 3. Kiểm tra & Xác minh P0.12 — Lỗi `sigmoid(2l)` của `bfree_official` **[ĐÃ XÁC MINH]**

### 3.1. Bằng chứng trực tiếp trong code Harness v1
Đã mở và kiểm tra trực tiếp cả 3 file notebook thực tế:
- `C:\Users\ADMIN\OneDrive - ptit.edu.vn\Documents\xla-eval-harness-ours.ipynb` (Cell 6, dòng 221)
- `C:\Users\ADMIN\Downloads\xla-eval-harness-b-free.ipynb` (Cell 6, `id="09aa6ff8"`)
- `C:\Users\ADMIN\Downloads\xla-eval-harness-ours (1).ipynb` (Cell 6, `id="bdcdf7d6"`)

Trong hàm `_load_bfree_official()`:
```python
    def forward_fn(m, x):
        l = m(x)                      # num_classes=1 -> pseudo 2-logit: softmax[:,1] = sigmoid(l)
        return torch.cat([-l, l], dim=1)
```
Và trong `ModelWrapper.score(self, pil_images)`:
```python
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = self.forward_fn(self.model, batch)
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].mean().item())
```

### 3.2. Chứng minh toán học sai lệch
Với đầu ra 1 logit $l \in \mathbb{R}$ của B-Free (`num_classes = 1`, trong đó xác suất chuẩn theo code gốc `code/utils/dmetrics.py:85` là `prob = expit(y_score) = sigmoid(l)`):
- Khi ghép `logits = torch.cat([-l, l], dim=1)`, vector 2 chiều là $z = [-l, +l]$.
- Khi qua `torch.softmax(logits, dim=-1)[:, 1]`, xác suất tính được là:
  $$p_{\text{harness}} = \frac{e^{l}}{e^{-l} + e^{l}} = \frac{1}{1 + e^{-2l}} = \sigma(2l) \neq \sigma(l)$$
- Tương đương việc **nhân đôi độ lớn logit ($2l$) hay chia nhiệt độ $T = 0.5$** trước khi lấy sigmoid!
- Đối chiếu với code gốc B-Free tại `code/main_bfree.py` và `code/utils/dmetrics.py:56,85`:
  - `balanced_cllr` và `balanced_ece_binary` trong `dmetrics.py` đều nhận đầu vào là **logit tự nhiên $l$**, bên trong dùng `expit(l)` tức $\sigma(l)$, tức hiệu hai logit phải là $l_1 - l_0 = l$ (ví dụ `[0, l]` hoặc `[-l/2, +l/2]`), **không phải** `[-l, +l]`.

### 3.3. Tác động đến các chỉ số cũ (v1)
1. **Không bị ảnh hưởng (bất biến dưới phép biến đổi đơn điệu tăng nghiêm ngặt $l \mapsto 2l$ khi chỉ có 1 crop):**
   - `AUC` (`roc_auc_score`).
   - `bAcc@0.5` trên 1 crop (vì $\sigma(2l) \ge 0.5 \iff 2l \ge 0 \iff l \ge 0 \iff \sigma(l) \ge 0.5$).
2. **Bị làm sai lệch trực tiếp:**
   - **`ECE` và `NLL` của `bfree_official`:** Vì $\sigma(2l)$ đẩy mọi xác suất về sát $0$ hoặc $1$ (over-confident giả tạo với nhiệt độ $T=0.5$), `NLL` và `ECE` của `bfree_official` trong bảng v1 bị phóng đại/lệch so với thực tế.
   - **`CALIB_THRESHOLD` (`thr_calib`):** Ngưỡng xác suất tối ưu bị biến dạng theo $\sigma(2l)$.
   - **Trường hợp `multi_crop=True`:** Trung bình $\frac{1}{K}\sum_k \sigma(2l_k) \neq \sigma\left(\frac{1}{K}\sum_k l_k\right)$ (lệch cả thứ hạng nếu dùng nhiều crop ngoài `Wrapper5crops`).

### 3.4. Cách sửa chuẩn xác trong Yardstick v2
1. **Chuẩn hóa đầu ra của `ModelWrapper` về logit tự nhiên $z \in \mathbb{R}$:**
   - Với mô hình 1-logit (`bfree_official`): $z = l = m(x).\text{squeeze}(-1)$ (hoặc nếu biểu diễn 2-logit thì phải là `torch.cat([-l / 2, l / 2], dim=1)` hoặc `torch.cat([torch.zeros_like(l), l], dim=1)` để $z_1 - z_0 = l$).
   - Với mô hình 2-logit (`ours`, `globalforge`): logit nhị phân tự nhiên tương ứng với $\text{softmax}(z)_1 = \sigma(z_1 - z_0)$ chính là:
     $$z = z_1 - z_0 = \text{logits}[:, 1] - \text{logits}[:, 0]$$
     (đúng như `code/eval/run_benchmarks.py:40` đã viết: `score = (logits[:, 1] - logits[:, 0]).item()`).
   - Khi gộp nhiều crop: lấy trung bình trên **logit** $\bar{z} = \frac{1}{K}\sum_{k=1}^K z_k$ (đúng chuẩn `Wrapper5crops.forward` dòng 57: `y = torch.mean(...)` trên logit), sau đó xác suất là $p = \sigma(\bar{z}) = \text{torch.sigmoid}(\bar{z})$.
2. **Truyền trực tiếp logit $\bar{z}$ vào các hàm `dmetrics.py` (`balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2`):** đảm bảo tương thích 100% với định nghĩa của tác giả B-Free.

---

## 4. Bảng tóm tắt vị trí Cell sẽ cập nhật khi đóng gói Notebook Harness v2

| Cell (theo `xla-eval-harness-ours (1).ipynb`) | Nội dung hiện tại | Thay đổi trong Yardstick v2 (`harness_version = 2`) |
|---|---|---|
| **Cell 2** (`id="f15431b5"`) | `MODEL_ID`, `BENCHMARKS_ENABLED` (`realdeg_bench_style: False`) | Thêm `PROTOCOL_MODE = "official"` (`"official"` \| `"unified_1crop"` \| `"unified_5crop"`), `HARNESS_VERSION = 2`; bật `realdeg_bench_style: True` |
| **Cell 3** (`id="df30088a"`) | `CONFIG-PATHS`, `AIGENBENCH_MAX_PER_GEN = 150`, `AIGENBENCH_MAX_REAL = 1500` | Tăng `AIGENBENCH_MAX_PER_GEN = 500`, `AIGENBENCH_MAX_REAL = 3000`; thêm `SEEN_GENERATORS` (13 gen), `CALIB_SEED = 2026`; thêm hàm băm `sha256_file` cho provenance |
| **Cell 6** (`id="bdcdf7d6"`) | `ModelWrapper`, `_load_ours`, `_load_bfree_official` (`cat([-l, l])`), `_load_globalforge` | Sửa P0.12 (`z = l` cho B-Free, `z = l1 - l0` cho 2-logit, `p = sigmoid(z)`); tách nhánh `PROTOCOL_MODE == "official"` (B-Free full-image `Wrapper5crops` không pre-upscale; GlobalForge official resize/crop/fp32) và `"unified_*"` |
| **Cell 7** (`id="e059485f"`) | `binary_ece`, `balanced_nll`, `find_optimal_threshold`, `compute_all_metrics` | Thay bằng bộ metric chuẩn `dmetrics.py` (`balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2`) + `fpr_at_tpr95`, `tpr_05`, `fpr_05`, `tpr_calib`, `fpr_calib` + Bootstrap CI 95% (1,000 lần) + Temperature Scaling $T$ |
| **Cell 8** (`id="06bee8af"`) | `ArrowSource`, `load_pil`, `score_items` | Thu thập thêm `(image_id, width, height, format, logit, prob)` để xuất `scores_{MODEL_ID}_{PROTOCOL_MODE}.parquet` (P0.9); truyền seed theo từng ảnh vào hàm suy giảm |
| **Cell 9** (`id="e7e1836d"`) | `DEGRADE_STRENGTHS` (7 op), `apply_degradation`, `apply_compound` | Tách `DEGRADE_INDIST_OPS` (4 op) và `DEGRADE_HELDOUT_OPS` (6 op mới: `webp`, `jpeg_subsampling`, `resize_lanczos_bilinear`, `poisson_iso_noise`, `sharpen`, `social_pipeline`); dùng RNG riêng theo từng ảnh |
| **Cell 10–11** (`id="ce90669f"`, `ba6e55a7`) | `build_ai_genbench_groups`, `build_benchmark_filelist`, `build_synthwildx_items` | Tách `calib_items` độc lập khỏi `test_items` (P0.7); sửa `build_synthwildx_items` tự động dò tìm thư mục con chứa `dataset_dict.json`/`state.json`/`.parquet`/ảnh (P0.4); thêm loader `wildrf/val` cho calibration |
| **Cell 12–14** (`id="07ef20fb"`, `1d9cb38a`, `2baa28f2`) | `run_ai_genbench`, `run_static_benchmark`, `run_synthwildx`, `run_realdeg` | Fit `CALIB_THRESHOLD` & $T$ trên `calib_items` + `wildrf/val` trước khi chấm test; xuất dòng tổng hợp tách biệt `ALL_SEEN`, `ALL_UNSEEN`, `ALL_UNSEEN_EXCL_ORIGIN` (P0.8); lưu `.parquet` điểm từng ảnh (P0.9) và thêm 5 cột provenance vào CSV (P0.10) |
