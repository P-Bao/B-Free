# AGENT_PLAN_V4 — Đánh giá & cải tiến B-Free × GlobalForge (hướng "chẩn đoán trước, ablation rộng")

> Nguồn: `danh-gia-va-plan-cai-tien-v4.md` + đối chiếu với repo (`repomix-output.md`).
> File này viết cho **AI agent thực thi**. Đọc hết mục 0–2 trước khi làm bất cứ việc gì.
> Nhãn: **[ĐÃ XÁC MINH]** = thấy trực tiếp trong code/log; **[GIẢ THUYẾT]** = cần thí nghiệm; **[AGENT-CẦN-KIỂM]** = suy ra từ đọc code, phải xác minh bằng test trước khi tin.

---

## 0. Mục tiêu & triết lý

Mô hình hiện tại (`ours`, LoRA r=16 trên DINOv2 ViT-B/14 reg4 + LIB + GSR + L_DCS) **thua B-Free official ngoài AI-GenBench** (WildRF AUC 0.943 vs 0.974; bfree_viral 0.893 vs 0.967) dù gần như tuyệt đối trên 13 generator đã thấy khi train.

Kế hoạch V4 đi theo thứ tự cứng:

```
P0 Yardstick v2 ─┬─> P1 Chẩn đoán ──> P2 Sửa recipe + Ablation ──> P3 Cải tiến cơ chế ──> P4 Đánh giá cuối
                 └─> P5 Dọn repo (song song, không chặn)
```

**Không chạy MS-DCS (v3, K=4) cho đến khi P2 xong.** v3 hiện **chưa từng chạy** (code đã có, `multi_severity.enabled=false`); mọi nhận định về hiệu quả của nó chỉ là giả thuyết.

Nguyên nhân gốc đã nhận diện (xem `W1..W9` trong tài liệu nguồn):

| Mã | Nội dung | Trạng thái |
|---|---|---|
| W1 | CE chỉ tính trên view sạch; view suy giảm chỉ vào InfoNCE với λ=0.01 (loss ≈ 0.0105 ⇒ đóng góp ≈ 1e-4 so với CE ≈ 0.2) → DCS gần như không tác động | ĐÃ XÁC MINH (`compute_loss` trong `networks/bfree_globalforge_vit.py`) |
| W2 | Không ablation; không init từ B-Free; GSR `proj` khởi tạo ngẫu nhiên, không gate; LIB α₀≈0.30; head mean-pool thay vì CLS | ĐÃ XÁC MINH |
| W3 | AIGB ≈ 64% mẫu train; real/fake AIGB khác `origin_dataset`; `ensure_min_size` upscale bicubic → nguy cơ shortcut | số liệu ĐÃ XÁC MINH, cơ chế GIẢ THUYẾT |
| W4 | So sánh harness không công bằng (U1–U7 trong `research/eval-diff.md`) | ĐÃ XÁC MINH |
| W5 | Val bão hòa (val bAcc 0.992 từ epoch 1; CE chạm sàn label-smoothing 0.1997 vs 0.1985) | ĐÃ XÁC MINH |
| W6 | Label smoothing 0.1 ⇒ ECE sàn ≈ 0.05; ECE không cân bằng lớp | ĐÃ XÁC MINH |
| W7 | 150 fake/generator, 1 seed, không CI; provenance checkpoint "ours" không truy ngược được | ĐÃ XÁC MINH |

---

## 1. Quy tắc bất biến (vi phạm = dừng và báo user)

Các quy tắc này lấy từ `agent.md`, `research/PATHS.md`, `research/log.md`, `research/eval-diff.md`. Nếu mâu thuẫn, **`research/PATHS.md` (đã xác minh bằng `ls`) thắng `agent.md`** về đường dẫn.

1. **Đường dẫn thật:** code làm việc ở worktree `B-Free-loss-backbone/` (nhánh `integration/loss-backbone`), thư mục `code/` bên trong. `B-Free/` (đối chiếu, HEAD hiện ở `integration/train-eval`) và `GlobalForge-BE0F/` là **chỉ đọc**. `agent.md` nhắc `B-Free-integration/` nhưng thư mục này **không tồn tại**.
2. **Commit:** tác giả `P-Bao <BaoPV.B23KH006@stu.ptit.edu.vn>`; **không** thêm trailer `Co-Authored-By`; chỉ commit file đã chủ ý sửa. Worktree là CRLF còn blob là LF (38 file hiện hiện "modified" chỉ vì khoảng trắng) → file nào sửa thì chuyển LF rồi mới commit.
3. **🔒 Cần user duyệt trước khi chạm** vào: metric, split, tiền xử lý của harness; phương pháp calib; bật RealDeg; mọi thay đổi làm đổi ý nghĩa số liệu cũ (theo `CLAUDE.md`, nhắc lại trong `eval-diff.md`). Các việc đánh dấu 🔒 bên dưới: viết đề xuất vào `research/` và **chờ OK**.
4. **Mọi thay đổi mặc định phải tắt được bằng config và đường cũ giữ bit-identical** (như MS-DCS đã làm: có `tests/ref_single_view.pt`). Thêm cờ mới ⇒ thêm vào `configs/bfree_dcs.yaml` **và** vào `RESUME_CHECK_KEYS` trong `train_lora.py` (nếu không, resume sẽ không phát hiện lệch cấu hình).
5. **Hyperparameter không hardcode:** để trong YAML/CLI (vì cần sweep).
6. **Gate bắt buộc trước khi tốn GPU:** `tests/overfit_test.py` (32 ảnh, loss → ≈0 trong ≤50 bước), `tests/resume_test.py` (RESUME PASS), test bit-identity của đường legacy, và test mới "bước 0 ≡ B-Free" (xem P2.B).
7. **Mọi quyết định tiêu chí thành công được ghi vào `research/` TRƯỚC khi chạy** (giữ thông lệ H1/H2). Kết quả 1 seed = "quan sát", không phải "chứng minh".
8. **Mọi CSV/parquet kết quả có cột:** `git_sha`, `ckpt_sha256`, `config_hash`, `seed`, `harness_version`.
9. **Không bịa số.** Chưa chạy ⇒ ghi `TODO: pending run`. Không điền ước lượng vào bảng kết quả.
10. **Gặp sai lệch so với giả định** (ví dụ D1 bác bỏ giả thuyết shortcut) ⇒ dừng, ghi `research/log.md` (provenance `ai`), báo user, **không tự đổi hướng**.
11. **Agent không có GPU Kaggle.** Việc huấn luyện/eval lớn là **USER-KAGGLE**: agent chuẩn bị notebook + `research/iter-N.md` theo mẫu (*Changed / Verified / You must do / Paste back*) rồi dừng chờ user dán kết quả.
12. **Môi trường CPU local:** conda env `xla-cpu` (py3.11, torch CPU 2.8.0, timm 1.0.22, peft 0.15.2). Không đụng env `face_recognition`. Kaggle chạy **offline**, đường dẫn viết cứng, fail-fast.
13. **Giấy phép:** code/dữ liệu B-Free chỉ phi lợi nhuận, phải ghi nguồn — nhắc trong báo cáo cuối.
14. `tests/` đang nằm trong `.gitignore` ⇒ commit test bằng `git add -f` (cho đến khi P5.4 sửa).

Ký hiệu cột **Ai làm**: `AGENT` (CPU/code), `USER-KAGGLE` (cần GPU), `🔒` (cần OK user).

---

## 2. Bản đồ repo (những chỗ sẽ sửa)

| Thành phần | File | Ghi chú đã xác minh |
|---|---|---|
| Mô hình | `code/networks/bfree_globalforge_vit.py` | `BFreeGlobalForgeViT`; `forward()` → `{"logits","cls"}` với `cls` = mean-pool patch token **sau LIB/GSR**; head gọi trực tiếp `model.fc_norm`, `head_drop`, `head`; `compute_loss()` → `(total, ce, dcs)` |
| LIB | `code/modules/lib_adapter.py` | Gaussian 3×3 depthwise cố định, `beta` init −0.85 (α≈0.30); `lib_tau` **không dùng**; trả `mask` không ai dùng |
| GSR | `code/modules/gsr_adapter.py` | single-head attn, che Chebyshev ≤ `window` (kể cả chính token), `out = proj(attn@V) + x`, **không gate, init mặc định** |
| DCS | `code/modules/dcs_loss.py` | `info_nce_loss` (symmetric, negative trong batch), `multi_severity_dcs` |
| Train | `code/train_lora.py` | `PairDataset` trả `(clean, degraded, label)`; `apply_degradation/apply_compound/realistic_lifecycle_degrade`; `load_pretrained_backbone` (chỉ DINOv2 safetensors); `apply_lora_to_backbone`; `main()`; `RESUME_CHECK_KEYS`; `run_epoch`; AdamW + CosineAnnealingLR |
| Checkpoint B-Free | `networks/__init__.py::get_network`, `wrapper5crops.py`, `BFreeGlobalForgeViT.load_checkpoint` | B-Free head 1 logit (`num_classes=1`); ours 2 logit |
| Metric | `code/utils/dmetrics.py` | có sẵn `balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far`, `calculate_eer2` |
| Config | `code/configs/bfree_dcs.yaml`, `loader.py` | có `multi_severity` (mặc định OFF) |
| Harness | `notebooks/xla-eval-harness-*.ipynb` (không có trong repomix) | 3 notebook dùng chung 1 codebase; `ModelWrapper` quyết định crop/score — **agent phải mở file thật trước khi sửa** |
| Nhật ký | `research/` (`log.md`, `iter-N.md`, `eval-diff.md`, `eval-changelog.md`, `PATHS.md`) | cập nhật theo mẫu hiện có |

---

## 3. P0 — Yardstick v2 (🔒 cần user duyệt; ~1 ngày, GPU thấp)

**Mục tiêu:** một thước đo công bằng, đủ lớn, có CI; mọi kết luận sau đó dựa vào nó.

**Bước đầu (AGENT, không cần duyệt):** viết `research/yardstick-v2.md` tổng hợp P0.1–P0.11, đối chiếu từng mục với `eval-diff.md` (U1–U7) và mở các notebook harness để ghi lại vị trí cell cần sửa. Sau đó xin OK user cho các mục 🔒.

| # | Việc | Chi tiết / tiêu chí xong | Ai |
|---|---|---|---|
| P0.1 | 🔒 **Chính sách crop** | Báo cáo song song 2 giao thức: (i) *official-per-model*: B-Free dùng 5 crop trên embedding toàn ảnh, không pre-upscale (đúng `Wrapper5crops` + `replicate_wrap`); GlobalForge theo preprocessing gốc fp32; (ii) *unified*: cùng một chính sách crop/resize cho mọi mô hình. Mỗi số ghi rõ thuộc giao thức nào | AGENT + 🔒 |
| P0.2 | 🔒 **GlobalForge off-protocol (U3)** | Dựng lại 224 crop, resize-to-1296, JPEG q100, fp32 như bản chính thức; hoặc loại khỏi bảng bAcc, chỉ giữ AUC kèm chú thích | AGENT + 🔒 |
| P0.3 | 🔒 **Bật RealDeg, seed theo ảnh, có toán tử held-out** | Held-out không có trong train: WebP, JPEG chroma-subsampling, Lanczos/bilinear resize, Poisson/ISO noise, sharpen, pipeline "mạng xã hội" (resize→JPEG→resize→JPEG). Báo cáo tách *in-distribution* vs *held-out*. **Bỏ brightness/contrast/saturation khỏi tuyên bố độ bền** (AUC không đổi) | AGENT + 🔒 |
| P0.4 | **Sửa loader SynthWildX** | Có số cho cả 3 mô hình. Lỗi hiện tại: "neither a `Dataset` nor `DatasetDict` directory" — trỏ đúng thư mục chứa `dataset_dict.json`/`.arrow` | AGENT |
| P0.5 | 🔒 **Tăng cỡ mẫu + CI** | ≥ 500 fake/generator (hoặc toàn bộ); bootstrap 1,000 lần cho CI 95%; DeLong (AUC) / McNemar (bAcc) khi so cặp | AGENT + 🔒 |
| P0.6 | **Metric** | Thêm `balanced_ece_binary`, `balanced_nll_binary`, `pd_at_far(…,0.10)`, `calculate_eer2`, FPR@TPR95; **FPR/TPR riêng tại ngưỡng 0.5** | AGENT |
| P0.7 | 🔒 **Ngưỡng calib trên tập tách riêng** | Fit ngưỡng/temperature trên split calibration độc lập (phần giữ riêng của AIGB-train hoặc WildRF `val`), đánh giá 1 lần trên test. Giữ cột cũ để đối chiếu lịch sử | AGENT + 🔒 |
| P0.8 | **Aggregate seen/unseen** | Hai nhóm tách bạch cho mọi mô hình (13 generator seen = `PRIORITY_GENERATORS` trong `train_lora.py`). *Seen* chỉ để kiểm tra mô hình có học được gì, **không dùng để so sánh** | AGENT |
| P0.9 | **Lưu điểm từng ảnh** | `scores_{model}.parquet` (id, label, score, generator, benchmark, điều kiện) | AGENT |
| P0.10 | **Provenance** | Metadata theo quy tắc #8; viết `eval-changelog.md` cho `harness_version = 2` | AGENT |
| P0.11 | **Chạy lại mọi baseline** | B-Free official, GlobalForge, ours-v1/v2 (mọi checkpoint còn giữ). Sanity: B-Free official tái lập được số paper trên ≥1 bộ chung | USER-KAGGLE |
| P0.12 | **Kiểm tra cách tính xác suất của B-Free** *(bổ sung từ đối chiếu V5)* | Trong `ModelWrapper`, kiểm tra nếu `forward_fn` của `bfree_official` trả `cat([-l, l])` rồi harness `softmax` ⇒ p = `sigmoid(2l)` (không phải `sigmoid(l)` như comment ghi). AUC/bAcc@0.5 không đổi nhưng ECE/NLL/ngưỡng xác suất lệch. Nếu đúng: dùng logit gốc cho mọi metric. **[AGENT-CẦN-KIỂM]** | AGENT |

**Cổng P0:** có `eval_results_v2/*.csv` cho ≥3 mô hình, CI đầy đủ, RealDeg + SynthWildX có số liệu → báo user. **Từ sau cổng này, mọi gate so với bảng v2, không so với số cũ.**

---

## 4. P1 — Chẩn đoán không cần train (~1–2 ngày, đọc điểm số đã lưu)

Mục tiêu: xác nhận hoặc bác bỏ W1, W3 **trước** khi tốn compute. Kết quả ghi vào `research/diagnosis.md` (bảng: xác nhận / bác bỏ / chưa rõ).

| # | Giả thuyết | Thí nghiệm | Quyết định rút ra |
|---|---|---|---|
| D1 | Shortcut định dạng/độ phân giải (W3) | Thống kê theo lớp và `origin_dataset`/generator của AIGB: kích thước ảnh, bảng lượng tử JPEG, định dạng file; tương quan điểm số ↔ thuộc tính; đo độ lệch điểm khi re-encode (JPEG q95/85/75, PNG) và resize | Điểm dịch mạnh theo định dạng/kích thước (hơn B-Free) ⇒ ưu tiên P2.C |
| D2 | DCS vô hiệu (W1) | Chạy 200 bước, ghi **chuẩn gradient của CE và của λ·DCS** lên tham số trainable, tính tỷ lệ | Tỷ lệ < 1% ⇒ P2.A bắt buộc. **Nếu D2 bác bỏ W1 ⇒ DỪNG, báo user** |
| D3 | Dạng lỗi trên WildRF/viral | Tại ngưỡng 0.5 và calib: FPR vs TPR; điểm trung bình ảnh real | Phân biệt "real bị báo fake" (shortcut nén) với "fake bị bỏ sót" |
| D4 | Nghịch lý SD2.1 | So thuộc tính ảnh SD2.1 B-Free-train vs AIGB; chấm ours trên SD2.1 in-dist (COCO-SD2.1 val) vs SD2.1-AIGB | In-dist ≈1.0 nhưng AIGB-SD2.1 thấp ⇒ vấn đề là *shift hậu xử lý* |
| D5 | Nhiễu khởi tạo của module | CKA/cosine giữa đặc trưng DINOv2 gốc và sau LIB/GSR ở bước 0 và sau train | Định lượng W2 |

**Cổng P1:** nếu D1 *bác bỏ* shortcut ⇒ hạ ưu tiên P2.C. Nếu D2 *bác bỏ* W1 ⇒ dừng và báo user trước khi sang P2.

---

## 5. P2 — Sửa recipe & ablation (trọng tâm; ~3–5 ngày)

Mọi cờ mới mặc định **tắt** (đường `legacy` giữ bit-identical).

### P2.A — Cho view suy giảm tham gia phân loại (sửa W1) · không tốn thêm forward

View suy giảm đã được forward sẵn cho DCS, dùng nó cho CE **không tăng chi phí**.

Thêm vào `configs/bfree_dcs.yaml` (và `RESUME_CHECK_KEYS`): `loss.mode: legacy|dual_view`, `loss.w_deg`, `loss.lam_cons`, `loss.cons_T`, `loss.p_aug_on_clean`.

```python
# networks/bfree_globalforge_vit.py :: compute_loss  (chỉ nhánh loss.mode == "dual_view")
out_c = self.forward(images_clean)
out_d = self.forward(images_deg)                       # đã có trong v2
ce_c  = self.ce(out_c["logits"], labels)
ce_d  = self.ce(out_d["logits"], labels)               # MỚI: nhãn không đổi dưới suy giảm
cons  = F.kl_div(F.log_softmax(out_d["logits"] / T, -1),
                 F.softmax(out_c["logits"].detach() / T, -1),
                 reduction="batchmean") * T * T        # MỚI: nhất quán quyết định
dcs   = info_nce_loss(out_c["cls"], out_d["cls"], tau=self.dcs.tau)
total = ce_c + w_deg * ce_d + lam_cons * cons + lam_dcs * dcs
# Trả (total, ce, dcs) — giữ hợp đồng 3 scalar. Thêm self.last_terms = {ce_c, ce_d, cons} để log.
```

- Khởi điểm: `w_deg = 1.0`, `lam_cons ∈ {0, 0.5}`.
- Nên thêm suy giảm "kiểu B-Free" lên cả view sạch với xác suất thấp (p≈0.2).
- Cập nhật `run_epoch` và vòng val để log thêm `ce_c`, `ce_d`, `cons` vào `train_log.csv` (như `dcs_k*` đã làm cho MS mode). Cả 3 chỗ gọi `compute_loss` (batch finder ~dòng 671, train ~918, val ~959) phải tương thích.
- **Test:** `loss.mode: legacy` ≡ tham chiếu `tests/ref_single_view.pt` (bit-identical); `dual_view` với `w_deg=0, lam_cons=0` ≡ legacy.

### P2.B — Init từ B-Free & không phá đặc trưng pretrained (sửa W2)

1. **`--init_from {dinov2,bfree}`** trong `train_lora.py` (mặc định `dinov2`, giữ nguyên đường cũ). `bfree` dùng `BFreeGlobalForgeViT.load_checkpoint`.
   - **[AGENT-CẦN-KIỂM] hai điểm rủi ro trong `load_checkpoint` hiện tại:**
     - Checkpoint B-Free (qua `Wrapper5crops.load_state_dict`) có key `patch_embed.*` và các key timm **không có tiền tố `model.`** (ví dụ `blocks.0.attn.qkv.weight`). Hàm hiện chỉ remap `patch_embed.*` → `model.patch_embed.*`, nên các key còn lại có thể thành *unexpected* và bị bỏ qua im lặng với `strict=False`. Báo cáo `num_missing/num_unexpected` phải được kiểm tra, và cần thêm tiền tố `model.` cho mọi key không phải `patch_embed.*`.
     - B-Free dùng head **1 logit**, ours **2 logit**: `load_state_dict(strict=False)` **vẫn báo lỗi khi sai shape**. Cần xử lý head riêng. Chuyển 1 logit → 2 logit sao cho `z = l1 − l0 = l`: `W2 = [−w/2; +w/2]`, `b2 = [−b/2; +b/2]`.
   - Test: bước 0 phải khớp B-Free (xem mục 5 dưới).
2. **Zero-init có gate cho GSR:** `out = tokens + gate * proj(attn @ V)` với `gate = nn.Parameter(torch.zeros(1))` (hoặc zero-init `proj`). Khi đó bước 0 ≡ B-Free. Cần cờ `backbone.gsr_zero_init`.
3. **LIB gần identity:** `beta_init = −6` (α₀ ≈ 0.0025) thay vì −0.85; cờ `backbone.lib_beta_init`.
4. **Head CLS:** cờ `backbone.head_mode: cls|meanpool`. Với `cls`: lấy `feats = self.model.forward_features(x)` rồi `self.model.forward_head(feats)` (timm: lấy token 0 → `fc_norm` → `head_drop` → `head`). `L_DCS` vẫn dùng mean-pool patch. Mean-pool chỉ giữ như một ô ablation riêng.
5. **GATE BẮT BUỘC "bước 0 ≡ B-Free":** logits của mô hình khởi tạo trùng B-Free gốc (sai số số học, fp32, |Δ| < 1e-3) trên 64 ảnh 504×504. Đây là test thay thế overfit gate (overfit gate không bắt được lỗi kiểu này). File: `tests/test_init_bfree.py`. Yêu cầu có checkpoint B-Free thật ⇒ nếu không có trên CPU local, test này là **USER-KAGGLE** hoặc dùng checkpoint nhỏ do user cung cấp.

### P2.C — Cân bằng dữ liệu & giảm shortcut (sửa W3; điều chỉnh theo D1)

Cờ dữ liệu mới (thêm vào YAML/`RESUME_CHECK_KEYS`): `data.mix_preset`, `data.fmt_match`, `data.res_match`, `data.holdout_generators`.

| Biến thể | Mô tả | Cách làm gợi ý |
|---|---|---|
| `mix_bfree` | Cặp B-Free ≥ 60–70% mẫu; AIGB ≤ 20–30% | Đặt `--aigenbench_max_per_gen` sao cho AIGB ≤ 25% tổng mẫu; tính và **ghi vào log** (xấp xỉ ~750/generator nếu giữ 13 generator, `aigenbench_val_frac` 0.1 — agent tự tính lại theo số liệu thật) |
| `mix_current` | Như hiện tại (AIGB ≈ 64%) — đối chứng | giữ mặc định (`4000`) |
| `mix_noaigb` | Chỉ B-Free + extra | bỏ `--ai_genbench_dir` |
| `fmt_match` | Real và fake cùng đi qua chuỗi nén ngẫu nhiên cùng phân phối (JPEG chất lượng ngẫu nhiên, xác suất như nhau cho hai lớp) | thêm bước vào `__getitem__` các dataset AIGB/extra |
| `res_match` | Bỏ upscale bicubic bắt buộc (`ensure_min_size`, dùng ở `ExtraTrainDataset` và `ExtraArrowTrainDataset`); dùng replicate-wrap như B-Free hoặc upscale xác suất bằng nhau cho real/fake | sửa `ensure_min_size` có cờ |

**Leave-generator-out thật sự:** giữ lại ≥3–4 trong 13 generator `PRIORITY_GENERATORS` (không bao giờ đưa vào train) và dùng real **từ origin khác** để đo tổng quát hóa.

### P2.D — Loss & hiệu chuẩn (sửa W5, W6)

- **Label smoothing:** so `ls ∈ {0.0, 0.05, 0.1}`; sau train luôn fit **temperature scaling** trên split calibration (P0.7).
- **λ_dcs** ∈ {0 (tắt), 0.01, 0.1, 0.5, 1.0}; **τ** ∈ {0.07, 0.2}. Dùng tỷ lệ gradient D2 để chọn khoảng 5–30%.
- **DCS thay thế** nếu InfoNCE theo batch vẫn "quá dễ" (loss 0.12 → 0.01 chỉ vài epoch): negative cùng lớp khác nguồn, hoặc dùng cặp real↔fake cùng nội dung của B-Free làm hard negative.

### P2.E — Validation & chọn checkpoint (sửa W5)

Val tổng hợp (không bão hòa), checkpoint chọn theo trung bình có trọng số của:
1. SD2.1 in-distribution (chỉ để phát hiện hỏng);
2. AIGB leave-generator-out;
3. Val suy giảm với toán tử held-out (P0.3);
4. WildRF `val` (**tuyệt đối không dùng `test`**).

Early-stop theo thành phần 2–4; epoch mặc định giảm 4–5 nếu đường cong xác nhận. `--select` ghi vào `selected_ckpt.txt` (đã có); bỏ sự không nhất quán `best` vs `last` (notebook eval của ours hiện load `*_best.pth`, trong khi `iter-1.md` quyết định dùng `last`).

### P2.F — Lưới ablation (2 giai đoạn)

*Sàng lọc* (USER-KAGGLE): 25% dữ liệu, 3 epoch, 1 seed, ≈35–45 phút/run, ~10–12 run ≈ 6–8 giờ. Agent chuẩn bị **một notebook cho mỗi nhóm run** + bảng cấu hình.

| Run | Init | Module | Loss | Dữ liệu | Mục đích |
|---|---|---|---|---|---|
| R0 | — | — | — | — | B-Free official trong harness v2 (tham chiếu) |
| R1 | B-Free | không | dual-view CE | mix_bfree | **Recipe thuần** |
| R1b | B-Free | không | CE sạch (legacy) | mix_bfree | Xác nhận W1: aug-CE có giúp thật? |
| R2 | B-Free | +DCS | dual-view | mix_bfree | Giá trị riêng của DCS |
| R3 | B-Free | +LIB | dual-view | mix_bfree | Giá trị riêng của LIB |
| R4 | B-Free | +GSR | dual-view | mix_bfree | Giá trị riêng của GSR |
| R5 | B-Free | LIB+GSR+DCS | dual-view | mix_bfree | **Mô hình đầy đủ** |
| R5-d | DINOv2 | LIB+GSR+DCS | dual-view | mix_bfree | Có cần init B-Free không? |
| R5-a | B-Free | LIB+GSR+DCS | dual-view | mix_current | Tác động của AIGB-heavy |
| R5-f | B-Free | LIB+GSR+DCS | dual-view + fmt/res match | mix_bfree | Tác động chống shortcut |
| R5-l | B-Free | LIB+GSR+DCS | dual-view, ls=0 + temp-scale | mix_bfree | Tác động của label smoothing |

*Xác nhận* (USER-KAGGLE): chọn ≤3 cấu hình tốt nhất, **đủ dữ liệu, 3 seed (42, 43, 44)**, 4–5 epoch × ~45 phút ≈ 27–34 giờ GPU; chia phiên bằng `--end_epoch` + `--resume_from` (mỗi phiên ≤12h, dùng cùng loại GPU).

### P2.G — Tiêu chí thành công & bác bỏ (ghi vào `research/directions/D-recipe-v4/README.md` TRƯỚC KHI chạy)

Số tham chiếu (harness v1, sẽ đo lại ở P0): mean AUC {WildRF fb/reddit/twitter, viral} = **0.930** (ours-v2) vs **0.972** (B-Free).

| Mã | Giả thuyết | Chỉ số (trên **test**, 1 lần) | Dự đoán | Bác bỏ nếu |
|---|---|---|---|---|
| H-A | Aug-CE + init B-Free sửa lỗi tổng quát hóa ngoài miền (R1 vs R1b, R1 vs ours-v2) | Mean AUC WildRF+viral | R1 ≥ 0.965 và ≥ ours-v2 + 0.03 | R1 < 0.945 |
| H-B | Aug-CE nâng độ bền suy giảm | Mean bAcc RealDeg held-out + compound 3–5 | ≥ +10 điểm so với ours-v2 (phase-2: 0.674/0.610/0.582) | < +4 điểm |
| H-C | LIB/GSR/DCS có đóng góp độc lập | R5 − R1: mean AUC (WildRF+viral+AIGB unseen+SynthWildX) | ≥ +0.005, CI không chứa 0 | CI chứa 0 ⇒ **kết luận trung thực: module không đóng góp** |
| H-D | Cân bằng dữ liệu giảm shortcut | R5 vs R5-a: AUC WildRF+viral; \|Δ điểm khi re-encode\| | ≥ +0.01 | < +0.003 |
| H-E | Calibration | ECE cân bằng sau temperature scaling, WildRF/viral | ≤ 0.08 | > 0.15 |
| Guard | Không đánh mất nội lực | AUC AIGB *seen* và SD2.1 in-dist | giảm ≤ 0.003 so với ours-v2 | giảm > 0.01 |

**Cổng P2:** bảng ablation đầy đủ + quyết định "giữ/loại" từng module theo H-C, **trình user trước khi sang P3**.

---

## 6. P3 — Cải tiến cơ chế (chỉ sau P2, chỉ những gì P2 chứng minh có ích)

Mỗi mục là **một thay đổi một lần**, có run đối chứng và tiêu chí cố định như P2.G.

| # | Việc | Điều kiện kích hoạt | Ghi chú |
|---|---|---|---|
| P3.1 | **MS-DCS rút gọn** | H-A/H-B đạt **và** DCS có đóng góp (H-C) | K=2 bin ngẫu nhiên/step (`views_per_step=2`, code đã hỗ trợ) thay K=4 ⇒ chi phí ~3× thay vì 5×. Dùng bin **tách khỏi** toán tử RealDeg-eval để held-out thật sự held-out. Mốc so sánh là R5, không phải v2 |
| P3.2 | **Full fine-tune thay LoRA** | LoRA r=16 là điểm nghẽn (thử r ∈ {8,16,32,64} trước) | B-Free gốc full FT; LR phân tầng, LR thấp 1e-5–2e-5. `train_full_ft.py` hiện **cũ/lệch** — kiểm tra nó dùng cùng pipeline dữ liệu/loss mới trước |
| P3.3 | **Thiết kế lại GSR** | GSR có đóng góp ở P2 | window ∈ {1,3,5,7} (lưới 36×36), multi-head, không che chính token, đặt giữa các block |
| P3.4 | **LIB học được** | LIB có đóng góp | kernel học được/multi-scale; kiểm tra α hội tụ về đâu (α→~0 ⇒ LIB vô dụng) |
| P3.5 | **Hard negative theo cặp** | DCS có đóng góp nhưng yếu | cặp real↔fake cùng nội dung làm negative cho InfoNCE |
| P3.6 | **TTA & hiệu chuẩn suy luận** | Luôn | 5-crop official + temperature scaling; báo cáo tách phần đóng góp của TTA |

---

## 7. P4 — Đánh giá cuối & báo cáo

1. Bảng chính: B-Free official, GlobalForge (đúng giao thức), ours-v2 (cũ), cấu hình thắng — **CI 95% bootstrap**, 3 seed (mean ± sd), kiểm định ghép cặp.
2. Bảng công bằng tách: *seen / unseen / wild / RealDeg in-dist / RealDeg held-out / SynthWildX*.
3. Bảng ablation R1–R5 + kết luận H-C **kể cả nếu âm tính**.
4. Phân tích lỗi: FPR/TPR ở ngưỡng cố định, đường cong độ bền theo cường độ, ví dụ định tính.
5. Mục "Giới hạn" trung thực: rò rỉ miền giữa train/eval, 1 backbone, giấy phép phi lợi nhuận của B-Free.
6. Cập nhật `research/` (`log.md`, `iter-N.md`, `eval-changelog.md`).

---

## 8. P5 — Dọn repo & tái lập (AGENT, song song, ưu tiên thấp nhưng làm sớm)

| # | Việc |
|---|---|
| P5.1 | Gộp logic suy giảm về **một module** `degradations.py` (tham số có đơn vị rõ). Hiện nhân bản ≥4 nơi: `train_lora.apply_degradation`, `bfree_datasets/multi_severity.py::_apply`, `bfree_datasets/bfree_dataset.py::DegradationPipeline`, bản sao trong harness; noise định nghĩa khác nhau (`std=√strength` vs `σ/255` vs `std∈[0.02,0.1]`). Test: train và eval cho cùng ảnh khi cùng seed |
| P5.2 | Xóa/đánh dấu deprecated: `train_full_ft.py`, `bfree_datasets/bfree_dataset.py`, `eval/run_benchmarks.py`, `scripts/run_phase6_search.sh`; cập nhật `Dockerfile` + `requirements.txt` (hiện torch 2.0.1) theo môi trường thật (torch 2.8.0, timm 1.0.22, peft 0.15.2) |
| P5.3 | Bỏ hyperparameter chết `lib_tau` (hoặc làm cho có nghĩa); bỏ `mask` không dùng của LIB |
| P5.4 | Bỏ `tests/` khỏi `.gitignore`; thêm test: bước 0 ≡ B-Free, gate GSR=0 ⇒ identity, dual-view loss tách đúng các term, schema parquet score |
| P5.5 | Chạy thử `train_lora.py main()` end-to-end trên CPU với dữ liệu giả nhỏ (khoảng trống đã ghi ở `log.md`; lần trước user từ chối lệnh — **xin phép trước**) |
| P5.6 | Đồng bộ `PATHS.md`, `CLAUDE.md`, `agent.md` (điền bảng "Dữ kiện xác nhận" theo code thật: LIB/GSR/DCS đã đọc từ `REM.py`, `engine_finetune.py`) |
| P5.7 | Chuẩn hóa slug Kaggle (sửa typo `xla-trainng-stage-2`); lập bảng ánh xạ phiên bản ↔ notebook ↔ checkpoint ↔ SHA |

---

## 9. Ước lượng & rủi ro

| Giai đoạn | GPU ước tính | Ghi chú |
|---|---|---|
| P0 | ~6–10 h (chủ yếu eval) | cần user duyệt trước |
| P1 | ~2–4 h | chủ yếu phân tích điểm đã lưu |
| P2 sàng lọc | ~7–8 h | 1 phiên Kaggle |
| P2 xác nhận | ~27–34 h | 3–4 phiên, dùng resume |
| P3 | ~2–5 h/ô sàng lọc | chỉ ô được kích hoạt |
| P4 | ~8–10 h | |
| **Tổng** | **~55–80 h** | từ ~44 phút/epoch đã quan sát (batch 64, 2 view, ~146k mẫu); đo lại khi đổi dữ liệu/số view |

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| D1/D2 bác bỏ W1/W3 | Điểm không dịch theo định dạng; tỷ lệ gradient DCS đã ổn | Dừng, báo user; xem lại LoRA/head/dữ liệu nội dung |
| Aug-CE không đủ bắt kịp B-Free ở WildRF | R1 < 0.945 AUC | Thêm đúng bộ augmentation inpainted++ (cut-out, jitter), tăng p suy giảm nhánh sạch; cân nhắc full FT (P3.2) |
| H-C âm | CI chứa 0 | **Chấp nhận kết quả âm**, định vị lại dự án quanh recipe (aug-CE + bias-free + dữ liệu); không ép thêm module |
| Init B-Free làm kiệt khả năng học module mới | R5 ≈ R1, gate ≈ 0 | Tăng LR cho gate/module, thử R5-d, hoặc warm-up 2 giai đoạn |
| Vượt ngân sách GPU | Run xác nhận quá dài | Còn 2 seed cho cấu hình 2–3, giữ 3 seed cho cấu hình chính |
| Rò rỉ miền AIGB↔eval | Chênh seen/unseen lớn bất thường | dùng `AIGENBENCH_EXCLUDE_ORIGIN` (cơ chế loại `origin_dataset` trùng của harness) ở eval; ưu tiên bộ ngoài AIGB |

---

## 10. Việc cần user quyết định (agent hỏi ngay ở lượt làm việc đầu tiên)

1. 🔒 Duyệt P0.1–P0.8 (đổi giao thức đo, phương pháp calib, bật RealDeg held-out).
2. Xác nhận **checkpoint nào** sinh `eval_results_ours.csv` (v1 phase-2, v2, hay v3-stage-2) và còn giữ file không.
3. Hoãn MS-DCS (v3) đến sau P2, hay chạy song song như nhánh thăm dò độc lập (khi đó dùng recipe cũ, kết quả chỉ tham khảo).
4. Chốt ngưỡng bác bỏ H-A…H-E (chỉnh được, nhưng phải chốt **trước** khi chạy).
5. Ngân sách GPU/Kaggle còn lại để cắt/giữ lưới ablation.
6. Có checkpoint B-Free official và cách nạp (`weights/<model_name>/config.yaml`) trên Kaggle không — cần cho P2.B.

---

## 11. Checklist thực thi (cập nhật khi làm)

- [ ] Hỏi user 6 câu ở mục 10
- [ ] P0: `research/yardstick-v2.md` → OK user → sửa harness → chạy lại baseline → đóng băng bảng v2
- [ ] P0.12: kiểm tra `sigmoid(2l)` của B-Free
- [ ] P1: D1–D5 → `research/diagnosis.md` → báo user
- [ ] P2.A dual-view (+ test bit-identity) · P2.B init B-Free + gate zero-init + **test bước 0 ≡ B-Free** · P2.C dữ liệu · P2.D loss · P2.E val
- [ ] Ghi tiêu chí P2.G vào `research/directions/D-recipe-v4/README.md` **trước** khi chạy
- [ ] Sàng lọc R0–R5-l → chọn ≤3 → xác nhận 3 seed
- [ ] P3 theo điều kiện kích hoạt · P4 báo cáo · P5 dọn repo (song song)
