# Direction D-recipe-v4 — P2 Recipe Fix & Screening/Confirmation Ablation (V4)

> **Đăng ký trước khi chạy (Pre-registration theo Quy tắc #7 & Mục P2.G `AGENT_PLAN_V4.md`):**
> Tài liệu này khóa cứng các giả thuyết, chỉ số đánh giá và ngưỡng thành công/bác bỏ **trước khi** bấm chạy lưới Ablation P2.F trên Kaggle GPU.

## 1. Bối cảnh & Số tham chiếu
- Chẩn đoán P1 (`research/diagnosis.md`) đã **xác nhận** cả hai giả thuyết gốc:
  - **D1 (W3 — Shortcut nội suy/định dạng & mất cân bằng nguồn AI-GenBench):** AI-GenBench chiếm `63.93%` mẫu train; nhóm generator độ phân giải gốc $\ge 512\text{px}$ không qua `ensure_min_size(504, BICUBIC)` sụp đổ xuống `bAcc@0.5 = 78.88%` (thua `bfree_official` `-12.98%`).
  - **D2 (W1 — Bất cân xứng gradient DCS & thiếu $\mathcal{L}_{\text{CE}}^{\text{deg}}$):** Với $\lambda_{\text{dcs}}=0.01$, tỷ lệ chuẩn gradient $\|\nabla(\lambda\mathcal{L}_{\text{DCS}})\|_2 / \|\nabla\mathcal{L}_{\text{CE}}\|_2$ chỉ đạt `0.35% - 0.37%` ($< 1\%$); classifier head nhận `0.000%` gradient từ nhánh suy giảm.
- Số tham chiếu lịch sử (Harness v1, đối chiếu lại với `R0` trên Yardstick v2):
  - Mean AUC `{WildRF fb/reddit/twitter, viral}` = **0.930** (`ours-v2`) vs **0.972** (`bfree_official`).

## 2. Bảng Tiêu chí Thành công & Bác bỏ (P2.G — Khóa cứng)

| Mã | Giả thuyết | Chỉ số (trên tập **test**, chấm 1 lần) | Dự đoán (Pass) | Bác bỏ nếu (Reject) |
| :--- | :--- | :--- | :--- | :--- |
| **H-A** | Aug-CE (`dual_view`) + init B-Free sửa lỗi tổng quát hóa ngoài miền (`R1` vs `R1b`, `R1` vs `ours-v2`) | Mean AUC `WildRF + viral` | `R1 >= 0.965` và `>= ours-v2 + 0.03` | `R1 < 0.945` |
| **H-B** | Aug-CE nâng độ bền dưới suy giảm | Mean bAcc `RealDeg held-out + compound 3-5` | `>= +10` điểm % so với `ours-v2` (phase-2: `0.674 / 0.610 / 0.582`) | `< +4` điểm % |
| **H-C** | `LIB` / `GSR` / `DCS` có đóng góp độc lập | `R5 - R1` (và `R2/R3/R4 - R1`): Mean AUC (`WildRF + viral + AIGB unseen + SynthWildX`) | `>= +0.005`, Bootstrap CI 95% của $\Delta\text{AUC}$ không chứa `0` | CI 95% chứa `0` $\Rightarrow$ **kết luận trung thực: module không đóng góp, loại bỏ khỏi cấu hình cuối** |
| **H-D** | Cân bằng dữ liệu (`mix_bfree` + `fmt/res match`) giảm shortcut | `R5` vs `R5-a` (và `R5-f` vs `R5`): AUC `WildRF + viral`; $\|\Delta\text{điểm khi re-encode}\|$ | `>= +0.01` AUC | `< +0.003` AUC |
| **H-E** | Hiệu chuẩn xác suất (Calibration) | `balanced_ece_binary` sau Temperature Scaling trên `WildRF / viral` | `<= 0.08` | `> 0.15` |
| **Guard** | Không đánh mất nội lực in-distribution | AUC `AIGB seen` (13 gen) và `SD2.1 in-dist` | Giảm `<= 0.003` so với `ours-v2` | Giảm `> 0.01` |

## 3. Quy tắc Cổng P2 (Gate sang P3)
- Hoàn tất bảng kết quả Ablation P2.F (`R0` $\rightarrow$ `R5-l`).
- Đưa ra quyết định **"giữ / loại"** từng module (`LIB`, `GSR`, `DCS`) dựa trên kiểm định **H-C** và trình User duyệt trước khi bước sang P3.
