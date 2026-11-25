# GIẢI THÍCH CHI TIẾT CÁC THÔNG SỐ, ĐỒ THỊ VÀ TỆP OUTPUT TRONG DỰ ÁN (`graph.md`)

> **Dành cho người mới:** Tài liệu này giải thích từng con số, từng biểu đồ và từng tệp kết quả (`.csv`, `.md`, `.png`) xuất hiện trong [report.md](file:///Users/cuongvuthanh/Documents/B-Free/docs/report.md) và thư mục [outputs/](file:///Users/cuongvuthanh/Documents/B-Free/outputs). Bạn có thể mở tài liệu này song song khi xem ảnh đồ thị hoặc bảng số liệu để hiểu ngay: **đồ thị đó vẽ gì, trục tung/trục hoành là gì, thông số cao hay thấp là tốt, và nhìn vào đó ta kết luận được gì.**

---

## MỤC LỤC

1. [Từ điển Thông số: Hiểu bản chất mọi con số trong bảng và đồ thị](#1-từ-điển-thông-số-hiểu-bản-chất-mọi-con-số-trong-bảng-và-đồ-thị)
2. [Giải thích từng Hình ảnh & Đồ thị trong `report.md` (Hình 1 – Hình 10)](#2-giải-thích-từng-hình-ảnh--đồ-thị-trong-reportmd-hình-1--hình-10)
3. [Giải thích cấu trúc & ý nghĩa các tệp Output trong `outputs/`](#3-giải-thích-cấu-trúc--ý-nghĩa-các-tệp-output-trong-outputs)

---

## 1. Từ điển Thông số: Hiểu bản chất mọi con số trong bảng và đồ thị

### 1.1. Các thông số trong quá trình Huấn luyện (Training Logs)

| Thông số | Tên đầy đủ & Ý nghĩa dễ hiểu | Tốt khi nào? |
| :--- | :--- | :--- |
| **Epoch** | Một lượt mô hình học qua **toàn bộ** tập dữ liệu huấn luyện (ví dụ $102.901$ ảnh ở S1–S2 hoặc $196.501$ ảnh ở S3–S7). | S1–S6 chạy $2$ epoch để sàng lọc nhanh; S7 chạy $6$ epoch để học sâu. |
| **Train Loss / Val Loss** | **Độ sai lệch** khi đoán trên tập học (Train) và tập kiểm tra nội bộ (Val). Công thức đo xem xác suất mô hình đoán lệch bao xa so với nhãn đúng (`0` = Thật, `1` = Giả). | **Càng nhỏ càng tốt**, và `Val Loss` phải giảm đều cùng `Train Loss` (nếu `Train Loss` giảm mà `Val Loss` tăng vọt là bị học vẹt / Overfitting). |
| **Train CE Loss** | Thành phần mất mát phân loại nhị phân chuẩn (Cross-Entropy). | Càng nhỏ càng tốt. |
| **Train DCS Loss** | Thành phần mất mát tương phản (chỉ có ở bậc S4/S6) ép ảnh sạch và ảnh nén JPEG nặng có đặc trưng giống nhau. | Giảm dần theo epoch (ở S6 giảm từ $0,0611 \rightarrow 0,0390$). |

---

### 1.2. Các thông số Đánh giá không phụ thuộc ngưỡng (Threshold-Free Metrics)

Đây là nhóm thông số **công bằng nhất** để so sánh các mô hình AI với nhau, vì chúng đo khả năng **tách biệt 2 nhóm Ảnh Thật và Ảnh Giả** mà không bị ảnh hưởng bởi việc chọn vạch cắt $0,5$ hay $0,3$.

1. **AUC (Area Under the ROC Curve — Diện tích dưới đường cong ROC):**
   - **Ý nghĩa trực quan:** Hãy tưởng tượng bạn bốc ngẫu nhiên **1 ảnh giả** và **1 ảnh thật**. **AUC** chính là xác suất mô hình chấm điểm nghi ngờ cho bức ảnh giả **cao hơn** bức ảnh thật.
   - **Thang điểm:** Từ $0,5$ (đoán mò như tung đồng xu) đến $1,0$ (tách biệt hoàn hảo $100\%$). **Càng gần $1,0$ càng tốt.**
   - **Phân biệt `AUC Pooled` và `AUC Macro`:**
     - **AUC Pooled (Gộp chung):** Gom tất cả ảnh của mọi nền tảng trong 1 bộ dữ liệu (ví dụ gom cả Facebook + Instagram + LinkedIn + X thành $10.000$ ảnh của `ITW-SM`) thành một rổ lớn rồi tính $1$ giá trị AUC duy nhất. Tập nào nhiều ảnh hơn (như Instagram $3.385$ ảnh) sẽ ảnh hưởng mạnh hơn.
     - **AUC Macro (Trung bình đều):** Tính AUC riêng cho từng nền tảng con (Facebook riêng, Instagram riêng, LinkedIn riêng, X riêng) rồi **cộng lại chia đều** (mỗi nền tảng trọng số ngang nhau). Thước đo này giúp không bị che giấu điểm yếu ở các tập con ít ảnh hơn.

2. **AP (Average Precision — Độ chính xác trung bình):**
   - **Ý nghĩa:** Đo diện tích dưới đường cong Precision–Recall — phản ánh độ tin cậy khi mô hình xếp hạng ảnh giả lên đầu danh sách. **Càng gần $1,0$ càng tốt.**

3. **EER (Equal Error Rate — Tỷ lệ lỗi cân bằng):**
   - **Ý nghĩa:** Khi ta dịch chuyển ngưỡng quyết định sao cho tỷ lệ **báo nhầm ảnh thật thành giả (FPR)** đúng bằng tỷ lệ **bỏ sót ảnh giả (FNR)**, thì tỷ lệ lỗi tại điểm cân bằng đó gọi là EER.
   - **Chú ý:** Khác với AUC, **EER càng NHỎ càng tốt** (ví dụ EER $= 0,044$ tức chỉ lỗi $4,4\%$).

4. **Pd@10%FPR (Probability of Detection at 10% False Positive Rate):**
   - **Ý nghĩa:** Trong môi trường thực tế, nếu ta chấp nhận tối đa chỉ được báo nhầm $10\%$ ảnh thật, thì mô hình bắt được bao nhiêu phần trăm ảnh giả? **Càng gần $1,0$ ($100\%$) càng tốt.**

---

### 1.3. Các thông số Đánh giá tại ngưỡng cố định $\tau = 0,5$ (Threshold-Dependent Metrics)

Khi đưa mô hình đi dùng thực tế với quy tắc mặc định *"xác suất $\ge 0,5$ thì kết luận Giả, $< 0,5$ thì kết luận Thật"*, ta đo 3 chỉ số:

1. **TPR (True Positive Rate — Độ nhạy bắt ảnh giả):** Trong $100$ ảnh AI, mô hình bắt đúng được bao nhiêu ảnh? (**Càng gần $100\%$ càng tốt**).
2. **TNR (True Negative Rate — Độ đặc hiệu bảo vệ ảnh thật):** Trong $100$ ảnh chụp thật, mô hình bảo vệ đúng (không vu oan là ảnh AI) được bao nhiêu ảnh? (**Càng gần $100\%$ càng tốt**).
3. **bAcc (Balanced Accuracy — Độ chính xác cân bằng):**
   $$\text{bAcc} = \frac{\text{TPR} + \text{TNR}}{2}$$
   Tại sao dùng `bAcc` thay vì Accuracy thường? Vì nếu một tập test có $90$ ảnh giả và $10$ ảnh thật, mô hình đoán bừa toàn bộ là giả vẫn đạt Accuracy $90\%$ nhưng `bAcc` chỉ có $50\%$. `bAcc` ép mô hình phải giỏi cả 2 lớp.

> [!IMPORTANT]
> **Mẹo đọc bảng cực quan trọng cho người mới:**
> Nếu thấy một mô hình có **AUC rất cao** (ví dụ $0,97$ hoặc $0,93$) nhưng **bAcc @ 0,5 lại thấp** (ví dụ $0,84$ hoặc $0,54$), điều đó **KHÔNG** có nghĩa là mô hình kém! Nó chỉ có nghĩa là **ngưỡng cắt $0,5$ bị lệch** so với điểm chia tối ưu của mô hình đó (ví dụ toàn bộ ảnh thật bị chấm $0,05$, toàn bộ ảnh giả bị chấm $0,35$ — rõ ràng tách biệt $100\%$ nhưng vì đều $< 0,5$ nên ở ngưỡng $0,5$ bị tính là đoán sai ảnh giả).

---

### 1.4. Các thông số Kiểm định Thống kê & Chi phí

| Thông số | Ý nghĩa dễ hiểu | Cách đọc nhanh |
| :--- | :--- | :--- |
| **$\Delta\text{AUC}$** | Hiệu số AUC giữa mô hình của nhóm (`Ours S7`) trừ đi mô hình đối thủ (`ref`). | Dương ($+$) là mình thắng; Âm ($-$) là mình thua. |
| **$[95\%\text{ CI}]$** | **Khoảng tin cậy $95\%$ từ Paired Bootstrap ($1.000$ lần):** Lấy mẫu ngẫu nhiên $1.000$ lần trên cùng tập ảnh để xem $\Delta\text{AUC}$ dao động trong khoảng nào. | Nếu cả hai đầu `[cận dưới, cận trên]` đều $> 0$ (ký hiệu **▲**) $\rightarrow$ **Thắng chắc chắn, không phải ăn may**. Nếu chứa số $0$ (ký hiệu **≈** / `n.s.`) $\rightarrow$ **Hòa về mặt thống kê**. |
| **$p_{\text{boot}}$ & $q_{\text{BH}}$** | Trị số $p$ kiểm định Bootstrap và trị số $q$ sau hiệu chỉnh **Benjamini–Hochberg** (tránh lỗi kiểm định nhiều tập cùng lúc). | **$q_{\text{BH}} < 0,05$** $\rightarrow$ Khác biệt có ý nghĩa thống kê (`ours better`). |
| **GPU-sec / image** | Số giây 1 GPU T4 cần để chấm xong 1 bức ảnh (và **images / GPU-sec** là số ảnh xử lý được trong 1 giây). | Càng nhanh (images/sec càng lớn) càng dễ triển khai thực tế. |

---

## 2. Giải thích từng Hình ảnh & Đồ thị trong `report.md` (Hình 1 – Hình 10)

### 2.1. Nhóm hình minh họa Phương pháp B-Free (Hình 1 – Hình 3)

1. **Hình 1 — [teaser.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/teaser.png):**
   - **Vẽ gì?** Minh họa sự đối lập giữa cách train cũ (ảnh thật và ảnh giả khác hẳn nội dung $\rightarrow$ mô hình học vẹt ngữ nghĩa) và cách train B-Free (căn chỉnh nội dung + mô phỏng nén $\rightarrow$ mô hình nhìn đúng vào dấu vết AI).
2. **Hình 2 — [content_aug.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/content_aug.png):**
   - **Vẽ gì?** Sơ đồ tạo **6 biến thể ảnh giả từ 1 ảnh thật MS-COCO** bằng Stable Diffusion 2.1 (`selfconditioned`, `inpainted_samecat`, `inpainted_diffcat`, có và không giữ nền gốc `_origBG`). Giúp loại bỏ thiên lệch nội dung ngay từ dữ liệu.
3. **Hình 3 — [aug_strategies.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/aug_strategies.png):**
   - **Vẽ gì?** Các chiến lược đưa ảnh có kích thước bất kỳ về ô chuẩn $504 \times 504$ (cắt ngẫu nhiên, đệm viền/ghép mảnh khi ảnh quá nhỏ) mà **không dùng phép co kéo (Resize)** làm méo dấu vết pixel.

---

### 2.2. Nhóm đồ thị Đường cong Huấn luyện (Hình 4, Hình 6, Hình 7)

Các tệp ảnh: [s2_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s2_training_curves.png), [s3_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s3_training_curves.png), [s6_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s6_training_curves.png), và 3 seed của S7 ([s7_seed42_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s7_seed42_training_curves.png), [s7_seed54_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s7_seed54_training_curves.png), [s7_seed68_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s7_seed68_training_curves.png)).

- **Cách đọc chung:**
  - **Trục hoành (X):** Số `Epoch` (1–2 cho S2/S3/S6; 1–6 cho S7).
  - **Đường Train Loss & Val Loss:** Cả hai cùng đi xuống đều đặn qua từng epoch là dấu hiệu mô hình **học thật, không bị Overfitting**.
  - **Đường Val AUC & Val bAcc:** Đi lên tiệm cận trần. Ở **S7 (6 epochs)**, cả 3 seed `42, 54, 68` đều có đồ thị gần như trùng khít nhau, hội tụ về `Val AUC` $= 0,9969 \pm 0,0002$ và `Val bAcc` $= 96,92\%$ — chứng minh thuật toán cực kỳ ổn định, không phụ thuộc may rủi hạt giống ngẫu nhiên.
  - **Điểm cần chú ý ở [s6_training_curves.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/s6_training_curves.png):** Có thêm đường `Train DCS Loss` giảm từ $0,0611 \rightarrow 0,0390$, nhưng thời gian train tăng gấp $1,5$ lần ($2,77$ giờ so với $1,85$ giờ của S3) vì mỗi ảnh phải chạy qua mạng 2 lần (bản sạch và bản nén nặng).

---

### 2.3. Nhóm đồ thị Đối sánh S7 (3 Seeds) vs. B-Free Official (Hình 8 & Hình 9)

1. **Hình 8 — [macro_auc_by_benchmark.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/macro_auc_by_benchmark.png):**
   - **Vẽ gì?** Biểu đồ cột so sánh `AUC Macro` giữa **S7 (cột của nhóm, kèm vạch sai số $\pm\text{std}$ của 3 seed)** và **B-Free Official** trên 3 bộ chuẩn mạng xã hội.
   - **Kết quả nhìn thấy ngay:** Cột của S7 cao hơn hẳn B-Free gốc trên **ITW-SM** ($0,9533$ vs $0,9410$), **WildRF** ($0,9929$ vs $0,9823$) và nhỉnh hơn trên **B-Free Viral** ($0,9736$ vs $0,9700$). Thanh sai số cực ngắn chứng tỏ cả 3 seed chạy ra kết quả rất đồng đều.

2. **Hình 9a — [delta_auc_all.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/delta_auc_all.png):**
   - **Vẽ gì?** Trục đứng thẳng nét đứt là **vạch số $0$** (hòa nhau). Mỗi dòng là 1 nền tảng mạng xã hội (Facebook, Instagram, LinkedIn, X, Reddit, Twitter, B-Free Viral), chấm tròn là mức tăng $\Delta\text{AUC}$ và thanh ngang là khoảng tin cậy $95\%$ Bootstrap.
   - **Cách đọc:**
     - Thanh ngang nằm **hẳn bên phải vạch $0$** $\rightarrow$ S7 **thắng chắc chắn** ($5/8$ tập con: LinkedIn $+0,0376$, Reddit $+0,0264$, ITW-SM Facebook $+0,0065$, WildRF Facebook $+0,0049$, Instagram $+0,0033$).
     - Thanh ngang **cắt ngang vạch $0$** $\rightarrow$ **Hòa thống kê** ($3/8$ tập con: X, Twitter, B-Free Viral). **Không có tập nào S7 bị thua.**

3. **Hình 9b — [auc_dot_all.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/auc_dot_all.png):**
   - **Vẽ gì?** Vẽ riêng điểm AUC của từng seed (`42, 54, 68`) bên cạnh điểm của `B-Free Official`. Cho thấy dù lấy bất kỳ seed nào của S7 thì vẫn vượt B-Free gốc trên các tập khó như LinkedIn và Reddit.

4. **Hình 9c — [roc_pooled.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/roc_pooled.png):**
   - **Vẽ gì?** Đường cong ROC gộp cho 3 bộ chuẩn. Trục hoành là tỷ lệ báo nhầm ảnh thật (FPR), trục tung là tỷ lệ bắt đúng ảnh giả (TPR). **Đường nào ôm sát lên góc trên-bên trái hơn thì mô hình đó mạnh hơn.** Đường của S7 nằm hẳn phía trên B-Free Official trên ITW-SM và WildRF.

5. **Hình 9d — [thr_acc_all.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/thr_acc_all.png):**
   - **Vẽ gì?** Bóc tách hành vi tại ngưỡng $0,5$ qua 3 chỉ số `bAcc`, `TPR (Fake)` và `TNR (Real)`.
   - **Điều thú vị nhìn thấy:** Trên toàn bộ 8 tập con, cột **TNR (Real)** của S7 cao chót vót ($95,9\%–99,9\%$), vượt xa B-Free gốc (đặc biệt trên Reddit $95,9\%$ vs $80,8\%$ và LinkedIn $99,0\%$ vs $88,9\%$). Đổi lại trên B-Free Viral và LinkedIn, cột TPR của S7 thấp hơn một chút do hiệu ứng **lệch ngưỡng thận trọng** (đã giải thích ở Mục 1.3).

---

### 2.4. Nhóm đồ thị Tổng hợp 10 Mô hình cho Slide (Hình 10 & `5_eval_final_merged/figs/`)

Các hình này nằm trong [outputs/5_eval_final_merged/figs/](file:///Users/cuongvuthanh/Documents/B-Free/outputs/5_eval_final_merged/figs) (và bản sao trong [docs/figs/](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/)), so sánh **Ours S7 (3 seeds)** với **B-Free Official** và **8 mô hình hiện đại khác** (`DDA`, `CommFor`, `GlobalForge DINOv3`, `GAPL`, `Co-Spy`, `Effort`, `C2P-CLIP`, `SAFE`) trên $13.288$ ảnh:

1. **[auc_bars_by_benchmark.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/auc_bars_by_benchmark.png):**
   - **Vẽ gì?** Biểu đồ cột so sánh AUC của cả 10 mô hình trên 3 bộ chuẩn.
   - **Kết quả:** S7 đứng **Hạng 1** trên ITW-SM ($0,964$) và WildRF ($0,991$), đứng **Hạng 2** trên B-Free Viral ($0,974$, chỉ sau `DDA` $0,986$). Tính trung bình cả 3 bộ chuẩn, S7 giữ **Hạng 1/10 ($0,976$)**.

2. **[auc_heatmap_subsets.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/auc_heatmap_subsets.png):**
   - **Vẽ gì?** **Bản đồ nhiệt (Heatmap):** Hàng là 10 mô hình, Cột là 8 nền tảng mạng xã hội + cột trung bình. Màu càng đậm/sáng (tùy bảng màu) ứng với AUC càng cao gần $1,0$.
   - **Cách đọc:** Nhìn hàng trên cùng (**Ours S7**) đều màu rực rỡ ở cả 8 cột (từ $0,915$ đến $0,998$), trong khi các hàng dưới cùng (`Effort`, `C2P-CLIP`, `SAFE`) bị nhạt màu ($0,43–0,68$), cho thấy chúng sụp đổ trước ảnh nén/thu phóng mạng xã hội.

3. **[delta_auc_vs_refs.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/delta_auc_vs_refs.png):**
   - **Vẽ gì?** So sánh hiệu số $\Delta\text{AUC} = \text{Ours S7} - \text{Từng đối thủ}$ kèm khoảng tin cậy $95\%$ Bootstrap.
   - **Kết quả:** Trên ITW-SM và WildRF, S7 **thắng cả 9/9 đối thủ** (thanh CI nằm hoàn toàn $> 0$). Trên B-Free Viral, S7 thắng $7/9$, hòa B-Free gốc (`n.s.`), và chỉ thấp hơn nhẹ `DDA` ($-0,012$).

4. **[operating_point_thr0.5.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/operating_point_thr0.5.png):**
   - **Vẽ gì?** Biểu đồ phân tán/thanh cho thấy **TPR (Fake)** và **TNR (Real)** tại ngưỡng cố định $0,5$ của cả 10 mô hình.
   - **Tại sao hình này rất quan trọng để tránh hiểu nhầm?**
     - `GlobalForge (DINOv3)` đạt `TPR = 100%` nhưng `TNR = 7%–12%` (tức gần như bức ảnh nào nó cũng hô là ảnh AI!).
     - `C2P-CLIP` và `SAFE` đạt `TNR = 99%–100%` nhưng `TPR ≈ 0%–35%` (ảnh nào cũng đoán là ảnh thật!).
     - **Chỉ có S7, B-Free Official và DDA** là giữ được cả TPR và TNR ở mức cao đồng đều.

5. **[roc_pooled_10models.png](file:///Users/cuongvuthanh/Documents/B-Free/docs/figs/roc_pooled_10models.png):**
   - **Vẽ gì?** 10 đường cong ROC xếp chồng lên nhau trên 3 bộ chuẩn. Đường của S7 (và DDA trên Viral) nổi bật nhất ở góc trên-trái, trong khi `C2P-CLIP`, `SAFE`, `Effort` nằm sát đường chéo $45^\circ$ (ngang đoán mò).

6. **[auc_vs_cost.png](file:///Users/cuongvuthanh/Documents/B-Free/outputs/5_eval_final_merged/figs/auc_vs_cost.png):**
   - **Vẽ gì?** Biểu đồ đánh đổi giữa **Tốc độ suy luận (GPU-sec/image)** và **Độ chính xác (AUC trung bình)** trên GPU T4. Ví dụ: `CommFor` rất nhanh ($0,004\text{s/ảnh}$, $271\text{ ảnh/s}$) nhưng AUC chỉ $0,934$; `GlobalForge (DINOv3)` rất chậm ($0,322\text{s/ảnh}$, $3,1\text{ ảnh/s}$) mà AUC chỉ $0,892$; `DDA` tốn $0,072\text{s/ảnh}$ ($14\text{ ảnh/s}$) với AUC $0,957$.

---

## 3. Giải thích cấu trúc & ý nghĩa các tệp Output trong `outputs/`

Thư mục [outputs/](file:///Users/cuongvuthanh/Documents/B-Free/outputs) được sắp xếp theo đúng **5 giai đoạn** từ lúc thử nghiệm huấn luyện đến khi tổng hợp bảng cho slide:

| Thư mục | Các tệp quan trọng bên trong | Ý nghĩa & Nội dung chứa đựng |
| :--- | :--- | :--- |
| **`1_training_ablation/`** | `S1_baseline`, `S2_online_aug`, `S3_add_aigenbench`, `S6_dcs_class_aware` (`train_log.csv`, `run_config.yaml`, `training_curves.png`) | Lưu toàn bộ nhật ký, cấu hình và biểu đồ đường cong của **giai đoạn sàng lọc 2 epoch** để kiểm tra từng thay đổi đơn lẻ. |
| **`2_training_s7_multiseed/`** | `S7_seed42/`, `S7_seed54/`, `S7_seed68/` (`train_log.csv`, `training_curves.png`) | Lưu nhật ký **huấn luyện 6 epoch** của công thức tốt nhất (S3) trên 3 hạt giống ngẫu nhiên độc lập để chứng minh tính ổn định. |
| **`3_eval_ablation_s3_vs_s6/`** | `eval_results_s3_seed42.csv`, `eval_results_s6_seed42.csv`, `eval_s3_6.log` | Kết quả **Cổng kiểm định trung gian (Gate Test)**: chứng minh S6 làm tụt $-0,0233$ AUC trên AI-GenBench (và S5 lỗi khóa `gsr.gate`), dẫn đến quyết định chọn S3 lên S7. |
| **`4_eval_s7_multiseed/`** | `s7_final_summary_per_generator.csv`, `eval_results_s7_seed*.csv`, `scores_s7_seed*.parquet` | Chứa **điểm xác suất dự đoán thô của từng bức ảnh** (`.parquet`) và bảng tổng hợp **S7 trên từng bộ sinh trong 36 bộ sinh của AI-GenBench** + 3 tập mạng xã hội. |
| **`5_eval_final_merged/`** | • [slide_notes.md](file:///Users/cuongvuthanh/Documents/B-Free/outputs/5_eval_final_merged/slide_notes.md) & [slide_tables.md](file:///Users/cuongvuthanh/Documents/B-Free/outputs/5_eval_final_merged/slide_tables.md) & `slide_tables.xlsx`<br>• `metrics_pooled.csv`, `metrics_macro.csv`, `metrics_per_subset.csv`<br>• `paired_bootstrap_auc.csv`, `provenance_models.csv`<br>• `figs/` (**16 ảnh PNG**: 6 biểu đồ + 10 ảnh bảng `table_*.png` sẵn sàng dán vào Slide) | Thư mục tổng hợp cuối cùng của notebook [xla-eval-final-merged.ipynb](file:///Users/cuongvuthanh/Documents/B-Free/notebooks/xla-eval-final-merged.ipynb), đối sánh toàn diện **12 mô hình (10 dòng)** với đầy đủ kiểm định Paired Bootstrap và các bảng/biểu đồ sẵn dùng cho bài thuyết trình. |
