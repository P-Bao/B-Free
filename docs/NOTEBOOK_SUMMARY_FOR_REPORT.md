# Tóm tắt nhiệm vụ theo notebook

Tài liệu này dùng cho người viết báo cáo. Mục tiêu là mô tả vai trò của từng notebook trong pipeline, không khẳng định số liệu cuối cùng.

## Nhánh và nguồn dữ liệu

- Nhánh repo chính xác sử dụng trong tóm tắt này: `B-Free @ integration/train-eval` từ `https://github.com/P-Bao/B-Free.git`, HEAD `130a365`.
- Nhánh `integration/loss-backbone` chỉ là nhánh trong bản notebook cục bộ và không nên dùng làm nhánh hiện hành khi truy cập repo.
- Bộ dữ liệu huấn luyện thêm ở các bậc từ S3 trở đi được tạo bằng [AI-GenBench](https://github.com/MI-BioLab/AI-GenBench).
- Các notebook cục bộ trong `notebooks/` chỉ là bản làm việc. Các notebook Kaggle do bạn liệt kê mới là nguồn cuối để viết báo cáo.

## Kế hoạch và chiến lược

Nội dung dưới đây là phần kế hoạch và chiến lược trích từ `AGENT_PLAN_V5.md`, được tóm gọn lại để người đọc hiểu đúng bối cảnh trước khi mở notebook hoặc code repo.

### 1. Chiến lược chung

Pipeline này đi theo hướng **ablation từng bậc**. Thay vì thử nhiều thay đổi cùng lúc, mỗi bậc chỉ thay đổi một nhóm biến nhất định để có thể giải thích được hiệu ứng.

Các nguyên tắc chính:

- Sửa phép đo trước, cải thiện mô hình sau. Nếu harness chưa công bằng, mọi kết quả so sánh đều không có ý nghĩa.
- Mỗi rung chỉ thay đổi một nhóm biến, phần còn lại giữ nguyên từ cấu hình gốc.
- Không chọn checkpoint hoặc siêu tham số theo tập test.
- Không dùng `bAcc@calib` làm chỉ số chính; số chính là `bAcc@0.5`.
- Không so sánh kết quả cũ khi protocol đánh giá chưa thống nhất.
- Mỗi lần chạy đều phải giữ provenance: `git_sha`, `ckpt_sha256`, `config_hash`, `seed`, `harness_version`.
- Các gate đánh giá được chốt trước khi chạy để tránh dò số.

### 2. Bốn pha kế hoạch

| Pha | Ý nghĩa | Vai trò trong notebook |
|---|---|---|
| Phase 0 | Sửa phép đo và protocol v2. | Thống nhất cách crop, trung bình logit, metric, lưu điểm theo ảnh, và bootstrap CI. |
| Phase 1 | Xây dựng pipeline huấn luyện đúng recipe B-Free. | Chuẩn bị `online_aug`, paired real/fake, CE thuần, đầu CLS, validation augmented. |
| Phase 2 | Chạy thang ablation S1 đến S7. | Đây là phần hiện diện rõ nhất trong các notebook Kaggle training. |
| Phase 3–4 | Chốt gate, chạy S7 nhiều seed, tổng hợp kết quả và viết báo cáo. | Dẫn tới `xla-eval-final-stage-1`. |

### 3. Thang ablation S1 đến S7

| Rung | Thay đổi chính | Mục đích |
|---|---|---|
| S1 | DINOv2 + LoRA, đầu CLS, CE trên ảnh sạch, không module, không AI-GenBench. | Làm baseline sạch. |
| S2 | Thêm online augmentation kiểu B-Free. | Kiểm tra giả thuyết thiếu augmentation là nguyên nhân chính. |
| S3 | Thêm AI-GenBench 25%. | Kiểm tra giá trị của dữ liệu bổ sung trên OOD. |
| S4 | Fine-tune toàn bộ thay vì LoRA. | Trong kế hoạch, nhưng chưa chạy trong chuỗi notebook hiện có. |
| S5 | S3 + LIB/GSR zero-init + đầu `cls_plus_patch`. | Kiểm tra module kiến trúc mà không làm lệch bước khởi tạo. |
| S6 | S3 + DCS class-aware. | Kiểm tra DCS với negative cùng nội dung, không kéo real về fake. |
| S7 | Chạy cấu hình tốt nhất nhiều seed. | Đo kết quả cuối và độ biến động theo seed. |

### 4. Cách đánh giá trong kế hoạch

Hệ thống đánh giá dùng protocol v2:

- Mỗi ảnh được chia thành 5 crop theo lưới patch 504 px.
- Mỗi crop cho một điểm logit, sau đó lấy trung bình logit.
- Điểm này được dùng cho AUC, `bAcc@0.5`, balanced ECE/NLL, `Pd@10%FAR`, EER và `FPR@95%TPR`.
- Kết quả được lưu theo từng ảnh để có thể tính CI và so sánh paired.
- Điểm số không được chọn hoặc hiệu chỉnh theo tập test.

Về mặt chiến lược, các benchmark dùng để kiểm tra khác biệt domain:

- AI-GenBench: kiểm tra khả năng khái quát trên nhiều generator.
- WildRF: kiểm tra dữ liệu ngoài, đa nền tảng.
- SynthWildX: kiểm tra dữ liệu synth có cấu trúc khác.
- B-Free viral images: kiểm tra ảnh lan truyền thực tế.

Trong tóm tắt này, danh sách benchmark cuối cùng vẫn cần lấy từ notebook `xla-eval-final-stage-1`, vì các notebook Kaggle cuối có thể đã loại bỏ hoặc bổ sung dataset so với kế hoạch gốc.

## Trình tự nhiệm vụ

| Notebook | Nhiệm vụ chính | Ghi chú |
|---|---|---|
| [`xla-training-s1`](https://www.kaggle.com/code/baopv051/xla-training-s1) | Huấn luyện bậc nền S1: DINOv2 + LoRA, đầu phân loại CLS, CE trên ảnh sạch, không LIB/GSR/DCS, không thêm AI-GenBench. | Dùng làm baseline sạch trước khi thêm augmentation hoặc dữ liệu ngoài. |
| [`xla-training-s2`](https://www.kaggle.com/code/baopv051/xla-training-s2) | Huấn luyện S2: thêm online augmentation theo recipe B-Free lên trên S1. | Dùng để kiểm tra ảnh hưởng của augmentation trước khi thêm dữ liệu ngoài. |
| [`xla-training-s3`](https://www.kaggle.com/code/baopv051/xla-training-s3) | Huấn luyện S3: giữ augmentation của S2 và bổ sung thêm dữ liệu AI-GenBench. | Đây là bậc bắt đầu dùng AI-GenBench như nguồn huấn luyện bổ sung. |
| [`xla-training-s5`](https://www.kaggle.com/code/baopv051/xla-training-s5) | Huấn luyện S5: trên cơ sở S3, thêm LIB/GSR ở trạng thái zero-init và đầu `cls_plus_patch`. | Kiểm tra giá trị của các module cấu trúc khi chúng khởi tạo không đổi điểm số so với S3. |
| [`xla-training-s6`](https://www.kaggle.com/code/baopv051/xla-training-s6) | Huấn luyện S6: trên cơ sở S3, bật DCS class-aware. | Kiểm tra giá trị của loss contrastive/DCS khi negative được chọn theo lớp. |
| [`xla-training-s7-seed-42`](https://www.kaggle.com/code/baopv051/xla-training-s7-seed-42) | Huấn luyện S7 seed 42 từ cấu hình được chọn sau S3–S6. | Dùng cho kết quả cuối và đo biến động theo seed. |
| [`xla-training-s7-seed-54`](https://www.kaggle.com/code/baopv051/xla-training-s7-seed-54) | Huấn luyện S7 seed 54. | Cùng cấu hình với S7 seed 42, chỉ khác seed. |
| [`xla-training-s7-seed-68`](https://www.kaggle.com/code/baopv051/xla-training-s7-seed-68) | Huấn luyện S7 seed 68. | Cùng cấu hình với S7 seed 42, chỉ khác seed. |
| [`xla-eval-s3-6`](https://www.kaggle.com/code/baopv051/xla-eval-s3-6) | Đánh giá các checkpoint trung gian S3, S5, S6 để so sánh và chọn cấu hình S7. | Đánh giá theo protocol 5-crop + trung bình logit; không dùng checkpoint trung gian làm kết quả cuối. |
| [`xla-eval-final-stage-1`](https://www.kaggle.com/code/baopv051/xla-eval-final-stage-1) | Đánh giá cuối cho giai đoạn 1 sau các rung S7. | Notebook này là nguồn chính cho bảng kết quả cuối và danh sách dataset cuối. |

## Đánh giá và bộ dữ liệu

Do các notebook Kaggle cuối không giống hoàn toàn các notebook cục bộ, phần đánh giá phải ghi theo notebook `xla-eval-final-stage-1` thay vì cố định theo kế hoạch cũ. Các hệ thống đánh giá trong notebook làm việc có nhắc tới AI-GenBench, WildRF, SynthWildX và B-Free viral images, nhưng danh sách bộ dữ liệu cuối cùng có thể đã bị loại bỏ hoặc bổ sung thêm. Khi viết báo cáo, nên mô tả thành hai dòng rõ ràng:

1. Bộ dữ liệu huấn luyện: B-Free data + AI-GenBench ở các bậc S3 trở đi.
2. Bộ dữ liệu đánh giá: lấy trực tiếp từ cấu hình của `xla-eval-final-stage-1`, có ghi rõ những bộ bị loại và những bộ được thêm so với kế hoạch.

Không nên ghi số liệu cuối trước khi đối chiếu đúng nhánh `integration/train-eval`, đúng notebook Kaggle cuối, và đúng danh sách benchmark đã dùng.
