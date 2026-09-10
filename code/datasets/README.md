## Tập dữ liệu

Thông tin chi tiết hơn tại [trang dự án của B-Free](https://grip-unina.github.io/B-Free) và trong [bài báo](https://arxiv.org/abs/2412.17671).

Bộ dữ liệu tại liên kết trên chứa các thư mục và tệp sau:
- **COCO_real_512**: hình ảnh thực từ COCO (ảnh cắt lớn nhất ở trung tâm được điều chỉnh kích thước thành 512x512)
- **SD2.1_selfconditioned**: hình ảnh tự điều kiện
- **SD2.1_selfconditioned_origBG**: hình ảnh tự điều chỉnh với nền gốc được khôi phục
- **SD2.1_inpainted_samecat**: một đối tượng được thay thế bằng một đối tượng cùng loại.
- **SD2.1_inpainted_samecat_origBG**: tương tự như *inpainted_samecat*, nhưng đã khôi phục lại hình nền gốc.
- **SD2.1_inpainted_diffcat**: một đối tượng được thay thế bằng một đối tượng thuộc danh mục khác.
- **SD2.1_inpainted_diffcat_origBG**: tương tự như *inpainted_diffcat*, nhưng đã khôi phục lại hình nền gốc.
- **mask**: mặt nạ đối tượng được sử dụng để tạo ra *inpainted_samecat* và để thay thế nền cho *inpainted_samecat_origBG* và *selfconditioned_origBG*
- **bbox**: các hộp giới hạn được sử dụng để tạo ra *inpainted_diffcat* và để thay thế nền cho *inpainted_diffcat_origBG*
- **train_list.csv**: chứa ID cho tập dữ liệu huấn luyện và thông tin bổ sung cho mỗi hình ảnh, chẳng hạn như danh mục đối tượng.
- **valid_list.csv**: chứa ID cho tập dữ liệu kiểm định và thông tin bổ sung cho mỗi hình ảnh, chẳng hạn như danh mục đối tượng.

Lưu ý rằng cả tập dữ liệu *huấn luyện* và *kiểm tra* đều được chia từ **tập dữ liệu huấn luyện** MS-COCO 2017.


## md5sum

- 41741e8e81e61455d01455452fcf15ae COCO_real_512.zip:
- a96855ced782ce09b6139ba68af94dd0 SD2.1_selfcondition.zip:
- b2e8530fa6904bc25e6fe6f1ed45149d SD2.1_selfcondition_origBG.zip:
- fc3ba27bc97d60bcf18ca57931aebd3b SD2.1_inpaint_samecat.zip:
- 466fed55a78dacc1f82bd95f6eb7cd5c SD2.1_inpaint_samecat_origBG.zip:
- 5b00dc28d18809c2efc943c89c3f5514 SD2.1_inpainted_diffcat.zip:
- d289c6b0fe4b19f229f4578e61ca8032 SD2.1_inpainted_diffcat_origBG.zip:
- 0294cb1611ad1c50af72d6cace3492ca masks_and_bbox.zip
- 06581da53bf81741501396d71eb7ec5f train_list.csv:
- 72e7b4be2d049029dbedca7e75af91b0 valid_list.csv:


## Bibtex

Nếu bạn sử dụng bộ dữ liệu này, vui lòng trích dẫn nguồn:

```
@inproceedings{Guillaro2025biasfree,
  Tiêu đề={Một mô hình huấn luyện không thiên vị cho việc phát hiện hình ảnh do AI tạo ra một cách tổng quát hơn},
  tác giả={Guillaro, Fabrizio và Zingarini, Giada và Usman, Ben và Sud, Avneesh và Cozzolino, Davide và Verdoliva, Luisa},
  booktitle={Hội nghị IEEE/CVF về Thị giác máy tính và Nhận dạng mẫu (CVPR)},
  năm={2025}
}
```


## Giấy phép

Bản quyền (c) 2025 Nhóm Nghiên cứu Xử lý Hình ảnh thuộc Đại học Federico II của Naples ('GRIP-UNINA').

Bản quyền đã được bảo lưu.

Phần mềm này chỉ được sử dụng, sao chép và chỉnh sửa cho mục đích thông tin và phi lợi nhuận.

Bằng cách tải xuống và/hoặc sử dụng bất kỳ tệp nào trong số này, bạn ngầm đồng ý với tất cả các điều khoản.
các điều khoản của giấy phép, như được quy định trong tài liệu LICENSE.txt
(Đã bao gồm trong gói sản phẩm này)