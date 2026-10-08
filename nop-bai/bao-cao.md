# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Thiều Quang Vinh |
| MSSV | 2A202602877 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/zinhcandoit/K4-L3-DAY21-ThieuQuangVinh-2A202602877-CI-CD-for-AI-Systems |
| Ngày nộp | 08/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần 3 đạt F1 cao nhất (0.7149), vượt ngưỡng chất lượng (>= 0.65). Ta ưu tiên lần 3 vì F1 cân bằng tốt giữa Precision và Recall trên lớp thiểu số, dù lần 1 có Accuracy cao hơn (0.8780). Về trade-off: giảm learning rate xuống 0.05 và 50 cây khiến mô hình underfitting (F1 tụt còn 0.6051). Tăng lên 200 cây và độ sâu 5 giúp bắt trọn quan hệ phi tuyến phức tạp.

---

## 2. Vì Sao Quality Gate Đặt Trên F1 Thay Vì Accuracy

Tập Adult mất cân bằng lớp: lớp dương chiếm 24.8%, lớp âm chiếm 75.2%. Nếu mô hình đoán toàn bộ là âm, Accuracy vẫn đạt 75.2% nhưng hoàn toàn vô dụng. F1 lớp dương là trung bình điều hòa giữa Precision và Recall, phản ánh chính xác năng lực phát hiện người thu nhập cao mà không bị lớp đa số lấn át. Tránh dùng `average="weighted"` vì trọng số lớp đa số sẽ thổi phồng chỉ số giả tạo.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Push workflow bị chặn | Token thiếu quyền `workflow` | Tạo PAT mới có chọn quyền `workflow` |
| CI chạy `dvc pull` lỗi 401 | Thiếu `sa-key.json` ở root trên runner | Ghi secret ra `sa-key.json` tại root của runner CI |
| API lỗi khi tải model trên VM | VM cài `scikit-learn 1.7.2` lệch bản 1.4.2 | Cố định `scikit-learn==1.4.2` trên VM đồng bộ với CI |

---

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi tăng mẫu từ 22.361 lên 44.722, F1 tăng từ 0.7149 lên 0.7354 và Accuracy tăng từ 0.8740 lên 0.8820. Thêm dữ liệu giúp mô hình khái quát tốt hơn các mẫu thiểu số. Pipeline CI/CD tự động huấn luyện lại và release thành công khi có commit DVC mới mà không cần thao tác thủ công.

---

## 5. Phần Bonus Đã Thực Hiện

- [x] Bonus 1 - Tracking MLflow DagsHub: Kết nối tracking URI đến server DagsHub của repo (`https://dagshub.com/zinhcandoit/...`).
- [x] Bonus 2 - Điều chỉnh ngưỡng quyết định: Quét ngưỡng 0.1-0.9, ngưỡng tối ưu 0.30 giúp F1 đạt 0.7537 (tăng từ 0.7354).
- [x] Bonus 3 - Báo cáo Precision / Recall: Tự động xuất Confusion Matrix và Classification Report ra `outputs/detail.txt` vào Artifacts.
- [x] Bonus 4 - Rollback an toàn: So sánh F1 trong Quality Gate, hủy Release nếu F1 mới thấp hơn model đang chạy trên GCS.
- [x] Bonus 5 - Cảnh báo Data Drift: Kiểm tra tỷ lệ lớp dương (24.78%), cảnh báo nếu lệch quá 5% so với mốc 24.80%.