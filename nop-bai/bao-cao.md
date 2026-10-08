# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Thiều Quang Vinh |
| MSSV | 2A202602877 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/zinhcandoit/K4-L3-DAY21-ThieuQuangVinh-2A202602877-CI-CD-for-AI-Systems |
| Ngày nộp | 07/10/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ siêu tham số ở lần chạy thứ 3 được lựa chọn vì đạt điểm F1-score cao nhất (0.7149), vượt qua ngưỡng chất lượng yêu cầu (>= 0.65) và tối ưu hóa khả năng dự đoán lớp thiểu số. Điều đáng chú ý là lần chạy có accuracy cao nhất lại là lần 1 (0.8780) chứ không phải lần 3 (0.8740). Sự sai lệch này minh chứng rằng accuracy phản ánh chủ yếu độ chính xác của lớp đa số, trong khi F1-score đánh giá sự hài hòa thực sự giữa precision và recall trên lớp mục tiêu. Giữa `n_estimators` và `learning_rate` tồn tại mối quan hệ đánh đổi chặt chẽ: khi giảm `learning_rate` xuống 0.05 và chỉ dùng 50 cây ở lần 2, mô hình bị underfitting nghiêm trọng khiến F1 giảm mạnh xuống 0.6051. Do đó, việc duy trì `learning_rate=0.1` đồng thời tăng số cây lên 200 và độ sâu lên 5 cho phép mô hình tích lũy đủ các bộ phân loại yếu để bắt trọn các đặc trưng phi tuyến tính phức tạp.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult mang đặc trưng mất cân bằng lớp rõ rệt khi lớp dương (thu nhập > 50K) chỉ chiếm khoảng 24.8%, trong khi lớp âm (thu nhập <= 50K) chiếm tới 75.2%. Nếu một mô hình ngây thơ luôn dự đoán nhãn là thu nhập thấp cho mọi trường hợp, độ chính xác (accuracy) của nó vẫn đạt tới 75.2% trên toàn bộ tập dữ liệu và thậm chí đạt gần 87% trên tập holdout, tạo ra ảo tưởng về một mô hình có hiệu năng cao nhưng thực tế lại hoàn toàn vô dụng. Ngược lại, chỉ số F1 của lớp dương là trung bình điều hòa giữa Precision và Recall, đo lường chính xác năng lực phát hiện đúng các cá nhân có thu nhập cao mà không bị áp đảo bởi số lượng lớn của lớp âm. Hơn nữa, ta tuyệt đối không sử dụng average="weighted" hay average="macro" khi tính F1 vì trọng số của lớp đa số sẽ kéo điểm số lên cao giả tạo, làm mất đi tính nghiêm ngặt của ngưỡng kiểm định chất lượng (Quality Gate) trong pipeline CI/CD.

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lỗi `ModuleNotFoundError: No module named 'pkg_resources'` khi chạy train | Phiên bản `setuptools>=82` trên Python 3.12 đã gỡ bỏ hoàn toàn module `pkg_resources` mà thư viện MLflow cần dùng | Hạ phiên bản `setuptools<82` hoặc nâng cấp MLflow lên phiên bản mới không còn phụ thuộc `pkg_resources` |
| Lỗi `ImportError` liên quan đến `FallbackAsyncAdaptedQueuePool` trong MLflow | Phiên bản MLflow cũ gọi import lớp kết nối không tồn tại trong cấu trúc thư viện SQLAlchemy hiện hành | Nâng cấp đồng bộ gói thư viện `mlflow` và `sqlalchemy` lên phiên bản tương thích mới nhất |
| Lỗi `UntrustedTypesFoundException` do `sklearn.tree._tree.Tree` bị chặn khi log model | MLflow mặc định sử dụng `skops` để serialize model nhằm ngăn chặn rủi ro bảo mật từ pickle và chặn kiểu cây | Thêm tham số `skops_trusted_types=["sklearn.tree._tree.Tree"]` vào lời gọi hàm `mlflow.sklearn.log_model` |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** Khi bổ sung thêm dữ liệu `train_batch2`, chỉ số F1 và Accuracy có thể chỉ biến động nhẹ do hai tập dữ liệu được lấy mẫu ngẫu nhiên từ cùng một phân phối gốc và mô hình đã bão hòa tri thức từ 22.361 mẫu đầu tiên. Điều cốt lõi mà Bước 3 chứng minh thành công là tính tự động hóa khép kín của pipeline MLOps: hệ thống tự động kích hoạt huấn luyện lại và kiểm định chất lượng ngay khi phát hiện thay đổi dữ liệu từ commit DVC.

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___
