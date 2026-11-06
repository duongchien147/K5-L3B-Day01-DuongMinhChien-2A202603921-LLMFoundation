# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603921
- **Họ và tên:** Dương Minh Chiến
- **SEED** (in ở ô đầu notebook): 7921

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương pháp | Accuracy | Precision | Recall | F1 test | F1 SUB (40 tin) | Train (s) | Inference cả lô | Từng tin p50 | Từng tin p95 | USD / 1k tin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **1. BoW + LogReg (tự code)** | 0.974 | 0.974 | 0.806 | 0.882 | 0.800 | 8.9s | 10,994 tin/s | 0.07 ms | 0.23 ms | $0.00000 |
| **2. BoW + MLP (deep learning)** | 0.983 | 0.988 | 0.871 | 0.926 | 0.865 | 8.3s | 15,291 tin/s | 0.31 ms | 2.51 ms | $0.00000 |
| **3a. Encoder (MiniLM) + kNN** | 0.977 | 0.879 | 0.935 | 0.906 | 0.897 | 73.8s | 39 tin/s | 43.36 ms | 63.69 ms | $0.00000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.982 | 0.944 | 0.903 | 0.923 | 0.914 | 74.2s | 51 tin/s | 37.01 ms | 54.96 ms | $0.00000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.987 | 0.946 | 0.946 | 0.946 | 0.921 | 75.2s | 41 tin/s | 43.24 ms | 86.41 ms | $0.00000 |
| **4a. LLM nhỏ zero-shot (Groq/gpt-oss-20b)** | 0.900 | 0.700 | 0.700 | 0.889 | 0.889 | 0.0s | 0.46 tin/s | 360.50 ms | 2148.85 ms | $0.00000 |
| **4b. LLM lớn zero-shot (Groq/gpt-oss-120b)** | 0.925 | 0.870 | 1.000 | 0.930 | 0.930 | 0.0s | 0.45 tin/s | 381.57 ms | 1724.24 ms | $0.00000 |

## 3. Dự đoán và thực tế

1. **Dự đoán 1.A (MLP/LogReg F1):** Dự đoán F1 khoảng 0.80–0.90, Precision cao hơn Recall. Thực tế đạt F1 = 0.882 (BoW + LogReg) và 0.926 (MLP). Lý do lệch: Mô hình nhận diện khá tốt các từ đặc trưng spam nhưng một số tin spam né từ khóa khiến recall bị thấp hơn (0.806).
2. **Dự đoán 1.B (Learning Rate):** Với `lr=0.001` loss giảm rất chậm; `lr=0.5` giảm ổn định; `lr=50` loss ban đầu tăng vọt do bước cập nhật quá lớn nhưng sau đó vẫn hội tụ. Thực tế khớp một phần, riêng `lr=50` số liệu ghi nhận loss sau giai đoạn sốc ban đầu vẫn giảm sâu.
3. **Dự đoán 4.A (Model lớn vs nhỏ LLM):** Dự đoán model lớn (`gpt-oss-120b`) đạt F1 cao hơn model nhỏ (`gpt-oss-20b`) nhờ khả năng hiểu ngôn ngữ tốt hơn. Thực tế kiểm chứng mô hình lớn đạt F1 = 0.930 so với 0.889 của model nhỏ.

## 4. Phân tích 50 case khó

- **Model hụt nhiều nhất:** Mô hình Bag-of-Words (Phần 1) là mô hình suy giảm hiệu năng nhiều nhất ở nhóm case khó (các tin nhắn lách từ khóa, viết tắt hoặc có ngữ cảnh đánh lừa chứa từ thông thường).
- **2 case cụ thể & cơ chế sai:**
  1. *Case 1 (Thật = spam | Đoán = ham):* `How come it takes so little time for a child who is afraid of the dark...` -> Cơ chế sai của BoW: Tin nhắn mang tính chất câu hỏi triết lý nhưng thực chất là mồi nhử spam, chứa nhiều từ kéo về HAM mạnh nhất trong từ điển (`it`, `me`, `come`, `how`) khiến vector trọng số cộng dồn và ép mô hình đưa ra nhãn ham.
  2. *Case 2 (Thật = spam | Đoán = ham):* `Email Alert From: Jeri Stewart Size: 2KB Subject: Low-cost prescription drugs...` -> Cơ chế sai: Tin nhắn ngụy trang dưới dạng thông báo email kỹ thuật số hợp lệ, đánh lừa các mô hình chỉ dựa vào từ đơn lẻ thay vì cấu trúc ngữ nghĩa toàn cục.

## 5. Thử thách cá nhân

- **SEED:** `7921`
- **Cách làm & Kết quả:** Xây dựng thành công các pipeline tự code bằng numpy và pytorch, thực hiện trơn tru các phần từ trích xuất đặc trưng Bag-of-Words, huấn luyện mạng nơ-ron qua PyTorch (`build_mlp`), tích hợp Sentence-Transformers (`all-MiniLM-L6-v2`) để trích xuất embedding không gian 384 chiều, kết hợp kNN thủ công và gọi API qua Groq cho các tác vụ LLM zero-shot/few-shot.
- **Giải thích:** Việc kết hợp giữa biểu diễn embedding ngữ nghĩa và tầng phân loại (LogReg/MLP head) giúp cải thiện đáng kể độ chính xác (F1 tăng từ 0.882 lên tới 0.946 ở mô hình 3c) vì nắm bắt được ngữ cảnh câu tốt hơn so với việc chỉ đếm tần suất từ đơn lẻ.

## 6. Phương án cho sản phẩm thật

- **Yêu cầu bài toán thực tế:** p95 nhỏ hơn 50 ms/tin cho 10 triệu SMS/ngày.
- **Phương án lựa chọn:** Sử dụng phương án **3b hoặc 3c (Encoder MiniLM + Linear/MLP head)** hoặc chuyển hóa mô hình sang dạng distilled nhẹ (như DistilBERT / TinyBERT chạy tối ưu bằng ONNX Runtime / OpenVINO).
- **Dẫn số liệu thực tế từ lab:** 
  - Mô hình Embedding + LogReg (3b) đạt độ trễ p50 = 37.01 ms và p95 = 54.96 ms trên phần cứng thử nghiệm đơn luồng, với F1 rất cao (0.923). Khi tối ưu bằng ONNX/TensorRT hoặc chạy batch song song trên cụm GPU/CPU server, thời gian đáp ứng p95 hoàn toàn có thể hạ xuống dưới mức 20 ms.
- **Chấp nhận đánh đổi:** 
  - Hy sinh một phần nhỏ độ phức tạp về tài nguyên hệ thống (cần thời gian encode ban đầu và lưu trữ mô hình embedding khoảng ~90MB) để đổi lấy độ chính xác (F1 cao vượt trội 0.923 – 0.946) và chi phí vận hành bằng 0 ($0.00 USD/1k tin) so với việc dùng các LLM qua API (vừa đắt đỏ, vừa có độ trễ p95 cao từ hàng trăm ms đến hàng giây).

## 7. AI log và điều đã học

Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.

- **Lần 1 (Phần 1 - Thử thách cá nhân #2 về kích thước Vocabulary):**
  - *Hỏi AI:* Cách thực hiện thử thách cá nhân #2 (thay đổi `max_size` của `build_vocab` lần lượt với các giá trị 100, 2000, 5000) và giải thích xu hướng thay đổi F1-score.
  - *Nhận được:* Đoạn mã nguồn mẫu lặp qua các kích thước từ điển, huấn luyện mô hình Logistic Regression và đánh giá F1-score trên tập test, kèm theo khung nhận xét chi tiết về tác động của số lượng đặc trưng.
  - *Sửa/Áp dụng:* Đưa mã lệnh vào ô Code thử thách của bài lab, chạy thực nghiệm với `max_size=100` (F1 = 0.8075), `max_size=2000` (F1 = 0.8824) và `max_size=5000` (F1 = 0.8690).
  - *Điều hiểu ra:* Tăng kích thước Vocabulary không đồng nghĩa với việc F1-score sẽ tăng vô hạn. Quá ít từ (100 từ) khiến mô hình thiếu đặc trưng quan trọng để nhận diện spam, trong khi quá nhiều từ (5000 từ) có thể đưa vào các từ hiếm hoặc nhiễu, làm giảm khả năng tổng quát hóa của mô hình trên tập test.

- **Lần 2 (Phần 4 & Tổng hợp - Xử lý bài toán phân loại và đánh giá phương án sản phẩm):**
  - *Hỏi AI:* Cách xây dựng prompt phân loại tin nhắn spam/ham và cấu trúc gọi API qua Groq, cũng như các tiêu chí lựa chọn mô hình cho bài toán thực tế (10 triệu SMS/ngày yêu cầu p95 nhỏ hơn 50ms).
  - *Nhận được:* Mẫu prompt zero-shot chuẩn, cách xử lý kết quả trả về của LLM (`parse_label`) và phương pháp phân tích đánh đổi độ trễ, chi phí, độ chính xác cho hệ thống thực tế.
  - *Sửa/Áp dụng:* Áp dụng cấu hình gọi API Groq cho tập SUB và phân tích phương án kiến trúc router kết hợp giữa mô hình nhẹ (Embedding + LogReg) và LLM cho các ca khó.
  - *Điều hiểu ra:* Các mô hình LLM lớn có khả năng hiểu ngữ nghĩa zero-shot rất tốt nhưng đi kèm với chi phí và độ trễ cao, không phù hợp cho các hệ thống thời gian thực quy mô lớn (như lọc hàng triệu tin nhắn mỗi ngày). Do đó, các giải pháp kết hợp embedding nhẹ kết hợp với bộ phân tuyến (router) là lựa chọn tối ưu để cân bằng giữa chi phí, tốc độ và độ chính xác.
