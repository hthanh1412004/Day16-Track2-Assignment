# Báo cáo Lab 16 — AWS CPU LightGBM

1. Tôi triển khai bài lab trên AWS tại region `us-east-1`, sử dụng Compute Node `t3.micro` (2 vCPU, khoảng 1 GiB RAM). Source dựa trên commit `554e983`, với `cpu_instance_type` được điều chỉnh từ `t3.medium` xuống `t3.micro` (bị lỗi với `t3.medium`, đã hỏi Lab Coach và được đồng ý cho đổi).
2. Dataset Credit Card Fraud Detection có 284.807 dòng, trong đó 492 dòng thuộc lớp fraud. Dữ liệu được chia theo tỷ lệ 60% train, 20% validation và 20% test, tương ứng 170.883/56.962/56.962 dòng, dùng seed 16 và stratify theo nhãn `Class`.
3. Lần đo được ghi lúc 18:32 ngày 03/10/2026 (GMT+7): thời gian đọc dữ liệu là 2,400 giây, thời gian huấn luyện là 3,524 giây và LightGBM dừng sớm tại iteration 68.
4. Trên tập test, mô hình đạt AUC 0,976848; Accuracy 0,999508; F1 0,847826; Precision 0,906977 và Recall 0,795918. AUC được tính từ xác suất dự đoán, còn các chỉ số phân loại dùng ngưỡng 0,5.
5. Latency dự đoán một dòng là 1,178 ms; batch 1.000 dòng mất khoảng 0,003017 giây, tương ứng throughput khoảng 331.403 dòng/giây. Kết quả là median sau warm-up, với 50 lần đo một dòng và 10 lần đo batch.
6. Trong lúc benchmark trên cùng cấu hình, tiến trình Python sử dụng khoảng 69,4% CPU và 21,3% RAM; hệ thống dùng 243 MiB trên tổng 914 MiB. Network ghi nhận RX khoảng 37,86 MB và TX khoảng 1,70 MB tại thời điểm chụp.
7. AWS Billing lúc 21:20 ngày 03/10/2026 ghi nhận mức sử dụng cao nhất theo region là 0,14 USD tại US East (N. Virginia), trong khi estimated grand total và số tiền theo dịch vụ hiện hiển thị 0,00 USD. Tài khoản đang dùng Free Plan nên credit bù chi phí; vì vậy 0,00 USD là số tiền phải trả tại thời điểm chụp, không có nghĩa hạ tầng không phát sinh mức sử dụng.
8. Tôi đã tải `benchmark.py` và `benchmark_result.json` về laptop, chạy `terraform destroy` thành công và xác nhận `terraform state list` không còn tài nguyên. Tôi cũng kiểm tra lại AWS Console để bảo đảm các tài nguyên EC2, NAT Gateway, ALB, Elastic IP và VPC của bài lab đã được xóa.
