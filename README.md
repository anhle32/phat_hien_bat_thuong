# 🛡️ Ứng dụng Phát Hiện Giao Dịch Bất Thường (Anomaly Detection Web App)

Ứng dụng Web này được chuyển đổi tự động từ Notebook huấn luyện mô hình Machine Learning phát hiện bất thường trong giao dịch tài chính. Hệ thống sử dụng thuật toán học máy không giám sát **Isolation Forest (Rừng cô lập)** để đánh giá mức độ rủi ro và nhận diện hành vi gian lận tiềm ẩn.

## 🚀 Tính Năng Chính Của Ứng Dụng
Ứng dụng được cấu trúc phân vùng giao diện khoa học theo nguyên tắc thiết kế của Streamlit, bao gồm thanh cấu hình và 4 thẻ chức năng chính:
1. **Thanh điều khiển (Sidebar):** Quản lý nạp dữ liệu gốc và tinh chỉnh động các siêu tham số mô hình học máy (`Contamination`, `n_estimators`, `max_features`, `random_state`).
2. **Tab 📊 Tổng quan dữ liệu:** Hiển thị trực quan dữ liệu thô, các chiều kích thước dòng/cột và bảng thống kê mô tả phân phối chuẩn của các biến số.
3. **Tab 📈 Trực quan hóa biến số:** Vẽ các đồ thị động trực quan hóa phân bổ lượng tiền giao dịch, khung thời gian phân tán và tần suất loại giao dịch/kênh thực hiện.
4. **Tab 🎯 Kết quả phát hiện bất thường:** Thống kê định lượng tổng số giao dịch bất thường, biểu đồ phân tích mật độ điểm cô lập, bản đồ phân tán rủi ro và bộ lọc danh sách nghi vấn rủi ro cao.
5. **Tab 🔮 Trực tuyến & Dự báo lô:** Cung cấp 2 chế độ: Nhập tay kiểm thử tức thời 1 giao dịch mới hoặc Tải lên danh sách lớn dạng file để dán nhãn bất thường hàng loạt và xuất file báo cáo.

---

## 🛠️ Hướng Dẫn Cài Đặt Và Vận Hành

### Bước 1: Khởi tạo môi trường ảo và Cài đặt thư viện phụ thuộc
Đảm bảo bạn đã cài đặt Python (phiên bản khuyến nghị `>=3.9`). Chạy lệnh sau trong Terminal/Command Prompt tại thư mục dự án để cài đặt tất cả thư viện cần thiết:

```bash
pip install -r requirements.txt
