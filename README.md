Xử Lý Ngôn Ngữ Tự Nhiên (NLP)
Dự án này tập trung vào việc xây dựng các ứng dụng xử lý ngôn ngữ tự nhiên (NLP) bằng Python, bao gồm chatbot, phân tích văn bản và các công cụ hỗ trợ ngôn ngữ tiếng Việt.

Cấu trúc thư mục
ChatBot/ – Mô hình chatbot dựa trên học máy.
app/ – Ứng dụng web sử dụng Flask để triển khai giao diện người dùng.
.XLNNTN/ – Tài nguyên và dữ liệu huấn luyện.

Hướng dẫn cài đặt

Yêu cầu
Python 3.8 trở lên
pip

Các bước cài đặt
1. Sao chép dự án về máy:
    git clone https://github.com/QuocTien004/XuLyNgonNguTuNhien.git
    cd XuLyNgonNguTuNhien
2. Tạo và kích hoạt môi trường ảo (tuỳ chọn):
   python -m venv venv
   source venv/bin/activate
3. Cài đặt các thư viện phụ thuộc:
   pip install -r requirements.txt

Cách sử dụng
Chạy ứng dụng web:
    cd app
    python app.py
Ứng dụng sẽ chạy tại http://localhost:5000.

Sử dụng chatbot:
    cd ChatBot
    python chatbot.py

Công nghệ sử dụng
    Python
    Flask
    NLTK / spaCy / scikit-learn
    HTML / CSS / Jinja2


