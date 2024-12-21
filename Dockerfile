# Bước 1: Chọn image Python làm nền tảng
FROM python:3.9-slim

# Bước 2: Cài đặt các thư viện hệ thống cần thiết (nếu có)
# Cài đặt các thư viện như build-essential, git, và các công cụ khác nếu cần thiết cho môi trường
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Bước 3: Tạo thư mục làm việc cho ứng dụng
WORKDIR /app

# Bước 4: Copy file requirements.txt vào container
# Đây là nơi bạn liệt kê tất cả các thư viện Python cần thiết cho dự án
COPY requirements.txt requirements.txt

# Bước 5: Cài đặt các thư viện Python từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bước 6: Copy toàn bộ mã nguồn vào container
COPY . .

# Bước 7: Tiền xử lý dữ liệu (nếu có)
# Chạy một script Python để xử lý dữ liệu trước khi huấn luyện (nếu cần)
# Ví dụ: python src/data/preprocessing.py
RUN python src/data/preprocessing.py

# Bước 8: Huấn luyện mô hình
# Chạy một script Python để huấn luyện mô hình
# Ví dụ: python src/models/train.py
RUN python src/models/train.py

# Bước 9: Expose cổng cho ứng dụng web (nếu sử dụng Gradio, Flask hoặc FastAPI)
EXPOSE 7860

# Bước 10: Chạy ứng dụng khi container bắt đầu
# Ví dụ: Nếu sử dụng Gradio cho giao diện người dùng
CMD ["python", "app.py"]
