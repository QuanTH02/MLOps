### Yêu cầu:
- Python
- Power Shell: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

### Cài đặt thư viện, môi trường:
- py -m venv venv
- .\venv\Scripts\activate
- pip install -r requirements.txt

### Run code:
- cd src/app
- py app.py

## Data sẽ cập nhật mỗi ngày, nhưng nếu bạn muốn crawl data mới và train lại thì:
### Crawl new data:
- cd data/crawl/main
- py main_update.py

### Train model:
- cd src/models
- py feature_selection.py




### Cấu trúc thư mục
project-mlops/
├── requirements.txt        # Danh sách thư viện
├── data/                   # Chứa dữ liệu thô hoặc đã xử lý
│   ├── crawl/              # Thư mục crawl data
│   ├── raw/
│   └── processed/
├── src/                    # Source code chính
│   ├── __init__.py
|   ├── app/                # Nơi để chạy app
    ├── constant/           # Chứa các biến không thay đổi
│   ├── data/               # Tiền xử lý dữ liệu
│   └── models/             # Huấn luyện và lưu mô hình
├── .github/                
│   └── workflows/          # CI/CD cho GitHub Actions
│       └── update.yml          # Workflow chính
└── README.md               # Mô tả dự án