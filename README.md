### Yêu cầu:
- Python

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