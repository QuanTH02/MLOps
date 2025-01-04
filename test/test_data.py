import pandas as pd
from constant import DATA_FILE_NAME


def validate_data(file_path: str) -> bool:
    """
    Kiểm tra tính hợp lệ của dữ liệu trong file CSV.
    :param file_path: Đường dẫn đến file CSV.
    :return: True nếu dữ liệu hợp lệ, False nếu không hợp lệ.
    """
    # Đọc file CSV
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Không thể đọc file CSV: {e}")
        return False

    # Danh sách các trường bắt buộc
    required_fields = [
        "movie_name",
        "mpaa",
        "budget",
        "runtime",
        "screens",
        "opening_week",
        "domestic_box_office",
        "ratings",
        "user_vote",
        "country",
        "genres",
        "critic_vote",
        "meta_score",
        "sequel",
        "month",
        "year",
    ]

    # 1. Kiểm tra các trường bắt buộc
    for field in required_fields:
        if field not in data.columns:
            print(f"Thiếu trường bắt buộc: {field}")
            return False

    # 2. Kiểm tra các giá trị rỗng
    if data.isnull().any().any():
        empty_fields = data.columns[data.isnull().any()].tolist()
        print(f"Các trường có giá trị rỗng: {empty_fields}")
        return False

    return True


if __name__ == "__main__":
    file_path = DATA_FILE_NAME
    is_valid = validate_data(file_path)
    if not is_valid:
        print("Dữ liệu không hợp lệ.")
    else:
        print("Dữ liệu hợp lệ.")
