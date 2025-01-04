from test_data import validate_data
from test_model import (
    predict_with_feature_selection_without_opening_week,
    predict_with_feature_selection,
)

from constant import (
    DATA_FILE_NAME,
    MODEL_EFR,
    MODEL_GB_WITHOUT_OPENING_WEEK,
    MODEL_RF_WITHOUT_OPENING_WEEK,
    MODEL_RF,
    MODEL_GB,
)

if __name__ == "__main__":
    file_path = DATA_FILE_NAME
    is_valid = validate_data(file_path)
    if not is_valid:
        print("Dữ liệu không hợp lệ")
    else:
        print("Dữ liệu hợp lệ")

        list_file_name = [MODEL_EFR + MODEL_RF, MODEL_EFR + MODEL_GB]
        list_file_name_without_opening_week = [
            MODEL_EFR + MODEL_RF_WITHOUT_OPENING_WEEK,
            MODEL_EFR + MODEL_GB_WITHOUT_OPENING_WEEK,
        ]
        index = 0
        for file_name in list_file_name:
            if not predict_with_feature_selection(
                file_name,
                1,
                2021,
                "PG-13",
                15000000,
                103,
                3427,
                24727437,
                72082999,
                7.2,
                355000,
                88.32,
                0,
                "Drama Horror Mystery Sci-Fi Thriller",
                "United States",
            ):
                index += 1
        for file_name in list_file_name_without_opening_week:
            if not predict_with_feature_selection_without_opening_week(
                file_name,
                1,
                2021,
                "PG-13",
                15000000,
                103,
                3427,
                355000,
                88.32,
                0,
                "Drama Horror Mystery Sci-Fi Thriller",
                "United States",
            ):
                index += 1

        if index > 0:
            print("Mô hình không hợp lệ")
        else:
            print("Mô hình huấn luyện hợp lệ")
