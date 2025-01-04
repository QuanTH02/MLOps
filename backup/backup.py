import os
import shutil


def backup_latest_version(
    local_root: str = "./data_version/",
    local_data_path: str = "../data/processed/final_merged.csv",
    local_model_path: str = "../src/models/model_efa",
):
    """
    Sao chép dữ liệu và mô hình từ phiên bản mới nhất trong thư mục `data_version`
    vào các vị trí hiện tại.

    :param local_root: Đường dẫn thư mục gốc chứa các thư mục phiên bản
    :param local_data_path: Đường dẫn file dữ liệu cục bộ hiện tại
    :param local_model_path: Đường dẫn thư mục mô hình cục bộ hiện tại
    """

    version_folders = [
        name
        for name in os.listdir(local_root)
        if os.path.isdir(os.path.join(local_root, name))
    ]

    if not version_folders:
        print("Không có phiên bản nào trong thư mục data_version.")
        return

    version_folders.sort(key=lambda x: int(x.split("_")[1]))
    latest_version_folder = version_folders[-1]

    latest_version_path = os.path.join(local_root, latest_version_folder)
    latest_data_path = os.path.join(latest_version_path, "final_data.csv")
    latest_model_path = os.path.join(latest_version_path, "models")

    if os.path.exists(latest_data_path):
        shutil.copy(latest_data_path, local_data_path)
        print(f"Đã sao chép dữ liệu từ {latest_data_path} vào {local_data_path}")
    else:
        print(f"Không tìm thấy file {latest_data_path}")

    if os.path.exists(latest_model_path) and os.path.isdir(latest_model_path):
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        shutil.copytree(latest_model_path, local_model_path)
        print(f"Đã sao chép mô hình từ {latest_model_path} vào {local_model_path}")
    else:
        print(f"Không tìm thấy thư mục mô hình {latest_model_path}")


if __name__ == "__main__":
    backup_latest_version()
