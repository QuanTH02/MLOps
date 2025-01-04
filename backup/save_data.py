import os
import shutil


def save_locally(
    local_data_path: str = "../data/processed/final_merged.csv",
    local_model_path: str = "../src/models/model_efa",
    local_root: str = "./data_version/",
):
    """
    Lưu file final_data.csv và thư mục models vào thư mục version_{index} trong thư mục cục bộ.
    Số phiên bản {index} được tính bằng cách đếm số thư mục hiện tại trong data_version.

    :param local_data_path: Đường dẫn file dữ liệu cục bộ
    :param local_model_path: Đường dẫn thư mục model cục bộ
    :param local_root: Đường dẫn thư mục gốc cục bộ
    """
    version_count = len(
        [
            name
            for name in os.listdir(local_root)
            if os.path.isdir(os.path.join(local_root, name))
        ]
    )

    version_index = version_count + 1
    version_folder = os.path.join(local_root, f"version_{version_index}")

    os.makedirs(version_folder, exist_ok=True)

    if os.path.exists(local_data_path):
        shutil.copy(local_data_path, os.path.join(version_folder, "final_data.csv"))
        print(f"Đã lưu {local_data_path} vào {version_folder}")
    else:
        print(f"Không tìm thấy {local_data_path}")

    if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
        dest_model_path = os.path.join(version_folder, "models")
        if os.path.exists(dest_model_path):
            shutil.rmtree(dest_model_path)
        shutil.copytree(local_model_path, dest_model_path)
        print(f"Đã lưu thư mục {local_model_path} vào {version_folder}")
    else:
        print(f"Không tìm thấy thư mục {local_model_path}")


if __name__ == "__main__":
    save_locally()
