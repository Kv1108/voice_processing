import os

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def get_all_files(folder_path, extension=".wav"):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]
