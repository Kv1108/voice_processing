import os

def ensure_folder_exists(folder_path):
    """
    Ensures a folder exists. Creates it if it doesn't.
    Args:
        folder_path (str): Path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Created folder: {folder_path}")

def get_all_files(folder_path, extension=".wav"):
    """
    Retrieves all files with a specific extension in a folder.
    Args:
        folder_path (str): Folder to search in.
        extension (str): File extension to filter by.
    Returns:
        list: List of file paths with the specified extension.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]
