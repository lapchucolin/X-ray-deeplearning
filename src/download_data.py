import kagglehub
import shutil
import os
from pathlib import Path

# Credentials from environment or hardcoded (using hardcoded as per previous session, though env is better practice)
os.environ['KAGGLE_USERNAME'] = 'lapchucolin'
os.environ['KAGGLE_KEY'] = 'b086bf04c8b389f641b3a256522adf0f'

def download_and_setup_data():
    print("Downloading dataset with kagglehub...")
    download_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Dataset downloaded to: {download_path}")
    
    target_root = Path("data/raw")
    if target_root.exists(): shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    
    source_path = Path(download_path)
    print(f"Copying files from {source_path} to {target_root}...")
    shutil.copytree(source_path, target_root, dirs_exist_ok=True)
    print("Copy completed.")

if __name__ == "__main__":
    download_and_setup_data()
