import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Optional, Callable

class ChestXRayDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: Optional[str] = None, transform: Optional[Callable] = None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir) if root_dir else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data_frame)

    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data_frame.iloc[idx]['path']
        if self.root_dir:
            full_img_path = self.root_dir / img_path
        else:
            full_img_path = Path(img_path)

        try:
            image = Image.open(full_img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image at {full_img_path}. Error: {e}")

        label = int(self.data_frame.iloc[idx]['label'])

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
