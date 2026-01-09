import os
import glob
import argparse
import random
import logging
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_patient_id(filename: str) -> str:
    parts = filename.split('_')
    if len(parts) > 1 and parts[0].startswith('person'):
        return parts[0]
    return filename

def get_image_files(data_dir: Path) -> List[Path]:
    extensions = ['*.jpeg', '*.jpg', '*.png']
    files = []
    for ext in extensions:
        files.extend(list(data_dir.rglob(ext)))
    return files

def create_splits(raw_data_path: str, output_path: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    random.seed(seed)
    raw_path = Path(raw_data_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)
    all_files = get_image_files(raw_path)
    
    if not all_files:
        logger.error(f"No images found in {raw_path}.")
        return

    data = []
    for p in all_files:
        label = p.parent.name.upper()
        if label not in ['NORMAL', 'PNEUMONIA']: continue
            
        data.append({
            'path': str(p.resolve()),
            'filename': p.name,
            'label': 0 if label == 'NORMAL' else 1,
            'label_str': label,
            'patient_id': parse_patient_id(p.name)
        })

    df = pd.DataFrame(data)
    unique_patients = df['patient_id'].unique()
    train_patients, test_patients = train_test_split(unique_patients, test_size=test_size, random_state=seed)
    train_patients, val_patients = train_test_split(train_patients, test_size=val_size / (1 - test_size), random_state=seed)
    
    train_df = df[df['patient_id'].isin(train_patients)]
    val_df = df[df['patient_id'].isin(val_patients)]
    test_df = df[df['patient_id'].isin(test_patients)]
    
    train_df.to_csv(out_path / 'train.csv', index=False)
    val_df.to_csv(out_path / 'val.csv', index=False)
    test_df.to_csv(out_path / 'test.csv', index=False)
    logger.info(f"Saved splits to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='data/raw')
    parser.add_argument('--out_path', type=str, default='data/processed')
    args = parser.parse_args()
    create_splits(args.raw_path, args.out_path)
