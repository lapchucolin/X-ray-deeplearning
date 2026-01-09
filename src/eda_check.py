import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

def run_eda():
    csv_path = 'data/processed/train.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Total Training Images: {len(df)}")
    print(df['label_str'].value_counts())

    try:
        samples = pd.concat([
            df[df['label_str'] == 'NORMAL'].sample(8, random_state=42),
            df[df['label_str'] == 'PNEUMONIA'].sample(8, random_state=42)
        ]).reset_index(drop=True)
    except:
        print("Not enough data to sample.")
        return

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(samples):
            try:
                img = Image.open(samples.iloc[i]['path']).convert('RGB')
                ax.imshow(img, cmap='gray')
                ax.set_title(samples.iloc[i]['label_str'])
                ax.axis('off')
            except: pass
                
    plt.tight_layout()
    plt.savefig('notebooks/eda_grid.png')
    print("Saved notebooks/eda_grid.png")

if __name__ == "__main__":
    run_eda()
