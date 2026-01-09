import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import logging
import yaml
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.dataset import ChestXRayDataset
from src.model import PneumoniaResNet
from src.transforms import get_transforms
from src.utils import load_checkpoint, get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def main(config_path: str, checkpoint_path: str):
    with open(config_path, 'r') as f: config = yaml.safe_load(f)
    device = get_device()
    processed_path = Path(config['data']['processed_path'])
    test_csv = processed_path / 'test.csv'
    
    if not test_csv.exists():
        logger.error(f"Test split not found at {test_csv}.")
        return

    test_loader = DataLoader(
        ChestXRayDataset(str(test_csv), transform=get_transforms('test', config['data']['img_size'])),
        batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers']
    )
    
    model = PneumoniaResNet(config['model']['num_classes'], config['model']['pretrained']).to(device)
    if not checkpoint_path: checkpoint_path = "best_model.pth"
    try:
        load_checkpoint(checkpoint_path, model)
    except FileNotFoundError:
        logger.error("No checkpoint found.")
        return

    y_true, y_pred = evaluate(model, test_loader, device)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"F1: {f1_score(y_true, y_pred):.4f}")
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='')
    args = parser.parse_args()
    main(args.config, args.checkpoint)
