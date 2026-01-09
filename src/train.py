import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import ChestXRayDataset
from src.model import PneumoniaResNet
from src.transforms import get_transforms
from src.utils import seed_everything, save_checkpoint, get_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    with open(config_path, 'r') as f: return yaml.safe_load(f)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / total, 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

def main(config_path: str):
    config = load_config(config_path)
    seed_everything(config['train']['seed'])
    device = get_device()
    logger.info(f"Using device: {device}")
    
    processed_path = Path(config['data']['processed_path'])
    train_csv, val_csv = processed_path / 'train.csv', processed_path / 'val.csv'
    
    if not train_csv.exists() or not val_csv.exists():
        logger.error(f"Data splits not found at {processed_path}. Please run src/prepare_splits.py.")
        return

    train_loader = DataLoader(ChestXRayDataset(str(train_csv), transform=get_transforms('train', config['data']['img_size'])), 
                              batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    val_loader = DataLoader(ChestXRayDataset(str(val_csv), transform=get_transforms('val', config['data']['img_size'])), 
                            batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    model = PneumoniaResNet(config['model']['num_classes'], config['model']['pretrained']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    
    best_val_acc = 0.0
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"Epoch [{epoch+1}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_val_acc': best_val_acc}, "best_model.pth")
            logger.info(f"New best model saved with Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()
    main(args.config)
