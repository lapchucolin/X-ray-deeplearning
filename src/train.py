import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

def train_one_epoch(model, loader, criterion, optimizer, device, writer, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training", leave=False)
    for i, (images, labels) in enumerate(pbar):
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
        
        # Log step loss
        writer.add_scalar('Batch/Loss', loss.item(), epoch * len(loader) + i)
        
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

def main(config_path: str, dry_run: bool = False):
    config = load_config(config_path)
    if dry_run:
        logger.warning("DRY RUN MODE: Setting epochs=1, batch_size=2, and limiting data.")
        config['train']['epochs'] = 1
        config['data']['batch_size'] = 2

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
    if dry_run:
        # Limit to 10 batches for speed
        train_loader = [x for i, x in enumerate(train_loader) if i < 5]

    val_loader = DataLoader(ChestXRayDataset(str(val_csv), transform=get_transforms('val', config['data']['img_size'])), 
                            batch_size=config['data']['batch_size'], shuffle=False, num_workers=config['data']['num_workers'])
    
    model = PneumoniaResNet(config['model']['num_classes'], config['model']['pretrained']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    writer = SummaryWriter('runs')
    
    best_val_acc = 0.0
    for epoch in range(config['train']['epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, writer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        logger.info(f"Epoch [{epoch+1}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
        
        if val_acc > best_val_acc:

            best_val_acc = val_acc
            save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'best_val_acc': best_val_acc}, "best_model.pth")
            logger.info(f"New best model saved with Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Run a single epoch on a subset for debugging')
    args = parser.parse_args()
    main(args.config, args.dry_run)
