import torch
import torch.nn as nn
from torchvision import models

class PneumoniaResNet(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super(PneumoniaResNet, self).__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
