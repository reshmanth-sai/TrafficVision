import torch
import torch.nn as nn
from torchvision import models

# Empirical optimal multi-label classification thresholds (evaluated on validation split)
# th=0.4 maximizes macro-F1 (0.7307 FT / 0.6387 Scratch) by optimizing minority class recall (bus & motorcycle)
CLASSIFIER_THRESHOLD_FINETUNED = 0.4
CLASSIFIER_THRESHOLD_SCRATCH = 0.4
DEFAULT_CLASSIFIER_THRESHOLD = 0.4

def get_vehicle_classifier(pretrained=True, num_classes=4):
    """
    Constructs a ResNet18 multi-label scene classifier.
    
    Args:
        pretrained (bool): If True, uses ImageNet weights (DEFAULT). If False, random initialization.
        num_classes (int): Number of multi-label classes (default: 4 -> bus, car, motorcycle, truck).
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    in_features = model.fc.in_features
    # Replace final linear layer for 4-class multi-label output logits
    model.fc = nn.Linear(in_features, num_classes)
    
    return model
