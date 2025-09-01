from torchvision import models
import torch.nn as nn

def make_model(name="resnet18", num_classes=10):
    if name == "resnet18":
        model = models.resnet18(weights=None)
    elif name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
