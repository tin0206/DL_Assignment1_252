import torch.nn as nn
import timm
from torchvision import models

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Load pre-trained ResNet50 weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Replace the final fully connected layer for multi-label classification
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

class ViT(nn.Module):
    def __init__(self, num_classes):
        super(ViT, self).__init__()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.vit(x)

def get_model(model_name, num_classes=10, device="cuda"):
    # MODELS
    # ========================
    if model_name == "resnet50":
        model = CNN(num_classes)
    elif model_name == "vit":
        model = ViT(num_classes)
    else:
        raise ValueError("Model not supported")

    return model.to(device)
