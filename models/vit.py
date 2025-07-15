import torch
import torch.nn as nn
from .functions import GRL
import timm

# ==== TODO: Modify the discriminator/classifier architecture ====
# - Try adding BatchNorm, Dropout, more hidden layers, or using LayerNorm for features.
# - Compare the performance of a simple linear head vs a head with dropout/layer norm.
# - Try using a different backbone (e.g. vit_small, vit_largeResNet, EfficientNet, etc.)

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, 2)
        )
    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        return self.net(x)

class ViT_DANN(nn.Module):
    def __init__(self, num_classes=65, pretrained=True):
        super().__init__()
        # self.vit = 
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        feat_dim = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.domain_discriminator = Discriminator(feat_dim)

    def forward(self, x, alpha=0.0):
        features = self.vit(x)
        class_logits = self.classifier(features)
        domain_logits = self.domain_discriminator(features, alpha)
        return class_logits, domain_logits

class ViT(nn.Module):
    def __init__(self, num_classes=65, pretrained=True):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)
        
    def forward(self, x):
        return self.vit(x)
