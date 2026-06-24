import torch
from torch import nn


class DGACNN(nn.Module):
    """Small CNN for 3x5x5 DGA-RGB inputs with an explicit embedding interface."""
    def __init__(self, num_classes=7, feature_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * 5 * 5, feature_dim), nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def extract_features(self, x):
        return self.features(x)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features), features
