import torch
from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import torch.nn.functional as F

class EfficientNet_Embedding(nn.Module):
    def __init__(self, embedding_dim=1792, pretrained=True):
        super(EfficientNet_Embedding, self).__init__()
        
        # Load EfficientNet-B4 backbone
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4') if pretrained else EfficientNet.from_name('efficientnet-b4')
        
        # image size 64x64
        self.efficientnet._conv_stem = nn.Conv2d(
            1, 48, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Change global pooling to fit small images
        self.efficientnet._avg_pooling = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer to reduce dimensionality
        self.fc = nn.Linear(1792, embedding_dim)  # Output of EfficientNet-B4 is 1792

    def forward(self, x):
        # Input x: [batch_size, channels, 64, 64]
        
        # Pass through EfficientNet backbone
        features = self.efficientnet.extract_features(x)  # [batch_size, 1792, height, width]
        
        # Global Average Pooling
        features = F.adaptive_avg_pool2d(features, (1, 1))  # [batch_size, 1792, 1, 1]
        features = features.view(features.size(0), -1)      # Flatten to [batch_size, 1792]
        
        # Fully connected layer for embedding
        embedding = self.fc(features)  # [batch_size, embedding_dim]
        
        return embedding


class ConEfficientNet(nn.Module):
    def __init__(self, embedding_dim=1792, num_classes=100, pretrained=False):
        super(ConEfficientNet, self).__init__()

        # Backbone feature extractor
        self.encoder = EfficientNet_Embedding(embedding_dim=embedding_dim, pretrained=pretrained)

        # Linear classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Feature extraction
        embedding = self.encoder(x)  # [batch_size, embedding_dim]

        # Classification
        logits = self.classifier(embedding)  # [batch_size, num_classes]

        return logits
