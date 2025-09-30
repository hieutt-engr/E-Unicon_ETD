# networks/efficient_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class EfficientNet_Embedding(nn.Module):
    def __init__(self, embedding_dim=1792, pretrained=True):
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0') if pretrained \
                            else EfficientNet.from_name('efficientnet-b0')
        # stem 1-channel
        in_ch = 1
        self.efficientnet._conv_stem = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # GAP
        self.efficientnet._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # map to embedding_dim
        self.fc = nn.Linear(1280, embedding_dim)  # b0 -> 1280, b4 -> 1792

    def forward(self, x):  # x: [B,1,64,64]
        feat = self.efficientnet.extract_features(x)      # [B,C,H',W']
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # [B,1280]
        h = self.fc(feat)                                 # [B,embedding_dim]
        return h

class SmallDecoder(nn.Module):
    """Nhẹ: h(1792) -> x̂(1x64x64)"""
    def __init__(self, in_dim=1792, base_ch=64, out_hw=(64,64)):
        super().__init__()
        self.out_hw = out_hw
        self.fc = nn.Linear(in_dim, base_ch*4*4)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),      # 8x8
            nn.Conv2d(base_ch, base_ch//2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),      # 16x16
            nn.Conv2d(base_ch//2, base_ch//4, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),      # 32x32
            nn.Conv2d(base_ch//4, base_ch//8, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),      # 64x64
            nn.Conv2d(base_ch//8, 1, 3, 1, 1),
        )
    def forward(self, h):
        B = h.size(0)
        x = self.fc(h).view(B, 64, 4, 4)
        x_hat = self.up(x)
        # đảm bảo đúng size
        x_hat = F.interpolate(x_hat, size=self.out_hw, mode='bilinear', align_corners=False)
        return x_hat

class VICRegProjector(nn.Module):
    def __init__(self, in_dim=1792, hid=2048, out=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Linear(hid, out),
        )
    def forward(self, h):
        return self.net(h)

class ConEfficientNet(nn.Module):
    def __init__(self, embedding_dim=1792, feat_dim=128, head='mlp',
                 pretrained=False, use_decoder=True, use_vicreg=True):
        super().__init__()
        self.encoder = EfficientNet_Embedding(embedding_dim=embedding_dim, pretrained=pretrained)

        # projection head (contrastive) -> z
        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, feat_dim)
            )
        elif head == 'linear':
            self.head = nn.Linear(embedding_dim, feat_dim)
        else:
            raise NotImplementedError

        self.use_decoder = use_decoder
        self.use_vicreg  = use_vicreg
        self.decoder = SmallDecoder(in_dim=embedding_dim) if use_decoder else None
        self.vicproj = VICRegProjector(in_dim=embedding_dim)   if use_vicreg  else None

        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim

    # giữ API cũ: trả z (cho phase 2/classifier)
    def forward(self, x):
        h = self.encoder(x)             # [B,emb]
        z = F.normalize(self.head(h), dim=1)  # [B,feat]
        return z

    # API cho phase 1 (LRL): trả thêm h, x_hat, p
    def forward_lrl(self, x):
        h = self.encoder(x)                       # [B,emb]
        z = F.normalize(self.head(h), dim=1)      # [B,feat]
        x_hat = self.decoder(h) if self.use_decoder else None
        p = self.vicproj(h)  if self.use_vicreg  else None
        return z, h, x_hat, p


class LinearClassifier(nn.Module):
    """
    Phân loại tuyến tính.
    - Mặc định: input_dim=128 để dùng projection feature
      (nếu muốn phân loại trên 1792-D thì set input_dim=1792)
    """
    def __init__(self, input_dim=128, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, features: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        if return_embeddings:
            return features
        return self.fc(features)