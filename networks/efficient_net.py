# networks/efficient_net.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


class EfficientNet_Embedding(nn.Module):
    """
    Encoder EfficientNet-B4 cho ảnh xám 1 kênh.
    - Input kỳ vọng: x ∈ R^{B, 1, 37, 28} (đã chuẩn hoá ở DatasetPrepare)
    - Mặc định resize nội bộ lên 64x64 trước khi trích đặc trưng.
    - Trả về: embedding_dim (mặc định 1792)
    """
    def __init__(self, embedding_dim: int = 1792, pretrained: bool = True,
                 target_size=(64, 64), do_resize: bool = True):
        super().__init__()
        self.target_size = target_size
        self.do_resize = do_resize

        # Load EfficientNet-B4
        self.efficientnet = (EfficientNet.from_pretrained('efficientnet-b4')
                             if pretrained else EfficientNet.from_name('efficientnet-b4'))

        # Sửa stem về 1 kênh (grayscale)
        self.efficientnet._conv_stem = nn.Conv2d(
            in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Pooling thích nghi
        self.efficientnet._avg_pooling = nn.AdaptiveAvgPool2d(1)

        # FC để cố định đầu ra về embedding_dim (B4 có 1792)
        self.fc = nn.Linear(1792, embedding_dim)

        # (tuỳ chọn) giảm downsample quá sớm nếu muốn:
        # self.efficientnet._blocks[0]._depthwise_conv.stride = (1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, 37, 28]
        """
        assert x.dim() == 4 and x.size(1) == 1, f"Expected [B,1,37,28], got {tuple(x.shape)}"

        # Resize lên kích thước ổn cho EfficientNet
        if self.do_resize and self.target_size is not None:
            x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)  # [B,1,64,64]

        # Trích đặc trưng
        feats = self.efficientnet.extract_features(x)      # [B, 1792, h, w]
        feats = F.adaptive_avg_pool2d(feats, (1, 1))       # [B, 1792, 1, 1]
        feats = feats.flatten(1)                           # [B, 1792]

        emb = self.fc(feats)                               # [B, embedding_dim]
        return emb


class ConEfficientNet(nn.Module):
    """
    Encoder + Projection Head cho contrastive.
    - Encoder trả embedding_dim (1792)
    - Head trả feat_dim (128) và chuẩn hoá L2 (cho UniCon/SupCon)
    """
    def __init__(self, embedding_dim=1792, feat_dim=128, head='mlp', pretrained=False,
                 target_size=(64, 64), do_resize=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoder = EfficientNet_Embedding(
            embedding_dim=embedding_dim, pretrained=pretrained,
            target_size=target_size, do_resize=do_resize
        )

        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, feat_dim)
            )
        elif head == 'linear':
            self.head = nn.Linear(embedding_dim, feat_dim)
        else:
            raise NotImplementedError(f"Projection head '{head}' not supported.")

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        x: [B, 1, 37, 28]  (đÃ chuẩn hoá ở DatasetPrepare)
        return_embedding=False -> trả feature 128-D (chuẩn hoá L2) cho contrastive
        return_embedding=True  -> trả embedding 1792-D cho CE (nếu muốn)
        """
        emb = self.encoder(x)  # [B, 1792]
        if return_embedding:
            return emb
        feat = F.normalize(self.head(emb), dim=1)  # [B, 128]
        return feat


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
