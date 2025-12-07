import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ==========================================
# 1. ENCODER (EfficientNet-V2-S)
# ==========================================
class EfficientNet_Embedding(nn.Module):
    def __init__(self, embedding_dim=1792, pretrained=True):
        super().__init__()
        
        # Load V2-S từ torchvision
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_v2_s(weights=weights)
        
        # V2-S mặc định nhận 3 channels -> Tốt, không cần sửa input layer
        
        # Thay thế Classifier cuối cùng bằng Identity để lấy feature vector
        # V2-S feature dim trước classifier là 1280
        self.feature_dim = self.model.classifier[1].in_features # 1280
        self.model.classifier = nn.Identity()
        
        # Projection layer: 1280 -> embedding_dim (1792)
        self.fc = nn.Linear(self.feature_dim, embedding_dim)

    def forward(self, x):  
        # x: [B, 3, H, W]
        # EfficientNet V2 extract features
        # Lưu ý: torchvision model forward trả về logit nếu có classifier,
        # nhưng ta đã gán classifier = Identity nên nó trả về feature [B, 1280]
        feat = self.model(x) 
        h = self.fc(feat) # [B, embedding_dim]
        return h

# ==========================================
# 2. DECODER (Hỗ trợ Flexible Size)
# ==========================================
class SmallDecoder(nn.Module):
    def __init__(self, in_dim=1792, base_ch=64, out_hw=(128,128)):
        super().__init__()
        self.out_hw = out_hw
        
        # Input: [B, 1792] -> Reshape thành [B, 64, 8, 8] (Tăng spatial size ban đầu)
        self.map_size = 8 
        self.fc = nn.Linear(in_dim, base_ch * self.map_size * self.map_size)
        
        self.up = nn.Sequential(
            # 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch, base_ch//2, 3, 1, 1), nn.ReLU(inplace=True),
            
            # 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch//2, base_ch//4, 3, 1, 1), nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_ch//4, base_ch//8, 3, 1, 1), nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128 (Thêm 1 lớp upsample nữa cho ảnh to)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            # Output layer: 3 Channels (RGB)
            nn.Conv2d(base_ch//8, 3, 3, 1, 1), 
        )
        
    def forward(self, h):
        B = h.size(0)
        x = self.fc(h).view(B, 64, self.map_size, self.map_size)
        x_hat = self.up(x)
        # Đảm bảo khớp size cuối cùng (kể cả nếu out_hw khác 128)
        if x_hat.shape[-2:] != self.out_hw:
            x_hat = F.interpolate(x_hat, size=self.out_hw, mode='bilinear', align_corners=False)
        return x_hat

# ==========================================
# 3. VICREG PROJECTOR (Giữ nguyên)
# ==========================================
class VICRegProjector(nn.Module):
    def __init__(self, in_dim=1792, hid=2048, out=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.BatchNorm1d(hid), nn.ReLU(inplace=True),
            nn.Linear(hid, out),
        )
    def forward(self, h):
        return self.net(h)

# ==========================================
# 4. MAIN MODEL (ConEfficientNet - V2)
# ==========================================
class ConEfficientNet(nn.Module):
    def __init__(self, embedding_dim=1792, feat_dim=128, head='mlp',
                 pretrained=False, use_decoder=True, use_vicreg=True,
                 num_prototypes=1000, target_size=(128, 128)): # [NEW] Thêm tham số size
        super().__init__()
        
        self.encoder = EfficientNet_Embedding(embedding_dim=embedding_dim, pretrained=pretrained)

        if head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, feat_dim)
            )
        elif head == 'linear':
            self.head = nn.Linear(embedding_dim, feat_dim)
        
        self.use_decoder = use_decoder
        self.use_vicreg  = use_vicreg
        
        # Truyền target_size vào Decoder để nó biết đường Reconstruct
        self.decoder = SmallDecoder(in_dim=embedding_dim, out_hw=target_size) if use_decoder else None
        self.vicproj = VICRegProjector(in_dim=embedding_dim) if use_vicreg else None

        self.feat_dim = feat_dim
        self.embedding_dim = embedding_dim

        self.prototypes = nn.Linear(feat_dim, num_prototypes, bias=False)

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.weight.data.copy_(w)

    def forward(self, x):
        h = self.encoder(x)             
        z = F.normalize(self.head(h), dim=1)
        return z

    def forward_lrl(self, x):
        h = self.encoder(x)                       
        z = F.normalize(self.head(h), dim=1)      
        x_hat = self.decoder(h) if self.use_decoder else None
        p = self.vicproj(h) if self.use_vicreg else None
        return z, h, x_hat, p

# ==========================================
# 5. LINEAR CLASSIFIER (Giữ nguyên)
# ==========================================
# class LinearClassifier(nn.Module):
#     def __init__(self, input_dim=128, num_classes=7):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, num_classes)

#     def forward(self, features):
#         return self.fc(features)
    

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=7):
        super().__init__()
        # [MODIFIED] Chuyển từ Linear sang MLP 2 lớp
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), # Chống overfitting
            nn.Linear(256, num_classes)
        )

    def forward(self, features):
        return self.net(features)