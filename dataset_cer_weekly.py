import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------- 0) Chuẩn hoá Global ----------
def fit_global_hour_stats(X_benign_train: np.ndarray):
    X = np.asarray(X_benign_train, dtype=np.float32)
    # Tính mean/std toàn cục (scalar) để giữ nguyên biên độ tương đối giữa các giờ
    mu = X.mean().astype(np.float32)
    sd = X.std().astype(np.float32)
    return mu, sd

def apply_global_hour_norm(x_seq: np.ndarray, mu: float, sd: float):
    x = np.asarray(x_seq, np.float32)
    # Log1p để xử lý outliers lớn
    x = np.log1p(np.clip(x, 0.0, None))
    # Z-score đơn giản
    return (x - mu) / (sd + 1e-6)

# ---------- 1) ENCODER 2D: CALENDAR HEATMAP ----------
@torch.no_grad()
def enc_calendar_heatmap_torch(x168_t: torch.Tensor, target_size=(224, 224)) -> torch.Tensor:
    """
    Biến đổi chuỗi 168h -> Ảnh [3, 224, 224]
    """
    # 1. Reshape thành lưới 7 ngày x 24 giờ
    # x168_t: [168] -> [1, 1, 7, 24] (Batch, Channel, Height, Width)
    grid = x168_t.view(1, 1, 7, 24)
    
    # 2. Resize lên kích thước ảnh (Upsampling)
    # Dùng mode='nearest' để giữ nguyên các khối pixel sắc nét (tránh làm mờ ranh giới ngày/giờ)
    # Điều này giúp CNN nhận diện rõ ràng các ô bị "thủng" (Attack H5) hoặc lệch (H6)
    img_resized = F.interpolate(grid, size=target_size, mode='nearest') # [1, 1, 224, 224]
    
    # 3. Loại bỏ batch dimension ảo
    img_2d = img_resized.squeeze(0) # [1, 224, 224]
    
    # 4. Nhân bản thành 3 kênh (RGB) cho EfficientNet
    img_rgb = img_2d.repeat(3, 1, 1) # [3, 224, 224]
    
    return img_rgb

# ---------- 2) Augment 1D (Two-Crop) ----------
class TwoCropTemporal:
    """
    Augmentation trên dữ liệu gốc 1D trước khi biến thành ảnh
    """
    is_two_crop = True
    def __init__(self, max_shift=4, p_roll=0.5, p_jitter=0.5, noise_std=0.05, seed=2025):
        self.max_shift = int(max_shift)
        self.p_roll = float(p_roll)
        self.p_jitter = float(p_jitter)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def _aug_1d(self, x: torch.Tensor) -> torch.Tensor:
        v = x.clone()
        # Roll (Dịch thời gian)
        if self.rng.random() < self.p_roll and self.max_shift > 0:
            sh = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
            v = torch.roll(v, shifts=sh, dims=0)
        # Jitter (Nhiễu)
        if self.rng.random() < self.p_jitter:
            v = v + self.noise_std * torch.randn_like(v)
        return v

    def __call__(self, x_seq: torch.Tensor):
        q = self._aug_1d(x_seq)
        k = self._aug_1d(x_seq)
        return q, k

# ---------- 3) Dataset Class ----------
class DatasetPrepare_CER(Dataset):
    def __init__(self, root_dir, is_train=True, split_name=None, mu_hour=None, sd_hour=None,
                 encoder='heatmap', encoder_kwargs=None, transform=None, target_size=(224, 224), log_e=True):
        super().__init__()
        
        if split_name is None: split = 'train' if is_train else 'val'
        else: split = split_name
            
        self.filename = os.path.join(root_dir, f'{split}.npz')
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist!")

        dat = np.load(self.filename, allow_pickle=True)
        self.X = dat['data'].astype(np.float32)

        if 'label' in dat: self.atk = dat['label'].astype(np.int64)
        elif 'atk' in dat: self.atk = dat['atk'].astype(np.int64)
        else: raise KeyError("Missing label key")

        # Stats
        if mu_hour is None: self.mu = 0.0
        else: self.mu = float(mu_hour) # Scalar
            
        if sd_hour is None: self.sd = 1.0
        else: self.sd = float(sd_hour) # Scalar
            
        self.transform = transform
        self.target_size = tuple(target_size)

    def __len__(self): return len(self.X)

    def _norm_seq(self, x_np: np.ndarray) -> torch.Tensor:
        x = apply_global_hour_norm(x_np, self.mu, self.sd)
        return torch.from_numpy(x).float()

    def __getitem__(self, idx: int):
        x_np = np.asarray(self.X[idx], dtype=np.float32) # (168,)
        
        # Binary Label
        raw_label = int(self.atk[idx]) 
        y = 0 if raw_label == 0 else 1

        # 1. Normalize 1D
        x_seq = self._norm_seq(x_np)

        # 2. Augment & Encode 2D
        if self.transform is not None and getattr(self.transform, 'is_two_crop', False):
            # Augment trên 1D
            q_1d, k_1d = self.transform(x_seq)
            
            # Encode sang 2D Image
            q_img = enc_calendar_heatmap_torch(q_1d, target_size=self.target_size)
            k_img = enc_calendar_heatmap_torch(k_1d, target_size=self.target_size)
            
            return (q_img, k_img), torch.tensor(y, dtype=torch.long)

        # Single view
        if self.transform is not None:
             # Nếu có augment single-view (cho classifier training phase)
             x_seq = self.transform._aug_1d(x_seq)
             
        img = enc_calendar_heatmap_torch(x_seq, target_size=self.target_size)
        return img, torch.tensor(y, dtype=torch.long)