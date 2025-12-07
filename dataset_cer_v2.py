import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------- 0) Chuẩn hoá Global (Giữ nguyên) ----------
def fit_global_hour_stats(X_benign_train: np.ndarray):
    X = np.asarray(X_benign_train, dtype=np.float32)
    mu = X.mean(axis=0).astype(np.float32)
    sd = X.std(axis=0).astype(np.float32)
    return mu, sd

def apply_global_hour_norm(x24: np.ndarray, mu_h: np.ndarray, sd_h: np.ndarray):
    x = np.asarray(x24, np.float32)
    x = np.log1p(np.clip(x, 0.0, None))
    return (x - mu_h) / (sd_h + 1e-6)

# ---------- 1) ENCODER 3 KÊNH TRỰC TIẾP (CHANNEL-WISE SIGNAL) ----------
@torch.no_grad()
def enc_channel_wise_torch(x24_t: torch.Tensor, up_hw=(64,64)) -> torch.Tensor:
    """
    Biến đổi 1D -> 2D bằng cách vẽ biểu đồ diện tích (Area Chart) lên 3 kênh.
    Mỗi kênh đại diện cho một đặc tính vật lý quan trọng.
    """
    H, W = up_hw
    img = torch.zeros((3, H, W), dtype=torch.float32)
    
    # 1. Chuẩn bị dữ liệu
    v = x24_t.flatten() # Global Normed
    
    # --- CHANNEL 0 (R): RAW VALUE (Biên độ) ---
    # Mục đích: Phân biệt Class 0 (Cao) với Class 1, 4 (Thấp)
    # Map khoảng giá trị [-4, 4] vào chiều cao ảnh [0, H]
    GLOBAL_MIN, GLOBAL_MAX = -4.0, 4.0
    v_val = (v - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN)
    v_val = v_val.clamp(0.0, 1.0)
    
    # --- CHANNEL 1 (G): GRADIENT (Độ biến thiên) ---
    # Mục đích: Phân biệt Class 1 (Mượt) với Class 2, 3 (Gai)
    # Tính đạo hàm: |x_t - x_{t-1}|
    grad = torch.abs(v[1:] - v[:-1])
    grad = torch.cat([torch.tensor([0.0], device=v.device), grad])
    # Phóng đại gradient nhỏ (Non-linear boost)
    grad = torch.sqrt(grad + 1e-6) * 2.0 
    v_grad = grad.clamp(0.0, 1.0)
    
    # --- CHANNEL 2 (B): CUMULATIVE SUM (Tích phân) ---
    # Mục đích: Phân biệt Class 5 (Mất đoạn) với Class 6 (Đảo ngược)
    # Cumsum giúp lưu giữ thông tin thứ tự thời gian (Temporal Order)
    cumsum = torch.cumsum(v - v.min(), dim=0) # Dịch để dương trước khi cộng
    if cumsum.max() > 1e-5:
        v_cum = cumsum / cumsum.max() # Norm về 0-1
    else:
        v_cum = torch.zeros_like(cumsum)
        
    # 2. VẼ LÊN ẢNH (Dạng Area Chart - Tô màu từ dưới lên)
    # Trải rộng 24 điểm ra W pixel (Interpolation 1D)
    x_indices = torch.linspace(0, 23, steps=W) # [W]
    
    # Hàm nội suy tuyến tính 1D để làm mượt đường vẽ
    def interpolate_1d(signal, indices):
        # signal: [24], indices: [W]
        idx_floor = indices.floor().long().clamp(0, 23)
        idx_ceil = indices.ceil().long().clamp(0, 23)
        alpha = indices - idx_floor.float()
        return signal[idx_floor] * (1 - alpha) + signal[idx_ceil] * alpha

    vals_W = interpolate_1d(v_val, x_indices)   # [W]
    grads_W = interpolate_1d(v_grad, x_indices) # [W]
    cums_W  = interpolate_1d(v_cum, x_indices)  # [W]
    
    # Map sang chiều cao H
    h_vals = (vals_W * (H-1)).long()
    h_grads = (grads_W * (H-1)).long()
    h_cums = (cums_W * (H-1)).long()
    
    # Vector hóa việc tô màu (nhanh hơn vòng lặp)
    # Tạo lưới tọa độ Y: [H, W]
    y_grid = torch.arange(H-1, -1, -1, device=v.device).unsqueeze(1).expand(H, W)
    
    # Kênh R: Tô nếu y <= h_val
    img[0][y_grid <= h_vals] = 1.0
    
    # Kênh G: Tô nếu y <= h_grad
    img[1][y_grid <= h_grads] = 1.0
    
    # Kênh B: Tô nếu y <= h_cum (Tô kiểu đường kẻ dày 2px để ko che lấp)
    # Hoặc tô full area cũng được, CNN sẽ tự học cách phối hợp
    img[2][y_grid <= h_cums] = 1.0

    return img

# ---------- 2) Two-crop augment (Masking OFF) ----------
class TwoCropTemporal:
    is_two_crop = True
    def __init__(self, max_shift=2, p_roll=0.6, p_jitter=0.5, noise_std=0.03, p_mask=0.0, mask_k=(1,2), seed=2025):
        self.max_shift = int(max_shift)
        self.p_roll = float(p_roll)
        self.p_jitter = float(p_jitter)
        self.noise_std = float(noise_std)
        self.p_mask = 0.0 # Tắt Masking
        self.mask_k = mask_k
        self.rng = np.random.default_rng(seed)

    def _aug_1d(self, x: torch.Tensor) -> torch.Tensor:
        v = x.clone()
        if self.rng.random() < self.p_roll and self.max_shift > 0:
            sh = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
            v = torch.roll(v, shifts=sh, dims=0)
        if self.rng.random() < self.p_jitter:
            v = v + self.noise_std * torch.randn_like(v)
        return v

    def __call__(self, x24_t: torch.Tensor):
        q = self._aug_1d(x24_t)
        k = self._aug_1d(x24_t)
        return q, k

# ---------- 3) Dataset 2D ----------
class DatasetPrepare_CER(Dataset):
    def __init__(self, root_dir, is_train=True, split_name=None, mu_hour=None, sd_hour=None,
                 encoder='channel', encoder_kwargs=None, transform=None, target_size=(64, 64), log_e=True):
        super().__init__()
        
        if split_name is None: split = 'train' if is_train else 'val'
        else: split = split_name
            
        self.filename = os.path.join(root_dir, f'{split}.npz')
        if not os.path.isfile(self.filename): raise FileNotFoundError(f"{self.filename} does not exist!")

        dat = np.load(self.filename, allow_pickle=True)
        self.X = dat['data'].astype(np.float32)

        if 'label' in dat: self.atk = dat['label'].astype(np.int64)
        elif 'atk' in dat: self.atk = dat['atk'].astype(np.int64)
        else: raise KeyError(f"File {self.filename} missing 'label' or 'atk'")

        self.mu_h = np.asarray(mu_hour, np.float32)
        self.sd_h = np.asarray(sd_hour, np.float32)

        self.encoder = encoder.lower()
        self.target_size = tuple(target_size)
        self.transform = transform

    def __len__(self): return len(self.X)

    def _norm_24(self, x_np: np.ndarray) -> torch.Tensor:
        x = apply_global_hour_norm(x_np, self.mu_h, self.sd_h)
        return torch.from_numpy(x).float()

    def _encode_24_to_img(self, x24_t: torch.Tensor) -> torch.Tensor:
        # [FIX] Luôn dùng Encoder mới
        return enc_channel_wise_torch(x24_t, up_hw=self.target_size)

    # def __getitem__(self, idx: int):
    #     x_np = np.asarray(self.X[idx], dtype=np.float32)
    #     y    = int(self.atk[idx]) 

    #     x24_t = self._norm_24(x_np)

    #     if self.transform is not None and getattr(self.transform, 'is_two_crop', False):
    #         q1d, k1d = self.transform(x24_t)
    #         q_img = self._encode_24_to_img(q1d)
    #         k_img = self._encode_24_to_img(k1d)
    #         return (q_img, k_img), torch.tensor(y, dtype=torch.long)

    #     img = self._encode_24_to_img(x24_t)
    #     return img, torch.tensor(y, dtype=torch.long)
    
    def __getitem__(self, idx: int):
        x_np = np.asarray(self.X[idx], dtype=np.float32)
        
        # [MODIFIED] Logic chuyển đổi Binary
        raw_label = int(self.atk[idx]) 
        
        # Nếu là 0 (Benign) -> 0
        # Nếu là 1..6 (Attack) -> 1
        y = 0 if raw_label == 0 else 1

        x24_t = self._norm_24(x_np)

        if self.transform is not None and getattr(self.transform, 'is_two_crop', False):
            q1d, k1d = self.transform(x24_t)
            q_img = self._encode_24_to_img(q1d)
            k_img = self._encode_24_to_img(k1d)
            return (q_img, k_img), torch.tensor(y, dtype=torch.long)

        img = self._encode_24_to_img(x24_t)
        return img, torch.tensor(y, dtype=torch.long)