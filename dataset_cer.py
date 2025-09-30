# ===== dataset_etd2d_preserve_patterns.py =====
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------- 0) Chuẩn hoá đúng bản chất ETD ----------
def fit_global_hour_stats(X_benign_train: np.ndarray):
    """
    Tính (mu_h, sd_h) theo GIỜ trên benign train. X: (N,24)
    """
    X = np.asarray(X_benign_train, dtype=np.float32)
    mu = X.mean(axis=0).astype(np.float32)   # (24,)
    sd = X.std(axis=0).astype(np.float32)
    return mu, sd

def apply_global_hour_norm(x24: np.ndarray, mu_h: np.ndarray, sd_h: np.ndarray):
    """
    log1p + z-score theo giờ. Trả ndarray (24,)
    """
    x = np.asarray(x24, np.float32)
    x = np.log1p(np.clip(x, 0.0, None))
    return (x - mu_h) / (sd_h + 1e-6)

# ---------- 1) Encoders (tạo ảnh 24x24 rồi upscale 64x64) ----------
@torch.no_grad()
def enc_recurrence_torch(x24_t, sigma=0.25, up_hw=(64,64), diag_alpha=0.5):
    v = x24_t.flatten().float()
    D = torch.abs(v[:,None] - v[None,:])
    M = torch.exp(-D / float(sigma))
    if diag_alpha > 0:
        M = M - torch.eye(24, device=M.device)*diag_alpha
        M = torch.clamp(M, 0.0, 1.0)
    M = M.unsqueeze(0).unsqueeze(0)
    return F.interpolate(M, size=up_hw, mode='bilinear', align_corners=False).squeeze(0)


@torch.no_grad()
def enc_corr_outer_torch(x24_t: torch.Tensor, up_hw=(64,64)) -> torch.Tensor:
    """
    x24_t: (24,) đã norm global-hour; trả ảnh [1,64,64] với Gram (outer product) đã scale về [0,1]
    """
    v = x24_t.flatten().float()
    # per-sample z-score chỉ để ổn định Gram; magnitude tuyệt đối đã giữ ở bước global-hour
    v = (v - v.mean()) / (v.std() + 1e-6)
    G = torch.outer(v, v)                            # (24,24)
    # scale về [0,1] cho ổn định loss/decoder
    G = (G - G.min()) / (G.max() - G.min() + 1e-6)
    G = G.unsqueeze(0).unsqueeze(0)                  # [1,1,24,24]
    G = F.interpolate(G, size=up_hw, mode='bilinear', align_corners=False)
    return G.squeeze(0)                              # [1,64,64]

# ---------- 2) Two-crop temporal augment (trước khi encode ảnh) ----------
class TwoCropTemporal:
    is_two_crop = True
    def __init__(self, max_shift=2, p_roll=0.6, p_jitter=0.6, noise_std=0.03, p_mask=0.35, mask_k=(1,2), seed=2025):
        self.max_shift = int(max_shift)
        self.p_roll = float(p_roll)
        self.p_jitter = float(p_jitter)
        self.noise_std = float(noise_std)
        self.p_mask = float(p_mask)
        self.mask_k = mask_k
        self.rng = np.random.default_rng(seed)

    def _aug_1d(self, x: torch.Tensor) -> torch.Tensor:
        v = x.clone()
        # roll (dịch giờ)
        if self.rng.random() < self.p_roll and self.max_shift > 0:
            sh = int(self.rng.integers(-self.max_shift, self.max_shift + 1))
            v = torch.roll(v, shifts=sh, dims=0)
        # jitter nhẹ
        if self.rng.random() < self.p_jitter:
            v = v + self.noise_std * torch.randn_like(v)
        # short mask (mất gói / bỏ giờ)
        if self.rng.random() < self.p_mask:
            k = int(self.rng.integers(self.mask_k[0], self.mask_k[1] + 1))
            idx = self.rng.choice(24, size=k, replace=False)
            v[idx] = v.mean()  # hoặc 0.0 / interpolate
        return v

    def __call__(self, x24_t: torch.Tensor):
        q = self._aug_1d(x24_t)
        k = self._aug_1d(x24_t)
        return q, k

# ---------- 3) Dataset 2D: giữ pattern ETD ----------
class DatasetPrepare_CER(Dataset):
    """
    CER ETD dataset (giữ pattern):
      - npz: keys: data (N,24), atk (N,) 0..6
      - trả về ảnh [1,64,64] đã encode từ vector 24h bằng RP/Corr
      - không instance-norm theo mẫu
      - nếu transform.is_two_crop == True: augment 1D rồi encode -> (q_img, k_img)
    """
    def __init__(self,
                root_dir,
                is_train=True,
                split_name=None,      # nếu muốn dùng 'test' thay vì 'val'
                mu_hour=None,
                sd_hour=None,
                encoder='rp',         # 'rp' | 'corr'
                encoder_kwargs=None,
                transform=None,       # nên là TwoCropTemporal cho SSL
                target_size=(64, 64),
                log_e=True):
        super().__init__()
        # ---- file ----
        if split_name is None:
            split = 'train' if is_train else 'val'  # dùng 'val' (khớp file bạn đã lưu)
        else:
            split = split_name
        self.filename = os.path.join(root_dir, f'{split}.npz')
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist!")

        dat = np.load(self.filename, allow_pickle=True)
        self.X  = dat['data'].astype(np.float32)   # (N,24)
        self.atk = dat['atk'].astype(np.int64)     # (N,) 0..6
        assert len(self.X) == len(self.atk), "data/atk length mismatch"

        # ---- norm theo giờ (global) ----
        if mu_hour is None or sd_hour is None:
            raise ValueError("Cần truyền mu_hour, sd_hour (tính từ benign train) để giữ pattern đúng.")
        self.mu_h = np.asarray(mu_hour, np.float32)
        self.sd_h = np.asarray(sd_hour, np.float32)

        # ---- encode ----
        self.encoder = encoder.lower()
        self.encoder_kwargs = {} if encoder_kwargs is None else dict(encoder_kwargs)
        self.target_size = tuple(target_size)

        # ---- transform (temporal two-crop) ----
        self.transform = transform
        self.log_e = bool(log_e)  # giữ tham số cũ cho tương thích

    def __len__(self): return len(self.X)

    def _norm_24(self, x_np: np.ndarray) -> torch.Tensor:
        # log1p + z-score theo giờ
        x = apply_global_hour_norm(x_np, self.mu_h, self.sd_h)  # (24,)
        return torch.from_numpy(x).float()

    def _encode_24_to_img(self, x24_t: torch.Tensor) -> torch.Tensor:
        if self.encoder == 'rp':
            return enc_recurrence_torch(x24_t, **({'sigma':0.25, 'up_hw':self.target_size} | self.encoder_kwargs))
        elif self.encoder == 'corr':
            return enc_corr_outer_torch(x24_t, up_hw=self.target_size)
        else:
            raise ValueError(f"Unknown encoder: {self.encoder}")

    def __getitem__(self, idx: int):
        x_np = np.asarray(self.X[idx], dtype=np.float32)
        y    = int(self.atk[idx])  # 0=benign, 1..6=attack types

        x24_t = self._norm_24(x_np)  # (24,)

        # Two-crop theo thời gian → encode ảnh từng view
        if self.transform is not None and getattr(self.transform, 'is_two_crop', False):
            q1d, k1d = self.transform(x24_t)
            q_img = self._encode_24_to_img(q1d)   # [1,64,64]
            k_img = self._encode_24_to_img(k1d)
            return (q_img, k_img), torch.tensor(y, dtype=torch.long)

        # Single-view (có thể thêm augment khác nếu cần)
        img = self._encode_24_to_img(x24_t)       # [1,64,64]
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)
