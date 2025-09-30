import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ==== helpers  ====
def to_gray1ch(x: torch.Tensor) -> torch.Tensor:
    """[37,28] -> [1,37,28]; [C,37,28] -> [1,37,28] (average channels)."""
    if x.dim() == 2:                 # [37,28]
        x = x.unsqueeze(0)           # -> [1,37,28]
    elif x.dim() == 3 and x.size(0) != 1:
        x = x.mean(dim=0, keepdim=True)
    return x

def instance_norm_2d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Instance normalization (per-sample) on H×W. x: [1,H,W] or [B,1,H,W]."""
    if x.dim() == 3:  # [1,H,W]
        mean = x.mean(dim=(1,2), keepdim=True)
        std  = x.std (dim=(1,2), keepdim=True).clamp_min(eps)
        return (x - mean) / std
    elif x.dim() == 4:  # [B,1,H,W]
        mean = x.mean(dim=(2,3), keepdim=True)
        std  = x.std (dim=(2,3), keepdim=True).clamp_min(eps)
        return (x - mean) / std
    return x

def day24_to_img64x64(vec24: torch.Tensor) -> torch.Tensor:
    """
    vec24: (24,) hoặc (1,24)
    Mapping: Hàng = thời gian (24 → nội suy lên 64), Cột = lặp lại để đủ 64.
    Out   : [1,64,64]
    """
    v = vec24.flatten().float()  # (24,)
    # nội suy 24 -> 64 theo trục dọc (row = time)
    v64 = F.interpolate(v.view(1,1,24,1), size=(64,1),
                        mode='bilinear', align_corners=False).view(64)  # (64,)
    img = v64.unsqueeze(0).repeat(64, 1).t()   # (64,64)
    return img.unsqueeze(0)  # [1,64,64]

class DatasetPrepare_CER(Dataset):
    """
    CER dataset: 
      - data[i]: (24,) float32 (một ngày)
      - label[i]: nhị phân (0=benign, 1=theft)
      - atk[i]:   multi-class (0=benign, 1..6=attack types)
    Trả về:
      data: Tensor [1,64,64]
      label: torch.long (atk type 0..6)
    """
    def __init__(self, root_dir, sequence_size, pad_size, embed,
                 max_time_position, log_e, transform=None, is_train=True,
                 do_instance_norm: bool = True, target_size=(64, 64)):
        super().__init__()
        split = 'train' if is_train else 'val'
        self.filename = os.path.join(root_dir, f'{split}.npz')
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f"{self.filename} does not exist!")

        dat = np.load(self.filename, allow_pickle=True)
        self.data  = dat['data']   # (N,24)
        self.atk   = dat['atk']    # (N,) 0..6
        self.transform = transform
        self.log_e = bool(log_e)
        self.do_instance_norm = bool(do_instance_norm)
        self.target_size = tuple(target_size)

        assert len(self.data) == len(self.atk), "data/atk length mismatch"

    def __len__(self):
        return len(self.data)

    def _preprocess(self, x_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x_np).float().flatten()  # (24,)
        if x.numel() != 24:
            raise ValueError(f"CER sample must have 24 values, got {x.numel()}")
        if self.log_e:
            x = torch.log1p(torch.clamp(x, min=0.0))
        x = day24_to_img64x64(x)  # -> [1,64,64]
        if self.do_instance_norm:
            x = instance_norm_2d(x)
        return x

    def __getitem__(self, idx):
        x_np = np.asarray(self.data[idx], dtype=np.float32)
        y    = int(self.atk[idx])   # 0=benign, 1..6=attack

        x = self._preprocess(x_np)

        if self.transform is not None and getattr(self.transform, 'is_two_crop', False):
            q, k = self.transform(x)
            return (q, k), torch.tensor(y, dtype=torch.long)
        elif self.transform is not None:
            x = self.transform(x)
            return x, torch.tensor(y, dtype=torch.long)
        else:
            return x, torch.tensor(y, dtype=torch.long)

class DatasetPrepare_ETD(Dataset):
    def __init__(self, root_dir, sequence_size, pad_size, embed,
                 max_time_position, log_e, transform=None, is_train=True,
                 do_instance_norm: bool = True, target_size=(64, 64)):
        super().__init__()
        split = 'train' if is_train else 'val'
        self.filename = os.path.join(root_dir, f'{split}.npz')
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(self.filename + ' does not exist!')

        dat = np.load(self.filename, allow_pickle=True)
        self.data  = dat['data']   # (N, 37, 28) or (N, 1, 37, 28)
        self.label = dat['label']  # (N,)

        self.transform = transform
        self.log_e = bool(log_e)
        self.do_instance_norm = do_instance_norm
        self.target_size = target_size

        assert len(self.data) == len(self.label), "data/label length mismatch"

    def __len__(self):
        return len(self.data)

    def _preprocess(self, x_np: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(x_np).float()            # [37,28] / [1,37,28]
        if self.log_e:
            x = torch.log1p(torch.clamp(x, min=0.0))

        x = to_gray1ch(x)                             # -> [1,37,28]
        if self.do_instance_norm:
            x = instance_norm_2d(x)                   # ~N(0,1)

        # [1, H, W] = [1, 64, 64]
        x = F.interpolate(x.unsqueeze(0), size=self.target_size,
                          mode='bilinear', align_corners=False).squeeze(0)  # [1,64,64]
        return x

    def __getitem__(self, idx):
        x_np = np.asarray(self.data[idx], dtype=np.float32)
        y    = int(self.label[idx])

        x = self._preprocess(x_np)                    # [1,64,64]

        # Apply transform: may return (q,k) or a single tensor
        if self.transform is not None:
            data = self.transform(x)
        else:
            data = x

        # Return in DataLoader format: (data, label)
        # data: torch.Tensor [1,64,64] or tuple(tensor, tensor)
        return data, torch.tensor(y, dtype=torch.long)
