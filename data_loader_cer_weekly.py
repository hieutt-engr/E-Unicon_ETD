import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_cer_weekly import DatasetPrepare_CER, TwoCropTemporal, fit_global_hour_stats

# ------------------ Utils ------------------
def init_seed(seed: int):
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _worker_init_fn(worker_id):
    seed = 199 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------ Main Loader ------------------
def data_preparing(config, args):
    init_seed(199)

    # Đường dẫn data tuần 168h
    root_dir = getattr(config, "data_path", getattr(config, "root_dir", "./prepared_etd_weekly_168"))
    train_npz = os.path.join(root_dir, "train.npz")
    
    if not os.path.isfile(train_npz):
        raise FileNotFoundError(f"Data not found: {root_dir}")

    # ---- 1) Stats ----
    print(f"Loading stats from: {train_npz}")
    dat_tr = np.load(train_npz, allow_pickle=True)
    X_tr = dat_tr["data"].astype(np.float32)
    
    if 'label' in dat_tr: t_tr = dat_tr['label'].astype(np.int64)
    else: t_tr = dat_tr['atk'].astype(np.int64)
        
    X_tr_benign = X_tr[t_tr == 0]
    if len(X_tr_benign) == 0: X_tr_benign = X_tr # Fallback
        
    mu_val, sd_val = fit_global_hour_stats(X_tr_benign)
    print(f"Global Stats: Mean={mu_val:.4f}, Std={sd_val:.4f}")

    # ---- 2) Augment ----
    two_crop = TwoCropTemporal(
        max_shift=4,      # Dịch tối đa 4 giờ trong tuần
        p_roll=0.5,
        p_jitter=0.5,
        noise_std=0.05,
        seed=2025
    )

    # ---- 3) Configuration ----
    # EfficientNetV2-S input chuẩn là 224x224 hoặc lớn hơn (đến 300, 384)
    # 224 là cân bằng tốt nhất giữa tốc độ và độ chi tiết cho lưới 7x24
    TARGET_SIZE = (224, 224) 
    print(f"Target Image Size: {TARGET_SIZE} (Calendar Heatmap)")

    # Loaders
    train_ds_contrastive = DatasetPrepare_CER(
        root_dir=root_dir, is_train=True,
        mu_hour=mu_val, sd_hour=sd_val,
        encoder='heatmap',
        transform=two_crop, 
        target_size=TARGET_SIZE,
    )

    train_ds_classifier = DatasetPrepare_CER(
        root_dir=root_dir, is_train=True,
        mu_hour=mu_val, sd_hour=sd_val,
        encoder='heatmap',
        transform=None,     
        target_size=TARGET_SIZE,
    )

    test_dataset = DatasetPrepare_CER(
        root_dir=root_dir, is_train=False,
        mu_hour=mu_val, sd_hour=sd_val,
        encoder='heatmap',
        transform=None,     
        target_size=TARGET_SIZE,
    )

    # ---- 4) Dataloaders ----
    num_workers = getattr(args, "num_workers", getattr(config, "num_workers", 8))
    # EfficientNet V2-S nặng hơn, giảm batch size nếu OOM
    batch_size  = getattr(config, "batch_size", 64) 

    train_loader = DataLoader(
        dataset=train_ds_contrastive,
        batch_size=batch_size,
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
    )

    train_classifier_loader = DataLoader(
        dataset=train_ds_classifier,
        batch_size=batch_size,
        shuffle=True, drop_last=True, pin_memory=True,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False, drop_last=False, pin_memory=True,
        num_workers=num_workers, worker_init_fn=_worker_init_fn,
    )

    print("Data Loaders ready (2D EfficientNet Mode).")
    return train_loader, train_classifier_loader, test_loader