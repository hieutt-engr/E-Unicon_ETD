import os
import numpy as np
import torch
from torch.utils.data import DataLoader
# Import class từ file dataset_cer.py
from dataset_cer_weekly import DatasetPrepare_CER, TwoCropTemporal, fit_global_hour_stats

# ------------------ utils ------------------
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

# ------------------ main ------------------
def data_preparing(config, args):
    init_seed(199)

    root_dir = getattr(config, "data_path", getattr(config, "root_dir", "./prepared_etd_multiclass"))
    train_npz = os.path.join(root_dir, "train.npz")
    test_npz  = os.path.join(root_dir, "val.npz")
    
    if not os.path.isfile(train_npz) or not os.path.isfile(test_npz):
        raise FileNotFoundError(f"train/val npz not found in {root_dir}")

    # ---- 1) Load thống kê từ Benign samples (Class 0) ----
    print(f"Loading stats from: {train_npz}")
    dat_tr = np.load(train_npz, allow_pickle=True)
    X_tr = dat_tr["data"].astype(np.float32)
    
    if 'label' in dat_tr:
        t_tr = dat_tr['label'].astype(np.int64)
    elif 'atk' in dat_tr:
        t_tr = dat_tr['atk'].astype(np.int64)
    else:
        raise KeyError("Missing label/atk key in train.npz")
        
    X_tr_benign = X_tr[t_tr == 0]
    if len(X_tr_benign) == 0:
        raise RuntimeError("No benign samples found to compute stats.")
        
    mu_h, sd_h = fit_global_hour_stats(X_tr_benign)

    # ---- 2) Augment: Two-crop theo thời gian (ĐÃ SỬA LẠI CHIẾN LƯỢC) ----
    two_crop = TwoCropTemporal(
        max_shift=4,      # Dịch chuyển thời gian nhẹ (OK)
        p_roll=0.6,
        p_jitter=0.5,     # Nhiễu nhẹ (OK)
        noise_std=0.03,   # Giảm noise xuống 0.03 để tránh nhầm với Class 3 (Zigzag)
        
        # [QUAN TRỌNG] TẮT MASKING ĐỂ CỨU CLASS 2
        p_mask=0.0,       
        mask_k=(1,2),
        seed=2025
    )

    # ---- 3) Datasets ----
    encoder_name = getattr(args, "encoder", getattr(config, "encoder", "rgb")).lower()
    TARGET_SIZE = (128, 128)
    print(f"Using Encoder: {encoder_name} (3 channels)")
    print(f"Target Size: {TARGET_SIZE}")

    # Contrastive Loader (Phase 1)
    train_ds_contrastive = DatasetPrepare_CER(
        root_dir=root_dir,
        is_train=True,
        mu_hour=mu_h, sd_hour=sd_h,
        encoder=encoder_name,
        encoder_kwargs={},
        transform=two_crop,  # Two-crop với mask=0
        target_size=TARGET_SIZE,
    )

    # Classifier Loader (Phase 2 - Linear Probing)
    train_ds_classifier = DatasetPrepare_CER(
        root_dir=root_dir,
        is_train=True,
        mu_hour=mu_h, sd_hour=sd_h,
        encoder=encoder_name,
        encoder_kwargs={},
        transform=None,     # Single-view, no augment
        target_size=TARGET_SIZE,
    )

    # Test Loader
    test_dataset = DatasetPrepare_CER(
        root_dir=root_dir,
        is_train=False,
        mu_hour=mu_h, sd_hour=sd_h,
        encoder=encoder_name,
        encoder_kwargs={},
        transform=None,     # Single-view, no augment
        target_size=TARGET_SIZE,
    )

    # ---- 4) Dataloaders ----
    num_workers = getattr(args, "num_workers", getattr(config, "num_workers", 8))
    batch_size  = getattr(config, "batch_size", 64)

    train_loader = DataLoader(
        dataset=train_ds_contrastive,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
    )

    train_classifier_loader = DataLoader(
        dataset=train_ds_classifier,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
    )

    print("Data Loaders ready (Masking Disabled).")
    return train_loader, train_classifier_loader, test_loader