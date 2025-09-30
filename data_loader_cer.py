# ==== data_loading_etd2d.py ====
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset_cer import DatasetPrepare_CER, TwoCropTemporal, fit_global_hour_stats

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
    """
    Chuẩn bị:
      - đọc ./prepared_etd/train.npz & test.npz
      - tính (mu_h, sd_h) từ BENIGN của train
      - tạo 3 loader:
          + train_loader: two-crop temporal (cho contrastive)
          + train_classifier_loader: single-view (cho CE)
          + test_loader: single-view
    Ghi chú:
      - KHÔNG dùng Normalize/Erasing 2D vì ảnh đã encode từ 24h.
      - Không instance-norm per-sample (dataset đã bỏ).
    """
    init_seed(199)

    root_dir = getattr(config, "root_dir", "./prepared_etd")
    train_npz = os.path.join(root_dir, "train.npz")
    test_npz  = os.path.join(root_dir, "val.npz")
    if not os.path.isfile(train_npz) or not os.path.isfile(test_npz):
        raise FileNotFoundError(f"train/val npz not found in {root_dir}")

    # ---- 1) Tính thống kê theo GIỜ từ benign-train để giữ pattern
    dat_tr = np.load(train_npz, allow_pickle=True)
    X_tr = dat_tr["data"].astype(np.float32)
    t_tr = dat_tr["atk"].astype(np.int64)
    X_tr_benign = X_tr[t_tr == 0]
    if len(X_tr_benign) == 0:
        raise RuntimeError("No benign samples in train.npz to compute hour-wise stats.")
    mu_h, sd_h = fit_global_hour_stats(X_tr_benign)

    # ---- 2) Augment: two-crop theo thời gian (roll/jitter/mask)
    two_crop = TwoCropTemporal(
        max_shift=4,
        p_roll=0.6,
        p_jitter=0.6,  
        noise_std=0.05,
        p_mask=0.5,
        mask_k=(1,2),
        seed=2025
    )

    # ---- 3) Datasets (encoder='rp' có thể đổi 'corr')
    encoder_name = getattr(args, "encoder", getattr(config, "encoder", "rp")).lower()

    # Contrastive (two-crop)
    train_ds_contrastive = DatasetPrepare_CER(
        root_dir=root_dir,
        is_train=True,                 # sử dụng train.npz
        mu_hour=mu_h, sd_hour=sd_h,
        encoder=encoder_name,          # 'rp' | 'corr'
        encoder_kwargs={},             # có thể truyền {'sigma':0.25} cho RP
        transform=two_crop,            # two-crop temporal
        target_size=(64, 64),
    )

    # Classifier (single view)
    train_ds_classifier = DatasetPrepare_CER(
        root_dir=root_dir,
        is_train=True,
        mu_hour=mu_h, sd_hour=sd_h,
        encoder=encoder_name,
        encoder_kwargs={},
        transform=None,                # single-view
        target_size=(64, 64),
    )

    # Test (single view)
    test_dataset = DatasetPrepare_CER(
        root_dir=root_dir,
        is_train=False,                # sử dụng test.npz
        mu_hour=mu_h, sd_hour=sd_h,
        encoder=encoder_name,
        encoder_kwargs={},
        transform=None,                # single-view
        target_size=(64, 64),
    )

    # ---- 4) Thông tin
    total = len(train_ds_contrastive) + len(test_dataset)
    train_ratio = round(len(train_ds_contrastive) / max(1, total) * 100)
    print(
        "TRAIN SIZE:", len(train_ds_contrastive),
        " TEST SIZE:", len(test_dataset),
        " SIZE:", total,
        " TRAIN RATIO:", train_ratio, "%"
    )
    print(
        "MODE:", getattr(config, "mode", "train"),
        " ENCODER:", encoder_name,
        " EPOCH:", getattr(args, "epoch", getattr(config, "epoch_num", None)),
        " BATCH SIZE:", getattr(args, "batch_size", getattr(config, "batch_size", None)),
        " LR:", getattr(args, "lr", getattr(config, "lr", None))
    )

    # ---- 5) Dataloaders
    num_workers = getattr(args, "num_workers", getattr(config, "num_workers", 4))
    batch_size  = getattr(config, "batch_size", 128)

    train_loader = DataLoader(
        dataset=train_ds_contrastive,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,                # ổn định cho contrastive
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    train_classifier_loader = DataLoader(
        dataset=train_ds_classifier,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )

    print("finish load data")
    return train_loader, train_classifier_loader, test_loader
