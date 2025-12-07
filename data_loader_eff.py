import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_eff import DatasetPrepare_ETD


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

def init_seed(seed: int):
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _worker_init_fn(worker_id):
    # mỗi worker có seed khác nhau để augment ngẫu nhiên ổn định
    seed = 199 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)

def data_preparing(config, args):
    init_seed(199)

    # 1-channel stats
    mean = (0.5,)
    std  = (0.5,)

    # --- transforms ---
    # test / classifier: 1 view (normalize)
    test_transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
    ])

    # train contrastive: two-crop (noise nhẹ + erasing + normalize)
    train_base = transforms.Compose([
        transforms.RandomApply([AddGaussianNoise(0., 0.05)], p=0.5),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        transforms.Normalize(mean=mean, std=std),
    ])
    train_transform = TwoCropTransform(train_base)

    # --- datasets ---
    # DatasetPrepare đã trả tensor [1,64,64] trước khi apply transform
    train_ds_contrastive = DatasetPrepare_ETD(
        root_dir=config.root_dir,
        sequence_size=config.window_size,
        pad_size=0,
        embed=config.d_model,
        max_time_position=config.max_time_position,
        log_e=config.log_e,
        transform=train_transform,         # two-crop
        is_train=True,
        do_instance_norm=True,
        target_size=(64, 64),
    )
    # classifier dùng 1 view để tránh tuple/list ở loader
    train_ds_classifier = DatasetPrepare_ETD(
        root_dir=config.root_dir,
        sequence_size=config.window_size,
        pad_size=0,
        embed=config.d_model,
        max_time_position=config.max_time_position,
        log_e=config.log_e,
        transform=test_transform,          # single-view
        is_train=True,
        do_instance_norm=True,
        target_size=(64, 64),
    )
    test_dataset = DatasetPrepare_ETD(
        root_dir=config.root_dir,
        sequence_size=config.window_size,
        pad_size=0,
        embed=config.d_model,
        max_time_position=config.max_time_position,
        log_e=config.log_e,
        transform=test_transform,          # single-view
        is_train=False,
        do_instance_norm=True,
        target_size=(64, 64),
    )

    # --- info print ---
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
        " INDIR:", getattr(args, "indir", ""),
        " WINDOW SIZE:", getattr(args, "window_size", config.window_size),
        " EPOCH:", getattr(args, "epoch", config.epoch_num),
        " BATCH SIZE:", getattr(args, "batch_size", config.batch_size),
        " LR:", getattr(args, "lr", getattr(config, "lr", None))
    )

    # --- dataloaders ---
    num_workers = getattr(args, "num_workers", getattr(config, "num_workers", 4))
    batch_size  = config.batch_size

    train_loader = DataLoader(
        dataset=train_ds_contrastive,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,               # bắt buộc cho contrastive ổn định
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        persistent_workers=(num_workers > 0),
    )
    train_classifier_loader = DataLoader(
        dataset=train_ds_classifier,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,               # nên drop_last để batch đều
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

    print('finish load data')
    return train_loader, train_classifier_loader, test_loader
