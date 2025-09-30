from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F
import torch
import os
import math
import random
import json
import pickle
import codecs
import torch
import datetime
import numpy as np
import torch.nn as nn
import pandas as pd
import torch.optim as optim

METRICS = {
    'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
    'euclidean': lambda gallery, query: euclidean_dist(query, gallery),
    'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
    'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
}

# def draw_confusion(label_y, pre_y, path):
#     confusion = confusion_matrix(label_y, pre_y)
#     print(confusion)
    
def draw_confusion(label_y, pre_y):
    cm = confusion_matrix(label_y, pre_y)
    # Calculate the confusion matrix

    # Print False Negatives for each class
    print("False Negatives for each class:")
    for i, label in enumerate(set(label_y)):
        false_negatives = sum(cm[i, :]) - cm[i, i]
        true_positives = cm[i, i]
        if (false_negatives + true_positives) > 0:
            fnr = false_negatives / (false_negatives + true_positives)
        else:
            fnr = 0.0
        print(f"Class {label}: {fnr:.4f}")
    print(cm)

def write_result(fin, label_y, pre_y):
    accuracy = accuracy_score(label_y, pre_y)
    precision = precision_score(label_y, pre_y)
    recall = recall_score(label_y, pre_y)
    f1 = f1_score(label_y, pre_y)
    print('  -- test result: ')
    print('    -- accuracy: ', accuracy)
    fin.write('    -- accuracy: ' + str(accuracy) + '\n')
    print('    -- recall: ', recall)
    fin.write('    -- recall: ' + str(recall) + '\n')
    print('    -- precision: ', precision)
    fin.write('    -- precision: ' + str(precision) + '\n')
    print('    -- f1 score: ', f1)
    fin.write('    -- f1 score: ' + str(f1) + '\n\n')
    report = classification_report(label_y, pre_y)
    fin.write(report)
    fin.write('\n\n')
    return f1, accuracy

def cal_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def print_model_info_pytorch(model):
    """
    Prints all the model parameters, both trainable and non-trainable, and calculates the model size.
    
    Args:
        model (torch.nn.Module): The PyTorch model.
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size
        else:
            non_trainable_params += param_size

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    # Calculate model size (assuming 32-bit floats for parameters)
    model_size = total_params * 4 / (1024 ** 2)  # Size in MB (4 bytes per float)
    print(f"Model size: {model_size:.2f} MB")
    
def l2_normalize(x):
    return x / x.norm(dim=1, keepdim=True)

def classify_feats(prototypes, classes, feats, targets, metric='euclidean', sigma=1.0):
    # Classify new examples with prototypes and return classification error
    # dist = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim=2)  # Squared euclidean distance
    # Calculate distances from query embeddings to prototypes
    # dist = torch.cdist(feats, prototypes)
    # dist = euclidean_dist(feats, prototypes)

    dist = METRICS[metric](prototypes, feats)
    preds = F.log_softmax(-dist, dim=1)
    labels = (classes[None, :] == targets[:, None]).long().argmax(dim=-1)

    with torch.no_grad():
        acc = (preds.argmax(dim=1) == labels).float().mean()
        f1 = f1_score(labels.cpu().numpy(), preds.argmax(dim=1).cpu().numpy(), average='weighted')
    return preds, labels, acc, f1

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
    
    
class TensorResize(nn.Module):
    """Resize tensor [C,H,W] -> [C,h,w] dùng bilinear (giữ C)."""
    def __init__(self, size=(64, 64)):
        super().__init__()
        self.size = size
    def forward(self, x):
        assert x.dim() == 3, f"Expect [C,H,W], got {tuple(x.shape)}"
        x4 = x.unsqueeze(0)                              # [1,C,H,W]
        x4 = F.interpolate(x4, size=self.size, mode='bilinear', align_corners=False)
        return x4.squeeze(0)
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine and epoch <= 1000:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2  # args.epochs
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def rand_bbox(size, lam):
    '''Getting the random box in CutMix'''
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def get_universum_standard(images, labels, opt):
    """Calculating Mixup-induced universum from a batch of images"""
    tmp = images.cpu()
    label = labels.cpu()
    bsz = tmp.shape[0]
    bs = len(label)
    class_images = [[] for i in range(max(label) + 1)]
    for i in label.unique():
        class_images[i] = np.where(label != i)[0]
    units = [tmp[random.choice(class_images[labels[i % bs]])] for i in range(bsz)]
    universum = torch.stack(units, dim=0).cuda()
    lamda = opt.lamda
    if not hasattr(opt, 'mix') or opt.mix == 'mixup':
        # Using Mixup
        universum = lamda * universum + (1 - lamda) * images
    else:
        # Using CutMix
        lam = 0
        while lam < 0.45 or lam > 0.55:
            # Since it is hard to control the value of lambda in CutMix,
            # we accept lambda in [0.45, 0.55].
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lamda)
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        universum[:, :, bbx1:bbx2, bby1:bby2] = images[:, :, bbx1:bbx2, bby1:bby2]
    return universum

@torch.no_grad()
def get_universum_etd(images: torch.Tensor, labels: torch.Tensor, opt) -> torch.Tensor:
    """
    images : [B, 1, H, W] (float, đã normalize/resize)
    labels : [B] (long)
    opt.lamda in [0,1]
    opt.mix   in {'mixup','cutmix'}
    return   : universum [B, 1, H, W] (cùng dtype/device)
    """
    assert images.dim() == 4 and labels.dim() == 1, "Bad shapes"
    device = images.device
    B, C, H, W = images.shape

    lam = float(getattr(opt, 'lamda', 0.5))
    lam = max(0.0, min(1.0, lam))
    mix_mode = str(getattr(opt, 'mix', 'mixup')).lower()

    # --- chọn chỉ mục khác lớp (vectorized) ---
    # Thử một hoán vị ngẫu nhiên; nếu còn trùng lớp, sửa lại các vị trí trùng
    perm = torch.randperm(B, device=device)
    same = (labels == labels[perm])
    if same.all():
        # batch 1 lớp hoặc hoán vị đụng lớp toàn bộ -> xoay 1 bước (nếu B>1)
        if B > 1:
            perm = torch.roll(torch.arange(B, device=device), shifts=1, dims=0)
        else:
            perm = torch.arange(B, device=device)  # B==1, đành dùng chính nó

    # Nếu vẫn còn vài phần tử trùng lớp (batch có >1 lớp nhưng perm xui):
    tries = 0
    while same.any() and tries < 5 and B > 1:
        idx_bad = same.nonzero(as_tuple=False).squeeze(1)            # các vị trí bị trùng
        # random re-assign chỉ những vị trí này
        cand = torch.randperm(B, device=device)
        perm[idx_bad] = cand[idx_bad]
        # không để i == perm[i]
        collide = (perm == torch.arange(B, device=device))
        perm[collide] = (perm[collide] + 1) % B
        same = (labels == labels[perm])
        tries += 1

    other = images.index_select(0, perm)  # [B,1,H,W]

    # --- mixup / cutmix ---
    if mix_mode == 'mixup':
        universum = lam * other + (1.0 - lam) * images  # [B,1,H,W]
    elif mix_mode == 'cutmix' and B > 0:
        # rand_bbox: vùng chèn theo lambda
        def rand_bbox(H, W, lam_val):
            # area giữ ~lam_val, clamp để không quá bé
            lam_clip = float(lam_val)
            lam_clip = min(max(lam_clip, 0.0), 1.0)
            cut_rat = (1.0 - lam_clip) ** 0.5
            cut_h = int(H * cut_rat)
            cut_w = int(W * cut_rat)
            cut_h = max(1, cut_h)
            cut_w = max(1, cut_w)
            cy = torch.randint(0, H, (1,), device=device).item()
            cx = torch.randint(0, W, (1,), device=device).item()
            y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
            x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
            return y1, y2, x1, x2

        universum = other.clone()
        # áp cho từng mẫu (đơn giản, ổn định)
        for b in range(B):
            y1, y2, x1, x2 = rand_bbox(H, W, lam)
            universum[b, :, y1:y2, x1:x2] = images[b, :, y1:y2, x1:x2]
    else:
        # mode không hợp lệ -> fallback mixup
        universum = lam * other + (1.0 - lam) * images

    return universum.to(dtype=images.dtype, device=device)

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, save_file)
    del state

def load_checkpoint(checkpoint_path, model, optimizer):
    print(f"==> Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])  # Load model weights
    optimizer.load_state_dict(checkpoint['optimizer'])  # Load optimizer state
    start_epoch = checkpoint['epoch']  # Load saved epoch
    opt = checkpoint['opt']  # Load saved options if needed

    return model, optimizer, start_epoch, opt