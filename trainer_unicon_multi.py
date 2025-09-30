import sys
import os
import math
import time
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from logger import Logger
from networks.transformer_model.scheduler import CosineWarmupScheduler
from networks.efficient_net import ConEfficientNet, LinearClassifier
from losses import UniConLoss_Standard, recon_loss, vicreg_loss
from config import Config, prepare_fin, parser_process
from utils import AverageMeter, get_universum_standard, save_model
from data_loader_cer import data_preparing

warnings.filterwarnings("ignore")

layout = {
    "CAE-Transformer": {
        "losses": ["Multiline", ["loss/train", "loss/test"]],
        "learning rate": ["Multiline", ["learning_rate/lr"]],
        "auc": ["Multiline", ["AUC"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
    },
}

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

@torch.no_grad()
def evaluate(model, classifier, test_loader, device):
    model.eval(); classifier.eval()
    ce = nn.CrossEntropyLoss().to(device)
    losses = AverageMeter()
    all_pred, all_true = [], []

    for images, labels in test_loader:                 # single-view
        x = images.float().to(device)                  # [B,1,64,64]
        y = labels.to(device)

        feat = model(x)                                # [B,128]
        logits = classifier(feat)                      # [B,K]
        loss = ce(logits, y)
        losses.update(loss.item(), y.size(0))

        all_pred.append(logits.argmax(1).cpu().numpy())
        all_true.append(y.cpu().numpy())

    y_pred = np.concatenate(all_pred); y_true = np.concatenate(all_true)
    acc  = (y_pred == y_true).mean()
    f1   = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted')
    cm   = confusion_matrix(y_true, y_pred, labels=np.arange(config.n_classes))

    print(f'[VAL] CE: {losses.avg:.4f} | Acc: {acc*100:.2f} | F1: {f1:.4f} | P: {prec:.4f} | R: {rec:.4f}')
    print('[VAL] Confusion Matrix:\n', cm)
    sys.stdout.flush()
    return {'loss': losses.avg, 'acc': acc, 'f1': f1, 'precision': prec, 'recall': rec, 'cm': cm}

if __name__ == '__main__':
    args = parser_process()
    config = Config(args)
    prepare_fin(config)

    if config.log_mode == 'train':
        logger = Logger('./logs/transformer_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), layout)
    else:
        logger = Logger('./logs/transformer_test', layout)
        print("Test mode")

    # ----- Data loader -----
    train_loader, train_classifier_loader, test_loader = data_preparing(config, args)
    print('finish load data')

    # ----- Model -----
    assert config.model == 'efficientnet', "This script supports efficientnet only."
    model = ConEfficientNet(
        embedding_dim=1792, feat_dim=128, head='mlp',
        pretrained=getattr(config, 'pretrained', False)
    ).to(config.device)
    classifier = LinearClassifier(input_dim=128, num_classes=config.n_classes).to(config.device)

    # ----- Losses -----
    if str(config.method).lower() == 'unicon':
        con_loss = UniConLoss_Standard(temperature=config.temp).to(config.device)

    # ----- Opt/Sched -----
    opt_enc = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    sch_enc = CosineWarmupScheduler(opt_enc, warmup=50, max_iters=max(1, config.epoch_num * len(train_loader)))

    opt_cls = optim.AdamW(
        classifier.parameters(),
        lr=getattr(config, 'lr_classifier', max(1e-3, 0.1*config.lr)),
        weight_decay=1e-4
    )
    total_phase2_epochs = max(1, config.epoch_num - max(0, config.epoch_start_classifier))
    sch_cls = CosineWarmupScheduler(
        opt_cls,
        warmup=min(50, total_phase2_epochs * max(1, len(train_classifier_loader))),
        max_iters=max(1, total_phase2_epochs * max(1, len(train_classifier_loader)))
    )

    # ===== Training loop =====
    start_epoch = -1
    os.makedirs(config.model_save_path, exist_ok=True)

    for epoch in range(start_epoch + 1, config.epoch_num):
        print(f'\n=== Epoch {epoch} / {config.epoch_num-1} ==='); sys.stdout.flush()

        # ---------- Phase 1: Contrastive (two-crop) ----------
        set_requires_grad(model, True); model.train()
        running_loss, n_samples = 0.0, 0

        for i, (images, labels) in enumerate(train_loader):
            image1, image2 = images[0].float().to(config.device), images[1].float().to(config.device)
            labels = labels.to(config.device)

            # DEBUG 1 lần đầu
            if epoch == start_epoch + 1 and i == 0:
                print("\n[DEBUG] Epoch 1, Iter 0")
                print("image1 shape:", image1.shape, "dtype:", image1.dtype,
                    "min:", image1.min().item(), "max:", image1.max().item())
                print("image2 shape:", image2.shape, "dtype:", image2.dtype,
                    "min:", image2.min().item(), "max:", image2.max().item())
                print("image1[0, 0, :5, :5]:\n", image1[0, 0, :5, :5].cpu())
                print("image2[0, 0, :5, :5]:\n", image2[0, 0, :5, :5].cpu())

            images = torch.cat([image1, image2], dim=0)

            universum = get_universum_standard(images, labels, config)
            uni_features = model(universum)
            bsz = labels.size(0)

            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            
            # ====== Losses ======
            # SupCon + Universum
            loss = con_loss(features=features, labels=labels, universum=uni_features)

            opt_enc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_enc.step()
            if sch_enc is not None:
                sch_enc.step()

            # logging
            running_loss += loss.item() * bsz
            n_samples += bsz

            if i % 100 == 0:
                print(f'[P1] it {i:05d} | loss {loss.item():.4f}')
                sys.stdout.flush()

        print(f'[P1] Train Loss (contrastive): {running_loss / max(1, n_samples):.4f}')


        # ---------- Phase 2: Classifier ----------
        if epoch >= config.epoch_start_classifier:
            model.eval(); set_requires_grad(model, False)
            classifier.train()

            run_ce, correct, total = 0.0, 0, 0

            for i, (images, labels) in enumerate(train_classifier_loader):
                # loader classifier là single-view, nhưng vẫn phòng trường hợp tuple
                x = images[0] if isinstance(images, (tuple, list)) else images
                x = x.float().to(config.device)
                y = labels.to(config.device)

                with torch.no_grad():
                    feat = model(x)                        # [B,128]
                logits = classifier(feat)                  # [B,K]
                ce = F.cross_entropy(logits, y)

                opt_cls.zero_grad(set_to_none=True)
                ce.backward()
                opt_cls.step()
                sch_cls.step()

                bs = y.size(0)
                run_ce += ce.item() * bs
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += bs

                if i % 100 == 0:
                    print(f'[P2] it {i:05d} | CE {ce.item():.4f}')
                    sys.stdout.flush()

            print(f'[P2] Train CE: {run_ce/max(1,total):.4f} | Acc: {correct/max(1,total):.4f}')

            # Validate ngay sau Phase 2
            _ = evaluate(model, classifier, test_loader, config.device)

            save_file = os.path.join(config.model_save_path, f'ckpt_epoch_{epoch}.pth')
            save_model(model, opt_enc, config, epoch, save_file)

            ckpt = 'ckpt_class_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(config.model_save_path, ckpt)
            save_model(classifier, opt_cls, config, epoch, save_file)

            set_requires_grad(model, True)  # mở lại cho epoch kế tiếp

    # save cuối
    torch.save({'encoder': model.state_dict(), 'classifier': classifier.state_dict()},
               os.path.join(config.model_save_path, f'{config.model_name}_final.pth'))
