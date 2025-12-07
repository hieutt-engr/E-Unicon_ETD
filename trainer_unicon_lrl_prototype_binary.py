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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

from logger import Logger
from networks.transformer_model.scheduler import CosineWarmupScheduler
from networks.efficient_v2_lrl_pro_224 import ConEfficientNet, LinearClassifier 
from losses import UniConLoss_ETD, recon_loss, vicreg_loss
from config import Config, prepare_fin, parser_process
from utils import AverageMeter, save_model
from data_loader_cer_weekly import data_preparing

warnings.filterwarnings("ignore")

# ----- LOSSES & UTILS -----

class ProtoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, prototypes_layer):
        prototypes_layer.weight.data = F.normalize(prototypes_layer.weight.data, dim=1)
        logits = prototypes_layer(features)
        scores, assigned_idx = torch.max(logits.detach(), dim=1)
        loss = F.cross_entropy(logits / self.temperature, assigned_idx)
        return loss

@torch.no_grad()
def generate_prototype_universum(features, prototypes_layer, alpha=0.5):
    sim_scores = prototypes_layer(features)
    _, assigned_idxs = torch.max(sim_scores, dim=1)
    
    B, num_proto = sim_scores.shape
    neg_idxs = torch.randint(0, num_proto, (B,), device=features.device)
    mask_collision = (neg_idxs == assigned_idxs)
    neg_idxs[mask_collision] = (neg_idxs[mask_collision] + 1) % num_proto
    
    proto_weights = prototypes_layer.weight.data
    c_neg = F.embedding(neg_idxs, proto_weights) 
    
    lam = torch.rand(B, device=features.device) * 0.4 + 0.3
    lam = lam.view(-1, 1)
    
    z_universum = lam * features + (1 - lam) * c_neg
    z_universum = F.normalize(z_universum, dim=1)
    return z_universum.detach()

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

@torch.no_grad()
def evaluate_binary(model, classifier, test_loader, device):
    model.eval(); classifier.eval()
    losses = AverageMeter()
    all_true, all_probs = [], []
    
    ce = nn.CrossEntropyLoss(weight=torch.tensor([2.0, 1.0]).to(device))

    for images, labels in test_loader:                 
        x = images.float().to(device)                  
        y = labels.to(device)

        feat = model(x)                                
        logits = classifier(feat)                      
        loss = ce(logits, y)
        losses.update(loss.item(), y.size(0))
        
        probs = F.softmax(logits, dim=1)[:, 1] 
        all_true.append(y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_probs = np.concatenate(all_probs)

    # [AUTO THRESHOLD SEARCH]
    best_f1 = 0; best_thresh = 0.5
    for thresh in np.arange(0.1, 0.99, 0.01):
        y_pred_t = (y_probs > thresh).astype(int)
        f1_t = f1_score(y_true, y_pred_t, pos_label=1)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh

    y_pred_final = (y_probs > best_thresh).astype(int)
    acc  = accuracy_score(y_true, y_pred_final)
    prec = precision_score(y_true, y_pred_final, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred_final, pos_label=1)
    try: auc = roc_auc_score(y_true, y_probs)
    except: auc = 0.0
    cm = confusion_matrix(y_true, y_pred_final)

    print(f'[VAL] Thresh: {best_thresh:.2f} | Acc: {acc*100:.2f} | F1: {best_f1:.4f} | P: {prec:.4f} | R: {rec:.4f} | AUC: {auc:.4f}')
    print('[VAL] Confusion Matrix:\n', cm)
    sys.stdout.flush()
    return {'loss': losses.avg, 'acc': acc, 'f1': best_f1, 'precision': prec, 'recall': rec, 'auc': auc}

# ----- MAIN -----

if __name__ == '__main__':
    args = parser_process()
    config = Config(args)
    config.n_classes = 2 # Binary
    
    args.encoder = 'heatmap'
    
    prepare_fin(config)

    if config.log_mode == 'train':
        logger = Logger('./logs/binary_efficientv2_' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), {})
    else:
        logger = Logger('./logs/binary_test', {})

    print("--- EFFICIENTNET-V2 WEEKLY TRAINING START ---")
    train_loader, train_classifier_loader, test_loader = data_preparing(config, args)
    
    num_prototypes = 2 
    
    # Model Init
    model = ConEfficientNet(
        embedding_dim=1792, feat_dim=128, head='mlp',
        pretrained=True,
        num_prototypes=num_prototypes,
        target_size=(224, 224)
    ).to(config.device)
    
    classifier = LinearClassifier(input_dim=128, num_classes=2).to(config.device)

    # Losses
    if str(config.method).lower() == 'unicon':
        con_loss = UniConLoss_ETD(temperature=config.temp).to(config.device)
    proto_loss_fn = ProtoNCELoss(temperature=0.07).to(config.device)

    # Optimizers
    opt_enc = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    sch_enc = CosineWarmupScheduler(opt_enc, warmup=50, max_iters=max(1, config.epoch_num * len(train_loader)))

    opt_cls = optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    sch_cls = CosineWarmupScheduler(opt_cls, warmup=50, max_iters=config.epoch_num * len(train_classifier_loader))

    # Placeholder
    opt_finetune = None
    sch_finetune = None

    start_epoch = -1
    os.makedirs(config.model_save_path, exist_ok=True)

    for epoch in range(start_epoch + 1, config.epoch_num):
        print(f'\n=== Epoch {epoch} / {config.epoch_num-1} ==='); sys.stdout.flush()

        # [E-STEP]
        if epoch == 0 or epoch % 5 == 0:
            print("Updating prototypes...")
            model.eval()
            proto_sum = torch.zeros(config.n_classes, 128).to(config.device)
            proto_count = torch.zeros(config.n_classes).to(config.device)
            with torch.no_grad():
                for images, labels in train_loader:
                    img = images[0].float().to(config.device)
                    y = labels.to(config.device)
                    z = model(img)
                    for c in range(config.n_classes):
                        mask = (y == c)
                        if mask.sum() > 0:
                            proto_sum[c] += z[mask].sum(dim=0)
                            proto_count[c] += mask.sum()
            new_centers = proto_sum / (proto_count.unsqueeze(1) + 1e-6)
            new_centers = F.normalize(new_centers, dim=1)
            model.prototypes.weight.data.copy_(new_centers)
            print("Prototypes updated.")

        # [PHASE 1] Contrastive
        set_requires_grad(model, True); model.train()
        model.normalize_prototypes()
        running_loss, n_samples = 0.0, 0

        for i, (images, labels) in enumerate(train_loader):
            image1, image2 = images[0].float().to(config.device), images[1].float().to(config.device)
            labels = labels.to(config.device)

            z_q, h_q, x_hat_q, p_q = model.forward_lrl(image1) 
            z_k, h_k, x_hat_k, p_k = model.forward_lrl(image2)

            z_u = generate_prototype_universum(z_q, model.prototypes, alpha=0.5)

            feats = torch.stack([z_q, z_k], dim=1)
            L_uni = con_loss(features=feats, labels=labels, universum=z_u)
            
            L_proto = (proto_loss_fn(z_q, model.prototypes) + proto_loss_fn(z_k, model.prototypes)) / 2
            
            L_rec = recon_loss(x_hat_q, image1) + recon_loss(x_hat_k, image2) if x_hat_q is not None else 0
            L_vic = vicreg_loss(p_q, p_k) if p_q is not None else 0

            loss = L_uni + 0.1 * L_rec + 0.1 * L_vic + 2.0 * L_proto

            opt_enc.zero_grad(set_to_none=True)
            loss.backward()
            opt_enc.step()
            sch_enc.step()
            model.normalize_prototypes()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        print(f'[P1] Train Loss: {running_loss / max(1, n_samples):.4f}')

        # [PHASE 2] Classifier
        if epoch >= 0: 
            FINE_TUNE_START_EPOCH = 30 
            is_finetuning = (epoch >= FINE_TUNE_START_EPOCH)
            
            if not is_finetuning:
                model.eval(); set_requires_grad(model, False)
                classifier.train()
                current_opt = opt_cls; current_sch = sch_cls
            else:
                model.train(); set_requires_grad(model, True)
                classifier.train()
                
                if opt_finetune is None:
                    print(f"\n[INFO] FULL FINE-TUNING START at Epoch {epoch}")
                    params = [{'params': model.parameters(), 'lr': 1e-5}, {'params': classifier.parameters(), 'lr': 1e-3}]
                    opt_finetune = optim.AdamW(params, weight_decay=1e-4)
                    remaining = config.epoch_num - epoch
                    sch_finetune = CosineWarmupScheduler(opt_finetune, warmup=20, max_iters=remaining*len(train_classifier_loader))
                current_opt = opt_finetune; current_sch = sch_finetune

            run_ce, correct, total = 0.0, 0, 0
            
            for i, (images, labels) in enumerate(train_classifier_loader):
                x = images[0] if isinstance(images, (tuple, list)) else images
                x = x.float().to(config.device)
                y = labels.to(config.device)

                feat = model(x)
                logits = classifier(feat)
                ce = F.cross_entropy(logits, y)

                current_opt.zero_grad(set_to_none=True)
                ce.backward()
                current_opt.step()
                if current_sch is not None: current_sch.step()

                bs = y.size(0)
                run_ce += ce.item() * bs
                correct += (logits.argmax(1) == y).sum().item()
                total += bs

            mode_str = "Fine-tune" if is_finetuning else "Lin-Probe"
            print(f'[P2] {mode_str} CE: {run_ce/total:.4f} | Acc: {correct/total:.4f}')

            # Validate
            _ = evaluate_binary(model, classifier, test_loader, config.device)

            save_file = os.path.join(config.model_save_path, f'ckpt_epoch_{epoch}.pth')
            save_model(model, opt_enc, config, epoch, save_file)
            save_model(classifier, opt_cls, config, epoch, os.path.join(config.model_save_path, f'ckpt_class_{epoch}.pth'))