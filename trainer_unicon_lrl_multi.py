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
from sklearn.cluster import KMeans

from logger import Logger
from networks.transformer_model.scheduler import CosineWarmupScheduler
from networks.efficient_v2_lrl_pro import ConEfficientNet, LinearClassifier 
from losses import UniConLoss_ETD, recon_loss, vicreg_loss, ProtoNCELoss
from config import Config, prepare_fin, parser_process
from utils import AverageMeter, save_model
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
    lam = lam.view(-1, 1) # [B, 1]
    
    z_universum = lam * features + (1 - lam) * c_neg
    z_universum = F.normalize(z_universum, dim=1)
    
    return z_universum.detach()

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

@torch.no_grad()
def evaluate(model, classifier, test_loader, device):
    model.eval(); classifier.eval()
    ce = nn.CrossEntropyLoss().to(device)
    losses = AverageMeter()
    all_pred, all_true = [], []

    for images, labels in test_loader:                 
        x = images.float().to(device)                  
        y = labels.to(device)

        feat = model(x)                                
        logits = classifier(feat)                      
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

# ----- MAIN EXECUTION -----
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
    num_prototypes = 7
    
    model = ConEfficientNet(
        embedding_dim=1792, feat_dim=128, head='mlp',
        pretrained=getattr(config, 'pretrained', False),
        num_prototypes=num_prototypes,
        target_size=(128, 128)
    ).to(config.device)
    
    classifier = LinearClassifier(input_dim=128, num_classes=config.n_classes).to(config.device)

    # ----- Losses -----
    if str(config.method).lower() == 'unicon':
        con_loss = UniConLoss_ETD(temperature=config.temp).to(config.device)

    # [NEW] ProtoNCE Loss
    proto_loss_fn = ProtoNCELoss(temperature=0.1).to(config.device)

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
    opt_finetune = None
    sch_finetune = None
    # ===== Training loop =====
    start_epoch = -1
    os.makedirs(config.model_save_path, exist_ok=True)

    for epoch in range(start_epoch + 1, config.epoch_num):
        print(f'\n=== Epoch {epoch} / {config.epoch_num-1} ==='); sys.stdout.flush()

        # [NEW LOGIC] Supervised Prototype Update
        if epoch == 0 or epoch % 5 == 0:
            print("Updating prototypes based on Class Centers (Supervised)...")
            model.eval()
            
            proto_sum = torch.zeros(config.n_classes, 128).to(config.device)
            proto_count = torch.zeros(config.n_classes).to(config.device)
            
            with torch.no_grad():
                for images, labels in train_loader:
                    img = images[0].float().to(config.device)
                    y = labels.to(config.device)
                    z = model(img) # [B, 128]
                    for c in range(config.n_classes):
                        mask = (y == c)
                        if mask.sum() > 0:
                            proto_sum[c] += z[mask].sum(dim=0)
                            proto_count[c] += mask.sum()
            

            new_centers = proto_sum / (proto_count.unsqueeze(1) + 1e-6)
            new_centers = F.normalize(new_centers, dim=1)
            
            model.prototypes.weight.data.copy_(new_centers)
            print("Prototypes updated to Class Centers.")

        # ---------- Phase 1: Contrastive (Hybrid) ----------
        set_requires_grad(model, True); model.train()
        
        model.prototypes.weight.data = F.normalize(model.prototypes.weight.data, dim=1)
        running_loss, n_samples = 0.0, 0

        for i, (images, labels) in enumerate(train_loader):
            image1, image2 = images[0].float().to(config.device), images[1].float().to(config.device)
            labels = labels.to(config.device)

            # 1. Forward 2 views
            z_q, h_q, x_hat_q, p_q = model.forward_lrl(image1) 
            z_k, h_k, x_hat_k, p_k = model.forward_lrl(image2)

            # 2. [NEW] Generate Hard Universum
            z_u = generate_prototype_universum(z_q, model.prototypes, alpha=0.5)

            # 3. Tính Losses
            # A. UniCon Loss (Instance Discrimination với Hard Negatives)
            feats = torch.stack([z_q, z_k], dim=1) # [B,2,D]
            L_uni = con_loss(features=feats, labels=labels, universum=z_u)

            # B. [NEW] ProtoNCE Loss (Semantic Clustering Structure)
            L_proto = (proto_loss_fn(z_q, model.prototypes) + proto_loss_fn(z_k, model.prototypes)) / 2

            # C. Reconstruction
            L_rec = 0.0
            if x_hat_q is not None and x_hat_k is not None:
                L_rec = recon_loss(x_hat_q, image1) + recon_loss(x_hat_k, image2)

            # D. VICReg
            L_vic = 0.0
            if (p_q is not None) and (p_k is not None):
                L_vic = vicreg_loss(p_q, p_k)

            alpha_loss = 0.1   # Recon
            beta_loss = 0.1    # VICReg
            
            gamma_loss = 1.0   # ProtoNCE (Quan trọng cho PCL)
            
            loss = L_uni + alpha_loss * L_rec + beta_loss * L_vic + gamma_loss * L_proto

            opt_enc.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_enc.step()
            if sch_enc is not None:
                sch_enc.step()
            
            model.prototypes.weight.data = F.normalize(model.prototypes.weight.data, dim=1)

            # Logging
            bs = labels.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            if i % 100 == 0:
                print(f'[P1] it {i:05d} | loss {loss.item():.4f}')
                with torch.no_grad():
                    sim_pos = (z_q * z_k).sum(1).mean()
                    sim_uni = (z_q * z_u).sum(1).mean()
                    print(f'[sim] pos={sim_pos:.3f} uni={sim_uni:.3f}')
                sys.stdout.flush()

        print(f'[P1] Train Loss (total): {running_loss / max(1, n_samples):.4f}')

        # ---------- Phase 2: Classifier / Fine-tuning ----------
        if epoch >= 0: 
            FINE_TUNE_START_EPOCH = 50 
            
            is_finetuning = (epoch >= FINE_TUNE_START_EPOCH)
            
            if not is_finetuning:
                # --- CHẾ ĐỘ LINEAR PROBING ---
                model.eval()               
                set_requires_grad(model, False) 
                classifier.train()

                current_opt = opt_cls 
                current_sch = sch_cls                
            else:
                # --- CHẾ ĐỘ FULL FINE-TUNING ---
                model.train()               
                set_requires_grad(model, True)  
                classifier.train()
                
                if opt_finetune is None:
                    print(f"\n[INFO] SWITCHING TO FULL FINE-TUNING at Epoch {epoch}")
                    params = [
                        {'params': model.parameters(), 'lr': 1e-5},
                        {'params': classifier.parameters(), 'lr': 1e-3} 
                    ]
                    opt_finetune = optim.AdamW(params, weight_decay=1e-4)

                    remaining_epochs = config.epoch_num - epoch
                    sch_finetune = CosineWarmupScheduler(
                        opt_finetune, 
                        warmup=10, 
                        max_iters=remaining_epochs * len(train_classifier_loader)
                    )
                
                current_opt = opt_finetune
                current_sch = sch_finetune

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
                if current_sch is not None:
                    current_sch.step()

                bs = y.size(0)
                run_ce += ce.item() * bs
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += bs

            mode_str = "Fine-tune" if is_finetuning else "Lin-Probe"
            print(f'[P2] {mode_str} CE: {run_ce/max(1,total):.4f} | Acc: {correct/max(1,total):.4f}')
            # Validate
            _ = evaluate(model, classifier, test_loader, config.device)

            save_file = os.path.join(config.model_save_path, f'ckpt_epoch_{epoch}.pth')
            save_model(model, opt_enc, config, epoch, save_file)

            ckpt = 'ckpt_class_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(config.model_save_path, ckpt)
            save_model(classifier, opt_cls, config, epoch, save_file)

            set_requires_grad(model, True)

    # Save final
    torch.save({'encoder': model.state_dict(), 'classifier': classifier.state_dict()},
               os.path.join(config.model_save_path, f'{config.model_name}_final.pth'))