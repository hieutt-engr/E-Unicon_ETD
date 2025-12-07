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
from networks.efficient_lrl_prototype import ConEfficientNet, LinearClassifier
from losses import UniConLoss_ETD, recon_loss, vicreg_loss, ProtoNCELoss
from config import Config, prepare_fin, parser_process
from utils import AverageMeter, save_model, generate_prototype_universum
from data_loader_eff import data_preparing

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
    cm   = confusion_matrix(y_true, y_pred)

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

    train_loader, train_classifier_loader, test_loader = data_preparing(config, args)
    print('finish load data')

    assert config.model == 'efficientnet', "This script supports efficientnet only."
    
    num_prototypes = 1000
    model = ConEfficientNet(
        embedding_dim=1792, feat_dim=128, head='mlp',
        pretrained=getattr(config, 'pretrained', False),
        num_prototypes=num_prototypes
    ).to(config.device)
    
    classifier = LinearClassifier(input_dim=128, num_classes=config.n_classes).to(config.device)

    if str(config.method).lower() == 'unicon':
        con_loss = UniConLoss_ETD(temperature=config.temp).to(config.device)

    proto_loss_fn = ProtoNCELoss(temperature=0.1).to(config.device)

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

    start_epoch = -1
    os.makedirs(config.model_save_path, exist_ok=True)

    for epoch in range(start_epoch + 1, config.epoch_num):
        print(f'\n=== Epoch {epoch} / {config.epoch_num-1} ==='); sys.stdout.flush()

        if epoch == 0 or epoch % 5 == 0:
            print("Clustering to update prototypes...")
            model.eval()
            all_feats = []
            with torch.no_grad():
                for images, _ in train_loader:
                    img = images[0].float().to(config.device)
                    feat = model(img)
                    all_feats.append(feat.cpu().numpy())
            
            all_feats = np.concatenate(all_feats, axis=0)
            
            kmeans = KMeans(n_clusters=num_prototypes, random_state=epoch).fit(all_feats)
            
            new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float).to(config.device)
            new_centers = F.normalize(new_centers, dim=1)
            model.prototypes.weight.data.copy_(new_centers)
            print("Prototypes updated.")

        set_requires_grad(model, True); model.train()
        model.normalize_prototypes()
        
        running_loss, n_samples = 0.0, 0

        for i, (images, labels) in enumerate(train_loader):
            q, k = images[0].float().to(config.device), images[1].float().to(config.device)
            y = labels.to(config.device)

            z1, h1, x_hat_q, p_q = model.forward_lrl(q)
            z2, h2, x_hat_k, p_k = model.forward_lrl(k)
            feats = torch.stack([z1, z2], dim=1)

            z_u = generate_prototype_universum(z1, model.prototypes, alpha=0.5)

            L_uni = con_loss(features=feats, labels=y, universum=z_u)

            L_rec = 0.0
            if x_hat_q is not None:
                L_rec = recon_loss(x_hat_q, q)

            L_vic = 0.0
            if p_q is not None and p_k is not None:
                L_vic = vicreg_loss(p_q, p_k)

            L_proto = (proto_loss_fn(z1, model.prototypes) + proto_loss_fn(z2, model.prototypes)) / 2

            alpha_loss = 0.2
            beta_loss = 0.5
            gamma_loss = 0.5
            loss = L_uni + alpha_loss * L_rec + beta_loss * L_vic + gamma_loss * L_proto

            opt_enc.zero_grad(set_to_none=True)
            loss.backward()
            opt_enc.step()
            sch_enc.step()
            
            model.normalize_prototypes()

            bs = y.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

            if i % 100 == 0:
                print(f'[P1] it {i:05d} | loss {loss.item():.4f}')
                sys.stdout.flush()

        print(f'[P1] Train Loss (contrastive): {running_loss / max(1, n_samples):.4f}')

        if epoch >= config.epoch_start_classifier:
            model.eval(); set_requires_grad(model, False)
            classifier.train()

            run_ce, correct, total = 0.0, 0, 0

            for i, (images, labels) in enumerate(train_classifier_loader):
                x = images[0] if isinstance(images, (tuple, list)) else images
                x = x.float().to(config.device)
                y = labels.to(config.device)

                with torch.no_grad():
                    feat = model(x)
                logits = classifier(feat)
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

            _ = evaluate(model, classifier, test_loader, config.device)

            save_file = os.path.join(config.model_save_path, f'ckpt_epoch_{epoch}.pth')
            save_model(model, opt_enc, config, epoch, save_file)

            ckpt = 'ckpt_class_epoch_{}.pth'.format(epoch)
            save_file = os.path.join(config.model_save_path, ckpt)
            save_model(classifier, opt_cls, config, epoch, save_file)

            set_requires_grad(model, True)

    torch.save({'encoder': model.state_dict(), 'classifier': classifier.state_dict()},
               os.path.join(config.model_save_path, f'{config.model_name}_final.pth'))