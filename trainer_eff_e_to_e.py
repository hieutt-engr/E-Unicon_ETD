import torch
import torch.nn as nn
import math
from dataset import DatasetPrepare
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import optuna
import os
import time
import warnings
import argparse
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
import pytz
import datetime
from logger import Logger
from networks.transformer_model.old_transformer import TransformerPredictor
from networks.transformer_model.scheduler import CosineWarmupScheduler
from networks.transformer_model.transformer import TransformerAnomalyPredictor
# from efficient_net import ConEfficientNet, LinearClassifier
from networks.efficient_net_ce import ConEfficientNet
from losses import SupConLoss, UniConLoss
from config import Config, prepare_fin, parser_process
from utils import *
from data_loader import data_preparing
import matplotlib.pyplot as plt

import shap
import lime
import lime.lime_tabular
import time

warnings.filterwarnings("ignore")

layout = {
    "CAE-Transformer": {
        "losses": ["Multiline", ["loss/train", "loss/test"]],
        "learning rate": ["Multiline", ["learning_rate/lr"]],
        "auc": ["Multiline", ["AUC"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
    },
}

if __name__ == '__main__':
    args = parser_process()
    config = Config(args)
    prepare_fin(config)
    
    # Set print options
    # torch.set_printoptions(profile="full")
    
    if config.log_mode == 'train':
        logger = Logger('./logs/transformer_'+ datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y-%m-%d %H:%M:%S"), layout)
    elif config.log_mode == 'test':
        logger = Logger('./logs/transformer_test', layout)
        print("Test mode")
        
    train_loader, test_loader = data_preparing(config, args)
    
    if config.model == 'old':
        model = TransformerPredictor(config)
    elif config.model == 'new':
        model = TransformerAnomalyPredictor(config)
        cal_model_size(model)
        print_model_info_pytorch(model)
    elif config.model == 'efficientnet':
        model = ConEfficientNet(embedding_dim=1792, num_classes=10, pretrained=False)
        model = model.to(config.device)
    
    start_epoch = -1
        
    loss_func = nn.CrossEntropyLoss().to(config.device)
    criterion = nn.CrossEntropyLoss().to(config.device)
    
    opt = optim.AdamW(model.parameters(), lr=config.lr)
    lr_scheduler = CosineWarmupScheduler(opt, warmup=50, max_iters=config.epoch_num*len(train_loader))
        
    mapk_scores = []
    auc_scores = []
    
    for epoch in range(start_epoch + 1, config.epoch_num):
        start_time = time.time()
        fin = open(config.result_file, 'a')
        print('--- epoch ', epoch)
        fin.write('-- epoch ' + str(epoch) + '\n')
        
        epoch_train_loss = 0  # Initialize epoch training loss
        model.train()
        
        correct_predictions = 0
        total_predictions = 0
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
    
        for i, sample_batch in enumerate(train_loader):
            batch_data = sample_batch['data'].type(torch.FloatTensor).to(config.device)
            # batch_data = sample_batch['data'].type(torch.IntTensor).to(config.device)
            # batch_mask = sample_batch['mask'].to(config.device)
            batch_label = sample_batch['label'].to(config.device)
            
            outputs = model(batch_data)                    # logits from model
            loss = criterion(outputs, batch_label)
            losses.update(loss.item(), batch_label.size(0))
            epoch_train_loss += loss.item() 
            
            # print("DATA: ", batch_data)
            # print("LABEL: ",batch_label)
            # print("OUT: ", out)
            # loss = loss_func(out, batch_label)
            
            # print("LOSS: ", loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == batch_label).sum().item()
            total_predictions += batch_label.size(0)
            
            if i % 20 == 0:
                print('iter {} loss: '.format(i), loss.item())
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        accuracy_train = correct_predictions / total_predictions

        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {accuracy_train:.4f}')
             
        torch.save(model, (config.model_save_path + config.model_name + '_model_{}.pth').format(epoch))
        
        train_time = time.time()
        print(f'Training time: {train_time - start_time} seconds')
        start_time = time.time()
        
        # test
        model.eval()
        label_y = []
        pre_y = []
        total_test_loss = 0  # Initialize total test loss
        
        losses = AverageMeter()
        total_pred = np.array([], dtype=int)
        total_label = np.array([], dtype=int) 
        
        with torch.no_grad():
            for j, test_sample_batch in enumerate(test_loader):
                test_data = test_sample_batch['data'].type(torch.FloatTensor).to(config.device)
                # test_mask = test_sample_batch['mask'].to(config.device)
                test_label = test_sample_batch['label'].to(config.device)
                
                outputs = model(test_data)
                
                loss = criterion(outputs, test_label)
                losses.update(loss.item(), test_label.size(0))
                
                _, pred = outputs.topk(1, 1, True, True)
                pred = pred.t().cpu().numpy().squeeze(0)

                total_pred = np.concatenate((total_pred, pred), axis=0)
                total_label = np.concatenate((total_label, test_label.cpu().numpy()), axis=0)
            
            acc = accuracy_score(total_label, total_pred)
            acc = acc * 100
            f1 = f1_score(total_label, total_pred, average='weighted')
            precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
            recall = recall_score(total_label, total_pred, average='weighted')
            conf_matrix = confusion_matrix(total_label, total_pred)
            
            # Print results
            print(f'Accuracy: {acc:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print('Confusion Matrix:')
            print(conf_matrix)
            
            
            # test_time = time.time()
            # print(f'Testing time: {test_time - start_time} seconds')
            # print('label_y: ', label_y)
            # print('pre_y: ', pre_y)

            # avg_test_loss = total_test_loss / len(test_loader)
            
            # # Calculate AUC
            # auc_value = roc_auc_score(label_y, pre_y)
            # auc_scores.append(auc_value)
            # print(f'Epoch {epoch+1}/{config.epoch_num}, AUC: {auc_value:.4f}')
            
            # draw_confusion(label_y, pre_y)
            # f1_test, accuracy_test = write_result(fin, label_y, pre_y)
        fin.close()
    
        # ============ TensorBoard logging ============# 
        # info = {
        #     'loss/train': avg_train_loss,
        #     'loss/test': avg_test_loss,
        #     'accuracy/train': accuracy_train,
        #     'accuracy/test': accuracy_test,
        #     'learning_rate/lr': opt.param_groups[0]['lr'],
        #     'AUC': auc_value
        # }
        
        # for tag, value in info.items():
        #     # Only apply .cpu() to tensors, skip for floats
        #     if isinstance(value, torch.Tensor):
        #         value = value.cpu()
        #     logger.scalar_summary(tag, value, epoch + 1)
        
    # auc_scores_df = pd.DataFrame(auc_scores, columns=['AUC'])
    # auc_scores_df.to_csv('auc_scores.csv', index=False)