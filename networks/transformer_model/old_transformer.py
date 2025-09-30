import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pandas as pd
import pytz
import math
import numpy as np

from .add_components import ConvAutoencoder1D, DNN, Autoencoder1D

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPredictor(nn.Module):
    def __init__(self, config):
        super(TransformerPredictor, self).__init__()
        self.pad_size = config.pad_size

        if config.mode == 'cae':
            self.cae = ConvAutoencoder1D(1, config.dout_mess).to(config.device)
        elif config.mode == 'dnn':
            self.dnn = DNN(28, config.dout_mess).to(config.device)
        elif config.mode == 'ae':
            self.ae = Autoencoder1D(28, config.dout_mess).to(config.device)
        
        self.dout_mess = config.dout_mess
        self.mode = config.mode
        self.device = config.device
        
        self.position_embedding = PositionalEncoding(config.d_model, dropout=0.0, max_len=config.max_time_position).to(config.device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers).to(config.device)
        self.fc = nn.Linear(config.d_model, config.classes_num).to(config.device)
        
        self.seq_len = config.window_size

    def forward(self, data):
        # Normal case
        # x = data.permute(2, 0, 1)
        
        # With AE 
        x = data
        
        mask = torch.from_numpy(np.full((data.shape[0], self.seq_len), False))
        # print("X: ", x.shape)
        # X:  torch.Size([128, 37, 28])
        
        if self.mode == 'cae':
            # Conv Autoencoder 1D =================================================   
            cae_out = torch.empty((x.shape[0], self.dout_mess, 0)).to(self.device)
            for i in range(self.pad_size):
                # shape of x[:, i, :] is (batch_size, 28)
                # tmp = self.cae(x[:, i, :]).unsqueeze(2)
                tmp = self.cae(x[:, i:i+1, :]).unsqueeze(2)
                cae_out = torch.concat((cae_out, tmp), dim=2)
                # sharp of cae_out is (batch_size, 20, 36)
                    
            # print("CAE OUT: ", cae_out.shape)
            # CAE OUT:  torch.Size([128, 20, 37]
            
            x = cae_out.permute(2, 0, 1)
            # sharp of x is (36, batch_size, 20)
        elif self.mode == 'dnn':
            dnn_out = torch.empty((x.shape[0], self.dout_mess, 0)).to(self.device)
            for i in range(self.pad_size):
                # shape of x[:, i, :] is (batch_size, 28)
                # tmp = self.cae(x[:, i, :]).unsqueeze(2)
                tmp = self.dnn(x[:, i, :]).unsqueeze(2)
                dnn_out = torch.concat((dnn_out, tmp), dim=2)
                
            x = dnn_out.permute(2, 0, 1)
        else:
            x = x.permute(1, 0, 2)
            
        out = self.position_embedding(x)
        out2 = self.transformer_encoder(out, src_key_padding_mask=mask.to(self.device))
        out = out2.permute(1, 0, 2)
        out = torch.sum(out, 1)
        out = self.fc(out)
        return out
    
class TransformerPredictor2(nn.Module):
    def __init__(self, config):
        super(TransformerPredictor2, self).__init__()
        self.pad_size = config.pad_size
        
        self.dout_mess = config.dout_mess
        self.mode = config.mode
        self.device = config.device
        
        self.position_embedding = PositionalEncoding(config.d_model, dropout=0.0, max_len=config.max_time_position).to(config.device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=config.d_model, nhead=config.nhead).to(config.device)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=config.num_layers).to(config.device)
        self.fc = nn.Linear(config.d_model, config.classes_num).to(config.device)
        
        self.seq_len = config.window_size

    def forward(self, data):
        # Normal case
        # x = data.permute(2, 0, 1)
        
        # With AE 
        x = data
        
        mask = torch.from_numpy(np.full((data.shape[0], self.seq_len), False))
        # print("X: ", x.shape)
        # X:  torch.Size([128, 37, 28])
        
        
        x = x.permute(1, 0, 2)
            
        out = self.position_embedding(x)
        out2 = self.transformer_encoder(out, src_key_padding_mask=mask.to(self.device))
        out = out2.permute(1, 0, 2)
        out = torch.sum(out, 1)
        out = self.fc(out)
        return out