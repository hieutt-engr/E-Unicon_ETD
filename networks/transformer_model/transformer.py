import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed import DataEmbedding, PositionalEmbedding
from .attn import AnomalyAttention, AttentionLayer
from .add_components import ConvAutoencoder1D, DNN, Autoencoder1D

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0, activation='gelu', output_attention=True, max_time_position=5000):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        # self.embedding = DataEmbedding(enc_in, d_model, dropout)
        
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = x + self.position_embedding(x)
        # enc_out, series, prior, sigmas = self.encoder(enc_out)
        # enc_out = self.projection(enc_out)
        
        enc_out, _, _, _ = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        return enc_out
    
class TransformerAnomalyPredictor(nn.Module):
    d_model = 512
    d_out = 16
    pr = True
    use_embedding = False
    
    embedding_size = 16
    window_size = 37
    
    def __init__(self, config):
        print(f"init TransformerAnomalyPredictor with {config.classes_num} classes")
        super(TransformerAnomalyPredictor, self).__init__()
        self.d_out = config.d_model
        self.layer = config.num_layers
        self.use_embedding = config.use_embedding
        self.embedding_size = config.embedding_size
        self.window_size = config.window_size
        self.mode = config.mode
        self.device = config.device

        self.encoder = AnomalyTransformer(win_size=self.window_size, enc_in=self.d_out, c_out=self.d_out, d_model=self.d_out, n_heads=self.d_out, e_layers=self.layer, d_ff=512, dropout=0.5, output_attention=True, max_time_position=10000).to(self.device)

        self.add_norm = config.add_norm
        self.num_class = config.classes_num

        self.cae = ConvAutoencoder1D(1, self.d_out).to(self.device)
        self.dnn = DNN(28, config.dout_mess).to(config.device)
        
        self.emb = nn.Sequential(
            nn.Linear(self.window_size*self.d_out, self.window_size*self.d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.BatchNorm1d(self.window_size*self.d_out),
            nn.Linear(self.window_size*self.d_out, self.embedding_size),
        ).to(self.device) if(self.use_embedding) else None
        
        self.fc = nn.Linear(self.embedding_size if self.use_embedding else self.d_out, self.num_class).to(config.device)
        
    def forward(self, x):
        
        if self.mode == 'cae':
            # Conv Autoencoder 1D =================================================   
            cae_out = torch.empty((x.shape[0], self.d_out, 0)).to(self.device)
            for i in range(x.shape[1]):
                # shape of x[:, i, :] is (batch_size, 28)
                # tmp = self.cae(x[:, i, :]).unsqueeze(2)
                tmp = self.cae(x[:, i:i+1, :]).unsqueeze(2)
                cae_out = torch.concat((cae_out, tmp), dim=2)
                # sharp of cae_out is (batch_size, 20, 36)
                    
            # print("CAE OUT: ", cae_out.shape)
            # CAE OUT:  torch.Size([128, 20, 37]
            
            x = cae_out.permute(0, 2, 1)
        elif self.mode == 'dnn':
            dnn_out = torch.empty((x.shape[0], self.d_out, 0)).to(self.device)
            for i in range(x.shape[1]):
                # shape of x[:, i, :] is (batch_size, 28)
                # tmp = self.cae(x[:, i, :]).unsqueeze(2)
                tmp = self.dnn(x[:, i, :]).unsqueeze(2)
                dnn_out = torch.concat((dnn_out, tmp), dim=2)
                
            x = dnn_out.permute(0, 2, 1)
        
        # Transformer
        x = self.encoder(x)
        x = self.emb(x.view(x.size(0), -1)) if (self.use_embedding) else torch.sum(x, 1)
        x = F.normalize(x, p=2, dim=-1) if (self.add_norm) else x
        x = torch.sigmoid(self.fc(x))
        return x
    
    def calculate_attention(self, features):
        max_pool = torch.max(features, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(features, dim=1, keepdim=True)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = torch.sigmoid(self.fc_att(combined))
        return attention_map