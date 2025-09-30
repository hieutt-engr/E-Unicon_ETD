import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Autoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Autoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return F.relu(x)

class ConvAutoencoder1D(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvAutoencoder1D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            # Original: input_dim -> 32
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            # Original: 32 -> 64
            nn.ReLU()
        )
        self.fc = nn.Linear(128*28, output_dim)
        # Original: 64*28
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class DNN(nn.Module):
    def __init__(self, d_in, d_out):  # config.slsum_count, config.dnn_out_d
        super(DNN, self).__init__()
        self.l1 = nn.Linear(d_in, 128)
        self.l2 = nn.Linear(128, 64)
        self.l3 = nn.Linear(64, d_out)

    def forward(self, x):
        # print('x: ', x.numpy ()[0])
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        # print('dnn out: ', out.detach().numpy()[0])
        return out