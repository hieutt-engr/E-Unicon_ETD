import torch.nn as nn
from config import Config, prepare_fin, parser_process
from data_loader import data_preparing
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt

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

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x
    
class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return x
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x
    
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            data = batch['data'].type(torch.FloatTensor).to(config.device)
            labels = batch['label'].to(config.device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
def evaluate_model(model, test_loader):
    model.eval()
    label_y = []
    pre_y = []
    prob_y = []
    with torch.no_grad():
        for batch in test_loader:
            data = batch['data'].type(torch.FloatTensor).to(config.device)
            labels = batch['label'].to(config.device)
            
            outputs = model(data)
            
            pre_y = np.concatenate([pre_y, torch.max(outputs, 1)[1].cpu().numpy()], 0)
            prob_y = np.concatenate([prob_y, torch.max(outputs, 1)[0].cpu().numpy()], 0)
            label_y = np.concatenate([label_y, labels.cpu().numpy()], 0)
            
    fpr, tpr, thresholds = roc_curve(label_y, prob_y)
    roc_auc = auc(fpr, tpr)
    
    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(label_y, pre_y)
    precision = precision_score(label_y, pre_y)
    recall = recall_score(label_y, pre_y)
    f1 = f1_score(label_y, pre_y)
    
    # Print the metrics
    print('\n========== Evaluation ==========')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')
    
    draw_confusion(label_y, pre_y)
    
    return fpr, tpr, roc_auc

args = parser_process()
config = Config(args)

model_caetrans = torch.load('./model/KSM_Bestt/KSM_Bestt_model_179.pth')
model_trans = torch.load('./modelKSM_Transformer/KSM_Transformer_model_90.pth')
config.root_dir = './data/ksm_transformer_best_result'

# print(model)

train_loader, test_loader = data_preparing(config, args)

model_caetrans.eval()
model_trans.eval()

fpr, tpr, roc_auc = evaluate_model(model_caetrans, test_loader)
np.savez('./metrics/caetrans_metrics.npz', fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    
fpr, tpr, roc_auc = evaluate_model(model_trans, test_loader)
np.savez('./metrics/trans_metrics.npz', fpr=fpr, tpr=tpr, roc_auc=roc_auc)

# Define models
cnn_lstm_model = CNN_LSTM(input_dim=37, hidden_dim=128, output_dim=2, num_layers=3).to(config.device)
lstm_model = LSTM(input_dim=37, hidden_dim=128, output_dim=2, num_layers=3).to(config.device)
rnn_model = RNN(input_dim=37, hidden_dim=128, output_dim=2, num_layers=3).to(config.device)
gru_model = GRU(input_dim=37, hidden_dim=128, output_dim=2, num_layers=3).to(config.device)

lr = 0.001
epoch = 200

# Train models
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_lstm_model.parameters(), lr=lr)
train_model(cnn_lstm_model, train_loader, criterion, optimizer, num_epochs=epoch)
fpr_cnn_lstm, tpr_cnn_lstm, roc_auc_cnn_lstm = evaluate_model(cnn_lstm_model, test_loader)
np.savez('./metrics/cnn_lstm_metrics.npz', fpr=fpr_cnn_lstm, tpr=tpr_cnn_lstm, roc_auc=roc_auc_cnn_lstm)

optimizer = torch.optim.Adam(lstm_model.parameters(), lr=lr)
train_model(lstm_model, train_loader, criterion, optimizer, num_epochs=epoch)
fpr_lstm, tpr_lstm, roc_auc_lstm = evaluate_model(lstm_model, test_loader)
np.savez('./metrics/lstm_metrics.npz', fpr=fpr_lstm, tpr=tpr_lstm, roc_auc=roc_auc_lstm)

optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
train_model(rnn_model, train_loader, criterion, optimizer, num_epochs=epoch)
fpr_rnn, tpr_rnn, roc_auc_rnn = evaluate_model(rnn_model, test_loader)
np.savez('./metrics/rnn_metrics.npz', fpr=fpr_rnn, tpr=tpr_rnn, roc_auc=roc_auc_rnn)

optimizer = torch.optim.Adam(gru_model.parameters(), lr=lr)
train_model(gru_model, train_loader, criterion, optimizer, num_epochs=epoch)
fpr_gru, tpr_gru, roc_auc_gru = evaluate_model(gru_model, test_loader)
np.savez('./metrics/gru_metrics.npz', fpr=fpr_gru, tpr=tpr_gru, roc_auc=roc_auc_gru)