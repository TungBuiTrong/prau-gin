import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import networkx as nx
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from torch_geometric.nn import Linear, BatchNorm, GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

class StarGCN(nn.Module):
    def __init__(self, n_classes= 5, n_hidden = 32, edge_dim = 63, 
                 num_gnn_layers = 2, dropout = 0.0, final_dropout= 0.25):

        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.dropout_rate = dropout 
        self.final_dropout = final_dropout
        self.batch_norms = nn.ModuleList()
        
        self.edge_emb = nn.Sequential(
            Linear(edge_dim, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU()
        )
        
        self.node_emb = nn.Sequential(
            Linear(2 * n_hidden, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU()
        )

        self.convs = nn.ModuleList()

        for _ in range(self.num_gnn_layers):
            
            conv = GCNConv(in_channels=self.n_hidden, out_channels=self.n_hidden, 
                           normalize=True, add_self_loops=True)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.final_mlp = nn.Sequential(
            Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(self.final_dropout)
        )
        
        self.output_layer = Linear(n_hidden, n_classes)

    def forward(self, x, edge_index, B_in, B_out):

        x = self.edge_emb(x)
        mat1 = torch.matmul(B_in, x) 
        mat2 = torch.matmul(B_out, x)
        x = self.node_emb(torch.cat((mat1, mat2), dim=1))

        for i in range(self.num_gnn_layers):
            x = self.batch_norms[i](self.convs[i](x, edge_index))
            x = F.dropout(x, self.dropout_rate, training=self.training)
            x = F.relu(x)

        x = self.final_mlp(x)
        x_mean = torch.mean(x, dim=0, keepdim=True, dtype=torch.float) 
        output = self.output_layer(x_mean)
        
        return output

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.B_in, data.B_out)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def evaluate(model, loader, criterion):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total_samples = 0
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
      for data in loader:       
        output = model(data.x, data.edge_index, data.B_in, data.B_out)
        preds = output.argmax(dim=1)
        target = data.y.view(-1)
        if target.dim() == 0:
            target = target.unsqueeze(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(target.cpu().numpy())

        loss = criterion(output, data.y)
        total_loss += loss.item()
        
    average_loss = total_loss / len(loader)
        
    cm = confusion_matrix(all_labels, all_preds)
    metrics_df = pd.DataFrame({
        "accuracy": [accuracy_score(all_labels, all_preds)],
        "macro_precision": [precision_score(all_labels, all_preds, average='macro', zero_division=np.nan)],
        "macro_recall": [recall_score(all_labels, all_preds, average='macro', zero_division=np.nan)],
        "macro_f1": [f1_score(all_labels, all_preds, average='macro', zero_division=np.nan)],
        "micro_precision": [precision_score(all_labels, all_preds, average='micro', zero_division=np.nan)],
        "micro_recall": [recall_score(all_labels, all_preds, average='micro', zero_division=np.nan)],
        "micro_f1": [f1_score(all_labels, all_preds, average='micro', zero_division=np.nan)],
        "weighted_precision": [precision_score(all_labels, all_preds, average='weighted', zero_division=np.nan)],
        "weighted_recall": [recall_score(all_labels, all_preds, average='weighted', zero_division=np.nan)],
        "weighted_f1": [f1_score(all_labels, all_preds, average='weighted', zero_division=np.nan)],
    }) 

    return metrics_df, cm, average_loss
    
pyg_file = 'D:/Research/NCS/Dataset/PrunedData/graph_data_edge_and_mlp_node.pt'
list_data = torch.load(pyg_file)
labels = []
for data in list_data:
    labels.extend(data.y.tolist())

print('Prepare data input model successful !')

label_counts = Counter(labels)
sorted_counts = dict(sorted(label_counts.items()))

total_samples = len(labels)
class_weights = {label: total_samples / count for label, count in sorted_counts.items()}
class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)

for label, weight in enumerate(class_weights_tensor):
    print(f"Class {label}: count = {sorted_counts[label]}, weight = {weight:.4f}")
    
w_results = "D:/Research/NCS/Dataset/Results/cat_classification_5d_weight.csv"
w_cm_results ="D:/Research/NCS/Dataset/Results/cat_classification_5d_weight_cm.csv"
lr = 0.001
epochs = 1000

for i in range(30):
    model = StarGCN(edge_dim=63, n_classes=5, n_hidden=64, edge_updates=False, final_dropout=0.25)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight = class_weights_tensor)
  
    train_data, test_data = train_test_split(list_data, test_size=0.2, random_state=42, stratify=labels)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)
    total_loss_epochs = 0.0

    train_metrics = pd.DataFrame()
    test_metrics = pd.DataFrame()   
    w_f1_final = 0.0  
    w_final_cm = pd.DataFrame() 
    w_final_metrics = pd.DataFrame()
    train_time = 0
    test_time = 0
    for epoch in range (epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        start = time.perf_counter()
        train_loss = train(model, temp_data, criterion, optimizer)
        train_metrics, _, _ = evaluate(model, temp_data, criterion)
        end = time.perf_counter()
        train_time = train_time + 1000*(end - start)

        start = time.perf_counter()
        test_metrics, cm, test_loss  = evaluate(model, test_data, criterion)
        end = time.perf_counter()
        test_time = test_time + 1000*(end - start)

        df_cm = pd.DataFrame(cm, index=range(5), columns=range(5))

        total_loss_epochs = total_loss_epochs + test_loss
         
        if w_f1_final < test_metrics["weighted_f1"].iloc[0]:
            w_f1_final = test_metrics["weighted_f1"].iloc[0]
            w_final_metrics = test_metrics
            w_final_cm = df_cm

        print(f"Train loss : {train_loss} --- Test loss: {test_loss}")
        print(f'Train accuracy: {train_metrics["accuracy"].iloc[0] * 100: .2f} -- Test accuracy: {test_metrics["accuracy"].iloc[0] * 100: .2f} -- Recall: {test_metrics["weighted_recall"].iloc[0] * 100: .2f} -- Precision: {test_metrics["weighted_precision"].iloc[0] * 100: .2f} -- F1 score: {test_metrics["weighted_f1"].iloc[0] * 100: .2f}')
        print("-" * 50)

    w_final_metrics['elapsed_time'] = train_time
    w_final_metrics['test_time'] = test_time
    w_final_metrics['eperiment'] = 'CAT_GCN'


    if not os.path.isfile(w_results):
        w_final_metrics.to_csv(w_results, index=False)
    else:
        w_final_metrics.to_csv(w_results, mode='a', header=False, index=False)
    
    if not os.path.isfile(w_cm_results):
        w_final_cm.to_csv(w_cm_results, index=False)
    else:
        w_final_cm.to_csv(w_cm_results, mode='a', header=False, index=False)
        
    print(f'Loss average: {total_loss_epochs / epochs}')
    print(f"Train accuracy: {train_metrics["accuracy"].iloc[0]}")

    print("Best model based on weight:")    
    print(f'Test accuracy: {w_final_metrics["accuracy"].iloc[0] * 100: .2f} -- Recall: {w_final_metrics["weighted_recall"].iloc[0] * 100: .2f} -- Precision: {w_final_metrics["weighted_precision"].iloc[0] * 100: .2f} -- F1 score: {w_final_metrics["weighted_f1"].iloc[0] * 100: .2f}')  
    print(f'Confusion Matrix:\n{w_final_cm}')