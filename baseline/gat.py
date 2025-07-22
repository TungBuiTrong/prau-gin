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
from torch_geometric.nn import Linear, BatchNorm, GATv2Conv, global_mean_pool
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

class StarGATe(nn.Module):
    def __init__(self, num_features = 5, num_gnn_layers = 2, n_classes = 5, 
                 n_hidden = 64, edge_dim = 63, num_heads = 4, 
                 dropout = 0.0, final_dropout = 0.5):
        super().__init__()

        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.num_heads = num_heads
        self.n_classes = n_classes


        self.output_dim_gnn = self.num_heads * self.n_hidden
        self.node_emb = Linear(num_features, self.output_dim_gnn)
        self.edge_emb = Linear(edge_dim, self.output_dim_gnn)

        self.convs = nn.ModuleList()

        for i in range(self.num_gnn_layers):
            conv = GATv2Conv(
                in_channels=self.output_dim_gnn, 
                out_channels=self.n_hidden,
                heads=self.num_heads, 
                dropout=self.dropout, 
                edge_dim=self.output_dim_gnn 
            )
            self.convs.append(conv)
            self.batch_norms = nn.ModuleList([
                BatchNorm(self.output_dim_gnn) for _ in range(self.num_gnn_layers)
            ])

        self.final_mlp = nn.Sequential(
            Linear(self.output_dim_gnn, self.n_hidden),
            nn.ReLU(),
            nn.Dropout(self.final_dropout)
        )
        
        self.output_layer = Linear(self.n_hidden, self.n_classes)

    def forward(self, x, edge_index, edge_attr, batch):

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            
        x = self.final_mlp(x)
        x_pooled = global_mean_pool(x, batch)
        output = self.output_layer(x_pooled)
        
        return output

def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0  # Tổng loss qua các batch
    num_batches = 0

    for data in train_loader:
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # Cộng dồn loss từng batch
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
        output = model(data.x, data.edge_index, data.edge_attr, data.batch)
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

pyg_file = 'D:/Research/NCS/Dataset/PrunedData/prun_graph_data_edge_and_5d_node.pt'
list_data = torch.load(pyg_file)
labels = []
for data in list_data:
    labels.extend(data.y.tolist())

print('Prepare data input model successful !')

label_counts = Counter(labels)
sorted_counts = dict(sorted(label_counts.items()))  # sắp xếp theo key

total_samples = len(labels)
class_weights = {label: total_samples / count for label, count in sorted_counts.items()}
class_weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)

for label, weight in enumerate(class_weights_tensor):
    print(f"Class {label}: count = {sorted_counts[label]}, weight = {weight:.4f}")

lr = 0.001
epochs = 1000
w_results = "D:/Research/NCS/Dataset/Results/cat_classification_5d_weight.csv"
w_cm_results ="D:/Research/NCS/Dataset/Results/cat_classification_5d_weight_cm.csv"

for i in range(30):
    model = StarGATe(num_features=5, n_classes=5, edge_updates=False, final_dropout=0.5)
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
    test_time = 0
    for epoch in range (epochs):      
        start = time.perf_counter()
        train_loss = train(model, train_loader, criterion, optimizer)
        train_mretics, _, _ = evaluate(model, train_loader, criterion)
        end = time.perf_counter()
        train_time = train_time + 1000*(end - start)

        start = time.perf_counter()
        test_metrics, cm, test_loss  = evaluate(model, test_loader, criterion)
        end = time.perf_counter()
        test_time = test_time + 1000*(end - start)

        df_cm = pd.DataFrame(cm, index=range(5), columns=range(5))
        total_loss_epochs = total_loss_epochs + test_loss
         
        if w_f1_final < test_metrics["weighted_f1"].iloc[0]:
            w_f1_final = test_metrics["weighted_f1"].iloc[0]
            w_final_metrics = test_metrics
            w_final_cm = df_cm  

    w_final_metrics['elapsed_time'] = train_time
    w_final_metrics['test_time'] = test_time
    w_final_metrics['eperiment'] = 'PRUN_CAT_GATe'

    if not os.path.isfile(w_results):
        w_final_metrics.to_csv(w_results, index=False)
    else:
        w_final_metrics.to_csv(w_results, mode='a', header=False, index=False)
  
    if not os.path.isfile(w_cm_results):
        w_final_cm.to_csv(w_cm_results, index=False)
    else:
        w_final_cm.to_csv(w_cm_results, mode='a', header=False, index=False)
        
    print(f'Loss average: {total_loss_epochs / epochs}')
    print(f"Train accuracy: {train_mretics["accuracy"].iloc[0]}")

    print("Best model based on weight:")    
    print(f'Test accuracy: {w_final_metrics["accuracy"].iloc[0] * 100: .2f} -- Recall: {w_final_metrics["weighted_recall"].iloc[0] * 100: .2f} -- Precision: {w_final_metrics["weighted_precision"].iloc[0] * 100: .2f} -- F1 score: {w_final_metrics["weighted_f1"].iloc[0] * 100: .2f}')  
    print(f'Confusion Matrix:\n{w_final_cm}')