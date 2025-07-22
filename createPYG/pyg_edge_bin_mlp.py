import os
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler

def create_mapping(df, ip_to_id, family_label, label_to_id):
    for ip in pd.concat([df['Src IP'], df['Dst IP']]).unique():
        if ip not in ip_to_id:
            ip_to_id[ip] = len(ip_to_id)

    if family_label not in label_to_id:
        label_to_id[family_label] = len(label_to_id)

def create_pyg_data(df, ip_to_id, cat_label, family_label, label_to_id):    
    df['Src IP'] = df['Src IP'].map(ip_to_id)
    df['Dst IP'] = df['Dst IP'].map(ip_to_id)
      
    columns_drop=['Flow ID', 'Src Port', 'Dst Port', 'Timestamp', 'Label', 'Family_Label']
    df.drop(columns=columns_drop, inplace=True)
    if 'Cat_Label' in df.columns:
        df.drop('Cat_Label', axis=1, inplace=True)
    edges_df = df.drop(columns=['Src IP', 'Dst IP'])
    G = nx.from_pandas_edgelist(df, 'Src IP', 'Dst IP',
                                        create_using=nx.MultiDiGraph(), edge_attr=edges_df.columns.tolist())
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    B_in = torch.zeros((num_nodes, num_edges), dtype=torch.float32)
    B_out = torch.zeros((num_nodes, num_edges), dtype=torch.float32)
    for i, node in enumerate(G.nodes):
        for j, edge in enumerate(G.edges):
            if node == edge[1]:
                B_in[i, j] = 1
            if node == edge[0]:
                B_out[i, j] = 1

    edges = list(G.edges())
    edge_index = torch.tensor([[int(src), int(dst)] for src, dst in edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([list(G.edges[edge].values()) for edge in G.edges], dtype=torch.float)

    family_label_id = label_to_id[family_label]
    y = torch.tensor([cat_label], dtype=torch.long)
    family = torch.tensor([family_label_id], dtype=torch.long)

    return Data(x=edge_attr, edge_index=edge_index, B_in = B_in, B_out=B_out, family = family, y=y), num_nodes

label_to_id = {}
def process_directory(directory, cat_label):

    pyg_data_list = []
    max_nodes = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                ip_to_id = {}
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                family_label = df['Family_Label'].unique()[0]
                create_mapping(df, ip_to_id, family_label, label_to_id)
                pyg_data, n_nodes = create_pyg_data(df, ip_to_id, cat_label, family_label, label_to_id)
                pyg_data_list.append(pyg_data)
                if max_nodes < n_nodes:
                    max_nodes = n_nodes

    print("Max nodes:", max_nodes)

    return pyg_data_list

benign_edge = "E:/MinMaxScaledData/1Benign"
sms_malware_edge = "E:/MinMaxScaledData/2Smsmalware"
ransomware_edge = "E:/MinMaxScaledData/3Ransomware"
adware_edge = "E:/MinMaxScaledData/PrunedData/4Adware"
scareware_edge = "E:/MinMaxScaledData/PrunedData/5Scareware"

folder_datas = []
data_list = []

folder_datas = process_directory(benign_edge, 0)
print(f"Benign graph {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas = process_directory(sms_malware_edge, 1)
print(f"SMSMalware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas = process_directory(ransomware_edge, 1)
print(f"Ransomware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas = process_directory(adware_edge, 1)
print(f"Adware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas = process_directory(scareware_edge, 1)
print(f"Scareware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

pyg_file = "E:/MinMaxScaledData/bin_graph_data_edge_and_5d_node.pt"
print(f"Total graph: {len(data_list)}")
torch.save(data_list, pyg_file)



