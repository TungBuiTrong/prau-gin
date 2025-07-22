import os
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
import networkx as nx
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
import matplotlib.pyplot as plt

def create_mapping(df, ip_to_id, family_label, label_to_id):
    for ip in pd.concat([df['Src IP'], df['Dst IP']]).unique():
        if ip not in ip_to_id:
            ip_to_id[ip] = len(ip_to_id)
    
    if family_label not in label_to_id:
        label_to_id[family_label] = len(label_to_id)

def compute_node_features(G):
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True)    
    closeness_centrality = nx.closeness_centrality(G)
    G_simple = nx.DiGraph(G)
    clustering_coefficient = nx.clustering(G_simple.to_undirected(as_view=True), nodes=G_simple.nodes())    
    node_features = []
    for node in sorted(G.nodes()):
        features = [
            in_degree.get(node, 0),            
            out_degree.get(node, 0),           
            betweenness_centrality.get(node, 0),  
            closeness_centrality.get(node, 0),     
            clustering_coefficient.get(node, 0)    
        ]
        node_features.append(features)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    
    scaler = MinMaxScaler()
    node_features_scaled = scaler.fit_transform(node_features_tensor.numpy())
    node_features_tensor = torch.tensor(node_features_scaled, dtype=torch.float)
    
    return node_features_tensor

def one_hot_node(graph):
    num_nodes = len(graph.nodes)
    feature_matrix = torch.zeros(num_nodes, 1024)

    node_to_index = {node: idx for idx, node in enumerate(graph.nodes)}

    for node in graph.nodes:
        index = node_to_index[node]
        feature_matrix[index, node] = 1
    
    return feature_matrix

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
    n_nodes = G.number_of_nodes()
    
    edges = list(G.edges())
    edge_index = torch.tensor([[int(src), int(dst)] for src, dst in edges], dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor([list(G.edges[edge].values()) for edge in G.edges], dtype=torch.float)
    node_features_tensor = compute_node_features(G)
    #node_features_tensor = one_hot_node(G)

    family_label_id = label_to_id[family_label]
    y = torch.tensor([cat_label], dtype=torch.long)
    family = torch.tensor([family_label_id], dtype=torch.long)

    return Data(x=node_features_tensor, edge_index=edge_index, edge_attr=edge_attr, family = family, y=y), n_nodes

label_to_id = {}
def process_directory(directory, cat_label, start_graph_id=0, output_mapping_file=None):
    pyg_data_list = []
    max_nodes = 0
    graph_id_counter = start_graph_id
    graph_id_mapping = []

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                ip_to_id = {}
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                family_label = df['Family_Label'].unique()[0]
                create_mapping(df, ip_to_id, family_label, label_to_id)
                pyg_data, n_nodes = create_pyg_data(df, ip_to_id, cat_label, family_label, label_to_id)

                pyg_data.graphid = torch.tensor([graph_id_counter], dtype=torch.long)            

                graph_id_mapping.append({'File Name': file, 'Graph ID': graph_id_counter})                
                graph_id_counter += 1
                pyg_data_list.append(pyg_data)
                if max_nodes < n_nodes:
                    max_nodes = n_nodes

    print("Max nodes:", max_nodes)
    
    if output_mapping_file:
        mapping_df = pd.DataFrame(graph_id_mapping)
        mapping_df.to_csv(output_mapping_file, mode='a', header=not os.path.exists(output_mapping_file), index=False)
        print(f"Graph ID mapping saved to {output_mapping_file}")

    return pyg_data_list, graph_id_counter

output_mapping_csv_file = "D:/Research/NCS/Dataset/graph_id_to_filename_mapping.csv"
if os.path.exists(output_mapping_csv_file):
    os.remove(output_mapping_csv_file)
    print(f"Removed existing mapping file: {output_mapping_csv_file}")


benign_edge = "D:/Research/NCS/Dataset/PrunedData/1Benign"
sms_malware_edge = "D:/Research/NCS/Dataset/PrunedData/2Smsmalware"
ransomware_edge = "D:/Research/NCS/Dataset/PrunedData/3Ransomware"
adware_edge = "D:/Research/NCS/Dataset/PrunedData/4Adware"
scareware_edge = "D:/Research/NCS/Dataset/PrunedData/5Scareware"

folder_datas = []
data_list = []
current_graph_id = 0

folder_datas, current_graph_id = process_directory(benign_edge, 0, current_graph_id, output_mapping_csv_file)
print(f"Benign graph {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas, current_graph_id = process_directory(sms_malware_edge, 1, current_graph_id, output_mapping_csv_file)
print(f"SMSMalware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas, current_graph_id = process_directory(ransomware_edge, 2, current_graph_id, output_mapping_csv_file)
print(f"Ransomware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas, current_graph_id = process_directory(adware_edge, 3, current_graph_id, output_mapping_csv_file)
print(f"Adware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

folder_datas, current_graph_id = process_directory(scareware_edge, 4, current_graph_id, output_mapping_csv_file)
print(f"Scareware graph: {len(folder_datas)}")
data_list.extend(folder_datas)

pyg_file = "D:/Research/NCS/Dataset/PrunedData/graph_data_edge_and_5d_node_wid.pt"
print(f"Total graph: {len(data_list)}")
torch.save(data_list, pyg_file)


