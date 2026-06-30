import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def construct_graph(preprocessed_dir, output_dir):
    print(f"Loading preprocessed data from {preprocessed_dir}...")
    edges_df = pd.read_parquet(os.path.join(preprocessed_dir, 'edges.parquet'))
    nodes_df = pd.read_parquet(os.path.join(preprocessed_dir, 'nodes.parquet'))
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Constructing Node Features (x)...")
    # Ensure nodes are sorted by node_id
    nodes_df = nodes_df.sort_values('node_id')
    node_feature_cols = ['total_sent_amount', 'sent_count', 'total_received_amount', 'received_count']
    x = torch.tensor(nodes_df[node_feature_cols].values, dtype=torch.float)
    
    print("Constructing Edge Indices (edge_index)...")
    source_nodes = edges_df['nameOrig_id'].values
    target_nodes = edges_df['nameDest_id'].values
    # PyG expects edge_index of shape [2, num_edges]
    edge_index = torch.tensor(np.array([source_nodes, target_nodes]), dtype=torch.long)
    
    print("Constructing Edge Features (edge_attr)...")
    # Identify edge feature columns (excluding identifiers and target labels)
    edge_feature_cols = [col for col in edges_df.columns if col not in [
        'step', 'type', 'nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud', 
        'nameOrig_id', 'nameDest_id'
    ]]
    
    edge_attr = torch.tensor(edges_df[edge_feature_cols].astype(float).values, dtype=torch.float)
    
    print("Constructing Labels (y)...")
    # In PaySim, fraud is a property of the transaction (edge classification)
    y = torch.tensor(edges_df['isFraud'].values, dtype=torch.float)
    
    # Store temporal information if doing dynamic GNN
    edge_time = torch.tensor(edges_df['step'].values, dtype=torch.long)
    
    print("Building PyG Data object...")
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_time = edge_time # Adding custom attribute for Temporal Graph Networks
    
    print(f"Graph Construction Complete:\n{data}")
    
    output_path = os.path.join(output_dir, 'paysim_graph.pt')
    torch.save(data, output_path)
    print(f"Saved graph to {output_path}")

if __name__ == "__main__":
    PREPROCESSED_DIR = r"d:\Papers\Animal Eye\data\preprocessed"
    OUTPUT_DIR = r"d:\Papers\Animal Eye\data\graph"
    
    construct_graph(PREPROCESSED_DIR, OUTPUT_DIR)
