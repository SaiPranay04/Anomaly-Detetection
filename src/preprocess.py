import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def preprocess_paysim(input_csv, output_dir, nrows=None):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv, nrows=nrows)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Extracting unique nodes...")
    # Nodes can be either Orig or Dest
    unique_orig = df['nameOrig'].unique()
    unique_dest = df['nameDest'].unique()
    all_nodes = np.unique(np.concatenate([unique_orig, unique_dest]))
    
    # Create node to ID mapping
    node_to_id = {node: idx for idx, node in enumerate(all_nodes)}
    
    print("Mapping nodes to IDs...")
    df['nameOrig_id'] = df['nameOrig'].map(node_to_id)
    df['nameDest_id'] = df['nameDest'].map(node_to_id)
    
    print("Engineering Edge Features...")
    # One hot encode 'type'
    type_dummies = pd.get_dummies(df['type'], prefix='type')
    df = pd.concat([df, type_dummies], axis=1)
    
    # Let's compute delta balances
    df['deltaOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['deltaDest'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    numerical_edge_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 
                               'oldbalanceDest', 'newbalanceDest', 'deltaOrig', 'deltaDest', 'step']
    
    scaler = StandardScaler()
    df[numerical_edge_features] = scaler.fit_transform(df[numerical_edge_features])
    
    print("Engineering Node Features...")
    # A simple way to get node features: total sent/received amounts and counts
    sent_stats = df.groupby('nameOrig_id').agg(
        total_sent_amount=('amount', 'sum'),
        sent_count=('amount', 'count')
    ).reset_index().rename(columns={'nameOrig_id': 'node_id'})
    
    received_stats = df.groupby('nameDest_id').agg(
        total_received_amount=('amount', 'sum'),
        received_count=('amount', 'count')
    ).reset_index().rename(columns={'nameDest_id': 'node_id'})
    
    # Merge to create node features dataframe
    node_df = pd.DataFrame({'node_id': range(len(all_nodes))})
    node_df = node_df.merge(sent_stats, on='node_id', how='left')
    node_df = node_df.merge(received_stats, on='node_id', how='left')
    node_df.fillna(0, inplace=True)
    
    # Scale node features
    node_features = ['total_sent_amount', 'sent_count', 'total_received_amount', 'received_count']
    node_scaler = StandardScaler()
    node_df[node_features] = node_scaler.fit_transform(node_df[node_features])
    
    print(f"Saving preprocessed files to {output_dir}...")
    df.to_parquet(os.path.join(output_dir, 'edges.parquet'), index=False)
    node_df.to_parquet(os.path.join(output_dir, 'nodes.parquet'), index=False)
    
    # Also save the node mappings just in case
    mapping_df = pd.DataFrame({'node_name': all_nodes, 'node_id': range(len(all_nodes))})
    mapping_df.to_parquet(os.path.join(output_dir, 'node_mapping.parquet'), index=False)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    # Define paths
    INPUT_CSV = r"d:\Papers\Animal Eye\PS_20174392719_1491204439457_log.csv\PS_20174392719_1491204439457_log.csv"
    OUTPUT_DIR = r"d:\Papers\Animal Eye\data\preprocessed"
    
    # We pass nrows=1000000 for initial testing to avoid OOM on standard laptops
    # The user can remove nrows to process the full 6.3M rows
    preprocess_paysim(INPUT_CSV, OUTPUT_DIR, nrows=1000000)
