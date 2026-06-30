import os
import torch
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import networkx as nx

def visualize_typology_subgraph():
    print("Loading Graph Data...")
    data_path = r"d:\Papers\Animal Eye\data\graph\paysim_graph.pt"
    data = torch.load(data_path, weights_only=False)
    
    num_edges = data.num_edges
    val_end = int(num_edges * 0.85)
    
    print("Loading Embeddings...")
    embeddings = np.load(r"d:\Papers\Animal Eye\data\embeddings\test_embeddings.npy")
    logits = np.load(r"d:\Papers\Animal Eye\data\embeddings\test_logits.npy")
    
    print("Running Clustering...")
    top_indices = np.argsort(logits)[-2000:]
    anomaly_embeddings = embeddings[top_indices]
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embeddings = reducer.fit_transform(anomaly_embeddings)
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(umap_embeddings)
    
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1) # Remove noise
    
    os.makedirs(r"d:\Papers\Animal Eye\data\visualizations", exist_ok=True)
    
    edge_index = data.edge_index.cpu().numpy()
    
    for label in unique_labels:
        print(f"Extracting Network for Typology {label}...")
        
        # Get the indices of the transactions in this cluster
        cluster_mask = cluster_labels == label
        cluster_test_indices = top_indices[cluster_mask]
        cluster_global_indices = val_end + cluster_test_indices
        
        # Extract the source and destination nodes for these transactions
        src_nodes = edge_index[0, cluster_global_indices]
        dst_nodes = edge_index[1, cluster_global_indices]
        
        # Build the NetworkX graph
        G = nx.DiGraph()
        for src, dst in zip(src_nodes, dst_nodes):
            G.add_edge(src, dst)
            
        # Draw the graph
        plt.figure(figsize=(12, 12))
        
        # Use spring layout for a nice aesthetic spread
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        
        nx.draw_networkx_nodes(G, pos, node_size=30, node_color='tomato', alpha=0.9, edgecolors='darkred')
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='dimgray', arrows=True, arrowsize=15)
        
        plt.title(f'Discovered Criminal Typology {label}\nNodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}', fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        
        out_path = rf"d:\Papers\Animal Eye\data\visualizations\typology_{label}_network.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved {out_path}")
        
    print("All typology visualizations generated successfully.")

if __name__ == "__main__":
    visualize_typology_subgraph()
