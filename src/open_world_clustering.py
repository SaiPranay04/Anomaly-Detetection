import os
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_anomalies():
    print("Loading extracted embeddings...")
    embeddings = np.load(r"d:\Papers\Animal Eye\data\embeddings\test_embeddings.npy")
    logits = np.load(r"d:\Papers\Animal Eye\data\embeddings\test_logits.npy")
    labels = np.load(r"d:\Papers\Animal Eye\data\embeddings\test_labels.npy")
    
    # 1. Define the Anomaly Pool
    # We want to investigate the transactions the model finds most suspicious.
    # We take the top 2,000 transactions with the highest anomaly scores (logits).
    print("Selecting Anomaly Pool (Top 2000 most suspicious transactions)...")
    top_indices = np.argsort(logits)[-2000:]
    
    anomaly_embeddings = embeddings[top_indices]
    anomaly_labels = labels[top_indices]
    
    # 2. Dimensionality Reduction (UMAP)
    print("Applying UMAP to reduce 32D embeddings to 2D...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embeddings = reducer.fit_transform(anomaly_embeddings)
    
    # 3. Density-Based Clustering (HDBSCAN)
    print("Applying HDBSCAN to discover distinct criminal typologies...")
    # HDBSCAN automatically finds dense clusters. min_cluster_size dictates the minimum 
    # number of transactions to form a 'typology'.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(umap_embeddings) # Or original 32D: anomaly_embeddings
    
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"HDBSCAN discovered {num_clusters} distinct typologies (excluding noise).")
    
    # 4. Visualization
    print("Generating visualization...")
    plt.figure(figsize=(12, 8))
    
    # Plot noise points (-1) in gray
    noise_mask = cluster_labels == -1
    plt.scatter(umap_embeddings[noise_mask, 0], umap_embeddings[noise_mask, 1], 
                c='lightgray', s=10, alpha=0.5, label='Noise (Unclustered)')
    
    # Plot discovered typologies
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1) # Remove noise
    
    palette = sns.color_palette("husl", len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        # Also let's see how much true fraud is in this discovered typology
        fraud_ratio = anomaly_labels[mask].sum() / mask.sum()
        
        plt.scatter(umap_embeddings[mask, 0], umap_embeddings[mask, 1], 
                    c=[palette[i]], s=30, alpha=0.8, 
                    label=f'Typology {label} (Fraud %: {fraud_ratio:.1%})')
        
    plt.title('Open-World Typology Discovery (UMAP + HDBSCAN)', fontsize=16)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs(r"d:\Papers\Animal Eye\data\visualizations", exist_ok=True)
    out_path = r"d:\Papers\Animal Eye\data\visualizations\typology_clusters.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {out_path}")

if __name__ == "__main__":
    cluster_anomalies()
