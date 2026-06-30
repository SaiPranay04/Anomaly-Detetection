import os
import torch
import numpy as np
from torch_geometric.data import Data
from gnn_model import AMLGraphSAGE

def extract():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load graph
    data_path = r"d:\Papers\Animal Eye\data\graph\paysim_graph.pt"
    data = torch.load(data_path, weights_only=False).to(device)
    
    # Instantiate Model
    in_channels = data.num_node_features
    edge_dim = data.num_edge_features
    hidden_channels = 64
    
    model = AMLGraphSAGE(in_channels, hidden_channels, edge_dim).to(device)
    model.load_state_dict(torch.load(r"d:\Papers\Animal Eye\models\gnn_encoder.pth", weights_only=True))
    model.eval()
    
    # Get Test Mask
    num_edges = data.num_edges
    val_end = int(num_edges * 0.85)
    test_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_mask[val_end:] = True
    
    print("Extracting embeddings for test set...")
    with torch.no_grad():
        logits, edge_representation = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
        
        test_logits = logits[test_mask].cpu().numpy()
        test_embeddings = edge_representation[test_mask].cpu().numpy()
        test_labels = data.y[test_mask].cpu().numpy()
        
    os.makedirs(r"d:\Papers\Animal Eye\data\embeddings", exist_ok=True)
    
    # Save the arrays
    np.save(r"d:\Papers\Animal Eye\data\embeddings\test_embeddings.npy", test_embeddings)
    np.save(r"d:\Papers\Animal Eye\data\embeddings\test_logits.npy", test_logits)
    np.save(r"d:\Papers\Animal Eye\data\embeddings\test_labels.npy", test_labels)
    
    print(f"Extracted {len(test_logits)} test embeddings with shape {test_embeddings.shape}")
    print("Saved to d:\\Papers\\Animal Eye\\data\\embeddings\\")

if __name__ == "__main__":
    extract()
