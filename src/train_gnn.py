import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from gnn_model import AMLGraphSAGE
import numpy as np

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_path = r"d:\Papers\Animal Eye\data\graph\paysim_graph.pt"
    print("Loading graph data...")
    data = torch.load(data_path, weights_only=False)
    data = data.to(device)
    
    num_edges = data.num_edges
    
    print("Creating Temporal Splits...")
    # Temporal split
    train_end = int(num_edges * 0.7)
    val_end = int(num_edges * 0.85)
    
    # Masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    train_mask[:train_end] = True
    
    val_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    val_mask[train_end:val_end] = True
    
    test_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    test_mask[val_end:] = True
    
    # Instantiate Model
    in_channels = data.num_node_features
    edge_dim = data.num_edge_features
    hidden_channels = 64
    
    print("Initializing Model & Optimizer...")
    model = AMLGraphSAGE(in_channels, hidden_channels, edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Compute positive weight for BCE
    num_pos = data.y[train_mask].sum()
    num_neg = train_mask.sum() - num_pos
    pos_weight = num_neg / (num_pos + 1e-5)
    print(f"BCE Positive Weight computed as: {pos_weight.item():.2f}")
    
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    epochs = 500
    patience = 50
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    print("Starting Full-Batch Training with Early Stopping...")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass for all nodes and edges
        logits, _ = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
        
        # Calculate loss only on training edges
        loss = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
                val_loss = criterion(val_logits[val_mask], data.y[val_mask])
                
                # Metrics
                val_probs = torch.sigmoid(val_logits[val_mask]).cpu().numpy()
                val_labels = data.y[val_mask].cpu().numpy()
                
                # To prevent errors if no fraud in val set
                if val_labels.sum() > 0:
                    auroc = roc_auc_score(val_labels, val_probs)
                    auprc = average_precision_score(val_labels, val_probs)
                    preds = (val_probs > 0.5).astype(int)
                    f1 = f1_score(val_labels, preds)
                else:
                    auroc, auprc, f1 = 0.0, 0.0, 0.0
                    
                print(f"Epoch {epoch+1:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val AUROC: {auroc:.4f} | Val AUPRC: {auprc:.4f} | Val F1: {f1:.4f}")
                
                # Early Stopping Logic
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    epochs_no_improve = 0
                    best_model_state = model.state_dict().copy()
                else:
                    epochs_no_improve += 10 # Evaluated every 10 epochs
                    
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}!")
                    break

    print("Restoring best model weights...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("Evaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(data.x, data.edge_index, data.edge_index, data.edge_attr)
        test_probs = torch.sigmoid(test_logits[test_mask]).cpu().numpy()
        test_labels = data.y[test_mask].cpu().numpy()
        
        if test_labels.sum() > 0:
            auroc = roc_auc_score(test_labels, test_probs)
            auprc = average_precision_score(test_labels, test_probs)
            preds = (test_probs > 0.5).astype(int)
            f1 = f1_score(test_labels, preds)
            print(f"Test AUROC: {auroc:.4f} | Test AUPRC: {auprc:.4f} | Test F1: {f1:.4f}")
        else:
            print("No fraud found in test set to calculate metrics.")
            
    os.makedirs(r"d:\Papers\Animal Eye\models", exist_ok=True)
    model_path = r"d:\Papers\Animal Eye\models\gnn_encoder.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()
