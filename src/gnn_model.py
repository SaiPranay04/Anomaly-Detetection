import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class AMLGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_dim, out_channels=1):
        super(AMLGraphSAGE, self).__init__()
        
        # Node embedding layers
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Edge classification MLP
        # Input: source_node_emb (hidden) + target_node_emb (hidden) + edge_attr (edge_dim)
        mlp_in = hidden_channels * 2 + edge_dim
        self.lin1 = torch.nn.Linear(mlp_in, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = torch.nn.Linear(hidden_channels // 2, out_channels)

    def encode_nodes(self, x, edge_index):
        # 2-layer GraphSAGE to get node embeddings
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.conv2(h, edge_index)
        return h

    def encode_edges(self, h, edge_label_index, edge_attr):
        # h: [num_nodes, hidden_channels]
        # edge_label_index: [2, batch_size] (source and target nodes for the batch)
        # edge_attr: [batch_size, edge_dim]
        
        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]
        
        src_h = h[src_idx]
        dst_h = h[dst_idx]
        
        # Concatenate src_h, dst_h, and edge_attr
        edge_emb = torch.cat([src_h, dst_h, edge_attr], dim=-1)
        
        # Pass through MLP
        out = F.relu(self.lin1(edge_emb))
        out = F.dropout(out, p=0.2, training=self.training)
        
        # This layer acts as our "Unknown Discovery" embedding space for Phase 3
        edge_representation = F.relu(self.lin2(out)) 
        
        # Final classification logit
        logits = self.lin3(edge_representation).squeeze(-1)
        
        return logits, edge_representation

    def forward(self, x, edge_index, edge_label_index, edge_attr):
        h = self.encode_nodes(x, edge_index)
        logits, edge_representation = self.encode_edges(h, edge_label_index, edge_attr)
        return logits, edge_representation
