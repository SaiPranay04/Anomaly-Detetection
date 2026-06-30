import os
import torch

def verify(graph_path):
    print(f"Loading {graph_path}...")
    data = torch.load(graph_path, weights_only=False)
    print(data)
    
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    print(f"Number of edge features: {data.num_edge_features}")
    
    fraud_edges = (data.y == 1).sum().item()
    print(f"Fraudulent edges in subset: {fraud_edges} / {data.num_edges} ({(fraud_edges/data.num_edges)*100:.2f}%)")

if __name__ == "__main__":
    GRAPH_PATH = r"d:\Papers\Animal Eye\data\graph\paysim_graph.pt"
    if os.path.exists(GRAPH_PATH):
        verify(GRAPH_PATH)
    else:
        print(f"File not found: {GRAPH_PATH}")
