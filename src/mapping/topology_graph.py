import torch
import numpy as np
from scipy.spatial import distance_matrix

def build_acoustic_topology(mic_coordinates: list, distance_threshold: float = 15.0):
    """
    Constructs the physical graph topology for the ST-GNN.
    
    Args:
        mic_coordinates (list of tuples): (x, y, z) spatial coordinates of each microphone in meters.
        distance_threshold (float): Maximum physical distance (in meters) to establish an edge.
        
    Returns:
        edge_index (torch.Tensor): Shape [2, num_edges] defining connected microphone nodes.
        edge_weight (torch.Tensor): Shape [num_edges] defining the signal weights (inverse distance).
    """
    num_nodes = len(mic_coordinates)
    coords_array = np.array(mic_coordinates)
    
    # 1. Compute the Euclidean distance between every microphone on the floor
    dist_matrix = distance_matrix(coords_array, coords_array)
    
    sources = []
    targets = []
    weights = []
    
    # 2. Build the graph edges
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist = dist_matrix[i, j]
                
                # Only connect nodes if they are within acoustic range of each other
                if dist <= distance_threshold:
                    sources.append(i)
                    targets.append(j)
                    
                    # 3. Apply Inverse Distance Weighting
                    # Sound intensity drops exponentially with distance.
                    # We add a small epsilon (1e-5) to prevent division by zero.
                    weight = 1.0 / (dist + 1e-5)
                    weights.append(weight)
                    
    # 4. Format for PyTorch Geometric
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    
    # Normalize weights to prevent gradient explosion in the GCN layers
    if len(weights) > 0:
        edge_weight = edge_weight / edge_weight.max()
        
    return edge_index, edge_weight

if __name__ == "__main__":
    # Example Production Configuration:
    # A server room with 4 microphones suspended at different coordinates (x, y, z)
    factory_mics = [
        (0.0, 0.0, 3.0),   # Node 0: Rack A entrance
        (5.0, 0.0, 3.0),   # Node 1: Rack A exit
        (0.0, 10.0, 3.0),  # Node 2: Rack B entrance
        (5.0, 10.0, 3.0)   # Node 3: Rack B exit
    ]
    
    edges, weights = build_acoustic_topology(factory_mics)
    
    print(f"[*] Topology generated for {len(factory_mics)} nodes.")
    print(f"[*] Edge Index Shape: {edges.shape}")
    print(f"[*] Edge Weights: {weights}")