import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from src.mapping.st_gnn_model import SpatioTemporalGNN
from src.forecasting.liquid_network import AcousticForecastingLNN
from src.mapping.topology_graph import build_acoustic_topology

# Configuration
EPOCHS = 50
BATCH_SIZE = 16
SEQ_LENGTH = 50
IN_CHANNELS = 64      # Mel-Spectrogram bins
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 256
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_mock_training_data(num_samples, num_nodes):
    """Simulates historical acoustic data and ground truth failure timelines."""
    # Shape: [Batch, Sequence, Nodes, Features] -> Flattened for GNN
    x = torch.randn(num_samples, SEQ_LENGTH, num_nodes * IN_CHANNELS).to(DEVICE)
    
    # Ground truth Time-to-Failure probabilities (0.0 = healthy, 1.0 = immediate failure)
    y_ttf = torch.rand(num_samples, 1).to(DEVICE)
    return x, y_ttf

def train():
    print(f"[*] Starting Murmur Training Pipeline on {DEVICE}...")
    
    # 1. Initialize Topology
    mics = [(0.0, 0.0, 3.0), (5.0, 0.0, 3.0), (0.0, 10.0, 3.0), (5.0, 10.0, 3.0)]
    edge_index, edge_weight = build_acoustic_topology(mics)
    edge_index, edge_weight = edge_index.to(DEVICE), edge_weight.to(DEVICE)
    num_nodes = len(mics)

    # 2. Initialize Models
    st_gnn = SpatioTemporalGNN(IN_CHANNELS, HIDDEN_CHANNELS, EMBEDDING_DIM, num_nodes).to(DEVICE)
    lnn = AcousticForecastingLNN(input_dim=EMBEDDING_DIM).to(DEVICE)
    
    # 3. Optimizers & Loss
    optimizer = optim.Adam(list(st_gnn.parameters()) + list(lnn.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss() # Mean Squared Error for TTF probability

    # 4. Generate Data
    x_train, y_train = generate_mock_training_data(100, num_nodes)

    mlflow.set_experiment("murmur_model_training")
    
    with mlflow.start_run():
        mlflow.log_params({"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "embedding_dim": EMBEDDING_DIM})
        
        for epoch in range(EPOCHS):
            st_gnn.train()
            lnn.train()
            
            optimizer.zero_grad()
            
            # Forward pass through ST-GNN
            embeddings = st_gnn(x_train, edge_index, edge_weight, batch_size=100, seq_length=SEQ_LENGTH)
            
            # The ST-GNN outputs a single embedding per batch sequence. 
            # We unsqueeze to simulate a sequence length of 1 for the LNN step.
            lnn_input = embeddings.unsqueeze(1)
            
            # Forward pass through Liquid Network
            ttf_predictions = lnn(lnn_input)
            
            # Compute Loss & Backpropagate
            loss = criterion(ttf_predictions, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{EPOCHS} | MSE Loss: {loss.item():.4f}")
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

    # 5. Save Artifacts for Inference Server
    os.makedirs("models", exist_ok=True)
    torch.save(st_gnn.state_dict(), "models/st_gnn_weights.pth")
    torch.save(lnn.state_dict(), "models/lnn_weights.pth")
    print("[*] Training complete. Weights saved to /models.")

if __name__ == "__main__":
    train()