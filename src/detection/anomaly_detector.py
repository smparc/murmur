import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class AcousticForecastingLNN(nn.Module):
    """
    A Liquid Neural Network (LNN) for continuous-time predictive maintenance.
    It adapts to shifting acoustic data to forecast machinery Time-to-Failure (TTF).
    """
    def __init__(self, input_dim: int, hidden_neurons: int = 64, output_dim: int = 1):
        super(AcousticForecastingLNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        
        # 1. Define the Wiring Architecture
        # NCP (Neural Circuit Policy) creates a sparse, biologically-inspired wiring 
        # diagram rather than a dense, fully connected layer. This drastically 
        # reduces compute overhead while maintaining high temporal adaptability.
        wiring = AutoNCP(hidden_neurons, output_dim)
        
        # 2. The Liquid Core (Closed-form Continuous-time network)
        # CfC is a highly optimized variant of an LNN that solves the underlying 
        # differential equations analytically, making it extremely fast for production.
        self.liquid_layer = CfC(input_dim, wiring, return_sequences=False)
        
        # 3. Output Projection
        # Maps the liquid state to a concrete prediction (e.g., remaining hours of life)
        self.fc_out = nn.Linear(output_dim, output_dim)
        
        # Optional: Sigmoid if outputting a failure probability (0.0 to 1.0)
        self.activation = nn.Sigmoid() 


    def forward(self, x, timespans=None):
        """
        Forward pass for the liquid network.
        
        Parameters:
        - x: Sequence of acoustic embeddings. Shape: [batch_size, seq_len, input_dim]
        - timespans: Optional irregular time intervals between observations. 
                     Crucial if the Kafka stream experiences latency/jitter.
        """
        # The CfC layer natively handles irregular time-series data if timespans are provided.
        # This makes it highly resilient to network interruptions on the factory floor.
        liquid_out, _ = self.liquid_layer(x, timespans=timespans)
        
        # Project to the final prediction metric
        prediction = self.fc_out(liquid_out)
        
        # Output a continuous probability curve of impending failure
        return self.activation(prediction)


if __name__ == "__main__":
    # Production Test: Initialize the network to accept the 256-dim embedding from the ST-GNN
    BATCH_SIZE = 16
    SEQ_LENGTH = 50   # Processing the last 50 acoustic frames
    EMBEDDING_DIM = 256
    
    # Initialize model
    model = AcousticForecastingLNN(input_dim=EMBEDDING_DIM)
    
    # Simulate a batch of incoming streaming data from the ST-GNN
    mock_streaming_data = torch.randn(BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM)
    
    # Simulate irregular arrival times (e.g., network jitter from the edge microphones)
    mock_timespans = torch.rand(BATCH_SIZE, SEQ_LENGTH)
    
    # Run continuous inference
    failure_probability = model(mock_streaming_data, timespans=mock_timespans)
    
    print("[*] Liquid Neural Network Initialized.")
    print(f"[*] Input sequence shape: {mock_streaming_data.shape}")
    print(f"[*] Predicted Failure Probabilities shape: {failure_probability.shape}")