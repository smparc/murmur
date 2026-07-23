"""
Anomaly Detection Module — the actual brain of Murmur.


Provides two complementary approaches:
    1. SpectrogramAutoencoder — unsupervised baseline that learns "normal" acoustic
       patterns and flags high-reconstruction-error frames as anomalies.
    2. AnomalyScorer — stateful online scorer that maintains running statistics
       and computes z-scores + adaptive thresholds per node.


This module fills the critical gap: the ST-GNN produces embeddings, but nothing
in the original system actually *decided* whether something was anomalous.
"""


import logging
from collections import deque
from dataclasses import dataclass, field


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


log = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# 1. Spectrogram Autoencoder (Unsupervised Anomaly Detection)
# ---------------------------------------------------------------------------


class SpectrogramAutoencoder(nn.Module):
    """
    Convolutional autoencoder trained on *normal* spectrograms.


    At inference, high reconstruction error → anomaly.
    This is the standard unsupervised anomaly detection approach for
    time-frequency representations (see DCASE challenge baselines).


    Input:  (B, 1, n_mels, time_frames)  e.g. (B, 1, 64, 32)
    Output: reconstruction + latent embedding
    """


    def __init__(self, n_mels: int = 64, latent_dim: int = 32):
        super().__init__()
        self.n_mels = n_mels
        self.latent_dim = latent_dim


        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )


        self.fc_encode = nn.Linear(128 * 2 * 2, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 2 * 2)


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_encode(h)


    def decode(self, z: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, 128, 2, 2)
        out = self.decoder(h)
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_embedding)."""
        z = self.encode(x)
        recon = self.decode(z, target_size=x.shape[2:])
        return recon, z


    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample anomaly score (reconstruction MSE).


        Returns:
            scores: (B,) — higher = more anomalous
        """
        with torch.no_grad():
            recon, _ = self.forward(x)
            # Per-sample MSE
            scores = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
        return scores



# ---------------------------------------------------------------------------
# 2. Online Anomaly Scorer (Adaptive Thresholding)
# ---------------------------------------------------------------------------


@dataclass
class NodeStats:
    """Running statistics for a single microphone node."""
    scores: deque = field(default_factory=lambda: deque(maxlen=500))
    mean: float = 0.0
    std: float = 1.0
    threshold: float = 3.0  # z-score threshold
    anomaly_count: int = 0
    total_count: int = 0


    def update(self, score: float):
        self.scores.append(score)
        self.total_count += 1
        if len(self.scores) >= 10:
            arr = np.array(self.scores)
            self.mean = float(arr.mean())
            self.std = float(arr.std()) + 1e-8


    
    def z_score_threshold(self) -> float:
        return self.mean + self.threshold * self.std



class AnomalyScorer:
    """
    Stateful anomaly scorer that maintains per-node running statistics.


    Combines autoencoder reconstruction error with adaptive z-score thresholds.
    Operates in two modes:
        1. LEARN — accumulates baseline statistics from normal operation
        2. DETECT — flags frames that exceed the adaptive threshold


    Usage:
        scorer = AnomalyScorer(autoencoder, num_nodes=4)


        # During normal operation (first N frames)
        result = scorer.score(node_id=2, spectrogram=spec)


        # result.is_anomaly will be True when reconstruction error exceeds
        # the adaptive threshold for that specific node.
    """


    def __init__(
        self,
        autoencoder: SpectrogramAutoencoder = None,
        num_nodes: int = 4,
        z_threshold: float = 3.0,
        warmup_frames: int = 50,
    ):
        self.autoencoder = autoencoder
        self.z_threshold = z_threshold
        self.warmup_frames = warmup_frames
        self.node_stats: dict[int, NodeStats] = {
            i: NodeStats(threshold=z_threshold) for i in range(num_nodes)
        }


    @dataclass
    class Result:
        node_id: int
        raw_score: float
        z_score: float
        threshold: float
        is_anomaly: bool
        is_warmup: bool
        severity: str  # "normal", "warning", "critical"


    def score(self, node_id: int, spectrogram: torch.Tensor) -> "AnomalyScorer.Result":
        """
        Score a single spectrogram frame for anomalies.


        Args:
            node_id: which microphone node
            spectrogram: (1, n_mels, time) or (1, 1, n_mels, time)


        Returns:
            Result with anomaly decision and metadata
        """
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.unsqueeze(0)  # Add batch dim


        # Compute reconstruction error
        if self.autoencoder is not None:
            raw_score = float(self.autoencoder.anomaly_score(spectrogram).item())
        else:
            # Fallback: use spectral energy as proxy
            raw_score = float(spectrogram.pow(2).mean().item())


        # Update node statistics
        stats = self.node_stats.get(node_id)
        if stats is None:
            stats = NodeStats(threshold=self.z_threshold)
            self.node_stats[node_id] = stats


        stats.update(raw_score)
        is_warmup = stats.total_count < self.warmup_frames


        # Compute z-score
        z_score = (raw_score - stats.mean) / stats.std if not is_warmup else 0.0


        # Decision
        is_anomaly = (not is_warmup) and (z_score > self.z_threshold)
        if is_anomaly:
            stats.anomaly_count += 1


        # Severity classification
        if not is_anomaly or is_warmup:
            severity = "normal"
        elif z_score > self.z_threshold * 2:
            severity = "critical"
        else:
            severity = "warning"


        return self.Result(
            node_id=node_id,
            raw_score=raw_score,
            z_score=round(z_score, 3),
            threshold=round(stats.z_score_threshold, 6),
            is_anomaly=is_anomaly,
            is_warmup=is_warmup,
            severity=severity,
        )


    def get_node_summary(self) -> dict:
        """Return summary stats for all nodes."""
        return {
            node_id: {
                "total_frames": s.total_count,
                "anomaly_count": s.anomaly_count,
                "anomaly_rate": round(s.anomaly_count / max(1, s.total_count), 4),
                "current_mean": round(s.mean, 6),
                "current_std": round(s.std, 6),
                "threshold": round(s.z_score_threshold, 6),
            }
            for node_id, s in self.node_stats.items()
        }



if __name__ == "__main__":
    # Quick test
    ae = SpectrogramAutoencoder(n_mels=64, latent_dim=32)
    scorer = AnomalyScorer(ae, num_nodes=4, warmup_frames=20)


    # Simulate 30 normal frames + 5 anomalous ones
    for i in range(30):
        spec = torch.randn(1, 1, 64, 32) * 0.1  # Low-energy normal
        result = scorer.score(node_id=0, spectrogram=spec)
        if not result.is_warmup:
            print(f"Frame {i}: score={result.raw_score:.4f} z={result.z_score:.2f} anomaly={result.is_anomaly}")


    print("\n--- Injecting anomaly ---")
    for i in range(5):
        spec = torch.randn(1, 1, 64, 32) * 2.0  # High-energy anomaly
        result = scorer.score(node_id=0, spectrogram=spec)
        print(f"Frame {30+i}: score={result.raw_score:.4f} z={result.z_score:.2f} anomaly={result.is_anomaly} severity={result.severity}")


    print("\nNode summary:", scorer.get_node_summary())