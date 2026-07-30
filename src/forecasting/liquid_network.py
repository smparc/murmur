"""
Liquid Neural Network for continuous-time predictive maintenance.

A Closed-form Continuous-time (CfC) core wrapped in a task head that forecasts
Time-to-Failure. CfC solves the underlying ODE analytically rather than by
numerical integration, which is what makes a genuinely continuous-time model
fast enough to serve online.

Two properties matter operationally:

- **Irregular sampling.** Edge microphones drop frames and Kafka delivery
  jitters. Passing real inter-arrival times as ``timespans`` lets the model
  integrate over the gap that actually occurred instead of assuming a uniform
  clock it never had.
- **Adaptivity.** The liquid state's time constants are input-dependent, so the
  network tracks a degradation profile that shifts underneath it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ncps.torch import CfC
from ncps.wirings import AutoNCP


class AcousticForecastingLNN(nn.Module):
    """
    Forecasts failure probability from a sequence of acoustic embeddings.

    Parameters
    ----------
    input_dim:
        Width of the incoming embedding (the ST-GNN's ``embedding_dim``).
    hidden_neurons:
        Total neurons in the neural-circuit wiring.
    output_dim:
        Predicted quantities. ``1`` is failure probability; more can carry
        auxiliary targets such as severity or a horizon estimate.
    dropout:
        Applied in the task head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_neurons: int = 64,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}")
        if output_dim < 1:
            raise ValueError(f"output_dim must be >= 1, got {output_dim}")

        # AutoNCP splits `hidden_neurons` into inter/command/motor layers, and
        # requires motor neurons to leave room for the rest of the circuit.
        motor_neurons = max(output_dim, hidden_neurons // 4)
        if motor_neurons >= hidden_neurons - 2:
            motor_neurons = max(1, hidden_neurons - 3)
        if motor_neurons < output_dim:
            raise ValueError(
                f"hidden_neurons ({hidden_neurons}) is too small to emit "
                f"{output_dim} outputs; use at least {output_dim + 3}"
            )

        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons
        self.output_dim = output_dim
        self.motor_neurons = motor_neurons

        # Sparse, biologically-inspired wiring rather than a dense layer: fewer
        # parameters and better-conditioned temporal dynamics.
        wiring = AutoNCP(hidden_neurons, motor_neurons)

        # return_sequences=True keeps the per-timestep states available, so the
        # same core serves both a single forecast and a full trajectory.
        self.liquid_layer = CfC(input_dim, wiring, return_sequences=True)

        # A real head. The previous implementation projected the motor output
        # through a Linear(output_dim, output_dim) — for the default output_dim
        # of 1 that is a scalar affine, i.e. two parameters doing nothing.
        self.head = nn.Sequential(
            nn.LayerNorm(motor_neurons),
            nn.Linear(motor_neurons, motor_neurons * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motor_neurons * 2, output_dim),
        )
        self.activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        timespans: torch.Tensor | None = None,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            ``(batch, seq_len, input_dim)`` sequence of acoustic embeddings.
        timespans:
            ``(batch, seq_len)`` inter-observation intervals in seconds.
            Omit for a uniform clock.
        return_sequence:
            When ``True``, return the failure probability at every timestep —
            a forecast trajectory rather than a single number.

        Returns
        -------
        ``(batch, output_dim)``, or ``(batch, seq_len, output_dim)`` when
        ``return_sequence`` is set.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (batch, seq_len, input_dim), got {tuple(x.shape)}")
        if x.size(-1) != self.input_dim:
            raise ValueError(f"expected input_dim {self.input_dim}, got {x.size(-1)}")
        if timespans is not None and timespans.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"timespans {tuple(timespans.shape)} must match x's "
                f"(batch, seq_len) = {tuple(x.shape[:2])}"
            )

        liquid_out, _ = self.liquid_layer(x, timespans=timespans)  # (B, S, motor)

        if return_sequence:
            return self.activation(self.head(liquid_out))

        # The final state summarises the whole observed history.
        return self.activation(self.head(liquid_out[:, -1]))

    @torch.no_grad()
    def predict_trajectory(
        self, x: torch.Tensor, timespans: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Per-timestep failure probabilities, for the dashboard forecast plot."""
        self.eval()
        return self.forward(x, timespans=timespans, return_sequence=True)


def main() -> None:  # pragma: no cover - manual inspection helper
    from src.settings import settings

    model = AcousticForecastingLNN(
        input_dim=settings.GNN_EMBEDDING_DIM,
        hidden_neurons=settings.LNN_HIDDEN_NEURONS,
    )
    x = torch.randn(16, settings.SEQ_LENGTH, settings.GNN_EMBEDDING_DIM)
    # Irregular arrival times, as produced by a jittery edge network.
    timespans = torch.rand(16, settings.SEQ_LENGTH) * 0.4 + 0.3

    point = model(x, timespans=timespans)
    trajectory = model.predict_trajectory(x, timespans=timespans)

    print("[*] Liquid Neural Network initialised.")
    print(f"[*] Input:      {tuple(x.shape)}")
    print(f"[*] Forecast:   {tuple(point.shape)}")
    print(f"[*] Trajectory: {tuple(trajectory.shape)}")
    print(f"[*] Parameters: {sum(p.numel() for p in model.parameters()) / 1e3:.1f}K")


if __name__ == "__main__":  # pragma: no cover
    main()
