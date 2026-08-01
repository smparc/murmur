"""
Published baseline scores, for context next to Murmur's own numbers.

An AUC in isolation says very little. 0.72 sounds mediocre and is in fact
roughly what the official DCASE autoencoder baseline achieves on pump audio —
industrial anomaly detection is simply hard, and the only way to read a result
is against what other systems get on the same data.

.. warning::

   The figures below are transcribed from the DCASE 2020 Task 2 baseline system
   results for the development set. They are provided as an orientation aid.
   **Verify them against the official results page before publishing any
   comparison**, and note that a number is only comparable if it was produced on
   the same split with the same metric definition — pAUC in particular is
   reported at ``p = 0.1`` and is not comparable to a full AUC.

   Source: https://dcase.community/challenge2020/
           task-unsupervised-detection-of-anomalous-sounds-results
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Baseline", "DCASE2020_AUTOENCODER", "lookup"]


@dataclass(frozen=True)
class Baseline:
    """A published score for one machine type."""

    machine_type: str
    auc: float
    pauc: float
    system: str
    citation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "machine_type": self.machine_type,
            "baseline_auc": self.auc,
            "baseline_pauc": self.pauc,
            "baseline_system": self.system,
        }


_SYSTEM = "DCASE 2020 Task 2 baseline (autoencoder)"
_CITATION = (
    "Koizumi et al., 'Description and Discussion on DCASE2020 Challenge Task 2', "
    "DCASE 2020 Workshop."
)


def _entry(machine_type: str, auc: float, pauc: float) -> Baseline:
    return Baseline(machine_type, auc, pauc, _SYSTEM, _CITATION)


#: Development-set averages, expressed on a 0-1 scale to match this package's
#: metric functions (the challenge reports them as percentages).
DCASE2020_AUTOENCODER: dict[str, Baseline] = {
    "toycar": _entry("ToyCar", 0.7877, 0.6758),
    "toyconveyor": _entry("ToyConveyor", 0.7253, 0.6043),
    "fan": _entry("fan", 0.6583, 0.5245),
    "pump": _entry("pump", 0.7289, 0.5999),
    "slider": _entry("slider", 0.8476, 0.6653),
    "valve": _entry("valve", 0.6628, 0.5098),
}


def lookup(machine_type: str) -> Baseline | None:
    """Find a baseline by machine type, case- and separator-insensitively."""
    key = machine_type.lower().replace("_", "").replace("-", "").replace(" ", "")
    return DCASE2020_AUTOENCODER.get(key)
