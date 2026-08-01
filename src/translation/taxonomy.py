"""
Grounded fault diagnosis by matching acoustic evidence against a taxonomy.

This exists to replace something that could not work. ``llm_decoder`` builds an
``EmbeddingProjector`` — a two-layer MLP from the 256-d ST-GNN embedding into the
LLM's 2048-d token space — and never trains it; the line that would load weights
is commented out. A randomly initialised projection produces a vector with no
relationship to any token the model knows, so the "diagnostics" downstream of it
are the LLM's prior conditioned on noise. They read fluently, which is precisely
the problem: fluent text about equipment that is fine is worse than no text.

The approach here inverts that. Rather than asking a language model to invent a
diagnosis, match the measured spectral evidence against a small catalogue of
documented fault signatures, and report the match *with its evidence and its
confidence*. Every diagnosis is traceable to a frequency band and a rule, and an
unrecognised signature comes back as "unrecognised" instead of as a confident
sentence.

The catalogue is small and deliberately so — it covers the failure modes the
simulator generates plus the common rotating-machinery modes they correspond to.
It is meant to be extended per-site: real diagnosis depends on what is installed
on the floor, and no general catalogue substitutes for that.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

from src.explain.saliency import AnomalyExplanation

__all__ = [
    "DEFAULT_TAXONOMY",
    "Diagnosis",
    "FaultSignature",
    "FaultTaxonomy",
    "SpectralCharacter",
]

SpectralCharacter = Literal["tonal", "broadband", "modulated", "impulsive"]


@dataclass(frozen=True)
class FaultSignature:
    """A documented acoustic fault mode."""

    name: str
    description: str
    frequency_bands_hz: tuple[tuple[float, float], ...]
    """Bands where this fault deposits energy."""
    character: SpectralCharacter
    likely_causes: tuple[str, ...]
    recommended_action: str
    urgency: Literal["monitor", "schedule", "urgent"]

    def band_overlap(self, low_hz: float, high_hz: float) -> float:
        """
        Fraction of ``[low_hz, high_hz]`` covered by this signature's bands.

        Overlap rather than a centre-frequency match: a fault occupies a range,
        an observed band occupies a range, and how much they intersect is a
        better similarity measure than how close their midpoints happen to fall.
        """
        if high_hz <= low_hz:
            return 1.0 if any(lo <= low_hz <= hi for lo, hi in self.frequency_bands_hz) else 0.0

        covered = 0.0
        for lo, hi in self.frequency_bands_hz:
            covered += max(0.0, min(high_hz, hi) - max(low_hz, lo))
        return min(1.0, covered / (high_hz - low_hz))


@dataclass(frozen=True)
class Diagnosis:
    """One candidate explanation for an anomaly, with its evidence."""

    signature: FaultSignature
    confidence: float
    evidence: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return self.signature.name

    def as_dict(self) -> dict:
        return {
            "fault": self.signature.name,
            "description": self.signature.description,
            "confidence": round(self.confidence, 4),
            "urgency": self.signature.urgency,
            "likely_causes": list(self.signature.likely_causes),
            "recommended_action": self.signature.recommended_action,
            "evidence": list(self.evidence),
        }

    def describe(self) -> str:
        """A sentence for an alert body or a log line."""
        return (
            f"{self.signature.name} ({self.confidence:.0%} confidence) — "
            f"{self.signature.description}. {self.signature.recommended_action}"
        )


#: Returned when the evidence matches nothing in the catalogue. Saying so is the
#: useful answer: an unrecognised signature on a machine that was quiet
#: yesterday still deserves a look, and inventing a named fault for it would
#: send someone to check the wrong component.
_UNRECOGNISED = FaultSignature(
    name="Unrecognised acoustic anomaly",
    description=(
        "Sound departs from this node's baseline but does not match a known "
        "fault signature"
    ),
    frequency_bands_hz=(),
    character="broadband",
    likely_causes=(
        "New or relocated equipment",
        "Process change",
        "A fault mode not yet in the catalogue",
    ),
    recommended_action="Inspect and, if a cause is identified, add it to the taxonomy.",
    urgency="monitor",
)


DEFAULT_TAXONOMY: tuple[FaultSignature, ...] = (
    FaultSignature(
        name="Bearing race defect",
        description=(
            "High-frequency tonal squeal with harmonics, typical of spalling on "
            "a bearing race"
        ),
        frequency_bands_hz=((2000.0, 4500.0), (5000.0, 7000.0)),
        character="tonal",
        likely_causes=(
            "Loss of lubrication",
            "Contamination ingress",
            "Fatigue spalling on the outer race",
        ),
        recommended_action=(
            "Schedule bearing inspection; check lubricant level and condition."
        ),
        urgency="schedule",
    ),
    FaultSignature(
        name="Pump cavitation",
        description=(
            "Broadband noise with transient impulses, consistent with vapour "
            "bubbles collapsing at the impeller"
        ),
        frequency_bands_hz=((1000.0, 8000.0),),
        character="impulsive",
        likely_causes=(
            "Suction pressure below NPSH required",
            "Blocked or restricted inlet strainer",
            "Fluid temperature above design",
        ),
        recommended_action=(
            "Check suction pressure and inlet restriction; cavitation erodes the "
            "impeller quickly once established."
        ),
        urgency="urgent",
    ),
    FaultSignature(
        name="Rotating imbalance",
        description=(
            "Low-frequency amplitude modulation at shaft rate, typical of mass "
            "imbalance or a bent shaft"
        ),
        frequency_bands_hz=((10.0, 250.0),),
        character="modulated",
        likely_causes=(
            "Material build-up on the rotor",
            "Lost balance weight",
            "Bent shaft or coupling misalignment",
        ),
        recommended_action="Balance the rotor; inspect coupling alignment.",
        urgency="schedule",
    ),
    FaultSignature(
        name="Gear mesh wear",
        description="Tonal energy at mesh frequency with sidebands, indicating tooth wear",
        frequency_bands_hz=((500.0, 2000.0),),
        character="tonal",
        likely_causes=("Tooth wear or pitting", "Backlash beyond tolerance"),
        recommended_action="Inspect gear teeth; check backlash and lubrication.",
        urgency="schedule",
    ),
    FaultSignature(
        name="Compressed air or fluid leak",
        description="Sustained high-frequency broadband hiss with no tonal structure",
        frequency_bands_hz=((4000.0, 8000.0),),
        character="broadband",
        likely_causes=("Failed seal or gasket", "Cracked line", "Loose fitting"),
        recommended_action="Trace with ultrasonic leak detector; a leak is pure energy cost.",
        urgency="schedule",
    ),
    FaultSignature(
        name="Electrical discharge",
        description="Impulsive high-frequency bursts, consistent with arcing or corona",
        frequency_bands_hz=((6000.0, 8000.0),),
        character="impulsive",
        likely_causes=("Insulation breakdown", "Loose electrical connection"),
        recommended_action=(
            "De-energise and inspect before the fault escalates; arcing is a fire risk."
        ),
        urgency="urgent",
    ),
)


class FaultTaxonomy:
    """Matches spectral evidence against a catalogue of fault signatures."""

    def __init__(
        self,
        signatures: Sequence[FaultSignature] = DEFAULT_TAXONOMY,
        min_confidence: float = 0.25,
    ):
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.signatures = tuple(signatures)
        self.min_confidence = min_confidence

    @staticmethod
    def _match_score(signature: FaultSignature, explanation: AnomalyExplanation) -> float:
        """
        Similarity between where the error landed and where this fault predicts it.

        The obvious scoring — sum of ``band.share * overlap`` — is wrong, and
        wrong in a way that is easy to miss: it rewards *breadth*. "Pump
        cavitation" spans 1-8 kHz, so it overlaps any high-frequency evidence at
        least as well as the narrow bearing signature does, and wins every
        high-frequency diagnosis by default. A catalogue scored that way would
        report cavitation for a bearing squeal.

        So both sides are treated as distributions and compared with the
        Bhattacharyya coefficient. The signature's predicted distribution is
        spread across everything it claims, which means a broad signature must
        pay for its breadth: predicting energy across 7 kHz and finding it in
        one band scores worse than predicting exactly that band.
        """
        weights = []
        for band in explanation.bands:
            overlap = signature.band_overlap(band.low_hz, band.high_hz)
            # Weight by bandwidth so a signature spanning many bands genuinely
            # dilutes its prediction across them.
            weights.append(overlap * max(band.high_hz - band.low_hz, 1.0))

        total_weight = sum(weights)
        if total_weight <= 0.0:
            return 0.0

        score = 0.0
        for weight, band in zip(weights, explanation.bands):
            score += math.sqrt((weight / total_weight) * band.share)
        return min(1.0, score)

    def diagnose(
        self, explanation: AnomalyExplanation, limit: int = 3
    ) -> list[Diagnosis]:
        """
        Rank candidate faults for an explained anomaly.

        Always returns at least one entry — ``_UNRECOGNISED`` when nothing
        clears ``min_confidence``.
        """
        if not explanation.bands:
            return [Diagnosis(_UNRECOGNISED, confidence=0.0)]

        scored: list[Diagnosis] = []
        for signature in self.signatures:
            score = self._match_score(signature, explanation)
            if score < self.min_confidence:
                continue

            matched: list[str] = []
            for band in explanation.bands:
                overlap = signature.band_overlap(band.low_hz, band.high_hz)
                # Only cite bands that carry real weight, or the evidence list
                # fills with negligible bands and stops being readable.
                if overlap > 0.4 and band.share > 0.1:
                    matched.append(
                        f"{band.share:.0%} of error in {band.describe().split(' (')[0]}"
                    )

            scored.append(
                Diagnosis(signature=signature, confidence=score, evidence=tuple(matched))
            )

        if not scored:
            dominant = explanation.dominant_band
            evidence = (
                (f"dominant energy {dominant.describe()}",) if dominant else ()
            )
            return [Diagnosis(_UNRECOGNISED, confidence=0.0, evidence=evidence)]

        scored.sort(key=lambda d: d.confidence, reverse=True)
        return scored[:limit]

    def best(self, explanation: AnomalyExplanation) -> Diagnosis:
        """The single most likely diagnosis."""
        return self.diagnose(explanation, limit=1)[0]
