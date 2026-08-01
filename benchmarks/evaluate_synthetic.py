"""
Score the anomaly detector against the simulator's ground truth.

Run it:

    python -m benchmarks.evaluate_synthetic
    python -m benchmarks.evaluate_synthetic --json results.json --seed 7

What the numbers mean, and what they do not: this measures the detector against
*synthetic* faults whose spectral signatures were written by hand in
``mock_edge_device``. Strong results here demonstrate that the scorer, the
baseline tracking and the thresholding all work as intended. They do not
demonstrate that the system generalises to a real machine — that is what
``benchmarks.evaluate_dataset`` is for. Treat this as a regression gate, not as
evidence of field performance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import torch

from benchmarks.features import mel_transform, to_log_mel
from benchmarks.metrics import format_table, lead_time, summarize
from benchmarks.scenario import Scenario, ScenarioConfig, generate_scenario
from src.detection.anomaly_detector import AnomalyScorer, SpectrogramAutoencoder

log = logging.getLogger(__name__)


def score_scenario(
    scenario: Scenario,
    autoencoder: SpectrogramAutoencoder | None = None,
    z_threshold: float = 3.0,
    warmup_frames: int = 50,
    device: torch.device | str = "cpu",
) -> dict[int, dict[str, list]]:
    """
    Replay a scenario through the detector, in emission order.

    Returns per-node parallel lists of ``scores``, ``labels``, ``predictions``
    and ``frame_indices``. Warm-up frames are excluded: during warm-up the
    scorer is contractually incapable of raising an anomaly, so counting those
    frames as correct rejections would inflate specificity with decisions the
    detector never actually made.
    """
    scorer = AnomalyScorer(
        autoencoder=autoencoder,
        num_nodes=scenario.config.num_nodes,
        warmup_frames=warmup_frames,
        z_threshold=z_threshold,
        window=max(500, warmup_frames * 2),
        device=device,
    )
    transform = mel_transform(device)

    per_node: dict[int, dict[str, list]] = {
        n: {"scores": [], "labels": [], "predictions": [], "frame_indices": []}
        for n in range(scenario.config.num_nodes)
    }

    for frame in scenario.frames:
        spectrogram = to_log_mel(frame.audio, transform, device)
        result = scorer.score(frame.node_id, spectrogram)
        if result.is_warmup:
            continue
        bucket = per_node[frame.node_id]
        bucket["scores"].append(result.normalized_score)
        bucket["labels"].append(1 if frame.is_faulty else 0)
        bucket["predictions"].append(result.is_anomaly)
        bucket["frame_indices"].append(frame.frame_index)

    return per_node


def evaluate(
    config: ScenarioConfig | None = None,
    autoencoder: SpectrogramAutoencoder | None = None,
    z_threshold: float = 3.0,
    consecutive: int = 3,
    device: torch.device | str = "cpu",
) -> dict:
    """Generate a scenario, score it, and reduce to a reportable summary."""
    config = config or ScenarioConfig()
    scenario = generate_scenario(config)
    per_node = score_scenario(
        scenario, autoencoder=autoencoder, z_threshold=z_threshold, device=device
    )

    all_scores: list[float] = []
    all_labels: list[int] = []
    all_predictions: list[bool] = []
    node_reports: dict[str, dict] = {}

    for node_id, bucket in per_node.items():
        all_scores.extend(bucket["scores"])
        all_labels.extend(bucket["labels"])
        all_predictions.extend(bucket["predictions"])

        # Lead time is per-node by nature: event frame indices only mean
        # something against that node's own timeline. Offsetting into the
        # concatenated stream would compare a node's alarms to another's faults.
        offset = _rebase_events(scenario.events_for(node_id), bucket["frame_indices"])
        node_timing = lead_time(
            bucket["predictions"], offset, scenario.frame_interval_s, consecutive
        )
        node_reports[str(node_id)] = node_timing.as_dict()

    merged_events: list[tuple[int, int]] = []
    cursor = 0
    for node_id, bucket in per_node.items():
        for onset, failure in _rebase_events(
            scenario.events_for(node_id), bucket["frame_indices"]
        ):
            merged_events.append((cursor + onset, cursor + failure))
        cursor += len(bucket["scores"])

    summary = summarize(
        all_scores,
        all_labels,
        all_predictions,
        merged_events,
        scenario.frame_interval_s,
        consecutive,
    )
    return {
        "summary": summary,
        "per_node": node_reports,
        "config": {
            "num_nodes": config.num_nodes,
            "frames_per_node": config.frames_per_node,
            "seed": config.seed,
            "z_threshold": z_threshold,
            "consecutive": consecutive,
            "frame_interval_s": scenario.frame_interval_s,
            "duration_s": scenario.duration_s,
            "positive_rate": scenario.positive_rate,
            "autoencoder": autoencoder is not None,
        },
    }


def _rebase_events(
    events: list[tuple[int, int]], frame_indices: list[int]
) -> list[tuple[int, int]]:
    """
    Translate event frame numbers into positions in the post-warm-up list.

    Dropping warm-up frames renumbers everything downstream; without this an
    onset at scenario frame 120 would be looked up at position 120 of a list
    that now starts at frame 50.
    """
    if not frame_indices:
        return []
    lookup = {frame: position for position, frame in enumerate(frame_indices)}
    last = len(frame_indices)

    rebased = []
    for onset, failure in events:
        start = next((lookup[f] for f in range(onset, failure) if f in lookup), None)
        if start is None:
            continue
        end = next(
            (lookup[f] for f in range(failure, frame_indices[-1] + 1) if f in lookup),
            last,
        )
        rebased.append((start, end))
    return rebased


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=400, help="frames per node")
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument(
        "--consecutive",
        type=int,
        default=3,
        help="consecutive alarm frames required to count as a sustained detection",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="path to trained autoencoder weights; omit to use the energy fallback",
    )
    parser.add_argument("--json", type=str, default=None, help="write results to a JSON file")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    autoencoder = None
    if args.weights:
        autoencoder = SpectrogramAutoencoder()
        autoencoder.load_state_dict(torch.load(args.weights, map_location="cpu"))
        autoencoder.eval()
        log.info("Loaded autoencoder weights from %s", args.weights)
    else:
        log.info("No --weights given: scoring with the frame-energy fallback.")

    results = evaluate(
        ScenarioConfig(num_nodes=args.nodes, frames_per_node=args.frames, seed=args.seed),
        autoencoder=autoencoder,
        z_threshold=args.z_threshold,
        consecutive=args.consecutive,
    )

    print()
    print(format_table(results["summary"], "Synthetic degradation benchmark"))
    print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        log.info("Wrote %s", args.json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
