"""
Score the detector against real recorded machine audio.

    python -m benchmarks.evaluate_dataset --dataset dcase --root ~/data/dcase2020
    python -m benchmarks.evaluate_dataset --dataset mimii --root ~/data/mimii --epochs 20

The autoencoder is trained on normal audio only and evaluated on a held-out mix,
per machine unit, exactly as the DCASE task defines it. Results are printed
alongside the published baseline for the same machine type where one is known.

This is the number worth quoting. Everything else in ``benchmarks/`` measures
Murmur against material Murmur generated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from benchmarks.baselines import lookup
from benchmarks.datasets import AudioSample, load_dataset, split_normal_train
from benchmarks.features import mel_transform, to_log_mel
from benchmarks.metrics import average_precision, partial_roc_auc, roc_auc
from src.detection.anomaly_detector import SpectrogramAutoencoder

log = logging.getLogger(__name__)


def extract_features(
    dataset, samples: list[AudioSample], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """
    Log-mel spectrograms for a list of samples, as ``(N, 1, n_mels, T)``.

    Clips are trimmed to the shortest in the batch. Padding instead would put a
    block of silence into the reconstruction target, and since silence is
    trivially reconstructable it would drag every score toward zero by an amount
    that depends on clip length rather than on machine condition.
    """
    transform = mel_transform(device)
    spectrograms = []
    for sample in samples:
        audio = dataset.load_audio(sample)
        spectrograms.append(to_log_mel(audio, transform, device))

    if not spectrograms:
        return torch.empty(0)

    min_frames = min(s.shape[-1] for s in spectrograms)
    stacked = torch.stack([s[..., :min_frames] for s in spectrograms])
    return stacked.unsqueeze(1)


def train_autoencoder(
    features: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> SpectrogramAutoencoder:
    """Fit the autoencoder on normal-only features."""
    torch.manual_seed(seed)
    n_mels = features.shape[-2]
    model = SpectrogramAutoencoder(n_mels=n_mels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction, _ = model(batch)
            loss = torch.nn.functional.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.size(0)
        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info("  epoch %2d/%d  loss %.6f", epoch + 1, epochs, total / len(features))

    model.eval()
    return model


def evaluate_group(
    dataset,
    samples: list[AudioSample],
    epochs: int,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, float] | None:
    """Train and score one machine unit. Returns ``None`` if it cannot be scored."""
    has_official_split = any(s.split == "train" for s in samples)
    if has_official_split:
        train_samples = [s for s in samples if s.split == "train" and s.label == 0]
        test_samples = [s for s in samples if s.split == "test"]
    else:
        train_samples, test_samples = split_normal_train(samples, seed=seed)

    if not train_samples or not test_samples:
        log.warning("  skipped: needs both training and test samples")
        return None
    if len({s.label for s in test_samples}) < 2:
        log.warning("  skipped: test fold has only one class, AUC undefined")
        return None

    train_features = extract_features(dataset, train_samples, device)
    model = train_autoencoder(train_features, epochs=epochs, device=device, seed=seed)

    test_features = extract_features(dataset, test_samples, device)
    with torch.no_grad():
        scores = model.anomaly_score(test_features.to(device)).cpu().tolist()
    labels = [s.label for s in test_samples]

    return {
        "auc": roc_auc(scores, labels),
        "pauc_10": partial_roc_auc(scores, labels, max_fpr=0.1),
        "average_precision": average_precision(scores, labels),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "anomaly_rate": sum(labels) / len(labels),
    }


def evaluate(
    dataset_name: str,
    root: str | Path,
    epochs: int = 20,
    device: torch.device | str = "cpu",
    seed: int = 0,
    limit_groups: int | None = None,
) -> dict:
    """Evaluate every machine unit in a dataset."""
    dataset = load_dataset(dataset_name, root)
    all_samples = dataset.samples()
    if not all_samples:
        raise RuntimeError(f"No samples found under {root}")

    by_group: dict[str, list[AudioSample]] = defaultdict(list)
    for sample in all_samples:
        by_group[sample.group].append(sample)

    groups = sorted(by_group)
    if limit_groups:
        groups = groups[:limit_groups]

    started = time.perf_counter()
    results: dict[str, dict] = {}
    for group in groups:
        log.info("Evaluating %s (%d samples)", group, len(by_group[group]))
        scored = evaluate_group(dataset, by_group[group], epochs, device, seed)
        if scored is None:
            continue

        machine_type = group.split("/")[0]
        baseline = lookup(machine_type)
        if baseline is not None:
            scored.update(baseline.as_dict())
            scored["auc_vs_baseline"] = scored["auc"] - baseline.auc
        results[group] = scored
        log.info("  AUC %.4f | pAUC@10%% %.4f", scored["auc"], scored["pauc_10"])

    if not results:
        raise RuntimeError("No group could be scored; check the dataset layout")

    return {
        "dataset": dataset_name,
        "root": str(root),
        "per_group": results,
        "mean_auc": float(np.mean([r["auc"] for r in results.values()])),
        "mean_pauc_10": float(np.mean([r["pauc_10"] for r in results.values()])),
        "groups_scored": len(results),
        "epochs": epochs,
        "seed": seed,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def format_report(results: dict) -> str:
    """Render per-group results as a markdown table, ready to paste into a README."""
    lines = [
        f"### {results['dataset'].upper()} — {results['groups_scored']} machine units",
        "",
        "| Machine | AUC | pAUC@10% | AP | Baseline AUC | Δ |",
        "| :-- | --: | --: | --: | --: | --: |",
    ]
    for group, row in sorted(results["per_group"].items()):
        baseline = row.get("baseline_auc")
        delta = row.get("auc_vs_baseline")
        lines.append(
            f"| {group} | {row['auc']:.4f} | {row['pauc_10']:.4f} | "
            f"{row['average_precision']:.4f} | "
            f"{f'{baseline:.4f}' if baseline is not None else '—'} | "
            f"{f'{delta:+.4f}' if delta is not None else '—'} |"
        )
    lines += [
        "",
        f"**Mean AUC {results['mean_auc']:.4f} | "
        f"Mean pAUC@10% {results['mean_pauc_10']:.4f}** "
        f"({results['epochs']} epochs, seed {results['seed']}, "
        f"{results['elapsed_s']}s)",
        "",
        "Baseline column: DCASE 2020 Task 2 autoencoder baseline. See "
        "`benchmarks/baselines.py` for the caveats before citing.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Murmur on public datasets")
    parser.add_argument("--dataset", required=True, choices=["mimii", "dcase", "ims"])
    parser.add_argument("--root", required=True, help="dataset root directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--limit-groups", type=int, default=None, help="evaluate only the first N units"
    )
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = evaluate(
        args.dataset,
        args.root,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        limit_groups=args.limit_groups,
    )

    print()
    print(format_report(results))
    print()

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        log.info("Wrote %s", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
