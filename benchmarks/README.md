# Benchmarks

Everything Murmur claims about its own accuracy is computed here.

Three entry points, in increasing order of how much they prove:

| Command | Needs a download | What it establishes |
| :-- | :-- | :-- |
| `python -m benchmarks.evaluate_synthetic` | No | The detector, baseline tracking and thresholding behave as designed. A regression gate. |
| `python -m benchmarks.evaluate_dataset` | Yes | Performance on real recorded machinery, next to published baselines. **The number worth quoting.** |
| `python -m benchmarks.perf` | No | The throughput and latency claims in the top-level README. |

---

## Synthetic degradation benchmark

Runs the edge simulator on a recorded schedule and scores the detector against
the fault type and severity that were actually injected.

```bash
python -m benchmarks.evaluate_synthetic
python -m benchmarks.evaluate_synthetic --frames 800 --seed 7 --json results.json
python -m benchmarks.evaluate_synthetic --weights models/autoencoder.pth
```

Current result with **no trained autoencoder** (frame-energy fallback), 4 nodes
× 300 frames, seed 1234:

| Metric | Value |
| :-- | --: |
| ROC AUC | 0.919 |
| pAUC @ 10% FPR | 0.788 |
| Average precision | 0.906 |
| Precision / Recall | 0.995 / 0.685 |
| False alarms per hour | 10.5 |
| Events detected | 4 / 4 |
| Mean lead time | 26.6 s |
| Mean detection delay | 13.5 s |

Reproduce with `python -m benchmarks.evaluate_synthetic --frames 300`.

> These are synthetic faults whose spectral signatures were written by hand in
> `mock_edge_device.py`. Strong numbers here show the pipeline works. They are
> **not** evidence of field performance — that requires the dataset benchmark
> below.

### Why lead time

An AUC cannot distinguish a detector that fires early enough to order a
replacement bearing from one that fires as the bearing seizes. Lead time —
seconds between the first *sustained* alarm and end-of-life — is the number a
maintenance planner actually acts on, and false alarms per hour is the number
that decides whether the alerting stays switched on. Both are reported per node
and in aggregate.

Detections must persist for `--consecutive` frames (default 3) to count. A
single frame over threshold is routinely a dropped tool or a passing forklift.

---

## Public dataset benchmark

```bash
python -m benchmarks.evaluate_dataset --dataset dcase --root ~/data/dcase2020
python -m benchmarks.evaluate_dataset --dataset mimii --root ~/data/mimii --epochs 30
python -m benchmarks.evaluate_dataset --dataset ims  --root ~/data/ims
```

The autoencoder trains on normal audio only and is scored per machine unit.
Output is a markdown table ready to paste into the top-level README, with the
published DCASE baseline alongside where one exists.

### Getting the data

None of these are vendored — they are large and separately licensed.

**DCASE 2020 Task 2** (recommended starting point — has an official split, so
results are directly comparable)

Download the development set from the
[task page](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds).
Extract to:

```
<root>/fan/train/normal_id_00_00000000.wav
<root>/fan/test/anomaly_id_00_00000000.wav
```

**MIMII** — real industrial fans, pumps, sliders and valves with genuine faults,
at three SNRs. From [Zenodo](https://zenodo.org/record/3384388). Extract to:

```
<root>/fan/id_00/normal/00000000.wav
<root>/fan/id_00/abnormal/00000000.wav
```

MIMII ships no train/test split; the harness carves a normal-only training fold
(70% by default, seeded).

**IMS bearing** — NASA run-to-failure vibration data, from the
[Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
Extract to:

```
<root>/1st_test/2003.10.22.12.06.24
```

IMS has no per-file labels, so the loader treats the final 15% of each run as
degraded. That is a *convention*, not ground truth — which makes IMS most useful
for lead-time analysis, where ordering is what matters, rather than for a
headline AUC.

### Reading the results

An AUC in isolation says very little. Industrial anomaly detection is hard: the
official DCASE autoencoder baseline scores around 0.66 on fan audio and 0.73 on
pump. `benchmarks/baselines.py` carries those reference figures so results can
be read in context — **check its caveats before citing any comparison.**

---

## Performance benchmarks

```bash
python -m benchmarks.perf
python -m benchmarks.perf --json perf.json --iterations 500
```

Measures serialization (MessagePack vs JSON), preprocessing throughput on
whatever device is available, and end-to-end frame latency percentiles.

---

## Metric definitions

| Metric | Definition |
| :-- | :-- |
| ROC AUC | Probability a random faulty frame scores above a random healthy one. |
| pAUC@10% | ROC AUC restricted to FPR ≤ 0.1, renormalised. The DCASE headline metric, and the only region of the curve a plant would operate in. |
| Average precision | Area under the precision-recall curve. Preferred when faults are rare. |
| Lead time | Seconds from the first sustained alarm to end-of-life. |
| Detection delay | Seconds from fault onset to that same alarm. |
| False alarms / hour | Alarms on healthy frames, per hour of healthy runtime. |

All implemented in `benchmarks/metrics.py` and unit-tested against scikit-learn
reference values in `tests/test_benchmark_metrics.py`.
