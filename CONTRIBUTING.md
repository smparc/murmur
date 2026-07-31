# Contributing to Murmur

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -e ".[dev]"
pre-commit install
```

CPU wheels are installed explicitly first — the default PyPI resolution pulls a
multi-gigabyte CUDA build that a development machine rarely needs.

## Before opening a pull request

```bash
ruff check src/ tests/ orchestration/
ruff format --check src/ tests/ orchestration/
pytest tests/ -m "not integration"
```

Integration tests need a broker:

```bash
docker compose -f docker-compose.kafka.yml up -d
pytest tests/ -m integration
```

## Conventions

- **Configuration goes through `src/settings.py`.** It validates at import time.
  Adding a setting means adding its validation rule too — a bad audio parameter
  otherwise surfaces as a shape mismatch three services downstream.
- **The API must not trust caller-supplied detection results.** Anomaly scores
  are computed in `src/inference/worker.py` from model output. An endpoint that
  accepts a score and forwards it to Prometheus is not detecting anything.
- **The dashboard reads structured fields, never generated prose.** Model output
  is not a stable interface.
- **Tests are seeded.** `conftest.py` reseeds every RNG per test. If an
  assertion is statistical, make its threshold wide enough to be meaningful
  rather than tight enough to be flaky.
- **No new dependency in the serving path without a reason.** MLOps tooling
  lives in the `mlops` extra so it stays out of the inference images.
