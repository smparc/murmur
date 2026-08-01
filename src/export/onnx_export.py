"""
ONNX export for edge deployment.

Acoustic monitoring is an edge problem. The microphones are already on the
factory floor; shipping every raw stream to a datacentre GPU costs bandwidth
continuously and adds a network partition to the failure path of a safety
system. A quantised autoencoder running on a $99 board next to the machine is
both cheaper and more robust than the same model behind a WAN link.

    python -m src.export.onnx_export --model autoencoder --output models/ae.onnx
    python -m src.export.onnx_export --model autoencoder --quantize

``onnx`` and ``onnxruntime`` are optional. Export needs neither — ``torch.onnx``
writes the file — but verification and quantization do, and both are skipped
with a clear message rather than an ImportError if they are absent.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from src.detection.anomaly_detector import SpectrogramAutoencoder
from src.settings import settings

log = logging.getLogger(__name__)

__all__ = ["export_autoencoder", "quantize", "verify_export"]


def export_autoencoder(
    output_path: str | Path,
    weights_path: str | Path | None = None,
    n_mels: int | None = None,
    time_frames: int = 16,
    opset: int = 17,
) -> Path:
    """
    Export :class:`SpectrogramAutoencoder` to ONNX.

    The time axis is exported as a dynamic dimension. Chunk length varies with
    hop size and Kafka framing, and a graph frozen to one length would have to be
    re-exported for every configuration change — or, worse, would silently accept
    a wrong-shaped input on a runtime that does not check.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_mels = n_mels or settings.N_MELS
    model = SpectrogramAutoencoder(n_mels=n_mels)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        log.info("Loaded weights from %s", weights_path)
    else:
        log.warning(
            "No weights supplied — exporting a randomly initialised model. "
            "Useful for shape and latency checks, useless for detection."
        )
    model.eval()

    example = torch.randn(1, 1, n_mels, time_frames)

    torch.onnx.export(
        model,
        example,
        str(output_path),
        input_names=["spectrogram"],
        output_names=["reconstruction", "latent"],
        dynamic_axes={
            "spectrogram": {0: "batch", 3: "time"},
            "reconstruction": {0: "batch", 3: "time"},
            "latent": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    log.info("Exported ONNX model to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def verify_export(
    onnx_path: str | Path,
    weights_path: str | Path | None = None,
    n_mels: int | None = None,
    time_frames: int = 16,
    tolerance: float = 1e-3,
) -> bool:
    """
    Check the exported graph reproduces the PyTorch model's output.

    Worth doing every time. Export silently changes semantics more often than is
    comfortable — adaptive pooling and interpolation, both used by this model,
    are common sources of divergence between opsets.

    ``weights_path`` must be whatever was passed to :func:`export_autoencoder`.
    Comparing against a freshly constructed model instead compares two different
    random initialisations, so the check fails for reasons that have nothing to
    do with the export and passes only when the tolerance is wide enough to be
    meaningless.
    """
    try:
        import numpy as np
        import onnxruntime
    except ImportError:
        log.warning("onnxruntime not installed; skipping verification")
        return False

    n_mels = n_mels or settings.N_MELS
    example = torch.randn(1, 1, n_mels, time_frames)

    model = SpectrogramAutoencoder(n_mels=n_mels)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        log.warning(
            "No weights supplied — the reference model is randomly initialised and "
            "will not match the exported graph. Only the output shape is checked."
        )
    model.eval()

    with torch.no_grad():
        torch_out, _ = model(example)

    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"spectrogram": example.numpy()})[0]

    if onnx_out.shape != tuple(torch_out.shape):
        log.error("Shape mismatch: onnx %s vs torch %s", onnx_out.shape, tuple(torch_out.shape))
        return False

    if not weights_path:
        # Two unrelated random initialisations; a numeric comparison here would
        # report a failure that says nothing about the export itself.
        return True

    difference = float(np.abs(onnx_out - torch_out.numpy()).max())
    log.info("Max absolute difference: %.2e", difference)
    return difference < tolerance


def quantize(onnx_path: str | Path, output_path: str | Path | None = None) -> Path | None:
    """
    Dynamic INT8 quantization, for memory-constrained edge hardware.

    Typically around 4x smaller. Accuracy impact is model-specific and must be
    measured, not assumed — re-run ``benchmarks.evaluate_dataset`` against the
    quantised graph before trusting it in the field.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        log.warning("onnxruntime not installed; cannot quantize")
        return None

    onnx_path = Path(onnx_path)
    output_path = Path(output_path or onnx_path.with_suffix(".int8.onnx"))

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )

    before = onnx_path.stat().st_size / 1024
    after = output_path.stat().st_size / 1024
    log.info("Quantized %.1f KB -> %.1f KB (%.1fx smaller)", before, after, before / after)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Murmur models to ONNX")
    parser.add_argument("--model", default="autoencoder", choices=["autoencoder"])
    parser.add_argument("--output", default="models/autoencoder.onnx")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--time-frames", type=int, default=16)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    path = export_autoencoder(
        args.output,
        weights_path=args.weights,
        time_frames=args.time_frames,
        opset=args.opset,
    )

    if not args.skip_verify:
        verify_export(path, weights_path=args.weights, time_frames=args.time_frames)
    if args.quantize:
        quantize(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
