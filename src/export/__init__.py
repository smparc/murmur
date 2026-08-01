"""
Model export for edge deployment.

See :mod:`src.export.onnx_export` for why the detector belongs next to the
machine rather than behind a WAN link.
"""

from src.export.onnx_export import export_autoencoder, quantize, verify_export

__all__ = ["export_autoencoder", "quantize", "verify_export"]
