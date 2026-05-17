"""
Deployment package — ONNX export, quantization, and edge inference.

Exports:
    export_macf_to_onnx: Export MACF to ONNX TreeEnsembleRegressor
    CausalForestExporter: JSON fallback exporter
    ONNXInferenceEngine: ONNX Runtime inference (C++)
    FastTreeInference: JSON Python inference (fallback)
    load_inference_engine: Auto-detect best available engine
"""

from src.deployment.model_export import (
    export_macf_to_onnx,
    export_all_macf_to_onnx,
    CausalForestExporter,
    ONNXInferenceEngine,
    FastTreeInference,
    load_inference_engine,
)

# Backward compatibility
CausalForestONNXExporter = CausalForestExporter

__all__ = [
    'export_macf_to_onnx',
    'export_all_macf_to_onnx',
    'CausalForestExporter',
    'CausalForestONNXExporter',
    'ONNXInferenceEngine',
    'FastTreeInference',
    'load_inference_engine',
]
