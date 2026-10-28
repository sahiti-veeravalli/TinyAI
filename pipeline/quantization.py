"""
Quantization Engine for TinyAI.
Applies:
1. Dynamic INT8 quantization via onnxruntime.quantization
2. Static INT8 quantization with calibration data reader
3. FP16 half-precision conversion for edge GPU / WebGPU
4. Benchmark file size, latency, and throughput verification.
"""

import os
import time
from typing import Dict, Any, Optional
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


class QuantizationEngine:
    """Manages post-training quantization and ONNX runtime optimization."""

    @staticmethod
    def quantize_to_int8(
        input_onnx_path: str,
        output_onnx_path: str,
        weight_type: QuantType = QuantType.QInt8
    ) -> Dict[str, Any]:
        """
        Dynamically quantizes floating-point weights to 8-bit integers.
        Reduces model size by up to 75% with minimal accuracy degradation.
        """
        os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
        
        # Check input file
        if not os.path.exists(input_onnx_path):
            raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")
            
        orig_size = os.path.getsize(input_onnx_path)

        # Apply ONNX dynamic quantization
        quantize_dynamic(
            model_input=input_onnx_path,
            model_output=output_onnx_path,
            weight_type=weight_type,
            per_channel=True,
            reduce_range=False
        )

        quant_size = os.path.getsize(output_onnx_path)
        compression_ratio = float(orig_size / max(quant_size, 1))

        return {
            "original_size_bytes": orig_size,
            "quantized_size_bytes": quant_size,
            "original_size_kb": round(orig_size / 1024, 2),
            "quantized_size_kb": round(quant_size / 1024, 2),
            "compression_ratio": round(compression_ratio, 2),
            "bit_width": 8,
            "output_path": output_onnx_path
        }

    @staticmethod
    def measure_onnx_latency(
        onnx_model_path: str,
        dummy_sample: np.ndarray,
        n_warmup: int = 20,
        n_iterations: int = 100
    ) -> Dict[str, float]:
        """Measures edge inference latency using onnxruntime CPU session."""
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 1  # Simulate single-core edge endpoint

        session = ort.InferenceSession(onnx_model_path, session_options, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name

        # Ensure correct float32 shape
        if dummy_sample.ndim == 1:
            dummy_sample = dummy_sample.reshape(1, -1)
        sample = dummy_sample.astype(np.float32)

        # Warmup
        for _ in range(n_warmup):
            _ = session.run(None, {input_name: sample})

        # Measurement
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = session.run(None, {input_name: sample})
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)  # ms

        latencies = np.array(latencies)
        return {
            "mean_ms": round(float(np.mean(latencies)), 3),
            "p50_ms": round(float(np.median(latencies)), 3),
            "p95_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_ms": round(float(np.percentile(latencies, 99)), 3),
            "throughput_qps": round(float(1000.0 / max(np.mean(latencies), 0.001)), 1)
        }
