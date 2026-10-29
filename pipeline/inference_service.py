"""
Production Inference Runtime Engine for TinyAI.
Features:
- Single-sample and Batch prediction support
- Input schema validation and automated preprocessing alignment
- Dynamic ONNX Runtime session caching with single-thread edge simulation
- Real measured execution latency per batch
- Seamless execution of compressed INT8/FP32 artifacts
"""

import os
import time
from typing import Dict, Any, List, Optional, Union
import numpy as np
import onnxruntime as ort

from backend.config import settings
from backend.security import safe_join
from pipeline.data_manager import DatasetManager, PreprocessorSchema


class InferenceEngine:
    """Manages cached ONNX Runtime sessions and executes high-speed edge inference."""

    _sessions: Dict[str, ort.InferenceSession] = {}

    @classmethod
    def get_session(cls, model_path: str) -> ort.InferenceSession:
        if model_path in cls._sessions:
            return cls._sessions[model_path]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model file not found at: {model_path}")
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # Standard edge thread profile
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        cls._sessions[model_path] = session
        return session

    @classmethod
    def predict(
        cls,
        model_path: str,
        features: Union[List[float], List[List[float]], Dict[str, Any], List[Dict[str, Any]]],
        schema: Optional[PreprocessorSchema] = None,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Executes real inference on the given model with input features.
        Accepts:
        - raw float array (pre-normalized)
        - dictionary or list of dictionaries with named features (applies schema preprocessor)
        """
        t0 = time.perf_counter_ns()
        session = cls.get_session(model_path)
        input_name = session.get_inputs()[0].name
        expected_shape = session.get_inputs()[0].shape

        # Format input array
        if isinstance(features, dict) or (isinstance(features, list) and len(features) > 0 and isinstance(features[0], dict)):
            if schema is None:
                raise ValueError("PreprocessorSchema is required when passing named dictionary features.")
            X_arr = DatasetManager.apply_preprocessor(features, schema)
        elif isinstance(features, list):
            if len(features) > 0 and isinstance(features[0], (int, float)):
                # Single 1D sample
                X_arr = np.array([features], dtype=np.float32)
            else:
                # Batch 2D list
                X_arr = np.array(features, dtype=np.float32)
        elif isinstance(features, np.ndarray):
            X_arr = features.astype(np.float32)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(1, -1)
        else:
            raise ValueError("Unsupported input format for features.")

        # Check feature dimension
        if len(expected_shape) > 1 and expected_shape[1] is not None:
            expected_dim = expected_shape[1]
            if isinstance(expected_dim, int) and X_arr.shape[1] != expected_dim:
                # Pad or truncate if minor discrepancy
                if X_arr.shape[1] < expected_dim:
                    pad = np.zeros((X_arr.shape[0], expected_dim - X_arr.shape[1]), dtype=np.float32)
                    X_arr = np.hstack([X_arr, pad])
                else:
                    X_arr = X_arr[:, :expected_dim]

        # Execute ONNX Runtime graph
        outputs = session.run(None, {input_name: X_arr})
        raw_out = outputs[0]
        t1 = time.perf_counter_ns()
        latency_ms = round((t1 - t0) / 1_000_000.0, 3)

        n_samples = X_arr.shape[0]
        results: List[Dict[str, Any]] = []

        if is_classification:
            # Handle binary classifier output
            if len(outputs) > 1 and isinstance(outputs[1], list):
                # skl2onnx probabilities
                probs = [list(d.values())[-1] for d in outputs[1]]
                preds = [int(p >= 0.5) for p in probs]
                for i in range(n_samples):
                    results.append({
                        "sample_index": i,
                        "prediction": preds[i],
                        "probability": round(float(probs[i]), 4),
                        "status": "ALERT" if preds[i] == 1 else "NORMAL"
                    })
            else:
                flat_out = raw_out.flatten()
                probs = 1.0 / (1.0 + np.exp(-np.clip(flat_out, -35.0, 35.0)))
                preds = (probs >= 0.5).astype(int)
                for i in range(n_samples):
                    p_val = float(probs[i]) if i < len(probs) else 0.5
                    pred_val = int(preds[i]) if i < len(preds) else 0
                    results.append({
                        "sample_index": i,
                        "prediction": pred_val,
                        "probability": round(p_val, 4),
                        "status": "ALERT" if pred_val == 1 else "NORMAL"
                    })
        else:
            flat_out = raw_out.flatten()
            for i in range(n_samples):
                v = float(flat_out[i]) if i < len(flat_out) else 0.0
                results.append({
                    "sample_index": i,
                    "prediction_value": round(v, 4)
                })

        return {
            "model_path": os.path.basename(model_path),
            "sample_count": n_samples,
            "latency_ms": latency_ms,
            "is_batch": n_samples > 1,
            "predictions": results
        }

