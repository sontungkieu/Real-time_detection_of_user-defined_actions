"""CPU latency benchmark helper for TensorFlow Lite models."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf


def benchmark_tflite_model(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    runs: int = 500,
    warmup: int = 50,
    out_path: str | Path | None = None,
) -> dict[str, object]:
    """Benchmark a TFLite model on the current machine's CPU."""

    model_path = Path(model_path)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    interpreter.resize_tensor_input(input_index, input_shape, strict=False)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    sample = _make_sample(input_shape, input_details[0]["dtype"], input_details[0].get("quantization"))
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])

    latencies_ms: list[float] = []
    for _ in range(runs):
        start = perf_counter()
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        latencies_ms.append((perf_counter() - start) * 1000)

    latencies = np.asarray(latencies_ms, dtype=np.float64)
    result = {
        "model_path": str(model_path),
        "input_shape": list(input_shape),
        "runs": runs,
        "warmup": warmup,
        "mean_ms": float(latencies.mean()),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "model_size_kb": model_path.stat().st_size / 1024,
        "note": "CPU benchmark on the developer machine, not Android device latency.",
    }

    output = Path(out_path) if out_path else model_path.with_suffix(".benchmark.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _make_sample(
    shape: tuple[int, ...],
    dtype: np.dtype,
    quantization: tuple[float, int] | None,
) -> np.ndarray:
    rng = np.random.default_rng(42)
    sample = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    dtype = np.dtype(dtype)
    if dtype == np.float32:
        return sample
    if dtype == np.float16:
        return sample.astype(np.float16)
    if np.issubdtype(dtype, np.integer):
        scale, zero_point = quantization or (0.0, 0)
        if scale:
            sample = sample / scale + zero_point
        info = np.iinfo(dtype)
        return np.clip(np.round(sample), info.min, info.max).astype(dtype)
    return sample.astype(dtype)
