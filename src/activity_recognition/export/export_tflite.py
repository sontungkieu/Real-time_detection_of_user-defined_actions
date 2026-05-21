"""TensorFlow Lite export helper."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def export_keras_to_tflite(
    model_path: str | Path,
    out_path: str | Path,
    float16: bool = False,
) -> dict[str, float | str]:
    """Convert a Keras model to TensorFlow Lite and return size metadata."""

    model_path = Path(model_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if float16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)

    keras_size = model_path.stat().st_size
    tflite_size = out_path.stat().st_size
    metadata = {
        "model_path": str(model_path),
        "tflite_path": str(out_path),
        "keras_size_mb": keras_size / (1024 * 1024),
        "tflite_size_mb": tflite_size / (1024 * 1024),
        "float16": float16,
    }
    print(f"Keras size: {metadata['keras_size_mb']:.3f} MB")
    print(f"TFLite size: {metadata['tflite_size_mb']:.3f} MB")
    return metadata
