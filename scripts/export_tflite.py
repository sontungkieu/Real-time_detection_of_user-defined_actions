#!/usr/bin/env python3
"""Export a Keras model to TensorFlow Lite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.export.export_tflite import export_keras_to_tflite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Path to a .keras model.")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory containing model.keras. Used when --model is omitted.",
    )
    parser.add_argument("--out", default=None, help="Output .tflite path.")
    parser.add_argument("--output", default=None, help="Alias for --out.")
    parser.add_argument(
        "--float16", action="store_true", help="Enable float16 weight quantization."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model
    if model_path is None and args.model_dir is not None:
        model_path = str(Path(args.model_dir) / "model.keras")
    out_path = args.out or args.output
    if model_path is None:
        raise SystemExit("Provide --model or --model-dir.")
    if out_path is None:
        raise SystemExit("Provide --out or --output.")

    metadata = export_keras_to_tflite(model_path, out_path, float16=args.float16)
    metadata_path = Path(out_path).with_suffix(".tflite.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
