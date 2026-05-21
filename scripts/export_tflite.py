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
    parser.add_argument("--model", required=True, help="Path to a .keras model.")
    parser.add_argument("--out", required=True, help="Output .tflite path.")
    parser.add_argument("--float16", action="store_true", help="Enable float16 weight quantization.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = export_keras_to_tflite(args.model, args.out, float16=args.float16)
    metadata_path = Path(args.out).with_suffix(".tflite.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
