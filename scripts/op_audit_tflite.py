#!/usr/bin/env python3
"""Audit TensorFlow Lite operators for a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.export.op_audit import audit_tflite_ops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a .tflite model.")
    parser.add_argument("--output", default=None, help="Optional audit JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_tflite_ops(args.model, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
