#!/usr/bin/env python3
"""Benchmark TensorFlow Lite latency on the current machine's CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.export.benchmark_tflite import benchmark_tflite_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a .tflite model.")
    parser.add_argument(
        "--input-shape", required=True, help="Comma-separated shape, e.g. 1,128,4."
    )
    parser.add_argument(
        "--runs", type=int, default=500, help="Measured inference runs."
    )
    parser.add_argument("--warmup", type=int, default=50, help="Warmup inference runs.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic input seed.")
    parser.add_argument(
        "--out", default=None, help="Optional benchmark JSON output path."
    )
    parser.add_argument("--output", default=None, help="Alias for --out.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_shape = tuple(int(part) for part in args.input_shape.split(","))
    benchmark_tflite_model(
        args.model,
        input_shape,
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
        out_path=args.out or args.output,
    )


if __name__ == "__main__":
    main()
