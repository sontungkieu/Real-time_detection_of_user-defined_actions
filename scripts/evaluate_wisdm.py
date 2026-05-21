#!/usr/bin/env python3
"""Evaluate a saved WISDM training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.training.evaluate import evaluate_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Path to the YAML config used for training."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory containing model.keras and metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_from_config(args.config, args.run_dir)


if __name__ == "__main__":
    main()
