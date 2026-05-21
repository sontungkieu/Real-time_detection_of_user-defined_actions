#!/usr/bin/env python3
"""Train a WISDM activity recognition model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.training.train import train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument(
        "--seed", type=int, default=None, help="Override split/model seed."
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(
        args.config, seed_override=args.seed, output_dir_override=args.output_dir
    )


if __name__ == "__main__":
    main()
