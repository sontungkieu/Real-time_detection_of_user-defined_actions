#!/usr/bin/env python3
"""Train a HAR activity recognition model from a YAML config."""

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
    parser.add_argument("--model", default=None, help="Override model.type.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Override learning rate."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_from_config(
        args.config,
        seed_override=args.seed,
        output_dir_override=args.output_dir,
        model_override=args.model,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        learning_rate_override=args.learning_rate,
    )


if __name__ == "__main__":
    main()
