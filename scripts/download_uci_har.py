#!/usr/bin/env python3
"""Download and extract the UCI HAR Dataset."""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
FALLBACK_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="data/raw/uci_har", help="Extraction directory."
    )
    parser.add_argument(
        "--force", action="store_true", help="Redownload existing archive."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = out_dir / "UCI HAR Dataset"
    if dataset_dir.exists() and not args.force:
        print(f"UCI HAR dataset already exists at {dataset_dir}")
        return

    archive_path = out_dir / "UCI_HAR_Dataset.zip"
    if args.force or not archive_path.exists():
        if not _download(UCI_HAR_URL, archive_path):
            if not _download(FALLBACK_URL, archive_path):
                _print_manual_instructions(out_dir)
                sys.exit(1)

    try:
        _extract_archive_chain(archive_path, out_dir, dataset_dir)
    except zipfile.BadZipFile:
        archive_path.unlink(missing_ok=True)
        if not _download(FALLBACK_URL, archive_path):
            _print_manual_instructions(out_dir)
            sys.exit(1)
        _extract_archive_chain(archive_path, out_dir, dataset_dir)
    print(f"Extracted UCI HAR dataset to {dataset_dir}")


def _download(url: str, out_path: Path) -> bool:
    print(f"Downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            with out_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
        return True
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"Download failed: {exc}")
        return False


def _extract_archive_chain(
    archive_path: Path, out_dir: Path, dataset_dir: Path
) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(out_dir)
    nested_archive = out_dir / "UCI HAR Dataset.zip"
    if nested_archive.exists() and not dataset_dir.exists():
        with zipfile.ZipFile(nested_archive) as archive:
            archive.extractall(out_dir)


def _print_manual_instructions(out_dir: Path) -> None:
    print(
        "\nManual download required. Download 'UCI HAR Dataset.zip' from:\n"
        "https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones\n"
        f"Then extract it under: {out_dir}\n"
        f"Expected directory: {out_dir / 'UCI HAR Dataset'}"
    )


if __name__ == "__main__":
    main()
