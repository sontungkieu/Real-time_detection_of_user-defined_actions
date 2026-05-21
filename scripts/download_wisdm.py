#!/usr/bin/env python3
"""Download or explain how to prepare the WISDM dataset."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

WISDM_URL = (
    "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="data/raw/wisdm", help="Directory for extracted WISDM files."
    )
    parser.add_argument(
        "--manual", action="store_true", help="Only print manual download instructions."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manual:
        _print_manual_instructions(out_dir)
        return

    archive_path = out_dir / "WISDM_ar_latest.tar.gz"
    try:
        print(f"Downloading WISDM from {WISDM_URL}")
        with urllib.request.urlopen(WISDM_URL, timeout=30) as response:
            with archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract(tar, out_dir)
        archive_path.unlink(missing_ok=True)
        print(f"WISDM files extracted to {out_dir}")
    except (urllib.error.URLError, TimeoutError, tarfile.TarError, OSError) as exc:
        archive_path.unlink(missing_ok=True)
        print(f"Automatic download failed: {exc}")
        _print_manual_instructions(out_dir)


def _print_manual_instructions(out_dir: Path) -> None:
    print(
        "\nManual WISDM setup:\n"
        "1. Download the WISDM Activity Recognition raw dataset from:\n"
        f"   {WISDM_URL}\n"
        "2. Extract it locally.\n"
        f"3. Place WISDM_ar_v1.1_raw.txt under: {out_dir}\n"
        "4. Re-run training with configs/wisdm_cnn1d.yaml or configs/wisdm_mlp.yaml.\n"
    )


def _safe_extract(tar: tarfile.TarFile, out_dir: Path) -> None:
    base = out_dir.resolve()
    for member in tar.getmembers():
        target = (out_dir / member.name).resolve()
        if not target.is_relative_to(base):
            raise tarfile.TarError(f"Unsafe path in archive: {member.name}")
    tar.extractall(out_dir)


if __name__ == "__main__":
    main()
