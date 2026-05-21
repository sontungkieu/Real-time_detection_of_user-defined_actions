#!/usr/bin/env python3
"""Configure persistent Qualcomm AI Hub credentials from a local env file."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.utils.secrets import get_qualcomm_api_key, mask_secret


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write the token to the user's local Qualcomm AI Hub config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = get_qualcomm_api_key(args.env_file)
    if not api_key:
        print(f"Missing QUALCOMM_AI_HUB_API_KEY in {args.env_file}")
        return 1

    print(f"Qualcomm AI Hub token found: {mask_secret(api_key)}")
    if not args.persist:
        print("Dry run only. Pass --persist to run persistent qai-hub configure.")
        print("This would write the token to the user's local Qualcomm AI Hub config.")
        return 0

    executable = shutil.which("qai-hub")
    if executable is None:
        print("Install optional dependency: uv pip install qai-hub python-dotenv")
        return 1

    configure = subprocess.run(
        [executable, "configure", "--api_token", api_key],
        check=False,
        capture_output=True,
        text=True,
    )
    if configure.returncode != 0:
        print("qai-hub configure failed.")
        if configure.stderr.strip():
            print(_redact_secret(configure.stderr.strip(), api_key))
        return configure.returncode

    print("Persistent Qualcomm AI Hub config updated outside the repository.")
    list_devices = subprocess.run(
        [executable, "list-devices"],
        check=False,
        capture_output=True,
        text=True,
    )
    if list_devices.stdout.strip():
        print(list_devices.stdout.strip())
    if list_devices.stderr.strip():
        print(list_devices.stderr.strip())
    return list_devices.returncode


def _redact_secret(text: str, secret: str) -> str:
    return text.replace(secret, "<token>") if secret else text


if __name__ == "__main__":
    raise SystemExit(main())
