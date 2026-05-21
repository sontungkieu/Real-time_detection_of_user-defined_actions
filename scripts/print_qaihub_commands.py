#!/usr/bin/env python3
"""Print manual Qualcomm AI Hub CLI commands without reading secrets."""

from __future__ import annotations

import argparse
import shutil
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/uci_har_tinytcn/model.tflite")
    parser.add_argument("--device", default="Samsung Galaxy S24 (Family)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supports_compute_unit = _submit_profile_help_supports_compute_unit()

    print("Configure:")
    print("qai-hub configure --api_token <YOUR_TOKEN>")
    print()
    print("List devices:")
    print("qai-hub list-devices")
    print()
    print("Profile CPU:")
    print(
        "qai-hub submit-profile-job "
        f"--model {args.model} "
        f'--device "{args.device}" '
        '--profile_options " --compute_unit cpu" '
        "--wait"
    )
    print()
    print("Profile NPU:")
    if supports_compute_unit:
        print(
            "qai-hub submit-profile-job "
            f"--model {args.model} "
            f'--device "{args.device}" '
            '--profile_options " --compute_unit npu" '
            "--wait"
        )
    else:
        print(
            "Check `qai-hub submit-profile-job --help` for compute-unit/profile options."
        )
    return 0


def _submit_profile_help_supports_compute_unit() -> bool:
    executable = shutil.which("qai-hub")
    if executable is None:
        return False

    result = subprocess.run(
        [executable, "submit-profile-job", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    return "compute_unit" in result.stdout or "compute-unit" in result.stdout


if __name__ == "__main__":
    raise SystemExit(main())
