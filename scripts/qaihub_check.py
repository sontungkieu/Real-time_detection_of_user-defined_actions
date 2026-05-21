#!/usr/bin/env python3
"""Check optional Qualcomm AI Hub environment setup without submitting jobs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.utils.secrets import get_qualcomm_api_key, mask_secret

INSTALL_MESSAGE = "Install optional dependency: uv pip install qai-hub python-dotenv"
CONFIGURE_MESSAGE = "Run: qai-hub configure --api_token <token>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device-filter", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = get_qualcomm_api_key(args.env_file)
    if not api_key:
        print(f"Missing QUALCOMM_AI_HUB_API_KEY in {args.env_file}")
        return 1

    print(f"Qualcomm AI Hub token found: {mask_secret(api_key)}")

    try:
        import qai_hub as hub
    except ImportError:
        print(INSTALL_MESSAGE)
        return 1

    client = _build_session_client(hub, api_key)
    if client is None:
        print(CONFIGURE_MESSAGE)
        return 1

    print("qai-hub import ok")
    if args.list_devices:
        return _list_devices(client, args.device_filter)
    return 0


def _build_session_client(hub, api_key: str):
    try:
        client_config = hub.ClientConfig(api_token=api_key)
        return hub.Client(client_config)
    except (AttributeError, TypeError):
        return None
    except Exception as exc:
        print(
            "Unable to initialize session client: "
            f"{type(exc).__name__}: {_redact_secret(str(exc), api_key)}"
        )
        return None


def _list_devices(client, device_filter: str) -> int:
    try:
        devices = client.get_devices()
    except Exception as exc:
        print(f"Unable to list Qualcomm AI Hub devices: {type(exc).__name__}: {exc}")
        print(CONFIGURE_MESSAGE)
        return 1

    normalized_filter = device_filter.lower().strip()
    matched = []
    for device in devices:
        name = str(getattr(device, "name", device))
        if normalized_filter and normalized_filter not in name.lower():
            continue
        matched.append(name)

    if not matched:
        print("No matching Qualcomm AI Hub devices found.")
        return 0

    for name in matched:
        print(name)
    return 0


def _redact_secret(text: str, secret: str) -> str:
    return text.replace(secret, "<token>") if secret else text


if __name__ == "__main__":
    raise SystemExit(main())
