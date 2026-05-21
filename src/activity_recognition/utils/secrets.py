"""Secret-loading helpers for optional local integrations."""

from __future__ import annotations

import os
from pathlib import Path

QUALCOMM_API_KEY_ENV = "QUALCOMM_AI_HUB_API_KEY"


def load_env_file(path: str = ".secrets/.env") -> None:
    """Load environment variables from a local env file if it exists.

    The function prefers python-dotenv when available, but keeps a tiny parser
    fallback so the core project does not depend on optional Qualcomm tooling.
    Existing environment variables are not overwritten.
    """

    env_path = Path(path)
    if not env_path.exists():
        return

    try:
        from dotenv import load_dotenv
    except ImportError:
        _load_env_file_fallback(env_path)
        return

    load_dotenv(env_path, override=False)


def get_qualcomm_api_key(env_path: str = ".secrets/.env") -> str | None:
    """Return the Qualcomm AI Hub API key from env or a local env file."""

    load_env_file(env_path)
    value = os.environ.get(QUALCOMM_API_KEY_ENV)
    return value.strip() if value and value.strip() else None


def mask_secret(secret: str) -> str:
    """Return a display-safe masked secret."""

    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}...{secret[-4:]}"


def _load_env_file_fallback(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = _strip_inline_comment(value.strip())
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _strip_inline_comment(value: str) -> str:
    quote_char = ""
    for index, char in enumerate(value):
        if char in {"'", '"'}:
            quote_char = "" if quote_char == char else char
        if (
            char == "#"
            and not quote_char
            and (index == 0 or value[index - 1].isspace())
        ):
            return value[:index].rstrip()
    return value
