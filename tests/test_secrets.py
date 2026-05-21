from __future__ import annotations

from activity_recognition.utils.secrets import get_qualcomm_api_key, mask_secret


def test_mask_secret_hides_short_values() -> None:
    assert mask_secret("abc123") == "******"


def test_mask_secret_keeps_only_edges_for_long_values() -> None:
    assert mask_secret("abcd1234wxyz") == "abcd...wxyz"


def test_missing_env_file_returns_none(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUALCOMM_AI_HUB_API_KEY", raising=False)

    assert get_qualcomm_api_key(str(tmp_path / "missing.env")) is None


def test_env_file_loads_qualcomm_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("QUALCOMM_AI_HUB_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OTHER=value\nQUALCOMM_AI_HUB_API_KEY=test-token-123\n",
        encoding="utf-8",
    )

    assert get_qualcomm_api_key(str(env_file)) == "test-token-123"
