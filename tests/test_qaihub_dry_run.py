from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_print_qaihub_commands_does_not_leak_token() -> None:
    env = os.environ.copy()
    env["QUALCOMM_AI_HUB_API_KEY"] = "secret-token-that-must-not-leak"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/print_qaihub_commands.py",
            "--model",
            "outputs/uci_har_tinytcn/model.tflite",
            "--device",
            "Samsung Galaxy S24 (Family)",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "secret-token-that-must-not-leak" not in result.stdout
    assert "<YOUR_TOKEN>" in result.stdout


def test_profile_qaihub_dry_run_does_not_need_credentials(tmp_path) -> None:
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"fake tflite payload")
    output_json = tmp_path / "profile_summary.json"

    env = os.environ.copy()
    env.pop("QUALCOMM_AI_HUB_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "scripts/profile_qualcomm_ai_hub.py",
            "--model",
            str(model_path),
            "--input-shape",
            "1,128,6",
            "--device",
            "Samsung Galaxy S24 (Family)",
            "--compute-unit",
            "cpu",
            "--output-json",
            str(output_json),
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "QUALCOMM_AI_HUB_API_KEY" not in result.stdout

    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["status"] == "pending"
    assert summary["compute_unit_requested"] == "cpu"
    assert summary["latency_ms"] is None
    assert "token" not in json.dumps(summary).lower()
