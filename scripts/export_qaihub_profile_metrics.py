#!/usr/bin/env python3
"""Export structured metrics from Qualcomm AI Hub profile logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from activity_recognition.utils.qaihub_metrics import (  # noqa: E402
    flatten_record_for_csv,
    layer_rows_from_record,
    parse_runtime_analysis_text,
    parse_runtime_log,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        nargs="*",
        default=[],
        help="Runtime log files downloaded from Qualcomm AI Hub.",
    )
    parser.add_argument(
        "--runtime-analysis",
        nargs="*",
        default=[],
        help=(
            "Runtime Analysis table files. Use JOB_ID=path to attach to a job, "
            "or JOB_ID=- to read one table from stdin."
        ),
    )
    parser.add_argument(
        "--output-json",
        default="outputs/qualcomm_ai_hub/profile_metrics.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/qualcomm_ai_hub/profile_metrics.csv",
    )
    parser.add_argument(
        "--layer-csv",
        default="outputs/qualcomm_ai_hub/profile_layers.csv",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = [parse_runtime_log(path) for path in args.logs]
    records_by_job = {
        record.get("job_id"): record for record in records if record.get("job_id")
    }

    for item in args.runtime_analysis:
        job_id, text = _read_runtime_analysis_item(item)
        analysis = parse_runtime_analysis_text(text, job_id=job_id)
        record = records_by_job.get(job_id)
        if record is None:
            record = _empty_record(job_id)
            records.append(record)
            if job_id:
                records_by_job[job_id] = record
        record["runtime_analysis"] = analysis

    summary_rows = [flatten_record_for_csv(record) for record in records]
    layer_rows = [row for record in records for row in layer_rows_from_record(record)]

    payload: dict[str, Any] = {
        "records": records,
        "summary_rows": summary_rows,
        "layer_rows": layer_rows,
    }
    write_json(payload, args.output_json)
    write_csv(summary_rows, args.output_csv)
    write_csv(layer_rows, args.layer_csv)

    print(f"Wrote {len(records)} profile records to {args.output_json}")
    print(f"Wrote summary CSV to {args.output_csv}")
    print(f"Wrote layer CSV to {args.layer_csv}")
    return 0


def _read_runtime_analysis_item(item: str) -> tuple[str | None, str]:
    if "=" in item:
        job_id, path_text = item.split("=", 1)
        job_id = job_id or None
    else:
        job_id, path_text = None, item

    if path_text == "-":
        return job_id, sys.stdin.read()
    return job_id, Path(path_text).read_text(encoding="utf-8")


def _empty_record(job_id: str | None) -> dict[str, Any]:
    return {
        "source_log": None,
        "job_id": job_id,
        "model_name": None,
        "device": {},
        "compute_unit_observed": None,
        "delegate": {},
        "qnn": {},
        "timings": {},
        "memory": {"unit": "MB", "source_unit": "kB"},
        "by_layer": {"entries": [], "summary": {}},
        "diagnostics": {},
    }


if __name__ == "__main__":
    raise SystemExit(main())
