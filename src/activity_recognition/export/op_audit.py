"""TensorFlow Lite operator audit helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import tensorflow as tf


def audit_tflite_ops(
    model_path: str | Path,
    out_path: str | Path | None = None,
) -> dict[str, Any]:
    """Inspect TFLite operator names and save a compact audit JSON."""

    model_path = Path(model_path)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    ops_details = getattr(interpreter, "_get_ops_details")()
    op_names = [str(op.get("op_name", "UNKNOWN")) for op in ops_details]
    counts = dict(sorted(Counter(op_names).items()))
    result = {
        "model": str(model_path),
        "ops": counts,
        "total_ops": len(op_names),
        "has_space_to_batch": "SPACE_TO_BATCH_ND" in counts,
        "has_batch_to_space": "BATCH_TO_SPACE_ND" in counts,
        "op_sequence": op_names,
    }

    output = Path(out_path) if out_path else model_path.with_suffix(".op_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
