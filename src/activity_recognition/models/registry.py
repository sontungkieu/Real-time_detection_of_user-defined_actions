"""Model registry for HAR architecture sweeps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorflow as tf

from activity_recognition.models.cnn1d import build_cnn1d
from activity_recognition.models.medium_conv1d import build_medium_conv1d
from activity_recognition.models.mlp import build_mlp
from activity_recognition.models.resnet1d_small import build_resnet1d_small
from activity_recognition.models.tiny_cnn1d import build_tiny_cnn1d
from activity_recognition.models.tiny_dscnn1d import build_tiny_dscnn1d
from activity_recognition.models.tinytcn import build_tinytcn

ModelBuilder = Callable[..., tf.keras.Model]

_BUILDERS: dict[str, ModelBuilder] = {
    "mlp": build_mlp,
    "cnn1d": build_cnn1d,
    "tinytcn": build_tinytcn,
    "tiny_cnn1d": build_tiny_cnn1d,
    "tiny_dscnn1d": build_tiny_dscnn1d,
    "medium_conv1d": build_medium_conv1d,
    "resnet1d_small": build_resnet1d_small,
}


def list_models() -> list[str]:
    """Return supported model names."""

    return sorted(_BUILDERS)


def build_model(
    name: str,
    input_shape: tuple[int, int],
    num_classes: int,
    config: dict[str, Any] | None = None,
) -> tf.keras.Model:
    """Build and compile a registered HAR model."""

    normalized_name = name.lower()
    if normalized_name not in _BUILDERS:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model {name!r}. Available models: {available}")

    config = dict(config or {})
    learning_rate = float(config.pop("learning_rate", 0.001))
    verbose = bool(config.pop("verbose", True))
    ignored_keys = {
        "type",
        "epochs",
        "batch_size",
        "description",
    }
    builder_kwargs = {
        key: value for key, value in config.items() if key not in ignored_keys
    }
    return _BUILDERS[normalized_name](
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
        verbose=verbose,
        **builder_kwargs,
    )


def main() -> int:
    """Small CLI for smoke-testing the registry."""

    for model_name in list_models():
        print(model_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
