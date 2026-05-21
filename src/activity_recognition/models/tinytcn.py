"""Tiny temporal convolution baseline for windowed motion signals."""

from __future__ import annotations

import tensorflow as tf


def build_tinytcn(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    verbose: bool = True,
) -> tf.keras.Model:
    """Build and compile a compact TFLite-friendly temporal CNN."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(
                32, kernel_size=3, dilation_rate=1, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Conv1D(
                32, kernel_size=3, dilation_rate=2, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv1D(
                64, kernel_size=3, dilation_rate=4, padding="same", activation="relu"
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="tinytcn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if verbose:
        print(f"TinyTCN parameters: {model.count_params():,}")
    return model
