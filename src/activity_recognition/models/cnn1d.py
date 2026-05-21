"""Lightweight 1D-CNN baseline for windowed motion signals."""

from __future__ import annotations

import tensorflow as tf


def build_cnn1d(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    verbose: bool = True,
) -> tf.keras.Model:
    """Build and compile a compact 1D-CNN classifier."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(
                32, kernel_size=5, padding="same", activation="relu"
            ),
            tf.keras.layers.Conv1D(
                64, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="wisdm_cnn1d",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if verbose:
        print(f"1D-CNN parameters: {model.count_params():,}")
    return model
