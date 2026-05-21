"""Small MLP baseline for windowed motion signals."""

from __future__ import annotations

import tensorflow as tf


def build_mlp(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    verbose: bool = True,
) -> tf.keras.Model:
    """Build and compile a compact MLP classifier."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="mlp",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if verbose:
        print(f"MLP parameters: {model.count_params():,}")
    return model
