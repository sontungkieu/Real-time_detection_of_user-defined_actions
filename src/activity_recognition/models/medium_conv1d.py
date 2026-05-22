"""Medium compute-heavy Conv1D HAR model."""

from __future__ import annotations

import tensorflow as tf


def build_medium_conv1d(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    verbose: bool = True,
) -> tf.keras.Model:
    """Build a larger Conv1D model for accelerator crossover experiments."""

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(
                64, kernel_size=7, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                64, kernel_size=5, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                64, kernel_size=5, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(
                128, kernel_size=5, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                128, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(
                128, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(
                128, kernel_size=3, padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="medium_conv1d",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if verbose:
        print(f"MediumConv1D parameters: {model.count_params():,}")
    return model
