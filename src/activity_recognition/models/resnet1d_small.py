"""Small residual Conv1D HAR model."""

from __future__ import annotations

import tensorflow as tf


def build_resnet1d_small(
    input_shape: tuple[int, int],
    num_classes: int,
    learning_rate: float = 0.001,
    dropout: float = 0.25,
    verbose: bool = True,
) -> tf.keras.Model:
    """Build a compact residual Conv1D model."""

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, kernel_size=5, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = _residual_block(x, filters=32, kernel_size=3)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = _residual_block(x, filters=64, kernel_size=3)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = _residual_block(x, filters=96, kernel_size=3)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet1d_small")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    if verbose:
        print(f"ResNet1D-small parameters: {model.count_params():,}")
    return model


def _residual_block(
    inputs: tf.Tensor, filters: int, kernel_size: int
) -> tf.keras.layers.Layer:
    shortcut = inputs
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="same")(
            shortcut
        )
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([shortcut, x])
    return tf.keras.layers.Activation("relu")(x)
