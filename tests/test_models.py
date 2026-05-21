import tensorflow as tf

from activity_recognition.models.cnn1d import build_cnn1d
from activity_recognition.models.mlp import build_mlp


def test_model_forward_passes():
    X = tf.zeros((2, 32, 4), dtype=tf.float32)
    mlp = build_mlp((32, 4), num_classes=3, verbose=False)
    cnn = build_cnn1d((32, 4), num_classes=3, verbose=False)

    assert mlp(X).shape == (2, 3)
    assert cnn(X).shape == (2, 3)
