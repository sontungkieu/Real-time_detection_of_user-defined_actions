import tensorflow as tf
import pytest

from activity_recognition.models.cnn1d import build_cnn1d
from activity_recognition.models.mlp import build_mlp
from activity_recognition.models.registry import build_model, list_models
from activity_recognition.models.tinytcn import build_tinytcn


def test_model_forward_passes():
    X = tf.zeros((2, 32, 4), dtype=tf.float32)
    mlp = build_mlp((32, 4), num_classes=3, verbose=False)
    cnn = build_cnn1d((32, 4), num_classes=3, verbose=False)
    tinytcn = build_tinytcn((32, 4), num_classes=3, verbose=False)

    assert mlp(X).shape == (2, 3)
    assert cnn(X).shape == (2, 3)
    assert tinytcn(X).shape == (2, 3)


def test_model_registry_builds_v5_models():
    model_names = {
        "tinytcn",
        "tiny_cnn1d",
        "tiny_dscnn1d",
        "medium_conv1d",
        "resnet1d_small",
    }
    assert model_names.issubset(set(list_models()))

    X = tf.zeros((1, 128, 6), dtype=tf.float32)
    for model_name in model_names:
        model = build_model(
            model_name,
            input_shape=(128, 6),
            num_classes=6,
            config={"verbose": False},
        )
        assert model(X).shape == (1, 6)


def test_model_registry_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("missing", input_shape=(128, 6), num_classes=6)
