"""Tests for the formal CNN stem branch-concatenation order."""

import pytest

tf = pytest.importorskip("tensorflow")

from core.models import _build_grid_preserving_cnn_stem


def _get_producing_layer_name(tensor) -> str:
    """Return the Keras layer or operation that produced a symbolic tensor."""
    history = tensor._keras_history

    operation = getattr(history, "operation", None)
    if operation is not None:
        return operation.name

    layer = getattr(history, "layer", None)
    if layer is not None:
        return layer.name

    raise AssertionError(
        "Unable to identify the Keras operation that produced the tensor."
    )


def test_cnn_stem_concatenates_formal_branch_order() -> None:
    """The six CNN branches must be concatenated in the formal order."""
    inputs = tf.keras.layers.Input(shape=(2, 9, 5, 3))
    outputs = _build_grid_preserving_cnn_stem(
        inputs,
        filters=4,
    )
    model = tf.keras.Model(inputs, outputs)

    concat = model.get_layer("grid_cnn_stem_concat")
    names = [
        _get_producing_layer_name(tensor)
        for tensor in concat.input
    ]

    assert names == [
        "grid_cnn_stem_local_3x3",
        "grid_cnn_stem_dilated_3x3",
        "grid_cnn_stem_meridional_3x1",
        "grid_cnn_stem_zonal_1x3",
        "grid_cnn_stem_broad_5x3",
        "grid_cnn_stem_raw_bypass",
    ]
