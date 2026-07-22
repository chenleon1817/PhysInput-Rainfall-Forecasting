import pytest

tf = pytest.importorskip("tensorflow")

from core.models import _build_grid_preserving_cnn_stem


def test_cnn_stem_concatenates_formal_branch_order():
    inputs = tf.keras.layers.Input(shape=(2, 9, 5, 3))
    outputs = _build_grid_preserving_cnn_stem(inputs, filters=4)
    model = tf.keras.Model(inputs, outputs)
    concat = model.get_layer("grid_cnn_stem_concat")
    names = [getattr(tensor._keras_history, "operation", tensor._keras_history.layer).name for tensor in concat.input]
    assert names == [
        "grid_cnn_stem_local_3x3",
        "grid_cnn_stem_dilated_3x3",
        "grid_cnn_stem_meridional_3x1",
        "grid_cnn_stem_zonal_1x3",
        "grid_cnn_stem_broad_5x3",
        "grid_cnn_stem_raw_bypass",
    ]
