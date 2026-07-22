import pytest

tf = pytest.importorskip("tensorflow")

from core.training_utils import (
    EpochProgressLogger,
    ValidationEventScoreCheckpointMetric,
    build_training_callbacks,
)


def test_formal_callback_order(tmp_path):
    config = {"training": {"batch_size": 1, "early_stopping_patience": 1, "reduce_lr_factor": 0.5, "reduce_lr_patience": 1, "minimum_learning_rate": 1e-7}, "checkpoint": {"event_threshold_mm_per_hour": 10, "event_score": {}, "early_stopping_min_delta": 1e-4}}
    callbacks = build_training_callbacks([[0]], [[0]], config, tmp_path / "best.weights.h5")
    assert isinstance(callbacks[0], ValidationEventScoreCheckpointMetric)
    assert isinstance(callbacks[1], tf.keras.callbacks.EarlyStopping)
    assert isinstance(callbacks[2], tf.keras.callbacks.ReduceLROnPlateau)
    assert isinstance(callbacks[3], EpochProgressLogger)
    assert isinstance(callbacks[4], tf.keras.callbacks.ModelCheckpoint)
