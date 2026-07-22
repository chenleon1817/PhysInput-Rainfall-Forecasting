import importlib.util
from pathlib import Path

import numpy as np
import pytest
tf = pytest.importorskip("tensorflow")


def test_debug_truncation_ignores_top_level_metadata():
    source = Path(__file__).resolve().parents[1] / "scripts" / "05_train_models.py"
    spec = importlib.util.spec_from_file_location("train_script", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = type("Args", (), {"debug_max_train_samples": 2, "debug_max_validation_samples": 3, "debug_max_test_samples": 4})()
    splits = {
        name: {"X_seq": np.zeros((5, 1)), "y_reg": np.zeros((5, 1)), "rain_seq_reg": np.zeros((5, 1, 1)), "valid_times": np.arange(5)}
        for name in ("train", "validation", "test")
    }
    splits["grid_points_order"] = [(0, 0)]
    splits["feature_columns"] = ["T"]
    module._truncate_debug_splits(splits, args)
    assert [len(splits[name]["y_reg"]) for name in ("train", "validation", "test")] == [2, 3, 4]
    assert splits["grid_points_order"] == [(0, 0)]
