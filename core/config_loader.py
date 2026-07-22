"""Validated configuration loading for the reproducible public workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SHARED_FEATURES = ["T", "P", "RH", "PWV", "U", "V", "DOY", "HOD", "dPWV+", "dPWV-", "lagged_rainfall"]
DIAGNOSTIC_FEATURES = ["specific_humidity_proxy", "pwv_grad_mag", "wind_convergence", "moisture_advection"]
FORMAL_MODEL_IDS = ["linear_arx", "bilstm", "convlstm", "cnn_bilstm_attention", "cnn_bilstm", "physinput_lite", "physinput_adv3", "physinput_conv3", "physinput_all4"]

def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required configuration file does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in configuration file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a JSON object.")
    return data

def load_experiment_config(path: str | Path = REPOSITORY_ROOT / "configs" / "experiment_config.json") -> dict[str, Any]:
    data = _read_json(path)
    validate_experiment_config(data)
    return data

def load_physinput_model_config(path: str | Path = REPOSITORY_ROOT / "configs" / "physinput_model_config.json") -> dict[str, Any]:
    data = _read_json(path)
    validate_physinput_model_config(data)
    return data

def validate_experiment_config(data: dict[str, Any]) -> None:
    required = {"project", "data", "split", "training", "sample_balance", "sample_weight", "models", "evaluation", "runtime", "outputs"}
    missing = sorted(required - set(data))
    if missing: raise ValueError(f"Experiment configuration is missing sections: {missing}")
    if data["data"].get("grid_shape") != [9, 5]: raise ValueError("The formal grid shape must be [9, 5].")
    if data["data"].get("input_hours") != 12 or data["data"].get("prediction_horizon") != 1: raise ValueError("The formal setup requires 12 input hours and a one-hour horizon.")
    if list(data["models"]) != FORMAL_MODEL_IDS: raise ValueError("The model list must contain exactly the nine formal model IDs in canonical order.")
    if resolve_thresholds(data["evaluation"].get("thresholds_mm_per_hour")) != [0.1, 2.5, 10.0]: raise ValueError("Formal event thresholds must be [0.1, 2.5, 10.0].")

def validate_physinput_model_config(data: dict[str, Any]) -> None:
    required = {"stage_name", "source_feature_analysis_artifact", "shared_input_features", "shared_derived_predictors", "physical_diagnostic_features", "physinput_variants", "physical_process_roles", "physinput_settings"}
    missing = sorted(required - set(data))
    if missing: raise ValueError(f"PhysInput configuration is missing fields: {missing}")
    if data["shared_input_features"] != SHARED_FEATURES: raise ValueError("Shared input features must match the formal ordered eleven-feature list.")
    if data["physical_diagnostic_features"] != DIAGNOSTIC_FEATURES: raise ValueError("Physical diagnostic features must match the formal ordered four-feature list.")
    expected = {"physinput_lite": DIAGNOSTIC_FEATURES[:2], "physinput_adv3": [DIAGNOSTIC_FEATURES[0], DIAGNOSTIC_FEATURES[1], DIAGNOSTIC_FEATURES[3]], "physinput_conv3": DIAGNOSTIC_FEATURES[:3], "physinput_all4": DIAGNOSTIC_FEATURES}
    if data["physinput_variants"] != expected: raise ValueError("PhysInput variant channel definitions are not the formal combinations.")
    settings = data["physinput_settings"]
    for key in ("fixed_physical_weights_used", "diagnostic_proportions_used", "separate_physical_branch_used", "test_time_postprocessing_used"):
        if settings.get(key) is not False: raise ValueError(f"PhysInput setting {key} must be false.")
    if settings.get("append_diagnostics_as_input_channels") is not True: raise ValueError("PhysInput diagnostics must be appended as input channels.")

def resolve_feature_columns(model_id: str, physinput_config: dict[str, Any]) -> list[str]:
    validate_physinput_model_config(physinput_config)
    if model_id in {"linear_arx", "bilstm", "convlstm", "cnn_bilstm_attention", "cnn_bilstm"}: return list(physinput_config["shared_input_features"])
    if model_id not in physinput_config["physinput_variants"]: raise ValueError(f"Unknown formal model ID: {model_id}")
    return list(physinput_config["shared_input_features"]) + list(physinput_config["physinput_variants"][model_id])

def resolve_thresholds(thresholds: Any) -> list[float]:
    values = list(thresholds.values()) if isinstance(thresholds, dict) else list(thresholds)
    values = [float(value) for value in values]
    if len(values) != 3 or not (0 <= values[0] < values[1] < values[2]): raise ValueError("Thresholds must be three ascending values.")
    return values
