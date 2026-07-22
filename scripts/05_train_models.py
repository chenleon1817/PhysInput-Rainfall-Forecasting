#!/usr/bin/env python3
"""Run the sole formal train, checkpoint, prediction, and export workflow."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

from core.config_loader import load_experiment_config, load_physinput_model_config
from core.data_processing import (
    append_lagged_rainfall_channel_train_validation_test,
    process_regional_data_for_training,
    select_model_input_channels,
)
from core.evaluation_utils import evaluate_regional_predictions
from core.models import (
    get_formal_model_ids,
    predict_linear_arx_baseline,
    train_linear_arx_baseline,
)
from core.result_export import (
    export_model_metrics,
    export_predictions_long,
    save_prediction_npz,
    write_json,
)
from core.runtime import setup_device
from core.training_utils import (
    build_intensity_sample_weight,
    train_regional_model,
    write_training_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train canonical rainfall models and export formal predictions.")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("converted_data"))
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment_config.json"))
    parser.add_argument("--physinput-config", type=Path, default=Path("configs/physinput_model_config.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-max-train-samples", type=int, default=64)
    parser.add_argument("--debug-max-validation-samples", type=int, default=64)
    parser.add_argument("--debug-max-test-samples", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace, config: dict) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_kind = "debug" if args.debug else "run"
    return Path(config["outputs"]["training_root"]) / f"{run_kind}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _safe_clean_run_directory(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    forbidden = {
        PROJECT_ROOT.resolve(),
        (PROJECT_ROOT / "outputs").resolve(),
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(resolved.anchor),
    }
    if resolved in forbidden or resolved.parent == resolved:
        raise ValueError(f"Refusing to delete unsafe output directory: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def prepare_run_directory(run_dir: Path, overwrite: bool) -> Path:
    if run_dir.exists() and any(run_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Run directory exists: {run_dir}; use --overwrite to recreate it.")
        _safe_clean_run_directory(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _append_in_shared_order(tensor: np.ndarray) -> np.ndarray:
    """Place lagged rainfall after the ten meteorological shared channels."""
    return np.concatenate([tensor[..., :10], tensor[..., -1:], tensor[..., 10:-1]], axis=-1)


def _truncate_debug_splits(splits: dict, args: argparse.Namespace) -> None:
    limits = {
        "train": args.debug_max_train_samples,
        "validation": args.debug_max_validation_samples,
        "test": args.debug_max_test_samples,
    }
    for split_name in ("train", "validation", "test"):
        split = splits[split_name]
        limit = min(max(1, int(limits[split_name])), len(split["y_reg"]))
        for key in ("X_seq", "y_reg", "rain_seq_reg", "valid_times"):
            split[key] = split[key][:limit]


def _save_predictions(directory: Path, name: str, split: dict, model_id: str, seed: int) -> Path:
    return save_prediction_npz(
        directory / name,
        split["y_reg"],
        split["y_pred"],
        split["valid_times"],
        split["grid_points_order"],
        model_id,
        seed,
    )


def _build_formal_run_config(config: dict, phys: dict, selected: list[str], raw_features: list[str], args: argparse.Namespace) -> dict:
    return {
        "seed": args.seed,
        "models": selected,
        "years": config["data"]["years"],
        "grid_shape": config["data"]["grid_shape"],
        "input_hours": config["data"]["input_hours"],
        "prediction_horizon": config["data"]["prediction_horizon"],
        "split": config["split"],
        "thresholds": config["evaluation"]["thresholds_mm_per_hour"],
        "loss_mode": config["training"]["loss_mode"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
        "learning_rate": config["training"]["learning_rate"],
        "sample_balance": config["sample_balance"],
        "sample_weight": config["sample_weight"],
        "checkpoint": config["checkpoint"],
        "meteorological_feature_order": raw_features,
        "physinput_diagnostic_combinations": phys["physinput_variants"],
        "debug": bool(args.debug),
        "formal_paper_reproduction": False if args.debug else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment_config)
    phys = load_physinput_model_config(args.physinput_config)
    config["training"]["seed"] = args.seed
    if args.debug:
        config["training"]["epochs"] = 2
    selected = get_formal_model_ids() if args.models == ["all"] else list(args.models)
    unknown = set(selected) - set(get_formal_model_ids())
    if unknown:
        raise ValueError(f"Unknown requested models: {sorted(unknown)}")
    setup_device()
    run_dir = prepare_run_directory(_run_dir(args, config), args.overwrite)
    raw_features = [name for name in phys["shared_input_features"] if name != "lagged_rainfall"] + phys["physical_diagnostic_features"]
    write_json(run_dir / "resolved_experiment_config.json", config)
    write_json(run_dir / "resolved_physinput_config.json", phys)
    write_json(run_dir / "formal_run_config.json", _build_formal_run_config(config, phys, selected, raw_features, args))
    prepared = process_regional_data_for_training(
        args.data_dir,
        config["data"]["years"],
        config["data"]["input_hours"],
        raw_features,
        tuple(config["data"]["grid_shape"]),
        config["split"],
        config["sample_balance"],
        run_dir / "sample_balance",
    )
    splits = {
        "train": {"X_seq": prepared["X_train"], "y_reg": prepared["y_train"], "rain_seq_reg": prepared["rain_seq_train"], "valid_times": prepared["times_train"]},
        "validation": {"X_seq": prepared["X_validation"], "y_reg": prepared["y_validation"], "rain_seq_reg": prepared["rain_seq_validation"], "valid_times": prepared["times_validation"]},
        "test": {"X_seq": prepared["X_test"], "y_reg": prepared["y_test"], "rain_seq_reg": prepared["rain_seq_test"], "valid_times": prepared["times_test"]},
    }
    for split in splits.values():
        split["grid_points_order"] = prepared["grid_points_order"]
    preprocessing_dir = run_dir / "preprocessing"
    preprocessing_dir.mkdir(exist_ok=True)
    joblib.dump(prepared["scalers"]["standard"], preprocessing_dir / "feature_scaler.joblib")
    preprocessing_metadata = {key: value for key, value in prepared.items() if key != "scalers"}
    write_json(preprocessing_dir / "preprocessing_metadata.json", preprocessing_metadata)
    train_x, validation_x, test_x, lag_stats = append_lagged_rainfall_channel_train_validation_test(
        splits["train"]["X_seq"], splits["validation"]["X_seq"], splits["test"]["X_seq"],
        splits["train"]["rain_seq_reg"], splits["validation"]["rain_seq_reg"], splits["test"]["rain_seq_reg"],
        splits["train"]["y_reg"], splits["validation"]["y_reg"], splits["test"]["y_reg"],
    )
    write_json(preprocessing_dir / "lagged_rainfall_scaler.json", lag_stats)
    for split_name, tensor in zip(("train", "validation", "test"), (train_x, validation_x, test_x)):
        splits[split_name]["X_seq"] = _append_in_shared_order(tensor)
    if args.debug:
        _truncate_debug_splits(splits, args)
    all_features = phys["shared_input_features"] + phys["physical_diagnostic_features"]
    manifest = {"seed": args.seed, "models": {}, "sample_balance": prepared["sample_balance"], "debug": bool(args.debug), "status": "running"}
    for model_id in selected:
        directory = run_dir / "models" / model_id
        directory.mkdir(parents=True, exist_ok=True)
        train, validation, test = splits["train"], splits["validation"], splits["test"]
        if model_id == "linear_arx":
            model = train_linear_arx_baseline(train["X_seq"][..., :10], train["y_reg"], train["rain_seq_reg"], args.seed)
            joblib.dump(model, directory / "linear_arx.joblib")
            validation_prediction = predict_linear_arx_baseline(model, validation["X_seq"][..., :10], validation["rain_seq_reg"])
            test_prediction = predict_linear_arx_baseline(model, test["X_seq"][..., :10], test["rain_seq_reg"])
            train_info = {"model_id": model_id, "sample_weight_used": False, "selected_checkpoint_type": None}
            write_json(directory / "train_info.json", train_info)
        else:
            X_train, feature_names, diagnostics = select_model_input_channels(train["X_seq"], all_features, model_id, phys)
            X_validation, _, _ = select_model_input_channels(validation["X_seq"], all_features, model_id, phys)
            X_test, _, _ = select_model_input_channels(test["X_seq"], all_features, model_id, phys)
            sample_weight, sample_weight_statistics = build_intensity_sample_weight(train["y_reg"], config)
            result = train_regional_model(model_id, X_train, train["y_reg"], X_validation, validation["y_reg"], sample_weight, sample_weight_statistics, config, directory, feature_names, diagnostics, phys)
            model = result["model"]
            validation_prediction = np.clip(model.predict(X_validation, batch_size=config["training"]["batch_size"], verbose=0), 0, None)
            test_prediction = np.clip(model.predict(X_test, batch_size=config["training"]["batch_size"], verbose=0), 0, None)
            train_info = result["train_info"]
        validation["y_pred"] = validation_prediction
        test["y_pred"] = test_prediction
        _save_predictions(directory, "validation_predictions.npz", validation, model_id, args.seed)
        _save_predictions(directory, "test_predictions.npz", test, model_id, args.seed)
        export_predictions_long(directory / "predictions_long.csv", test["valid_times"], test["grid_points_order"], test["y_reg"], test_prediction, model_id, args.seed)
        export_model_metrics(directory / "validation_metrics.json", evaluate_regional_predictions(validation["y_reg"], validation_prediction, config["evaluation"]["thresholds_mm_per_hour"]))
        export_model_metrics(directory / "test_metrics.json", evaluate_regional_predictions(test["y_reg"], test_prediction, config["evaluation"]["thresholds_mm_per_hour"]))
        manifest["models"][model_id] = train_info
    manifest["status"] = "completed"
    write_training_manifest(run_dir / "training_manifest.json", manifest)
    print(f"Training contract completed: {run_dir}")


if __name__ == "__main__":
    main()
