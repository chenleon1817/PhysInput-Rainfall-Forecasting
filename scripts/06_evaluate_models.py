#!/usr/bin/env python3
"""Evaluate saved formal test predictions without retraining."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from core.config_loader import load_experiment_config
from core.evaluation_utils import compute_valid_rain_mae, evaluate_regional_predictions
from core.models import get_formal_model_ids, get_model_display_name
from core.result_export import (
    export_evaluation_tables,
    export_physinput_ablation_at10,
    export_predictions_long,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate test predictions from completed training runs.")
    parser.add_argument("--training-run-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=Path("configs/experiment_config.json"))
    return parser.parse_args()


def _archive_identity(archive: np.lib.npyio.NpzFile) -> tuple:
    return archive["y_true"], archive["valid_times"], archive["grid_points_order"], int(archive["seed"])


def _assert_identity(reference: tuple, candidate: tuple, label: str) -> None:
    names = ("y_true", "valid_times", "grid_points_order", "seed")
    for name, expected, actual in zip(names, reference, candidate):
        if not np.array_equal(expected, actual):
            raise ValueError(f"Prediction archive identity mismatch for {label}: {name} differs.")


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.experiment_config)
    thresholds = config["evaluation"]["thresholds_mm_per_hour"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, events, intensity, pointwise, positive, confusion = [], [], [], [], [], []
    long_rows: dict[str, list[pd.DataFrame]] = {}
    available, seeds, run_references = set(), [], []
    for run_dir in args.training_run_dir:
        run_config = json.loads((run_dir / "formal_run_config.json").read_text(encoding="utf-8"))
        run_seed = int(run_config["seed"])
        run_reference = None
        for model_id in get_formal_model_ids():
            path = run_dir / "models" / model_id / "test_predictions.npz"
            if not path.is_file():
                continue
            archive = np.load(path, allow_pickle=True)
            identity = _archive_identity(archive)
            if identity[3] != run_seed:
                raise ValueError(f"Prediction archive seed differs from formal_run_config for {model_id} in {run_dir}.")
            if run_reference is None:
                run_reference = identity
            else:
                _assert_identity(run_reference, identity, f"{model_id} in {run_dir}")
            report = evaluate_regional_predictions(archive["y_true"], archive["y_pred"], thresholds)
            display = get_model_display_name(model_id)
            metrics.append({"seed": run_seed, "result_key": model_id, "model_id": model_id, "model_name": display, **report["continuous"], "MAE_valid_rain": compute_valid_rain_mae(archive["y_true"], archive["y_pred"])})
            for item in report["events"].values():
                row = {"seed": run_seed, "result_key": model_id, "model_id": model_id, "model_name": display, **item}
                events.append(row)
                if np.isclose(item["threshold_mm"], max(thresholds)):
                    confusion.append(row)
            intensity.append({"seed": run_seed, "result_key": model_id, "model_id": model_id, "model_name": display, **report["intensity"]})
            rows_path = args.output_dir / "_temporary_long.csv"
            export_predictions_long(rows_path, archive["valid_times"], archive["grid_points_order"], archive["y_true"], archive["y_pred"], model_id, run_seed)
            long_rows.setdefault(model_id, []).append(pd.read_csv(rows_path))
            rows_path.unlink()
            for index, (latitude, longitude) in enumerate(archive["grid_points_order"]):
                pointwise.append({"seed": run_seed, "model_id": model_id, "model_name": display, "point_idx": index, "latitude": latitude, "longitude": longitude, **{key: float(values[index]) if np.isfinite(values[index]) else np.nan for key, values in report["gridwise"].items()}})
            available.add(model_id)
        if run_reference is None:
            continue
        run_references.append(run_reference)
        seeds.append(run_seed)
        true_values = np.asarray(run_reference[0]).reshape(-1)
        for threshold in thresholds:
            observed = true_values >= float(threshold)
            positive.append({"seed": run_seed, "threshold_mm": float(threshold), "positive_grid_hour_samples": int(observed.sum()), "total_grid_hour_samples": int(observed.size), "positive_ratio": float(observed.mean())})
    if not metrics:
        raise FileNotFoundError("No test_predictions.npz files were found in the supplied training runs.")
    reference = run_references[0]
    for index, candidate in enumerate(run_references[1:], start=2):
        _assert_identity(reference[:3] + (reference[3],), candidate[:3] + (reference[3],), f"run {index}")
    export_evaluation_tables(args.output_dir, metrics, events, intensity)
    pd.DataFrame(pointwise).to_csv(args.output_dir / "pointwise_metrics_by_seed.csv", index=False, float_format="%.10g")
    pd.DataFrame(positive).to_csv(args.output_dir / "event_positive_sample_counts.csv", index=False, float_format="%.10g")
    pd.DataFrame(confusion).to_csv(args.output_dir / "heavy_rain_confusion_counts.csv", index=False, float_format="%.10g")
    export_physinput_ablation_at10(args.output_dir / "physinput_ablation_at10.csv", events, max(thresholds))
    long_dir = args.output_dir / "predictions_long"
    long_dir.mkdir(exist_ok=True)
    for model_id, frames in long_rows.items():
        pd.concat(frames, ignore_index=True).to_csv(long_dir / f"{model_id}_predictions_long.csv", index=False, float_format="%.10g")
    table_paths = ["formal_metrics_by_seed.csv", "formal_metrics_mean_std.csv", "formal_event_metrics_by_seed.csv", "formal_event_metrics_mean_std.csv", "heavy_rain_confusion_counts.csv", "prediction_intensity_distribution.csv", "event_positive_sample_counts.csv", "pointwise_metrics_by_seed.csv", "physinput_ablation_at10.csv"]
    write_json(args.output_dir / "evaluation_manifest.json", {"input_training_runs": [str(path) for path in args.training_run_dir], "seeds": sorted(set(seeds)), "available_models": sorted(available), "thresholds": thresholds, "test_sample_count": int(np.asarray(reference[0]).shape[0]), "grid_cell_count": int(np.asarray(reference[0]).shape[1]), "output_tables": table_paths, "created_at_utc": datetime.now(timezone.utc).isoformat()})
    required_long = ["cnn_bilstm", "physinput_conv3", "physinput_all4"]
    write_json(args.output_dir / "figure_input_manifest.json", {"formal_event_metrics_by_seed": "formal_event_metrics_by_seed.csv", "long_prediction_tables": [f"predictions_long/{model}_predictions_long.csv" for model in required_long if model in long_rows], "seed_column_available": True, "seeds": sorted(set(seeds))})
    print(f"Evaluation completed: {args.output_dir}")


if __name__ == "__main__":
    main()
