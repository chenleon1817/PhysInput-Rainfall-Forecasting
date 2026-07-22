"""Formal deep-learning training and validation-event checkpoint handling."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .models import DEEP_MODEL_IDS, build_model
from .runtime import reset_training_state

LOGGER = logging.getLogger(__name__)


class ValidationEventScoreCheckpointMetric(tf.keras.callbacks.Callback):
    """Compute the configured validation heavy-rain checkpoint score."""

    def __init__(
        self,
        X_validation,
        y_validation,
        batch_size=32,
        threshold=10.0,
        event_score=None,
    ):
        super().__init__()
        self.X_validation = np.asarray(X_validation)
        self.y_validation = np.asarray(y_validation)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.event_score = event_score or {}
        self.best_score = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        prediction = self.model.predict(
            self.X_validation, batch_size=self.batch_size, verbose=0
        )
        true = self.y_validation.astype(np.float32).reshape(-1)
        predicted = np.clip(np.asarray(prediction, dtype=np.float32), 0, None).reshape(-1)
        observed = true >= self.threshold
        forecast = predicted >= self.threshold
        hits = int(np.sum(observed & forecast))
        false_alarms = int(np.sum(~observed & forecast))
        misses = int(np.sum(observed & ~forecast))
        pod = hits / max(hits + misses, 1)
        far = false_alarms / max(hits + false_alarms, 1)
        csi = hits / max(hits + false_alarms + misses, 1)
        fbias = int(forecast.sum()) / max(int(observed.sum()), 1)
        score = (
            float(self.event_score.get("csi_weight", 1.0)) * csi
            + float(self.event_score.get("pod_weight", 0.20)) * pod
            + float(self.event_score.get("far_weight", -0.15)) * far
            - float(self.event_score.get("fbias_absolute_error_weight", 0.10))
            * abs(fbias - 1.0)
        )
        logs.update(
            {
                "val_event_score": float(score),
                "val_csi_10": float(csi),
                "val_pod_10": float(pod),
                "val_far_10": float(far),
                "val_fbias_10": float(fbias),
            }
        )
        self.best_score = max(self.best_score, float(score))


class EpochProgressLogger(tf.keras.callbacks.Callback):
    """Log the formal epoch fields after metric callbacks have populated logs."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        fields = {"epoch": epoch + 1}
        for key in (
            "loss",
            "val_loss",
            "val_event_score",
            "val_csi_10",
            "val_pod_10",
            "val_far_10",
            "val_fbias_10",
            "learning_rate",
            "lr",
        ):
            if key in logs:
                fields[key] = float(logs[key])
        LOGGER.info("Training epoch: %s", fields)


def build_intensity_sample_weight(y_train, configuration):
    """Return four-class target-intensity sample weights and statistics."""
    settings = configuration["sample_weight"]
    targets = np.asarray(y_train, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != 45:
        raise ValueError("Sample weights require targets with shape (samples, 45).")
    maxima = targets.max(axis=1)
    rain = float(settings["rain_threshold"])
    moderate = float(settings["moderate_threshold"])
    heavy = float(settings["heavy_threshold"])
    weights_config = settings["weights"]
    masks = {
        "no_rain": maxima < rain,
        "light_rain": (maxima >= rain) & (maxima < moderate),
        "moderate_rain": (maxima >= moderate) & (maxima < heavy),
        "heavy_rain": maxima >= heavy,
    }
    sample_weight = np.ones(len(targets), dtype=np.float32)
    for label, mask in masks.items():
        sample_weight[mask] = float(weights_config[label])
    counts = {label: int(mask.sum()) for label, mask in masks.items()}
    weighted = {label: float(counts[label] * weights_config[label]) for label in counts}
    total = int(len(targets))
    effective = float(sum(weighted.values()))
    return sample_weight, {
        "enabled": True,
        "method": "y_max_four_class",
        "thresholds": {"rain_threshold": rain, "moderate_threshold": moderate, "heavy_threshold": heavy},
        "weights": {key: float(value) for key, value in weights_config.items()},
        "class_counts": counts,
        "class_percent": {key: counts[key] / max(total, 1) * 100 for key in counts},
        "weighted_counts": weighted,
        "effective_percent": {key: weighted[key] / max(effective, 1) * 100 for key in weighted},
        "sample_weight_min": float(sample_weight.min()),
        "sample_weight_max": float(sample_weight.max()),
        "sample_weight_mean": float(sample_weight.mean()),
        "sample_weight_std": float(sample_weight.std()),
        "total_samples": total,
        "total_effective_samples": effective,
    }


def build_training_callbacks(X_validation, y_validation, configuration, checkpoint_path):
    """Return callbacks in the formal dependency order."""
    checkpoint = configuration["checkpoint"]
    training = configuration["training"]
    event_callback = ValidationEventScoreCheckpointMetric(
        X_validation,
        y_validation,
        batch_size=training["batch_size"],
        threshold=checkpoint["event_threshold_mm_per_hour"],
        event_score=checkpoint["event_score"],
    )
    return [
        event_callback,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_event_score", mode="max",
            min_delta=float(checkpoint["early_stopping_min_delta"]),
            patience=int(training["early_stopping_patience"]),
            restore_best_weights=False,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=float(training["reduce_lr_factor"]),
            patience=int(training["reduce_lr_patience"]),
            min_lr=float(training["minimum_learning_rate"]),
        ),
        EpochProgressLogger(),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_event_score", mode="max",
            save_best_only=True, save_weights_only=True,
        ),
    ]


def train_regional_model(
    model_id, X_train, y_train, X_validation, y_validation, sample_weight_train,
    sample_weight_statistics, configuration, output_dir, model_feature_names,
    diagnostic_feature_names, physinput_config=None,
):
    """Train one deep model and reload its best validation-event checkpoint."""
    if model_id not in DEEP_MODEL_IDS:
        raise ValueError(f"Unsupported deep-learning model ID: {model_id}")
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    sample_weight_train = np.asarray(sample_weight_train, dtype=np.float32)
    if sample_weight_train.ndim != 1 or sample_weight_train.shape[0] != np.asarray(y_train).shape[0]:
        raise ValueError("sample_weight must be one-dimensional and aligned to y_train.")
    if not np.isfinite(sample_weight_train).all() or np.any(sample_weight_train <= 0):
        raise ValueError("sample_weight must contain finite positive values.")
    reset_training_state(configuration["training"]["seed"])
    model = build_model(model_id, np.asarray(X_train).shape[1:], configuration, physinput_config)
    checkpoint_path = checkpoints / "best_val_event.weights.h5"
    callbacks = build_training_callbacks(X_validation, y_validation, configuration, checkpoint_path)
    history = model.fit(
        np.asarray(X_train), np.asarray(y_train),
        sample_weight=sample_weight_train,
        validation_data=(np.asarray(X_validation), np.asarray(y_validation)),
        epochs=int(configuration["training"]["epochs"]),
        batch_size=int(configuration["training"]["batch_size"]),
        callbacks=callbacks, verbose=1, shuffle=False,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best validation-event checkpoint was not created: {checkpoint_path}")
    model.load_weights(checkpoint_path)
    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    best_epoch = int(np.argmax(history_dict["val_event_score"])) + 1
    train_info = {
        "model_id": model_id,
        "sample_weight_used": True,
        "sample_weight_statistics": sample_weight_statistics,
        "selected_checkpoint_type": "best_val_event",
        "best_epoch": best_epoch,
        "best_validation_event_score": float(max(history_dict["val_event_score"])),
        "epochs_completed": len(history_dict["loss"]),
        "model_feature_names": list(model_feature_names),
        "diagnostic_feature_names": list(diagnostic_feature_names),
    }
    pd.DataFrame(history_dict).to_csv(output_dir / "history.csv", index=False)
    (output_dir / "train_info.json").write_text(json.dumps(train_info, indent=2), encoding="utf-8")
    return {"model": model, "history": history_dict, "train_info": train_info, "checkpoint_paths": {"best_val_event": str(checkpoint_path)}, "best_epoch": best_epoch}


def write_training_manifest(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
