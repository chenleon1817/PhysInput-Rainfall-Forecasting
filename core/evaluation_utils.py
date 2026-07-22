"""Formal continuous and threshold-event evaluation utilities."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def normalize_thresholds(thresholds=None):
    if thresholds is None: values = [0.1, 2.5, 10.0]
    elif isinstance(thresholds, dict): values = [thresholds.get("rain_mm", thresholds.get("rain", 0.1)), thresholds.get("moderate_mm", thresholds.get("moderate", 2.5)), thresholds.get("heavy_mm", thresholds.get("heavy", 10.0))]
    else: values = list(thresholds)
    values = [float(value) for value in values]
    if len(values) != 3 or not (0 <= values[0] < values[1] < values[2]): raise ValueError("Thresholds must be exactly three ascending values.")
    return values

def _flatten(y_true, y_pred):
    true = np.asarray(y_true, dtype=float); pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape: raise ValueError(f"Prediction shape mismatch: {true.shape} versus {pred.shape}.")
    if not np.isfinite(true).all() or not np.isfinite(pred).all(): raise ValueError("Predictions and targets must be finite.")
    return true.reshape(-1), np.clip(pred.reshape(-1), 0.0, None)

def clip_nonnegative_predictions(predictions): return np.clip(np.asarray(predictions, dtype=float), 0.0, None)
def compute_continuous_metrics(y_true, y_pred):
    true, pred = _flatten(y_true, y_pred); corr = np.nan if true.size == 0 or np.std(true) == 0 or np.std(pred) == 0 else float(np.corrcoef(true, pred)[0, 1])
    return {"MAE": float(mean_absolute_error(true, pred)), "RMSE": float(np.sqrt(mean_squared_error(true, pred))), "R2": float(r2_score(true, pred)), "Bias": float(np.mean(pred - true)), "Correlation": corr}
def compute_event_confusion_counts(y_true, y_pred, threshold):
    true, pred = _flatten(y_true, y_pred); observed, forecast = true >= threshold, pred >= threshold
    return {"hits": int(np.sum(observed & forecast)), "misses": int(np.sum(observed & ~forecast)), "false_alarms": int(np.sum(~observed & forecast)), "correct_negatives": int(np.sum(~observed & ~forecast)), "observed_event_count": int(np.sum(observed)), "predicted_event_count": int(np.sum(forecast))}
def compute_event_metrics(y_true, y_pred, thresholds=None):
    result = {}
    for threshold in normalize_thresholds(thresholds):
        counts = compute_event_confusion_counts(y_true, y_pred, threshold); h, m, f = counts["hits"], counts["misses"], counts["false_alarms"]
        result[threshold] = {"threshold_mm": threshold, "POD": h / max(h + m, 1), "FAR": f / max(h + f, 1), "CSI": h / max(h + m + f, 1), "FBIAS": (h + f) / max(h + m, 1), **counts}
    return result
def safe_pearson_correlation(x, y):
    if len(x) == 0 or np.std(x) == 0 or np.std(y) == 0: return np.nan
    return float(np.corrcoef(x,y)[0,1])
def compute_valid_rain_mae(y_true,y_pred,rain_threshold=.1):
    true,pred=_flatten(y_true,y_pred); mask=true>=rain_threshold
    return float(np.mean(np.abs(pred[mask]-true[mask]))) if mask.any() else np.nan
def compute_gridwise_metrics(y_true, y_pred):
    true, pred = np.asarray(y_true), clip_nonnegative_predictions(y_pred)
    if true.shape != pred.shape or true.ndim != 2: raise ValueError("Gridwise evaluation requires matching (samples, cells) arrays.")
    return {"MAE": np.mean(np.abs(pred - true), axis=0), "RMSE": np.sqrt(np.mean((pred - true) ** 2, axis=0)), "Bias":np.mean(pred-true,axis=0), "Correlation":np.asarray([safe_pearson_correlation(true[:,i],pred[:,i]) for i in range(true.shape[1])])}
def summarize_prediction_intensity(y_true, y_pred, thresholds=None):
    """Summarize distribution and observed-heavy-cell prediction behavior."""
    true, pred = _flatten(y_true, y_pred)
    rain, moderate, heavy = normalize_thresholds(thresholds)
    masks = {
        "no_rain": lambda values: values < rain,
        "light_rain": lambda values: (values >= rain) & (values < moderate),
        "moderate_rain": lambda values: (values >= moderate) & (values < heavy),
        "heavy_rain": lambda values: values >= heavy,
    }
    result = {}
    for prefix, values in (("true", true), ("pred", pred)):
        for name, predicate in masks.items():
            count = int(np.sum(predicate(values)))
            result[f"{prefix}_{name}"] = count
            result[f"{prefix}_{name}_percent"] = 100.0 * count / max(values.size, 1)
    result.update({
        "y_true_max": float(np.max(true)), "y_pred_max": float(np.max(pred)),
        "y_true_p99": float(np.percentile(true, 99)), "y_pred_p99": float(np.percentile(pred, 99)),
    })
    observed_heavy_predictions = pred[true >= heavy]
    result["pred_at_observed_heavy_count"] = int(observed_heavy_predictions.size)
    for name, function in (("mean", np.mean), ("median", np.median), ("p75", lambda x: np.percentile(x, 75)), ("p90", lambda x: np.percentile(x, 90)), ("max", np.max)):
        result[f"pred_at_observed_heavy_{name}"] = float(function(observed_heavy_predictions)) if observed_heavy_predictions.size else np.nan
    result["pred_at_observed_heavy_count_5_to_8"] = int(np.sum((observed_heavy_predictions >= 5) & (observed_heavy_predictions < 8)))
    result["pred_at_observed_heavy_count_8_to_10"] = int(np.sum((observed_heavy_predictions >= 8) & (observed_heavy_predictions < heavy)))
    result["pred_at_observed_heavy_count_ge_heavy"] = int(np.sum(observed_heavy_predictions >= heavy))
    return result
def evaluate_regional_predictions(y_true, y_pred, thresholds=None): return {"continuous": compute_continuous_metrics(y_true, y_pred), "events": compute_event_metrics(y_true, y_pred, thresholds), "gridwise": compute_gridwise_metrics(y_true, y_pred), "intensity": summarize_prediction_intensity(y_true, y_pred, thresholds)}
