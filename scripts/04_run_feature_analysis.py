#!/usr/bin/env python3
"""Run the formal feature-analysis workflow for regional rainfall forecasting.

This script performs the feature-analysis stage only. It does not construct
the downstream PhysInput diagnostic combinations used by model training.

The workflow includes:
1. Loading the annual model-input CSV files.
2. Renaming the legacy Vxxxxx variables.
3. Adding day-of-year, hour-of-day, and wind-representation variables.
4. Applying the formal chronological train/validation/test split.
5. Comparing U/V with wind-speed and wind-direction representations.
6. Generating the formal double-column correlation figure.
7. Generating the formal double-column PCA-loading heatmap.
8. Computing multi-seed permutation importance and its formal figure.
9. Selecting and checking physical-context candidate variables.
10. Generating one combined rainfall-response figure and six individual
    rainfall-response figures.
11. Exporting the feature-selection configuration and all supporting tables.

Only PDF figures required by the current PAA manuscript are generated.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Formal analysis constants
# ---------------------------------------------------------------------------

RAIN_OCCURRENCE_THRESHOLD = 0.1
HEAVY_RAIN_THRESHOLD = 2.5
EXTREME_RAIN_THRESHOLD = 10.0

DEFAULT_RENAME_MAP: Dict[str, str] = {
    "V11001": "T",
    "V11002": "U",
    "V11003": "V",
    "V07004": "P",
    "V13010": "RH",
    "V13011": "TP",
    "V13012": "PWV",
    "V13013": "dPWV+",
    "V13014": "dPWV-",
}

DISPLAY_NAME_MAP: Dict[str, str] = {
    "T": "T",
    "U": "U",
    "V": "V",
    "P": "P",
    "RH": "RH",
    "PWV": "PWV",
    "dPWV+": r"$\mathrm{dPWV}^{+}$",
    "dPWV-": r"$\mathrm{dPWV}^{-}$",
    "WS": "WS",
    "theta_sin": "WDs",
    "theta_cos": "WDc",
    "DOY": "DOY",
    "HOD": "HOD",
    "TP": "RF",
}

PHYSICAL_BASE_FEATURES = ["T", "P", "RH", "PWV"]
RAW_WIND_FEATURES = ["U", "V"]
INCREMENTAL_FEATURES = ["dPWV+", "dPWV-"]
REPARAMETERIZED_WIND_FEATURES = ["WS", "theta_sin", "theta_cos"]
TIME_FEATURES = ["DOY", "HOD"]
SHARED_DERIVED_PREDICTORS = ["lagged_rainfall"]

PHYSICAL_PROCESS_ROLES = {
    "PWV": "moisture availability",
    "RH": "moisture condition",
    "dPWV+": "moisture accumulation tendency",
    "dPWV-": "moisture depletion tendency",
    "T": "thermodynamic favorability",
    "P": "near-surface pressure state",
    "U": "dynamic modulation",
    "V": "dynamic modulation",
    "WS": "wind-speed modulation",
    "theta_sin": "wind-direction representation",
    "theta_cos": "wind-direction representation",
}

FEATURE_ANALYSIS_CONFIG: Dict[str, Any] = {
    "pca_max_rows": None,
    "wind_bootstrap_B": 100,
    "wind_bootstrap_n": 10_000,
    "wind_beta": 0.10,
    "wind_var_threshold": 0.95,
    "importance_train_max_rows": 80_000,
    "importance_eval_max_rows": 12_000,
    "importance_seeds": [42, 2024, 3407],
    "importance_n_repeats": 8,
    "importance_n_estimators": 300,
    "importance_max_depth": 8,
    "importance_min_samples_leaf": 5,
    "importance_n_jobs": -1,
    "physical_context_top_k": 6,
    "redundancy_corr_threshold": 0.85,
    "response_n_bins": 12,
    "response_min_count_per_bin": 30,
    "random_state": 42,
}

DOUBLE_COLUMN_PROFILE: Dict[str, float] = {
    "width": 7.15,
    "base_fs": 8.8,
    "title_fs": 9.2,
    "label_fs": 8.8,
    "tick_fs": 8.0,
    "legend_fs": 7.8,
    "annot_fs": 7.6,
    "cbar_fs": 7.8,
    "panel_fs": 9.2,
}

RESPONSE_FEATURE_ORDER = ["PWV", "T", "RH", "P", "U", "V"]

RESPONSE_AXIS_LABELS = {
    "PWV": r"PWV (mm)",
    "T": r"Temperature ($^\circ$C)",
    "P": r"Pressure (hPa)",
    "U": r"Zonal wind $U$ (m s$^{-1}$)",
    "V": r"Meridional wind $V$ (m s$^{-1}$)",
    "RH": r"Relative humidity (%)",
    "dPWV+": r"$\mathrm{dPWV}^{+}$ (mm h$^{-1}$)",
    "dPWV-": r"$\mathrm{dPWV}^{-}$ (mm h$^{-1}$)",
}

RESPONSE_TITLES = {
    "PWV": "PWV",
    "T": "Temperature",
    "P": "Pressure",
    "U": "Zonal wind",
    "V": "Meridional wind",
    "RH": "Relative humidity",
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def ensure_directory(path: Path) -> None:
    """Create a directory and all missing parent directories."""
    path.mkdir(parents=True, exist_ok=True)


def merge_unique_features(*feature_lists: Sequence[str]) -> List[str]:
    """Merge feature lists while preserving order and removing duplicates."""
    return list(
        dict.fromkeys(
            feature
            for feature_list in feature_lists
            for feature in feature_list
        )
    )


def generated_at() -> str:
    """Return the current local timestamp in ISO-8601 format."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def to_jsonable(value: Any) -> Any:
    """Convert NumPy, pandas, and path objects to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return None if not math.isfinite(numeric) else numeric
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def safe_feature_token(feature: str) -> str:
    """Convert a feature name to a filename-safe token."""
    return (
        feature.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "minus")
    )


def relative_path(path: Path, root: Path) -> str:
    """Return a portable path relative to the output root."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a formatted UTF-8 JSON file."""
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_pdf_figure(
    figure,
    output_path: Path,
    *,
    dpi: int = 600,
    pad_inches: float = 0.035,
) -> Path:
    """Save a publication-ready PDF figure and close it."""
    ensure_directory(output_path.parent)
    figure.savefig(
        output_path,
        format="pdf",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
        transparent=False,
    )
    plt.close(figure)
    return output_path


def configure_plotting() -> None:
    """Configure the common double-column PAA figure style."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "font.weight": "normal",
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": DOUBLE_COLUMN_PROFILE["base_fs"],
            "axes.titlesize": DOUBLE_COLUMN_PROFILE["title_fs"],
            "axes.titleweight": "normal",
            "axes.labelsize": DOUBLE_COLUMN_PROFILE["label_fs"],
            "axes.labelweight": "normal",
            "axes.linewidth": 0.75,
            "legend.fontsize": DOUBLE_COLUMN_PROFILE["legend_fs"],
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "xtick.labelsize": DOUBLE_COLUMN_PROFILE["tick_fs"],
            "ytick.labelsize": DOUBLE_COLUMN_PROFILE["tick_fs"],
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.40,
            "grid.alpha": 0.18,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.035,
        }
    )


def style_colorbar(colorbar, label: str) -> None:
    """Apply the common figure style to a colorbar."""
    if colorbar is None:
        return
    colorbar.set_label(
        label,
        fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
        labelpad=5,
    )
    colorbar.ax.tick_params(
        labelsize=DOUBLE_COLUMN_PROFILE["cbar_fs"],
        width=0.6,
        length=3,
    )


def style_axis(ax, *, grid_axis: str = "both") -> None:
    """Apply the common axis style used by line and bar figures."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=3.6,
        width=0.7,
        labelsize=DOUBLE_COLUMN_PROFILE["tick_fs"],
        pad=4,
    )
    ax.grid(
        True,
        axis=grid_axis,
        linestyle="--",
        linewidth=0.42,
        alpha=0.18,
    )


def set_clean_limits(
    ax,
    x_values: np.ndarray,
    y_series: Sequence[np.ndarray],
    *,
    top_padding: float = 0.20,
    bottom_padding: float = 0.02,
) -> None:
    """Set compact plot limits while retaining small visual margins."""
    x = np.asarray(x_values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size:
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        x_span = max(x_max - x_min, 1e-8)
        ax.set_xlim(x_min - 0.03 * x_span, x_max + 0.05 * x_span)

    valid_y = []
    for values in y_series:
        current = np.asarray(values, dtype=float)
        current = current[np.isfinite(current)]
        if current.size:
            valid_y.append(current)

    if not valid_y:
        return

    y_all = np.concatenate(valid_y)
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    y_span = max(y_max - y_min, 1e-8)

    if y_min >= 0:
        lower = min(-0.04 * max(y_max, 1e-8), y_min - bottom_padding * y_span)
    else:
        lower = y_min - bottom_padding * y_span

    upper = y_max + top_padding * y_span
    ax.set_ylim(lower, upper)


# ---------------------------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------------------------

def resolve_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated columns while retaining the first occurrence."""
    if frame.columns.duplicated().any():
        return frame.loc[:, ~frame.columns.duplicated()].copy()
    return frame


def rename_variables(
    frame: pd.DataFrame,
    rename_map: Dict[str, str] = DEFAULT_RENAME_MAP,
) -> pd.DataFrame:
    """Rename the supported legacy model-input variables."""
    available_mapping = {
        source: target
        for source, target in rename_map.items()
        if source in frame.columns
    }
    renamed = frame.rename(columns=available_mapping)
    return resolve_duplicate_columns(renamed)


def build_datetime_from_columns(frame: pd.DataFrame) -> pd.Series:
    """Build timestamps from supported legacy or semantic time columns."""
    if all(
        column in frame.columns
        for column in ["V04001", "V04002", "V04003", "V04004"]
    ):
        return pd.to_datetime(
            {
                "year": frame["V04001"].astype(int),
                "month": frame["V04002"].astype(int),
                "day": frame["V04003"].astype(int),
                "hour": frame["V04004"].astype(int),
            },
            errors="coerce",
        )

    if all(
        column in frame.columns
        for column in ["year", "month", "day", "hour"]
    ):
        return pd.to_datetime(
            {
                "year": frame["year"].astype(int),
                "month": frame["month"].astype(int),
                "day": frame["day"].astype(int),
                "hour": frame["hour"].astype(int),
            },
            errors="coerce",
        )

    if "valid_time" in frame.columns:
        return pd.to_datetime(frame["valid_time"], errors="coerce")

    raise KeyError(
        "No supported time columns were found. Expected "
        "V04001/V04002/V04003/V04004, year/month/day/hour, "
        "or valid_time."
    )


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-year and hour-of-day features."""
    output = frame.copy()
    timestamp = build_datetime_from_columns(output)
    output["DOY"] = timestamp.dt.dayofyear
    output["HOD"] = timestamp.dt.hour
    output["_valid_time"] = timestamp
    return output


def prepare_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add wind speed and trigonometric wind-direction representations."""
    output = frame.copy()
    if "U" in output.columns and "V" in output.columns:
        u = output["U"].astype(float)
        v = output["V"].astype(float)
        output["WS"] = np.sqrt(u**2 + v**2)
        direction = np.arctan2(v, u)
        output["theta_sin"] = np.sin(direction)
        output["theta_cos"] = np.cos(direction)
    return output


def get_valid_feature_columns(
    frame: pd.DataFrame,
    feature_list: Sequence[str],
) -> List[str]:
    """Keep columns that exist and contain at least one non-missing value."""
    return [
        feature
        for feature in feature_list
        if feature in frame.columns and frame[feature].notna().any()
    ]


def load_regional_data(data_dir: Path, years: Sequence[int]) -> pd.DataFrame:
    """Load the explicitly requested annual CSV files without silent skipping."""
    frames: List[pd.DataFrame] = []
    expected_columns: Optional[List[str]] = None

    for year in years:
        file_path = data_dir / f"data_{year}.csv"
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required annual data file was not found: {file_path}"
            )

        logging.info("Loading %s", file_path)
        current = pd.read_csv(file_path)

        if current.empty:
            raise ValueError(f"The annual data file is empty: {file_path}")

        if expected_columns is None:
            expected_columns = list(current.columns)
        elif list(current.columns) != expected_columns:
            raise ValueError(
                f"Column mismatch in {file_path}. All annual files must use "
                "the same columns and column order."
            )

        frames.append(current)

    data = pd.concat(frames, ignore_index=True)
    logging.info("Loaded %d rows from %d annual files.", len(data), len(frames))
    return data


def build_time_split(
    frame: pd.DataFrame,
    *,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
) -> tuple[pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Build the formal chronological data split."""
    if "_valid_time" not in frame.columns:
        raise KeyError("_valid_time is missing. Run add_time_features first.")

    timestamp = pd.to_datetime(frame["_valid_time"], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(
            f"{int(timestamp.isna().sum())} rows contain invalid timestamps."
        )

    train_end_ts = pd.Timestamp(train_end)
    validation_start_ts = pd.Timestamp(validation_start)
    validation_end_ts = pd.Timestamp(validation_end)
    test_start_ts = pd.Timestamp(test_start)

    train_mask = timestamp <= train_end_ts
    validation_mask = (
        (timestamp >= validation_start_ts)
        & (timestamp <= validation_end_ts)
    )
    test_mask = timestamp >= test_start_ts

    split_info = {
        "split_method": "chronological",
        "train_end": str(train_end_ts),
        "validation_start": str(validation_start_ts),
        "validation_end": str(validation_end_ts),
        "test_start": str(test_start_ts),
        "data_time_start": str(timestamp.min()),
        "data_time_end": str(timestamp.max()),
    }
    return train_mask, validation_mask, test_mask, split_info


# ---------------------------------------------------------------------------
# Wind-representation screening without figure generation
# ---------------------------------------------------------------------------

def effective_rank_from_correlation(
    matrix: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """Calculate the entropy-based effective rank of a correlation matrix."""
    correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.nan_to_num(
        correlation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    correlation = 0.5 * (correlation + correlation.T)

    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())

    if total <= epsilon:
        return 0.0

    probabilities = np.clip(eigenvalues / total, epsilon, 1.0)
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return float(np.exp(entropy))


def partial_corr_residual(
    response: np.ndarray,
    controls: np.ndarray,
) -> np.ndarray:
    """Residualize one vector against the supplied control variables."""
    if controls is None or controls.size == 0:
        return response

    model = LinearRegression()
    model.fit(controls, response)
    return response - model.predict(controls)


def mean_absolute_correlation(
    response: np.ndarray,
    predictors: np.ndarray,
    *,
    method: str = "spearman",
) -> float:
    """Return the mean absolute correlation across predictor columns."""
    values = []

    for column_index in range(predictors.shape[1]):
        predictor = predictors[:, column_index]

        if method == "spearman":
            correlation, _ = spearmanr(predictor, response)
        else:
            correlation = np.corrcoef(predictor, response)[0, 1]

        if np.isnan(correlation):
            correlation = 0.0

        values.append(abs(float(correlation)))

    return float(np.mean(values)) if values else 0.0


def evaluate_wind_representation(
    train_data: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    target: str = "TP",
    controls: Optional[Sequence[str]] = None,
    corr_method: str = "spearman",
    beta: float = 0.10,
    variance_threshold: float = 0.95,
) -> Dict[str, Any]:
    """Evaluate one wind representation using PCA and partial correlation."""
    controls = list(controls or [])
    wind_features = [
        column for column in group_columns if column in train_data.columns
    ]

    if target not in train_data.columns or not wind_features:
        return {
            "skipped": True,
            "reason": "Missing target or wind-representation columns.",
            "columns": wind_features,
        }

    columns = wind_features + controls + [target]
    work = (
        train_data[columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(work) < 3_000:
        return {
            "skipped": True,
            "reason": f"Not enough clean rows: {len(work)}",
            "columns": wind_features,
        }

    predictors = work[wind_features].astype(float).to_numpy()
    response = work[target].astype(float).to_numpy()
    control_values = (
        work[controls].astype(float).to_numpy()
        if controls
        else np.empty((len(work), 0))
    )

    standardized = StandardScaler().fit_transform(predictors)
    pca = PCA(n_components=None, random_state=42)
    principal_components = pca.fit_transform(standardized)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    dimensions_95 = int(
        np.searchsorted(cumulative, variance_threshold) + 1
    )
    dimensions_95 = max(
        1,
        min(dimensions_95, principal_components.shape[1]),
    )

    effective_rank = effective_rank_from_correlation(standardized)
    n_features = len(wind_features)
    complexity = (
        float((n_features - effective_rank) / (n_features - 1))
        if n_features > 1
        else 0.0
    )

    response_residual = partial_corr_residual(response, control_values)
    selected_components = principal_components[:, :dimensions_95]
    component_residuals = np.zeros_like(selected_components)

    for index in range(selected_components.shape[1]):
        component_residuals[:, index] = partial_corr_residual(
            selected_components[:, index],
            control_values,
        )

    partial_correlation_score = mean_absolute_correlation(
        response_residual,
        component_residuals,
        method=corr_method,
    )
    representation_score = float(
        partial_correlation_score - beta * complexity
    )

    return {
        "skipped": False,
        "columns": wind_features,
        "n": int(len(work)),
        "corr_method": corr_method,
        "variance_threshold": float(variance_threshold),
        "beta": float(beta),
        "dimensions_95": int(dimensions_95),
        "effective_rank": float(effective_rank),
        "pc1_explained_ratio": float(explained[0]),
        "complexity": float(complexity),
        "pc_target_partialcorr_score": float(partial_correlation_score),
        "representation_score": float(representation_score),
    }


def choose_wind_representation(
    train_data: pd.DataFrame,
    *,
    raw_columns: Optional[Sequence[str]] = None,
    reparameterized_columns: Optional[Sequence[str]] = None,
    target: str = "TP",
    controls: Optional[Sequence[str]] = None,
    corr_method: str = "spearman",
    beta: float = 0.10,
    variance_threshold: float = 0.95,
    bootstrap_iterations: int = 100,
    bootstrap_rows: int = 10_000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Select U/V or WS/direction using the established stability rule."""
    raw_columns = list(raw_columns or ["U", "V"])
    reparameterized_columns = list(
        reparameterized_columns
        or ["WS", "theta_sin", "theta_cos"]
    )
    controls = list(controls or [])

    raw_evaluation = evaluate_wind_representation(
        train_data,
        raw_columns,
        target=target,
        controls=controls,
        corr_method=corr_method,
        beta=beta,
        variance_threshold=variance_threshold,
    )
    reparameterized_evaluation = evaluate_wind_representation(
        train_data,
        reparameterized_columns,
        target=target,
        controls=controls,
        corr_method=corr_method,
        beta=beta,
        variance_threshold=variance_threshold,
    )

    required_columns = list(
        dict.fromkeys(
            [
                column
                for column in (
                    raw_columns
                    + reparameterized_columns
                    + controls
                    + [target]
                )
                if column in train_data.columns
            ]
        )
    )
    work = (
        train_data[required_columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(work) < 3_000:
        return {
            "skipped": True,
            "reason": f"Not enough clean rows: {len(work)}",
        }

    rng = np.random.RandomState(random_state)
    score_differences: List[float] = []
    raw_wins = 0

    for _ in range(int(bootstrap_iterations)):
        sample_size = min(int(bootstrap_rows), len(work))
        indices = rng.choice(
            len(work),
            size=sample_size,
            replace=True,
        )
        bootstrap_sample = work.iloc[indices].copy()

        raw_result = evaluate_wind_representation(
            bootstrap_sample,
            raw_columns,
            target=target,
            controls=controls,
            corr_method=corr_method,
            beta=beta,
            variance_threshold=variance_threshold,
        )
        reparameterized_result = evaluate_wind_representation(
            bootstrap_sample,
            reparameterized_columns,
            target=target,
            controls=controls,
            corr_method=corr_method,
            beta=beta,
            variance_threshold=variance_threshold,
        )

        if (
            raw_result.get("skipped", False)
            or reparameterized_result.get("skipped", False)
        ):
            continue

        difference = float(
            raw_result["representation_score"]
            - reparameterized_result["representation_score"]
        )
        score_differences.append(difference)

        if difference >= 0:
            raw_wins += 1

    win_rate_raw = float(
        raw_wins / max(1, len(score_differences))
    )
    difference_mean = (
        float(np.mean(score_differences))
        if score_differences
        else 0.0
    )
    difference_low = (
        float(np.quantile(score_differences, 0.025))
        if score_differences
        else 0.0
    )
    difference_high = (
        float(np.quantile(score_differences, 0.975))
        if score_differences
        else 0.0
    )

    epsilon = 0.002
    sufficient_bootstrap = len(score_differences) >= max(
        30,
        bootstrap_iterations // 3,
    )
    confidence_interval_crosses_zero = (
        difference_low < 0.0 < difference_high
    )

    if sufficient_bootstrap:
        if (
            confidence_interval_crosses_zero
            or abs(difference_mean) < epsilon
        ):
            prefer_raw = True
            decision_reason = (
                "no_significant_difference_prefer_uv"
            )
        elif difference_mean < -epsilon and win_rate_raw <= 0.40:
            prefer_raw = False
            decision_reason = (
                "reparameterized_wind_significantly_better"
            )
        else:
            prefer_raw = True
            decision_reason = (
                "uv_preferred_by_stability_rule"
            )
    else:
        raw_partial = float(
            raw_evaluation.get(
                "pc_target_partialcorr_score",
                -1e9,
            )
        )
        reparameterized_partial = float(
            reparameterized_evaluation.get(
                "pc_target_partialcorr_score",
                -1e9,
            )
        )
        prefer_raw = raw_partial >= reparameterized_partial
        decision_reason = (
            "fallback_partialcorr_prefer_uv"
            if prefer_raw
            else "fallback_partialcorr_prefer_reparameterized_wind"
        )

    selected = [
        column
        for column in (
            raw_columns if prefer_raw else reparameterized_columns
        )
        if column in train_data.columns
    ]
    discarded = [
        column
        for column in (
            reparameterized_columns if prefer_raw else raw_columns
        )
        if column in train_data.columns
    ]

    return {
        "skipped": False,
        "keep": selected,
        "drop": discarded,
        "uv_evaluation": raw_evaluation,
        "reparameterized_wind_evaluation": reparameterized_evaluation,
        "bootstrap": {
            "iterations_requested": int(bootstrap_iterations),
            "iterations_effective": int(len(score_differences)),
            "rows_per_bootstrap": int(min(bootstrap_rows, len(work))),
            "uv_win_rate": float(win_rate_raw),
            "difference_mean": float(difference_mean),
            "difference_ci95": [
                float(difference_low),
                float(difference_high),
            ],
            "score_differences": [
                float(value) for value in score_differences
            ],
        },
        "score_definition": (
            "difference = representation_score(UV) - "
            "representation_score(WS_direction)"
        ),
        "decision_reason": decision_reason,
        "decision_statistics": {
            "epsilon": float(epsilon),
            "confidence_interval_crosses_zero": bool(
                confidence_interval_crosses_zero
            ),
            "difference_mean": float(difference_mean),
            "difference_low": float(difference_low),
            "difference_high": float(difference_high),
            "uv_win_rate": float(win_rate_raw),
        },
        "target": target,
        "controls": controls,
        "corr_method": corr_method,
        "beta": float(beta),
        "variance_threshold": float(variance_threshold),
    }


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def plot_feature_correlation(
    train_data: pd.DataFrame,
    features: Sequence[str],
    figures_dir: Path,
    tables_dir: Path,
    output_root: Path,
) -> Dict[str, Any]:
    """Generate the formal double-column Pearson-correlation figure."""
    columns = [
        feature for feature in features if feature in train_data.columns
    ]
    if "TP" in train_data.columns and "TP" not in columns:
        columns.append("TP")

    if len(columns) < 3:
        return {
            "skipped": True,
            "reason": f"Not enough columns: {columns}",
        }

    work = (
        train_data[columns]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(work) < 2_000:
        return {
            "skipped": True,
            "reason": f"Not enough clean rows: {len(work)}",
        }

    correlation = work.corr(method="pearson")
    correlation_path = tables_dir / "correlation_key_predictors.csv"
    correlation.to_csv(
        correlation_path,
        encoding="utf-8-sig",
    )

    display_columns = [
        DISPLAY_NAME_MAP.get(column, column)
        for column in columns
    ]
    display_correlation = correlation.copy()
    display_correlation.index = display_columns
    display_correlation.columns = display_columns

    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN_PROFILE["width"], 4.85)
    )

    if HAS_SEABORN:
        sns.heatmap(
            display_correlation,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            linewidths=0.45,
            linecolor="white",
            cbar_kws={"shrink": 0.84, "pad": 0.035},
            annot_kws={
                "size": DOUBLE_COLUMN_PROFILE["annot_fs"],
                "weight": "normal",
            },
            ax=axis,
        )

        for text_object in axis.texts:
            x_position, y_position = text_object.get_position()
            column_index = int(round(x_position - 0.5))
            row_index = int(round(y_position - 0.5))
            if (
                0 <= row_index < display_correlation.shape[0]
                and 0 <= column_index < display_correlation.shape[1]
            ):
                value = display_correlation.iloc[
                    row_index,
                    column_index,
                ]
                text_object.set_color(
                    "white" if abs(value) >= 0.55 else "black"
                )
    else:
        image = axis.imshow(
            display_correlation.values,
            vmin=-1.0,
            vmax=1.0,
            cmap="RdBu_r",
        )
        axis.set_xticks(range(len(display_columns)))
        axis.set_yticks(range(len(display_columns)))
        axis.set_xticklabels(display_columns)
        axis.set_yticklabels(display_columns)
        figure.colorbar(
            image,
            ax=axis,
            shrink=0.84,
            pad=0.035,
        )

        for row_index in range(display_correlation.shape[0]):
            for column_index in range(display_correlation.shape[1]):
                value = display_correlation.iloc[
                    row_index,
                    column_index,
                ]
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=DOUBLE_COLUMN_PROFILE["annot_fs"],
                    color=(
                        "white" if abs(value) >= 0.55 else "black"
                    ),
                )

    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(
        axis="both",
        labelsize=DOUBLE_COLUMN_PROFILE["tick_fs"],
        length=0,
    )
    axis.set_xticklabels(
        axis.get_xticklabels(),
        rotation=0,
        ha="center",
    )
    axis.set_yticklabels(
        axis.get_yticklabels(),
        rotation=0,
    )

    colorbar = None
    if getattr(axis, "collections", None):
        try:
            colorbar = axis.collections[0].colorbar
        except Exception:
            colorbar = None

    style_colorbar(
        colorbar,
        r"Pearson correlation coefficient, $r$",
    )

    figure.subplots_adjust(
        left=0.16,
        right=0.88,
        bottom=0.12,
        top=0.97,
    )

    figure_path = figures_dir / "figure_feature_correlation.pdf"
    save_pdf_figure(figure, figure_path)

    return {
        "skipped": False,
        "figure": relative_path(figure_path, output_root),
        "table": relative_path(correlation_path, output_root),
        "n": int(len(work)),
        "features": columns,
    }


# ---------------------------------------------------------------------------
# PCA analysis
# ---------------------------------------------------------------------------

def assign_features_to_dominant_components(
    pca: PCA,
    features: Sequence[str],
    *,
    n_components: int,
) -> pd.DataFrame:
    """Assign each feature to the retained PC with maximum absolute loading."""
    retained_components = min(
        n_components,
        pca.components_.shape[0],
    )
    component_matrix = pca.components_[:retained_components, :]

    records = []
    for feature_index, feature in enumerate(features):
        absolute_loadings = np.abs(
            component_matrix[:, feature_index]
        )
        best_index = int(np.argmax(absolute_loadings))
        signed_loading = float(
            component_matrix[best_index, feature_index]
        )
        records.append(
            {
                "feature": feature,
                "display_name": DISPLAY_NAME_MAP.get(
                    feature,
                    feature,
                ),
                "assigned_component": f"PC{best_index + 1}",
                "loading": signed_loading,
                "absolute_loading": abs(signed_loading),
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(
            ["assigned_component", "absolute_loading"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )


def run_pca_analysis(
    train_data: pd.DataFrame,
    features: Sequence[str],
    figures_dir: Path,
    tables_dir: Path,
    output_root: Path,
    *,
    max_rows: Optional[int],
    random_state: int,
    n_display_components: int = 6,
) -> Dict[str, Any]:
    """Fit PCA, export loadings, and generate the formal loading heatmap."""
    pca_features = [
        feature for feature in features if feature in train_data.columns
    ]

    if len(pca_features) < 4:
        return {
            "skipped": True,
            "reason": f"Not enough PCA features: {pca_features}",
        }

    work = (
        train_data[pca_features]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(work) < 5_000:
        return {
            "skipped": True,
            "reason": f"Not enough clean rows: {len(work)}",
        }

    if max_rows is not None and len(work) > max_rows:
        work = work.sample(
            n=max_rows,
            random_state=random_state,
        )

    standardized = StandardScaler().fit_transform(
        work[pca_features].astype(float).to_numpy()
    )

    pca = PCA(n_components=None, random_state=random_state)
    pca.fit(standardized)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_display = min(
        n_display_components,
        pca.components_.shape[0],
    )

    full_loadings = pd.DataFrame(
        pca.components_.T,
        index=pca_features,
        columns=[
            f"PC{index + 1}"
            for index in range(pca.components_.shape[0])
        ],
    )
    loadings_path = tables_dir / "pca_loadings.csv"
    full_loadings.to_csv(
        loadings_path,
        encoding="utf-8-sig",
    )

    assignment = assign_features_to_dominant_components(
        pca,
        pca_features,
        n_components=n_display,
    )
    assignment_path = tables_dir / "pca_mode_assignment.csv"
    assignment.to_csv(
        assignment_path,
        index=False,
        encoding="utf-8-sig",
    )

    displayed_components = pca.components_[:n_display, :]
    component_labels = [
        f"PC{index + 1}\n({explained[index] * 100:.1f}%)"
        for index in range(n_display)
    ]
    display_features = [
        DISPLAY_NAME_MAP.get(feature, feature)
        for feature in pca_features
    ]

    heatmap_data = pd.DataFrame(
        displayed_components.T,
        index=display_features,
        columns=component_labels,
    )
    row_order = (
        heatmap_data.abs()
        .max(axis=1)
        .sort_values(ascending=False)
        .index
    )
    heatmap_data = heatmap_data.loc[row_order]

    heatmap_table_path = tables_dir / "pca_loading_heatmap.csv"
    heatmap_data.to_csv(
        heatmap_table_path,
        encoding="utf-8-sig",
    )

    maximum_loading = (
        float(np.nanmax(np.abs(heatmap_data.to_numpy())))
        if heatmap_data.size
        else 0.75
    )
    color_limit = max(
        0.20,
        np.ceil(maximum_loading * 10.0) / 10.0,
    )

    height = max(
        3.75,
        0.285 * len(heatmap_data.index) + 1.25,
    )
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN_PROFILE["width"], height)
    )

    if HAS_SEABORN:
        sns.heatmap(
            heatmap_data,
            cmap="RdBu_r",
            center=0.0,
            vmin=-color_limit,
            vmax=color_limit,
            linewidths=0.8,
            linecolor="#D9D9D9",
            cbar_kws={"shrink": 0.86, "pad": 0.035},
            ax=axis,
        )
    else:
        image = axis.imshow(
            heatmap_data.to_numpy(),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        axis.set_xticks(np.arange(heatmap_data.shape[1]))
        axis.set_yticks(np.arange(heatmap_data.shape[0]))
        axis.set_xticklabels(heatmap_data.columns)
        axis.set_yticklabels(heatmap_data.index)
        figure.colorbar(
            image,
            ax=axis,
            fraction=0.046,
            pad=0.04,
        )

    axis.set_xlabel(
        "Principal component",
        fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
    )
    axis.set_ylabel(
        "Predictor",
        fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
    )
    axis.tick_params(
        axis="x",
        rotation=0,
        labelsize=DOUBLE_COLUMN_PROFILE["tick_fs"],
        length=0,
    )
    axis.tick_params(
        axis="y",
        rotation=0,
        labelsize=DOUBLE_COLUMN_PROFILE["tick_fs"],
        length=0,
    )

    colorbar = None
    if getattr(axis, "collections", None):
        try:
            colorbar = axis.collections[0].colorbar
        except Exception:
            colorbar = None
    style_colorbar(colorbar, "PCA loading")

    figure.tight_layout(pad=0.45)
    figure_path = figures_dir / "figure_pca_loadings.pdf"
    save_pdf_figure(figure, figure_path)

    selected_all = assignment["feature"].tolist()
    selected_by_component = {
        component: group["feature"].tolist()
        for component, group in assignment.groupby(
            "assigned_component",
            sort=True,
        )
    }

    return {
        "skipped": False,
        "figure": relative_path(figure_path, output_root),
        "loadings_table": relative_path(
            loadings_path,
            output_root,
        ),
        "heatmap_table": relative_path(
            heatmap_table_path,
            output_root,
        ),
        "assignment_table": relative_path(
            assignment_path,
            output_root,
        ),
        "n": int(len(work)),
        "features": pca_features,
        "explained_variance_ratio": explained.tolist(),
        "cumulative_variance_ratio": cumulative.tolist(),
        "displayed_components": int(n_display),
        "selected_by_component": selected_by_component,
        "selected_all": selected_all,
        "analysis_sample_info": {
            "rows_used": int(len(work)),
            "max_rows": (
                None if max_rows is None else int(max_rows)
            ),
            "random_state": int(random_state),
        },
    }


# ---------------------------------------------------------------------------
# Shared-input selection
# ---------------------------------------------------------------------------

def finalize_shared_input_features(
    train_data: pd.DataFrame,
    *,
    base_features: Sequence[str],
    incremental_features: Sequence[str],
    wind_features: Sequence[str],
    time_features: Sequence[str],
    pca_result: Dict[str, Any],
    importance_result: Dict[str, Any],
    minimum_positive_importance: float = 0.0,
) -> Dict[str, Any]:
    """Build the shared meteorological input list from the analysis results."""
    pca_selected = set()
    if (
        isinstance(pca_result, dict)
        and not pca_result.get("skipped", False)
    ):
        pca_selected = set(
            pca_result.get("selected_all", [])
        )

    positive_importance = set()
    if (
        isinstance(importance_result, dict)
        and not importance_result.get("skipped", False)
    ):
        importance_table = pd.DataFrame(
            importance_result.get("table", [])
        )
        required = {"feature", "importance_mean"}

        if (
            not importance_table.empty
            and required.issubset(importance_table.columns)
        ):
            positive_importance = set(
                importance_table.loc[
                    importance_table["importance_mean"].astype(float)
                    > float(minimum_positive_importance),
                    "feature",
                ].astype(str)
            )

    candidates = merge_unique_features(
        list(base_features),
        list(incremental_features),
        list(wind_features),
        list(time_features),
    )
    candidates = get_valid_feature_columns(
        train_data,
        candidates,
    )

    selected = []

    for feature in merge_unique_features(
        list(base_features),
        list(wind_features),
        list(time_features),
    ):
        if feature in candidates and feature not in selected:
            selected.append(feature)

    for feature in incremental_features:
        if feature not in candidates:
            continue

        if (
            feature in pca_selected
            or feature in positive_importance
        ):
            selected.append(feature)

    summary_records = []
    for feature in candidates:
        if feature in base_features:
            group = "base"
        elif feature in incremental_features:
            group = "incremental"
        elif feature in wind_features:
            group = "wind"
        elif feature in time_features:
            group = "time"
        else:
            group = "other"

        summary_records.append(
            {
                "feature": feature,
                "selected_for_shared_input": int(
                    feature in selected
                ),
                "supported_by_pca": int(
                    feature in pca_selected
                ),
                "supported_by_importance": int(
                    feature in positive_importance
                ),
                "feature_group": group,
            }
        )

    return {
        "selected_input_features": selected,
        "dropped_features": [
            feature
            for feature in candidates
            if feature not in selected
        ],
        "summary_table": summary_records,
    }


# ---------------------------------------------------------------------------
# Multi-seed permutation importance
# ---------------------------------------------------------------------------

def stratified_sample_by_rainfall(
    frame: pd.DataFrame,
    n_rows: int,
    *,
    target_column: str = "TP",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows by rainfall intensity strata."""
    if len(frame) <= n_rows:
        return frame.copy()

    work = frame.copy()
    bins = [
        -np.inf,
        0.0,
        RAIN_OCCURRENCE_THRESHOLD,
        HEAVY_RAIN_THRESHOLD,
        EXTREME_RAIN_THRESHOLD,
        np.inf,
    ]
    labels = [
        "zero",
        "trace",
        "rain",
        "heavy",
        "extreme",
    ]
    work["_rainfall_bin"] = pd.cut(
        work[target_column].astype(float),
        bins=bins,
        labels=labels,
        right=False,
    )

    rng = np.random.RandomState(random_state)
    selected_parts = []
    total_rows = len(work)

    for label in labels:
        subset = work[work["_rainfall_bin"] == label]
        if subset.empty:
            continue

        subset_rows = max(
            1,
            int(round(n_rows * len(subset) / total_rows)),
        )
        subset_rows = min(subset_rows, len(subset))
        indices = rng.choice(
            subset.index.to_numpy(),
            size=subset_rows,
            replace=False,
        )
        selected_parts.append(work.loc[indices])

    sampled = pd.concat(
        selected_parts,
        axis=0,
    ).drop_duplicates()

    if len(sampled) < n_rows:
        remaining = work.drop(
            index=sampled.index,
            errors="ignore",
        )
        needed = min(
            n_rows - len(sampled),
            len(remaining),
        )
        if needed > 0:
            extra_indices = rng.choice(
                remaining.index.to_numpy(),
                size=needed,
                replace=False,
            )
            sampled = pd.concat(
                [sampled, work.loc[extra_indices]],
                axis=0,
            )

    sampled = sampled.drop(
        columns=["_rainfall_bin"],
        errors="ignore",
    )
    return sampled.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)


def calculate_single_seed_importance(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    features: Sequence[str],
    *,
    maximum_train_rows: int,
    maximum_validation_rows: int,
    n_repeats: int,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    n_jobs: int,
) -> Dict[str, Any]:
    """Calculate one random-seed permutation-importance result without a figure."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import (
        make_scorer,
        mean_squared_error,
        r2_score,
    )

    if (
        "TP" not in train_data.columns
        or "TP" not in validation_data.columns
    ):
        return {
            "skipped": True,
            "reason": "TP is missing.",
        }

    valid_features = [
        feature
        for feature in features
        if (
            feature in train_data.columns
            and feature in validation_data.columns
        )
    ]

    if len(valid_features) < 3:
        return {
            "skipped": True,
            "reason": f"Not enough features: {valid_features}",
        }

    train_work = (
        train_data[valid_features + ["TP"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    validation_work = (
        validation_data[valid_features + ["TP"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(train_work) < 3_000:
        return {
            "skipped": True,
            "reason": (
                f"Not enough clean training rows: {len(train_work)}"
            ),
        }

    if len(validation_work) < 1_000:
        return {
            "skipped": True,
            "reason": (
                "Not enough clean validation rows: "
                f"{len(validation_work)}"
            ),
        }

    if len(train_work) > maximum_train_rows:
        train_work = stratified_sample_by_rainfall(
            train_work,
            maximum_train_rows,
            random_state=random_state,
        )

    if len(validation_work) > maximum_validation_rows:
        validation_work = stratified_sample_by_rainfall(
            validation_work,
            maximum_validation_rows,
            random_state=random_state,
        )

    x_train = train_work[valid_features].astype(float)
    y_train = train_work["TP"].astype(float).to_numpy()
    x_validation = validation_work[valid_features].astype(float)
    y_validation = validation_work["TP"].astype(float).to_numpy()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(x_train, y_train)

    prediction = model.predict(x_validation)
    baseline_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_validation,
                prediction,
            )
        )
    )
    baseline_r2 = float(
        r2_score(
            y_validation,
            prediction,
        )
    )

    def rmse_metric(y_true, y_pred):
        return np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

    scorer = make_scorer(
        rmse_metric,
        greater_is_better=False,
    )
    permutation_result = permutation_importance(
        model,
        x_validation,
        y_validation,
        scoring=scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance_table = (
        pd.DataFrame(
            {
                "feature": valid_features,
                "importance_mean": (
                    permutation_result.importances_mean
                ),
                "importance_std_within_seed": (
                    permutation_result.importances_std
                ),
            }
        )
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return {
        "skipped": False,
        "random_state": int(random_state),
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "table": importance_table.to_dict(
            orient="records"
        ),
        "analysis_sample_info": {
            "train_rows_used": int(len(train_work)),
            "validation_rows_used": int(
                len(validation_work)
            ),
            "maximum_train_rows": int(
                maximum_train_rows
            ),
            "maximum_validation_rows": int(
                maximum_validation_rows
            ),
            "sampling_strategy": "rainfall_stratified",
            "rainfall_thresholds": [
                RAIN_OCCURRENCE_THRESHOLD,
                HEAVY_RAIN_THRESHOLD,
                EXTREME_RAIN_THRESHOLD,
            ],
        },
    }


def run_multi_seed_permutation_importance(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    features: Sequence[str],
    figures_dir: Path,
    tables_dir: Path,
    output_root: Path,
    *,
    maximum_train_rows: int,
    maximum_validation_rows: int,
    n_repeats: int,
    seeds: Sequence[int],
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    n_jobs: int,
) -> Dict[str, Any]:
    """Aggregate permutation importance across seeds and draw one formal figure."""
    seed_tables = []
    seed_results = []

    for seed in seeds:
        logging.info(
            "Calculating permutation importance for seed %d.",
            seed,
        )
        result = calculate_single_seed_importance(
            train_data,
            validation_data,
            features,
            maximum_train_rows=maximum_train_rows,
            maximum_validation_rows=maximum_validation_rows,
            n_repeats=n_repeats,
            random_state=seed,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
        )
        seed_results.append(result)

        if result.get("skipped", False):
            logging.warning(
                "Seed %d was skipped: %s",
                seed,
                result.get("reason"),
            )
            continue

        current = pd.DataFrame(result["table"])
        current["seed"] = int(seed)
        seed_tables.append(current)

    if not seed_tables:
        return {
            "skipped": True,
            "reason": "All random-seed importance runs failed.",
        }

    by_seed = pd.concat(seed_tables, ignore_index=True)
    by_seed["rank"] = by_seed.groupby(
        "seed"
    )["importance_mean"].rank(
        ascending=False,
        method="average",
    )

    by_seed_path = tables_dir / "feature_importance_by_seed.csv"
    by_seed.to_csv(
        by_seed_path,
        index=False,
        encoding="utf-8-sig",
    )

    aggregate = (
        by_seed.groupby("feature")
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
            rank_mean=("rank", "mean"),
            rank_std=("rank", lambda values: values.std(ddof=0)),
        )
        .reset_index()
    )

    top_five_frequency: Dict[str, int] = {}
    for seed in seeds:
        current = by_seed[by_seed["seed"] == seed]
        top_five = current.nlargest(
            5,
            "importance_mean",
        )["feature"].tolist()
        for feature in top_five:
            top_five_frequency[feature] = (
                top_five_frequency.get(feature, 0) + 1
            )

    aggregate["top5_frequency"] = aggregate["feature"].map(
        lambda feature: top_five_frequency.get(feature, 0)
    )
    aggregate["selected_top5"] = aggregate["feature"].isin(
        aggregate.nlargest(
            5,
            "importance_mean",
        )["feature"].tolist()
    ).astype(int)
    aggregate = aggregate.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    aggregate_path = (
        tables_dir / "feature_importance_multi_seed.csv"
    )
    aggregate.to_csv(
        aggregate_path,
        index=False,
        encoding="utf-8-sig",
    )

    stability = (
        by_seed.groupby("feature")
        .agg(
            n_seeds_valid=("seed", "nunique"),
            mean_importance=("importance_mean", "mean"),
            sd_importance=("importance_mean", "std"),
            mean_rank=("rank", "mean"),
            sd_rank=("rank", lambda values: values.std(ddof=0)),
        )
        .reset_index()
    )
    stability["cv_importance"] = (
        stability["sd_importance"].abs()
        / (stability["mean_importance"].abs() + 1e-8)
    )
    stability["stability_label"] = stability[
        "cv_importance"
    ].apply(
        lambda coefficient: (
            "stable"
            if coefficient < 0.3
            else (
                "medium"
                if coefficient < 0.6
                else "unstable"
            )
        )
    )

    stability_path = (
        tables_dir / "feature_importance_seed_stability.csv"
    )
    stability.to_csv(
        stability_path,
        index=False,
        encoding="utf-8-sig",
    )

    plot_data = aggregate.sort_values(
        "importance_mean",
        ascending=True,
    ).copy()
    plot_data["display_name"] = [
        DISPLAY_NAME_MAP.get(feature, feature)
        for feature in plot_data["feature"]
    ]

    y_positions = np.arange(len(plot_data))
    means = plot_data["importance_mean"].astype(float).to_numpy()
    standard_deviations = np.nan_to_num(
        plot_data["importance_std"].astype(float).to_numpy(),
        nan=0.0,
    )

    height = max(
        3.35,
        0.40 * len(plot_data) + 0.75,
    )
    figure, axis = plt.subplots(
        figsize=(DOUBLE_COLUMN_PROFILE["width"], height)
    )

    axis.barh(
        y_positions,
        means,
        xerr=standard_deviations,
        height=0.58,
        color="#4C78A8",
        edgecolor="#2F5D8A",
        linewidth=0.75,
        error_kw={
            "elinewidth": 0.9,
            "ecolor": "#2F5D8A",
            "capsize": 2.5,
            "capthick": 0.9,
        },
    )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(
        plot_data["display_name"].tolist(),
        fontsize=DOUBLE_COLUMN_PROFILE["tick_fs"],
    )
    axis.tick_params(axis="y", pad=5, length=0)
    axis.set_xlabel(
        r"Increase in RMSE (mm h$^{-1}$)",
        fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
    )
    axis.set_ylabel(
        "Physical predictor",
        fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
    )

    x_maximum = (
        float(np.max(means + standard_deviations))
        if len(means)
        else 1.0
    )
    x_maximum = max(x_maximum, 1e-6)
    axis.set_xlim(0.0, x_maximum * 1.18)
    axis.axvline(
        0.0,
        color="black",
        linewidth=0.8,
        alpha=0.65,
    )
    style_axis(axis, grid_axis="x")
    axis.grid(False, axis="y")

    text_padding = 0.02 * x_maximum
    for index, (mean, std) in enumerate(
        zip(means, standard_deviations)
    ):
        axis.text(
            max(mean + std + text_padding, 0.012),
            index,
            f"{mean:.3f}",
            va="center",
            ha="left",
            fontsize=DOUBLE_COLUMN_PROFILE["annot_fs"],
            color="black",
        )

    figure.tight_layout(pad=0.45)
    figure_path = (
        figures_dir / "figure_permutation_importance.pdf"
    )
    save_pdf_figure(figure, figure_path)

    valid_seed_results = [
        result
        for result in seed_results
        if not result.get("skipped", False)
    ]

    return {
        "skipped": False,
        "n_seeds": int(len(valid_seed_results)),
        "seeds_requested": [int(seed) for seed in seeds],
        "seeds_used": [
            int(result["random_state"])
            for result in valid_seed_results
        ],
        "figure": relative_path(figure_path, output_root),
        "by_seed_table": relative_path(
            by_seed_path,
            output_root,
        ),
        "multi_seed_table": relative_path(
            aggregate_path,
            output_root,
        ),
        "stability_table": relative_path(
            stability_path,
            output_root,
        ),
        "table": aggregate.to_dict(orient="records"),
        "baseline_by_seed": [
            {
                "seed": int(result["random_state"]),
                "baseline_rmse": float(
                    result["baseline_rmse"]
                ),
                "baseline_r2": float(
                    result["baseline_r2"]
                ),
            }
            for result in valid_seed_results
        ],
    }


# ---------------------------------------------------------------------------
# Physical-context candidate selection and redundancy diagnosis
# ---------------------------------------------------------------------------

def select_physical_context_candidates(
    importance_result: Dict[str, Any],
    *,
    excluded_features: Optional[Sequence[str]] = None,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Select the top physical candidate variables from the importance table."""
    excluded_features = list(
        excluded_features or ["DOY", "HOD"]
    )

    if (
        not isinstance(importance_result, dict)
        or importance_result.get("skipped", False)
    ):
        return {
            "skipped": True,
            "reason": "Importance results are unavailable.",
        }

    importance_table = pd.DataFrame(
        importance_result.get("table", [])
    )
    if (
        importance_table.empty
        or "feature" not in importance_table.columns
    ):
        return {
            "skipped": True,
            "reason": "The importance table is empty.",
        }

    work = importance_table.copy()
    work["feature"] = work["feature"].astype(str)
    work = work.loc[
        ~work["feature"].isin(excluded_features)
    ].copy()
    work["importance_mean"] = work[
        "importance_mean"
    ].astype(float)
    work = work.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)

    selected_table = work.head(top_k).copy()
    selected_table["physical_process_role"] = (
        selected_table["feature"].map(
            lambda feature: PHYSICAL_PROCESS_ROLES.get(
                feature,
                "candidate physical-context variable",
            )
        )
    )

    return {
        "skipped": False,
        "excluded_features": excluded_features,
        "top_k": int(top_k),
        "physical_context_variables": (
            selected_table["feature"].tolist()
        ),
        "selected_table": selected_table.to_dict(
            orient="records"
        ),
        "full_ranking": work.to_dict(orient="records"),
    }


def diagnose_candidate_redundancy(
    train_data: pd.DataFrame,
    features: Sequence[str],
    tables_dir: Path,
    output_root: Path,
    *,
    correlation_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Check pairwise Spearman redundancy among selected candidates."""
    valid_features = [
        feature for feature in features if feature in train_data.columns
    ]

    if len(valid_features) < 2:
        return {
            "skipped": True,
            "reason": "Fewer than two candidate variables are available.",
            "n_features_available": len(valid_features),
        }

    work = (
        train_data[valid_features]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(work) < 2_000:
        return {
            "skipped": True,
            "reason": f"Not enough clean rows: {len(work)}",
            "n_features_available": len(valid_features),
        }

    correlation = work.corr(method="spearman")
    correlation_path = (
        tables_dir / "physical_context_redundancy_corr.csv"
    )
    correlation.to_csv(
        correlation_path,
        encoding="utf-8-sig",
    )

    high_pairs = []
    for first_index in range(len(valid_features)):
        for second_index in range(
            first_index + 1,
            len(valid_features),
        ):
            first_feature = valid_features[first_index]
            second_feature = valid_features[second_index]
            value = float(
                correlation.loc[
                    first_feature,
                    second_feature,
                ]
            )

            if abs(value) >= correlation_threshold:
                high_pairs.append(
                    {
                        "feature_a": first_feature,
                        "feature_b": second_feature,
                        "spearman_rho": round(value, 4),
                        "absolute_rho": round(abs(value), 4),
                        "correlation_threshold": float(
                            correlation_threshold
                        ),
                    }
                )

    high_pairs_table = pd.DataFrame(
        high_pairs,
        columns=[
            "feature_a",
            "feature_b",
            "spearman_rho",
            "absolute_rho",
            "correlation_threshold",
        ],
    )
    high_pairs_path = (
        tables_dir
        / "physical_context_high_redundancy_pairs.csv"
    )
    high_pairs_table.to_csv(
        high_pairs_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "skipped": False,
        "n_features_available": int(len(valid_features)),
        "n_clean_samples": int(len(work)),
        "correlation_threshold": float(correlation_threshold),
        "correlation_table": relative_path(
            correlation_path,
            output_root,
        ),
        "high_redundancy_pairs_table": relative_path(
            high_pairs_path,
            output_root,
        ),
        "high_redundancy_pairs": high_pairs,
        "n_high_pairs": int(len(high_pairs)),
        "features_checked": valid_features,
    }


# ---------------------------------------------------------------------------
# Rainfall-response analysis
# ---------------------------------------------------------------------------

def response_xlabel(feature: str) -> str:
    """Return the formal x-axis label for one response variable."""
    return RESPONSE_AXIS_LABELS.get(
        feature,
        DISPLAY_NAME_MAP.get(feature, feature),
    )


def response_title(feature: str) -> str:
    """Return the formal panel title for one response variable."""
    return RESPONSE_TITLES.get(
        feature,
        DISPLAY_NAME_MAP.get(feature, feature),
    )


def calculate_binned_response(
    data: pd.DataFrame,
    feature: str,
    *,
    target: str,
    n_bins: int,
    minimum_count_per_bin: int,
) -> tuple[pd.DataFrame, float, int]:
    """Calculate quantile-binned mean and median rainfall responses."""
    work = (
        data[[feature, target]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    n_clean = len(work)

    if n_clean < max(
        500,
        n_bins * minimum_count_per_bin,
    ):
        raise ValueError(
            f"Not enough clean rows for {feature}: {n_clean}"
        )

    work = work.copy()
    work["bin"] = pd.qcut(
        work[feature],
        q=n_bins,
        duplicates="drop",
    )

    grouped = (
        work.groupby(
            "bin",
            observed=False,
        )[target]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    grouped = grouped.loc[
        grouped["count"] >= minimum_count_per_bin
    ].copy()

    if grouped.empty:
        raise ValueError(
            f"No valid response bins remained for {feature}."
        )

    grouped["bin_center"] = [
        interval.mid if hasattr(interval, "mid") else np.nan
        for interval in grouped["bin"]
    ]

    x_values = grouped["bin_center"].astype(float).to_numpy()
    y_values = grouped["mean"].astype(float).to_numpy()

    if np.isfinite(x_values).sum() >= 3:
        correlation, _ = spearmanr(x_values, y_values)
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0

    return grouped, float(correlation), int(n_clean)


def draw_response_axis(
    axis,
    feature: str,
    response_table: pd.DataFrame,
    *,
    panel_label: Optional[str] = None,
    show_y_label: bool = True,
    include_legend: bool = False,
) -> tuple[Any, Any]:
    """Draw one mean/median rainfall-response panel."""
    x_values = response_table["bin_center"].astype(float).to_numpy()
    mean_values = response_table["mean"].astype(float).to_numpy()
    median_values = response_table["median"].astype(float).to_numpy()

    mean_line, = axis.plot(
        x_values,
        mean_values,
        color="#1f77b4",
        marker="o",
        linewidth=1.15,
        markersize=3.0,
        label="Mean rainfall",
        zorder=2,
    )
    median_line, = axis.plot(
        x_values,
        median_values,
        color="#E68613",
        marker="s",
        linewidth=1.05,
        markersize=2.8,
        linestyle="--",
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Median rainfall",
        zorder=3,
    )

    title = response_title(feature)
    if panel_label:
        title = f"{panel_label} {title}"

    axis.set_title(
        title,
        fontsize=DOUBLE_COLUMN_PROFILE["title_fs"],
        fontweight="normal",
        loc="left" if panel_label else "center",
        pad=3,
    )
    axis.set_xlabel(
        response_xlabel(feature),
        fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
    )

    if show_y_label:
        axis.set_ylabel(
            r"Rainfall intensity (mm h$^{-1}$)",
            fontsize=DOUBLE_COLUMN_PROFILE["label_fs"],
        )
    else:
        axis.set_ylabel("")

    style_axis(axis)
    axis.axhline(
        0.0,
        color="0.6",
        linewidth=0.6,
        linestyle="--",
        alpha=0.45,
        zorder=1,
    )
    set_clean_limits(
        axis,
        x_values,
        [mean_values, median_values],
        top_padding=0.22,
        bottom_padding=0.02,
    )

    if include_legend:
        axis.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="0.72",
            facecolor="white",
            borderpad=0.30,
            labelspacing=0.28,
            handlelength=2.0,
            fontsize=DOUBLE_COLUMN_PROFILE["legend_fs"],
        )

    return mean_line, median_line


def run_rainfall_response_analysis(
    train_data: pd.DataFrame,
    features: Sequence[str],
    figures_dir: Path,
    tables_dir: Path,
    output_root: Path,
    *,
    target: str,
    n_bins: int,
    minimum_count_per_bin: int,
) -> Dict[str, Any]:
    """Generate six individual response figures and one combined figure."""
    if target not in train_data.columns:
        return {
            "skipped": True,
            "reason": f"{target} is missing.",
        }

    response_results: Dict[str, Dict[str, Any]] = {}
    ranking_records = []

    for feature in features:
        if feature not in train_data.columns:
            logging.warning(
                "Skipping rainfall response for missing feature %s.",
                feature,
            )
            continue

        try:
            response_table, correlation, n_clean = (
                calculate_binned_response(
                    train_data,
                    feature,
                    target=target,
                    n_bins=n_bins,
                    minimum_count_per_bin=(
                        minimum_count_per_bin
                    ),
                )
            )
        except ValueError as error:
            logging.warning("%s", error)
            continue

        token = safe_feature_token(feature)
        table_path = tables_dir / f"response_{token}.csv"
        response_table.to_csv(
            table_path,
            index=False,
            encoding="utf-8-sig",
        )

        figure, axis = plt.subplots(
            figsize=(DOUBLE_COLUMN_PROFILE["width"], 4.25)
        )
        draw_response_axis(
            axis,
            feature,
            response_table,
            include_legend=True,
        )
        figure.tight_layout(pad=0.55)

        figure_path = (
            figures_dir / f"figure_response_{token}.pdf"
        )
        save_pdf_figure(figure, figure_path)

        physical_role = PHYSICAL_PROCESS_ROLES.get(
            feature,
            "candidate physical-context variable",
        )
        response_results[feature] = {
            "figure": relative_path(
                figure_path,
                output_root,
            ),
            "table": relative_path(
                table_path,
                output_root,
            ),
            "binned_response_spearman": float(
                correlation
            ),
            "physical_process_role": physical_role,
            "n_clean_samples": int(n_clean),
            "n_valid_bins": int(len(response_table)),
        }

        ranking_records.append(
            {
                "feature": feature,
                "binned_response_spearman": float(
                    correlation
                ),
                "physical_process_role": physical_role,
            }
        )

    if not response_results:
        return {
            "skipped": True,
            "reason": "No valid response result was generated.",
        }

    combined_features = [
        feature
        for feature in RESPONSE_FEATURE_ORDER
        if feature in response_results
    ]
    if not combined_features:
        combined_features = list(response_results)[:6]
    combined_features = combined_features[:6]

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(DOUBLE_COLUMN_PROFILE["width"], 4.25),
    )
    axes_flat = np.asarray(axes).ravel()
    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    legend_handles = None
    legend_labels = None

    for index, axis in enumerate(axes_flat):
        if index >= len(combined_features):
            axis.axis("off")
            continue

        feature = combined_features[index]
        table_path = output_root / response_results[feature]["table"]
        response_table = pd.read_csv(table_path)

        mean_line, median_line = draw_response_axis(
            axis,
            feature,
            response_table,
            panel_label=panel_labels[index],
            show_y_label=(index % 3 == 0),
            include_legend=False,
        )

        if legend_handles is None:
            legend_handles = [mean_line, median_line]
            legend_labels = [
                "Mean rainfall",
                "Median rainfall",
            ]

    if legend_handles is not None:
        figure.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="0.72",
            facecolor="white",
            borderpad=0.30,
            labelspacing=0.28,
            handlelength=2.0,
            fontsize=DOUBLE_COLUMN_PROFILE["legend_fs"],
        )

    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.075,
        top=0.920,
        wspace=0.26,
        hspace=0.36,
    )

    combined_path = figures_dir / "figure_rainfall_response.pdf"
    save_pdf_figure(figure, combined_path)

    ranking_records = sorted(
        ranking_records,
        key=lambda record: abs(
            record["binned_response_spearman"]
        ),
        reverse=True,
    )

    return {
        "skipped": False,
        "target": target,
        "features": list(response_results),
        "combined_figure": relative_path(
            combined_path,
            output_root,
        ),
        "individual_outputs": response_results,
        "ranking_table": ranking_records,
    }


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_feature_analysis(
    data_dir: Path,
    years: Sequence[int],
    output_root: Path,
    *,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    clean_output: bool,
) -> Dict[str, Any]:
    """Run the complete formal feature-analysis workflow."""
    if clean_output and output_root.exists():
        shutil.rmtree(output_root)

    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    ensure_directory(figures_dir)
    ensure_directory(tables_dir)

    configure_plotting()
    config = FEATURE_ANALYSIS_CONFIG

    data = load_regional_data(data_dir, years)
    data = rename_variables(data)
    data = add_time_features(data)
    data = prepare_derived_features(data)

    train_mask, validation_mask, test_mask, split_info = build_time_split(
        data,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
    )

    train_data = data.loc[train_mask].copy()
    validation_data = data.loc[validation_mask].copy()
    test_rows = int(test_mask.sum())

    logging.info("Training rows: %d", len(train_data))
    logging.info("Validation rows: %d", len(validation_data))
    logging.info("Test rows: %d", test_rows)

    if train_data.empty:
        raise ValueError(
            "The feature-analysis training split is empty."
        )
    if validation_data.empty:
        raise ValueError(
            "The feature-analysis validation split is empty."
        )
    if test_rows == 0:
        logging.warning(
            "The feature-analysis test split is empty."
        )

    base_features = get_valid_feature_columns(
        train_data,
        PHYSICAL_BASE_FEATURES,
    )
    incremental_features = get_valid_feature_columns(
        train_data,
        INCREMENTAL_FEATURES,
    )
    time_features = get_valid_feature_columns(
        train_data,
        TIME_FEATURES,
    )

    wind_controls = [
        feature
        for feature in ["T", "P", "RH", "PWV", "DOY", "HOD"]
        if feature in train_data.columns
    ]
    wind_selection = choose_wind_representation(
        train_data,
        raw_columns=RAW_WIND_FEATURES,
        reparameterized_columns=REPARAMETERIZED_WIND_FEATURES,
        target="TP",
        controls=wind_controls,
        corr_method="spearman",
        beta=config["wind_beta"],
        variance_threshold=config["wind_var_threshold"],
        bootstrap_iterations=config["wind_bootstrap_B"],
        bootstrap_rows=config["wind_bootstrap_n"],
        random_state=config["random_state"],
    )
    wind_features = wind_selection.get("keep", [])

    correlation_features = [
        feature
        for feature in ["PWV", "T", "V", "RH", "P", "U"]
        if feature in train_data.columns
    ]
    correlation_result = plot_feature_correlation(
        train_data,
        correlation_features,
        figures_dir,
        tables_dir,
        output_root,
    )

    pca_features = merge_unique_features(
        base_features,
        incremental_features,
        wind_features,
        time_features,
    )
    pca_result = run_pca_analysis(
        train_data,
        pca_features,
        figures_dir,
        tables_dir,
        output_root,
        max_rows=config["pca_max_rows"],
        random_state=config["random_state"],
        n_display_components=6,
    )

    importance_features = merge_unique_features(
        base_features,
        incremental_features,
        wind_features,
    )
    importance_result = run_multi_seed_permutation_importance(
        train_data,
        validation_data,
        importance_features,
        figures_dir,
        tables_dir,
        output_root,
        maximum_train_rows=config[
            "importance_train_max_rows"
        ],
        maximum_validation_rows=config[
            "importance_eval_max_rows"
        ],
        n_repeats=config["importance_n_repeats"],
        seeds=config["importance_seeds"],
        n_estimators=config[
            "importance_n_estimators"
        ],
        max_depth=config["importance_max_depth"],
        min_samples_leaf=config[
            "importance_min_samples_leaf"
        ],
        n_jobs=config["importance_n_jobs"],
    )

    candidate_selection = select_physical_context_candidates(
        importance_result,
        excluded_features=["DOY", "HOD"],
        top_k=config["physical_context_top_k"],
    )
    physical_context_variables = (
        candidate_selection.get(
            "physical_context_variables",
            [],
        )
        if isinstance(candidate_selection, dict)
        else []
    )

    redundancy_result = diagnose_candidate_redundancy(
        train_data,
        physical_context_variables,
        tables_dir,
        output_root,
        correlation_threshold=config[
            "redundancy_corr_threshold"
        ],
    )

    response_result = run_rainfall_response_analysis(
        train_data,
        physical_context_variables,
        figures_dir,
        tables_dir,
        output_root,
        target="TP",
        n_bins=config["response_n_bins"],
        minimum_count_per_bin=config[
            "response_min_count_per_bin"
        ],
    )

    shared_input_selection = finalize_shared_input_features(
        train_data,
        base_features=base_features,
        incremental_features=incremental_features,
        wind_features=wind_features,
        time_features=time_features,
        pca_result=pca_result,
        importance_result=importance_result,
        minimum_positive_importance=0.0,
    )
    selected_shared_inputs = (
        shared_input_selection[
            "selected_input_features"
        ]
    )
    selected_input_features_for_all_models = merge_unique_features(
        selected_shared_inputs,
        SHARED_DERIVED_PREDICTORS,
    )

    selected_summary_path = (
        tables_dir / "selected_input_feature_summary.csv"
    )
    pd.DataFrame(
        shared_input_selection["summary_table"]
    ).to_csv(
        selected_summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    split_info.update(
        {
            "train_rows": int(len(train_data)),
            "validation_rows": int(
                len(validation_data)
            ),
            "test_rows": int(test_rows),
        }
    )

    selection_config = {
        "stage_name": (
            "feature_analysis_physical_context_screening"
        ),
        "generated_at": generated_at(),
        "selected_input_features_for_all_models": (
            selected_input_features_for_all_models
        ),
        "shared_derived_predictors": (
            SHARED_DERIVED_PREDICTORS
        ),
        "physical_context_variables": (
            physical_context_variables
        ),
        "physical_process_roles": {
            feature: PHYSICAL_PROCESS_ROLES.get(
                feature,
                "candidate physical-context variable",
            )
            for feature in physical_context_variables
        },
        "downstream_usage": {
            "purpose": (
                "Provide screened meteorological variables for "
                "downstream physical-diagnostic construction and "
                "model configuration."
            ),
            "diagnostic_combinations_defined_here": False,
        },
        "wind_representation_selection": wind_selection,
        "correlation_analysis": correlation_result,
        "pca": pca_result,
        "importance": importance_result,
        "candidate_selection": candidate_selection,
        "redundancy_diagnosis": redundancy_result,
        "response_diagnosis": response_result,
        "selected_input_summary_table": relative_path(
            selected_summary_path,
            output_root,
        ),
        "split_info": split_info,
    }

    selection_config_path = (
        output_root / "physical_context_selection_config.json"
    )
    save_json(selection_config_path, selection_config)

    individual_response_figures = {}
    if (
        isinstance(response_result, dict)
        and not response_result.get("skipped", True)
    ):
        individual_response_figures = {
            feature: values["figure"]
            for feature, values in response_result[
                "individual_outputs"
            ].items()
        }

    figure_manifest = {
        "feature_correlation": correlation_result.get(
            "figure"
        ),
        "pca_loadings": pca_result.get("figure"),
        "permutation_importance": importance_result.get(
            "figure"
        ),
        "rainfall_response_combined": response_result.get(
            "combined_figure"
        ),
        "rainfall_response_individual": (
            individual_response_figures
        ),
    }

    figure_manifest_path = output_root / "figure_manifest.json"
    save_json(figure_manifest_path, figure_manifest)

    run_summary = {
        "output_root": output_root.as_posix(),
        "physical_context_selection_config": (
            selection_config_path.as_posix()
        ),
        "figure_manifest": figure_manifest_path.as_posix(),
        "selected_input_features_for_all_models": (
            selected_input_features_for_all_models
        ),
        "physical_context_variables": (
            physical_context_variables
        ),
        "wind_features_kept": wind_features,
        "wind_features_dropped": wind_selection.get(
            "drop",
            [],
        ),
    }
    run_summary_path = output_root / "run_summary.json"
    save_json(run_summary_path, run_summary)

    return run_summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the formal double-column feature-analysis "
            "workflow for regional rainfall forecasting."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("converted_data"),
        help=(
            "Directory containing data_2020.csv through "
            "data_2025.csv."
        ),
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2020, 2021, 2022, 2023, 2024, 2025],
        help="Annual CSV files included in the analysis.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/feature_analysis"),
        help="Output directory for figures, tables, and JSON files.",
    )
    parser.add_argument(
        "--train-end",
        default="2024-06-30 23:59:59",
    )
    parser.add_argument(
        "--validation-start",
        default="2024-07-01 00:00:00",
    )
    parser.add_argument(
        "--validation-end",
        default="2024-12-31 23:59:59",
    )
    parser.add_argument(
        "--test-start",
        default="2025-01-01 00:00:00",
    )
    parser.add_argument(
        "--clean-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Delete the existing output directory before the run."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line feature-analysis workflow."""
    arguments = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    result = run_feature_analysis(
        data_dir=arguments.data_dir,
        years=arguments.years,
        output_root=arguments.output_root,
        train_end=arguments.train_end,
        validation_start=arguments.validation_start,
        validation_end=arguments.validation_end,
        test_start=arguments.test_start,
        clean_output=arguments.clean_output,
    )

    logging.info(
        "Feature analysis completed. Output directory: %s",
        result["output_root"],
    )
    logging.info(
        "Selected shared inputs: %s",
        result["selected_input_features_for_all_models"],
    )
    logging.info(
        "Physical-context candidates: %s",
        result["physical_context_variables"],
    )


if __name__ == "__main__":
    main()
