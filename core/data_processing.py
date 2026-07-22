"""Regional grid data preparation with explicit alignment and leakage checks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .physical_diagnostics import add_physical_diagnostic_features

FEATURE_NAME_MAPPING = {"V11001": "T", "V11002": "U", "V11003": "V", "V07004": "P", "V13010": "RH", "V13012": "PWV", "V13013": "dPWV+", "V13014": "dPWV-"}

def add_standard_and_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for raw, semantic in FEATURE_NAME_MAPPING.items():
        if raw in result and semantic not in result: result[semantic] = result[raw]
    if "datetime_utc" in result:
        dates = pd.to_datetime(result["datetime_utc"], errors="coerce")
        if "DOY" not in result: result["DOY"] = dates.dt.dayofyear
        if "HOD" not in result: result["HOD"] = dates.dt.hour
    return result

def load_annual_csv_data(data_dir: str | Path, years) -> pd.DataFrame:
    paths = [Path(data_dir) / f"data_{int(year)}.csv" for year in years]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing: raise FileNotFoundError(f"Required annual CSV files are missing: {missing}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

def validate_hourly_continuity(times) -> None:
    values = pd.DatetimeIndex(pd.to_datetime(times)).drop_duplicates().sort_values()
    if len(values) < 2: return
    gaps = np.diff(values.values.astype("datetime64[h]")).astype("timedelta64[h]").astype(int)
    if np.any(gaps != 1): raise ValueError("Regional data timestamps are not hourly continuous.")

def validate_complete_grid(frame: pd.DataFrame, time_col="datetime_utc", lat_col="V05002", lon_col="V05001", grid_shape=(9, 5)) -> list[tuple[float, float]]:
    if frame.duplicated([time_col, lat_col, lon_col]).any(): raise ValueError("Duplicate time-grid rows detected.")
    cells = [(float(lat), float(lon)) for lat, lon in frame[[lat_col, lon_col]].drop_duplicates().sort_values([lat_col, lon_col]).itertuples(index=False, name=None)]
    if len(cells) != grid_shape[0] * grid_shape[1]: raise ValueError(f"Expected {grid_shape[0] * grid_shape[1]} unique grid cells, found {len(cells)}.")
    expected = set(cells)
    for time, group in frame.groupby(time_col, sort=False):
        observed = {(float(lat), float(lon)) for lat, lon in group[[lat_col, lon_col]].itertuples(index=False, name=None)}
        if observed != expected: raise ValueError(f"Incomplete grid field at {time}.")
    return cells

def prepare_regional_grid_sequences(data_frame: pd.DataFrame, input_hours=12, target_grid_points=None, grid_shape=(9, 5), selected_features=None, rain_col="V13011") -> dict:
    """Create 12-hour input windows whose target is the immediately next hour."""
    if input_hours <= 0: raise ValueError("input_hours must be positive.")
    data = add_standard_and_derived_features(data_frame)
    if "datetime_utc" not in data:
        required = ["V04001", "V04002", "V04003", "V04004"]
        if not all(column in data for column in required): raise ValueError("Missing datetime_utc and component datetime columns.")
        data["datetime_utc"] = pd.to_datetime({"year": data["V04001"], "month": data["V04002"], "day": data["V04003"], "hour": data["V04004"]}, errors="raise")
    # Calendar features must be derived only after a valid timestamp exists.
    # Calling this twice is intentional: the first pass maps raw columns and
    # the second pass materializes DOY/HOD from datetime_utc.
    data = add_standard_and_derived_features(data)
    if not {"DOY", "HOD"}.issubset(data.columns):
        raise ValueError("DOY and HOD must exist before physical diagnostics and sequence construction.")
    data = add_physical_diagnostic_features(data, use_spatial_diagnostics=True)
    if rain_col not in data: raise ValueError(f"Missing rainfall target column: {rain_col}")
    features = list(selected_features or [])
    if not features: raise ValueError("Selected feature columns must be provided explicitly.")
    missing = [column for column in features if column not in data]
    if missing: raise ValueError(f"Missing selected feature columns: {missing}")
    data = data.sort_values(["datetime_utc", "V05002", "V05001"]).reset_index(drop=True)
    validate_hourly_continuity(data["datetime_utc"]); cells = validate_complete_grid(data, grid_shape=grid_shape)
    if target_grid_points is not None and list(target_grid_points) != cells: raise ValueError("Provided grid order does not match strict latitude-major ordering.")
    groups = [group.sort_values(["V05002", "V05001"]) for _, group in data.groupby("datetime_utc", sort=True)]
    fields = np.stack([group[features].to_numpy(dtype=np.float32).reshape(*grid_shape, len(features)) for group in groups])
    rainfall = np.stack([group[rain_col].to_numpy(dtype=np.float32) for group in groups])
    valid_times = np.asarray([group["datetime_utc"].iloc[0] for group in groups])
    if len(fields) <= input_hours: raise ValueError("Insufficient hourly fields for requested input window.")
    return {"X_seq": np.stack([fields[index:index + input_hours] for index in range(len(fields) - input_hours)]), "y_reg": rainfall[input_hours:], "rain_seq_reg": np.stack([rainfall[index:index + input_hours] for index in range(len(fields) - input_hours)]), "valid_times": valid_times[input_hours:], "grid_points_order": cells, "feature_columns": features}

def split_sequences_by_target_time(pack: dict, train_end="2024-06-30 23:59:59", validation_start="2024-07-01 00:00:00", validation_end="2024-12-31 23:59:59", test_start="2025-01-01 00:00:00") -> dict:
    """Split already-built windows by target valid time without overlap."""
    boundaries = {
        "train_end": pd.Timestamp(train_end),
        "validation_start": pd.Timestamp(validation_start),
        "validation_end": pd.Timestamp(validation_end),
        "test_start": pd.Timestamp(test_start),
    }
    if not (boundaries["train_end"] < boundaries["validation_start"]):
        raise ValueError("train_end must be before validation_start.")
    if not (boundaries["validation_start"] <= boundaries["validation_end"]):
        raise ValueError("validation_start must not be after validation_end.")
    if not (boundaries["validation_end"] < boundaries["test_start"]):
        raise ValueError("validation_end must be before test_start.")
    times = pd.to_datetime(pack["valid_times"], errors="raise")
    masks = {
        "train": times <= boundaries["train_end"],
        "validation": (times >= boundaries["validation_start"]) & (times <= boundaries["validation_end"]),
        "test": times >= boundaries["test_start"],
    }
    if any(np.any(first & second) for first, second in ((masks["train"], masks["validation"]), (masks["train"], masks["test"]), (masks["validation"], masks["test"]))):
        raise ValueError("Chronological split masks overlap.")
    empty = [name for name, mask in masks.items() if not np.any(mask)]
    if empty:
        raise ValueError(f"Chronological split(s) are empty: {empty}")
    output = {"grid_points_order": pack["grid_points_order"], "feature_columns": pack["feature_columns"]}
    for split, mask in masks.items():
        output[split] = {key: np.asarray(pack[key])[mask] for key in ("X_seq", "y_reg", "rain_seq_reg", "valid_times")}
    return output

def build_event_aware_five_class_keep_indices(y_train, valid_times_train, rain_threshold=0.1, moderate_threshold=2.5, near_heavy_threshold=7.5, heavy_threshold=10.0, event_neighbor_hours=3, keep_ratios=None, random_seed=42):
    """Build train-only keep indices using monthly five-class sampling."""
    y_train = np.asarray(y_train, dtype=np.float32)
    if y_train.ndim != 2 or y_train.shape[1] != 45: raise ValueError("y_train must have shape (samples, 45) for formal sampling.")
    if keep_ratios is None: keep_ratios={"no_rain":.60,"light":.80,"moderate":1.,"near_heavy":1.,"heavy":1.,"event_neighbor":1.}
    valid_times=pd.DatetimeIndex(pd.to_datetime(valid_times_train,errors="raise")); months=valid_times.to_period("M").astype(str).to_numpy(); rng=np.random.default_rng(int(random_seed)); maxima=y_train.max(axis=1)
    classes=np.full(len(maxima),"unknown",dtype=object); classes[maxima<rain_threshold]="no_rain"; classes[(maxima>=rain_threshold)&(maxima<moderate_threshold)]="light"; classes[(maxima>=moderate_threshold)&(maxima<near_heavy_threshold)]="moderate"; classes[(maxima>=near_heavy_threshold)&(maxima<heavy_threshold)]="near_heavy"; classes[maxima>=heavy_threshold]="heavy"
    heavy=maxima>=heavy_threshold; neighbours=np.zeros(len(maxima),dtype=bool)
    for timestamp in valid_times[heavy]: neighbours |= np.abs((valid_times-timestamp).total_seconds()) / 3600 <= event_neighbor_hours
    protected=heavy | (classes=="near_heavy") | neighbours; selected=[]; monthly={}; order=["no_rain","light","moderate","near_heavy","heavy"]
    for month in np.unique(months):
        month_indices=np.where(months==month)[0]; monthly[month]={}
        for label in order:
            indices=month_indices[classes[month_indices]==label]; keep=indices[protected[indices]]; ordinary=indices[~protected[indices]]; ratio=float(keep_ratios.get(label,1.)); number=len(ordinary) if ratio>=1 else max(1,int(round(len(ordinary)*ratio))) if len(ordinary) else 0; ordinary_keep=ordinary if ratio>=1 else (rng.choice(ordinary,size=number,replace=False) if number else np.array([],dtype=int)); current=np.concatenate([keep,ordinary_keep]); selected.append(current); monthly[month][label]={"before":int(len(indices)),"after":int(len(current)),"protected_before":int(protected[indices].sum()),"keep_ratio":ratio}
    indices=np.unique(np.concatenate(selected) if selected else np.array([],dtype=int)); rng.shuffle(indices)
    def counts(index):
        values=maxima[index]; return {"no_rain":int((values<rain_threshold).sum()),"light":int(((values>=rain_threshold)&(values<moderate_threshold)).sum()),"moderate":int(((values>=moderate_threshold)&(values<near_heavy_threshold)).sum()),"near_heavy":int(((values>=near_heavy_threshold)&(values<heavy_threshold)).sum()),"heavy":int((values>=heavy_threshold).sum()),"event_neighbor":int(neighbours[index].sum()),"protected_total":int(protected[index].sum()),"total":int(len(index))}
    before,after=counts(np.arange(len(y_train))),counts(indices)
    percentage_keys = ("no_rain", "light", "moderate", "near_heavy", "heavy")
    return indices,{"method":"event_aware_five_class_keep_ratio_train_only","random_seed":int(random_seed),"thresholds":{"rain_threshold":float(rain_threshold),"moderate_threshold":float(moderate_threshold),"near_heavy_threshold":float(near_heavy_threshold),"heavy_threshold":float(heavy_threshold),"event_neighbor_hours":int(event_neighbor_hours)},"keep_ratios":{k:float(v) for k,v in keep_ratios.items()},"before_counts":before,"after_counts":after,"before_percent":{key:100.0 * before[key] / max(before["total"], 1) for key in percentage_keys},"after_percent":{key:100.0 * after[key] / max(after["total"], 1) for key in percentage_keys},"per_month_stats":monthly}

def apply_training_sample_balance(train_split, sample_balance_config, output_dir):
    """Apply sampling only to the training split and save its manifest."""
    from pathlib import Path
    config=sample_balance_config; indices,manifest=build_event_aware_five_class_keep_indices(train_split["y_reg"],train_split["valid_times"],rain_threshold=config["rain_threshold_mm"],moderate_threshold=config["moderate_threshold_mm"],near_heavy_threshold=config["near_heavy_threshold_mm"],heavy_threshold=config["heavy_threshold_mm"],event_neighbor_hours=config["event_neighbor_hours"],keep_ratios=config["keep_ratios"],random_seed=config["seed"])
    balanced={key:np.asarray(train_split[key])[indices] for key in ("X_seq","y_reg","rain_seq_reg","valid_times")}; directory=Path(output_dir); directory.mkdir(parents=True,exist_ok=True); np.save(directory/"train_selected_indices.npy",indices); (directory/"sample_balance_manifest.json").write_text(__import__("json").dumps(manifest,indent=2),encoding="utf-8"); return balanced,manifest

def check_lagged_rainfall_no_target_leakage(rain_sequences, targets, atol=1e-6, maximum_equal_fraction=0.99) -> dict:
    rain_sequences, targets = np.asarray(rain_sequences), np.asarray(targets)
    if rain_sequences.ndim != 3 or targets.ndim != 2 or rain_sequences.shape[0] != targets.shape[0] or rain_sequences.shape[2] != targets.shape[1]: raise ValueError("Lagged rainfall and target shapes are incompatible.")
    equal = np.isclose(rain_sequences[:, -1], targets, rtol=0, atol=atol); fraction = float(np.mean(np.all(equal, axis=1)))
    if np.array_equal(rain_sequences[:, -1], targets) or fraction >= maximum_equal_fraction: raise ValueError("Lagged rainfall appears to contain target-time rainfall.")
    return {"status": "pass", "row_equal_fraction": fraction, "element_equal_fraction": float(np.mean(equal))}

def append_lagged_rainfall_channel(X, rain_sequences, targets):
    check_lagged_rainfall_no_target_leakage(rain_sequences, targets)
    X, rain = np.asarray(X, dtype=np.float32), np.asarray(rain_sequences, dtype=np.float32)
    if X.shape[:2] != rain.shape[:2] or X.shape[2] * X.shape[3] != rain.shape[-1]: raise ValueError("Rainfall sequences do not align with the feature tensor.")
    return np.concatenate([X, rain.reshape(*X.shape[:4], 1)], axis=-1)

def append_lagged_rainfall_channel_train_validation_test(X_train, X_validation, X_test, rain_seq_train, rain_seq_validation, rain_seq_test, y_train, y_validation, y_test, grid_shape=(9,5)):
    """Append standardized historical rainfall using training statistics."""
    checks={"train":check_lagged_rainfall_no_target_leakage(rain_seq_train,y_train),"validation":check_lagged_rainfall_no_target_leakage(rain_seq_validation,y_validation),"test":check_lagged_rainfall_no_target_leakage(rain_seq_test,y_test)}
    train=np.asarray(rain_seq_train,dtype=np.float32); mean=float(train.mean()); std=float(train.std()); std=1.0 if std < 1e-6 else std
    def append(X,rain):
        X=np.asarray(X,dtype=np.float32); rain=(np.asarray(rain,dtype=np.float32)-mean)/std
        if rain.shape != (X.shape[0],X.shape[1],X.shape[2]*X.shape[3]): raise ValueError("Rainfall sequences do not align with feature tensors.")
        return np.concatenate([X,rain.reshape(*X.shape[:4],1)],axis=-1)
    return append(X_train,rain_seq_train),append(X_validation,rain_seq_validation),append(X_test,rain_seq_test),{"mean":mean,"std":std,"normalization":"train_only_standardization","leakage_checks":checks}

def standardize_feature_splits(train, validation, test):
    """Compatibility wrapper for the formal training-only normalization."""
    train_norm, validation_norm, test_norm, scalers = normalize_regional_data(train, validation, test)
    return train_norm, validation_norm, test_norm, scalers["standard"]


def normalize_regional_data(X_train, X_validation=None, X_test=None, scalers=None):
    """Normalize regional tensors with the original reshape and NaN timing.

    The scaler is fit on balanced training samples only.  Each tensor is
    flattened in C order to (N*T*H*W, F), NaNs are replaced immediately before
    fitting/transforming, and the original tensor shape is restored unchanged.
    """
    def reshape_for_scaler(values):
        if values is None:
            return None, None
        values = np.asarray(values)
        if values.ndim == 3:
            n, t, f = values.shape
            return values.reshape(n * t, f), (n, t, f)
        if values.ndim == 4:
            n, t, f, g = values.shape
            return values.reshape(n * t, f * g), (n, t, f, g)
        if values.ndim == 5:
            n, t, h, w, f = values.shape
            return values.reshape(n * t * h * w, f), (n, t, h, w, f)
        raise ValueError(f"Unsupported X shape: {values.shape}")

    def restore_shape(values, shape):
        if len(shape) == 3:
            return values.reshape(shape)
        if len(shape) == 4:
            n, t, f, g = shape
            return values.reshape(n, t, f * g)
        return values.reshape(shape)

    train_2d, train_shape = reshape_for_scaler(X_train)
    if scalers is None:
        scaler = StandardScaler().fit(np.nan_to_num(train_2d, nan=0.0))
        scalers = {"standard": scaler}
    else:
        scaler = scalers["standard"]

    def transform(values):
        values_2d, shape = reshape_for_scaler(values)
        if values_2d is None:
            return None
        return restore_shape(scaler.transform(np.nan_to_num(values_2d, nan=0.0)), shape)

    return transform(X_train), transform(X_validation), transform(X_test), scalers


def _summarize_regional_targets(y_reg, thresholds):
    values = np.asarray(y_reg, dtype=np.float32)
    maxima = values.max(axis=1) if values.ndim == 2 else values
    rain, moderate, heavy = (float(thresholds[key]) for key in ("rain_threshold_mm", "moderate_threshold_mm", "heavy_threshold_mm"))
    masks = {
        "no_rain": maxima < rain,
        "light": (maxima >= rain) & (maxima < moderate),
        "moderate": (maxima >= moderate) & (maxima < heavy),
        "heavy": maxima >= heavy,
    }
    total = int(maxima.size)
    result = {f"n_{name}": int(mask.sum()) for name, mask in masks.items()}
    result.update({f"pct_{name}": 100.0 * result[f"n_{name}"] / max(total, 1) for name in masks})
    result["total"] = total
    return result


def process_regional_data_for_training(data_dir, years, input_hours, selected_features,
                                       grid_shape=(9, 5), split_config=None,
                                       sample_balance_config=None, sample_balance_dir=None):
    """Execute the only formal preprocessing order used by training.

    Full annual fields are sequenced before target-time splitting; balancing is
    train-only and precedes scaler fitting.  This deliberately does not split
    a raw DataFrame, which would drop boundary history from validation/test.
    """
    data = load_annual_csv_data(data_dir, years)
    full = prepare_regional_grid_sequences(data, input_hours=input_hours,
                                           grid_shape=grid_shape,
                                           selected_features=selected_features)
    split_config = split_config or {}
    splits = split_sequences_by_target_time(full, **split_config)
    sample_balance_config = sample_balance_config or {
        "rain_threshold_mm": 0.1, "moderate_threshold_mm": 2.5,
        "near_heavy_threshold_mm": 7.5, "heavy_threshold_mm": 10.0,
        "event_neighbor_hours": 3,
        "keep_ratios": {"no_rain": .60, "light": .80, "moderate": 1., "near_heavy": 1., "heavy": 1., "event_neighbor": 1.},
        "seed": 42,
    }
    before = _summarize_regional_targets(splits["train"]["y_reg"], sample_balance_config)
    balance_dir = Path(sample_balance_dir or "sample_balance")
    balanced, manifest = apply_training_sample_balance(splits["train"], sample_balance_config, balance_dir)
    train_x, validation_x, test_x, scalers = normalize_regional_data(
        balanced["X_seq"], splits["validation"]["X_seq"], splits["test"]["X_seq"])
    thresholds = dict(sample_balance_config)
    return {
        "X_train": train_x, "y_train": balanced["y_reg"], "rain_seq_train": balanced["rain_seq_reg"], "times_train": balanced["valid_times"],
        "X_validation": validation_x, "y_validation": splits["validation"]["y_reg"], "rain_seq_validation": splits["validation"]["rain_seq_reg"], "times_validation": splits["validation"]["valid_times"],
        "X_test": test_x, "y_test": splits["test"]["y_reg"], "rain_seq_test": splits["test"]["rain_seq_reg"], "times_test": splits["test"]["valid_times"],
        "scalers": scalers, "grid_points_order": full["grid_points_order"], "feature_columns": full["feature_columns"],
        "thresholds_used": thresholds, "sample_balance": manifest,
        "train_selected_indices": np.load(balance_dir / "train_selected_indices.npy"),
        "train_stats_before_balance": before,
        "train_stats_after_balance": _summarize_regional_targets(balanced["y_reg"], thresholds),
        "validation_stats": _summarize_regional_targets(splits["validation"]["y_reg"], thresholds),
        "test_stats": _summarize_regional_targets(splits["test"]["y_reg"], thresholds),
    }

def select_model_input_channels(X_full, feature_names, model_id, physinput_config):
    from .config_loader import resolve_feature_columns
    required = resolve_feature_columns(model_id, physinput_config); feature_names = list(feature_names); missing = [name for name in required if name not in feature_names]
    if missing: raise ValueError(f"Input tensor is missing model channels: {missing}")
    indices = [feature_names.index(name) for name in required]; diagnostics = required[11:]
    return np.asarray(X_full)[..., indices], required, diagnostics
