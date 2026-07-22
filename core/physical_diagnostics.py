"""Formula-preserving physical diagnostic calculations on raw fields."""

from __future__ import annotations

import numpy as np
import pandas as pd

def compute_specific_humidity_proxy(temperature, pressure, relative_humidity, eps: float = 1e-6):
    """Return the original unit-adaptive specific-humidity proxy."""
    temperature = pd.Series(temperature, dtype=float)
    pressure = pd.Series(pressure, dtype=float)
    relative_humidity = pd.Series(relative_humidity, dtype=float)
    temp_c = np.where(temperature.median() > 100.0, temperature - 273.15, temperature)
    pressure_hpa = np.where(pressure.median() > 2000.0, pressure / 100.0, pressure)
    rh_percent = np.clip(np.where(relative_humidity.median() <= 1.5, relative_humidity * 100.0, relative_humidity), 0.0, 100.0)
    saturation = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5 + eps))
    vapour = (rh_percent / 100.0) * saturation
    return 0.622 * vapour / (pressure_hpa - 0.378 * vapour + eps)

def _spatial_derivatives(frame: pd.DataFrame, time_col: str, lat_col: str, lon_col: str, eps: float):
    required = [time_col, lat_col, lon_col, "PWV", "U", "V"]
    missing = [column for column in required if column not in frame]
    if missing: raise ValueError(f"Missing columns for spatial diagnostics: {missing}")
    work = frame.copy(); work[time_col] = pd.to_datetime(work[time_col]); work = work.sort_values([time_col, lat_col, lon_col]).reset_index(drop=True)
    lats = np.sort(work[lat_col].astype(float).unique()); lons = np.sort(work[lon_col].astype(float).unique()); times = pd.Index(work[time_col].drop_duplicates().sort_values())
    if len(lats) < 2 or len(lons) < 2: raise ValueError("At least a 2 x 2 grid is required for spatial diagnostics.")
    full_index = pd.MultiIndex.from_product([times, lats, lons], names=[time_col, lat_col, lon_col])
    indexed = work.set_index([time_col, lat_col, lon_col]).sort_index()
    if indexed.index.has_duplicates: raise ValueError("Duplicate time-grid rows detected.")
    cube = indexed[["PWV", "U", "V"]].reindex(full_index)
    if cube.isna().any().any(): raise ValueError("Incomplete time-grid fields cannot be used for spatial diagnostics.")
    shape = (len(times), len(lats), len(lons)); pwv, u, v = (cube[column].to_numpy(dtype=np.float64).reshape(shape) for column in ("PWV", "U", "V"))
    radius = 6_371_000.0; lat_rad = np.deg2rad(lats); lon_rad = np.deg2rad(lons); cos_lat = np.clip(np.cos(lat_rad), eps, None)[None, :, None]
    dp_dphi = np.gradient(pwv, lat_rad, axis=1, edge_order=1); dp_dlambda = np.gradient(pwv, lon_rad, axis=2, edge_order=1)
    du_dlambda = np.gradient(u, lon_rad, axis=2, edge_order=1); dv_dphi = np.gradient(v, lat_rad, axis=1, edge_order=1)
    dp_dy = dp_dphi / radius; dp_dx = dp_dlambda / (radius * cos_lat); du_dx = du_dlambda / (radius * cos_lat); dv_dy = dv_dphi / radius
    return indexed, full_index, np.sqrt(dp_dx**2 + dp_dy**2), -(du_dx + dv_dy - v * np.tan(lat_rad)[None, :, None] / radius), -(u * dp_dx + v * dp_dy)

def compute_pwv_gradient_magnitude(frame: pd.DataFrame, **kwargs) -> np.ndarray:
    return _spatial_derivatives(frame, kwargs.get("time_col", "datetime_utc"), kwargs.get("lat_col", "V05002"), kwargs.get("lon_col", "V05001"), kwargs.get("eps", 1e-6))[2]

def compute_wind_convergence(frame: pd.DataFrame, **kwargs) -> np.ndarray:
    return _spatial_derivatives(frame, kwargs.get("time_col", "datetime_utc"), kwargs.get("lat_col", "V05002"), kwargs.get("lon_col", "V05001"), kwargs.get("eps", 1e-6))[3]

def compute_moisture_advection(frame: pd.DataFrame, **kwargs) -> np.ndarray:
    return _spatial_derivatives(frame, kwargs.get("time_col", "datetime_utc"), kwargs.get("lat_col", "V05002"), kwargs.get("lon_col", "V05001"), kwargs.get("eps", 1e-6))[4]

def add_spatial_physical_diagnostics(frame: pd.DataFrame, time_col="datetime_utc", lat_col="V05002", lon_col="V05001", eps=1e-6) -> pd.DataFrame:
    indexed, full_index, gradient, convergence, advection = _spatial_derivatives(frame, time_col, lat_col, lon_col, eps)
    diagnostics = pd.DataFrame({"pwv_grad_mag": gradient.reshape(-1), "wind_convergence": convergence.reshape(-1), "moisture_advection": advection.reshape(-1)}, index=full_index)
    for column in diagnostics:
        indexed[column] = diagnostics.loc[indexed.index, column].to_numpy(dtype=np.float32)
    result = indexed.reset_index()
    if not np.isfinite(result[["pwv_grad_mag", "wind_convergence", "moisture_advection"]].to_numpy(dtype=float)).all(): raise ValueError("Spatial diagnostics contain non-finite values.")
    return result

def add_physical_diagnostic_features(frame: pd.DataFrame, time_col="datetime_utc", lat_col="V05002", lon_col="V05001", eps=1e-6, use_spatial_diagnostics=True) -> pd.DataFrame:
    """Append the four formal downstream physical diagnostics."""
    required = ["PWV", "RH", "T", "P", "U", "V", "dPWV+", "dPWV-", time_col, lat_col, lon_col]
    missing = [column for column in required if column not in frame]
    if missing: raise ValueError(f"Missing columns for physical diagnostics: {missing}")
    result = frame.copy().sort_values([time_col, lat_col, lon_col]).reset_index(drop=True)
    result["specific_humidity_proxy"] = compute_specific_humidity_proxy(result["T"], result["P"], result["RH"], eps)
    if use_spatial_diagnostics: return add_spatial_physical_diagnostics(result, time_col, lat_col, lon_col, eps)
    for column in ("pwv_grad_mag", "wind_convergence", "moisture_advection"): result[column] = 0.0
    if not np.isfinite(result[["specific_humidity_proxy", "pwv_grad_mag", "wind_convergence", "moisture_advection"]].to_numpy(dtype=float)).all(): raise ValueError("Physical diagnostics contain non-finite values.")
    return result
