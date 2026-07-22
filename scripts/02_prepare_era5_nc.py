#!/usr/bin/env python3
"""Merge and preprocess monthly ERA5 NetCDF files.

This script reproduces the confirmed NetCDF-stage workflow:

1. Read and merge monthly ERA5 single-level NetCDF files.
2. Normalize the time coordinate and optional ERA5 dimensions.
3. Validate the variables, hourly timestamps, and the 9 x 5 grid.
4. Calculate 2 m relative humidity from temperature and dewpoint.
5. Convert precipitation, temperature, dewpoint, and pressure units.
6. Calculate the original 24-hour PWV growth and decline indicators.
7. Crop the historical buffer and save one standardized NetCDF file.

The PWV calculations intentionally preserve the extrema-based method used in
the original data preparation code. They are not replaced by a simple
one-hour finite difference.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
from scipy.signal import argrelextrema


REQUIRED_VARIABLES = (
    "t2m",
    "d2m",
    "sp",
    "u10",
    "v10",
    "tp",
    "tcwv",
)

TIME_NAME = "valid_time"
LATITUDE_NAME = "latitude"
LONGITUDE_NAME = "longitude"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare standardized ERA5 NetCDF data for the study."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw/era5"),
        help="Directory containing monthly ERA5 NetCDF files.",
    )
    parser.add_argument(
        "--pattern",
        default="ERA5_southern_guangxi_*.nc",
        help="Glob pattern used to discover monthly files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path(
            "data/processed/era5_southern_guangxi_202008_202508.nc"
        ),
    )
    parser.add_argument(
        "--study-start",
        default="2020-08-01T00:00:00",
        help="First formal timestamp retained in the output.",
    )
    parser.add_argument(
        "--study-end",
        default="2025-08-31T23:00:00",
        help="Last formal timestamp retained in the output.",
    )
    parser.add_argument("--expected-latitudes", type=int, default=9)
    parser.add_argument("--expected-longitudes", type=int, default=5)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def discover_files(input_dir: Path, pattern: str) -> list[Path]:
    """Return sorted monthly NetCDF paths."""
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No NetCDF files matched '{pattern}' in {input_dir.resolve()}."
        )
    return files


def normalize_time_coordinate(ds: xr.Dataset) -> xr.Dataset:
    """Rename the detected time coordinate to 'valid_time'."""
    if TIME_NAME in ds.dims or TIME_NAME in ds.coords:
        return ds

    if "time" in ds.dims or "time" in ds.coords:
        return ds.rename({"time": TIME_NAME})

    raise ValueError("The dataset contains neither 'valid_time' nor 'time'.")


def collapse_singleton_number(ds: xr.Dataset) -> xr.Dataset:
    """Remove a singleton ERA5 ensemble-number dimension when present."""
    if "number" in ds.dims:
        if ds.sizes["number"] != 1:
            raise ValueError(
                f"Unexpected non-singleton 'number' dimension: {ds.sizes['number']}."
            )
        ds = ds.isel(number=0, drop=True)

    if "number" in ds.variables and "number" not in ds.dims:
        ds = ds.drop_vars("number")

    return ds


def collapse_expver(ds: xr.Dataset) -> xr.Dataset:
    """Combine non-missing values across the ERA5 'expver' dimension."""
    if "expver" not in ds.dims:
        if "expver" in ds.variables:
            ds = ds.drop_vars("expver")
        return ds

    expver_size = ds.sizes["expver"]
    rebuilt = xr.Dataset(coords={name: coord for name, coord in ds.coords.items() if name != "expver"})

    for name, data_array in ds.data_vars.items():
        if "expver" not in data_array.dims:
            rebuilt[name] = data_array
            continue

        combined = data_array.isel(expver=0, drop=True)
        for index in range(1, expver_size):
            combined = combined.combine_first(
                data_array.isel(expver=index, drop=True)
            )
        rebuilt[name] = combined

    rebuilt.attrs = dict(ds.attrs)
    return rebuilt


def normalize_monthly_dataset(path: Path) -> xr.Dataset:
    """Open, normalize, load, and close one monthly dataset."""
    logging.info("Reading %s", path)
    with xr.open_dataset(path) as source:
        ds = normalize_time_coordinate(source)
        ds = collapse_singleton_number(ds)
        ds = collapse_expver(ds)
        ds = ds.load()

    missing = [name for name in REQUIRED_VARIABLES if name not in ds.data_vars]
    if missing:
        raise ValueError(f"{path.name} is missing variables: {missing}")

    required_coords = (TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME)
    missing_coords = [name for name in required_coords if name not in ds.coords]
    if missing_coords:
        raise ValueError(f"{path.name} is missing coordinates: {missing_coords}")

    return ds[list(REQUIRED_VARIABLES)]


def assert_matching_grid(reference: xr.Dataset, candidate: xr.Dataset, path: Path) -> None:
    """Require identical latitude and longitude coordinates across files."""
    for coord_name in (LATITUDE_NAME, LONGITUDE_NAME):
        if not np.array_equal(
            reference[coord_name].values,
            candidate[coord_name].values,
        ):
            raise ValueError(
                f"Coordinate mismatch for '{coord_name}' in {path.name}."
            )


def merge_monthly_files(files: Iterable[Path]) -> xr.Dataset:
    """Load and concatenate all monthly files along valid_time."""
    datasets: list[xr.Dataset] = []
    reference: xr.Dataset | None = None

    for path in files:
        ds = normalize_monthly_dataset(path)
        if reference is None:
            reference = ds
        else:
            assert_matching_grid(reference, ds, path)
        datasets.append(ds)

    merged = xr.concat(
        datasets,
        dim=TIME_NAME,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="exact",
    )
    merged = merged.sortby(TIME_NAME)

    times = merged[TIME_NAME].values
    _, unique_indices = np.unique(times, return_index=True)
    if len(unique_indices) != len(times):
        logging.warning(
            "Removing %d duplicate timestamps.",
            len(times) - len(unique_indices),
        )
        merged = merged.isel({TIME_NAME: np.sort(unique_indices)})

    return merged


def validate_hourly_time_axis(ds: xr.Dataset) -> None:
    """Require a strictly continuous hourly time axis."""
    times = ds[TIME_NAME].values.astype("datetime64[h]")
    if len(times) < 2:
        raise ValueError("At least two timestamps are required.")

    differences = np.diff(times).astype("timedelta64[h]").astype(np.int64)
    invalid = np.flatnonzero(differences != 1)
    if invalid.size:
        index = int(invalid[0])
        raise ValueError(
            "The time axis is not hourly continuous between "
            f"{times[index]} and {times[index + 1]}."
        )


def validate_grid(ds: xr.Dataset, expected_latitudes: int, expected_longitudes: int) -> None:
    """Validate the fixed 9 x 5 regional grid."""
    n_lat = int(ds.sizes[LATITUDE_NAME])
    n_lon = int(ds.sizes[LONGITUDE_NAME])

    if n_lat != expected_latitudes or n_lon != expected_longitudes:
        raise ValueError(
            f"Expected a {expected_latitudes} x {expected_longitudes} grid, "
            f"but found {n_lat} x {n_lon}."
        )

    if n_lat * n_lon != 45:
        raise ValueError(f"Expected 45 grid cells, but found {n_lat * n_lon}.")


def calculate_relative_humidity(t2m_kelvin: xr.DataArray, d2m_kelvin: xr.DataArray) -> xr.DataArray:
    """Calculate 2 m relative humidity with the Magnus formulation."""
    temperature_celsius = t2m_kelvin - 273.15
    dewpoint_celsius = d2m_kelvin - 273.15

    saturation_vapour_pressure = 6.112 * np.exp(
        17.67 * temperature_celsius / (temperature_celsius + 243.5)
    )
    actual_vapour_pressure = 6.112 * np.exp(
        17.67 * dewpoint_celsius / (dewpoint_celsius + 243.5)
    )

    relative_humidity = 100.0 * (
        actual_vapour_pressure / saturation_vapour_pressure
    )
    relative_humidity = relative_humidity.clip(min=0.0, max=100.0)
    relative_humidity = relative_humidity.where(
        np.isfinite(relative_humidity)
    )
    relative_humidity.name = "rh2m"
    relative_humidity.attrs = {
        "long_name": "2 metre relative humidity",
        "units": "%",
        "derived_from": "t2m and d2m",
        "formula": (
            "RH = 100 * e/es using the Magnus formulation with "
            "temperature and dewpoint temperature"
        ),
    }
    return relative_humidity


def convert_units(ds: xr.Dataset) -> xr.Dataset:
    """Convert ERA5 variables to the units used by the model data."""
    converted = ds.copy()

    converted["tp"] = converted["tp"] * 1000.0
    converted["tp"].attrs.update(
        {
            "units": "mm",
            "long_name": "Total precipitation",
            "original_units": "m",
        }
    )

    converted["t2m"] = converted["t2m"] - 273.15
    converted["t2m"].attrs.update(
        {
            "units": "degree_Celsius",
            "long_name": "2 metre temperature",
            "original_units": "K",
        }
    )

    converted["d2m"] = converted["d2m"] - 273.15
    converted["d2m"].attrs.update(
        {
            "units": "degree_Celsius",
            "long_name": "2 metre dewpoint temperature",
            "original_units": "K",
        }
    )

    converted["sp"] = converted["sp"] / 100.0
    converted["sp"].attrs.update(
        {
            "units": "hPa",
            "long_name": "Surface pressure",
            "original_units": "Pa",
        }
    )

    return converted


def find_robust_extrema(
    pwv_window: np.ndarray,
) -> tuple[int, int, float, float]:
    """Find extrema using the rules from the original PWV processing code."""
    try:
        maximum_indices = argrelextrema(pwv_window, np.greater, order=1)[0]
        minimum_indices = argrelextrema(pwv_window, np.less, order=1)[0]
    except Exception:
        maximum_indices = np.array([], dtype=int)
        minimum_indices = np.array([], dtype=int)

    maximum_index = int(np.argmax(pwv_window))
    minimum_index = int(np.argmin(pwv_window))
    maximum_value = float(pwv_window[maximum_index])
    minimum_value = float(pwv_window[minimum_index])

    if len(maximum_indices) > 0:
        maximum_index = int(maximum_indices[-1])
        maximum_value = float(pwv_window[maximum_index])

        if len(minimum_indices) > 0:
            minima_before_maximum = minimum_indices[
                minimum_indices < maximum_index
            ]
            if len(minima_before_maximum) > 0:
                minimum_index = int(minima_before_maximum[-1])
                minimum_value = float(pwv_window[minimum_index])
            elif maximum_index > 0:
                preceding_values = pwv_window[:maximum_index]
                local_index = int(np.argmin(preceding_values))
                minimum_index = local_index
                minimum_value = float(preceding_values[local_index])
        elif maximum_index > 0:
            preceding_values = pwv_window[:maximum_index]
            local_index = int(np.argmin(preceding_values))
            minimum_index = local_index
            minimum_value = float(preceding_values[local_index])

    elif len(minimum_indices) > 0:
        minimum_index = int(minimum_indices[-1])
        minimum_value = float(pwv_window[minimum_index])

        if minimum_index < len(pwv_window) - 1:
            following_values = pwv_window[minimum_index + 1 :]
            local_index = int(np.argmax(following_values))
            maximum_index = minimum_index + 1 + local_index
            maximum_value = float(following_values[local_index])

    if minimum_index >= maximum_index:
        if pwv_window[0] <= pwv_window[-1]:
            minimum_index = 0
            minimum_value = float(pwv_window[0])
            maximum_index = len(pwv_window) - 1
            maximum_value = float(pwv_window[-1])
        else:
            maximum_index = 0
            maximum_value = float(pwv_window[0])
            minimum_index = len(pwv_window) - 1
            minimum_value = float(pwv_window[-1])

        if len(pwv_window) > 2:
            middle_values = pwv_window[1:-1]
            if pwv_window[0] <= pwv_window[-1]:
                local_index = int(np.argmin(middle_values))
                candidate_index = local_index + 1
                if pwv_window[candidate_index] < minimum_value:
                    minimum_index = candidate_index
                    minimum_value = float(pwv_window[candidate_index])
            else:
                local_index = int(np.argmax(middle_values))
                candidate_index = local_index + 1
                if pwv_window[candidate_index] > maximum_value:
                    maximum_index = candidate_index
                    maximum_value = float(pwv_window[candidate_index])

    return maximum_index, minimum_index, maximum_value, minimum_value


def calculate_pwv_rates_for_series(
    pwv_series: np.ndarray,
    window_hours: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the original PWV growth and decline indicators."""
    values = np.asarray(pwv_series, dtype=np.float64)
    growth = np.full(values.shape, np.nan, dtype=np.float32)
    decline = np.full(values.shape, np.nan, dtype=np.float32)

    for index in range(len(values)):
        if index < window_hours or np.isnan(values[index]):
            continue

        start_index = index - window_hours
        pwv_window = values[start_index:index]

        if len(pwv_window) != window_hours or np.isnan(pwv_window).any():
            continue

        (
            maximum_index,
            minimum_index,
            maximum_value,
            minimum_value,
        ) = find_robust_extrema(pwv_window)

        maximum_global_index = start_index + maximum_index
        minimum_global_index = start_index + minimum_index

        rise_time = index - minimum_global_index
        if rise_time > 0:
            growth[index] = (maximum_value - minimum_value) / rise_time

        fall_time = index - maximum_global_index
        if fall_time > 0:
            decline[index] = (values[index] - maximum_value) / fall_time

    return growth, decline


def add_pwv_rate_variables(ds: xr.Dataset, window_hours: int = 24) -> xr.Dataset:
    """Calculate PWV rates independently for every regional grid cell."""
    pwv = ds["tcwv"].transpose(
        TIME_NAME,
        LATITUDE_NAME,
        LONGITUDE_NAME,
    )
    values = np.asarray(pwv.values, dtype=np.float64)

    growth = np.full(values.shape, np.nan, dtype=np.float32)
    decline = np.full(values.shape, np.nan, dtype=np.float32)

    n_lat = values.shape[1]
    n_lon = values.shape[2]

    for latitude_index in range(n_lat):
        for longitude_index in range(n_lon):
            point_growth, point_decline = calculate_pwv_rates_for_series(
                values[:, latitude_index, longitude_index],
                window_hours=window_hours,
            )
            growth[:, latitude_index, longitude_index] = point_growth
            decline[:, latitude_index, longitude_index] = point_decline

    result = ds.copy()
    coordinates = {
        TIME_NAME: pwv[TIME_NAME],
        LATITUDE_NAME: pwv[LATITUDE_NAME],
        LONGITUDE_NAME: pwv[LONGITUDE_NAME],
    }

    result["pwv_growth_rate"] = xr.DataArray(
        growth,
        dims=(TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME),
        coords=coordinates,
        attrs={
            "long_name": "24-hour PWV growth indicator",
            "units": "kg m-2 h-1",
            "window_hours": window_hours,
            "method": "Extrema-based method preserved from the original workflow",
        },
    )
    result["pwv_decline_rate"] = xr.DataArray(
        decline,
        dims=(TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME),
        coords=coordinates,
        attrs={
            "long_name": "24-hour PWV decline indicator",
            "units": "kg m-2 h-1",
            "window_hours": window_hours,
            "method": "Extrema-based method preserved from the original workflow",
        },
    )

    return result


def build_quality_report(ds: xr.Dataset, source_files: list[Path]) -> dict:
    """Build a machine-readable data quality summary."""
    time_values = ds[TIME_NAME].values
    report = {
        "source_file_count": len(source_files),
        "source_files": [path.name for path in source_files],
        "time_start": str(time_values[0]),
        "time_end": str(time_values[-1]),
        "time_steps": int(ds.sizes[TIME_NAME]),
        "latitude_points": int(ds.sizes[LATITUDE_NAME]),
        "longitude_points": int(ds.sizes[LONGITUDE_NAME]),
        "grid_cells": int(
            ds.sizes[LATITUDE_NAME] * ds.sizes[LONGITUDE_NAME]
        ),
        "variables": list(ds.data_vars),
        "variable_statistics": {},
    }

    for name in ds.data_vars:
        data = np.asarray(ds[name].values)
        finite = data[np.isfinite(data)]
        report["variable_statistics"][name] = {
            "nan_count": int(np.isnan(data).sum()),
            "finite_count": int(finite.size),
            "minimum": float(np.min(finite)) if finite.size else None,
            "maximum": float(np.max(finite)) if finite.size else None,
            "mean": float(np.mean(finite)) if finite.size else None,
        }

    return report


def save_netcdf(ds: xr.Dataset, output_file: Path) -> None:
    """Save the processed dataset with lossless compression."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    encoding = {}
    for name, data_array in ds.data_vars.items():
        if data_array.dtype.kind in {"f", "c"}:
            encoding[name] = {
                "zlib": True,
                "complevel": 2,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
            }

    ds.to_netcdf(output_file, encoding=encoding)


def main() -> None:
    """Run the complete NetCDF preparation workflow."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    files = discover_files(args.input_dir, args.pattern)
    logging.info("Found %d monthly NetCDF files.", len(files))

    ds = merge_monthly_files(files)
    ds = ds.sortby(LATITUDE_NAME).sortby(LONGITUDE_NAME)
    validate_hourly_time_axis(ds)
    validate_grid(
        ds,
        expected_latitudes=args.expected_latitudes,
        expected_longitudes=args.expected_longitudes,
    )

    logging.info("Calculating 2 m relative humidity.")
    ds["rh2m"] = calculate_relative_humidity(ds["t2m"], ds["d2m"])

    logging.info("Converting physical units.")
    ds = convert_units(ds)

    logging.info("Calculating 24-hour PWV growth and decline indicators.")
    ds = add_pwv_rate_variables(ds, window_hours=24)

    logging.info(
        "Cropping the historical buffer and retaining %s to %s.",
        args.study_start,
        args.study_end,
    )
    ds = ds.sel(
        {
            TIME_NAME: slice(
                np.datetime64(args.study_start),
                np.datetime64(args.study_end),
            )
        }
    )

    if ds.sizes[TIME_NAME] == 0:
        raise ValueError("The requested formal study interval is empty.")

    validate_hourly_time_axis(ds)
    validate_grid(
        ds,
        expected_latitudes=args.expected_latitudes,
        expected_longitudes=args.expected_longitudes,
    )

    ds.attrs.update(
        {
            "title": "Processed ERA5 data for regional rainfall-field forecasting",
            "source_dataset": "reanalysis-era5-single-levels",
            "study_start": args.study_start,
            "study_end": args.study_end,
            "grid_description": (
                f"{args.expected_latitudes} latitude points x "
                f"{args.expected_longitudes} longitude points"
            ),
            "pwv_rate_window_hours": 24,
        }
    )

    save_netcdf(ds, args.output_file)

    report = build_quality_report(ds, files)
    report_path = args.output_file.with_suffix(".quality.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("Processed NetCDF saved to %s", args.output_file)
    logging.info("Quality report saved to %s", report_path)


if __name__ == "__main__":
    main()
