#!/usr/bin/env python3
"""Export the processed ERA5 NetCDF dataset to annual model CSV files.

The script combines the former NetCDF-to-CSV conversion and column-mapping
steps. It reads one standardized NetCDF file, validates the 45-cell hourly
grid, and writes one CSV file per year.

Two naming modes are supported:
- legacy: reproduces the Vxxxxx column names used by the original model data.
- semantic: writes descriptive English column names.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


TIME_NAME = "valid_time"
LATITUDE_NAME = "latitude"
LONGITUDE_NAME = "longitude"

REQUIRED_VARIABLES = (
    "t2m",
    "u10",
    "v10",
    "sp",
    "rh2m",
    "tp",
    "tcwv",
    "pwv_growth_rate",
    "pwv_decline_rate",
)

LEGACY_MAPPING = {
    LONGITUDE_NAME: "V05001",
    LATITUDE_NAME: "V05002",
    "t2m": "V11001",
    "u10": "V11002",
    "v10": "V11003",
    "sp": "V07004",
    "rh2m": "V13010",
    "tp": "V13011",
    "tcwv": "V13012",
    "pwv_growth_rate": "V13013",
    "pwv_decline_rate": "V13014",
}

SEMANTIC_MAPPING = {
    LONGITUDE_NAME: "longitude",
    LATITUDE_NAME: "latitude",
    "t2m": "temperature_c",
    "u10": "u_wind_m_s",
    "v10": "v_wind_m_s",
    "sp": "surface_pressure_hpa",
    "rh2m": "relative_humidity_pct",
    "tp": "precipitation_mm",
    "tcwv": "pwv_kg_m2",
    "pwv_growth_rate": "dPWV_plus",
    "pwv_decline_rate": "dPWV_minus",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export annual model CSV files from processed ERA5 NetCDF."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path(
            "data/processed/era5_southern_guangxi_202008_202508.nc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("converted_data"),
    )
    parser.add_argument(
        "--naming",
        choices=("legacy", "semantic"),
        default="legacy",
        help="Column naming convention used in the output CSV files.",
    )
    parser.add_argument(
        "--float-format",
        default="%.6f",
        help="Floating-point format passed to pandas.to_csv.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def validate_dataset(ds: xr.Dataset) -> None:
    """Validate the required coordinates, variables, and fixed grid."""
    missing_coords = [
        name
        for name in (TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME)
        if name not in ds.coords
    ]
    if missing_coords:
        raise ValueError(f"Missing coordinates: {missing_coords}")

    missing_variables = [
        name for name in REQUIRED_VARIABLES if name not in ds.data_vars
    ]
    if missing_variables:
        raise ValueError(f"Missing data variables: {missing_variables}")

    n_lat = int(ds.sizes[LATITUDE_NAME])
    n_lon = int(ds.sizes[LONGITUDE_NAME])
    if (n_lat, n_lon) != (9, 5):
        raise ValueError(f"Expected a 9 x 5 grid, found {n_lat} x {n_lon}.")


def validate_long_table(frame: pd.DataFrame) -> None:
    """Require exactly 45 unique grid rows for every timestamp."""
    counts = frame.groupby(TIME_NAME, sort=False).size()
    invalid = counts[counts != 45]
    if not invalid.empty:
        first_time = invalid.index[0]
        raise ValueError(
            f"Expected 45 rows at every timestamp. "
            f"{first_time} contains {int(invalid.iloc[0])} rows."
        )

    duplicates = frame.duplicated(
        subset=[TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME]
    )
    if duplicates.any():
        raise ValueError(
            f"Found {int(duplicates.sum())} duplicate time-grid rows."
        )


def add_calendar_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, day, and hour columns from valid_time."""
    result = frame.copy()
    result[TIME_NAME] = pd.to_datetime(result[TIME_NAME])
    result["year"] = result[TIME_NAME].dt.year.astype(np.int16)
    result["month"] = result[TIME_NAME].dt.month.astype(np.int8)
    result["day"] = result[TIME_NAME].dt.day.astype(np.int8)
    result["hour"] = result[TIME_NAME].dt.hour.astype(np.int8)
    return result


def apply_naming(frame: pd.DataFrame, naming: str) -> pd.DataFrame:
    """Apply the selected public or legacy model column names."""
    if naming == "legacy":
        renamed = frame.rename(columns=LEGACY_MAPPING)
        renamed = renamed.rename(
            columns={
                "year": "V04001",
                "month": "V04002",
                "day": "V04003",
                "hour": "V04004",
            }
        )
        columns = [
            "V04001",
            "V04002",
            "V04003",
            "V04004",
            "V05001",
            "V05002",
            "V11001",
            "V11002",
            "V11003",
            "V07004",
            "V13010",
            "V13011",
            "V13012",
            "V13013",
            "V13014",
        ]
        return renamed[columns]

    renamed = frame.rename(columns=SEMANTIC_MAPPING)
    columns = [
        "year",
        "month",
        "day",
        "hour",
        "longitude",
        "latitude",
        "temperature_c",
        "u_wind_m_s",
        "v_wind_m_s",
        "surface_pressure_hpa",
        "relative_humidity_pct",
        "precipitation_mm",
        "pwv_kg_m2",
        "dPWV_plus",
        "dPWV_minus",
    ]
    return renamed[columns]


def export_year(
    ds: xr.Dataset,
    year: int,
    output_dir: Path,
    naming: str,
    float_format: str,
) -> dict:
    """Export one calendar year and return a manifest record."""
    yearly = ds.sel({TIME_NAME: slice(f"{year}-01-01", f"{year}-12-31T23:59:59")})
    if yearly.sizes.get(TIME_NAME, 0) == 0:
        return {"year": year, "status": "empty"}

    frame = (
        yearly[list(REQUIRED_VARIABLES)]
        .to_dataframe()
        .reset_index()
    )
    frame = frame[
        [TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME, *REQUIRED_VARIABLES]
    ]
    frame = frame.sort_values(
        [TIME_NAME, LATITUDE_NAME, LONGITUDE_NAME],
        kind="mergesort",
    ).reset_index(drop=True)

    validate_long_table(frame)
    frame = add_calendar_columns(frame)
    output_frame = apply_naming(frame, naming=naming)

    output_path = output_dir / (f"data_{year}.csv" if naming == "legacy" else f"data_{year}_semantic.csv")
    output_frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
        float_format=float_format,
    )

    logging.info(
        "Saved %s with %d rows and %d columns.",
        output_path,
        len(output_frame),
        len(output_frame.columns),
    )
    return {
        "year": year,
        "status": "written",
        "file": output_path.name,
        "rows": int(len(output_frame)),
        "columns": list(output_frame.columns),
    }


def main() -> None:
    """Run the annual CSV export workflow."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.input_file.exists():
        raise FileNotFoundError(args.input_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(args.input_file) as source:
        ds = source.load()

    validate_dataset(ds)
    ds = ds.sortby(TIME_NAME).sortby(LATITUDE_NAME).sortby(LONGITUDE_NAME)

    years = sorted(
        np.unique(pd.DatetimeIndex(ds[TIME_NAME].values).year).tolist()
    )

    records = [
        export_year(
            ds=ds,
            year=int(year),
            output_dir=args.output_dir,
            naming=args.naming,
            float_format=args.float_format,
        )
        for year in years
    ]

    manifest = {
        "source_netcdf": str(args.input_file),
        "naming": args.naming,
        "grid_cells_per_hour": 45,
        "files": records,
    }
    manifest_path = args.output_dir / f"csv_export_manifest_{args.naming}.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logging.info("CSV export completed. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
