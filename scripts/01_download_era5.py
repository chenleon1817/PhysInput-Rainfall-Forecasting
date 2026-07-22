#!/usr/bin/env python3
"""Download monthly ERA5 single-level data for the study domain.

The script downloads the variables used by the rainfall-field forecasting
workflow. It also supports a configurable pre-study buffer so that derived
features requiring historical context can be calculated for the first formal
study timestamp.

Before running this script:
1. Install cdsapi.
2. Configure the CDS credentials in ~/.cdsapirc.
3. Accept the ERA5 dataset licence in the Climate Data Store.
"""

from __future__ import annotations

import argparse
import calendar
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable



DATASET_NAME = "reanalysis-era5-single-levels"

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "surface_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "total_column_water_vapour",
]

# CDS area order: north, west, south, east.
DEFAULT_AREA = (23.0, 108.0, 21.0, 109.0)


@dataclass(frozen=True)
class MonthlyRequest:
    """A single monthly CDS request."""

    year: int
    month: int
    days: tuple[str, ...]


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-like date or datetime string."""
    normalized = value.strip().replace("Z", "")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{value}'. Use YYYY-MM-DD or YYYY-MM-DDTHH:MM."
        ) from exc


def month_start(value: datetime) -> datetime:
    """Return the first day of the month containing *value*."""
    return datetime(value.year, value.month, 1)


def next_month(value: datetime) -> datetime:
    """Return the first day of the month after *value*."""
    if value.month == 12:
        return datetime(value.year + 1, 1, 1)
    return datetime(value.year, value.month + 1, 1)


def iter_monthly_requests(start: datetime, end: datetime) -> Iterable[MonthlyRequest]:
    """Yield the exact day list needed for each month in a datetime interval."""
    cursor = month_start(start)

    while cursor <= end:
        month_end_day = calendar.monthrange(cursor.year, cursor.month)[1]
        first_day = start.day if (cursor.year, cursor.month) == (start.year, start.month) else 1
        last_day = end.day if (cursor.year, cursor.month) == (end.year, end.month) else month_end_day

        days = tuple(f"{day:02d}" for day in range(first_day, last_day + 1))
        yield MonthlyRequest(cursor.year, cursor.month, days)
        cursor = next_month(cursor)


def build_request(item: MonthlyRequest, area: tuple[float, float, float, float]) -> dict:
    """Build a CDS API request for one month."""
    return {
        "product_type": ["reanalysis"],
        "variable": ERA5_VARIABLES,
        "year": [f"{item.year:04d}"],
        "month": [f"{item.month:02d}"],
        "day": list(item.days),
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": list(area),
    }


def validate_download(path: Path) -> None:
    """Raise an error when a downloaded file is missing or empty."""
    if not path.exists():
        raise RuntimeError(f"The CDS request completed but no file was created: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"The downloaded file is empty: {path}")


def download_month(
    client,
    item: MonthlyRequest,
    output_dir: Path,
    area: tuple[float, float, float, float],
    retries: int,
    retry_wait_seconds: int,
) -> dict:
    """Download one monthly NetCDF file and return a manifest record."""
    output_name = f"ERA5_southern_guangxi_{item.year:04d}_{item.month:02d}.nc"
    output_path = output_dir / output_name

    if output_path.exists() and output_path.stat().st_size > 0:
        logging.info("Skipping existing file: %s", output_path)
        return {
            "file": output_name,
            "status": "skipped_existing",
            "size_bytes": output_path.stat().st_size,
            "days": list(item.days),
        }

    request = build_request(item, area)
    temporary_path = output_path.with_suffix(".nc.part")

    for attempt in range(1, retries + 2):
        try:
            if temporary_path.exists():
                temporary_path.unlink()

            logging.info(
                "Downloading %04d-%02d, days %s to %s",
                item.year,
                item.month,
                ",".join(item.days),
                output_path,
            )
            client.retrieve(DATASET_NAME, request, str(temporary_path))
            validate_download(temporary_path)
            temporary_path.replace(output_path)
            validate_download(output_path)

            logging.info(
                "Downloaded %s (%.2f MB)",
                output_path.name,
                output_path.stat().st_size / (1024**2),
            )
            return {
                "file": output_name,
                "status": "downloaded",
                "size_bytes": output_path.stat().st_size,
                "days": list(item.days),
            }

        except Exception as exc:
            logging.exception(
                "Download attempt %d failed for %04d-%02d",
                attempt,
                item.year,
                item.month,
            )
            if temporary_path.exists():
                temporary_path.unlink()

            if attempt > retries:
                return {
                    "file": output_name,
                    "status": "failed",
                    "error": str(exc),
                    "days": list(item.days),
                }

            time.sleep(retry_wait_seconds)

    raise RuntimeError("Unreachable download state.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download monthly ERA5 single-level NetCDF files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/era5"),
        help="Directory used for downloaded monthly NetCDF files.",
    )
    parser.add_argument(
        "--study-start",
        type=parse_datetime,
        default=datetime(2020, 8, 1, 0, 0),
        help="First formal study timestamp.",
    )
    parser.add_argument(
        "--study-end",
        type=parse_datetime,
        default=datetime(2025, 8, 31, 23, 0),
        help="Last formal study timestamp.",
    )
    parser.add_argument(
        "--history-buffer-hours",
        type=int,
        default=24,
        help="Hours downloaded before the formal study start.",
    )
    parser.add_argument(
        "--area",
        type=float,
        nargs=4,
        metavar=("NORTH", "WEST", "SOUTH", "EAST"),
        default=DEFAULT_AREA,
        help="CDS area in north west south east order.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-wait-seconds", type=int, default=20)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ERA5 download workflow."""
    args = parse_args()

    if args.study_end < args.study_start:
        raise ValueError("--study-end must not be earlier than --study-start.")
    if args.history_buffer_hours < 0:
        raise ValueError("--history-buffer-hours must be non-negative.")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actual_start = args.study_start - timedelta(hours=args.history_buffer_hours)
    area = tuple(float(value) for value in args.area)

    logging.info("Dataset: %s", DATASET_NAME)
    logging.info("Formal study interval: %s to %s", args.study_start, args.study_end)
    logging.info("Actual download interval: %s to %s", actual_start, args.study_end)
    logging.info("Area [N, W, S, E]: %s", area)
    logging.info("Variables: %s", ", ".join(ERA5_VARIABLES))

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "The cdsapi package is required for downloading ERA5 data. "
            "Install it with: pip install \"cdsapi>=0.7.7\""
        ) from exc

    client = cdsapi.Client()
    records = []

    for item in iter_monthly_requests(actual_start, args.study_end):
        records.append(
            download_month(
                client=client,
                item=item,
                output_dir=args.output_dir,
                area=area,
                retries=args.retries,
                retry_wait_seconds=args.retry_wait_seconds,
            )
        )

    manifest = {
        "dataset": DATASET_NAME,
        "variables": ERA5_VARIABLES,
        "area_north_west_south_east": list(area),
        "study_start": args.study_start.isoformat(),
        "study_end": args.study_end.isoformat(),
        "history_buffer_hours": args.history_buffer_hours,
        "actual_download_start": actual_start.isoformat(),
        "files": records,
    }

    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    failed = [record for record in records if record["status"] == "failed"]
    if failed:
        raise RuntimeError(
            f"{len(failed)} monthly download(s) failed. See {manifest_path}."
        )

    logging.info("Download workflow completed. Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
