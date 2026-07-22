# Data

The formal period is August 2020 through August 2025 over a latitude-major 9 x 5 grid (45 cells). Each sample contains the preceding 12 hourly fields and predicts the next-hour rainfall field. Raw data, NetCDF files, and converted full CSV files are intentionally excluded from version control.

`dPWV+` and `dPWV-` are produced during NetCDF preparation with the original 24-hour extrema-based method. CSV export stores them in `V13013` and `V13014`; model-stage code reads these values and never recomputes them from one-hour differences.
