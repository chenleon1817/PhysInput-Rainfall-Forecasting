# Data

The formal period is August 2020 through August 2025 over a latitude-major 9 x 5 grid (45 cells). Each sample contains the preceding 12 hourly fields and predicts the next-hour rainfall field. Raw data, NetCDF files, and converted full CSV files are intentionally excluded from version control.

`dPWV+` and `dPWV-` are produced during NetCDF preparation with the original 24-hour extrema-based method. CSV export stores them in `V13013` and `V13014`; model-stage code reads these values and never recomputes them from one-hour differences.

## Input-definition reconciliation

The current public modular NetCDF preparation workflow computes 24-hour extrema-based PWV growth and decline indicators, which are subsequently mapped to the model-input fields.

The revised manuscript defines the formal training predictors dPWV+ and dPWV- using positive and negative changes between consecutive hourly PWV values.

These are different feature definitions. The public preprocessing path should not be presented as behaviorally equivalent to the original formal training-input construction until the historical implementation and input provenance have been reconciled.

The historical exploratory feature-analysis implementation is retained for provenance. This documentation note does not modify the original reported experimental results.
