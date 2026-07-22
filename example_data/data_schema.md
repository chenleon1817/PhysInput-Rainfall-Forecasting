# Example Data Schema

The example files contain one continuous complete 9 x 5 regional window from each available year.

## Files

| Year | File | Start | End | Hours | Rows | Peak rainfall |
|---:|---|---|---|---:|---:|---:|
| 2020 | `sample_data_2020.csv` | `2020-09-17T19:00:00` | `2020-09-20T18:00:00` | 72 | 3240 | 14.283180 |
| 2021 | `sample_data_2021.csv` | `2021-04-24T21:00:00` | `2021-04-27T20:00:00` | 72 | 3240 | 27.150154 |
| 2022 | `sample_data_2022.csv` | `2022-08-24T01:00:00` | `2022-08-27T00:00:00` | 72 | 3240 | 26.567460 |
| 2023 | `sample_data_2023.csv` | `2023-10-17T23:00:00` | `2023-10-20T22:00:00` | 72 | 3240 | 28.627396 |
| 2024 | `sample_data_2024.csv` | `2024-09-05T10:00:00` | `2024-09-08T09:00:00` | 72 | 3240 | 24.735450 |
| 2025 | `sample_data_2025.csv` | `2025-06-01T05:00:00` | `2025-06-04T04:00:00` | 72 | 3240 | 28.480530 |

## Columns

| Column | Data type |
|---|---|
| `valid_time` | `datetime64[ns]` |
| `year` | `int64` |
| `month` | `int64` |
| `day` | `int64` |
| `hour` | `int64` |
| `longitude` | `float64` |
| `latitude` | `float64` |
| `temperature_c` | `float64` |
| `u_wind_m_s` | `float64` |
| `v_wind_m_s` | `float64` |
| `surface_pressure_hpa` | `float64` |
| `relative_humidity_pct` | `float64` |
| `precipitation_mm` | `float64` |
| `pwv_kg_m2` | `float64` |
| `pwv_growth_rate` | `float64` |
| `pwv_decline_rate` | `float64` |

## Integrity requirements

- Every timestamp contains exactly 45 unique grid cells.
- Timestamps are continuous at an hourly interval within each file.
- Rows are sorted by time, latitude, and longitude.
- Values are copied from the complete processed annual datasets.
- These samples are intended for examples and lightweight tests.
- These samples are not sufficient for formal feature analysis, model training, or paper-result reproduction.
