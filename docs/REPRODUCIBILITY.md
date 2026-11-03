# Reproducibility

## Current validation boundary

The repository configures deterministic TensorFlow settings before model
imports and resets state before each deep model. Deterministic configuration
does not substitute for the unavailable formal data, exact environment lock,
or required old-versus-new regression comparisons.
Use Python with the packages in `requirements.txt`, `configs/experiment_config.json`, and `configs/physinput_model_config.json`. The workflow uses deterministic seed handling, 12-hour input, one-hour horizon, the fixed 9 x 5 grid, and the published chronological split. Complete formal data are required to reproduce paper metrics; example data support only lightweight validation.

## Relationship to the Formal Experiments

The reported manuscript results originate from the archived formal experiments conducted using ERA5 hourly reanalysis data and a fixed chronological data split.

The public repository provides a modular implementation, experiment configurations, lightweight test data, and selected archived results. The complete ERA5 datasets, original model checkpoints, and full prediction archives are not publicly included.

Selected unit tests verify diagnostic calculations, data alignment, sampling behavior, and evaluation metrics. These tests do not establish complete behavioral equivalence with the original training workflow.

The archived results have undergone internal consistency and paired evaluation checks. Exact end-to-end reproduction from the publicly available repository remains dependent on the original data, preprocessing artifacts, and historical experimental environment.
