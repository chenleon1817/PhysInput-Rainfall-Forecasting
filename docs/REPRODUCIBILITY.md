# Reproducibility

## Current validation boundary

The repository configures deterministic TensorFlow settings before model
imports and resets state before each deep model. Deterministic configuration
does not substitute for the unavailable formal data, exact environment lock,
or required old-versus-new regression comparisons.
Use Python with the packages in `requirements.txt`, `configs/experiment_config.json`, and `configs/physinput_model_config.json`. The workflow uses deterministic seed handling, 12-hour input, one-hour horizon, the fixed 9 x 5 grid, and the published chronological split. Complete formal data are required to reproduce paper metrics; example data support only lightweight validation.
