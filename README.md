# PhysInput-Rainfall-Forecasting

This repository provides a modular TensorFlow implementation and selected archived experimental results for regional hourly rainfall-field forecasting with physically motivated diagnostic input augmentation.

The study uses 12 preceding hours of ERA5 meteorological information to predict next-hour rainfall over a 9 × 5 grid in southern Guangxi, China.

Four diagnostic inputs are considered:

- q*: near-surface moisture proxy
- G: precipitable water vapour (PWV) gradient magnitude
- C: 10 m horizontal wind convergence
- A: PWV advection proxy

Five configurations are evaluated using a shared CNN–BiLSTM architecture: CNN-BiLSTM, PhysInput-Lite, PhysInput-Adv3, PhysInput-Conv3, and PhysInput-All4. Additional reference models are included for comparison.

The study focuses on configuration-dependent forecasting performance, including continuous rainfall prediction errors and heavy-rainfall detection trade-offs.

The repository contains the modular implementation, lightweight example data, selected reported results, and validation documentation. The complete ERA5 datasets, original model checkpoints, and full formal prediction archives are not included.

The publicly available implementation has undergone selected unit and structural tests. Complete behavioral equivalence with the original formal training workflow has not yet been established.

## Workflow

The modular workflow supports nine forecasting configurations using the preceding 12 hourly time steps to predict the next-hour rainfall field over a 9 × 5 grid.

The formal model set is Linear-ARX, BiLSTM, ConvLSTM,
CNN-BiLSTM-Attention, CNN-BiLSTM, PhysInput-Lite, PhysInput-Adv3,
PhysInput-Conv3, and PhysInput-All4. `example_data/` supports interface and
structural checks only. `reported_results/` archives lightweight outputs from
the original formal experiment; complete ERA5 data, model checkpoints, and
full prediction tables are intentionally not uploaded.

Install dependencies with `pip install -r requirements.txt`. Prepare ERA5, export CSV data, run feature analysis, train, evaluate, then create paper figures:

TensorFlow is required for model construction, training, evaluation, and
TensorFlow-dependent tests. The exact GPU or CPU backend depends on the user's
environment. Full ERA5 data and model checkpoints are not included.

```powershell
python scripts/01_download_era5.py
python scripts/02_prepare_era5_nc.py
python scripts/03_export_model_csv.py
python scripts/04_run_feature_analysis.py
python scripts/05_train_models.py --models all --seed 42
python scripts/06_evaluate_models.py --training-run-dir outputs/training/<run_id> --output-dir outputs/evaluation/<evaluation_id>
python scripts/07_make_paper_figures.py --evaluation-dir outputs/evaluation/<run> --output-dir outputs/paper_figures --seed 42
```

For a structurally identical two-epoch debug run, use a selected model and an explicit output directory:

```powershell
python scripts/05_train_models.py --debug --models cnn_bilstm --output-dir outputs/training/debug_example
```

Run repository checks with `python -m compileall core scripts tests` and `pytest -q`.

The modular repository is the only supported execution path. Original large reference files are intentionally not included in the public repository. The lightweight files in `example_data/` support structural checks only; complete formal data are required to reproduce paper numbers. See `docs/` for data, model, training, evaluation, figure, and reproducibility details. Equivalence status is reported candidly in `docs/EQUIVALENCE_VALIDATION.md`; do not treat an untested stage as validated. 

## Figure Generation Scope

The public script `scripts/07_make_paper_figures.py` produces historical plotting layouts rather than the complete revised manuscript figure set.

See `docs/PAPER_FIGURES.md` for details.

## Implementation and Reproducibility Scope

The public modular preprocessing workflow has not yet been fully reconciled with the historical formal training-input construction, including the dPWV input definitions.

The archived formal results and the public implementation have different validation scopes. Passing repository unit tests does not establish complete end-to-end reproduction of the manuscript results.

See `docs/REPRODUCIBILITY.md` and `docs/EQUIVALENCE_VALIDATION.md` for further details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
