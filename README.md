# PhysInput-Rainfall-Forecasting

This repository accompanies a Pattern Analysis and Applications submission on next-hour rainfall-field prediction over a 9 x 5 grid in southern Guangxi, China. It uses 12 preceding hourly ERA5 regional fields to predict rainfall at all 45 cells.

## Workflow

This repository trains nine formal models from 12 hourly input fields to predict the next hour over a 9 x 5 grid. The formal split is train through 2024-06-30, validation from 2024-07-01 through 2024-12-31, and test from 2025-01-01 onward.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
