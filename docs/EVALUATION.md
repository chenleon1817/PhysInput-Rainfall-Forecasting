# Evaluation

Evaluation flattens all sample-cell values and clips predictions to non-negative values. Continuous metrics are MAE, RMSE, R2, Bias, and correlation. For an event threshold, POD is hits/(hits+misses), FAR is false alarms/(hits+false alarms), CSI is hits/(hits+misses+false alarms), and FBIAS is predicted events/observed events. Thresholds are 0.1, 2.5, and 10.0 mm h-1. By-seed, mean/std, and long-prediction outputs are written by `scripts/06_evaluate_models.py`.

Training writes each model's `test_predictions.npz`; evaluation reads those archives directly using `--training-run-dir` and does not retrain models.
