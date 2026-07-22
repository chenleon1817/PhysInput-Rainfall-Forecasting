# Formal Logic Restoration

This repair restores only the final formal protocol from the archived mixed
sources; retired historical model routes remain excluded.

| Original source function or behavior | Destination | Restoration |
| --- | --- | --- |
| `grid_level_intensity_weighted_huber_loss` | `core/models.py` | Per-cell intensity-weighted regression loss |
| `build_event_aware_five_class_keep_indices` | `core/data_processing.py` | Training-only monthly sampling |
| `append_lagged_rainfall_channel_train_val_test` | `core/data_processing.py` | Three-split rainfall scaling and leakage checks |
| `build_intensity_sample_weight` | `core/training_utils.py` | Four-class sample weights with statistics |
| `ValidationEventScoreCheckpointMetric` | `core/training_utils.py` | Validation heavy-rain checkpoint metric |
| `train_regional_model_variant` | `core/training_utils.py` | Best-event checkpoint training protocol |
| training result exports | `core/result_export.py`, `scripts/05_train_models.py` | NPZ, JSON, CSV, and long-form outputs |
| final figure drawing functions | `scripts/07_make_paper_figures.py` | Confirmed heatmap and fixed-window process figures |
| processed PWV-rate mapping | `core/data_processing.py`, `scripts/03_export_model_csv.py` | Preserve V13013/V13014 values |

The active training entry point delegates annual loading, sequence creation,
target-time splitting, train-only balancing, and feature scaling to
`process_regional_data_for_training`. The CNN stem uses the formal branch
concatenation order: local, dilated, meridional, zonal, broad, raw bypass.
This document records modular restoration work only; it is not evidence of a
complete old-versus-new numerical equivalence result.
