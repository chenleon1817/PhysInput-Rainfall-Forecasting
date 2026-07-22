# Current-to-Target Mapping

## Direct-completion status

The formal modular data path is now centralized in
`process_regional_data_for_training`. Evaluation consumes saved test archives
only. It validates archive identity within a run and writes a single combined
long-prediction table per model when multiple seeds are evaluated.
| Current source | Destination | Decision |
| --- | --- | --- |
| configs.py | core/config_loader.py and configs/*.json | rewritten |
| processing_utils.py | core/data_processing.py, core/physical_diagnostics.py, core/evaluation_utils.py | split and cleaned |
| regional_model.py | core/models.py | moved and simplified to formal registry |
| training_utils.py | core/training_utils.py, core/result_export.py, core/models.py | split |
| runtime_utils.py | core/runtime.py | moved |
| training.py | scripts/05_train_models.py and scripts/06_evaluate_models.py | split |
| plotting_utils.py | core/result_export.py | obsolete figure generation removed |
| redraw_final9_paper_figures (3).py | scripts/07_make_paper_figures.py | rewritten as sole figure entry |
| physical_context_selection_config.json | configs/physinput_model_config.json | renamed and cleaned |
