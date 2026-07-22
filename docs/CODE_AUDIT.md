# Code Audit

## 2026-07-22 direct completion update

Repository-level safeguards now include debug split truncation that ignores
top-level metadata, safe output-directory replacement, cross-model prediction
identity checks, multi-seed long-prediction aggregation, and population
standard deviations in summary tables. Full data and original-code comparison
artifacts remain external requirements.
Phase-0 source inventory recorded nine root modules: configuration, processing, model, runtime, training, training utilities, plotting, final paper figures, and the model-stage JSON. SHA-256 hashes and sizes were collected before edits in the execution log; a temporary backup was created outside the repository. The original source contained duplicate numerical imports, font loading, Chinese comments, garbled text, working-directory configuration search, historical model registries, development labels, and old plotting paths. Its physical diagnostics, formal seed 42, learning rate 0.0005, batch size 32, 20 epochs, early stopping patience 8, scheduler patience 3, plain Huber loss, and thresholds 0.1/2.5/10.0 were extracted into the public structure. The original feature-analysis artifact is missing from `reported_results/feature_analysis/` and was not fabricated.

| Source file | Bytes | SHA-256 |
| --- | ---: | --- |
| configs.py | 16,379 | 97F82EAB21AD6BECF5054B862BA43D84CE46BD3BDD997CFE69C70AD49E886541 |
| physical_context_selection_config.json | 1,648 | 7A9162014EB4CAF9A2B2EFA783187FB02F93CAD97989BF11A06970018BAB90A1 |
| plotting_utils.py | 82,461 | 3D0BFC3F8982C05F6AC356B36CB94CD3C19A59EC8CAD96927C126FC3021E50A0 |
| processing_utils.py | 75,630 | F568C57FA811B85668AEA71B2EC2D24BCCE83989D591DBBE1E9E78B5D5F1C0E4 |
| redraw_final9_paper_figures (3).py | 42,431 | DBEF66D21FCE4D8E6391EBBEEE7ADAD19665C15773BB312E3714FDE564B829D7 |
| regional_model.py | 17,075 | 85096A395C2AF903E4A0E69B846E2FBA0064963108B07B9D3078B9B23427E2B8 |
| runtime_utils.py | 8,070 | C61D399498E68843E588F5CB39A20E49309B36EE0551EDECDE5CCC088BCF02B1 |
| training.py | 152,496 | DBBD64D56D1AFEE5D22905C35B9ADBAAAAC7D4D1597A516EF161A5EAF530A124 |
| training_utils.py | 65,883 | 19873FCB26F7A66F995A5798E00937BFB55C575B83165C9E4BB17637CC53729D |

Before removal, the legacy and refactored diagnostic outputs were compared on a complete synthetic grid for `specific_humidity_proxy`, `pwv_grad_mag`, `wind_convergence`, `moisture_advection`, `dPWV+`, and `dPWV-`; the comparison passed with `rtol=1e-7` and `atol=1e-7`.
