# Code-Paper Consistency

The public repository defines the nine-model registry, 12-hour input window, one-hour forecast horizon, 9 × 5 grid, chronological data split, and categorical verification thresholds.

The archived feature-analysis configuration is available at `reported_results/feature_analysis/physical_context_selection_config.json`.

The revised manuscript investigates four meteorologically motivated diagnostic inputs through hierarchical configurations of the shared CNN-BiLSTM backbone.

The public plotting scripts do not reproduce the complete revised manuscript figure set.

The historical formal training-input construction and the public modular preprocessing implementation require further behavioral-equivalence verification, particularly for the dPWV input definitions.

## Diagnostic Input Configurations

| Configuration | Input channels | Additional diagnostic inputs |
|---|---:|---|
| CNN-BiLSTM | 11 | None |
| PhysInput-Lite | 13 | q*, G |
| PhysInput-Adv3 | 14 | q*, G, A |
| PhysInput-Conv3 | 14 | q*, G, C |
| PhysInput-All4 | 15 | q*, G, C, A |

All five configurations use a common CNN-BiLSTM backbone architecture, while input dimensionality and trainable parameter counts differ.

## Evaluation Scope

The manuscript reports continuous rainfall prediction errors and categorical verification at precipitation thresholds of 0.1, 2.5, and 10 mm h⁻¹.

For the 10 mm h⁻¹ threshold, the archived formal results show higher POD for PhysInput-All4 relative to the CNN-BiLSTM baseline, accompanied by a higher FAR.

Paired block-bootstrap uncertainty analysis was conducted using archived predictions and ERA5 reference fields. These results concern the original fixed experimental runs and do not constitute a full rerun of the public modular implementation.

## Source of Reported Results

The manuscript tables and figures are based on archived formal outputs. The `reported_results/` directory contains selected lightweight research outputs rather than the complete prediction archive.
