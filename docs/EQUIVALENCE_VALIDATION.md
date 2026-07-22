# Equivalence validation status

The public repository contains only the modular execution path.  Original
large source files are external, temporary behavioral references and are not
included here.

| Stage | Compared object | Result | Tolerance |
|---|---|---|---|
| Data loading | rows, columns, times | Not executed: original reader unavailable in this workspace | exact |
| Diagnostics | four physical fields | Existing synthetic-grid unit test passed | unit-test assertions |
| Sequences | X/y/rain/time | Existing alignment and grid-integrity unit tests passed | exact test cases |
| Split | indices | Existing target-time split unit test passed | exact test cases |
| Sampling | selected indices | Existing event-aware sampler unit test passed | exact test cases |
| Scaling | mean/scale/output | Not executed against original external normalizer | strict floating tolerance |
| Models | layers/params/config | Not executed against original model builder | exact |
| Initialization | weights | Not executed against original model builder | strict floating tolerance |
| One epoch | losses/predictions | Not executed against original trainer | strict floating tolerance |
| Metrics | all outputs | Existing metric unit tests passed; no external comparison run | exact test cases |
| Figure inputs | CSV content/window | Existing figure-output unit test passed; no external comparison run | exact test cases |

The required original `regional_model.py`, `training.py`, `training_utils.py`,
`configs.py`, and `runtime_utils.py` references were not present in the
provided workspace at validation time. Consequently this repository must not
yet be described as behaviorally equivalent to the original direct training
workflow. Supply those temporary references and the full formal data to run
the remaining old-versus-new comparisons.

Current repository checks completed on 2026-07-22: `compileall`, CLI help
checks, and 21 unit tests. TensorFlow was available for the TensorFlow-specific
tests in that run. These are modular consistency checks, not old-versus-new
equivalence tests.
