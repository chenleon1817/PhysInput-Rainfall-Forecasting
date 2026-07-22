# Model Configuration

The shared input order is T, P, RH, PWV, U, V, DOY, HOD, dPWV+, dPWV-, and lagged rainfall. Diagnostics are a specific-humidity proxy, PWV-gradient magnitude, wind convergence, and moisture advection. PhysInput appends only the configured diagnostic channels to the shared input; it has no separate physical branch, gate, or post-processing. `configs/physinput_model_config.json` is the source of truth for diagnostic combinations and is validated before use.
