# formula_one_race_prediction is race by race predictive modeling for 2026 formula one season. (it's probably mercedes this year)

### The pipeline would look something like:
First, establish a feature set from already completed circuits in 2026. fully validated features show up in multiple races.
Second, for each new race weekend, extract those same features from qualifying telemetry. This is the input to the model.
Third, the model outputs predicted finishing positions or probabilities.



## Features.py
The registry — every feature is one entry in FEATURE_REGISTRY. To add time-to-full-throttle per corner later, or any new 2026 channel that emerges, you just add a dict entry. Nothing else changes.

Graceful degradation — two layers of it. The requires field skips features if a channel is missing entirely. The try/except inside _compute_features_for_lap catches anything that fails at runtime. Columns that are entirely NaN across all laps get dropped automatically with a warning.

The target variable delta_to_session_best is the gap between each lap and the fastest lap in the session 

