# formula_one_race_prediction

Predicts F1 qualifying order and podium probabilities using telemetry data from FastF1.

Built during the 2026 season with the new reg cycle in mind.

---

## What it does

Pulls qualifying telemetry, engineers features from driver inputs (throttle, brake, RPM, gear), and trains a ranking model to predict finishing order. Outputs an expected top 3 with podium probabilities for each driver.

---

## Key design decisions

**Qualifying telemetry only** — no practice sessions, no championship standings. Clean signal, available before the race, and directly reflects car and driver performance.

**Lap-level training, driver-level prediction** — each qualifying lap is a training row, aggregated up to one row per driver before prediction. Gives more training data than race-level aggregation.

**Target variable is delta to session best** — gap between each lap and the fastest lap in the session. Normalises across circuits and conditions.

**Two models:**
- LightGBM LambdaRank for predicted position — treats drivers as competing against each other within a race rather than independently
- Calibrated Random Forest for podium probability — more stable at low sample sizes, calibrated so probabilities are honest

**Dynamic feature selection** — Random Forest importance reruns every time you retrain. As more races are added the selected features can shift. Adding a new feature means one entry in the registry in `features.py`, nothing else.

**Speed variables excluded** — average speed is circular with lap time. Features focus on driver inputs: throttle application, braking behaviour, RPM, gear usage.

**DRS excluded** — removed in 2026 regs, replaced by battery boost (MOE). Boost channel is included but fails gracefully on pre-2026 data.

---

## Pipeline

```
features.py   — telemetry feature registry and per-lap extraction
dataset.py    — loads multiple race weekends, builds training dataframe
model.py      — feature selection, trains ranking + podium models, evaluates
predict.py    — loads trained models, predicts a new qualifying session
main.py       — CLI to orchestrate everything
```

---

## Usage

```bash
# Install
pip install -r requirements.txt

# Train on Australia and China 2026
python main.py train

# Predict Japan qualifying
python main.py predict --year 2026 --gp Japan

# Score the prediction after the race
python main.py evaluate --year 2026 --gp Japan
```

Add new races to `TRAINING_RACES` in `main.py` after each weekend and retrain:

```bash
python main.py train --reload   # reuses saved data, just retrains models
```

---

## Adding a new feature

Open `src/features.py` and add an entry to `FEATURE_REGISTRY`:

```python
"my_new_feature": {
    "fn":          lambda tel: tel["Channel"].some_operation(),
    "description": "What this measures",
    "requires":    ["Channel"],  # telemetry channels needed
},
```

That's it. Feature selection will automatically consider it next time you retrain.

---

## Evaluation metrics

After each race `evaluate_prediction` reports:
- Top 3 overlap — how many of the predicted top 3 were actually top 3
- Mean position error — average places off across the full grid
- Spearman ρ — rank correlation between predicted and actual order
- Podium probability calibration — were high probability drivers actually fast

---

## Notes

- Model accuracy will improve as more 2026 races are added to training data
- 2026 is a clean-sheet reg cycle so pre-2026 telemetry patterns may not transfer
- Predictions are qualifying pace proxies for race finish — reliability, safety cars, and strategy are not modelled
- LightGBM must be installed separately: `pip install lightgbm`
- (it's probably mercedes this year)
