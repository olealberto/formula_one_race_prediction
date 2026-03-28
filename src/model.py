"""
model.py
========
Trains a model on the driver-level dataset produced by dataset.py.

Responsibilities:
- Dynamic feature selection via Random Forest importance
- Train a ranking model (LightGBM LambdaRank) to predict qualifying order
- Train a secondary classifier for podium probability
- Persist trained models and feature metadata to disk
- Evaluate on held-out races with interpretable metrics

Design notes:
- Feature selection reruns every time the model is retrained, so as new
  races are added the selected features can shift automatically
- LambdaRank is used for the primary model because drivers within a race
  are competing against each other — a ranking loss is more appropriate
  than treating each driver as independent
- A calibrated Random Forest is used for podium probability because the
  dataset is small and RF is more stable than LightGBM at low sample sizes
- Both models are saved alongside the feature list used, so predictions
  always use the correct feature set

Requirements:
    pip install lightgbm scikit-learn pandas numpy joblib
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import ndcg_score

from src.features import get_feature_names, FEATURE_REGISTRY


# ── CONFIG ────────────────────────────────────────────────────────────────────
TOP_N_FEATURES   = 8     # How many features to select dynamically
MIN_RACES        = 2     # Minimum races needed before training
LIGHTGBM_PARAMS  = {
    "objective":       "lambdarank",
    "metric":          "ndcg",
    "ndcg_eval_at":    [3, 5],
    "learning_rate":   0.05,
    "num_leaves":      15,       # kept small given low data volume
    "min_data_in_leaf": 3,
    "n_estimators":    200,
    "verbose":         -1,
}
# ─────────────────────────────────────────────────────────────────────────────


# ── FEATURE SELECTION ─────────────────────────────────────────────────────────

def select_features(
    driver_df: pd.DataFrame,
    top_n: int = TOP_N_FEATURES,
) -> list[str]:
    """
    Dynamically select the most predictive features using Random Forest
    importance. Uses race_position as target if available, otherwise
    falls back to delta_to_session_best from qualifying.

    Returns a list of the top_n feature names.
    """
    feature_cols = [c for c in get_feature_names(driver_df)
                    if c in driver_df.columns]

    if not feature_cols:
        raise ValueError("No feature columns found in driver_df.")

    # Use race position if available, otherwise qualifying delta
    if "race_position" in driver_df.columns and driver_df["race_position"].notna().sum() > 5:
        target_col = "race_position"
        print("\n📊 Feature selection target: race_position")
    else:
        target_col = "delta_to_session_best"
        print("\n📊 Feature selection target: delta_to_session_best (no race results)")

    X = driver_df[feature_cols].copy()
    y = driver_df[target_col].copy()

    mask = y.notna() & X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)

    importance_df = pd.DataFrame({
        "feature":    feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    selected = importance_df.head(top_n)["feature"].tolist()

    print(f"\n📊 Selected top {top_n} features:")
    for i, row in importance_df.head(top_n).iterrows():
        desc = FEATURE_REGISTRY.get(row["feature"], {}).get("description", "")
        print(f"   {importance_df.index.get_loc(i)+1:>2}. {row['feature']:<30} "
              f"importance={row['importance']:.4f}   {desc}")

    return selected, importance_df


# ── RANKING MODEL (LightGBM LambdaRank) ──────────────────────────────────────

def _build_lambdarank_data(
    training_df: pd.DataFrame,
    feature_cols: list[str],
):
    """
    Build LightGBM Dataset for LambdaRank.
    Each session is a query group.
    training_df can be lap-level (one row per lap) or driver-level
    (one row per driver) — both work, lap-level gives more training signal.
    Label is inverted race_position so P1 gets the highest label.
    """
    df = training_df.dropna(subset=feature_cols + ["race_position"]).copy()

    max_pos = df.groupby("session_label")["race_position"].transform("max")
    df["rank_label"] = (max_pos - df["race_position"] + 1).astype(int)

    X      = df[feature_cols].values
    labels = df["rank_label"].values

    group_sizes = (
        df.groupby("session_label", sort=False)
        .size()
        .values
    )

    dataset = lgb.Dataset(
        X,
        label=labels,
        group=group_sizes,
        feature_name=feature_cols,
        free_raw_data=False,
    )

    return dataset, df


def train_ranking_model(
    lap_training_df: pd.DataFrame,
    feature_cols: list[str],
) -> lgb.Booster:
    """
    Train LambdaRank on lap-level training data.
    Each qualifying lap is one training row, tagged with its driver's
    race finishing position. Gives ~10x more rows than driver-level training.
    """
    print(f"\n⏳ Training LambdaRank on {len(lap_training_df)} lap-level rows...")

    dataset, _ = _build_lambdarank_data(lap_training_df, feature_cols)

    model = lgb.train(
        LIGHTGBM_PARAMS,
        dataset,
        num_boost_round=LIGHTGBM_PARAMS["n_estimators"],
        valid_sets=[dataset],
        callbacks=[lgb.early_stopping(20, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    print(f"✅ LambdaRank trained. Best iteration: {model.best_iteration}")
    return model


# ── PODIUM PROBABILITY MODEL (Calibrated Random Forest) ──────────────────────

def train_podium_model(
    driver_df: pd.DataFrame,
    feature_cols: list[str],
) -> CalibratedClassifierCV:
    """
    Train a calibrated Random Forest classifier to predict podium (top 3).
    Calibration converts raw scores to proper probabilities.
    """
    print("\n⏳ Training podium probability model...")

    df = driver_df.dropna(subset=feature_cols + ["podium"]).copy()
    X  = df[feature_cols].values
    y  = df["podium"].values

    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,       # shallow trees given small dataset
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    # Isotonic calibration with cross-val to get honest probabilities
    # cv=3 because we have limited data
    calibrated = CalibratedClassifierCV(base_rf, cv=3, method="isotonic")
    calibrated.fit(X, y)

    print("✅ Podium probability model trained.")
    return calibrated


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate_leave_one_race_out(
    driver_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Leave-one-race-out cross validation.
    For each race, train on all others, predict on the held-out race.
    Returns a DataFrame with predictions vs actuals per driver per race.
    """
    print("\n⏳ Running leave-one-race-out evaluation...")

    sessions = driver_df["session_label"].unique()
    if len(sessions) < MIN_RACES:
        print(f"⚠️  Need at least {MIN_RACES} races for evaluation. "
              f"Currently have {len(sessions)}.")
        return pd.DataFrame()

    all_results = []

    for held_out in sessions:
        train_df = driver_df[driver_df["session_label"] != held_out]
        test_df  = driver_df[driver_df["session_label"] == held_out].copy()

        test_df = test_df.dropna(subset=feature_cols)
        if test_df.empty:
            continue

        # Train ranking model on everything except held-out race
        try:
            ranking_model = train_ranking_model(train_df, feature_cols)
            scores = ranking_model.predict(test_df[feature_cols].values)
            test_df["predicted_score"]    = scores
            test_df["predicted_position"] = (
                test_df["predicted_score"]
                .rank(ascending=False)
                .astype(int)
            )
        except Exception as e:
            print(f"   ⚠️  Ranking model failed for {held_out}: {e}")
            continue

        # Train podium model
        try:
            podium_model  = train_podium_model(train_df, feature_cols)
            podium_probs  = podium_model.predict_proba(
                test_df[feature_cols].values
            )[:, 1]
            test_df["podium_probability"] = podium_probs
        except Exception as e:
            print(f"   ⚠️  Podium model failed for {held_out}: {e}")
            test_df["podium_probability"] = np.nan

        all_results.append(test_df)

    if not all_results:
        return pd.DataFrame()

    results_df = pd.concat(all_results, ignore_index=True)

    # Print summary
    print(f"\n{'═'*60}")
    print(f"  LEAVE-ONE-RACE-OUT EVALUATION RESULTS")
    print(f"{'═'*60}")

    for session in results_df["session_label"].unique():
        sub = results_df[results_df["session_label"] == session].copy()
        sub = sub.sort_values("predicted_position")

        # Top 3 accuracy: how many of the predicted top 3 were actually top 3
        pred_top3   = set(sub.head(3)["driver"])
        actual_top3 = set(sub[sub["quali_position"] <= 3]["driver"])
        overlap     = len(pred_top3 & actual_top3)

        print(f"\n  {session}")
        print(f"  Top 3 overlap: {overlap}/3  "
              f"({pred_top3} predicted vs {actual_top3} actual)")
        print(f"  {'Driver':<6} {'Pred':>5} {'Actual':>7} {'Podium%':>8}")
        print(f"  {'─'*30}")
        for _, row in sub.head(5).iterrows():
            prob = f"{row['podium_probability']*100:.0f}%" \
                   if not pd.isna(row.get("podium_probability")) else "  —"
            print(f"  {row['driver']:<6} {int(row['predicted_position']):>5} "
                  f"{int(row['quali_position']):>7} {prob:>8}")

    print(f"\n{'═'*60}")
    return results_df


# ── PERSIST & LOAD ────────────────────────────────────────────────────────────

def save_models(
    ranking_model,
    podium_model,
    feature_cols: list[str],
    importance_df: pd.DataFrame,
    save_dir: str = "models",
):
    """Save trained models, selected features, and importance scores."""
    os.makedirs(save_dir, exist_ok=True)

    ranking_path    = os.path.join(save_dir, "ranking_model.lgb")
    podium_path     = os.path.join(save_dir, "podium_model.pkl")
    features_path   = os.path.join(save_dir, "selected_features.json")
    importance_path = os.path.join(save_dir, "feature_importance.csv")

    ranking_model.save_model(ranking_path)
    joblib.dump(podium_model, podium_path)

    with open(features_path, "w") as f:
        json.dump(feature_cols, f, indent=2)

    importance_df.to_csv(importance_path, index=False)

    print(f"\n💾 Models saved to {save_dir}/")
    print(f"   ranking_model.lgb")
    print(f"   podium_model.pkl")
    print(f"   selected_features.json")
    print(f"   feature_importance.csv")


def load_models(save_dir: str = "models"):
    """Load previously saved models and feature list."""
    ranking_path  = os.path.join(save_dir, "ranking_model.lgb")
    podium_path   = os.path.join(save_dir, "podium_model.pkl")
    features_path = os.path.join(save_dir, "selected_features.json")

    if not all(os.path.exists(p) for p in
               [ranking_path, podium_path, features_path]):
        raise FileNotFoundError(
            f"Models not found in {save_dir}. Run train() first."
        )

    ranking_model = lgb.Booster(model_file=ranking_path)
    podium_model  = joblib.load(podium_path)

    with open(features_path) as f:
        feature_cols = json.load(f)

    print(f"✅ Models loaded from {save_dir}/")
    print(f"   Features: {feature_cols}")
    return ranking_model, podium_model, feature_cols


# ── MAIN TRAIN ENTRY POINT ────────────────────────────────────────────────────

def train(
    driver_df: pd.DataFrame,
    lap_training_df: pd.DataFrame,
    save_dir: str = "models",
    top_n_features: int = TOP_N_FEATURES,
    run_evaluation: bool = True,
    force_features: list = None,
) -> tuple:
    """
    Full training pipeline:
    1. Select features dynamically OR use hardcoded feature list
    2. Optionally run leave-one-race-out evaluation
    3. Train LambdaRank on lap-level data (more rows = better learning)
    4. Train podium classifier on driver-level data
    5. Save everything to disk

    Parameters
    ----------
    driver_df       : one row per driver per race — used for podium model and evaluation
    lap_training_df : one row per qualifying lap with race_position target — used for LambdaRank
    save_dir        : where to save models
    top_n_features  : how many features to select dynamically
    run_evaluation  : whether to run LORO evaluation before final training
    force_features  : if provided, skip dynamic selection and use this list

    Returns
    -------
    ranking_model, podium_model, feature_cols, importance_df
    """
    n_races = driver_df["session_label"].nunique()
    n_laps  = len(lap_training_df)
    print(f"\n{'═'*55}")
    print(f"  TRAINING on {n_races} race(s), {n_laps} lap training rows")
    print(f"{'═'*55}")

    if n_races < MIN_RACES:
        print(f"⚠️  Only {n_races} race(s) available. "
              f"Minimum recommended is {MIN_RACES}.")

    # Step 1: Feature selection — run on driver_df for stable importance ranking
    if force_features:
        feature_cols = force_features
        importance_df = pd.DataFrame({
            "feature":    force_features,
            "importance": [1/len(force_features)] * len(force_features),
        })
        print(f"\n📊 Using hardcoded validated features: {feature_cols}")
    else:
        feature_cols, importance_df = select_features(driver_df, top_n_features)

    # Step 2: Evaluation on driver-level data
    if run_evaluation and n_races >= MIN_RACES:
        evaluate_leave_one_race_out(driver_df, feature_cols)

    # Step 3: LambdaRank trained on lap-level data for more signal
    ranking_model = train_ranking_model(lap_training_df, feature_cols)

    # Step 4: Podium classifier on driver-level data
    podium_model  = train_podium_model(driver_df, feature_cols)

    # Step 5: Save
    save_models(ranking_model, podium_model, feature_cols,
                importance_df, save_dir)

    return ranking_model, podium_model, feature_cols, importance_df