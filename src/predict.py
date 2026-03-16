"""
predict.py
==========
Takes a new qualifying session, extracts features, runs trained models,
and outputs expected top three with podium probabilities.

Always uses qualifying telemetry as input.
Fetches race results separately for evaluation if available.

Usage:
    from src.predict import predict_race

    results = predict_race(
        year     = 2026,
        gp       = "Japan",
        save_dir = "models",
    )

Or from the command line via main.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import fastf1
warnings.filterwarnings("ignore")

from src.features import extract_features_from_session
from src.dataset  import _aggregate_to_driver_level, _load_race_results
from src.model    import load_models


# ── PROBABILITY CONVERSION ────────────────────────────────────────────────────

def _ranking_scores_to_probabilities(
    scores: np.ndarray,
    top_n: int = 3,
    temperature: float = 2.0,
) -> np.ndarray:
    """
    Convert raw LambdaRank scores to podium probabilities using a
    softmax with temperature scaling.

    Higher temperature = more spread out probabilities (more uncertainty).
    Lower temperature = more confident predictions.

    With limited training data, a higher temperature is more honest.
    """
    scaled = scores * temperature
    exp    = np.exp(scaled - scaled.max())  # numerical stability
    probs  = exp / exp.sum()
    return probs


def _build_prediction_table(
    driver_df: pd.DataFrame,
    ranking_scores: np.ndarray,
    podium_probs: np.ndarray,
    softmax_probs: np.ndarray,
) -> pd.DataFrame:
    """
    Assemble the final prediction DataFrame sorted by predicted position.
    """
    df = driver_df.copy().reset_index(drop=True)

    df["ranking_score"]       = ranking_scores
    df["podium_probability"]  = podium_probs
    df["top3_probability"]    = softmax_probs

    # Predicted position: rank by ranking score descending (higher = better)
    df["predicted_position"] = (
        df["ranking_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    # Blend the two probability signals:
    # - podium_probability: calibrated RF, trained on binary top-3 label
    # - top3_probability:   softmax over ranking scores, captures relative gaps
    # Simple average; can be tuned as more data accumulates
    df["blended_podium_pct"] = (
        (df["podium_probability"] + df["top3_probability"]) / 2 * 100
    ).round(1)

    return df.sort_values("predicted_position").reset_index(drop=True)


# ── CORE PREDICTION FUNCTION ──────────────────────────────────────────────────

def predict_race(
    year:      int,
    gp:        str,
    save_dir:  str  = "models",
    cache_dir: str  = "./f1_cache",
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Load qualifying telemetry, extract features, and predict race finishing order.

    Always uses Q session as input — race telemetry is never used.
    Fetches race results separately and merges them onto the output
    so evaluate_prediction() can score against actual race finish.

    Parameters
    ----------
    year      : race year  (e.g. 2026)
    gp        : grand prix name (e.g. "Japan")
    save_dir  : directory where trained models are saved
    cache_dir : FastF1 cache directory
    verbose   : print prediction table to console

    Returns
    -------
    DataFrame with one row per driver, sorted by predicted position, columns:
        driver, predicted_position, blended_podium_pct,
        quali_position (actual qualifying position),
        race_position  (actual race finish, if race has happened),
        + all feature values used by the model
    """
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    # ── Load models ───────────────────────────────────────────────────────────
    ranking_model, podium_model, feature_cols = load_models(save_dir)

    # ── Always load qualifying telemetry as input ─────────────────────────────
    print(f"\n⏳ Loading {year} {gp} Q for prediction...")
    try:
        sess = fastf1.get_session(year, gp, "Q")
        sess.load(telemetry=True, weather=False, messages=False)
    except Exception as e:
        raise RuntimeError(f"Could not load {year} {gp} Q: {e}")

    lap_df = extract_features_from_session(sess)
    if lap_df.empty:
        raise ValueError("No valid laps extracted from qualifying session.")

    # ── Aggregate to driver level ─────────────────────────────────────────────
    driver_df = _aggregate_to_driver_level(lap_df)

    # ── Fetch race results separately for evaluation ──────────────────────────
    # This will be None if the race hasn't happened yet — handled gracefully
    print(f"   Fetching race results for evaluation (if available)...")
    race_results = _load_race_results(year, gp, cache_dir)
    if not race_results.empty:
        driver_df = driver_df.merge(race_results, on="driver", how="left")

    # ── Check feature availability ────────────────────────────────────────────
    missing = [f for f in feature_cols if f not in driver_df.columns]
    if missing:
        print(f"⚠️  Missing features: {missing}")
        print(   "   These will be filled with column means from available data.")
        for f in missing:
            driver_df[f] = np.nan

    # Fill any remaining NaNs with column means (last resort)
    X = driver_df[feature_cols].copy()
    X = X.fillna(X.mean())

    # ── Run models ────────────────────────────────────────────────────────────
    ranking_scores = ranking_model.predict(X.values)
    podium_probs   = podium_model.predict_proba(X.values)[:, 1]
    softmax_probs  = _ranking_scores_to_probabilities(ranking_scores, top_n=3)

    # ── Build output ──────────────────────────────────────────────────────────
    predictions = _build_prediction_table(
        driver_df, ranking_scores, podium_probs, softmax_probs
    )

    # ── Print results ─────────────────────────────────────────────────────────
    if verbose:
        _print_predictions(predictions, year, gp, feature_cols)

    return predictions


# ── DISPLAY ───────────────────────────────────────────────────────────────────

def _print_predictions(
    predictions: pd.DataFrame,
    year: int,
    gp: str,
    feature_cols: list[str],
):
    has_race  = "race_position" in predictions.columns and \
                predictions["race_position"].notna().any()
    has_quali = "quali_position" in predictions.columns

    # Show race result if available, otherwise qualifying
    actual_col   = "race_position" if has_race else \
                   "quali_position" if has_quali else None
    actual_label = "Race" if has_race else "Quali" if has_quali else None

    print(f"\n{'═'*62}")
    print(f"  PREDICTED RACE FINISH — {year} {gp}")
    if actual_col:
        print(f"  Actual shown: {actual_label} position")
    print(f"{'═'*62}")

    top3 = predictions.head(3)
    print(f"\n  🏆 EXPECTED TOP 3")
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(top3.iterrows()):
        actual_str = ""
        if actual_col and not pd.isna(row.get(actual_col)):
            actual_str = f"  (actual P{int(row[actual_col])})"
        print(f"     {medals[i]} P{i+1}  {row['driver']:<6}  "
              f"{row['blended_podium_pct']:>5.1f}% podium chance"
              f"{actual_str}")

    print(f"\n  {'Pos':>4}  {'Driver':<6}  {'Podium%':>8}  ", end="")
    if actual_col:
        print(f"{actual_label:>7}  {'Δ Pos':>6}", end="")
    print()
    print(f"  {'─'*50}")

    for _, row in predictions.iterrows():
        pred_pos   = int(row["predicted_position"])
        actual_str = ""
        delta_str  = ""
        if actual_col and not pd.isna(row.get(actual_col)):
            actual    = int(row[actual_col])
            delta     = actual - pred_pos
            actual_str = f"{actual:>7}"
            delta_str  = f"  {delta:>+5}" if delta != 0 else f"  {'✓':>5}"

        print(f"  {pred_pos:>4}  {row['driver']:<6}  "
              f"{row['blended_podium_pct']:>7.1f}%"
              f"{actual_str}{delta_str}")

    print(f"\n  Features used: {feature_cols}")
    print(f"{'═'*62}")


# ── ACCURACY SUMMARY (post-race) ──────────────────────────────────────────────

def evaluate_prediction(predictions: pd.DataFrame) -> dict:
    """
    After a race weekend, compare predictions to actual results.
    Uses race_position if available, otherwise falls back to quali_position.
    """
    # Prefer race_position as the ground truth
    if "race_position" in predictions.columns and predictions["race_position"].notna().sum() > 3:
        actual_col  = "race_position"
        actual_top3 = set(predictions[predictions["race_position"] <= 3]["driver"])
        print("\n📊 Evaluating against: race_position")
    elif "quali_position" in predictions.columns:
        actual_col  = "quali_position"
        actual_top3 = set(predictions[predictions["quali_position"] <= 3]["driver"])
        print("\n📊 Evaluating against: quali_position (no race results yet)")
    else:
        raise ValueError("No position data available for evaluation.")

    df = predictions.dropna(subset=[actual_col, "predicted_position"])

    # Top 3 overlap
    pred_top3  = set(df[df["predicted_position"] <= 3]["driver"])
    top3_hits  = len(pred_top3 & actual_top3)

    # Mean absolute error on position
    mae = (df["predicted_position"] - df[actual_col]).abs().mean()

    # Spearman rank correlation
    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["predicted_position"], df[actual_col])

    # Podium probability calibration
    if "blended_podium_pct" in df.columns:
        top3_avg_prob = df[df[actual_col] <= 3]["blended_podium_pct"].mean()
        rest_avg_prob = df[df[actual_col]  > 3]["blended_podium_pct"].mean()
    else:
        top3_avg_prob = rest_avg_prob = np.nan

    metrics = {
        "evaluated_against": actual_col,
        "top3_hits":         top3_hits,
        "top3_overlap":      f"{top3_hits}/3",
        "mae_positions":     round(mae, 2),
        "spearman_rho":      round(rho, 3),
        "spearman_pval":     round(pval, 3),
        "top3_avg_prob":     round(top3_avg_prob, 1),
        "rest_avg_prob":     round(rest_avg_prob, 1),
    }

    print(f"\n{'═'*50}")
    print(f"  POST-RACE ACCURACY SUMMARY")
    print(f"{'═'*50}")
    print(f"  Evaluated against   : {actual_col}")
    print(f"  Top 3 overlap       : {metrics['top3_overlap']}")
    print(f"  Mean position error : {metrics['mae_positions']} places")
    print(f"  Spearman ρ          : {metrics['spearman_rho']} "
          f"(p={metrics['spearman_pval']})")
    print(f"  Avg podium prob — actual top 3 : {metrics['top3_avg_prob']}%")
    print(f"  Avg podium prob — rest of grid : {metrics['rest_avg_prob']}%")
    print(f"{'═'*50}")

    return metrics
