"""
prerace_predictor.py
====================
A lightweight pre-race predictor that uses only race results and
qualifying gaps from previous rounds — no telemetry required.

This runs BEFORE a race weekend's qualifying session and gives a
rough predicted finishing order based on season form so far.

Clearly labelled as a fun/rough predictor. Not a substitute for
the main telemetry pipeline.

Usage:
    python prerace_predictor.py

Configure TARGET_RACE at the top to set which race you're predicting.
"""

import os
import numpy as np
import pandas as pd
import fastf1
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TARGET_YEAR  = 2026
TARGET_RACE  = "Japan"

# Races to use as form data — add new rounds as the season progresses
FORM_RACES = [
    {"year": 2026, "gp": "Australia"},
    {"year": 2026, "gp": "China"},
]

# How much to weight recent races vs older ones
# First entry = oldest race weight, last = most recent
RECENCY_WEIGHTS = [0.35, 0.65]

# How much the race over/underperformance adjusts the pace score
# 0 = ignore race performance, 1 = equal weight with pace
RACE_PERFORMANCE_WEIGHT = 0.3

CACHE_DIR = "./f1_cache"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_weekend_data(year: int, gp: str) -> pd.DataFrame:
    """
    Load qualifying delta to pole and race finishing position
    for every driver at a given race weekend.
    Returns one row per driver.
    """
    print(f"\n⏳ Loading {year} {gp}...")

    records = []

    # ── Qualifying ────────────────────────────────────────────────────────────
    try:
        quali = fastf1.get_session(year, gp, "Q")
        quali.load(telemetry=False, weather=False, messages=False)

        best_laps = (
            quali.laps
            .dropna(subset=["LapTime"])
            .groupby("Driver")["LapTime"]
            .min()
            .dt.total_seconds()
            .reset_index()
        )
        best_laps.columns = ["driver", "quali_time_s"]
        pole_time = best_laps["quali_time_s"].min()
        best_laps["quali_delta"] = best_laps["quali_time_s"] - pole_time
        best_laps["quali_position"] = best_laps["quali_time_s"].rank(method="min").astype(int)

    except Exception as e:
        print(f"   ⚠️  Could not load qualifying: {e}")
        return pd.DataFrame()

    # ── Race ──────────────────────────────────────────────────────────────────
    try:
        race = fastf1.get_session(year, gp, "R")
        race.load(telemetry=False, weather=False, messages=False)

        results = race.results[["Abbreviation", "Position", "Status"]].copy()
        results.columns = ["driver", "race_position", "status"]

        # Handle DNFs
        classified   = results["race_position"].notna().sum()
        dnf_mask     = results["race_position"].isna()
        results.loc[dnf_mask, "race_position"] = range(
            int(classified) + 1,
            int(classified) + 1 + dnf_mask.sum()
        )
        results["race_position"] = results["race_position"].astype(int)
        results["dnf"]           = (~results["status"].str.contains(
                                        "Finished|Lap", na=False
                                    )).astype(int)

    except Exception as e:
        print(f"   ⚠️  Could not load race results: {e}")
        results = pd.DataFrame(columns=["driver", "race_position", "dnf"])

    # ── Merge ─────────────────────────────────────────────────────────────────
    df = best_laps.merge(results[["driver", "race_position", "dnf"]],
                         on="driver", how="left")

    # Positions gained/lost from qualifying to race
    df["positions_gained"] = df["quali_position"] - df["race_position"]

    df["year"] = year
    df["gp"]   = gp

    print(f"   ✅ {len(df)} drivers loaded.")
    return df


def build_form_table(form_races: list, weights: list) -> pd.DataFrame:
    """
    Build a combined form table across all past races.
    Returns one row per driver with weighted pace and performance scores.
    """
    all_dfs = []
    for cfg, w in zip(form_races, weights):
        df = load_weekend_data(cfg["year"], cfg["gp"])
        if df.empty:
            continue
        df["weight"] = w
        all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No form data loaded.")

    combined = pd.concat(all_dfs, ignore_index=True)

    # ── Pace score ────────────────────────────────────────────────────────────
    # Normalise qualifying delta within each race (0 = pole, higher = slower)
    # then compute weighted average across races
    combined["quali_delta_norm"] = combined.groupby("gp")["quali_delta"].transform(
        lambda x: x / x.max()
    )

    pace_score = (
        combined
        .groupby("driver")
        .apply(lambda g: np.average(g["quali_delta_norm"], weights=g["weight"]))
        .reset_index()
    )
    pace_score.columns = ["driver", "pace_score"]
    # Lower pace score = faster

    # ── Race performance score ────────────────────────────────────────────────
    # Average positions gained/lost, excluding DNFs
    perf_score = (
        combined[combined["dnf"] == 0]
        .groupby("driver")
        .apply(lambda g: np.average(g["positions_gained"], weights=g["weight"]))
        .reset_index()
    )
    perf_score.columns = ["driver", "avg_positions_gained"]

    # ── Combine ───────────────────────────────────────────────────────────────
    form = pace_score.merge(perf_score, on="driver", how="left")
    form["avg_positions_gained"] = form["avg_positions_gained"].fillna(0)

    # Normalise performance score to same scale as pace score
    max_gain = form["avg_positions_gained"].abs().max()
    if max_gain > 0:
        form["perf_score_norm"] = -form["avg_positions_gained"] / max_gain * \
                                   form["pace_score"].std()
    else:
        form["perf_score_norm"] = 0

    # Final score: pace adjusted by race over/underperformance
    form["final_score"] = (
        (1 - RACE_PERFORMANCE_WEIGHT) * form["pace_score"] +
        RACE_PERFORMANCE_WEIGHT       * form["perf_score_norm"]
    )

    # Predicted position: rank by final score ascending (lower = better)
    form["predicted_position"] = form["final_score"].rank(method="min").astype(int)

    return form.sort_values("predicted_position").reset_index(drop=True)


def print_prediction(form: pd.DataFrame, year: int, gp: str):
    """Print the pre-race prediction table."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}

    print(f"\n{'═'*62}")
    print(f"  PRE-RACE PREDICTION — {year} {gp} Grand Prix")
    print(f"  Based on: Australia & China 2026 form")
    print(f"  ⚠️  Rough predictor — no telemetry, no circuit data")
    print(f"{'═'*62}")

    print(f"\n  🏆 PREDICTED TOP 3")
    for _, row in form.head(3).iterrows():
        medal = medals.get(int(row["predicted_position"]), "  ")
        gain  = row["avg_positions_gained"]
        trend = f"+{gain:.1f} pos/race" if gain > 0 else f"{gain:.1f} pos/race"
        print(f"     {medal} P{int(row['predicted_position'])}  "
              f"{row['driver']:<6}  pace score: {row['pace_score']:.3f}  "
              f"race trend: {trend}")

    print(f"\n  {'Pos':>4}  {'Driver':<6}  {'Pace Score':>10}  "
          f"{'Avg Pos Gained':>14}  {'Final Score':>11}")
    print(f"  {'─'*52}")

    for _, row in form.iterrows():
        print(f"  {int(row['predicted_position']):>4}  "
              f"{row['driver']:<6}  "
              f"{row['pace_score']:>10.3f}  "
              f"{row['avg_positions_gained']:>+14.1f}  "
              f"{row['final_score']:>11.4f}")

    print(f"\n{'═'*62}")
    print(f"  Pace score    : weighted qualifying delta to pole (lower = faster)")
    print(f"  Avg pos gained: mean positions gained from quali to race finish")
    print(f"  Final score   : pace ({int((1-RACE_PERFORMANCE_WEIGHT)*100)}%) + "
          f"race performance ({int(RACE_PERFORMANCE_WEIGHT*100)}%)")
    print(f"{'═'*62}\n")


def main():
    print(f"\n🏎️  PRE-RACE PREDICTOR — {TARGET_YEAR} {TARGET_RACE} GP")
    print(f"   (No telemetry — results-based form model)")

    if len(FORM_RACES) != len(RECENCY_WEIGHTS):
        raise ValueError("FORM_RACES and RECENCY_WEIGHTS must have the same length.")

    form = build_form_table(FORM_RACES, RECENCY_WEIGHTS)
    print_prediction(form, TARGET_YEAR, TARGET_RACE)

    # Save
    out_path = f"./data/processed/prerace_{TARGET_YEAR}_{TARGET_RACE}.csv"
    os.makedirs("./data/processed", exist_ok=True)
    form.to_csv(out_path, index=False)
    print(f"💾 Saved → {out_path}")


if __name__ == "__main__":
    main()
