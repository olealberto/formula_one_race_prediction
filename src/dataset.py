"""
dataset.py
==========
Builds the training dataset by loading qualifying sessions across
multiple race weekends, extracting features via features.py, and
aggregating from lap level to driver level.

Also loads race finishing positions from the corresponding race session
so the model can learn qualifying telemetry -> race result, not just
qualifying telemetry -> qualifying position.

The output is two DataFrames:
    - lap_df    : one row per lap (used for model training)
    - driver_df : one row per driver per race (used for ranking and evaluation)
                  includes both quali_position and race_position as targets

Usage:
    from src.dataset import build_dataset

    RACES = [
        {"year": 2026, "gp": "Australia", "session": "Q"},
        {"year": 2026, "gp": "China",     "session": "Q"},
    ]

    lap_df, driver_df = build_dataset(RACES, cache_dir="./f1_cache")
"""

import os
import pandas as pd
import numpy as np
import fastf1
import warnings
warnings.filterwarnings("ignore")

from src.features import extract_features_from_session, get_feature_names


# ── AGGREGATION SPEC ──────────────────────────────────────────────────────────
#
# Defines how each feature is aggregated from lap level to driver level.
# "best"   = value from the driver's fastest lap (min lap_time_s)
# "mean"   = mean across all laps
# "min"    = minimum value across all laps
#
# Reasoning: for qualifying prediction we care about peak performance,
# so most features use "best". Mean captures consistency.
#
AGGREGATION_SPEC = {
    "throttle_mean":             "best",
    "throttle_pct_full":         "best",
    "throttle_pct_off":          "best",
    "brake_pct":                 "best",
    "brake_presses":             "best",
    "rpm_mean":                  "best",
    "rpm_max":                   "best",
    "gear_mean":                 "best",
    "gear_changes":              "best",
    "throttle_application_rate": "best",
    "boost_pct":                 "best",
    "boost_mean":                "best",
}


def _load_session(year: int, gp: str, session_type: str, cache_dir: str):
    """Load a FastF1 session with error handling."""
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=True, weather=False, messages=False)
        return session
    except Exception as e:
        print(f"⚠️  Could not load {year} {gp} {session_type}: {e}")
        return None


def _load_race_results(year: int, gp: str, cache_dir: str) -> pd.DataFrame:
    """
    Load race finishing positions for a given GP.
    Returns a DataFrame with columns: driver, race_position, race_podium.
    DNFs are assigned positions after all classified finishers.
    Returns empty DataFrame if race session unavailable.
    """
    try:
        race = fastf1.get_session(year, gp, "R")
        race.load(telemetry=False, weather=False, messages=False)

        results = race.results[["Abbreviation", "Position", "Status"]].copy()
        results.columns = ["driver", "race_position", "status"]

        # Position is NaN for DNFs — assign them positions after last finisher
        classified = results["race_position"].notna().sum()
        dnf_mask   = results["race_position"].isna()
        results.loc[dnf_mask, "race_position"] = range(
            int(classified) + 1,
            int(classified) + 1 + dnf_mask.sum()
        )

        results["race_position"] = results["race_position"].astype(int)
        results["race_podium"]   = (results["race_position"] <= 3).astype(int)
        results["dnf"]           = (~results["status"].str.contains(
                                        "Finished|Lap", na=False
                                    )).astype(int)

        print(f"   ✅ Race results loaded for {year} {gp} "
              f"({len(results)} drivers, "
              f"{dnf_mask.sum()} DNF/DNS)")
        return results[["driver", "race_position", "race_podium", "dnf"]]

    except Exception as e:
        print(f"   ⚠️  Could not load race results for {year} {gp}: {e}")
        return pd.DataFrame()


def _aggregate_to_driver_level(
    lap_df: pd.DataFrame,
    race_results: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Aggregate lap-level features to one row per driver per session.

    For each driver we compute:
    - Features from their personal best lap (fastest lap_time_s)
    - Their best lap time and delta to session best
    - Their qualifying position (rank by best lap time within session)
    - Their race finishing position (if race_results provided)

    race_results : DataFrame with driver, race_position, race_podium, dnf
                   If provided, merged onto driver_df so the model can learn
                   qualifying telemetry -> race result
    """
    records = []
    feature_cols = get_feature_names(lap_df)

    for (session_label, driver), group in lap_df.groupby(["session_label", "driver"]):
        # Best lap = fastest recorded lap for this driver in this session
        best_lap = group.loc[group["lap_time_s"].idxmin()]

        record = {
            "session_label":         session_label,
            "year":                  best_lap["year"],
            "gp":                    best_lap["gp"],
            "session_type":          best_lap["session_type"],
            "driver":                driver,
            "best_lap_time_s":       best_lap["lap_time_s"],
            "delta_to_session_best": best_lap["delta_to_session_best"],
            "n_laps":                len(group),
        }

        # Aggregate each feature per the aggregation spec
        for feat in feature_cols:
            if feat not in group.columns:
                continue
            agg = AGGREGATION_SPEC.get(feat, "best")
            if agg == "best":
                record[feat] = best_lap[feat] if feat in best_lap.index else np.nan
            elif agg == "mean":
                record[f"{feat}_mean"] = group[feat].mean()
            elif agg == "min":
                record[f"{feat}_min"] = group[feat].min()

        records.append(record)

    driver_df = pd.DataFrame(records)

    # Add qualifying position within each session (rank by best lap time)
    driver_df["quali_position"] = (
        driver_df
        .groupby("session_label")["best_lap_time_s"]
        .rank(method="min")
        .astype(int)
    )

    # Merge race results if provided
    if race_results is not None and not race_results.empty:
        # Drop session_label from race_results before merging to avoid
        # column conflicts — match on driver only within each session
        # since driver_df already has session_label
        race_results_clean = race_results.drop(
            columns=["session_label"], errors="ignore"
        )
        # Merge per session to avoid cross-race contamination
        merged_parts = []
        for label, group_df in driver_df.groupby("session_label"):
            merged = group_df.merge(race_results_clean, on="driver", how="left")
            merged_parts.append(merged)
        driver_df = pd.concat(merged_parts, ignore_index=True)
        driver_df["podium"] = driver_df["race_podium"].fillna(0).astype(int)
        print(f"   ✅ Race results merged. "
              f"{driver_df['race_position'].notna().sum()} drivers matched.")
    else:
        driver_df["race_position"] = np.nan
        driver_df["race_podium"]   = np.nan
        driver_df["dnf"]           = np.nan
        driver_df["podium"] = (driver_df["quali_position"] <= 3).astype(int)
        print("   ℹ️  No race results available. Using quali_position as target.")

    return driver_df.sort_values(
        ["session_label", "quali_position"]
    ).reset_index(drop=True)


def build_dataset(
    races: list[dict],
    cache_dir: str = "./f1_cache",
    save_path: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build lap-level and driver-level training datasets from a list of race configs.

    Parameters
    ----------
    races     : list of dicts with keys "year", "gp", "session"
                e.g. [{"year": 2026, "gp": "Australia", "session": "Q"}]
    cache_dir : path to FastF1 cache directory
    save_path : optional path prefix to save CSVs
                e.g. "data/processed" → saves lap_df and driver_df there

    Returns
    -------
    lap_df    : DataFrame with one row per lap
    driver_df : DataFrame with one row per driver per race
    """
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    all_lap_dfs       = []
    all_race_results  = []

    for cfg in races:
        year    = cfg["year"]
        gp      = cfg["gp"]
        session = cfg.get("session", "Q")

        print(f"\n{'─'*55}")
        print(f"  {year} {gp} {session}")
        print(f"{'─'*55}")

        sess = _load_session(year, gp, session, cache_dir)
        if sess is None:
            continue

        lap_df = extract_features_from_session(sess)
        if lap_df.empty:
            continue

        # Load corresponding race results
        print(f"   Loading race results for {year} {gp}...")
        race_results = _load_race_results(year, gp, cache_dir)

        # Aggregate to driver level and merge race results
        driver_lap_df = lap_df.copy()
        all_lap_dfs.append(driver_lap_df)

        # Store race results keyed by session label for later merging
        session_label = f"{year}_{gp}_Q".replace(" ", "_")
        if not race_results.empty:
            race_results["session_label"] = session_label
            all_race_results.append(race_results)

    if not all_lap_dfs:
        print("\n❌ No data loaded. Check race configs and internet connection.")
        return pd.DataFrame(), pd.DataFrame()

    # Combine all sessions
    combined_lap_df = pd.concat(all_lap_dfs, ignore_index=True)

    # Drop feature columns that are NaN across the entire dataset
    feature_cols = get_feature_names(combined_lap_df)
    all_nan = [c for c in feature_cols if combined_lap_df[c].isna().all()]
    if all_nan:
        print(f"\n   Dropping globally unavailable features: {all_nan}")
        combined_lap_df = combined_lap_df.drop(columns=all_nan)

    # Combine race results across all weekends
    combined_race_results = (
        pd.concat(all_race_results, ignore_index=True)
        if all_race_results else pd.DataFrame()
    )

    # Aggregate to driver level with race results merged in
    driver_df = _aggregate_to_driver_level(
        combined_lap_df,
        race_results=combined_race_results if not combined_race_results.empty else None,
    )

    # Summary
    print(f"\n{'═'*55}")
    print(f"  DATASET SUMMARY")
    print(f"{'═'*55}")
    print(f"  Sessions loaded : {combined_lap_df['session_label'].nunique()}")
    print(f"  Total laps      : {len(combined_lap_df)}")
    print(f"  Total drivers   : {combined_lap_df['driver'].nunique()}")
    print(f"  Features active : {len(get_feature_names(combined_lap_df))}")
    print(f"  Driver rows     : {len(driver_df)}")
    print(f"{'═'*55}")

    # Optionally persist to disk
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        lap_path    = os.path.join(save_path, "lap_df.csv")
        driver_path = os.path.join(save_path, "driver_df.csv")
        combined_lap_df.to_csv(lap_path,    index=False)
        driver_df.to_csv(driver_path,       index=False)
        print(f"\n💾 Saved lap_df    → {lap_path}")
        print(f"💾 Saved driver_df → {driver_path}")

    return combined_lap_df, driver_df


def load_saved_dataset(save_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved lap_df and driver_df from disk.
    Useful for iterating on model.py without re-downloading telemetry.
    """
    lap_path    = os.path.join(save_path, "lap_df.csv")
    driver_path = os.path.join(save_path, "driver_df.csv")

    if not os.path.exists(lap_path) or not os.path.exists(driver_path):
        raise FileNotFoundError(
            f"No saved dataset found at {save_path}. "
            "Run build_dataset() with save_path first."
        )

    lap_df    = pd.read_csv(lap_path)
    driver_df = pd.read_csv(driver_path)
    print(f"✅ Loaded dataset from {save_path}")
    print(f"   {len(lap_df)} laps, {len(driver_df)} driver-race rows")
    return lap_df, driver_df
