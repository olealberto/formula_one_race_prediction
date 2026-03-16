"""
features.py
===========
Feature registry and extraction for the F1 qualifying telemetry pipeline.

Design principles:
- All features are registered in FEATURE_REGISTRY as named, self-contained functions
- Adding a new feature = adding one entry to the registry, nothing else
- Features fail gracefully if a telemetry channel is missing
- One row per lap is the output unit; aggregation to driver level happens in dataset.py

Usage:
    from src.features import extract_features_from_session

    session = fastf1.get_session(2026, "Japan", "Q")
    session.load(telemetry=True)
    df = extract_features_from_session(session)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── FEATURE REGISTRY ──────────────────────────────────────────────────────────
#
# Each entry is a dict with:
#   "fn"          : callable(tel) -> float   — takes a telemetry DataFrame, returns a scalar
#   "description" : str                      — human readable explanation
#   "requires"    : list[str]                — telemetry channels needed (used for graceful skipping)
#
# To add a new feature, just add an entry here. Nothing else needs to change.
#
FEATURE_REGISTRY = {

    # ── Throttle ──────────────────────────────────────────────────────────────
    "throttle_mean": {
        "fn":          lambda tel: tel["Throttle"].mean(),
        "description": "Mean throttle position across the lap (0-100)",
        "requires":    ["Throttle"],
    },
    "throttle_pct_full": {
        "fn":          lambda tel: (tel["Throttle"] >= 99).mean() * 100,
        "description": "% of lap spent at full throttle (>=99)",
        "requires":    ["Throttle"],
    },
    "throttle_pct_off": {
        "fn":          lambda tel: (tel["Throttle"] <= 1).mean() * 100,
        "description": "% of lap spent off throttle (<=1)",
        "requires":    ["Throttle"],
    },

    # ── Brake ─────────────────────────────────────────────────────────────────
    "brake_pct": {
        "fn":          lambda tel: tel["Brake"].astype(float).mean() * 100,
        "description": "% of lap spent braking",
        "requires":    ["Brake"],
    },
    "brake_presses": {
        "fn":          lambda tel: int(
                           ((~tel["Brake"].astype(bool).shift(1, fill_value=False))
                            & tel["Brake"].astype(bool)).sum()
                       ),
        "description": "Number of discrete brake press events per lap",
        "requires":    ["Brake"],
    },

    # ── RPM ───────────────────────────────────────────────────────────────────
    "rpm_mean": {
        "fn":          lambda tel: tel["RPM"].mean(),
        "description": "Mean engine RPM across the lap",
        "requires":    ["RPM"],
    },
    "rpm_max": {
        "fn":          lambda tel: tel["RPM"].max(),
        "description": "Maximum engine RPM recorded in the lap",
        "requires":    ["RPM"],
    },

    # ── Gear ──────────────────────────────────────────────────────────────────
    "gear_mean": {
        "fn":          lambda tel: tel["nGear"].mean(),
        "description": "Mean gear across the lap (proxy for overall speed profile)",
        "requires":    ["nGear"],
    },
    "gear_changes": {
        "fn":          lambda tel: int((tel["nGear"].diff().abs() > 0).sum()),
        "description": "Total number of gear changes in the lap",
        "requires":    ["nGear"],
    },

    # ── Throttle application rate (time-to-full-throttle proxy) ───────────────
    # Mean rate of change when throttle is increasing — captures how aggressively
    # the driver/car applies power out of corners.
    "throttle_application_rate": {
        "fn":          lambda tel: (
                           tel["Throttle"]
                           .diff()
                           .clip(lower=0)   # only increasing throttle
                           .mean()
                       ),
        "description": "Mean rate of throttle increase — proxy for corner exit aggression",
        "requires":    ["Throttle"],
    },

    # ── Battery boost / Manual Override Energy (2026+) ────────────────────────
    # DRS was removed for 2026. The electrical deployment channel may appear
    # under different names depending on FastF1 version. Tried in priority order.
    "boost_pct": {
        "fn":          lambda tel: _boost_pct(tel),
        "description": "% of lap with battery boost / MOE active (2026+, skipped if unavailable)",
        "requires":    [],   # graceful skip handled inside _boost_pct
    },
    "boost_mean": {
        "fn":          lambda tel: _boost_mean(tel),
        "description": "Mean battery boost deployment level (2026+, skipped if unavailable)",
        "requires":    [],
    },
}

# ── BOOST HELPERS (2026+) ─────────────────────────────────────────────────────
# Centralised channel name resolution so both boost features stay in sync.

_BOOST_CANDIDATES = ["EngineBoost", "MGU_K_Deploy", "ERS_Deploy", "Boost"]


def _resolve_boost_col(tel):
    """Return the first available boost channel name, or None."""
    for col in _BOOST_CANDIDATES:
        if col in tel.columns:
            return col
    return None


def _boost_pct(tel):
    col = _resolve_boost_col(tel)
    if col is None:
        return np.nan
    return (tel[col] > 0).mean() * 100


def _boost_mean(tel):
    col = _resolve_boost_col(tel)
    if col is None:
        return np.nan
    return tel[col].mean()


# ── CORE EXTRACTION ───────────────────────────────────────────────────────────

def _compute_features_for_lap(tel: pd.DataFrame) -> dict:
    """
    Run every registered feature function against a single lap's telemetry.
    Returns a dict of {feature_name: value}.
    Features that raise an exception or require missing channels return NaN.
    """
    result = {}
    for name, spec in FEATURE_REGISTRY.items():
        # Skip if any required channel is missing
        if any(ch not in tel.columns for ch in spec["requires"]):
            result[name] = np.nan
            continue
        try:
            result[name] = spec["fn"](tel)
        except Exception:
            result[name] = np.nan
    return result


def extract_features_from_session(
    session,
    session_label: str = None,
) -> pd.DataFrame:
    """
    Extract per-lap telemetry features from a loaded FastF1 session.

    Parameters
    ----------
    session      : loaded fastf1.core.Session
    session_label: optional string to tag rows (e.g. "2026_Australia_Q")

    Returns
    -------
    DataFrame with one row per valid lap, columns:
        session_label, year, gp, session_type,
        driver, lap_number, lap_time_s, delta_to_session_best,
        + all features in FEATURE_REGISTRY
    """
    print(f"\n⏳ Extracting features from {session.event['EventName']} "
          f"{session.name} {session.event.year}...")

    # Session-level metadata
    year       = session.event.year
    gp         = session.event["EventName"]
    sess_type  = session.name
    label      = session_label or f"{year}_{gp}_{sess_type}".replace(" ", "_")

    # Use all laps for qualifying (pick_quicklaps filters too aggressively for Q)
    # but still drop laps with no recorded time
    laps = session.laps.dropna(subset=["LapTime"])

    # Session best lap time — used to compute delta
    session_best_s = laps["LapTime"].dt.total_seconds().min()

    records = []
    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()
            if tel is None or tel.empty or len(tel) < 50:
                continue

            lap_time_s = lap["LapTime"].total_seconds()
            if pd.isna(lap_time_s) or lap_time_s <= 0:
                continue

            features = _compute_features_for_lap(tel)

            record = {
                "session_label":         label,
                "year":                  year,
                "gp":                    gp,
                "session_type":          sess_type,
                "driver":                lap["Driver"],
                "lap_number":            int(lap["LapNumber"]),
                "lap_time_s":            lap_time_s,
                # Target variable: gap to the fastest lap in the session
                # Positive = slower than session best
                "delta_to_session_best": lap_time_s - session_best_s,
                **features,
            }
            records.append(record)

        except Exception:
            continue

    df = pd.DataFrame(records)

    if df.empty:
        print("⚠️  No valid laps extracted.")
        return df

    # Drop feature columns that are entirely NaN
    # (e.g. boost channels on pre-2026 sessions)
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        print(f"   Dropping {len(all_nan_cols)} unavailable feature(s): {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)

    n_drivers = df["driver"].nunique()
    n_laps    = len(df)
    print(f"✅ Extracted {n_laps} laps across {n_drivers} drivers.")
    print(f"   Active features: {[c for c in df.columns if c in FEATURE_REGISTRY]}")

    return df


def get_feature_names(df: pd.DataFrame) -> list:
    """Return the list of feature columns present in an extracted DataFrame."""
    return [c for c in df.columns if c in FEATURE_REGISTRY]


def describe_features() -> pd.DataFrame:
    """Print a summary of all registered features."""
    rows = [
        {
            "feature":     name,
            "requires":    ", ".join(spec["requires"]) or "—",
            "description": spec["description"],
        }
        for name, spec in FEATURE_REGISTRY.items()
    ]
    return pd.DataFrame(rows)
