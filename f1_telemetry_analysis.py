"""
F1 Telemetry Correlation Analysis
===================================
Analyzes which telemetry variables are most closely correlated
with lap time for a given race session using FastF1.

Requirements:
    pip install fastf1 matplotlib seaborn scikit-learn pandas numpy

Usage:
    python f1_telemetry_analysis.py
    
    Or customize the config at the top of the file.
"""

import os
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
YEAR        = 2024
GRAND_PRIX  = "Monza"       # e.g. "Monza", "Silverstone", "Monaco"
SESSION     = "R"           # R = Race, Q = Qualifying, FP1/FP2/FP3 = Practice
CACHE_DIR   = "./f1_cache"  # Local cache to speed up repeated runs
TOP_N       = 10            # How many top correlations to highlight
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def load_session(year, gp, session_type):
    """Load and return a FastF1 session."""
    print(f"\n⏳ Loading {year} {gp} {session_type} session...")
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    print(f"✅ Loaded. {len(session.laps)} total laps found.")
    return session


def _extract_boost(tel):
    """
    Try to extract the 2026 battery boost / Manual Override Energy channel.
    FastF1's exact channel name for this isn't finalised yet, so we try a
    priority list and return whichever is present. Returns empty dict if none found.
    """
    candidates = ["EngineBoost", "MGU_K_Deploy", "ERS_Deploy", "Boost"]
    for col in candidates:
        if col in tel.columns:
            return {
                "Boost_pct":  (tel[col] > 0).mean() * 100,   # % time boost active
                "Boost_mean": tel[col].mean(),
            }
    return {}   # pre-2026 or channel not yet mapped — just skip


def extract_telemetry_features(session):
    """
    For each valid lap, compute aggregate telemetry statistics
    and return a flat DataFrame with one row per lap.
    """
    print("\n⏳ Extracting telemetry features per lap (this may take a minute)...")

    records = []
    laps = session.laps.pick_quicklaps()  # Drop outliers / in/out laps

    for _, lap in laps.iterlaps():
        try:
            tel = lap.get_telemetry()
            if tel.empty or len(tel) < 50:
                continue

            lap_time_s = lap["LapTime"].total_seconds()
            if pd.isna(lap_time_s) or lap_time_s <= 0:
                continue

            record = {
                "LapTime_s":         lap_time_s,
                "Driver":            lap["Driver"],
                "LapNumber":         lap["LapNumber"],

                # ── Throttle (0–100) ───────────────────────────────────────
                "Throttle_mean":     tel["Throttle"].mean(),
                "Throttle_pct_full": (tel["Throttle"] >= 99).mean() * 100,
                "Throttle_pct_off":  (tel["Throttle"] <= 1).mean() * 100,

                # ── Brake ─────────────────────────────────────────────────
                "Brake_pct":         tel["Brake"].astype(float).mean() * 100,

                # ── RPM ────────────────────────────────────────────────────
                "RPM_mean":          tel["RPM"].mean(),
                "RPM_max":           tel["RPM"].max(),

                # ── Gear ───────────────────────────────────────────────────
                "nGear_mean":        tel["nGear"].mean(),
                "nGear_changes":     (tel["nGear"].diff().abs() > 0).sum(),

                # ── Battery Boost / Manual Override Energy (2026+) ─────────
                # DRS removed in 2026; replaced by push-to-pass boost.
                # Skipped silently for pre-2026 sessions.
                **_extract_boost(tel),
            }
            records.append(record)

        except Exception:
            continue

    df = pd.DataFrame(records)
    print(f"✅ Extracted features for {len(df)} laps across {df['Driver'].nunique()} drivers.")
    return df


def compute_correlations(df):
    """Pearson correlation between each telemetry feature and LapTime."""
    feature_cols = [c for c in df.columns if c not in ("LapTime_s", "Driver", "LapNumber")]
    corr = df[feature_cols + ["LapTime_s"]].corr()["LapTime_s"].drop("LapTime_s")
    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation"]
    corr_df["AbsCorr"] = corr_df["Correlation"].abs()
    corr_df = corr_df.sort_values("AbsCorr", ascending=False).reset_index(drop=True)
    return corr_df


def compute_feature_importance(df):
    """Random Forest feature importance as a secondary ranking method."""
    feature_cols = [c for c in df.columns if c not in ("LapTime_s", "Driver", "LapNumber")]
    X = df[feature_cols].dropna()
    y = df.loc[X.index, "LapTime_s"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)

    imp_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp_df


def plot_results(df, corr_df, imp_df, year, gp, session_type):
    """Generate a multi-panel figure summarising the analysis."""
    fig = plt.figure(figsize=(18, 14), facecolor="#0f0f0f")
    fig.suptitle(
        f"F1 Telemetry → Lap Time Analysis\n{year} {gp} · {session_type}",
        fontsize=18, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    RED   = "#e8002d"
    GOLD  = "#ffd700"
    GREY  = "#888888"
    WHITE = "#f0f0f0"
    BG    = "#1a1a1a"

    # ── Panel 1: Correlation bar chart ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(BG)
    top = corr_df.head(TOP_N).copy()
    colors = [RED if v < 0 else GOLD for v in top["Correlation"]]
    bars = ax1.barh(top["Feature"][::-1], top["Correlation"][::-1], color=colors[::-1], edgecolor="none")
    ax1.axvline(0, color=GREY, linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Pearson Correlation with Lap Time", color=WHITE, fontsize=9)
    ax1.set_title(f"Top {TOP_N} Correlations", color=WHITE, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=WHITE, labelsize=8)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#333333")
    ax1.text(0.98, 0.02, "Red = faster  Gold = slower",
             transform=ax1.transAxes, color=GREY, fontsize=7, ha="right")

    # ── Panel 2: RF Feature importance ───────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(BG)
    top_imp = imp_df.head(TOP_N)
    ax2.barh(top_imp["Feature"][::-1], top_imp["Importance"][::-1], color=GOLD, edgecolor="none")
    ax2.set_xlabel("Random Forest Importance", color=WHITE, fontsize=9)
    ax2.set_title(f"Top {TOP_N} RF Feature Importances", color=WHITE, fontsize=11, fontweight="bold")
    ax2.tick_params(colors=WHITE, labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#333333")

    # ── Panel 3: Scatter – best correlated feature vs lap time ───────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(BG)
    best_feature = corr_df.iloc[0]["Feature"]
    drivers = df["Driver"].unique()
    cmap = plt.cm.get_cmap("tab20", len(drivers))
    for i, drv in enumerate(drivers):
        sub = df[df["Driver"] == drv]
        ax3.scatter(sub[best_feature], sub["LapTime_s"],
                    label=drv, alpha=0.7, s=25, color=cmap(i))
    ax3.set_xlabel(best_feature, color=WHITE, fontsize=9)
    ax3.set_ylabel("Lap Time (s)", color=WHITE, fontsize=9)
    ax3.set_title(f"Strongest Predictor: {best_feature}", color=WHITE, fontsize=11, fontweight="bold")
    ax3.tick_params(colors=WHITE, labelsize=8)
    ax3.legend(fontsize=6, ncol=2, facecolor="#2a2a2a", edgecolor="#444",
               labelcolor="white", loc="upper right")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333333")

    # ── Panel 4: Full correlation heatmap (all features) ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(BG)
    feature_cols = [c for c in df.columns if c not in ("Driver", "LapNumber")]
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, ax=ax4,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor="#111",
        cbar_kws={"shrink": 0.7},
        annot=False
    )
    ax4.set_title("Feature Correlation Matrix", color=WHITE, fontsize=11, fontweight="bold")
    ax4.tick_params(colors=WHITE, labelsize=6)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")

    plt.savefig("f1_telemetry_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n📊 Plot saved → f1_telemetry_analysis.png")
    plt.show()


def print_summary(corr_df, imp_df):
    """Print a readable summary table."""
    print("\n" + "═" * 55)
    print("  TELEMETRY VARIABLE CORRELATION WITH LAP TIME")
    print("═" * 55)
    print(f"  {'Rank':<5} {'Feature':<25} {'Correlation':>11}  {'Direction'}")
    print("─" * 55)
    for i, row in corr_df.head(TOP_N).iterrows():
        direction = "↓ faster laps" if row["Correlation"] < 0 else "↑ slower laps"
        print(f"  {i+1:<5} {row['Feature']:<25} {row['Correlation']:>+.4f}   {direction}")
    print("═" * 55)

    print("\n" + "═" * 55)
    print("  RANDOM FOREST FEATURE IMPORTANCE RANKING")
    print("═" * 55)
    print(f"  {'Rank':<5} {'Feature':<25} {'Importance':>10}")
    print("─" * 55)
    for i, row in imp_df.head(TOP_N).iterrows():
        print(f"  {i+1:<5} {row['Feature']:<25} {row['Importance']:>10.4f}")
    print("═" * 55)


def main():
    session  = load_session(YEAR, GRAND_PRIX, SESSION)
    df       = extract_telemetry_features(session)

    if df.empty:
        print("❌ No valid laps found. Try a different session.")
        return

    corr_df  = compute_correlations(df)
    imp_df   = compute_feature_importance(df)

    print_summary(corr_df, imp_df)
    plot_results(df, corr_df, imp_df, YEAR, GRAND_PRIX, SESSION)

    # Optionally save the raw feature data
    df.to_csv("f1_lap_features.csv", index=False)
    print("💾 Raw features saved → f1_lap_features.csv")


if __name__ == "__main__":
    main()
