"""
main.py
=======
Orchestrates the full F1 qualifying prediction pipeline.

Three modes:
    train    — build dataset from past races and train models
    predict  — predict an upcoming race from its qualifying session
    evaluate — score a completed prediction against actual results

Usage:
    python main.py train
    python main.py predict --year 2026 --gp Japan
    python main.py evaluate --year 2026 --gp Japan

Configuration is at the top of this file.
"""

import argparse
import pandas as pd

from src.dataset import build_dataset, load_saved_dataset
from src.model   import train
from src.predict import predict_race, evaluate_prediction


# ── CONFIG ────────────────────────────────────────────────────────────────────

# Races used for training — extend this list as the season progresses
TRAINING_RACES = [
    {"year": 2026, "gp": "Australia", "session": "Q"},
    {"year": 2026, "gp": "China",     "session": "Q"},
    # Add new races here after each weekend:
    # {"year": 2026, "gp": "Japan",     "session": "Q"},
]

CACHE_DIR   = "./f1_cache"
DATA_DIR    = "./data/processed"
MODELS_DIR  = "./models"

# ─────────────────────────────────────────────────────────────────────────────


def cmd_train(args):
    """Build dataset and train models."""
    print("\n🏎️  F1 PREDICTION PIPELINE — TRAIN MODE")

    # Build or reload dataset
    if args.reload:
        print("\n📂 Loading saved dataset...")
        lap_df, driver_df = load_saved_dataset(DATA_DIR)
    else:
        lap_df, driver_df = build_dataset(
            races     = TRAINING_RACES,
            cache_dir = CACHE_DIR,
            save_path = DATA_DIR,
        )

    if driver_df.empty:
        print("❌ No training data available.")
        return

    # Train
    ranking_model, podium_model, feature_cols, importance_df = train(
        driver_df      = driver_df,
        save_dir       = MODELS_DIR,
        top_n_features = args.top_n,
        run_evaluation = not args.skip_eval,
    )

    print("\n✅ Training complete.")


def cmd_predict(args):
    """Predict an upcoming race."""
    print(f"\n🏎️  F1 PREDICTION PIPELINE — PREDICT MODE")
    print(f"   {args.year} {args.gp}")

    predictions = predict_race(
        year      = args.year,
        gp        = args.gp,
        save_dir  = MODELS_DIR,
        cache_dir = CACHE_DIR,
    )

    # Save predictions for later evaluation
    out_path = f"{DATA_DIR}/predictions_{args.year}_{args.gp}.csv"
    predictions.to_csv(out_path, index=False)
    print(f"\n💾 Predictions saved → {out_path}")


def cmd_evaluate(args):
    """Score a completed prediction against actual results."""
    print(f"\n🏎️  F1 PREDICTION PIPELINE — EVALUATE MODE")
    print(f"   {args.year} {args.gp}")

    pred_path = f"{DATA_DIR}/predictions_{args.year}_{args.gp}.csv"
    try:
        predictions = pd.read_csv(pred_path)
    except FileNotFoundError:
        print(f"❌ No saved predictions found at {pred_path}.")
        print(   "   Run predict mode first.")
        return

    metrics = evaluate_prediction(predictions)
    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="F1 Qualifying Telemetry Prediction Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # train
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--reload", action="store_true",
        help="Reload saved dataset instead of re-downloading telemetry"
    )
    train_parser.add_argument(
        "--top-n", type=int, default=8,
        help="Number of features to select dynamically (default: 8)"
    )
    train_parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip leave-one-race-out evaluation during training"
    )

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict a race")
    predict_parser.add_argument("--year", type=int, required=True)
    predict_parser.add_argument("--gp",   type=str, required=True)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a prediction")
    eval_parser.add_argument("--year", type=int, required=True)
    eval_parser.add_argument("--gp",   type=str, required=True)

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
