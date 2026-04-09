"""
33_create_ensemble.py

Bootstrap-aggregated XGBoost ensemble (bagging).

Trains N_MODELS identical XGBoost models, each on a bootstrap resample
(sampled with replacement, n = total training observations) of the full
training set. Ensemble prediction is the mean of all model outputs.

Inputs
------
Feature-level hyperparams:
  --gold-dir          Path to gold variant directory
  --n-players         Player slots per side {5, 7, 10}

Model-level hyperparams (defaults = tuning3 rank-2 winner):
  --max-depth              default 6
  --min-child-weight       default 3
  --subsample              default 0.8
  --colsample-bytree       default 0.6
  --reg-lambda             default 1.0
  --reg-alpha              default 0.0
  --gamma                  default 0.1
  --learning-rate          default 0.02
  --num-boost-round        default 64   (mean_best_round from CV)
  --n-models               default 20
  --seed                   default 42   (RNG seed for bootstrap draws)

Training / evaluation:
  --train-years   e.g.  2015 2016 ... 2024
  --test-years    e.g.  2025
  --out-dir       directory to save models, predictions, config

Outputs (all written to --out-dir)
-------
  model_00.ubj … model_19.ubj   — individual XGBoost models
  config.json                   — all hyperparams + training metadata
  ensemble_predictions.csv      — per-game predictions on test years
  performance.json              — LL / Brier / accuracy vs Elo

Usage
-----
  python notebooks/33_create_ensemble.py \\
    --gold-dir data/gold/stage1/hM7_Linj14_tau150_hT7 \\
    --n-players 7 \\
    --max-depth 6 --min-child-weight 3 \\
    --subsample 0.8 --colsample-bytree 0.6 \\
    --reg-lambda 1.0 --reg-alpha 0.0 --gamma 0.1 \\
    --learning-rate 0.02 --num-boost-round 64 \\
    --n-models 20 --seed 42 \\
    --train-years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \\
    --test-years 2025 \\
    --out-dir data/ensemble/default
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss

# ---------------------------------------------------------------------------
# Feature column builder (identical to 32_run_xgboost_cv.py)
# ---------------------------------------------------------------------------

PLAYER_MODEL_FEATURES = [
    "m_ewma_pre", "q_pre", "days_since_first_report_pre",
    "days_since_last_dnp_pre", "consec_dnps_pre", "played_last_game_pre",
    "minutes_last_game_pre", "days_since_last_played_pre",
    "injury_present_flag_pre",
]
RECENT_FORM_FEATURES = [
    "net_rtg_ewma_pre", "efg_ewma_pre", "tov_pct_ewma_pre",
    "orb_pct_ewma_pre", "ftr_ewma_pre",
]
STYLE_FEATURES = [
    "off_3pa_rate_pre", "def_3pa_allowed_pre", "off_2pa_rate_pre",
    "def_2pa_allowed_pre", "off_tov_pct_pre", "def_forced_tov_pre",
]
SCHEDULE_FEATURES = [
    "days_rest_pre", "is_b2b_pre", "games_last_4_days_pre",
    "games_last_7_days_pre", "travel_miles_pre", "timezone_shift_hours_pre",
]


def build_feature_cols(n_players: int) -> list[str]:
    cols = []
    for side in ("home", "away"):
        for slot in range(1, n_players + 1):
            for feat in PLAYER_MODEL_FEATURES:
                cols.append(f"{side}_p{slot}_{feat}")
    for feat in RECENT_FORM_FEATURES + STYLE_FEATURES + SCHEDULE_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")
    return cols


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold(gold_dir: Path, season: int) -> pd.DataFrame:
    p = gold_dir / f"game_xgboost_input_{season}_REGPST.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing gold file: {p}")
    df = pd.read_csv(p)
    df["season"] = df["season"].astype(int)
    cold_start = (df["home_p1_m_ewma_pre"] == 0) | (df["away_p1_m_ewma_pre"] == 0)
    if cold_start.any():
        df = df[~cold_start].reset_index(drop=True)
    return df


def make_dmatrix(df: pd.DataFrame, feature_cols: list[str]) -> xgb.DMatrix:
    avail = [c for c in feature_cols if c in df.columns]
    dm = xgb.DMatrix(
        df[avail].values.astype(float),
        label=df["home_win"].values.astype(float) if "home_win" in df.columns else None,
        base_margin=df["base_margin"].values.astype(float),
        feature_names=avail,
        missing=np.nan,
    )
    return dm


# ---------------------------------------------------------------------------
# Bootstrap ensemble
# ---------------------------------------------------------------------------

def bootstrap_dmatrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    rng: np.random.Generator,
) -> xgb.DMatrix:
    """Sample df with replacement (same n) and return a DMatrix."""
    idx = rng.integers(0, len(df), size=len(df))
    return make_dmatrix(df.iloc[idx].reset_index(drop=True), feature_cols)


def train_ensemble(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    num_boost_round: int,
    n_models: int,
    seed: int,
) -> list[xgb.Booster]:
    """
    Train n_models XGBoost models on bootstrap resamples of train_df.
    Returns list of trained Booster objects.
    """
    models = []
    for i in range(n_models):
        model_seed = seed + i
        rng = np.random.default_rng(model_seed)
        dm = bootstrap_dmatrix(train_df, feature_cols, rng)
        model = xgb.train(
            {**xgb_params, "seed": model_seed},
            dm,
            num_boost_round=num_boost_round,
            verbose_eval=False,
        )
        models.append(model)
        print(f"  trained model {i+1:2d}/{n_models}  (seed={model_seed})")
    return models


def ensemble_predict(
    models: list[xgb.Booster],
    dm: xgb.DMatrix,
) -> np.ndarray:
    """Average predictions across all models."""
    preds = np.stack([m.predict(dm) for m in models], axis=0)
    return preds.mean(axis=0)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, p_pred: np.ndarray, label: str) -> dict:
    p = np.clip(p_pred, 1e-7, 1 - 1e-7)
    ll = log_loss(y_true, p)
    brier = brier_score_loss(y_true, p)
    acc = float(np.mean((p >= 0.5) == y_true.astype(bool)))
    print(f"  {label:<20}  LL={ll:.5f}  Brier={brier:.5f}  Acc={acc:.4f}")
    return {"log_loss": ll, "brier": brier, "accuracy": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Bootstrap XGBoost ensemble.")

    # Data
    ap.add_argument("--gold-dir", type=Path, default=Path("data/gold/stage1/hM7_Linj14_tau150_hT7"))
    ap.add_argument("--n-players", type=int, default=7)
    ap.add_argument("--train-years", type=int, nargs="+",
                    default=list(range(2015, 2025)))
    ap.add_argument("--test-years", type=int, nargs="+", default=[2025])
    ap.add_argument("--out-dir", type=Path, default=Path("data/ensemble/default"))

    # Model hyperparams (tuning3 rank-2 defaults)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-child-weight", type=int, default=3)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.6)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--learning-rate", type=float, default=0.02)
    ap.add_argument("--num-boost-round", type=int, default=64)

    # Ensemble
    ap.add_argument("--n-models", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"gold_dir       : {args.gold_dir}")
    print(f"n_players      : {args.n_players}")
    print(f"train_years    : {args.train_years}")
    print(f"test_years     : {args.test_years}")
    print(f"max_depth      : {args.max_depth}")
    print(f"min_child_wt   : {args.min_child_weight}")
    print(f"subsample      : {args.subsample}")
    print(f"colsample_bt   : {args.colsample_bytree}")
    print(f"reg_lambda     : {args.reg_lambda}")
    print(f"reg_alpha      : {args.reg_alpha}")
    print(f"gamma          : {args.gamma}")
    print(f"learning_rate  : {args.learning_rate}")
    print(f"num_boost_round: {args.num_boost_round}")
    print(f"n_models       : {args.n_models}")
    print(f"seed           : {args.seed}")
    print("=" * 60)

    feature_cols = build_feature_cols(args.n_players)
    print(f"\nFeature cols: {len(feature_cols)}")

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "nthread": -1,
    }

    # --- Load training data ---
    print("\nLoading training data...")
    train_dfs = []
    for yr in args.train_years:
        try:
            train_dfs.append(load_gold(args.gold_dir, yr))
            print(f"  loaded {yr}: {len(train_dfs[-1])} rows")
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    if not train_dfs:
        raise RuntimeError("No training data loaded.")
    train_df = pd.concat(train_dfs, ignore_index=True)
    train_df = train_df.dropna(subset=["home_win", "base_margin"])
    print(f"  total training rows: {len(train_df)}")

    # --- Train ensemble ---
    print(f"\nTraining {args.n_models} bootstrap models ({len(train_df)} obs each)...")
    models = train_ensemble(
        train_df, feature_cols, xgb_params,
        num_boost_round=args.num_boost_round,
        n_models=args.n_models,
        seed=args.seed,
    )

    # --- Save models ---
    print("\nSaving models...")
    model_paths = []
    for i, model in enumerate(models):
        path = args.out_dir / f"model_{i:02d}.ubj"
        model.save_model(str(path))
        model_paths.append(str(path))
    print(f"  saved {len(models)} models to {args.out_dir}")

    # --- Evaluate on test years ---
    print("\nEvaluating on test years...")
    pred_records = []
    perf_by_year = {}

    for yr in args.test_years:
        try:
            test_df = load_gold(args.gold_dir, yr)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue
        test_df = test_df.dropna(subset=["home_win", "base_margin"])
        dm_test = make_dmatrix(test_df, feature_cols)

        p_ensemble = np.clip(ensemble_predict(models, dm_test), 1e-7, 1 - 1e-7)
        p_elo = np.clip(test_df["p_elo"].values, 1e-7, 1 - 1e-7)
        y = test_df["home_win"].values.astype(float)

        print(f"\n  Year {yr} (n={len(test_df)}):")
        perf_elo = evaluate(y, p_elo, "Elo")
        perf_ens = evaluate(y, p_ensemble, "Ensemble XGB")

        perf_by_year[yr] = {
            "n": len(test_df),
            "elo": perf_elo,
            "ensemble": perf_ens,
            "delta_ll": perf_ens["log_loss"] - perf_elo["log_loss"],
        }

        # Individual model predictions for variance estimate
        dm_test2 = make_dmatrix(test_df, feature_cols)
        individual_preds = np.stack(
            [np.clip(m.predict(dm_test2), 1e-7, 1 - 1e-7) for m in models], axis=0
        )
        pred_std = individual_preds.std(axis=0)

        out = test_df[["game_id", "game_date", "home_team_id", "away_team_id",
                        "season", "home_win", "p_elo", "base_margin"]].copy()
        out["p_ensemble"] = p_ensemble
        out["pred_std"]   = pred_std
        out["test_year"]  = yr
        pred_records.append(out)

    # --- Summary ---
    print("\n--- Summary ---")
    for yr, perf in perf_by_year.items():
        delta = perf["delta_ll"]
        print(f"  {yr}: LL={perf['ensemble']['log_loss']:.5f}  "
              f"delta_vs_elo={delta:+.5f}")

    # --- Save predictions and config ---
    if pred_records:
        preds_df = pd.concat(pred_records, ignore_index=True)
        preds_path = args.out_dir / "ensemble_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        print(f"\nSaved predictions: {preds_path}")

    config = {
        "gold_dir": str(args.gold_dir),
        "n_players": args.n_players,
        "train_years": args.train_years,
        "test_years": args.test_years,
        "n_models": args.n_models,
        "seed": args.seed,
        "num_boost_round": args.num_boost_round,
        "xgb_params": xgb_params,
        "feature_cols": feature_cols,
        "model_paths": model_paths,
        "n_train": len(train_df),
    }
    config_path = args.out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))

    perf_path = args.out_dir / "performance.json"
    perf_path.write_text(json.dumps(perf_by_year, indent=2))

    print(f"Saved config   : {config_path}")
    print(f"Saved perf     : {perf_path}")

    return models, preds_df if pred_records else None, perf_by_year


# ---------------------------------------------------------------------------
# Public API for import
# ---------------------------------------------------------------------------

def load_ensemble(out_dir: Path) -> tuple[list[xgb.Booster], dict]:
    """
    Load a saved ensemble from disk.
    Returns (models, config).
    """
    config = json.loads((out_dir / "config.json").read_text())
    models = []
    for path in config["model_paths"]:
        m = xgb.Booster()
        m.load_model(path)
        models.append(m)
    return models, config


def predict_from_ensemble(
    models: list[xgb.Booster],
    df: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Generate ensemble predictions for an arbitrary DataFrame.
    df must contain base_margin and all feature_cols (NaN OK).
    Returns array of win probabilities (home team).
    """
    dm = make_dmatrix(df, feature_cols)
    return np.clip(ensemble_predict(models, dm), 1e-7, 1 - 1e-7)


if __name__ == "__main__":
    main()
