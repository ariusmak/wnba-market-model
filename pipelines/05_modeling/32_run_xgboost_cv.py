"""
32_run_xgboost_cv.py

Walk-forward cross-validation for the XGBoost correction model.
Evaluation protocol: nested early stopping (no test-year leakage).

For each test_year Y in 2020..2024:
  1. ES-train on 2015..Y-2, ES-val on Y-1  → find best_round
  2. Retrain on 2015..Y-1 for exactly best_round trees (no early stopping)
  3. Predict on Y → honest OOF log-loss

Feature hyperparam (affects column selection only, no dataset rebuild needed):
  --n-players       N_players  ∈ {5, 7, 10}   player slots included per side

Model hyperparams (tuned defaults — confirmed by XGB_tuning3 + follow-up grid):
  --max-depth               default 6      (tuning3: d=6 >> d=5,7,8,9)
  --min-child-weight        default 2      (tuning3: mcw=2 best; mcw=1 excluded from search)
  --subsample               default 0.9   (fixed throughout tuning)
  --colsample-bytree        default 0.8   (tuning3: cbt=0.8 > 0.6 > 1.0)
  --reg-lambda              default 0.5   (tuning3: rl=0.5 dominated; lower values worse)
  --reg-alpha               default 0.0   (fixed throughout tuning)
  --gamma                   default 0.5   (tuning3: monotone 0→0.1→0.5; follow-up confirmed 0.5 is peak)
  --learning-rate           default 0.03  (tuning3: monotone 0.01→0.02→0.03; follow-up confirmed 0.03 is peak)
  --num-boost-round         default 3000  (ES terminates early; ~16 rounds at lr=0.03)
  --early-stopping-rounds   default 150

Usage:
  python 32_run_xgboost_cv.py \\
    --gold-dir data/gold/variants/hM5_Linj14 \\
    --n-players 7 \\
    --max-depth 3 --min-child-weight 1 \\
    --subsample 0.8 --colsample-bytree 0.8 \\
    --reg-lambda 1 --reg-alpha 0 --gamma 0 \\
    --learning-rate 0.03 \\
    --num-boost-round 3000 --early-stopping-rounds 75

  # Or point at the default gold dir:
  python 32_run_xgboost_cv.py --gold-dir data/gold --n-players 7
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, brier_score_loss

# ---------------------------------------------------------------------------
# Constants / column lists (must match 30_build_game_xgboost_input.py)
# ---------------------------------------------------------------------------

PLAYER_MODEL_FEATURES = [
    "m_ewma_pre",
    "q_pre",
    "days_since_first_report_pre",
    "days_since_last_dnp_pre",
    "consec_dnps_pre",
    "played_last_game_pre",
    "minutes_last_game_pre",
    "days_since_last_played_pre",
    "injury_present_flag_pre",
]

PLAYER_DEBUG_FEATURES = ["player_id", "player_name", "strength_pre"]

RECENT_FORM_FEATURES = [
    "net_rtg_ewma_pre",
    "efg_ewma_pre",
    "tov_pct_ewma_pre",
    "orb_pct_ewma_pre",
    "ftr_ewma_pre",
]

STYLE_FEATURES = [
    "off_3pa_rate_pre",
    "def_3pa_allowed_pre",
    "off_2pa_rate_pre",
    "def_2pa_allowed_pre",
    "off_tov_pct_pre",
    "def_forced_tov_pre",
]

SCHEDULE_FEATURES = [
    "days_rest_pre",
    "is_b2b_pre",
    "games_last_4_days_pre",
    "games_last_7_days_pre",
    "travel_miles_pre",
    "timezone_shift_hours_pre",
]

# Columns never used as XGBoost features
METADATA_COLS = {
    "game_id", "game_ts", "game_date", "season", "is_playoff",
    "home_team_id", "away_team_id", "home_franchise_id", "away_franchise_id",
    "home_elo_pre", "away_elo_pre", "p_elo", "base_margin", "home_win",
    "home_origin_city_pre", "away_origin_city_pre",
    "home_current_city_pre", "away_current_city_pre",
}


# ---------------------------------------------------------------------------
# Feature column builder
# ---------------------------------------------------------------------------

def build_feature_cols(n_players: int) -> list[str]:
    """
    Return the ordered list of XGBoost feature columns for a given N_players.
    Player slots p1..n_players are included; p(n_players+1)..12 are dropped.
    Debug player columns (player_id, player_name, strength_pre) are excluded.
    """
    cols = []
    for side in ("home", "away"):
        for slot in range(1, n_players + 1):
            for feat in PLAYER_MODEL_FEATURES:
                cols.append(f"{side}_p{slot}_{feat}")

    for feat in RECENT_FORM_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")

    for feat in STYLE_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")

    for feat in SCHEDULE_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")

    return cols


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_gold(gold_dir: Path, season: int) -> pd.DataFrame:
    p = gold_dir / f"game_xgboost_input_{season}_REGPST.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing gold file: {p}")
    df = pd.read_csv(p)
    df["season"] = df["season"].astype(int)

    # Exclude cold-start games where either team's p1 has m_ewma=0.
    # This affects only the first ~9 games of 2015 (no prior-season seed data).
    cold_start = (df["home_p1_m_ewma_pre"] == 0) | (df["away_p1_m_ewma_pre"] == 0)
    n_dropped = cold_start.sum()
    if n_dropped:
        df = df[~cold_start].reset_index(drop=True)
        print(f"  [load_gold {season}] dropped {n_dropped} cold-start rows (p1 m_ewma=0)")

    return df


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def run_cv(
    gold_dir: Path,
    n_players: int,
    # model hyperparams
    max_depth: int,
    min_child_weight: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    reg_alpha: float,
    gamma: float,
    learning_rate: float,
    num_boost_round: int,
    early_stopping_rounds: int,
    nthread: int = -1,
    train_start: int = 2015,
    test_years: list[int] = None,
) -> dict:
    if test_years is None:
        test_years = list(range(2020, 2025))

    feature_cols = build_feature_cols(n_players)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "seed": 42,
        "nthread": nthread,
    }

    fold_results = []

    for test_year in test_years:
        es_val_year = test_year - 1
        es_train_years = list(range(train_start, es_val_year))

        # --- Step 1: build ES-train (2015..Y-2) and ES-val (Y-1) ---
        es_train_dfs = []
        for y in es_train_years:
            try:
                es_train_dfs.append(load_gold(gold_dir, y))
            except FileNotFoundError as e:
                print(f"  WARNING: {e} — skipping year {y}")
        if not es_train_dfs:
            print(f"  SKIP fold test_year={test_year}: no ES-train data")
            continue

        try:
            es_val_df = load_gold(gold_dir, es_val_year)
        except FileNotFoundError as e:
            print(f"  SKIP fold test_year={test_year}: missing ES-val year {es_val_year}: {e}")
            continue

        try:
            test_df = load_gold(gold_dir, test_year)
        except FileNotFoundError as e:
            print(f"  SKIP fold test_year={test_year}: {e}")
            continue

        es_train_df = pd.concat(es_train_dfs, ignore_index=True)

        # Drop rows with missing target/base_margin
        es_train_df = es_train_df.dropna(subset=["home_win", "base_margin"])
        es_val_df   = es_val_df.dropna(subset=["home_win", "base_margin"])
        test_df     = test_df.dropna(subset=["home_win", "base_margin"])

        avail_feats = [c for c in feature_cols if c in es_train_df.columns]
        missing = set(feature_cols) - set(avail_feats)
        if missing:
            print(f"  WARNING fold {test_year}: {len(missing)} feature cols missing")

        def make_dm(df):
            return xgb.DMatrix(
                df[avail_feats].values.astype(float),
                label=df["home_win"].values.astype(float),
                base_margin=df["base_margin"].values.astype(float),
                feature_names=avail_feats, missing=np.nan,
            )

        dm_es_train = make_dm(es_train_df)
        dm_es_val   = make_dm(es_val_df)

        # Early-stop run to determine best_round
        es_model = xgb.train(
            xgb_params,
            dm_es_train,
            num_boost_round=num_boost_round,
            evals=[(dm_es_val, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        best_round = es_model.best_iteration + 1

        # --- Step 2: retrain on full 2015..Y-1 for best_round trees ---
        full_train_dfs = es_train_dfs + [es_val_df]
        full_train_df  = pd.concat(full_train_dfs, ignore_index=True)
        full_train_df  = full_train_df.dropna(subset=["home_win", "base_margin"])

        dm_full_train = make_dm(full_train_df)
        dm_test       = make_dm(test_df)

        final_model = xgb.train(
            xgb_params,
            dm_full_train,
            num_boost_round=best_round,
            verbose_eval=False,
        )

        preds = final_model.predict(dm_test)
        preds = np.clip(preds, 1e-7, 1 - 1e-7)
        y_test = test_df["home_win"].values.astype(float)

        ll    = log_loss(y_test, preds)
        brier = brier_score_loss(y_test, preds)

        fold_results.append({
            "test_year": test_year,
            "n_es_train": len(es_train_df),
            "n_full_train": len(full_train_df),
            "n_test": len(test_df),
            "log_loss": ll,
            "brier": brier,
            "best_round": best_round,
        })
        print(f"  fold {test_year}: logloss={ll:.5f}  brier={brier:.5f}  "
              f"best_round={best_round}  n_full_train={len(full_train_df)}  n_test={len(test_df)}")

    if not fold_results:
        return {"error": "no folds completed"}

    log_losses   = [r["log_loss"]    for r in fold_results]
    briers       = [r["brier"]       for r in fold_results]
    best_rounds  = [r["best_round"]  for r in fold_results]

    summary = {
        "mean_log_loss":   float(np.mean(log_losses)),
        "std_log_loss":    float(np.std(log_losses)),
        "mean_brier":      float(np.mean(briers)),
        "mean_best_round": float(np.mean(best_rounds)),
        "folds": fold_results,
        "config": {
            "gold_dir": str(gold_dir),
            "n_players": n_players,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "num_boost_round": num_boost_round,
            "early_stopping_rounds": early_stopping_rounds,
        },
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Walk-forward XGBoost CV. See module docstring for full param list."
    )
    # Data
    ap.add_argument("--gold-dir", type=Path, default=Path("data/gold"),
                    help="Directory containing game_xgboost_input_{year}_REGPST.csv files")
    ap.add_argument("--output-json", type=Path, default=None,
                    help="If set, write JSON results to this path")

    # Feature hyperparam
    ap.add_argument("--n-players", type=int, default=7,
                    help="Player slots per side to include as features {5,7,10}")

    # Model hyperparams — tuned defaults (XGB_tuning3 + follow-up grid)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-child-weight", type=int, default=2)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=0.5)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--learning-rate", type=float, default=0.03)
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--early-stopping-rounds", type=int, default=150)
    ap.add_argument("--nthread", type=int, default=-1,
                    help="XGBoost nthread (-1 = all cores)\")")

    # CV window
    ap.add_argument("--train-start", type=int, default=2015)
    ap.add_argument("--test-years", type=int, nargs="+", default=None,
                    help="Test years (default: 2020 2021 2022 2023 2024)")

    args = ap.parse_args()

    print("=" * 60)
    print(f"gold_dir      : {args.gold_dir}")
    print(f"n_players     : {args.n_players}")
    print(f"max_depth     : {args.max_depth}")
    print(f"min_child_wt  : {args.min_child_weight}")
    print(f"subsample     : {args.subsample}")
    print(f"colsample_bt  : {args.colsample_bytree}")
    print(f"reg_lambda    : {args.reg_lambda}")
    print(f"reg_alpha     : {args.reg_alpha}")
    print(f"gamma         : {args.gamma}")
    print(f"learning_rate : {args.learning_rate}")
    print(f"num_boost_rnd : {args.num_boost_round}")
    print(f"early_stop    : {args.early_stopping_rounds}")
    print("=" * 60)

    results = run_cv(
        gold_dir=args.gold_dir,
        n_players=args.n_players,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        nthread=args.nthread,
        train_start=args.train_start,
        test_years=args.test_years,
    )

    print("\n--- Summary ---")
    print(f"mean_log_loss    : {results.get('mean_log_loss', 'N/A'):.5f}")
    print(f"std_log_loss     : {results.get('std_log_loss', 'N/A'):.5f}")
    print(f"mean_brier       : {results.get('mean_brier', 'N/A'):.5f}")
    print(f"mean_best_round  : {results.get('mean_best_round', 'N/A'):.0f}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nwrote: {args.output_json}")


if __name__ == "__main__":
    main()
