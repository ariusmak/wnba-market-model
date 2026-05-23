"""Final model: train the locked Elo + XGBoost pipeline and emit predictions.

Reproduces the exact protocol used in `trading_results2.ipynb` and
`return_investigation.ipynb`:

  1. Train on 2015..(holdout_year - 2) with early stopping on year (holdout_year - 1).
  2. Refit on 2015..(holdout_year - 1) for `best_round` trees.
  3. Predict on `holdout_year` (default 2025) using base_margin = logit(p_elo).

Usage
-----
    # Train + predict 2025 holdout, write CSV + model artifact
    python -m organized.pipelines.05_modeling.final_model \
        --holdout-year 2025 \
        --out-csv organized/outputs/final_model_predictions_2025.csv \
        --model-out organized/outputs/final_model_2025.json

    # Use as a library
    from organized.pipelines.modeling.final_model import train_final_model
    booster, preds = train_final_model(holdout_year=2025)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR_DEFAULT = PROJECT_ROOT / "data" / "gold"

XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    reg_alpha=0.0,
    gamma=0.1,
    learning_rate=0.02,
    seed=42,
    nthread=-1,
)
N_PLAYERS = 7
MAX_ROUNDS = 3000
EARLY_STOP = 150
CLIP_EPS = 1e-7
LABEL_COL = "home_win"

PLAYER_FEATS = [
    "m_ewma_pre", "q_pre", "days_since_first_report_pre", "days_since_last_dnp_pre",
    "consec_dnps_pre", "played_last_game_pre", "minutes_last_game_pre",
    "days_since_last_played_pre", "injury_present_flag_pre",
]
FORM_FEATS = [
    "net_rtg_ewma_pre", "efg_ewma_pre", "tov_pct_ewma_pre",
    "orb_pct_ewma_pre", "ftr_ewma_pre",
]
STYLE_FEATS = [
    "off_3pa_rate_pre", "def_3pa_allowed_pre", "off_2pa_rate_pre",
    "def_2pa_allowed_pre", "off_tov_pct_pre", "def_forced_tov_pre",
]
SCHED_FEATS = [
    "days_rest_pre", "is_b2b_pre", "games_last_4_days_pre",
    "games_last_7_days_pre", "travel_miles_pre", "timezone_shift_hours_pre",
]


def build_feature_cols(n: int = N_PLAYERS) -> list[str]:
    cols = [
        f"{s}_p{k}_{f}"
        for s in ("home", "away")
        for k in range(1, n + 1)
        for f in PLAYER_FEATS
    ]
    for f in FORM_FEATS + STYLE_FEATS + SCHED_FEATS:
        cols += [f"home_{f}", f"away_{f}"]
    return cols


def build_block_cols(n: int = N_PLAYERS) -> dict[str, list[str]]:
    """Return feature columns grouped by feature block.

    Used by ablations and feature-importance notebooks. Keys: 'player', 'form',
    'style', 'schedule'. Each value lists the columns belonging to that block.
    """
    player = [
        f"{s}_p{k}_{f}"
        for s in ("home", "away")
        for k in range(1, n + 1)
        for f in PLAYER_FEATS
    ]
    form = [f"{s}_{f}" for s in ("home", "away") for f in FORM_FEATS]
    style = [f"{s}_{f}" for s in ("home", "away") for f in STYLE_FEATS]
    sched = [f"{s}_{f}" for s in ("home", "away") for f in SCHED_FEATS]
    return {"player": player, "form": form, "style": style, "schedule": sched}


def feature_block(name: str) -> str:
    """Return the block label for a given column name (player/form/style/schedule/elo/other)."""
    if name in ("base_margin", "p_elo"):
        return "elo"
    if name.startswith("home_p") or name.startswith("away_p"):
        # Player slot like home_p1_m_ewma_pre — but be careful not to match plain home_pace
        rest = name.split("_", 2)[1]
        if rest and rest[0] == "p" and rest[1:].isdigit():
            return "player"
    for f in FORM_FEATS:
        if name.endswith(f):
            return "form"
    for f in STYLE_FEATS:
        if name.endswith(f):
            return "style"
    for f in SCHED_FEATS:
        if name.endswith(f):
            return "schedule"
    return "other"


FEAT_COLS = build_feature_cols(N_PLAYERS)
BLOCK_COLS = build_block_cols(N_PLAYERS)


def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def load_year(year: int, gold_dir: Path = GOLD_DIR_DEFAULT) -> pd.DataFrame:
    """Load a single season; drop the cold-start rows where p1 EWMA minutes are zero."""
    df = pd.read_csv(gold_dir / f"game_xgboost_input_{year}_REGPST.csv")
    cold = (df["home_p1_m_ewma_pre"] == 0) | (df["away_p1_m_ewma_pre"] == 0)
    return df[~cold].reset_index(drop=True)


def make_dm(df: pd.DataFrame, use_bm: bool = True) -> xgb.DMatrix:
    avail = [c for c in FEAT_COLS if c in df.columns]
    y = df[LABEL_COL].values.astype(float) if LABEL_COL in df.columns else None
    dm = xgb.DMatrix(
        df[avail].values.astype(float), label=y,
        feature_names=avail, missing=np.nan,
    )
    if use_bm and "base_margin" in df.columns:
        dm.set_base_margin(df["base_margin"].values.astype(float))
    return dm


def kalshi_taker_fee(n_contracts: float, price: float) -> float:
    raw = 0.07 * n_contracts * price * (1 - price)
    return math.ceil(raw * 100) / 100


def train_final_model(
    holdout_year: int = 2025,
    train_start: int = 2015,
    gold_dir: Path = GOLD_DIR_DEFAULT,
    verbose: bool = True,
) -> tuple[xgb.Booster, pd.DataFrame, dict]:
    """Train the locked pipeline and return (booster, predictions_df, metadata).

    `predictions_df` has one row per holdout game with columns:
        game_id, game_ts, game_date, home_team_id, away_team_id,
        home_win, p_elo, p_full_model.
    """
    if holdout_year - train_start < 2:
        raise ValueError(
            f"holdout_year ({holdout_year}) must be at least train_start+2 "
            f"({train_start + 2}) so we have an early-stopping validation year."
        )

    all_data = {yr: load_year(yr, gold_dir) for yr in range(train_start, holdout_year + 1)}
    es_tr = pd.concat(
        [all_data[yr] for yr in range(train_start, holdout_year - 1)],
        ignore_index=True,
    )
    val_df = all_data[holdout_year - 1]
    full_tr = pd.concat(
        [all_data[yr] for yr in range(train_start, holdout_year)],
        ignore_index=True,
    )
    test_df = all_data[holdout_year]

    if verbose:
        print(
            f"Train ES: {train_start}..{holdout_year - 2} "
            f"({len(es_tr)} games)  |  ES-val: {holdout_year - 1} "
            f"({len(val_df)} games)  |  Test: {holdout_year} ({len(test_df)} games)"
        )

    m_es = xgb.train(
        XGB_PARAMS, make_dm(es_tr), MAX_ROUNDS,
        evals=[(make_dm(val_df), "val")],
        early_stopping_rounds=EARLY_STOP, verbose_eval=False,
    )
    best_round = m_es.best_iteration + 1

    m_final = xgb.train(XGB_PARAMS, make_dm(full_tr), best_round, verbose_eval=False)

    p_full = clip(m_final.predict(make_dm(test_df)))
    p_elo = clip(test_df["p_elo"].values.astype(float))

    preds = test_df[
        ["game_id", "game_ts", "game_date", "home_team_id", "away_team_id", LABEL_COL]
    ].copy()
    preds["p_elo"] = p_elo
    preds["p_full_model"] = p_full

    meta = {"best_round": int(best_round), "holdout_year": int(holdout_year),
            "train_start": int(train_start), "n_features": len(FEAT_COLS)}

    if verbose and LABEL_COL in test_df.columns:
        y = test_df[LABEL_COL].values.astype(float)
        meta["holdout_logloss_full"] = float(log_loss(y, p_full))
        meta["holdout_logloss_elo"] = float(log_loss(y, p_elo))
        print(
            f"best_round={best_round}  |  holdout log-loss: "
            f"full={meta['holdout_logloss_full']:.4f}  "
            f"elo={meta['holdout_logloss_elo']:.4f}"
        )

    return m_final, preds, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout-year", type=int, default=2025)
    ap.add_argument("--train-start", type=int, default=2015)
    ap.add_argument("--gold-dir", type=Path, default=GOLD_DIR_DEFAULT)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="If set, write predictions to this CSV path.")
    ap.add_argument("--model-out", type=Path, default=None,
                    help="If set, save the booster to this path (.json or .ubj).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    booster, preds, meta = train_final_model(
        holdout_year=args.holdout_year,
        train_start=args.train_start,
        gold_dir=args.gold_dir,
        verbose=not args.quiet,
    )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        preds.to_csv(args.out_csv, index=False)
        print(f"Predictions written: {args.out_csv}")

    if args.model_out is not None:
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(args.model_out))
        print(f"Model saved: {args.model_out}")

    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
