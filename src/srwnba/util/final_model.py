"""
Final production model for the WNBA prediction pipeline.

Trains an XGBoost model on a full training CSV (gold-level game_xgboost_input),
then produces home-win probability predictions for new games. Intended as the
model layer that connects to the live trading system.

All hyperparameters are imported from organized/config/final_hyperparams.py
(the single source of truth). Do NOT hardcode params here.

Training procedure:
  1. Load training CSV, drop cold-start rows (home/away p1 m_ewma == 0)
  2. Train on ALL remaining rows for XGB_PRODUCTION_NUM_BOOST_ROUND trees
  3. Model is ready to predict

Input requirements:
  - Training CSV: any gold-level game_xgboost_input file (e.g. 2015_2024_REGPST.csv)
  - Prediction input: a DataFrame (or CSV) with the same column schema as the gold table.
    Must include the 160 feature columns + either 'p_elo' or 'base_margin' for the Elo prior.
    The 'home_win' label column is NOT required for prediction.

Output:
  - p_home: production home-win probability (float between 0 and 1)

──────────────────────────────────────────────────────────────────────
USAGE — Python (for live trading integration):
──────────────────────────────────────────────────────────────────────

    from src.srwnba.util.final_model import FinalModel
    import pandas as pd

    # Step 1: Train the model once (takes ~2 seconds)
    model = FinalModel("data/gold/game_xgboost_input_2015_2024_REGPST.csv")

    # Step 2: Predict a single game — pass a 1-row DataFrame with gold-table columns
    game_row = pd.read_csv("data/gold/game_xgboost_input_2025_REGPST.csv").iloc[[0]]
    p_home = model.predict_single(game_row)   # returns float, e.g. 0.6214

    # Step 3: Or predict multiple games at once
    result = model.predict(games_df)
    # result = {
    #     "game_id": ["uuid1", "uuid2", ...],   # if game_id column exists
    #     "p_elo":   [0.55, 0.48, ...],          # Elo-only baseline probabilities
    #     "p_raw":   [0.62, 0.45, ...],          # XGBoost-corrected probabilities
    #     "p_home":  [0.62, 0.45, ...],          # same as p_raw (home-win prob)
    # }

──────────────────────────────────────────────────────────────────────
USAGE — Command line:
──────────────────────────────────────────────────────────────────────

    # Predict all games in a CSV:
    python -m srwnba.util.final_model \\
        --train-csv data/gold/game_xgboost_input_2015_2024_REGPST.csv \\
        --input-csv data/gold/game_xgboost_input_2025_REGPST.csv

    # Predict a single game by ID:
    python -m srwnba.util.final_model \\
        --train-csv data/gold/game_xgboost_input_2015_2024_REGPST.csv \\
        --input-csv data/gold/game_xgboost_input_2025_REGPST.csv \\
        --game-id "127815af-ec83-4409-b0c3-4140a357a60c"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ── Import locked hyperparams from organized/config/final_hyperparams.py ──
# This is the single source of truth for all model config:
#   XGB_PARAMS: dict with max_depth=6, min_child_weight=3, gamma=0.1,
#               colsample_bytree=0.6, subsample=0.8, reg_lambda=1.0,
#               reg_alpha=0.0, learning_rate=0.02, seed=42
#   XGB_PRODUCTION_NUM_BOOST_ROUND: locked final-fit tree count
#   N_PLAYERS: 7 (player slots per team)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "config"))
from final_hyperparams import (
    XGB_PARAMS,
    XGB_PRODUCTION_NUM_BOOST_ROUND,
)
from srwnba.util.model_schema import ELO_PROB_COL, FEAT_COLS, LABEL_COL

# ── Feature definitions ─────────────────────────────────────────────
# These must exactly match the gold table columns built by the feature
# pipeline (pipelines/04_gold/). 9 features per player slot, 7 slots
# per side = 126 player features + 10 form + 12 style + 12 schedule = 160 total.
# Elo probability is passed via base_margin, NOT as an ordinary feature.

CLIP_EPS = 1e-6


def clip_probs(p):
    return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def logit(p):
    p = clip_probs(np.asarray(p, dtype=float))
    return np.log(p / (1 - p))


def _make_dmatrix(df, feature_cols, has_label=True):
    """Build an xgb.DMatrix from a gold-table DataFrame.

    - Extracts the 160 ordinary feature columns
    - Sets base_margin = logit(p_elo) so XGBoost learns a correction on top of Elo
    - Accepts either 'p_elo' (probability) or pre-computed 'base_margin' (logit)
    - Missing player slots should be NaN (the gold table uses NULL), handled by XGBoost natively
    """
    avail = [c for c in feature_cols if c in df.columns]
    X = df[avail].values.astype(float)
    y = df[LABEL_COL].values.astype(float) if has_label and LABEL_COL in df.columns else None
    dm = xgb.DMatrix(X, label=y, feature_names=avail, missing=np.nan)
    # Set Elo prior as base_margin — XGBoost predicts logit(p) = base_margin + tree_output
    if ELO_PROB_COL in df.columns:
        p_elo = clip_probs(df[ELO_PROB_COL].values)
        dm.set_base_margin(logit(p_elo))
    elif "base_margin" in df.columns:
        dm.set_base_margin(df["base_margin"].values.astype(float))
    return dm


def _cold_start_mask(df):
    """Filter out cold-start games where top player EWMA minutes are zero.

    This happens at the start of the first modeled season (2015) before
    enough games have been played for m_ewma to become informative.
    See CLAUDE.md §3 — first 9 games of 2015 are excluded for this reason.
    """
    col = "home_p1_m_ewma_pre"
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return (df[col] != 0) & (df["away_p1_m_ewma_pre"] != 0)


class FinalModel:
    """Train once on full training data, then predict individual games."""

    def __init__(self, train_csv: str | Path):
        """Load training CSV and train the locked production model.

        Production training uses all cold-start-filtered rows from the provided
        CSV and exactly XGB_PRODUCTION_NUM_BOOST_ROUND trees. The tree count is
        not reselected from the final season in the training data.
        """
        train_df = pd.read_csv(train_csv)
        train_df = train_df[_cold_start_mask(train_df)].reset_index(drop=True)
        print(f"[FinalModel] Training rows (cold-start filtered): {len(train_df)}")

        # Production tree count is locked by validation. Do not reselect it from
        # the last season in the training CSV, which may be a tiny partial year.
        self.best_round = int(XGB_PRODUCTION_NUM_BOOST_ROUND)
        self.best_round_source = "locked_config"
        print(f"[FinalModel] Locked production trees: {self.best_round}")

        dm_full = _make_dmatrix(train_df, FEAT_COLS)
        self.model = xgb.train(
            XGB_PARAMS, dm_full, self.best_round, verbose_eval=False
        )
        print(f"[FinalModel] Final model trained ({self.best_round} trees, {len(FEAT_COLS)} features)")

    def predict(self, input_df: pd.DataFrame) -> dict:
        """Predict home-win probability for one or more games.

        Args:
            input_df: DataFrame with the same columns as the gold table.
                      Must contain feature columns + p_elo (or base_margin).
                      Can be a single row or multiple rows.

        Returns:
            dict with keys:
                game_id:   list of game IDs (if column exists)
                p_elo:     list of Elo-only probabilities
                p_raw:     list of raw XGBoost probabilities
                p_home:    same as p_raw (home-win probability)
        """
        dm = _make_dmatrix(input_df, FEAT_COLS, has_label=False)
        p_raw = clip_probs(self.model.predict(dm))

        result = {
            "p_raw": p_raw.tolist(),
            "p_home": p_raw.tolist(),
        }
        if "game_id" in input_df.columns:
            result["game_id"] = input_df["game_id"].tolist()
        if ELO_PROB_COL in input_df.columns:
            result["p_elo"] = clip_probs(input_df[ELO_PROB_COL].values).tolist()

        return result

    def predict_single(self, input_df: pd.DataFrame) -> float:
        """Convenience: predict a single game, return home-win probability as a float."""
        return self.predict(input_df)["p_home"][0]


# ── CLI entry point ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WNBA Final Model — train & predict")
    parser.add_argument("--train-csv", required=True, help="Path to full training CSV (e.g. 2015-2024)")
    parser.add_argument("--input-csv", required=True, help="Path to CSV containing game(s) to predict")
    parser.add_argument("--game-id", default=None, help="If set, filter input CSV to this game_id")
    args = parser.parse_args()

    model = FinalModel(args.train_csv)

    input_df = pd.read_csv(args.input_csv)
    if args.game_id:
        input_df = input_df[input_df["game_id"] == args.game_id]
        if input_df.empty:
            print(f"ERROR: game_id '{args.game_id}' not found in {args.input_csv}")
            sys.exit(1)

    result = model.predict(input_df)

    for i, p in enumerate(result["p_home"]):
        gid = result.get("game_id", [f"row_{i}"])[i]
        p_elo = result.get("p_elo", [None])[i]
        elo_str = f"  p_elo={p_elo:.4f}" if p_elo is not None else ""
        print(f"  {gid}: p_home={p:.4f}{elo_str}")


if __name__ == "__main__":
    main()
