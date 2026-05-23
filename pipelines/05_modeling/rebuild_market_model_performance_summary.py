"""Regenerate `market_model_performance_summary.csv` (and the disagreement-band
tables) using the de-duplicated Kalshi/Polymarket loaders so the per-source
log-loss/Brier/accuracy numbers match the forecasting summary on the same
n = 310 holdout set.

Reads cached holdout predictions from
    organized/data/model_comparison/holdout_model_comparison_2025.csv
to skip retraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from outputs import OUTPUTS_DIR, save_table  # noqa: E402
from markets import (  # noqa: E402
    load_kalshi_pretipoff_probs, load_polymarket_pretipoff_probs,
)


HOLDOUT_CSV = (
    Path(__file__).resolve().parents[1].parent
    / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"
)


def eval_metrics(y, p, label):
    mask = p.notna()
    yv, pv = y[mask].values, np.clip(p[mask].values, 0.01, 0.99)
    return {
        "source":   label,
        "n":        int(mask.sum()),
        "log_loss": float(log_loss(yv, pv)),
        "brier":    float(brier_score_loss(yv, pv)),
        "accuracy": float(((pv > 0.5) == yv).mean()),
    }


def summary_block(df, label):
    return pd.DataFrame([
        eval_metrics(df["home_win"], df["model_prob"],  "XGB + Elo (model)"),
        eval_metrics(df["home_win"], df["elo_prob"],    "Elo only"),
        eval_metrics(df["home_win"], df["kalshi_prob"], "Kalshi pre-tipoff"),
        eval_metrics(df["home_win"], df["poly_prob"],   "Polymarket pre-tipoff"),
    ]).assign(scope=label)


def main() -> None:
    holdout = pd.read_csv(HOLDOUT_CSV)
    model_preds = (
        holdout[holdout["model"] == "xgb_with_elo"]
        [["game_id", "home_win", "pred_prob"]]
        .rename(columns={"pred_prob": "model_prob"})
    )
    elo_preds = (
        holdout[holdout["model"] == "elo"]
        [["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "elo_prob"})
    )

    kalshi = load_kalshi_pretipoff_probs()
    poly   = load_polymarket_pretipoff_probs()

    comp = (
        model_preds.merge(elo_preds, on="game_id")
                  .merge(kalshi, on="game_id", how="left")
                  .merge(poly,   on="game_id", how="left")
    )
    print(f"comp rows: {len(comp)}, unique gid: {comp['game_id'].nunique()}")
    print(f"  with Kalshi:     {comp['kalshi_prob'].notna().sum()}")
    print(f"  with Polymarket: {comp['poly_prob'].notna().sum()}")

    common = comp[comp["kalshi_prob"].notna() & comp["poly_prob"].notna()]
    summary = pd.concat(
        [
            summary_block(comp,   "all available"),
            summary_block(common, f"common (n={len(common)})"),
        ],
        ignore_index=True,
    )
    save_table(summary, "market_model_performance_summary")
    print(summary.to_string(index=False))

    # Also rebuild the disagreement-band tables since they depend on comp
    band_bins  = [0, 0.05, 0.10, 0.20, 1.0]
    band_label = ["0-5%", "5-10%", "10-20%", "20%+"]

    def disagreement_table(df, model_col, market_col, market_name, y_col="home_win"):
        d = df.dropna(subset=[market_col]).copy()
        d["diff"] = (d[model_col] - d[market_col]).abs()
        d["band"] = pd.cut(d["diff"], bins=band_bins, labels=band_label, right=True)
        rows = []
        for band in band_label:
            sub = d[d["band"] == band]
            if len(sub) == 0:
                continue
            y = sub[y_col].values
            mp = np.clip(sub[model_col].values, 0.01, 0.99)
            mkt = np.clip(sub[market_col].values, 0.01, 0.99)
            rows.append({
                "band": band, "n": len(sub),
                "model_ll":  log_loss(y, mp),
                f"{market_name}_ll": log_loss(y, mkt),
                "model_acc": ((mp > 0.5) == y).mean(),
                f"{market_name}_acc": ((mkt > 0.5) == y).mean(),
                "mean_diff": sub["diff"].mean(),
            })
        return pd.DataFrame(rows)

    bands_k = disagreement_table(comp, "model_prob", "kalshi_prob", "kalshi")
    bands_p = disagreement_table(comp, "model_prob", "poly_prob",   "poly")
    save_table(bands_k, "market_kalshi_disagreement_band_table")
    save_table(bands_p, "market_polymarket_disagreement_band_table")


if __name__ == "__main__":
    main()
