"""Generate `trade_log_return_by_edge_bucket.png` (+ matching CSV).

Mean per-trade log return, bucketed by model-implied edge, under the
**actual selected half-Kelly strategy** (edge_min = 0.05, norm_min = 0.25 —
the best half-Kelly config from `trade_half_kelly_best_config_table.csv`).
Half-Kelly sizing is used because per-trade log return `log(W_after/W_before)`
is well-defined under it; fixed-$1 wipes the stake on a loss and yields
undefined log returns.

The earlier draft used a wider filter (edge ≥ 0.05, norm = 0) that included
trades that would not be taken under the live strategy. This version applies
the live thresholds, so every bucket reflects only trades the strategy
actually enters.

Outputs:
    organized/outputs/trade_log_return_by_edge_bucket.png
    organized/outputs/trade_log_return_by_edge_bucket_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
PIPELINE_DIR = Path(__file__).resolve().parent
for d in (ANALYSIS_DIR, PIPELINE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from outputs import save_fig, save_table  # noqa: E402
from walkforward import fit_holdout_models  # noqa: E402
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_ideal, add_trade_returns, BANKROLL_INIT,
)
from final_model import load_year, LABEL_COL  # noqa: E402

MODEL_LABELS = {"elo": "Elo", "full_model": "Full Model"}
MODEL_COLORS = {"elo": "#3498db", "full_model": "#e74c3c"}

# Buckets start at 5% because the live half-Kelly filter is edge ≥ 0.05.
EDGE_BIN_EDGES  = [0.05, 0.10, 0.15, 0.20, 0.30, 1.0]
EDGE_BIN_LABELS = ["5–10%", "10–15%", "15–20%", "20–30%", "30%+"]

# Live half-Kelly thresholds (from trade_half_kelly_best_config_table.csv).
LIVE_EDGE_MIN = 0.05
LIVE_NORM_MIN = 0.25


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    print("Training holdout models …")
    preds_df, _ = fit_holdout_models(
        {"xgb_with_elo": {"type": "xgb", "use_bm": True}, "elo": {"type": "elo"}},
        holdout_year=2025, verbose=False,
    )
    model_w = (
        preds_df[preds_df["model"] == "xgb_with_elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_full_model"})
    )
    elo_w = (
        preds_df[preds_df["model"] == "elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_elo"})
    )

    test_df = (
        load_year(2025)
        .dropna(subset=[LABEL_COL, "base_margin"])
        [["game_id", "game_ts", "game_date", "home_team_id", "away_team_id", LABEL_COL]]
        .copy()
    )
    test_df["game_ts"] = pd.to_datetime(test_df["game_ts"], utc=True)
    test_df["game_date"] = pd.to_datetime(test_df["game_date"])
    signals = test_df.merge(model_w, on="game_id").merge(elo_w, on="game_id")

    print("Building Kalshi trading index …")
    idx = build_kalshi_trading_index(signals)
    ticker_info, pretip = idx["ticker_info"], idx["pretip"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bucket_rows = []
    for ax, (model_name, model_col) in zip(
        axes, [("elo", "p_elo"), ("full_model", "p_full_model")]
    ):
        # Live half-Kelly thresholds (edge ≥ 0.05, norm ≥ 0.25).
        ents = collect_entries(
            ticker_info, pretip, model_col,
            LIVE_EDGE_MIN, LIVE_NORM_MIN, "half_life",
        )
        trades = run_kelly_ideal(ents, 0.5, BANKROLL_INIT)
        if not trades:
            continue
        tdf = add_trade_returns(pd.DataFrame(trades))
        tdf["edge_bucket"] = pd.cut(tdf["edge"], bins=EDGE_BIN_EDGES, labels=EDGE_BIN_LABELS)
        stats = (
            tdf.groupby("edge_bucket", observed=True)
            .agg(
                n=("pnl", "count"),
                mean_log_ret=("log_ret", "mean"),
                median_log_ret=("log_ret", "median"),
                hit_rate=("won", "mean"),
                total_pnl=("pnl", "sum"),
            )
            .reset_index()
        )
        stats["model"] = model_name
        bucket_rows.append(stats)

        color = MODEL_COLORS[model_name]
        bars = ax.bar(
            range(len(stats)),
            stats["mean_log_ret"].values,
            color=color, alpha=0.7, width=0.6,
        )
        for x, bar, n, val in zip(range(len(stats)), bars, stats["n"], stats["mean_log_ret"]):
            offset = 0.005 if val >= 0 else -0.005
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                f"n={int(n)}",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats["edge_bucket"].astype(str), fontsize=9)
        ax.set_xlabel("Model edge at entry")
        ax.set_ylabel("Mean per-trade log return")
        ax.set_title(MODEL_LABELS[model_name], fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.grid(axis="y", alpha=0.3)

    # Common y-limits so the two panels are comparable at a glance
    if bucket_rows:
        ymin = min(s["mean_log_ret"].min() for s in bucket_rows)
        ymax = max(s["mean_log_ret"].max() for s in bucket_rows)
        pad = 0.20 * max(ymax - ymin, 1e-3)
        for ax in axes:
            ax.set_ylim(ymin - pad, ymax + pad)

    plt.suptitle(
        "Mean Log Return by Model-Implied Edge Bucket "
        "(Half-Kelly best config: edge ≥ 0.05, norm ≥ 0.25)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "trade_log_return_by_edge_bucket")
    plt.close(fig)

    bucket_tbl = pd.concat(bucket_rows, ignore_index=True)
    save_table(bucket_tbl, "trade_log_return_by_edge_bucket_table")
    print(bucket_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
