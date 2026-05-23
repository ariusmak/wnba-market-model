"""Regenerate `trade_hit_rate_by_edge_bucket.png` (+ matching CSV).

Bucket the **live half-Kelly strategy** trades by model-implied edge and
plot hit rate per bucket. Uses half-Kelly sizing for consistency with the
companion mean-log-return figure (`trade_log_return_by_edge_bucket.png`);
applies the live half-Kelly thresholds `edge_min = 0.05`, `norm_min = 0.25`
(from `trade_half_kelly_best_config_table.csv`) so every trade plotted is
one the strategy actually enters. Hit rate is sizing-invariant — it depends
only on the trade set — so this still answers the same edge-calibration
question as before, just on the live half-Kelly trade set.

Outputs:
    organized/outputs/trade_hit_rate_by_edge_bucket.png
    organized/outputs/trade_returns_by_edge_bucket_table.csv
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
from trading import collect_entries, run_kelly_ideal, BANKROLL_INIT  # noqa: E402
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
        ents = collect_entries(
            ticker_info, pretip, model_col,
            LIVE_EDGE_MIN, LIVE_NORM_MIN, "half_life",
        )
        trades = run_kelly_ideal(ents, 0.5, BANKROLL_INIT)
        if not trades:
            continue
        bdf = pd.DataFrame(trades)
        bdf["edge_bucket"] = pd.cut(bdf["edge"], bins=EDGE_BIN_EDGES, labels=EDGE_BIN_LABELS)
        stats = (
            bdf.groupby("edge_bucket", observed=True)
            .agg(n=("pnl", "count"), hit_rate=("won", "mean"),
                 mean_pnl=("pnl", "mean"), total_pnl=("pnl", "sum"))
            .reset_index()
        )
        stats["model"] = model_name
        bucket_rows.append(stats)

        color = MODEL_COLORS[model_name]
        bars = ax.bar(range(len(stats)), stats["hit_rate"] * 100,
                      color=color, alpha=0.7, width=0.6)
        for x, bar, n in zip(range(len(stats)), bars, stats["n"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"n={int(n)}", ha="center", fontsize=8)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(stats["edge_bucket"].astype(str), fontsize=9)
        ax.set_xlabel("Model edge at entry")
        ax.set_ylabel("Hit rate (%)")
        ax.set_title(MODEL_LABELS[model_name], fontweight="bold")
        ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Hit Rate by Model-Implied Edge Bucket "
        "(Half-Kelly best config: edge ≥ 0.05, norm ≥ 0.25)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "trade_hit_rate_by_edge_bucket")
    plt.close(fig)

    bucket_tbl = pd.concat(bucket_rows, ignore_index=True)
    save_table(bucket_tbl, "trade_returns_by_edge_bucket_table")
    print(bucket_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
