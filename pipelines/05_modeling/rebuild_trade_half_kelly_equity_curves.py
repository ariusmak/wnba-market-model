"""Regenerate `trade_half_kelly_equity_curves.png` with payout-time ordering.

The engines compound the bankroll in entry-timestamp order. When trades enter
on different days from the games they settle on, plotting bankroll vs.
game_date produces a curve that loops back on itself and looks like a bug.
This script re-sorts each model's trades by `game_ts` (settlement / payout
time) and plots `BANKROLL_INIT + cumsum(pnl)` so the curves are strictly
monotonic in time. Per-trade pnl is unchanged, terminal wealth matches the
engine output to within rounding.

Usage:
    python organized/pipelines/05_modeling/rebuild_trade_half_kelly_equity_curves.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
PIPELINE_DIR = Path(__file__).resolve().parent
for d in (ANALYSIS_DIR, PIPELINE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from outputs import save_fig  # noqa: E402
from walkforward import fit_holdout_models  # noqa: E402
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_ideal, equity_by_payout, BANKROLL_INIT,
)
from final_model import load_year, LABEL_COL  # noqa: E402

MODEL_LABELS = {"elo": "Elo", "full_model": "Full Model"}
MODEL_COLORS = {"elo": "#3498db", "full_model": "#e74c3c"}


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

    # Best half-Kelly threshold pair we use throughout the trading notebooks.
    BEST_EDGE_MIN, BEST_NORM_MIN = 0.05, 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name, model_col in [("elo", "p_elo"), ("full_model", "p_full_model")]:
        ents = collect_entries(
            ticker_info, pretip, model_col,
            BEST_EDGE_MIN, BEST_NORM_MIN, "half_life",
        )
        trades = run_kelly_ideal(ents, 0.5, BANKROLL_INIT)
        if not trades:
            continue
        tdf = pd.DataFrame(trades)
        eq = equity_by_payout(tdf, bankroll_init=BANKROLL_INIT, ts_col="game_ts")
        ret = (eq["display_bankroll"].iloc[-1] - BANKROLL_INIT) / BANKROLL_INIT
        ax.step(
            eq["game_ts"], eq["display_bankroll"], where="post",
            color=MODEL_COLORS[model_name], linewidth=1.8,
            label=f"{MODEL_LABELS[model_name]} ({ret:.0%} return)",
        )

    ax.axhline(BANKROLL_INIT, color="gray", linestyle=":", alpha=0.5,
               label="Starting bankroll")
    ax.set_xlabel("Settlement date")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("Half-Kelly Equity Curves for Elo and Full Model Strategies",
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    plt.tight_layout()
    save_fig(fig, "trade_half_kelly_equity_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
