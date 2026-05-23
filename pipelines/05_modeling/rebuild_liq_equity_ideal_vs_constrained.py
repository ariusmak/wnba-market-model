"""Regenerate `liq_equity_ideal_vs_constrained.png` with payout-time ordering.

Same fix as `rebuild_trade_half_kelly_equity_curves.py`: trades are sorted by
`game_ts` (settlement / payout time) and the plotted bankroll is
`BANKROLL_INIT + cumsum(pnl)` over that order. Eliminates the loop-back when
entries on a single day cover games that settle on different days.

Usage:
    python organized/pipelines/05_modeling/rebuild_liq_equity_ideal_vs_constrained.py
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
    collect_entries, run_kelly_ideal, run_kelly_sweep, equity_by_payout,
)
from final_model import load_year, LABEL_COL  # noqa: E402

BANKROLL_REAL = 5000.0
BEST_EDGE_MIN, BEST_NORM_MIN = 0.05, 0.25


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
    ticker_info, pretip, wt = idx["ticker_info"], idx["pretip"], idx["wt"]

    print("Running half-Kelly under ideal and liquidity-constrained execution …")
    entries = collect_entries(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, "half_life",
    )
    tdf_ideal = pd.DataFrame(run_kelly_ideal(entries, 0.5, BANKROLL_REAL))
    tdf_sweep = pd.DataFrame(run_kelly_sweep(entries, 0.5, wt, BANKROLL_REAL))

    eq_ideal = equity_by_payout(tdf_ideal, bankroll_init=BANKROLL_REAL, ts_col="game_ts")
    eq_sweep = equity_by_payout(tdf_sweep, bankroll_init=BANKROLL_REAL, ts_col="game_ts")

    ret_ideal = (eq_ideal["display_bankroll"].iloc[-1] - BANKROLL_REAL) / BANKROLL_REAL
    ret_sweep = (eq_sweep["display_bankroll"].iloc[-1] - BANKROLL_REAL) / BANKROLL_REAL

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(
        eq_ideal["game_ts"], eq_ideal["display_bankroll"], where="post",
        color="#e74c3c", linewidth=1.8,
        label=f"Ideal (infinite liquidity, {ret_ideal:.0%})",
    )
    ax.step(
        eq_sweep["game_ts"], eq_sweep["display_bankroll"], where="post",
        color="#3498db", linewidth=1.8, linestyle="--",
        label=f"Liquidity-constrained ({ret_sweep:.0%})",
    )
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Settlement date")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title(
        "Ideal and Liquidity-Constrained Half-Kelly Equity Curves "
        "($5,000 start)",
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    plt.tight_layout()
    save_fig(fig, "liq_equity_ideal_vs_constrained")
    plt.close(fig)


if __name__ == "__main__":
    main()
