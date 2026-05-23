"""For each trade in the uncapped half-Kelly $5k sweep backtest, compute
our order's share of the historical Kalshi tape inside that game's entry
window. Reports two slices:

    share_of_total      = n_filled / total tape volume in window
    share_of_qualifying = n_filled / tape volume at or below our entry price

Outputs:
    organized/outputs/trade_volume_share_per_trade.csv
    organized/outputs/trade_volume_share_distribution.png
"""
from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
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
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_sweep, fill_diagnostics, add_trade_returns,
)
from final_model import load_year, LABEL_COL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    print("Loading cached holdout predictions ...")
    holdout = pd.read_csv(HOLDOUT_CSV)
    model_w = (
        holdout[holdout["model"] == "xgb_with_elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_full_model"})
    )
    elo_w = (
        holdout[holdout["model"] == "elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_elo"})
    )
    test_2025 = (
        load_year(2025).dropna(subset=[LABEL_COL, "base_margin"])
        [["game_id","game_ts","game_date","home_team_id","away_team_id", LABEL_COL]]
        .copy()
    )
    test_2025["game_ts"] = pd.to_datetime(test_2025["game_ts"], utc=True)
    signals = test_2025.merge(model_w, on="game_id").merge(elo_w, on="game_id")

    print("Building Kalshi trading index ...")
    idx = build_kalshi_trading_index(signals, pred_cols=("p_full_model", "p_elo"))
    ticker_info, pretip, wt = idx["ticker_info"], idx["pretip"], idx["wt"]

    print("Running uncapped half-Kelly sweep ...")
    ents  = collect_entries(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
    )
    raw   = run_kelly_sweep(ents, KELLY_FRACTION, wt, BANKROLL_REAL)
    tdf   = add_trade_returns(pd.DataFrame(raw))
    print(f"  trades: {len(tdf)}")

    fdf = fill_diagnostics(tdf, wt)
    fdf = fdf.merge(tdf[["game_id","won","wager","fill_pct"]],
                     on="game_id", suffixes=("","_t"))

    fdf["our_filled"] = tdf.set_index("game_id").loc[fdf["game_id"], "n_contracts"].values
    fdf["share_of_total"]      = np.where(
        fdf["total_volume"] > 0,
        fdf["our_filled"] / fdf["total_volume"], np.nan,
    )
    fdf["share_of_qualifying"] = np.where(
        fdf["volume_at_or_below"] > 0,
        fdf["our_filled"] / fdf["volume_at_or_below"], np.nan,
    )

    save_table(
        fdf[[
            "game_id","side","entry_px","wager","our_filled",
            "n_needed","total_volume","volume_at_or_below","vol_at_price",
            "fill_pct","share_of_total","share_of_qualifying","won",
        ]],
        "trade_volume_share_per_trade",
    )

    def _stats(s, label):
        s = pd.Series(s).dropna()
        return {
            "metric":    label,
            "n":         int(len(s)),
            "mean":      float(s.mean()),
            "std":       float(s.std()),
            "min":       float(s.min()),
            "p05":       float(np.percentile(s,  5)),
            "p25":       float(np.percentile(s, 25)),
            "median":    float(np.percentile(s, 50)),
            "p75":       float(np.percentile(s, 75)),
            "p95":       float(np.percentile(s, 95)),
            "max":       float(s.max()),
            "share_above_25pct": float((s >= 0.25).mean()),
            "share_above_50pct": float((s >= 0.50).mean()),
            "share_above_75pct": float((s >= 0.75).mean()),
        }

    summary = pd.DataFrame([
        _stats(fdf["share_of_total"],      "share of total window volume"),
        _stats(fdf["share_of_qualifying"], "share of qualifying-price volume (<= entry)"),
    ])
    save_table(summary, "trade_volume_share_summary_table")
    print()
    print(summary.to_string(index=False))

    # ---- distribution figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (col, label, color) in zip(axes, [
        ("share_of_total",      "Share of Total Window Volume",            "#3498db"),
        ("share_of_qualifying", "Share of Qualifying-Price Volume (<= entry)", "#e74c3c"),
    ]):
        s = fdf[col].dropna()
        bins = np.linspace(0, 1, 21)
        ax.hist(s.clip(upper=0.999), bins=bins, color=color, alpha=0.85,
                edgecolor="white")
        for marker, name, ls in [
            (s.median(), "median", "--"),
            (s.mean(),   "mean",   ":"),
        ]:
            ax.axvline(marker, color="black", linestyle=ls, alpha=0.7)
            ax.text(marker, ax.get_ylim()[1] * 0.92,
                    f" {name}={marker*100:.1f}%", fontsize=9, va="top")
        ax.set_xlabel(label); ax.set_ylabel("Trades")
        ax.set_title(label, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%","25%","50%","75%","100%"])
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Per-Trade Volume Share — Uncapped Half-Kelly Sweep, $5,000 bankroll",
        fontweight="bold", fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_fig(fig, "trade_volume_share_distribution")
    plt.close(fig)


if __name__ == "__main__":
    main()
