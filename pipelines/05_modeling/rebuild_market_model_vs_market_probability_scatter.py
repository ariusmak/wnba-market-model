"""Regenerate `market_model_vs_market_probability_scatter.png` from the
deduplicated Kalshi / Polymarket loaders. The previous run was built from
the inflated 366-row comp DataFrame (Polymarket condition_id duplicates
fanned out the merge); this script uses the per-game deduped loaders so
each panel scatters one point per game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from outputs import save_fig  # noqa: E402
from markets import (  # noqa: E402
    load_kalshi_pretipoff_probs, load_polymarket_pretipoff_probs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

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
    poly = load_polymarket_pretipoff_probs()

    comp = (
        model_preds.merge(elo_preds, on="game_id")
                  .merge(kalshi, on="game_id", how="left")
                  .merge(poly,   on="game_id", how="left")
    )
    assert len(comp) == comp["game_id"].nunique(), \
        "duplicate game_ids — dedup loaders not in effect"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (mcol, mname) in zip(
        axes, [("kalshi_prob", "Kalshi"), ("poly_prob", "Polymarket")],
    ):
        sub = comp.dropna(subset=[mcol])
        home = sub[sub["home_win"] == 1]
        away = sub[sub["home_win"] == 0]
        ax.scatter(home[mcol], home["model_prob"], c="#2ecc71", alpha=0.5, s=20)
        ax.scatter(away[mcol], away["model_prob"], c="#e74c3c", alpha=0.5, s=20)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel(f"{mname} implied P(home win)")
        ax.set_ylabel("Model P(home win)")
        ax.set_title(f"Model vs {mname} (n={len(sub)})", fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(handles=[
            Line2D([], [], color="k", ls="--", alpha=0.3, label="y = x"),
            Line2D([], [], marker="o", color="w", markerfacecolor="#2ecc71",
                   markersize=6, label="Home win"),
            Line2D([], [], marker="o", color="w", markerfacecolor="#e74c3c",
                   markersize=6, label="Away win"),
        ], loc="upper left")
    plt.tight_layout()
    save_fig(fig, "market_model_vs_market_probability_scatter")
    plt.close(fig)
    print(f"comp rows: {len(comp)}")
    print(f"  Kalshi panel n: {comp['kalshi_prob'].notna().sum()}")
    print(f"  Polymarket panel n: {comp['poly_prob'].notna().sum()}")


if __name__ == "__main__":
    main()
