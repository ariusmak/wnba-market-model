"""Regenerate `feature_block_ablation_logloss.png` from the cached summary
CSV with tighter y-axis limits so the per-block differences are visually clear.

Run:
    python organized/pipelines/05_modeling/rebuild_feature_block_ablation_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from outputs import OUTPUTS_DIR, save_fig  # noqa: E402

ABL_ORDER  = ["elo", "full", "no_player", "no_form", "no_style", "no_schedule"]
ABL_COLORS = {
    "elo": "#7f8c8d", "full": "#3498db", "no_player": "#e74c3c",
    "no_form": "#e67e22", "no_style": "#9b59b6", "no_schedule": "#2ecc71",
}
ABL_LABELS = {
    "elo": "Elo only", "full": "Full",
    "no_player": "− Player", "no_form": "− Form",
    "no_style": "− Style", "no_schedule": "− Schedule",
}

PERIODS = ("OOF 2020–2024", "2025 Holdout")


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    summary_path = OUTPUTS_DIR / "feature_block_ablation_summary.csv"
    summary = pd.read_csv(summary_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, period in zip(axes, PERIODS):
        sub = (
            summary[summary["period"] == period]
            .set_index("model")
            .loc[ABL_ORDER]
        )
        vals = sub["log_loss"].values
        colors = [ABL_COLORS[m] for m in ABL_ORDER]
        bars = ax.bar(range(len(ABL_ORDER)), vals, color=colors,
                      edgecolor="white", width=0.7)
        ax.set_xticks(range(len(ABL_ORDER)))
        ax.set_xticklabels(
            [ABL_LABELS[m] for m in ABL_ORDER],
            rotation=15, ha="right", fontsize=9,
        )
        ax.set_ylabel("Log loss")
        ax.set_title(f"Block Ablation — {period}", fontweight="bold")

        # Tight y-limits so the millis-scale gaps are visible. Pad ~15% of
        # the range above and below; pin the lower edge so bars don't float.
        v_min, v_max = vals.min(), vals.max()
        span = max(v_max - v_min, 1e-4)
        pad_lo = span * 0.30
        pad_hi = span * 0.55  # extra headroom for the value labels
        ax.set_ylim(v_min - pad_lo, v_max + pad_hi)

        full_ll = sub.loc["full", "log_loss"]
        for bar, val in zip(bars, vals):
            delta = val - full_ll
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + span * 0.04,
                f"{val:.4f}\nΔ={delta:+.4f}",
                ha="center", va="bottom", fontsize=8,
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_fig(fig, "feature_block_ablation_logloss")
    plt.close(fig)


if __name__ == "__main__":
    main()
