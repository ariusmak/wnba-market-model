"""Regenerate `liq_capacity_by_starting_bankroll.png` from the cached
`liq_execution_summary_table.csv`, dropping the middle "Absolute PnL"
panel so only Return % and Fill Rate are shown.

Run:
    python organized/pipelines/05_modeling/rebuild_liq_capacity_by_starting_bankroll.py
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


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    rdf = pd.read_csv(OUTPUTS_DIR / "liq_execution_summary_table.csv")
    rdf = rdf.sort_values("starting_bankroll").reset_index(drop=True)
    bankrolls = rdf["starting_bankroll"].values

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.plot(bankrolls, rdf["ideal_return"] * 100, "o-",
            color="#e74c3c", lw=2, ms=6, label="Ideal")
    ax.plot(bankrolls, rdf["liq_constrained_return"] * 100, "s--",
            color="#3498db", lw=2, ms=6, label="Liquidity-constrained")
    ax.set_xlabel("Starting bankroll ($)")
    ax.set_ylabel("Return (%)")
    ax.set_title("Return % by Starting Bankroll", fontweight="bold")
    ax.set_xscale("log"); ax.set_xticks(bankrolls)
    ax.set_xticklabels([f"${b:,.0f}" for b in bankrolls], rotation=45, fontsize=8)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)

    ax = axes[1]
    ax.plot(bankrolls, rdf["mean_fill_rate"] * 100, "D-",
            color="#2ecc71", lw=2, ms=6, label="Mean")
    ax.plot(bankrolls, rdf["median_fill_rate"] * 100, "^--",
            color="#27ae60", lw=2, ms=6, label="Median")
    ax.set_xlabel("Starting bankroll ($)")
    ax.set_ylabel("Fill rate (%)")
    ax.set_title("Fill Rate by Starting Bankroll", fontweight="bold")
    ax.set_xscale("log"); ax.set_xticks(bankrolls)
    ax.set_xticklabels([f"${b:,.0f}" for b in bankrolls], rotation=45, fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=10)

    fig.suptitle("Strategy Capacity by Starting Bankroll", fontweight="bold", y=1.02)
    plt.tight_layout()
    save_fig(fig, "liq_capacity_by_starting_bankroll")
    plt.close(fig)


if __name__ == "__main__":
    main()
