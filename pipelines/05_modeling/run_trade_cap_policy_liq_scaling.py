"""Cap-policy sweep with v1 liquidity cap under 2x and 3x tape-scaling.

Re-runs the same 7-policy sweep from `run_trade_cap_policy_sweep_with_liq.py`
under three liquidity assumptions:

    1.0x  (2025 baseline)
    2.0x  (synthetic — every historical trade's `count` x2)
    3.0x  (synthetic — every historical trade's `count` x3)

The scaling propagates linearly through the v1 liquidity-cap pools
(visible-depth, recent-3h, cumulative) AND through the sweep's fill
capacity, since both are sourced from the same `wt` DataFrame.

This is a sensitivity test for "if the 2026 Kalshi WNBA tape is N times
thicker than 2025, does the cumulative-30%-cap stop binding?"

Outputs (in organized/outputs/):
    trade_cap_policy_liq_scaling_summary.csv
    trade_cap_policy_liq_scaling_per_trade.csv
    trade_cap_policy_liq_scaling_metrics_grid.png
    trade_cap_policy_liq_scaling_binding_distribution.png
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
from trading import LABEL, collect_all_snapshots  # noqa: E402
from final_model import load_year, LABEL_COL, kalshi_taker_fee  # noqa: E402

# Reuse functions from the v1 liq sweep script
from run_trade_cap_policy_sweep_with_liq import (  # noqa: E402
    POLICIES, POLICY_COLORS, BANKROLL_REAL, BEST_EDGE_MIN, BEST_NORM_MIN,
    KELLY_FRACTION, ENTRY_WINDOW, TIMING_CUTOFF_H, MAX_VISIBLE_DEPTH_PARTICIPATION,
    RECENT_VOLUME_WINDOW_HOURS, MAX_RECENT_QUALIFYING_VOLUME_PARTICIPATION,
    COLD_START_BANKROLL_CAP, COLD_START_VISIBLE_DEPTH_PARTICIPATION,
    MAX_CUMULATIVE_QUALIFYING_VOLUME_SHARE, VISIBLE_WINDOW_MINUTES,
    policy_cap, collect_entries_cutoff, first_qualification_map,
    liquidity_caps, run_sweep_with_liq, evaluate, by_period,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

SCALINGS = [1.0, 2.0, 3.0]
SCALING_LABELS = {1.0: "1.0x (2025 baseline)", 2.0: "2.0x", 3.0: "3.0x"}


def scale_tape(wt: pd.DataFrame, factor: float) -> pd.DataFrame:
    if factor == 1.0:
        return wt
    out = wt.copy()
    out["count"] = out["count"] * factor
    return out


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_scaling_metrics(summary: pd.DataFrame, save_name: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    pmap = list(POLICIES); x = np.arange(len(pmap))
    width = 0.27
    sc_color = {1.0: "#3498db", 2.0: "#9b59b6", 3.0: "#e74c3c"}

    for ax, col, ylabel, title, fmt in [
        (axes[0, 0], "total_return",       "Return (x bankroll)",
         "Total Return (multiple)",   lambda v: f"{1+v:.1f}x"),
        (axes[0, 1], "sharpe_per_trade",   "Per-trade Sharpe",
         "Per-Trade Sharpe",          lambda v: f"{v:+.3f}"),
        (axes[1, 0], "max_drawdown",       "Max drawdown ($)",
         "Max Drawdown",              lambda v: f"${v:,.0f}"),
        (axes[1, 1], "mean_realized_wager_pct",
         "Mean realized stake (% of bankroll)",
         "Realized Stake %",          lambda v: f"{v*100:.1f}%"),
    ]:
        for i, sc in enumerate(SCALINGS):
            sub = summary[summary["scaling"] == sc].set_index("policy").reindex(pmap)
            offset = (i - 1) * width
            vals = sub[col].values
            ax.bar(x + offset, vals if col != "total_return" else (1.0 + vals),
                   width=width, color=sc_color[sc], alpha=0.85,
                   label=SCALING_LABELS[sc], edgecolor="white")
            for xi, v in zip(x + offset, vals):
                ax.text(xi, (1.0 + v) if col == "total_return" else v + (
                            v * 0.02 if abs(v) > 1 else 0.005),
                        fmt(v), ha="center", fontsize=7, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICIES[p] for p in pmap], fontsize=8,
                           rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Cap-Policy Sweep with v1 Liquidity Cap — Tape Scaling Sensitivity "
        f"(T-{int(TIMING_CUTOFF_H)}h gate, ${BANKROLL_REAL:,.0f} bankroll)",
        fontweight="bold", fontsize=12, y=1.0,
    )
    plt.tight_layout()
    save_fig(fig, save_name); plt.close(fig)


def plot_scaling_binding(summary: pd.DataFrame, save_name: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    palette = {
        "share_kelly_binding":      ("#2ecc71", "Kelly target"),
        "share_portfolio_binding":  ("#e67e22", "Portfolio cap"),
        "share_rolling_binding":    ("#3498db", "Rolling liq cap"),
        "share_cumulative_binding": ("#9b59b6", "Cumulative liq cap"),
        "share_sweepshort_binding": ("#7f8c8d", "Sweep tape exhausted"),
    }

    pmap = list(POLICIES); x = np.arange(len(pmap))
    for ax, sc in zip(axes, SCALINGS):
        sub = summary[summary["scaling"] == sc].set_index("policy").reindex(pmap)
        bottoms = np.zeros(len(pmap))
        for col, (color, lbl) in palette.items():
            vals = sub[col].values
            ax.bar(x, vals * 100, bottom=bottoms * 100, color=color, alpha=0.85,
                   edgecolor="white", label=lbl)
            for xi, v, b in zip(x, vals, bottoms):
                if v >= 0.06:
                    ax.text(xi, (b + v / 2) * 100, f"{v*100:.0f}%",
                            ha="center", va="center", fontsize=7, color="white",
                            fontweight="bold")
            bottoms = bottoms + vals
        ax.set_xticks(x); ax.set_xticklabels(
            [POLICIES[p] for p in pmap], fontsize=8, rotation=25, ha="right",
        )
        ax.set_title(SCALING_LABELS[sc], fontweight="bold")
        ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        if sc == SCALINGS[0]:
            ax.set_ylabel("Share of trades (%)")
        if sc == SCALINGS[-1]:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.85)

    fig.suptitle("Binding-Constraint Distribution — Tape-Scaling Sensitivity",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    save_fig(fig, save_name); plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
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
    ticker_info, pretip, wt_base = idx["ticker_info"], idx["pretip"], idx["wt"]

    first_qual = first_qualification_map(ticker_info, pretip)
    ents = collect_entries_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
        cutoff_hours=TIMING_CUTOFF_H,
    )
    print(f"Qualifying entries (T-{int(TIMING_CUTOFF_H)}h gate): {len(ents)}")

    summary_rows, per_trade_rows = [], []
    for sc in SCALINGS:
        wt = scale_tape(wt_base, sc)
        print(f"\n=== Tape scaling {SCALING_LABELS[sc]} ===")
        for name in POLICIES:
            tdf = run_sweep_with_liq(
                ents, KELLY_FRACTION, wt, BANKROLL_REAL,
                policy_name=name, starting_bankroll=BANKROLL_REAL,
                first_qual=first_qual,
            )
            s = evaluate(name, tdf, BANKROLL_REAL)
            s["scaling"] = sc
            s["scaling_label"] = SCALING_LABELS[sc]
            summary_rows.append(s)
            if len(tdf) > 0:
                tdf = tdf.assign(scaling=sc, policy=name)
                per_trade_rows.append(tdf)
            print(f"  {name:<26s}  trades={s['trades']}  ret={s['total_return']:+.2%}  "
                  f"DD=${s['max_drawdown']:,.0f}  sharpe={s['sharpe_per_trade']:+.3f}  "
                  f"realized-stake={s['mean_realized_wager_pct']*100:.2f}%  "
                  f"cum-liq-bind={s['share_cumulative_binding']*100:.0f}%")

    summary_df = pd.DataFrame(summary_rows)
    save_table(summary_df, "trade_cap_policy_liq_scaling_summary")

    if per_trade_rows:
        per_trade = pd.concat(per_trade_rows, ignore_index=True)
        keep = [
            "scaling","policy","game_id","game_ts","entry_ts","side","edge",
            "kelly_f","cap_pct_used","wager_kelly","wager_post_portfolio",
            "rolling_cap","cumulative_cap","allowed_dollars","wager",
            "n_contracts","fill_pct","entry_px_actual","binding",
            "visible_pool","recent_pool","cumulative_pool",
            "fee","pnl","won","bankroll","bankroll_before",
        ]
        save_table(per_trade[keep], "trade_cap_policy_liq_scaling_per_trade")

    plot_scaling_metrics(summary_df, "trade_cap_policy_liq_scaling_metrics_grid")
    plot_scaling_binding(summary_df, "trade_cap_policy_liq_scaling_binding_distribution")


if __name__ == "__main__":
    main()
