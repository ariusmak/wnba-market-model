"""Sweep T-Nh pre-tipoff cutoffs and compare half-Kelly $5k sweep performance.

Tests cutoffs at 0, 1, 2, 3, 4, 6, 8, 12 hours before tipoff. For each cutoff,
records trade count, return, drawdown, mean log return, per-trade Sharpe, hit
rate, fill rate, and lead-time stats. Plots the resulting curves so the
hypothesis ("late-window trades face informed flow we can't see") can be read
as a continuous response, not just two points.

Outputs:
    organized/outputs/trade_cutoff_sweep_summary.csv
    organized/outputs/trade_cutoff_sweep_metrics.png
    organized/outputs/trade_cutoff_sweep_equity_curves.png
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
    run_kelly_sweep, add_trade_returns, equity_by_payout, LABEL,
)
from final_model import load_year, LABEL_COL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"

CUTOFFS_HOURS = [0, 1, 2, 3, 4, 6, 8, 12]


def collect_entries_with_cutoff(ticker_info, pretip, model_col,
                                 edge_min, norm_edge_min, entry_window,
                                 cutoff_hours: float, label_col=LABEL):
    key_map = {"half_life": "entry_start_half",
                "two_thirds_life": "entry_start_twothirds"}
    key = key_map[entry_window]
    out = []
    for tkr, info in ticker_info.items():
        if tkr not in pretip:
            continue
        cutoff_ts = info["game_ts"]
        if cutoff_hours > 0:
            cutoff_ts = info["game_ts"] - pd.Timedelta(hours=cutoff_hours)
        if cutoff_ts <= info[key]:
            continue
        p, hw = info[model_col], info[label_col]
        candle_df = pretip[tkr]
        eligible = candle_df[(candle_df["ts"] >= info[key]) &
                              (candle_df["ts"] <= cutoff_ts)]
        if eligible.empty:
            continue
        for _, row in eligible.iterrows():
            if row["ts"].minute % 15 != 0:
                continue
            yb, ya = row["yes_bid_close"], row["yes_ask_close"]
            if not (0 < yb < 1 and 0 < ya < 1):
                continue
            q_yes, q_no = ya, 1 - yb
            edge_yes, edge_no = p - q_yes, (1 - p) - q_no
            if edge_yes >= edge_no:
                side, entry_px, edge, p_side = "YES", q_yes, edge_yes, p
            else:
                side, entry_px, edge, p_side = "NO", q_no, edge_no, 1 - p
            if edge < edge_min:
                continue
            norm_edge = edge / entry_px if entry_px > 0 else 0
            if norm_edge_min > 0 and norm_edge < norm_edge_min:
                continue
            out.append({
                "game_id": info["game_id"], "game_ts": info["game_ts"],
                "side": side, "entry_px": entry_px,
                "entry_ts": row["ts"], "edge": edge, "norm_edge": norm_edge,
                "p_side": p_side, "p_model": p, "home_win": hw,
            })
            break
    return out


def evaluate(cutoff_hours, ticker_info, pretip, wt):
    ents = collect_entries_with_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
        cutoff_hours=cutoff_hours,
    )
    raw = run_kelly_sweep(ents, KELLY_FRACTION, wt, BANKROLL_REAL)
    if not raw:
        return None
    tdf = add_trade_returns(pd.DataFrame(raw))
    eq  = equity_by_payout(tdf, bankroll_init=BANKROLL_REAL, ts_col="game_ts")
    fb_engine = float(tdf["bankroll"].iloc[-1])
    dd_engine = float((tdf["bankroll"].cummax() - tdf["bankroll"]).max())
    mean_lr   = float(tdf["log_ret"].mean())
    std_lr    = float(tdf["log_ret"].std())
    sharpe    = mean_lr / std_lr if std_lr > 0 else float("nan")
    lead_h = ((tdf["game_ts"] - tdf["entry_ts"]).dt.total_seconds() / 3600.0)
    return {
        "cutoff_hours":      int(cutoff_hours),
        "trades":            int(len(tdf)),
        "hit_rate":          float(tdf["won"].mean()),
        "total_return":      (fb_engine - BANKROLL_REAL) / BANKROLL_REAL,
        "final_bankroll":    fb_engine,
        "max_drawdown":      dd_engine,
        "mean_log_return":   mean_lr,
        "std_log_return":    std_lr,
        "sharpe_per_trade":  sharpe,
        "mean_fill_rate":    float(tdf["fill_pct"].mean()),
        "median_lead_h":     float(lead_h.median()),
        "min_lead_h":        float(lead_h.min()),
    }, tdf, eq


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

    rows, frames = [], {}
    for h in CUTOFFS_HOURS:
        result = evaluate(h, ticker_info, pretip, wt)
        if result is None:
            print(f"  T-{h}h: no trades")
            continue
        summary, tdf, eq = result
        rows.append(summary)
        frames[h] = (tdf, eq)
        print(f"  T-{h}h: trades={summary['trades']}  "
              f"return={summary['total_return']:+.2%}  "
              f"DD=${summary['max_drawdown']:.0f}  "
              f"sharpe={summary['sharpe_per_trade']:+.3f}  "
              f"hit={summary['hit_rate']:.1%}  "
              f"min-lead={summary['min_lead_h']:.1f}h")

    summary_tbl = pd.DataFrame(rows).sort_values("cutoff_hours").reset_index(drop=True)
    save_table(summary_tbl, "trade_cutoff_sweep_summary")
    print()
    print(summary_tbl.to_string(index=False))

    # ---- Metrics curve figure (4-panel) ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = summary_tbl["cutoff_hours"].values

    ax = axes[0, 0]
    ax.plot(x, summary_tbl["total_return"] * 100, "o-", color="#3498db", lw=2, ms=7)
    for xi, v in zip(x, summary_tbl["total_return"]):
        ax.text(xi, v * 100 + 30, f"{v*100:+.0f}%", ha="center", fontsize=8)
    ax.set_xlabel("Cutoff (hours before tipoff)")
    ax.set_ylabel("Total return (%)")
    ax.set_title("Total Return", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, summary_tbl["sharpe_per_trade"], "s-", color="#e74c3c", lw=2, ms=7)
    for xi, v in zip(x, summary_tbl["sharpe_per_trade"]):
        ax.text(xi, v + 0.005, f"{v:+.3f}", ha="center", fontsize=8)
    ax.set_xlabel("Cutoff (hours before tipoff)")
    ax.set_ylabel("Per-trade Sharpe (mean / std log return)")
    ax.set_title("Per-Trade Sharpe", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, summary_tbl["hit_rate"] * 100, "D-", color="#2ecc71", lw=2, ms=7)
    for xi, v in zip(x, summary_tbl["hit_rate"]):
        ax.text(xi, v * 100 + 0.4, f"{v*100:.1f}%", ha="center", fontsize=8)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cutoff (hours before tipoff)")
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Hit Rate", fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x, summary_tbl["trades"], width=0.7, color="#9b59b6", alpha=0.75,
           edgecolor="white")
    for xi, v in zip(x, summary_tbl["trades"]):
        ax.text(xi, v + 1.5, f"{int(v)}", ha="center", fontsize=8)
    ax.set_xlabel("Cutoff (hours before tipoff)")
    ax.set_ylabel("Trades taken")
    ax.set_title("Number of Trades", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Pre-Tipoff Cutoff Sweep — Half-Kelly Sweep, $5,000 bankroll",
                 fontweight="bold", fontsize=13, y=1.00)
    plt.tight_layout()
    save_fig(fig, "trade_cutoff_sweep_metrics")
    plt.close(fig)

    # ---- Overlaid equity curves ----
    cmap = plt.cm.viridis(np.linspace(0.05, 0.92, len(frames)))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, h in zip(cmap, sorted(frames.keys())):
        tdf, eq = frames[h]
        r = next(rr for rr in rows if rr["cutoff_hours"] == h)
        label = f"T-{h}h ({r['total_return']:+.0%}, n={r['trades']})"
        if h == 0:
            label = f"Baseline (no cutoff) ({r['total_return']:+.0%}, n={r['trades']})"
        ax.step(eq["game_ts"], eq["display_bankroll"], where="post",
                color=color, linewidth=1.6, label=label)
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6,
               label=f"Starting bankroll (${BANKROLL_REAL:,.0f})")
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title(
        "2025 Half-Kelly Sweep — Equity Curves Across Pre-Tipoff Cutoffs",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "trade_cutoff_sweep_equity_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
