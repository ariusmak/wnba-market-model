"""Backtest the half-Kelly $5k sweep strategy with a T-3h pre-tipoff cutoff.

Hypothesis: late-breaking news (lineup announcements, last-minute scratches,
travel updates) tends to surface in the final hours before tipoff. Our model
can't see it; informed flow can. Cutting the entry window 3 hours before
tipoff should remove the trades most exposed to that informed-flow risk.

Compares two configurations on the 2025 holdout, all else equal:

    Baseline           : entries allowed up to game_ts (existing strategy)
    T-3h cutoff        : entries only at ts ≤ game_ts − 3h

Both use the live half-Kelly best config (edge_min = 0.05, norm_min = 0.25),
half-Kelly fraction = 0.5, $5,000 starting bankroll, sweep execution against
the historical Kalshi tape.

Outputs:
    organized/outputs/trade_t3_cutoff_comparison_summary.csv
    organized/outputs/trade_t3_cutoff_equity_curves.png
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
    collect_entries, run_kelly_sweep, add_trade_returns, equity_by_payout,
    LABEL,
)
from final_model import load_year, LABEL_COL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL = 5000.0
BEST_EDGE_MIN = 0.05
BEST_NORM_MIN = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW = "half_life"
CUTOFF_HOURS = 3


# --------------------------------------------------------------------------- #
# Cutoff-aware entry collector                                                #
# --------------------------------------------------------------------------- #

def collect_entries_with_cutoff(ticker_info, pretip, model_col,
                                 edge_min, norm_edge_min, entry_window,
                                 cutoff_hours: float | None,
                                 label_col=LABEL):
    """Same as `collect_entries`, but trades only enter at ts ≤ game_ts − cutoff_hours.

    Setting cutoff_hours=None (or 0) reproduces the baseline behaviour.
    """
    key_map = {"half_life": "entry_start_half",
                "two_thirds_life": "entry_start_twothirds"}
    if entry_window not in key_map:
        raise ValueError(f"unknown entry_window: {entry_window}")
    key = key_map[entry_window]

    out, skipped_no_window, skipped_no_qualifying = [], 0, 0
    for tkr, info in ticker_info.items():
        if tkr not in pretip:
            continue
        cutoff_ts = info["game_ts"]
        if cutoff_hours and cutoff_hours > 0:
            cutoff_ts = info["game_ts"] - pd.Timedelta(hours=cutoff_hours)
        if cutoff_ts <= info[key]:
            skipped_no_window += 1
            continue

        p, hw = info[model_col], info[label_col]
        candle_df = pretip[tkr]
        eligible = candle_df[(candle_df["ts"] >= info[key]) &
                              (candle_df["ts"] <= cutoff_ts)]
        if eligible.empty:
            skipped_no_qualifying += 1
            continue

        entered = False
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
                "game_id":  info["game_id"],
                "game_ts":  info["game_ts"],
                "side":     side,
                "entry_px": entry_px,
                "entry_ts": row["ts"],
                "edge":     edge,
                "norm_edge": norm_edge,
                "p_side":   p_side,
                "p_model":  p,
                "home_win": hw,
            })
            entered = True
            break
        if not entered:
            skipped_no_qualifying += 1
    return out, {
        "skipped_no_window":     skipped_no_window,
        "skipped_no_qualifying": skipped_no_qualifying,
    }


# --------------------------------------------------------------------------- #
# Per-config metrics                                                          #
# --------------------------------------------------------------------------- #

def evaluate(label, ticker_info, pretip, wt, *, cutoff_hours):
    ents, diag = collect_entries_with_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW, cutoff_hours=cutoff_hours,
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

    # Lead-time diagnostic: hours from entry to settlement
    lead_h = ((tdf["game_ts"] - tdf["entry_ts"]).dt.total_seconds() / 3600.0)
    return {
        "config":            label,
        "cutoff_hours":      cutoff_hours if cutoff_hours else 0,
        "intended_entries":  len(ents),
        "trades":            int(len(tdf)),
        "skipped_no_window": diag["skipped_no_window"],
        "skipped_no_qual":   diag["skipped_no_qualifying"],
        "hit_rate":          float(tdf["won"].mean()),
        "total_return":      (fb_engine - BANKROLL_REAL) / BANKROLL_REAL,
        "final_bankroll":    fb_engine,
        "max_drawdown":      dd_engine,
        "mean_log_return":   mean_lr,
        "std_log_return":    std_lr,
        "sharpe_per_trade":  sharpe,
        "mean_fill_rate":    float(tdf["fill_pct"].mean()),
        "median_fill_rate":  float(tdf["fill_pct"].median()),
        "median_lead_h":     float(lead_h.median()),
        "min_lead_h":        float(lead_h.min()),
    }, tdf, eq


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

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
    test_2025["game_ts"]   = pd.to_datetime(test_2025["game_ts"], utc=True)
    test_2025["game_date"] = pd.to_datetime(test_2025["game_date"])
    signals = test_2025.merge(model_w, on="game_id").merge(elo_w, on="game_id")

    print("Building Kalshi trading index ...")
    idx = build_kalshi_trading_index(signals, pred_cols=("p_full_model", "p_elo"))
    ticker_info = idx["ticker_info"]
    pretip      = idx["pretip"]
    wt          = idx["wt"]

    configs = [
        ("Baseline (no cutoff)",                None),
        (f"T-{CUTOFF_HOURS}h cutoff",           CUTOFF_HOURS),
    ]
    rows, frames = [], {}
    for label, hrs in configs:
        result = evaluate(label, ticker_info, pretip, wt, cutoff_hours=hrs)
        if result is None:
            print(f"  {label}: no trades")
            continue
        summary, tdf, eq = result
        rows.append(summary)
        frames[label] = (tdf, eq)
        print(f"  {label}: trades={summary['trades']}  "
              f"return={summary['total_return']:+.2%}  "
              f"DD=${summary['max_drawdown']:.0f}  "
              f"sharpe={summary['sharpe_per_trade']:+.3f}  "
              f"hit={summary['hit_rate']:.1%}  "
              f"median-lead={summary['median_lead_h']:.1f}h  "
              f"min-lead={summary['min_lead_h']:.1f}h")

    summary_tbl = pd.DataFrame(rows)
    save_table(summary_tbl, "trade_t3_cutoff_comparison_summary")
    print()
    print(summary_tbl.to_string(index=False))

    # ---- Equity curves ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    palette = {"Baseline (no cutoff)": "#3498db",
               f"T-{CUTOFF_HOURS}h cutoff": "#e74c3c"}
    for label, (tdf, eq) in frames.items():
        r = next(rr for rr in rows if rr["config"] == label)
        ax.step(eq["game_ts"], eq["display_bankroll"], where="post",
                color=palette.get(label, "black"), linewidth=1.8,
                label=f"{label} ({r['total_return']:+.0%}, n={r['trades']})")
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6,
               label=f"Starting bankroll (${BANKROLL_REAL:,.0f})")
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title(
        f"2025 Half-Kelly Sweep: Baseline vs T-{CUTOFF_HOURS}h Pre-Tipoff Cutoff "
        f"(${BANKROLL_REAL:,.0f} bankroll)",
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "trade_t3_cutoff_equity_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
