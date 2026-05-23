"""Dynamic price-aware half-Kelly sweep vs. static (price-sorted) sweep.

Current code (`trading.run_kelly_sweep`) computes the half-Kelly target ONCE
at entry, at price q_max, producing a fixed contract target n_ideal. The
sweep then sorts qualifying historical trades by price (cheapest first) and
fills up to n_ideal contracts. This is a best-case execution assumption.

The new (dynamic) engine implements the user's correction:

  - walk historical trades in time order from entry_ts onward
  - at each trade printed at our-side price p, compute Kelly fraction at p:
        f*(p) = (p_side - p) / (1 - p)
    and half-Kelly stake at p:
        wager_target(p) = fraction * f*(p) * bankroll_at_entry
  - if already_filled_dollars < wager_target(p):
        take_dollars = min(trade_cost, wager_target(p) - already_filled)
        take_count   = take_dollars / p
        already_filled += take_dollars
  - never pay more than q_max (the entry filter)
  - keep walking until end of entry window

This means: as price falls below q_max during the window, the bot keeps
filling because half-Kelly at lower prices is larger. As price rises back
toward q_max, headroom shrinks and the bot stops. If price falls again,
it resumes.

Strategy config (identical for both engines, fair comparison):
  half-Kelly best (edge_min = 0.05, norm_min = 0.25), half-life entry window,
  T-8h timing gate, half-Kelly fraction = 0.5, $5,000 starting bankroll.

Outputs (in organized/outputs/):
  trade_dynamic_kelly_summary.csv
  trade_dynamic_kelly_per_trade.csv
  trade_dynamic_kelly_equity_curves.png
  trade_dynamic_kelly_position_sizes.png
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
from final_model import load_year, LABEL_COL, kalshi_taker_fee  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"
TIMING_CUTOFF_H = 8.0


def collect_entries_cutoff(ticker_info, pretip, model_col, edge_min,
                            norm_edge_min, entry_window, cutoff_hours,
                            label_col=LABEL):
    key_map = {"half_life": "entry_start_half",
               "two_thirds_life": "entry_start_twothirds"}
    key = key_map[entry_window]
    out = []
    for tkr, info in ticker_info.items():
        if tkr not in pretip: continue
        cutoff_ts = info["game_ts"] - pd.Timedelta(hours=cutoff_hours) \
                    if cutoff_hours and cutoff_hours > 0 else info["game_ts"]
        if cutoff_ts <= info[key]: continue
        p, hw = info[model_col], info[label_col]
        candle_df = pretip[tkr]
        eligible = candle_df[(candle_df["ts"] >= info[key]) &
                              (candle_df["ts"] <= cutoff_ts)]
        if eligible.empty: continue
        for _, row in eligible.iterrows():
            if row["ts"].minute % 15 != 0: continue
            yb, ya = row["yes_bid_close"], row["yes_ask_close"]
            if not (0 < yb < 1 and 0 < ya < 1): continue
            q_yes, q_no = ya, 1 - yb
            edge_yes, edge_no = p - q_yes, (1 - p) - q_no
            if edge_yes >= edge_no:
                side, entry_px, edge, p_side = "YES", q_yes, edge_yes, p
            else:
                side, entry_px, edge, p_side = "NO", q_no, edge_no, 1 - p
            if edge < edge_min: continue
            norm_edge = edge / entry_px if entry_px > 0 else 0
            if norm_edge_min > 0 and norm_edge < norm_edge_min: continue
            out.append({
                "game_id": info["game_id"], "game_ts": info["game_ts"],
                "side": side, "entry_px": entry_px,
                "entry_ts": row["ts"], "edge": edge, "norm_edge": norm_edge,
                "p_side": p_side, "p_model": p, "home_win": hw,
            })
            break
    return out


# --------------------------------------------------------------------------- #
# Dynamic (price-aware) half-Kelly sweep                                      #
# --------------------------------------------------------------------------- #

def run_kelly_sweep_dynamic(entries, fraction, trade_data, bankroll_init):
    """Time-ordered, price-aware fill. See module docstring for the rule."""
    ents = sorted(entries, key=lambda x: x["entry_ts"])
    bankroll = bankroll_init
    out = []
    for e in ents:
        kf_qmax = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_kelly_at_qmax = kf_qmax * fraction * bankroll
        if wager_kelly_at_qmax < 0.01:
            continue

        gtr = trade_data[trade_data["game_id"] == e["game_id"]].copy()
        if gtr.empty:
            continue
        gtr = gtr[gtr["ts"] >= e["entry_ts"]].sort_values("ts")
        if gtr.empty:
            continue
        gtr["our_price"] = gtr["yes_price"] if e["side"] == "YES" else gtr["no_price"]
        # Drop trades above q_max (we'd never lift them) and trades that
        # cross our_price out of (0, 1).
        gtr = gtr[(gtr["our_price"] <= e["entry_px"]) &
                  (gtr["our_price"] > 0) & (gtr["our_price"] < 1)]
        if gtr.empty:
            continue

        already_dollars = 0.0
        already_count   = 0.0
        # Diagnostics
        n_pause_events = 0  # times we hit headroom 0 and stopped on a trade
        n_resume_events = 0  # times we resumed after pausing
        was_paused = False

        for _, t in gtr.iterrows():
            p = float(t["our_price"])
            kf_p = (e["p_side"] - p) / (1 - p)
            if kf_p <= 0:
                continue
            wager_target_p = fraction * kf_p * bankroll
            headroom = wager_target_p - already_dollars
            if headroom <= 0:
                n_pause_events += 1
                was_paused = True
                continue
            if was_paused:
                n_resume_events += 1
                was_paused = False
            trade_cost = float(t["count"]) * p
            take_cost  = min(trade_cost, headroom)
            take_n     = take_cost / p
            already_dollars += take_cost
            already_count   += take_n

        if already_count == 0:
            continue
        vwap = already_dollars / already_count
        n_actual = already_count
        wager_actual = already_dollars
        fee = kalshi_taker_fee(n_actual, vwap)
        won_payoff = 1.0 if (e["side"] == "YES" and e["home_win"] == 1) \
                     or (e["side"] == "NO" and e["home_win"] == 0) else 0.0
        pnl = n_actual * won_payoff - wager_actual - fee
        bankroll_before = bankroll
        bankroll += pnl

        # For comparison with the static engine's n_ideal target:
        n_ideal_at_qmax = wager_kelly_at_qmax / e["entry_px"]
        fill_ratio_vs_qmax = n_actual / n_ideal_at_qmax if n_ideal_at_qmax > 0 else 0

        out.append({
            **e, "kelly_f_qmax":   kf_qmax,
            "wager_kelly_at_qmax": wager_kelly_at_qmax,
            "wager":               wager_actual,
            "n_ideal_at_qmax":     n_ideal_at_qmax,
            "n_contracts":         n_actual,
            "fill_ratio_vs_qmax":  fill_ratio_vs_qmax,
            "entry_px_actual":     vwap,
            "n_pause_events":      int(n_pause_events),
            "n_resume_events":     int(n_resume_events),
            "fee":                 fee,
            "pnl":                 pnl,
            "won":                 int(pnl > 0),
            "bankroll":            bankroll,
            "bankroll_before":     bankroll_before,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Metrics + plots                                                             #
# --------------------------------------------------------------------------- #

def evaluate(label, tdf, bankroll_init):
    if len(tdf) == 0:
        return {"engine": label, "trades": 0}
    tdf = tdf.copy()
    tdf["log_ret"] = np.log(tdf["bankroll"] / tdf["bankroll_before"])
    tdf["wager_pct"] = tdf["wager"] / tdf["bankroll_before"]
    fb = float(tdf["bankroll"].iloc[-1])
    dd = float((tdf["bankroll"].cummax() - tdf["bankroll"]).max())
    mean_lr = float(tdf["log_ret"].mean())
    std_lr  = float(tdf["log_ret"].std())
    return {
        "engine":              label,
        "trades":              int(len(tdf)),
        "hit_rate":            float(tdf["won"].mean()),
        "total_return":        (fb - bankroll_init) / bankroll_init,
        "final_bankroll":      fb,
        "max_drawdown":        dd,
        "mean_log_return":     mean_lr,
        "std_log_return":      std_lr,
        "sharpe_per_trade":    mean_lr / std_lr if std_lr > 0 else float("nan"),
        "mean_wager":          float(tdf["wager"].mean()),
        "median_wager":        float(tdf["wager"].median()),
        "max_wager":           float(tdf["wager"].max()),
        "mean_wager_pct":      float(tdf["wager_pct"].mean()),
        "median_wager_pct":    float(tdf["wager_pct"].median()),
        "max_wager_pct":       float(tdf["wager_pct"].max()),
        "mean_entry_px_actual":float(tdf["entry_px_actual"].mean()),
    }


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

    idx = build_kalshi_trading_index(signals, pred_cols=("p_full_model", "p_elo"))
    ticker_info, pretip, wt = idx["ticker_info"], idx["pretip"], idx["wt"]

    ents = collect_entries_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
        cutoff_hours=TIMING_CUTOFF_H,
    )
    print(f"Qualifying entries (T-{int(TIMING_CUTOFF_H)}h gate): {len(ents)}")

    print("Running STATIC sweep (existing) ...")
    static = pd.DataFrame(run_kelly_sweep(ents, KELLY_FRACTION, wt, BANKROLL_REAL))
    if len(static):
        static = add_trade_returns(static)
        static["wager_pct"] = static["wager"] / static["bankroll_before"]

    print("Running DYNAMIC sweep (price-aware) ...")
    dyn = run_kelly_sweep_dynamic(ents, KELLY_FRACTION, wt, BANKROLL_REAL)
    if len(dyn):
        dyn = add_trade_returns(dyn)
        dyn["wager_pct"] = dyn["wager"] / dyn["bankroll_before"]

    static_s = evaluate("Static (existing)",  static, BANKROLL_REAL)
    dyn_s    = evaluate("Dynamic (proposed)", dyn,    BANKROLL_REAL)
    summary  = pd.DataFrame([static_s, dyn_s])
    save_table(summary, "trade_dynamic_kelly_summary")
    print()
    print(summary.to_string(index=False))

    # Per-trade alignment for paired analysis (inner join on game_id)
    if len(static) and len(dyn):
        s = static[["game_id","game_ts","entry_ts","side","edge","entry_px",
                    "p_side","wager","n_contracts","entry_px_actual","won","pnl"]].rename(
            columns=lambda c: c if c == "game_id" else c + "_static"
        )
        d = dyn[["game_id","wager","n_contracts","entry_px_actual","won","pnl",
                 "n_pause_events","n_resume_events","fill_ratio_vs_qmax"]].rename(
            columns=lambda c: c if c == "game_id" else c + "_dynamic"
        )
        paired = s.merge(d, on="game_id", how="outer")
        save_table(paired, "trade_dynamic_kelly_per_trade")

    # ---- Equity curves (settlement-time) ----
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for label, tdf, color in [
        ("Static (existing)",  static, "#3498db"),
        ("Dynamic (proposed)", dyn,    "#e74c3c"),
    ]:
        if len(tdf) == 0: continue
        eq = tdf.copy()
        eq["display_bankroll"] = BANKROLL_REAL + eq["pnl"].cumsum()
        eq = eq.sort_values("game_ts")
        ret = (eq["display_bankroll"].iloc[-1] - BANKROLL_REAL) / BANKROLL_REAL
        ax.step(eq["game_ts"], eq["display_bankroll"], where="post",
                color=color, linewidth=1.8,
                label=f"{label} ({ret:+.0%}, n={len(tdf)})")
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6,
               label=f"Starting bankroll (${BANKROLL_REAL:,.0f})")
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title("Dynamic vs. Static Half-Kelly Sweep — Equity Curves",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "trade_dynamic_kelly_equity_curves"); plt.close(fig)

    # ---- Position-size distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins_pct = np.linspace(0, 20, 41)
    bins_dol = np.linspace(0, max(static["wager"].max(), dyn["wager"].max()) * 1.05, 41)
    for ax, col_data, label_xform, title, bins, fmt in [
        (axes[0], "wager_pct", lambda v: v*100,
         "Position size — % of portfolio", bins_pct, "%"),
        (axes[1], "wager", lambda v: v,
         "Position size — raw dollars",   bins_dol, "$"),
    ]:
        for label, tdf, color in [
            ("Static",  static, "#3498db"),
            ("Dynamic", dyn,    "#e74c3c"),
        ]:
            if len(tdf) == 0: continue
            vals = label_xform(tdf[col_data].values)
            ax.hist(vals, bins=bins, color=color, alpha=0.5,
                    edgecolor="white", label=f"{label} (med {np.median(vals):.1f}{fmt})")
        ax.set_xlabel(title); ax.set_ylabel("Trades")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.suptitle("Position-Size Distributions: Static vs. Dynamic Sweep",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, "trade_dynamic_kelly_position_sizes"); plt.close(fig)


if __name__ == "__main__":
    main()
