"""Sweep maximum-exposure caps and rerun the half-Kelly $5k sweep backtest.

Default strategy config: live half-Kelly best thresholds (edge_min = 0.05,
norm_min = 0.25), half-life entry window, half-Kelly fraction = 0.5,
$5,000 starting bankroll, sweep execution against historical Kalshi tape.

Add: cap each trade's intended wager at `max_pct * bankroll_before` before
the liquidity sweep is applied. This is the standard Kelly-cap technique
for clipping the tail of position sizes (max ideal wager observed in the
uncapped strategy is ~34% of bankroll). Caps tested:
    6%, 8%, 10%, 12%, 15%   (plus the uncapped baseline)

Outputs:
    organized/outputs/trade_max_exposure_sweep_summary.csv
    organized/outputs/trade_max_exposure_sweep_metrics.png
    organized/outputs/trade_max_exposure_sweep_equity_curves.png
"""
from __future__ import annotations

import sys, warnings, math
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
    collect_entries, add_trade_returns, equity_by_payout,
)
from final_model import load_year, LABEL_COL, kalshi_taker_fee  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"

EXPOSURE_CAPS = [None, 0.06, 0.08, 0.10, 0.12, 0.15]


# --------------------------------------------------------------------------- #
# Capped sweep engine                                                         #
# --------------------------------------------------------------------------- #

def run_kelly_sweep_with_cap(entries, fraction, trade_data, bankroll_init,
                              max_pct: float | None):
    """Sweep execution with a per-trade exposure cap.

    For each trade the ideal half-Kelly wager is first computed, then capped
    at `max_pct * bankroll_before` (no cap if `max_pct is None`), then
    fulfilled by sweeping historical taker volume at <= entry price (VWAP fill).

    Returns the same per-trade record schema as `trading.run_kelly_sweep`,
    plus a `cap_binding` flag indicating whether the cap was the constraint.
    """
    ents = sorted(entries, key=lambda x: x["entry_ts"])
    bankroll = bankroll_init
    out = []
    for e in ents:
        kf = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_kelly = kf * fraction * bankroll
        if wager_kelly < 0.01:
            continue
        cap_value = (max_pct * bankroll) if max_pct is not None else float("inf")
        wager_ideal = min(wager_kelly, cap_value)
        cap_binding = max_pct is not None and wager_kelly > cap_value

        n_ideal = wager_ideal / e["entry_px"]
        gtr = trade_data[trade_data["game_id"] == e["game_id"]].copy()
        if gtr.empty:
            continue
        gtr["our_price"] = gtr["yes_price"] if e["side"] == "YES" else gtr["no_price"]
        qual = gtr[gtr["our_price"] <= e["entry_px"]].sort_values("our_price")
        if qual.empty:
            continue
        filled, cost = 0.0, 0.0
        for _, t in qual.iterrows():
            take = min(t["count"], n_ideal - filled)
            filled += take
            cost += take * t["our_price"]
            if filled >= n_ideal:
                break
        if filled == 0:
            continue
        vwap = cost / filled
        n_actual = filled
        wager_actual = cost
        fee = kalshi_taker_fee(n_actual, vwap)
        won_payoff = 1.0 if (e["side"] == "YES" and e["home_win"] == 1) \
                     or (e["side"] == "NO" and e["home_win"] == 0) else 0.0
        pnl = n_actual * won_payoff - wager_actual - fee
        bankroll += pnl
        out.append({
            **e, "kelly_f": kf,
            "wager_kelly":      wager_kelly,
            "wager_ideal":      wager_ideal,
            "wager":            wager_actual,
            "n_ideal":          n_ideal,
            "n_contracts":      n_actual,
            "fill_pct":         n_actual / n_ideal if n_ideal > 0 else 0,
            "entry_px_actual":  vwap,
            "cap_binding":      cap_binding,
            "fee":              fee,
            "pnl":              pnl,
            "won":              int(pnl > 0),
            "bankroll":         bankroll,
        })
    return out


# --------------------------------------------------------------------------- #
# Per-cap evaluation                                                          #
# --------------------------------------------------------------------------- #

def evaluate(cap, ticker_info, pretip, wt):
    ents = collect_entries(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
    )
    raw = run_kelly_sweep_with_cap(
        ents, KELLY_FRACTION, wt, BANKROLL_REAL, max_pct=cap,
    )
    if not raw:
        return None
    tdf = add_trade_returns(pd.DataFrame(raw))
    eq  = equity_by_payout(tdf, bankroll_init=BANKROLL_REAL, ts_col="game_ts")

    fb_engine    = float(tdf["bankroll"].iloc[-1])
    dd_engine    = float((tdf["bankroll"].cummax() - tdf["bankroll"]).max())
    mean_lr      = float(tdf["log_ret"].mean())
    std_lr       = float(tdf["log_ret"].std())
    sharpe       = mean_lr / std_lr if std_lr > 0 else float("nan")

    bb = tdf["bankroll"] - tdf["pnl"]
    intended_wager_pct = (tdf["wager_ideal"] / bb).mean()
    realized_wager_pct = (tdf["wager"] / bb).mean()
    cap_bind_share     = float(tdf["cap_binding"].mean()) if "cap_binding" in tdf.columns else 0.0

    return {
        "max_exposure_pct":       (cap * 100) if cap is not None else float("nan"),
        "label":                  ("Uncapped" if cap is None else f"{int(cap*100)}% cap"),
        "trades":                 int(len(tdf)),
        "hit_rate":               float(tdf["won"].mean()),
        "total_return":           (fb_engine - BANKROLL_REAL) / BANKROLL_REAL,
        "final_bankroll":         fb_engine,
        "max_drawdown":           dd_engine,
        "mean_log_return":        mean_lr,
        "std_log_return":         std_lr,
        "sharpe_per_trade":       sharpe,
        "mean_fill_rate":         float(tdf["fill_pct"].mean()),
        "mean_intended_wager_pct": float(intended_wager_pct),
        "mean_realized_wager_pct": float(realized_wager_pct),
        "share_trades_cap_bound":  cap_bind_share,
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
    test_2025["game_ts"] = pd.to_datetime(test_2025["game_ts"], utc=True)
    signals = test_2025.merge(model_w, on="game_id").merge(elo_w, on="game_id")

    print("Building Kalshi trading index ...")
    idx = build_kalshi_trading_index(signals, pred_cols=("p_full_model", "p_elo"))
    ticker_info, pretip, wt = idx["ticker_info"], idx["pretip"], idx["wt"]

    rows, frames = [], {}
    for cap in EXPOSURE_CAPS:
        result = evaluate(cap, ticker_info, pretip, wt)
        if result is None:
            print(f"  cap={cap}: no trades")
            continue
        summary, tdf, eq = result
        rows.append(summary)
        frames[summary["label"]] = (tdf, eq)
        print(f"  {summary['label']:<10s}  trades={summary['trades']}  "
              f"return={summary['total_return']:+.2%}  "
              f"DD=${summary['max_drawdown']:,.0f}  "
              f"sharpe={summary['sharpe_per_trade']:+.3f}  "
              f"realized-stake={summary['mean_realized_wager_pct']*100:.2f}%  "
              f"cap-bind={summary['share_trades_cap_bound']*100:.0f}%")

    summary_tbl = pd.DataFrame(rows)
    save_table(summary_tbl, "trade_max_exposure_sweep_summary")
    print()
    print(summary_tbl.to_string(index=False))

    # ---- Metrics curves (uncapped is plotted at x = 100 for visual reference) ----
    finite = summary_tbl[summary_tbl["max_exposure_pct"].notna()]\
        .sort_values("max_exposure_pct").reset_index(drop=True)
    uncapped = summary_tbl[summary_tbl["max_exposure_pct"].isna()]\
        .reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = finite["max_exposure_pct"].values

    ax = axes[0, 0]
    ax.plot(x, finite["total_return"] * 100, "o-", color="#3498db", lw=2, ms=7, label="Capped")
    if len(uncapped):
        ax.axhline(uncapped["total_return"].iloc[0] * 100, color="#7f8c8d",
                   linestyle="--", alpha=0.7,
                   label=f"Uncapped ({uncapped['total_return'].iloc[0]*100:+.0f}%)")
    for xi, v in zip(x, finite["total_return"]):
        ax.text(xi, v * 100 + 30, f"{v*100:+.0f}%", ha="center", fontsize=8)
    ax.set_xlabel("Max exposure cap (% of bankroll)")
    ax.set_ylabel("Total return (%)")
    ax.set_title("Total Return", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(x, finite["sharpe_per_trade"], "s-", color="#e74c3c", lw=2, ms=7, label="Capped")
    if len(uncapped):
        ax.axhline(uncapped["sharpe_per_trade"].iloc[0], color="#7f8c8d",
                   linestyle="--", alpha=0.7,
                   label=f"Uncapped ({uncapped['sharpe_per_trade'].iloc[0]:+.3f})")
    for xi, v in zip(x, finite["sharpe_per_trade"]):
        ax.text(xi, v + 0.005, f"{v:+.3f}", ha="center", fontsize=8)
    ax.set_xlabel("Max exposure cap (% of bankroll)")
    ax.set_ylabel("Per-trade Sharpe")
    ax.set_title("Per-Trade Sharpe", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x, finite["max_drawdown"], "v-", color="#9b59b6", lw=2, ms=7, label="Capped")
    if len(uncapped):
        ax.axhline(uncapped["max_drawdown"].iloc[0], color="#7f8c8d",
                   linestyle="--", alpha=0.7,
                   label=f"Uncapped (${uncapped['max_drawdown'].iloc[0]:,.0f})")
    for xi, v in zip(x, finite["max_drawdown"]):
        ax.text(xi, v + 700, f"${v:,.0f}", ha="center", fontsize=8)
    ax.set_xlabel("Max exposure cap (% of bankroll)")
    ax.set_ylabel("Max drawdown ($)")
    ax.set_title("Max Drawdown", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, finite["share_trades_cap_bound"] * 100, "D-",
            color="#2ecc71", lw=2, ms=7, label="Cap-bound trade share")
    ax.plot(x, finite["mean_realized_wager_pct"] * 100, "o-",
            color="#e67e22", lw=2, ms=7, label="Mean realized stake %")
    ax.plot(x, finite["mean_intended_wager_pct"] * 100, "s--",
            color="#3498db", lw=1.5, ms=6, alpha=0.85,
            label="Mean intended stake %")
    for xi, v in zip(x, finite["share_trades_cap_bound"]):
        ax.text(xi, v * 100 + 1.5, f"{v*100:.0f}%", ha="center", fontsize=7.5,
                color="#2ecc71")
    ax.set_xlabel("Max exposure cap (% of bankroll)")
    ax.set_ylabel("%")
    ax.set_title("Cap Bite & Stake Sizes", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Maximum-Exposure Cap Sweep — Half-Kelly Sweep, $5,000 bankroll",
        fontweight="bold", fontsize=13, y=1.0,
    )
    plt.tight_layout()
    save_fig(fig, "trade_max_exposure_sweep_metrics")
    plt.close(fig)

    # ---- Overlaid equity curves ----
    palette = plt.cm.viridis(np.linspace(0.05, 0.92, len(frames)))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for color, (label, (tdf, eq)) in zip(palette, frames.items()):
        r = next(rr for rr in rows if rr["label"] == label)
        ax.step(eq["game_ts"], eq["display_bankroll"], where="post",
                color=color, linewidth=1.6,
                label=f"{label} ({r['total_return']:+.0%}, n={r['trades']})")
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6,
               label=f"Starting bankroll (${BANKROLL_REAL:,.0f})")
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title(
        "2025 Half-Kelly Sweep — Equity Curves Across Maximum-Exposure Caps",
        fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "trade_max_exposure_sweep_equity_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
