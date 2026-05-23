"""Decompose where the cap-vs-uncapped return gap is paid.

For each cap level in {6, 8, 10, 12, 15}%, run the half-Kelly $5k sweep
backtest in lock-step with the uncapped baseline. For each trade, compute:

    log_ret_uncapped  = log(W_after / W_before) under uncapped sizing
    log_ret_capped    = log(W_after / W_before) under that cap
    delta_log_ret     = log_ret_uncapped − log_ret_capped
    cap_bound         = whether the cap actually clipped this trade

The terminal-wealth gap between the two strategies is exactly
exp(sum delta_log_ret) (multiplied by starting bankroll). Reporting
delta_log_ret bucketed by **season period** answers "when in the season
was the cap-cost paid?" — the user's intuition that most of the give-up
sits in early-season trades is testable here.

Outputs:
    organized/outputs/trade_max_exposure_upside_per_trade.csv
    organized/outputs/trade_max_exposure_upside_by_period.csv
    organized/outputs/trade_max_exposure_upside_decomposition.png
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
from trading import collect_entries, add_trade_returns  # noqa: E402
from final_model import load_year, LABEL_COL, kalshi_taker_fee  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"

CAPS = [0.06, 0.08, 0.10, 0.12, 0.15]
PRIMARY_CAP = 0.10  # the cap used in the deep-dive figure


# --------------------------------------------------------------------------- #
# Sweep engine with optional cap                                              #
# --------------------------------------------------------------------------- #

def run_sweep(entries, fraction, trade_data, bankroll_init, max_pct=None):
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
            "game_id":     e["game_id"],
            "game_ts":     e["game_ts"],
            "entry_ts":    e["entry_ts"],
            "side":        e["side"],
            "edge":        e["edge"],
            "kelly_f":     kf,
            "wager_kelly": wager_kelly,
            "wager_ideal": wager_ideal,
            "wager":       wager_actual,
            "n_actual":    n_actual,
            "won":         int(pnl > 0),
            "fee":         fee,
            "pnl":         pnl,
            "bankroll":    bankroll,
            "cap_binding": cap_binding,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Decomposition                                                               #
# --------------------------------------------------------------------------- #

def decompose(uncapped: pd.DataFrame, capped: pd.DataFrame,
              cap_value: float) -> pd.DataFrame:
    """Aligned per-trade view of uncapped vs. capped log returns.

    Both engines see the *same* trade set in the *same* order (only the
    sizing differs and only the per-trade bankroll diverges as we go), so
    we can index by (game_id, entry_ts) for a clean inner join.
    """
    u = uncapped.copy()
    u["bankroll_before_unc"] = u["bankroll"] - u["pnl"]
    u["log_ret_unc"]         = np.log(u["bankroll"] / u["bankroll_before_unc"])

    c = capped.copy()
    c["bankroll_before_cap"] = c["bankroll"] - c["pnl"]
    c["log_ret_cap"]         = np.log(c["bankroll"] / c["bankroll_before_cap"])

    keep_u = ["game_id","entry_ts","game_ts","side","edge","won",
              "wager_ideal","wager","pnl","bankroll","log_ret_unc"]
    keep_c = ["game_id","entry_ts","wager_ideal","wager","pnl","bankroll",
              "log_ret_cap","cap_binding"]
    merged = (
        u[keep_u].rename(columns={"wager_ideal":"wager_ideal_unc",
                                  "wager":"wager_unc",
                                  "pnl":"pnl_unc",
                                  "bankroll":"bankroll_unc"})
        .merge(
            c[keep_c].rename(columns={"wager_ideal":"wager_ideal_cap",
                                      "wager":"wager_cap",
                                      "pnl":"pnl_cap",
                                      "bankroll":"bankroll_cap"}),
            on=["game_id","entry_ts"], how="outer",
        )
    )
    # Trades that exist in only one frame can't contribute to a paired delta
    merged = merged.dropna(subset=["log_ret_unc","log_ret_cap"]).copy()
    merged["delta_log_ret"] = merged["log_ret_unc"] - merged["log_ret_cap"]
    merged["cap_pct"] = cap_value * 100
    merged["game_date"] = pd.to_datetime(merged["game_ts"]).dt.tz_convert(None).dt.normalize()
    merged["month"]     = pd.to_datetime(merged["game_ts"]).dt.tz_convert(None).dt.to_period("M")
    return merged.sort_values("game_ts").reset_index(drop=True)


def by_period(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cap_pct, period), sub in detail.groupby(["cap_pct", "month"]):
        sum_delta = float(sub["delta_log_ret"].sum())
        share = sum_delta / detail[detail["cap_pct"] == cap_pct]["delta_log_ret"].sum()
        rows.append({
            "cap_pct":             cap_pct,
            "period":              str(period),
            "n_trades":            int(len(sub)),
            "n_cap_bound":         int(sub["cap_binding"].sum()),
            "sum_delta_log_ret":   sum_delta,
            "share_of_total_gap":  float(share),
            "wealth_multiplier":   float(np.exp(sum_delta)),
            "mean_uncapped_log_ret": float(sub["log_ret_unc"].mean()),
            "mean_capped_log_ret":   float(sub["log_ret_cap"].mean()),
        })
    return pd.DataFrame(rows).sort_values(["cap_pct","period"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# --------------------------------------------------------------------------- #

def plot_decomposition(detail_primary: pd.DataFrame,
                        period_tbl: pd.DataFrame,
                        save_name: str) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    cap = detail_primary["cap_pct"].iloc[0] / 100
    detail = detail_primary.sort_values("game_ts").reset_index(drop=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: equity curves (settlement-time, cumulative log) ---
    ax = axes[0, 0]
    eq_unc = BANKROLL_REAL * np.exp(detail["log_ret_unc"].cumsum())
    eq_cap = BANKROLL_REAL * np.exp(detail["log_ret_cap"].cumsum())
    ax.step(detail["game_ts"], eq_unc, where="post", color="#3498db", lw=1.8,
            label=f"Uncapped (final ${eq_unc.iloc[-1]:,.0f})")
    ax.step(detail["game_ts"], eq_cap, where="post", color="#e74c3c", lw=1.8,
            label=f"{int(cap*100)}% cap (final ${eq_cap.iloc[-1]:,.0f})")
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title("Equity Curves: Uncapped vs Cap", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))

    # --- Panel B: cumulative log-return gap ---
    ax = axes[0, 1]
    cum_gap = detail["delta_log_ret"].cumsum()
    ax.step(detail["game_ts"], cum_gap, where="post", color="#9b59b6", lw=1.8,
            label="Cumulative Δlog-return (uncapped − cap)")
    bound = detail[detail["cap_binding"]]
    won_bound = bound[bound["won"] == 1]; lost_bound = bound[bound["won"] == 0]
    ax.scatter(won_bound["game_ts"], won_bound["delta_log_ret"].cumsum() if False else
               cum_gap.reindex(won_bound.index),
               color="#2ecc71", s=35, edgecolors="white", linewidth=0.5, zorder=3,
               label=f"Cap-bound win (n={len(won_bound)})")
    ax.scatter(lost_bound["game_ts"], cum_gap.reindex(lost_bound.index),
               color="#e74c3c", s=35, edgecolors="white", linewidth=0.5, zorder=3,
               label=f"Cap-bound loss (n={len(lost_bound)})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Settlement date")
    ax.set_ylabel("Cumulative Δ log-return")
    ax.set_title("Where the Cap Cost Is Paid (Cumulative)", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))

    # --- Panel C: monthly delta-log-return contribution ---
    ax = axes[1, 0]
    mtbl = period_tbl[period_tbl["cap_pct"] == cap * 100].sort_values("period")
    x = np.arange(len(mtbl))
    bars = ax.bar(x, mtbl["sum_delta_log_ret"], width=0.65, color="#9b59b6",
                  alpha=0.75, edgecolor="white")
    for xi, n_bound, share, v in zip(x, mtbl["n_cap_bound"],
                                      mtbl["share_of_total_gap"],
                                      mtbl["sum_delta_log_ret"]):
        ax.text(xi, v + 0.015, f"{share*100:.0f}%\n(bound={int(n_bound)})",
                ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(mtbl["period"], fontsize=9, rotation=15)
    ax.set_xlabel("Month")
    ax.set_ylabel("Sum of Δ log-return contributions")
    ax.set_title("Monthly Cap-Cost Contribution", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # --- Panel D: top 10 single-trade contributors ---
    ax = axes[1, 1]
    top = detail.sort_values("delta_log_ret", ascending=False).head(10).iloc[::-1]
    ypos = np.arange(len(top))
    colors = ["#2ecc71" if w else "#e74c3c" for w in top["won"]]
    ax.barh(ypos, top["delta_log_ret"], color=colors, alpha=0.85, edgecolor="white")
    labels = [
        f"{ts.strftime('%m-%d')}  edge={ed:.2f}  "
        f"{'cap' if cb else 'no-cap'}"
        for ts, ed, cb in zip(top["game_ts"], top["edge"], top["cap_binding"])
    ]
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=8)
    for yi, v in zip(ypos, top["delta_log_ret"]):
        ax.text(v + 0.001, yi, f"{v:+.4f}", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Δ log-return contribution")
    ax.set_title("Top 10 Single-Trade Contributors to the Cap-Cost Gap",
                 fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Maximum-Exposure Cap Decomposition — Uncapped vs. {int(cap*100)}% cap "
        f"(half-Kelly sweep, ${BANKROLL_REAL:,.0f} bankroll)",
        fontweight="bold", fontsize=13, y=1.0,
    )
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


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
    ticker_info, pretip, wt = idx["ticker_info"], idx["pretip"], idx["wt"]

    ents = collect_entries(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
    )
    print(f"  qualifying entries: {len(ents)}")

    print("Running uncapped baseline ...")
    uncapped = run_sweep(ents, KELLY_FRACTION, wt, BANKROLL_REAL, max_pct=None)

    detail_frames, period_frames = [], []
    for cap in CAPS:
        capped = run_sweep(ents, KELLY_FRACTION, wt, BANKROLL_REAL, max_pct=cap)
        detail = decompose(uncapped, capped, cap_value=cap)
        per    = by_period(detail)
        detail_frames.append(detail)
        period_frames.append(per)
        total_gap = float(detail["delta_log_ret"].sum())
        wealth_mult = np.exp(total_gap)
        bound_n = int(detail["cap_binding"].sum())
        print(f"  cap={int(cap*100)}%  total delta-log-ret={total_gap:+.4f}  "
              f"wealth-mult={wealth_mult:.3f}x  bound-trades={bound_n}/{len(detail)}")

    detail_all = pd.concat(detail_frames, ignore_index=True)
    period_all = pd.concat(period_frames, ignore_index=True)

    save_table(detail_all, "trade_max_exposure_upside_per_trade")
    save_table(period_all, "trade_max_exposure_upside_by_period")

    detail_primary = detail_all[detail_all["cap_pct"] == PRIMARY_CAP * 100].copy()
    plot_decomposition(detail_primary, period_all,
                        "trade_max_exposure_upside_decomposition")

    print()
    print("Per-period decomposition (10% cap):")
    print(period_all[period_all["cap_pct"] == PRIMARY_CAP * 100][
        ["period","n_trades","n_cap_bound","sum_delta_log_ret",
         "share_of_total_gap","wealth_multiplier"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
