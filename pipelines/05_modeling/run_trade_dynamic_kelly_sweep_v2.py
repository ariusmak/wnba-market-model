"""Dynamic price-aware half-Kelly sweep — v2 with ask-lift filter.

Fix vs v1: the v1 engine walked ALL historical trades in time order
(including bid-hits, where someone sold at the prevailing bid). A real
buyer can only lift the ASK, not pay the bid — bid-hit prints were
fictitious liquidity that the v1 engine "walked down" into. v2 filters to
ask-lifts only (taker_side == our side), so the price sequence is the
actual sequence of best-asks a buyer could have lifted.

Same fix applied to the static engine for a fair comparison.

Engines:
  Static (price-sorted, ask-lifts only)  — cheapest-first sweep
  Dynamic (time-ordered, price-aware)    — walk asks chronologically,
                                            recompute Kelly at each price,
                                            cap each fill at headroom under
                                            half-Kelly at that price

Config: half-Kelly best (edge 0.05, norm 0.25), T-8h gate, $5k bankroll.

Outputs (in organized/outputs/):
  trade_dynamic_kelly_v2_summary.csv
  trade_dynamic_kelly_v2_per_trade.csv
  trade_dynamic_kelly_v2_equity_curves.png
  trade_dynamic_kelly_v2_position_sizes.png
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
from trading import add_trade_returns, LABEL  # noqa: E402
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


def asklift_prints(trade_data: pd.DataFrame, game_id, side, entry_ts):
    """Return ask-lift prints in this game, AFTER entry_ts, on our side, with
    our-side price column."""
    g = trade_data[trade_data["game_id"] == game_id].copy()
    if g.empty:
        return g
    g = g[g["ts"] >= entry_ts]
    target = "yes" if side == "YES" else "no"
    g = g[g["taker_side"] == target]
    if g.empty:
        return g
    g["our_price"] = g["yes_price"] if side == "YES" else g["no_price"]
    return g.sort_values("ts")


# --------------------------------------------------------------------------- #
# Static (price-sorted, ask-lifts only)                                       #
# --------------------------------------------------------------------------- #

def run_static_asklift(entries, fraction, trade_data, bankroll_init):
    ents = sorted(entries, key=lambda x: x["entry_ts"])
    bankroll = bankroll_init
    out = []
    for e in ents:
        kf = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_kelly = kf * fraction * bankroll
        if wager_kelly < 0.01:
            continue
        n_ideal = wager_kelly / e["entry_px"]

        g = asklift_prints(trade_data, e["game_id"], e["side"], e["entry_ts"])
        if g.empty: continue
        g = g[(g["our_price"] <= e["entry_px"]) & (g["our_price"] > 0) & (g["our_price"] < 1)]
        if g.empty: continue
        g = g.sort_values("our_price")  # cheapest-first sweep

        filled, cost = 0.0, 0.0
        for _, t in g.iterrows():
            take = min(t["count"], n_ideal - filled)
            filled += take
            cost += take * t["our_price"]
            if filled >= n_ideal:
                break
        if filled == 0: continue

        vwap = cost / filled
        n_actual = filled
        wager_actual = cost
        fee = kalshi_taker_fee(n_actual, vwap)
        won_payoff = 1.0 if (e["side"] == "YES" and e["home_win"] == 1) \
                     or (e["side"] == "NO" and e["home_win"] == 0) else 0.0
        pnl = n_actual * won_payoff - wager_actual - fee
        bankroll_before = bankroll
        bankroll += pnl
        out.append({
            **e, "kelly_f_qmax": kf, "wager_kelly_at_qmax": wager_kelly,
            "wager": wager_actual, "n_ideal_at_qmax": n_ideal,
            "n_contracts": n_actual,
            "fill_ratio_vs_qmax": n_actual / n_ideal if n_ideal > 0 else 0,
            "entry_px_actual": vwap, "fee": fee, "pnl": pnl,
            "won": int(pnl > 0),
            "bankroll": bankroll, "bankroll_before": bankroll_before,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Dynamic (time-ordered, ask-lifts only, price-aware sizing)                  #
# --------------------------------------------------------------------------- #

def run_dynamic_asklift(entries, fraction, trade_data, bankroll_init):
    ents = sorted(entries, key=lambda x: x["entry_ts"])
    bankroll = bankroll_init
    out = []
    for e in ents:
        kf_qmax = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_kelly_at_qmax = kf_qmax * fraction * bankroll
        if wager_kelly_at_qmax < 0.01:
            continue

        g = asklift_prints(trade_data, e["game_id"], e["side"], e["entry_ts"])
        if g.empty: continue
        g = g[(g["our_price"] <= e["entry_px"]) & (g["our_price"] > 0) & (g["our_price"] < 1)]
        if g.empty: continue
        # g is already time-sorted from asklift_prints; keep that ordering.

        already_dollars = 0.0
        already_count   = 0.0
        n_pause_events  = 0
        n_resume_events = 0
        was_paused = False
        for _, t in g.iterrows():
            p = float(t["our_price"])
            kf_p = (e["p_side"] - p) / (1 - p)
            if kf_p <= 0:
                continue
            wager_target_p = fraction * kf_p * bankroll
            headroom = wager_target_p - already_dollars
            if headroom <= 0:
                n_pause_events += 1; was_paused = True; continue
            if was_paused:
                n_resume_events += 1; was_paused = False
            trade_cost = float(t["count"]) * p
            take_cost  = min(trade_cost, headroom)
            take_n     = take_cost / p
            already_dollars += take_cost
            already_count   += take_n

        if already_count == 0: continue
        vwap = already_dollars / already_count
        n_actual = already_count
        wager_actual = already_dollars
        fee = kalshi_taker_fee(n_actual, vwap)
        won_payoff = 1.0 if (e["side"] == "YES" and e["home_win"] == 1) \
                     or (e["side"] == "NO" and e["home_win"] == 0) else 0.0
        pnl = n_actual * won_payoff - wager_actual - fee
        bankroll_before = bankroll
        bankroll += pnl

        n_ideal_at_qmax = wager_kelly_at_qmax / e["entry_px"]
        out.append({
            **e, "kelly_f_qmax": kf_qmax, "wager_kelly_at_qmax": wager_kelly_at_qmax,
            "wager": wager_actual, "n_ideal_at_qmax": n_ideal_at_qmax,
            "n_contracts": n_actual,
            "fill_ratio_vs_qmax": n_actual / n_ideal_at_qmax if n_ideal_at_qmax > 0 else 0,
            "entry_px_actual": vwap,
            "n_pause_events": int(n_pause_events),
            "n_resume_events": int(n_resume_events),
            "fee": fee, "pnl": pnl, "won": int(pnl > 0),
            "bankroll": bankroll, "bankroll_before": bankroll_before,
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
        "engine": label,
        "trades": int(len(tdf)),
        "hit_rate": float(tdf["won"].mean()),
        "total_return": (fb - bankroll_init) / bankroll_init,
        "final_bankroll": fb,
        "max_drawdown": dd,
        "mean_log_return": mean_lr,
        "std_log_return": std_lr,
        "sharpe_per_trade": mean_lr / std_lr if std_lr > 0 else float("nan"),
        "mean_wager": float(tdf["wager"].mean()),
        "median_wager": float(tdf["wager"].median()),
        "max_wager": float(tdf["wager"].max()),
        "mean_wager_pct": float(tdf["wager_pct"].mean()),
        "median_wager_pct": float(tdf["wager_pct"].median()),
        "max_wager_pct": float(tdf["wager_pct"].max()),
        "mean_entry_px_actual": float(tdf["entry_px_actual"].mean()),
        "mean_fill_ratio": float(tdf["fill_ratio_vs_qmax"].mean())
            if "fill_ratio_vs_qmax" in tdf.columns else float("nan"),
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
    print(f"  wt size: {len(wt):,} trades  |  ask-lifts on yes: "
          f"{(wt['taker_side']=='yes').sum():,}  on no: {(wt['taker_side']=='no').sum():,}")

    ents = collect_entries_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
        cutoff_hours=TIMING_CUTOFF_H,
    )
    print(f"Qualifying entries (T-{int(TIMING_CUTOFF_H)}h gate): {len(ents)}")

    print("Running STATIC sweep (cheapest-first, ask-lifts only) ...")
    static = run_static_asklift(ents, KELLY_FRACTION, wt, BANKROLL_REAL)
    if len(static): static = add_trade_returns(static)

    print("Running DYNAMIC sweep (time-ordered, ask-lifts only, price-aware) ...")
    dyn = run_dynamic_asklift(ents, KELLY_FRACTION, wt, BANKROLL_REAL)
    if len(dyn): dyn = add_trade_returns(dyn)

    static_s = evaluate("Static (ask-lifts only)",  static, BANKROLL_REAL)
    dyn_s    = evaluate("Dynamic (ask-lifts only)", dyn,    BANKROLL_REAL)
    summary  = pd.DataFrame([static_s, dyn_s])
    save_table(summary, "trade_dynamic_kelly_v2_summary")
    print()
    print(summary.to_string(index=False))

    if len(static) and len(dyn):
        s = static[["game_id","entry_ts","side","edge","entry_px","p_side",
                    "wager","n_contracts","entry_px_actual","won","pnl"]].rename(
            columns=lambda c: c if c == "game_id" else c + "_static"
        )
        d = dyn[["game_id","wager","n_contracts","entry_px_actual","won","pnl",
                 "n_pause_events","n_resume_events","fill_ratio_vs_qmax"]].rename(
            columns=lambda c: c if c == "game_id" else c + "_dynamic"
        )
        paired = s.merge(d, on="game_id", how="outer")
        save_table(paired, "trade_dynamic_kelly_v2_per_trade")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for label, tdf, color in [
        ("Static (ask-lifts)",  static, "#3498db"),
        ("Dynamic (ask-lifts)", dyn,    "#e74c3c"),
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
    ax.set_title("Dynamic vs. Static Half-Kelly Sweep (Ask-Lifts Only)",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "trade_dynamic_kelly_v2_equity_curves"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bins_pct = np.linspace(0, 25, 51)
    max_w = max(static["wager"].max() if len(static) else 0,
                dyn["wager"].max() if len(dyn) else 0)
    bins_dol = np.linspace(0, max_w * 1.05, 41)
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
    fig.suptitle("Position-Size Distributions (Ask-Lifts Only)",
                 fontweight="bold", fontsize=13, y=1.02)
    plt.tight_layout()
    save_fig(fig, "trade_dynamic_kelly_v2_position_sizes"); plt.close(fig)


if __name__ == "__main__":
    main()
