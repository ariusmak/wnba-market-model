"""Phase-conditioned exposure-cap policy sweep WITH the v1 liquidity cap.

Builds on `run_trade_cap_policy_sweep.py` by layering the user's v1 liquidity
cap rule on top of every portfolio-cap policy. Same 7 policies, same T-8h
timing gate, same Kalshi 2025 holdout. The only change is how much of the
qualifying tape the bot is allowed to consume per trade.

v1 liquidity-cap spec (parameters at module top)
------------------------------------------------
    visible_depth_cap         = 0.25 * visible_cost_at_or_below_qmax_now
    qualifying_volume_cap     = 0.15 * traded_cost_at_or_below_qmax_last_3h_ex_self
    cold_start_cap            = min(0.01 * bankroll,
                                    0.15 * visible_cost_at_or_below_qmax_now)
    effective_volume_cap      = max(qualifying_volume_cap, cold_start_cap)
    rolling_liquidity_cap     = min(visible_depth_cap, effective_volume_cap)
    cumulative_cap            = 0.30 * traded_cost_at_or_below_qmax_since_first_qualification_ex_self

    allowed_dollars = min(remaining_desired,
                           rolling_liquidity_cap,
                           cumulative_cap − already_filled)

Order of operations on each entry:
    half_kelly_size  = kelly_f * 0.5 * bankroll
    after_portfolio  = min(half_kelly_size, policy_cap * bankroll)
    after_liq        = min(after_portfolio, rolling_liq_cap, cumulative_cap)
    actual_fill      = sweep(qualifying_tape, after_liq)

Approximations
--------------
  - Our historical data has trade tape (`wt`) but not order-book snapshots,
    so `visible_cost_at_or_below_qmax_now` is approximated as the qualifying-
    price tape volume in the last `VISIBLE_WINDOW_MINUTES` minutes (default 15).
    This proxies for "instantaneous executable depth at q_max."
  - Our entry simulation is single-cadence per market (one entry at the first
    qualifying snapshot). The cumulative cap is computed since the *first
    qualifying snapshot* (from `collect_all_snapshots`), which is generally
    earlier than `entry_ts` because the model can re-qualify after gaps.
  - All trades in the historical tape are ex-self by construction (the bot
    was never in the live order flow), so the "_ex_self" tags are automatic.

Outputs (in organized/outputs/):
    trade_cap_policy_liq_summary.csv
    trade_cap_policy_liq_per_trade.csv
    trade_cap_policy_liq_period_contribution.csv
    trade_cap_policy_liq_equity_curves.png
    trade_cap_policy_liq_metrics_grid.png
    trade_cap_policy_liq_first10_paths.png
    trade_cap_policy_liq_binding_distribution.png
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"
TIMING_CUTOFF_H = 8.0

# v1 liquidity cap parameters
MAX_VISIBLE_DEPTH_PARTICIPATION             = 0.25
RECENT_VOLUME_WINDOW_HOURS                  = 3.0
MAX_RECENT_QUALIFYING_VOLUME_PARTICIPATION  = 0.15
COLD_START_BANKROLL_CAP                     = 0.01
COLD_START_VISIBLE_DEPTH_PARTICIPATION      = 0.15
MAX_CUMULATIVE_QUALIFYING_VOLUME_SHARE      = 0.30
VISIBLE_WINDOW_MINUTES                      = 15   # tape proxy for "now" depth


POLICIES = {
    "static_12":             "Static 12%",
    "static_15":             "Static 15%",
    "static_20":             "Static 20%",
    "uncapped":              "Uncapped",
    "phase_double_then_12":  "20% until 2x bankroll → 12%",
    "phase_10k_then_12":     "20% until $10k → 12%",
    "edge_conditioned":      "Edge-conditioned 12/15/20",
}
POLICY_COLORS = {
    "static_12":             "#3498db",
    "static_15":             "#2ecc71",
    "static_20":             "#e67e22",
    "uncapped":              "#7f8c8d",
    "phase_double_then_12":  "#9b59b6",
    "phase_10k_then_12":     "#f1c40f",
    "edge_conditioned":      "#e74c3c",
}


def policy_cap(name: str, *, bankroll: float, edge: float,
               starting_bankroll: float) -> float | None:
    if name == "static_12":   return 0.12
    if name == "static_15":   return 0.15
    if name == "static_20":   return 0.20
    if name == "uncapped":    return None
    if name == "phase_double_then_12":
        return 0.20 if bankroll < 2.0 * starting_bankroll else 0.12
    if name == "phase_10k_then_12":
        return 0.20 if bankroll < 10000.0 else 0.12
    if name == "edge_conditioned":
        if edge >= 0.10:  return 0.20
        if edge >= 0.075: return 0.15
        return 0.12
    raise ValueError(f"unknown policy: {name}")


# --------------------------------------------------------------------------- #
# Cutoff-aware entry collector + first-qualification map                      #
# --------------------------------------------------------------------------- #

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


def first_qualification_map(ticker_info, pretip):
    """Return dict {game_id: first_qual_ts}.

    Uses the same edge / norm thresholds as the live entry, but searches the
    *full* candle window (no T-8h cutoff) because cumulative liquidity should
    be measured from the model's first opinion on the market.
    """
    snaps = collect_all_snapshots(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
    )
    if not snaps:
        return {}
    df = pd.DataFrame(snaps)
    return df.sort_values("entry_ts").groupby("game_id")["entry_ts"].first().to_dict()


# --------------------------------------------------------------------------- #
# v1 liquidity cap                                                            #
# --------------------------------------------------------------------------- #

def liquidity_caps(game_trades: pd.DataFrame, *, side: str, entry_px: float,
                    entry_ts: pd.Timestamp, first_qual_ts: pd.Timestamp,
                    bankroll: float):
    """Return (rolling_cap, cumulative_cap) in dollars, plus diagnostic pools."""
    if game_trades.empty:
        return 0.0, 0.0, {"visible_pool": 0.0, "recent_pool": 0.0,
                          "cumulative_pool": 0.0}
    g = game_trades.copy()
    g["our_price"] = g["yes_price"] if side == "YES" else g["no_price"]
    qual = g[g["our_price"] <= entry_px].copy()
    qual["cost"] = qual["count"] * qual["our_price"]

    visible_lo = entry_ts - pd.Timedelta(minutes=VISIBLE_WINDOW_MINUTES)
    recent_lo  = entry_ts - pd.Timedelta(hours=RECENT_VOLUME_WINDOW_HOURS)

    visible_pool    = float(qual.loc[qual["ts"] >= visible_lo, "cost"].sum())
    recent_pool     = float(qual.loc[qual["ts"] >= recent_lo,  "cost"].sum())
    cumulative_pool = float(qual.loc[qual["ts"] >= first_qual_ts, "cost"].sum())

    visible_cap   = MAX_VISIBLE_DEPTH_PARTICIPATION * visible_pool
    qual_vol_cap  = MAX_RECENT_QUALIFYING_VOLUME_PARTICIPATION * recent_pool
    cold_cap      = min(COLD_START_BANKROLL_CAP * bankroll,
                        COLD_START_VISIBLE_DEPTH_PARTICIPATION * visible_pool)
    effective_vol = max(qual_vol_cap, cold_cap)
    rolling_cap   = min(visible_cap, effective_vol)

    cumulative_cap = MAX_CUMULATIVE_QUALIFYING_VOLUME_SHARE * cumulative_pool

    return rolling_cap, cumulative_cap, {
        "visible_pool":   visible_pool,
        "recent_pool":    recent_pool,
        "cumulative_pool":cumulative_pool,
        "visible_cap":    visible_cap,
        "qual_vol_cap":   qual_vol_cap,
        "cold_cap":       cold_cap,
        "rolling_cap":    rolling_cap,
        "cumulative_cap": cumulative_cap,
    }


# --------------------------------------------------------------------------- #
# Sweep with portfolio + liquidity caps                                       #
# --------------------------------------------------------------------------- #

def run_sweep_with_liq(entries, fraction, trade_data, bankroll_init,
                       policy_name: str, starting_bankroll: float,
                       first_qual: dict):
    ents = sorted(entries, key=lambda x: x["entry_ts"])
    bankroll = bankroll_init
    out = []
    for e in ents:
        kf = max((e["p_side"] - e["entry_px"]) / (1 - e["entry_px"]), 0)
        wager_kelly = kf * fraction * bankroll
        if wager_kelly < 0.01:
            continue

        cap_pct = policy_cap(
            policy_name, bankroll=bankroll, edge=e["edge"],
            starting_bankroll=starting_bankroll,
        )
        portfolio_cap_dollars = (cap_pct * bankroll) if cap_pct is not None \
                                else float("inf")

        gtr = trade_data[trade_data["game_id"] == e["game_id"]].copy()
        if gtr.empty:
            continue

        first_qual_ts = first_qual.get(e["game_id"], e["entry_ts"])
        rolling_cap, cumulative_cap, diag = liquidity_caps(
            gtr, side=e["side"], entry_px=e["entry_px"],
            entry_ts=e["entry_ts"], first_qual_ts=first_qual_ts,
            bankroll=bankroll,
        )

        # Identify which constraint binds. Order of competition:
        candidates = {
            "kelly":         wager_kelly,
            "portfolio":     portfolio_cap_dollars,
            "rolling_liq":   rolling_cap,
            "cumulative_liq":cumulative_cap,
        }
        # We need the smallest, but with a tie-break that respects order of
        # operations (kelly < portfolio < rolling < cumulative on equality).
        names_in_order = ["kelly", "portfolio", "rolling_liq", "cumulative_liq"]
        binding = min(names_in_order, key=lambda k: candidates[k])
        allowed_dollars = candidates[binding]
        if allowed_dollars <= 0.01:
            continue

        # Sweep qualifying tape up to allowed_dollars.
        gtr["our_price"] = gtr["yes_price"] if e["side"] == "YES" else gtr["no_price"]
        qual = gtr[gtr["our_price"] <= e["entry_px"]].sort_values("our_price")
        if qual.empty:
            continue
        cost, filled = 0.0, 0.0
        for _, t in qual.iterrows():
            t_cost = t["count"] * t["our_price"]
            take_cost = min(t_cost, allowed_dollars - cost)
            if take_cost <= 0:
                break
            take_n = take_cost / t["our_price"]
            cost += take_cost
            filled += take_n
            if cost >= allowed_dollars - 1e-9:
                break
        if filled == 0:
            continue

        # If sweep filled less than allowed, the qualifying tape was the binder
        if cost < allowed_dollars - 1e-6:
            binding = "sweep_short"

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
            **e, "kelly_f": kf,
            "cap_pct_used":           cap_pct if cap_pct is not None else float("nan"),
            "wager_kelly":            wager_kelly,
            "wager_post_portfolio":   min(wager_kelly, portfolio_cap_dollars),
            "rolling_cap":            rolling_cap,
            "cumulative_cap":         cumulative_cap,
            "allowed_dollars":        allowed_dollars,
            "wager":                  wager_actual,
            "n_contracts":            n_actual,
            "fill_pct":               (cost / allowed_dollars) if allowed_dollars > 0 else 0,
            "entry_px_actual":        vwap,
            "binding":                binding,
            "visible_pool":           diag["visible_pool"],
            "recent_pool":            diag["recent_pool"],
            "cumulative_pool":        diag["cumulative_pool"],
            "fee":                    fee,
            "pnl":                    pnl,
            "won":                    int(pnl > 0),
            "bankroll":               bankroll,
            "bankroll_before":        bankroll_before,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def worst_first_n_drawdown(tdf: pd.DataFrame, n: int = 10) -> float:
    sub = tdf.head(n)
    if len(sub) == 0: return 0.0
    series = sub["bankroll"].values
    peak = np.maximum.accumulate(series)
    return float((peak - series).max())


def max_consec_losses(tdf: pd.DataFrame, n: int = 10) -> int:
    sub = tdf.head(n)
    if len(sub) == 0: return 0
    losses = (sub["won"] == 0).values.astype(int)
    best = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


def evaluate(name: str, tdf: pd.DataFrame, bankroll_init: float) -> dict:
    if len(tdf) == 0:
        return {"policy": name, "label": POLICIES[name], "trades": 0}
    tdf = tdf.copy()
    tdf["log_ret"] = np.log(tdf["bankroll"] / tdf["bankroll_before"])
    fb = float(tdf["bankroll"].iloc[-1])
    dd = float((tdf["bankroll"].cummax() - tdf["bankroll"]).max())
    mean_lr = float(tdf["log_ret"].mean())
    std_lr  = float(tdf["log_ret"].std())
    sharpe  = mean_lr / std_lr if std_lr > 0 else float("nan")
    bind    = tdf["binding"].value_counts(normalize=True)
    return {
        "policy":              name,
        "label":               POLICIES[name],
        "trades":              int(len(tdf)),
        "hit_rate":            float(tdf["won"].mean()),
        "total_return":        (fb - bankroll_init) / bankroll_init,
        "final_bankroll":      fb,
        "max_drawdown":        dd,
        "worst_first10_dd":    worst_first_n_drawdown(tdf, 10),
        "max_consec_losses_first10": max_consec_losses(tdf, 10),
        "mean_log_return":     mean_lr,
        "std_log_return":      std_lr,
        "sharpe_per_trade":    sharpe,
        "mean_fill_rate":      float(tdf["fill_pct"].mean()),
        "mean_realized_wager_pct": float((tdf["wager"] / tdf["bankroll_before"]).mean()),
        "share_kelly_binding":      float(bind.get("kelly", 0.0)),
        "share_portfolio_binding":  float(bind.get("portfolio", 0.0)),
        "share_rolling_binding":    float(bind.get("rolling_liq", 0.0)),
        "share_cumulative_binding": float(bind.get("cumulative_liq", 0.0)),
        "share_sweepshort_binding": float(bind.get("sweep_short", 0.0)),
    }


def by_period(tdf: pd.DataFrame) -> pd.DataFrame:
    if len(tdf) == 0: return pd.DataFrame()
    df = tdf.copy()
    df["log_ret"] = np.log(df["bankroll"] / df["bankroll_before"])
    df["month"] = pd.to_datetime(df["game_ts"]).dt.tz_convert(None).dt.to_period("M")
    out = (
        df.groupby("month").agg(
            n_trades=("pnl", "count"),
            sum_log_ret=("log_ret", "sum"),
            hit_rate=("won", "mean"),
        ).reset_index()
    )
    total = out["sum_log_ret"].sum()
    out["share_of_total"] = np.where(total != 0, out["sum_log_ret"] / total, np.nan)
    out["wealth_multiplier"] = np.exp(out["sum_log_ret"])
    out["month"] = out["month"].astype(str)
    return out


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #

def plot_equity_curves(trade_frames: dict, save_name: str) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    for name, tdf in trade_frames.items():
        if len(tdf) == 0: continue
        eq = tdf.copy()
        eq["display_bankroll"] = BANKROLL_REAL + eq["pnl"].cumsum()
        eq = eq.sort_values("game_ts")
        ret = (eq["display_bankroll"].iloc[-1] - BANKROLL_REAL) / BANKROLL_REAL
        ax.step(eq["game_ts"], eq["display_bankroll"], where="post",
                color=POLICY_COLORS[name], linewidth=1.7,
                label=f"{POLICIES[name]} ({ret:+.0%})")
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6,
               label=f"Starting bankroll (${BANKROLL_REAL:,.0f})")
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title("Cap-Policy Equity Curves with v1 Liquidity Cap "
                 f"(T-{int(TIMING_CUTOFF_H)}h gate, ${BANKROLL_REAL:,.0f})",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, save_name); plt.close(fig)


def plot_metrics_grid(summary: pd.DataFrame, save_name: str) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "figure.dpi": 120})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    pmap = list(summary["policy"]); x = np.arange(len(pmap))
    colors = [POLICY_COLORS[p] for p in pmap]
    labels = [POLICIES[p] for p in pmap]

    ax = axes[0, 0]
    ax.bar(x, summary["total_return"] * 100, color=colors, alpha=0.85, width=0.7)
    for xi, v in zip(x, summary["total_return"]):
        ax.text(xi, v * 100 + 5, f"{v*100:+.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Total return (%)"); ax.set_title("Total Return", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, summary["sharpe_per_trade"], color=colors, alpha=0.85, width=0.7)
    for xi, v in zip(x, summary["sharpe_per_trade"]):
        ax.text(xi, v + 0.005, f"{v:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Per-trade Sharpe"); ax.set_title("Per-Trade Sharpe", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x, summary["max_drawdown"], color=colors, alpha=0.85, width=0.7)
    for xi, v in zip(x, summary["max_drawdown"]):
        ax.text(xi, v + max(50, v * 0.02), f"${v:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Max drawdown ($)"); ax.set_title("Max Drawdown", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x, summary["worst_first10_dd"], color=colors, alpha=0.85, width=0.7)
    for xi, v, k in zip(x, summary["worst_first10_dd"],
                          summary["max_consec_losses_first10"]):
        ax.text(xi, v + max(5, v * 0.02), f"${v:,.0f}\n{int(k)} loss streak",
                ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Worst first-10-trade drawdown ($)")
    ax.set_title("First-10-Trade Stress", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Cap-Policy Comparison with v1 Liquidity Cap — "
        f"T-{int(TIMING_CUTOFF_H)}h Gate, Half-Kelly, ${BANKROLL_REAL:,.0f}",
        fontweight="bold", fontsize=13, y=1.0,
    )
    plt.tight_layout()
    save_fig(fig, save_name); plt.close(fig)


def plot_first10_paths(trade_frames: dict, save_name: str) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, tdf in trade_frames.items():
        if len(tdf) < 1: continue
        sub = tdf.head(10).reset_index(drop=True)
        path = np.concatenate([[BANKROLL_REAL], sub["bankroll"].values])
        ax.plot(range(len(path)), path, marker="o",
                color=POLICY_COLORS[name], linewidth=1.6, markersize=5,
                label=POLICIES[name])
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Trade index (0 = season start)")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("First-10-Trade Bankroll Paths (with v1 Liquidity Cap)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="best"); ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, save_name); plt.close(fig)


def plot_binding(summary: pd.DataFrame, save_name: str) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(13, 5.2))
    pmap = list(summary["policy"]); x = np.arange(len(pmap))
    labels = [POLICIES[p] for p in pmap]
    palette = {
        "kelly":         ("#2ecc71", "Half-Kelly target binds"),
        "portfolio":     ("#e67e22", "Portfolio cap binds"),
        "rolling_liq":   ("#3498db", "Rolling liquidity cap binds"),
        "cumulative_liq":("#9b59b6", "Cumulative liquidity cap binds"),
        "sweep_short":   ("#7f8c8d", "Sweep tape exhausted"),
    }
    bottoms = np.zeros(len(pmap))
    for kind, (color, lbl) in palette.items():
        col = "share_" + ("kelly" if kind == "kelly"
                            else "portfolio" if kind == "portfolio"
                            else "rolling" if kind == "rolling_liq"
                            else "cumulative" if kind == "cumulative_liq"
                            else "sweepshort") + "_binding"
        vals = summary[col].values
        ax.bar(x, vals * 100, bottom=bottoms * 100,
               color=color, alpha=0.85, edgecolor="white", label=lbl)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 0.06:
                ax.text(xi, (b + v / 2) * 100, f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
        bottoms = bottoms + vals
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=18, ha="right")
    ax.set_ylabel("Share of trades (%)")
    ax.set_title("Binding Constraint Distribution per Trade (v1 Liquidity Cap)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
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
    ticker_info, pretip, wt = idx["ticker_info"], idx["pretip"], idx["wt"]

    print(f"Building first-qualification map ...")
    first_qual = first_qualification_map(ticker_info, pretip)
    print(f"  first-qual entries: {len(first_qual)}")

    print(f"Collecting entries with T-{int(TIMING_CUTOFF_H)}h cutoff ...")
    ents = collect_entries_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
        cutoff_hours=TIMING_CUTOFF_H,
    )
    print(f"  qualifying entries: {len(ents)}")

    summaries, trade_frames, period_frames = [], {}, []
    for name in POLICIES:
        tdf = run_sweep_with_liq(
            ents, KELLY_FRACTION, wt, BANKROLL_REAL,
            policy_name=name, starting_bankroll=BANKROLL_REAL,
            first_qual=first_qual,
        )
        trade_frames[name] = tdf
        s = evaluate(name, tdf, BANKROLL_REAL)
        summaries.append(s)
        ptbl = by_period(tdf); ptbl.insert(0, "policy", name)
        period_frames.append(ptbl)
        print(f"  {name:<26s}  trades={s['trades']}  ret={s['total_return']:+.2%}  "
              f"DD=${s['max_drawdown']:,.0f}  sharpe={s['sharpe_per_trade']:+.3f}  "
              f"first10-DD=${s['worst_first10_dd']:,.0f}  "
              f"first10-LL={s['max_consec_losses_first10']}  "
              f"realized-stake={s['mean_realized_wager_pct']*100:.2f}%")

    summary_df = pd.DataFrame(summaries)
    save_table(summary_df, "trade_cap_policy_liq_summary")

    per_trade = pd.concat([
        tdf.assign(policy=name)[[
            "policy","game_id","game_ts","entry_ts","side","edge",
            "kelly_f","cap_pct_used","wager_kelly","wager_post_portfolio",
            "rolling_cap","cumulative_cap","allowed_dollars","wager",
            "n_contracts","fill_pct","entry_px_actual","binding",
            "visible_pool","recent_pool","cumulative_pool",
            "fee","pnl","won","bankroll","bankroll_before",
        ]]
        for name, tdf in trade_frames.items() if len(tdf) > 0
    ], ignore_index=True)
    save_table(per_trade, "trade_cap_policy_liq_per_trade")

    period_df = pd.concat(period_frames, ignore_index=True)
    save_table(period_df, "trade_cap_policy_liq_period_contribution")

    plot_equity_curves(trade_frames,    "trade_cap_policy_liq_equity_curves")
    plot_metrics_grid(summary_df,        "trade_cap_policy_liq_metrics_grid")
    plot_first10_paths(trade_frames,     "trade_cap_policy_liq_first10_paths")
    plot_binding(summary_df,             "trade_cap_policy_liq_binding_distribution")

    print()
    print("Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
