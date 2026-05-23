"""Phase-conditioned exposure-cap policy sweep with T-8h timing gate.

Runs seven candidate cap policies on the live half-Kelly $5k Kalshi sweep
backtest. All policies share the same edge / norm thresholds, the same
sweep liquidity model, and the same timing gate (no new entries after
T-8h before tipoff). Single-entry-per-market is enforced by
`collect_entries`, so "no adds after T-4h" is automatic.

Policies tested
---------------
A. Static 12%
B. Static 15%
C. Static 20%
D. Uncapped (half-Kelly only, no portfolio cap)
E. Phase: 20% until bankroll doubles ($10,000), then 12%
F. Phase: 20% until bankroll reaches $10,000 (same point given the start),
   alternative wording — kept as separate policy for sanity (identical here)
G. Edge-conditioned: edge >= 0.10 -> 20%; edge >= 0.075 -> 15%; else 12%

(Policies E and F are identical at $5,000 starting bankroll because doubling
hits exactly $10k. Both are reported anyway because the user asked for both.)

Outputs (in organized/outputs/):
    trade_cap_policy_summary.csv
    trade_cap_policy_per_trade.csv
    trade_cap_policy_period_contribution.csv
    trade_cap_policy_equity_curves.png
    trade_cap_policy_metrics_grid.png
    trade_cap_policy_first10_paths.png
    trade_cap_policy_binding_distribution.png
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
from trading import LABEL  # noqa: E402
from final_model import load_year, LABEL_COL, kalshi_taker_fee  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"
TIMING_CUTOFF_H = 8.0  # T-8h timing gate


# --------------------------------------------------------------------------- #
# Policy definitions                                                          #
# --------------------------------------------------------------------------- #

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
        if edge >= 0.10: return 0.20
        if edge >= 0.075: return 0.15
        return 0.12
    raise ValueError(f"unknown policy: {name}")


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


# --------------------------------------------------------------------------- #
# Cutoff-aware entry collector (T-Nh timing gate)                             #
# --------------------------------------------------------------------------- #

def collect_entries_cutoff(ticker_info, pretip, model_col, edge_min,
                            norm_edge_min, entry_window, cutoff_hours,
                            label_col=LABEL):
    key_map = {"half_life": "entry_start_half",
               "two_thirds_life": "entry_start_twothirds"}
    key = key_map[entry_window]
    out = []
    for tkr, info in ticker_info.items():
        if tkr not in pretip:
            continue
        cutoff_ts = info["game_ts"] - pd.Timedelta(hours=cutoff_hours) \
                    if cutoff_hours and cutoff_hours > 0 else info["game_ts"]
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
# Sweep engine with policy cap                                                #
# --------------------------------------------------------------------------- #

def run_sweep_with_policy(entries, fraction, trade_data, bankroll_init,
                           policy_name: str, starting_bankroll: float):
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
        cap_value = (cap_pct * bankroll) if cap_pct is not None else float("inf")
        cap_binding = cap_pct is not None and wager_kelly > cap_value
        wager_post_cap = min(wager_kelly, cap_value)

        n_post_cap = wager_post_cap / e["entry_px"]
        gtr = trade_data[trade_data["game_id"] == e["game_id"]].copy()
        if gtr.empty: continue
        gtr["our_price"] = gtr["yes_price"] if e["side"] == "YES" else gtr["no_price"]
        qual = gtr[gtr["our_price"] <= e["entry_px"]].sort_values("our_price")
        if qual.empty: continue
        filled, cost = 0.0, 0.0
        for _, t in qual.iterrows():
            take = min(t["count"], n_post_cap - filled)
            filled += take
            cost += take * t["our_price"]
            if filled >= n_post_cap: break
        if filled == 0: continue

        liq_binding = filled < n_post_cap - 1e-9
        # Determine binding constraint: order of operations is kelly -> cap -> liq.
        if liq_binding:
            binding = "liquidity"
        elif cap_binding:
            binding = "cap"
        else:
            binding = "kelly"

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
            "cap_pct_used":   cap_pct if cap_pct is not None else float("nan"),
            "wager_kelly":    wager_kelly,
            "wager_post_cap": wager_post_cap,
            "wager":          wager_actual,
            "n_post_cap":     n_post_cap,
            "n_contracts":    n_actual,
            "fill_pct":       n_actual / n_post_cap if n_post_cap > 0 else 0,
            "entry_px_actual":vwap,
            "binding":        binding,
            "fee":            fee,
            "pnl":            pnl,
            "won":            int(pnl > 0),
            "bankroll":       bankroll,
            "bankroll_before":bankroll_before,
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def worst_first_n_drawdown(tdf: pd.DataFrame, n: int = 10) -> float:
    sub = tdf.head(n)
    if len(sub) == 0:
        return 0.0
    series = sub["bankroll"].values
    running_peak = np.maximum.accumulate(series)
    return float((running_peak - series).max())


def max_consec_losses(tdf: pd.DataFrame, n: int = 10) -> int:
    sub = tdf.head(n)
    if len(sub) == 0:
        return 0
    losses = (sub["won"] == 0).values.astype(int)
    best = cur = 0
    for x in losses:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


def evaluate(name: str, tdf: pd.DataFrame, bankroll_init: float) -> dict:
    if len(tdf) == 0:
        return {"policy": name, "trades": 0}
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
        "share_kelly_binding":     float(bind.get("kelly", 0.0)),
        "share_cap_binding":       float(bind.get("cap", 0.0)),
        "share_liq_binding":       float(bind.get("liquidity", 0.0)),
    }


def by_period(tdf: pd.DataFrame) -> pd.DataFrame:
    """Per-month sum of log returns, share of total."""
    if len(tdf) == 0:
        return pd.DataFrame()
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
    ax.set_title("Cap-Policy Equity Curves — T-8h Gate, Half-Kelly Sweep, $5k Bankroll",
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

    pmap = list(summary["policy"])
    x = np.arange(len(pmap))
    colors = [POLICY_COLORS[p] for p in pmap]
    labels = [POLICIES[p] for p in pmap]

    ax = axes[0, 0]
    ax.bar(x, summary["total_return"] * 100, color=colors, alpha=0.85, width=0.7)
    for xi, v in zip(x, summary["total_return"]):
        ax.text(xi, v * 100 + 30, f"{v*100:+.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Total return (%)")
    ax.set_title("Total Return", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, summary["sharpe_per_trade"], color=colors, alpha=0.85, width=0.7)
    for xi, v in zip(x, summary["sharpe_per_trade"]):
        ax.text(xi, v + 0.005, f"{v:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Per-trade Sharpe")
    ax.set_title("Per-Trade Sharpe", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x, summary["max_drawdown"], color=colors, alpha=0.85, width=0.7)
    for xi, v in zip(x, summary["max_drawdown"]):
        ax.text(xi, v + 700, f"${v:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Max drawdown ($)")
    ax.set_title("Max Drawdown", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    ax.bar(x, summary["worst_first10_dd"], color=colors, alpha=0.85, width=0.7)
    for xi, v, k in zip(x, summary["worst_first10_dd"],
                         summary["max_consec_losses_first10"]):
        ax.text(xi, v + 30, f"${v:,.0f}\n{int(k)} loss streak",
                ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Worst first-10-trade drawdown ($)")
    ax.set_title("First-10-Trade Stress", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cap-Policy Comparison — T-8h Gate, Half-Kelly Sweep, $5k Bankroll",
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
    ax.set_xlabel("Trade index (0 = start of season)")
    ax.set_ylabel("Bankroll ($)")
    ax.set_title("First-10-Trade Bankroll Paths", fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, save_name); plt.close(fig)


def plot_binding_distribution(summary: pd.DataFrame, save_name: str) -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})
    fig, ax = plt.subplots(figsize=(12, 4.8))
    pmap = list(summary["policy"])
    x = np.arange(len(pmap))
    labels = [POLICIES[p] for p in pmap]

    bottoms = np.zeros(len(pmap))
    palette = {"kelly": "#2ecc71", "cap": "#e67e22", "liquidity": "#3498db"}
    legend_label = {"kelly": "Half-Kelly target binds (no cap, full fill)",
                    "cap":   "Portfolio cap binds",
                    "liquidity": "Liquidity (sweep) binds"}
    for kind in ["kelly", "cap", "liquidity"]:
        col = f"share_{'liq' if kind == 'liquidity' else kind}_binding"
        vals = summary[col].values
        ax.bar(x, vals * 100, bottom=bottoms * 100,
               color=palette[kind], alpha=0.85, label=legend_label[kind],
               edgecolor="white")
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 0.05:
                ax.text(xi, (b + v / 2) * 100, f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold")
        bottoms = bottoms + vals
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=18, ha="right")
    ax.set_ylabel("Share of trades (%)")
    ax.set_title("Binding-Constraint Distribution per Trade",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
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

    print(f"Collecting entries with T-{TIMING_CUTOFF_H}h cutoff ...")
    ents = collect_entries_cutoff(
        ticker_info, pretip, "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
        cutoff_hours=TIMING_CUTOFF_H,
    )
    print(f"  qualifying entries: {len(ents)}")

    summaries, trade_frames, period_frames = [], {}, []
    for name in POLICIES:
        tdf = run_sweep_with_policy(
            ents, KELLY_FRACTION, wt, BANKROLL_REAL,
            policy_name=name, starting_bankroll=BANKROLL_REAL,
        )
        trade_frames[name] = tdf
        summaries.append(evaluate(name, tdf, BANKROLL_REAL))
        ptbl = by_period(tdf); ptbl.insert(0, "policy", name)
        period_frames.append(ptbl)
        s = summaries[-1]
        print(f"  {name:<26s}  trades={s['trades']}  ret={s['total_return']:+.2%}  "
              f"DD=${s['max_drawdown']:,.0f}  sharpe={s['sharpe_per_trade']:+.3f}  "
              f"first10-DD=${s['worst_first10_dd']:,.0f}  "
              f"first10-LL={s['max_consec_losses_first10']}  "
              f"realized-stake={s['mean_realized_wager_pct']*100:.2f}%")

    summary_df = pd.DataFrame(summaries)
    save_table(summary_df, "trade_cap_policy_summary")

    per_trade = pd.concat([
        tdf.assign(policy=name)[[
            "policy","game_id","game_ts","entry_ts","side","edge",
            "kelly_f","cap_pct_used","wager_kelly","wager_post_cap","wager",
            "n_post_cap","n_contracts","fill_pct","entry_px_actual",
            "binding","fee","pnl","won","bankroll","bankroll_before",
        ]]
        for name, tdf in trade_frames.items() if len(tdf) > 0
    ], ignore_index=True)
    save_table(per_trade, "trade_cap_policy_per_trade")

    period_df = pd.concat(period_frames, ignore_index=True)
    save_table(period_df, "trade_cap_policy_period_contribution")

    plot_equity_curves(trade_frames, "trade_cap_policy_equity_curves")
    plot_metrics_grid(summary_df, "trade_cap_policy_metrics_grid")
    plot_first10_paths(trade_frames, "trade_cap_policy_first10_paths")
    plot_binding_distribution(summary_df, "trade_cap_policy_binding_distribution")

    print()
    print("Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
