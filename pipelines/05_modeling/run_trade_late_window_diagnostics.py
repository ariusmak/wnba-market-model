"""Three diagnostic studies of late-window trade pathology.

All three run on the live half-Kelly best config (edge_min = 0.05,
norm_min = 0.25) over the 2025 Kalshi holdout. Trades, fills, and log returns
come from the same sweep-execution simulation already used in the cutoff
sweep so numbers reconcile across appendix figures.

(1) First-qualification-time buckets — bucket trades by their actual entry
    lead time (game_ts − entry_ts in hours). Per bucket: trade count, hit
    rate, mean edge, mean fill rate, mean log return, and average closing
    move (mid price drift from entry to tipoff on our side). This separates
    "late trades are bad because they're late" from "late trades are bad
    because they reflect a different signal regime."

(2) Late favorable-move toxicity — compute mid-price drift on our side
    between T-8h and entry. If trades whose price moved IN OUR FAVOR before
    we entered (we chased a move that already happened) systematically
    underperform, that's adverse selection: we were the late, less-informed
    flow taking the worst of the queue.

(3) Signal persistence classes — for each game, walk every 15-min qualifying
    snapshot and classify the entry pattern: early & continuous, early but
    with mid-window dropouts, or late-only (only qualified inside the last
    few hours). Compare hit rate / log return across classes.

Outputs:
    organized/outputs/trade_first_qual_time_buckets.csv
    organized/outputs/trade_first_qual_time_buckets.png
    organized/outputs/trade_late_favorable_move.csv
    organized/outputs/trade_late_favorable_move.png
    organized/outputs/trade_signal_persistence.csv
    organized/outputs/trade_signal_persistence.png
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
    collect_entries, collect_all_snapshots, run_kelly_sweep,
    add_trade_returns,
)
from final_model import load_year, LABEL_COL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

BANKROLL_REAL  = 5000.0
BEST_EDGE_MIN  = 0.05
BEST_NORM_MIN  = 0.25
KELLY_FRACTION = 0.5
ENTRY_WINDOW   = "half_life"

# Hour-before-tipoff buckets for analysis #1
LEAD_BIN_EDGES  = [0, 1, 2, 4, 8, 12, 17, 1e6]
LEAD_BIN_LABELS = ["T-1 to T-0", "T-2 to T-1", "T-4 to T-2",
                    "T-8 to T-4", "T-12 to T-8", "T-17 to T-12", "T-17+"]

# Persistence threshold (hours)
EARLY_THRESHOLD_H = 8


# --------------------------------------------------------------------------- #
# Candle-price helper                                                         #
# --------------------------------------------------------------------------- #

def market_mid_for(side: str, candles_df: pd.DataFrame, ts: pd.Timestamp) -> float:
    """Mid price on our side, evaluated at the latest candle ≤ ts.

    YES side: mid = (yes_bid + yes_ask) / 2
    NO  side: mid = 1 − (yes_bid + yes_ask) / 2
    """
    sub = candles_df[candles_df["ts"] <= ts]
    if sub.empty:
        return float("nan")
    last = sub.iloc[-1]
    yes_mid = 0.5 * (last["yes_bid_close"] + last["yes_ask_close"])
    return float(yes_mid) if side == "YES" else float(1.0 - yes_mid)


# --------------------------------------------------------------------------- #
# Build inputs                                                                #
# --------------------------------------------------------------------------- #

def build_inputs():
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

    idx = build_kalshi_trading_index(signals, pred_cols=("p_full_model", "p_elo"))
    return signals, idx


def build_trade_frame(idx) -> pd.DataFrame:
    """Run the live half-Kelly sweep strategy and augment trades with
    candle-derived diagnostic columns."""
    ents = collect_entries(
        idx["ticker_info"], idx["pretip"], "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
    )
    raw = run_kelly_sweep(ents, KELLY_FRACTION, idx["wt"], BANKROLL_REAL)
    if not raw:
        raise SystemExit("No trades generated.")
    tdf = add_trade_returns(pd.DataFrame(raw))

    # Map game_id -> ticker (so we can look up candles)
    gid_to_tkr = {info["game_id"]: tkr for tkr, info in idx["ticker_info"].items()}

    lead_h, mid_at_t8, mid_at_entry, mid_at_close = [], [], [], []
    for _, t in tdf.iterrows():
        gid = t["game_id"]; side = t["side"]
        tkr = gid_to_tkr.get(gid)
        candles = idx["pretip"].get(tkr) if tkr else None
        if candles is None:
            lead_h.append(np.nan); mid_at_t8.append(np.nan)
            mid_at_entry.append(np.nan); mid_at_close.append(np.nan); continue
        lead = (t["game_ts"] - t["entry_ts"]).total_seconds() / 3600.0
        m_t8 = market_mid_for(side, candles, t["game_ts"] - pd.Timedelta(hours=8))
        m_en = market_mid_for(side, candles, t["entry_ts"])
        m_cl = market_mid_for(side, candles, t["game_ts"])
        lead_h.append(lead); mid_at_t8.append(m_t8)
        mid_at_entry.append(m_en); mid_at_close.append(m_cl)

    tdf["lead_h"]               = lead_h
    tdf["mid_at_T_minus_8"]     = mid_at_t8
    tdf["mid_at_entry"]         = mid_at_entry
    tdf["mid_at_close"]         = mid_at_close
    tdf["entry_drift_from_T8"]  = tdf["mid_at_entry"] - tdf["mid_at_T_minus_8"]
    tdf["closing_move"]         = tdf["mid_at_close"] - tdf["mid_at_entry"]
    return tdf


# --------------------------------------------------------------------------- #
# Analysis 1: first-qualification-time buckets                                #
# --------------------------------------------------------------------------- #

def analysis_first_qual_buckets(tdf: pd.DataFrame) -> pd.DataFrame:
    df = tdf.copy()
    df["lead_bucket"] = pd.cut(df["lead_h"], bins=LEAD_BIN_EDGES,
                                 labels=LEAD_BIN_LABELS, right=False)
    rows = []
    for label in LEAD_BIN_LABELS:
        sub = df[df["lead_bucket"] == label]
        if len(sub) == 0:
            continue
        rows.append({
            "lead_bucket":          label,
            "trades":               int(len(sub)),
            "hit_rate":             float(sub["won"].mean()),
            "mean_edge":            float(sub["edge"].mean()),
            "mean_fill_rate":       float(sub["fill_pct"].mean()),
            "mean_log_return":      float(sub["log_ret"].mean()),
            "mean_closing_move":    float(sub["closing_move"].mean()),
            "median_closing_move":  float(sub["closing_move"].median()),
        })
    return pd.DataFrame(rows)


def plot_first_qual_buckets(tbl: pd.DataFrame, save_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    x = np.arange(len(tbl)); w = 0.7

    ax = axes[0, 0]
    bars = ax.bar(x, tbl["hit_rate"] * 100, color="#2ecc71", alpha=0.8, width=w)
    for xi, n, v in zip(x, tbl["trades"], tbl["hit_rate"]):
        ax.text(xi, v * 100 + 0.8, f"n={int(n)}", ha="center", fontsize=8)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tbl["lead_bucket"], fontsize=9, rotation=15)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Hit Rate by First-Qualification Lead Time", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    ax.bar(x, tbl["mean_log_return"], color="#e74c3c", alpha=0.8, width=w)
    for xi, v in zip(x, tbl["mean_log_return"]):
        ax.text(xi, v + (0.001 if v >= 0 else -0.003),
                f"{v:+.4f}", ha="center", fontsize=8,
                va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tbl["lead_bucket"], fontsize=9, rotation=15)
    ax.set_ylabel("Mean per-trade log return")
    ax.set_title("Mean Log Return by Lead Bucket", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    ax.bar(x, tbl["mean_edge"], color="#3498db", alpha=0.8, width=w, label="Mean edge")
    ax.set_xticks(x); ax.set_xticklabels(tbl["lead_bucket"], fontsize=9, rotation=15)
    ax.set_ylabel("Mean edge at entry")
    ax.set_title("Mean Edge at Entry by Lead Bucket", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for xi, v in zip(x, tbl["mean_edge"]):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)

    ax = axes[1, 1]
    ax.bar(x, tbl["mean_closing_move"], color="#9b59b6", alpha=0.8, width=w)
    for xi, v in zip(x, tbl["mean_closing_move"]):
        offset = 0.001 if v >= 0 else -0.003
        ax.text(xi, v + offset, f"{v:+.4f}", ha="center", fontsize=8,
                va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tbl["lead_bucket"], fontsize=9, rotation=15)
    ax.set_ylabel("Mean closing move (mid_close − mid_entry)")
    ax.set_title("Average Closing Move After Entry", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Trade Quality by First-Qualification Lead Bucket",
                 fontweight="bold", fontsize=13, y=1.0)
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis 2: late favorable-move toxicity                                    #
# --------------------------------------------------------------------------- #

def analysis_late_favorable_move(tdf: pd.DataFrame) -> pd.DataFrame:
    """Cross-tab by (entry_drift_from_T8 sign × lead_h class).

    Favorable late drift = entry_drift_from_T8 > 0 (our side's mid moved
    UP between T-8h and our entry — we entered after a move toward our
    predicted direction).

    Lead class: late (≤ 4h before tipoff) vs not-late (> 4h).
    """
    df = tdf.dropna(subset=["entry_drift_from_T8"]).copy()
    df["drift_class"] = np.where(
        df["entry_drift_from_T8"] > 0.005, "favorable late drift",
        np.where(df["entry_drift_from_T8"] < -0.005, "adverse late drift",
                  "flat"),
    )
    df["lead_class"]  = np.where(df["lead_h"] <= 4, "late (<=T-4h)", "not-late (>T-4h)")
    rows = []
    for (drift, lead), sub in df.groupby(["drift_class", "lead_class"]):
        if len(sub) == 0:
            continue
        rows.append({
            "drift_class":          drift,
            "lead_class":           lead,
            "trades":               int(len(sub)),
            "hit_rate":             float(sub["won"].mean()),
            "mean_edge":            float(sub["edge"].mean()),
            "mean_drift_T8_entry":  float(sub["entry_drift_from_T8"].mean()),
            "mean_log_return":      float(sub["log_ret"].mean()),
            "mean_closing_move":    float(sub["closing_move"].mean()),
        })
    out = pd.DataFrame(rows)
    # Sort with deterministic order
    drift_order = ["favorable late drift", "flat", "adverse late drift"]
    lead_order  = ["late (<=T-4h)", "not-late (>T-4h)"]
    out["drift_class"] = pd.Categorical(out["drift_class"], categories=drift_order, ordered=True)
    out["lead_class"]  = pd.Categorical(out["lead_class"],  categories=lead_order,  ordered=True)
    return out.sort_values(["drift_class", "lead_class"]).reset_index(drop=True)


def plot_late_favorable_move(tbl: pd.DataFrame, raw: pd.DataFrame, save_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- left: hit rate by drift × lead cell ---
    ax = axes[0]
    drift_order = list(tbl["drift_class"].cat.categories)
    lead_order  = list(tbl["lead_class"].cat.categories)
    x = np.arange(len(drift_order)); w = 0.38
    color_map = {"late (<=T-4h)": "#e74c3c", "not-late (>T-4h)": "#3498db"}
    for i, lead in enumerate(lead_order):
        sub = tbl[tbl["lead_class"] == lead].set_index("drift_class").reindex(drift_order)
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, sub["hit_rate"] * 100, width=w,
                      color=color_map[lead], alpha=0.85, label=lead)
        for xi, n, v in zip(x + offset, sub["trades"], sub["hit_rate"]):
            if pd.notna(v):
                ax.text(xi, v * 100 + 0.8, f"n={int(n)}", ha="center", fontsize=7.5)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(drift_order, fontsize=9)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Hit Rate: Late Favorable Drift vs. Other Trades",
                 fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    # --- right: scatter colored by win/loss ---
    ax = axes[1]
    df = raw.dropna(subset=["entry_drift_from_T8"]).copy()
    won = df[df["won"] == 1]; lost = df[df["won"] == 0]
    ax.scatter(won["entry_drift_from_T8"], won["lead_h"], alpha=0.55, s=30,
               color="#2ecc71", edgecolors="none", label=f"Won (n={len(won)})")
    ax.scatter(lost["entry_drift_from_T8"], lost["lead_h"], alpha=0.55, s=30,
               color="#e74c3c", edgecolors="none", label=f"Lost (n={len(lost)})")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axhline(4, color="gray", linestyle=":", alpha=0.7,
               label="Late-trade boundary (T-4h)")
    ax.set_xlabel("Mid drift on our side, T-8h → entry")
    ax.set_ylabel("Entry lead time (hours before tipoff)")
    ax.set_title("Scatter: Drift × Lead Time, by Outcome", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(True, alpha=0.3)

    fig.suptitle("Late Favorable-Move Toxicity Diagnostic",
                 fontweight="bold", fontsize=13, y=1.0)
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Analysis 3: signal persistence                                              #
# --------------------------------------------------------------------------- #

def classify_persistence(idx, tdf: pd.DataFrame) -> pd.DataFrame:
    """Walk every qualifying snapshot per game and classify the entry pattern.

    Classes:
      early_stable                 : first qual time is ≥ EARLY_THRESHOLD_H AND
                                     qualifying snapshots are continuous.
      early_disappeared_then_late  : first qual time is ≥ EARLY_THRESHOLD_H AND
                                     there is at least one 30-min gap between
                                     qualifying snapshots.
      late_only                    : first qual time is < EARLY_THRESHOLD_H.
    """
    snaps = collect_all_snapshots(
        idx["ticker_info"], idx["pretip"], "p_full_model",
        BEST_EDGE_MIN, BEST_NORM_MIN, ENTRY_WINDOW,
    )
    if not snaps:
        return pd.DataFrame()
    snap_df = pd.DataFrame(snaps)

    classes = {}
    for gid, sub in snap_df.groupby("game_id"):
        sub = sub.sort_values("entry_ts")
        first_ts = sub["entry_ts"].iloc[0]
        last_ts  = sub["entry_ts"].iloc[-1]
        game_ts  = sub["game_ts"].iloc[0]
        first_qual_h = (game_ts - first_ts).total_seconds() / 3600.0
        # gaps: contiguous 15-min snapshots from first_ts to last_ts?
        diffs = sub["entry_ts"].diff().dropna().dt.total_seconds() / 60.0
        has_gap = bool((diffs > 30.0).any())  # > 30 min between consecutive quals = gap

        if first_qual_h >= EARLY_THRESHOLD_H and not has_gap:
            cls = "early_stable"
        elif first_qual_h >= EARLY_THRESHOLD_H and has_gap:
            cls = "early_disappeared_then_late"
        else:
            cls = "late_only"
        classes[gid] = {
            "persistence_class": cls,
            "first_qual_h":      float(first_qual_h),
            "last_qual_h":       float((game_ts - last_ts).total_seconds() / 3600.0),
            "n_qual_snapshots":  int(len(sub)),
            "has_gap":           has_gap,
        }
    persist_df = pd.DataFrame.from_dict(classes, orient="index").reset_index().rename(
        columns={"index": "game_id"},
    )
    out = tdf.merge(persist_df, on="game_id", how="left")
    return out


def analysis_persistence(tdf: pd.DataFrame) -> pd.DataFrame:
    df = tdf.dropna(subset=["persistence_class"])
    rows = []
    order = ["early_stable", "early_disappeared_then_late", "late_only"]
    for cls in order:
        sub = df[df["persistence_class"] == cls]
        if len(sub) == 0:
            continue
        rows.append({
            "persistence_class": cls,
            "trades":            int(len(sub)),
            "hit_rate":          float(sub["won"].mean()),
            "mean_edge":         float(sub["edge"].mean()),
            "mean_lead_h":       float(sub["lead_h"].mean()),
            "median_lead_h":     float(sub["lead_h"].median()),
            "mean_log_return":   float(sub["log_ret"].mean()),
            "mean_closing_move": float(sub["closing_move"].mean()),
            "mean_first_qual_h": float(sub["first_qual_h"].mean()),
            "mean_n_snapshots":  float(sub["n_qual_snapshots"].mean()),
        })
    return pd.DataFrame(rows)


def plot_persistence(tbl: pd.DataFrame, save_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(tbl))
    labels = [c.replace("_", " ").title() for c in tbl["persistence_class"]]
    color_map = {"early_stable": "#2ecc71",
                  "early_disappeared_then_late": "#e67e22",
                  "late_only": "#e74c3c"}
    colors = [color_map[c] for c in tbl["persistence_class"]]

    ax = axes[0]
    bars = ax.bar(x, tbl["hit_rate"] * 100, color=colors, alpha=0.85, width=0.65)
    for xi, n, v in zip(x, tbl["trades"], tbl["hit_rate"]):
        ax.text(xi, v * 100 + 0.8, f"n={int(n)}", ha="center", fontsize=8)
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Hit rate (%)")
    ax.set_title("Hit Rate by Signal-Persistence Class", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, tbl["mean_log_return"], color=colors, alpha=0.85, width=0.65)
    for xi, v in zip(x, tbl["mean_log_return"]):
        ax.text(xi, v + (0.0015 if v >= 0 else -0.003),
                f"{v:+.4f}", ha="center", fontsize=8,
                va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean per-trade log return")
    ax.set_title("Mean Log Return by Persistence Class", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Signal Persistence — early threshold = T-{EARLY_THRESHOLD_H}h",
        fontweight="bold", fontsize=13, y=1.0,
    )
    plt.tight_layout()
    save_fig(fig, save_name)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    print("Loading cached holdout predictions ...")
    signals, idx = build_inputs()
    print(f"  signals: {len(signals)} games  |  matched tickers: "
          f"{len(idx['ticker_info'])}  |  with candles: {len(idx['pretip'])}")

    print("Generating live half-Kelly sweep trades + diagnostics ...")
    tdf = build_trade_frame(idx)
    print(f"  trades: {len(tdf)}")

    # ---- Analysis 1 ----
    tbl1 = analysis_first_qual_buckets(tdf)
    save_table(tbl1, "trade_first_qual_time_buckets")
    plot_first_qual_buckets(tbl1, "trade_first_qual_time_buckets")
    print("\n=== Analysis 1: First-qualification lead-time buckets ===")
    print(tbl1.to_string(index=False))

    # ---- Analysis 2 ----
    tbl2 = analysis_late_favorable_move(tdf)
    save_table(tbl2, "trade_late_favorable_move")
    plot_late_favorable_move(tbl2, tdf, "trade_late_favorable_move")
    print("\n=== Analysis 2: Late favorable-move toxicity ===")
    print(tbl2.to_string(index=False))

    # ---- Analysis 3 ----
    tdf_p = classify_persistence(idx, tdf)
    tbl3 = analysis_persistence(tdf_p)
    save_table(tbl3, "trade_signal_persistence")
    plot_persistence(tbl3, "trade_signal_persistence")
    print("\n=== Analysis 3: Signal persistence classes ===")
    print(tbl3.to_string(index=False))


if __name__ == "__main__":
    main()
