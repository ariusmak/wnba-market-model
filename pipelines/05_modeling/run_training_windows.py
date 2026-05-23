"""Driver script that mirrors `training_windows.ipynb` and writes all outputs.

Saves:
    organized/outputs/training_windows_logloss_per_fold.csv
    organized/outputs/training_windows_logloss_summary.csv
    organized/outputs/training_windows_trading_summary.csv
    organized/outputs/training_windows_logloss_comparison.png
    organized/outputs/training_windows_equity_curves.png
"""
from __future__ import annotations

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
PIPELINE_DIR = Path(__file__).resolve().parent
for d in (ANALYSIS_DIR, PIPELINE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from outputs import save_fig, save_table  # noqa: E402
from walkforward import walk_forward_models, fit_holdout_models  # noqa: E402
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_sweep, add_trade_returns, equity_by_payout,
)
from final_model import load_year, LABEL_COL  # noqa: E402

OOF_YEARS = list(range(2020, 2025))
HOLDOUT_YEAR = 2025
BANKROLL_REAL = 5000.0
BEST_EDGE_MIN = 0.05
BEST_NORM_MIN = 0.25

MODEL_SPEC = {"xgb_with_elo": {"type": "xgb", "use_bm": True}}

METHODS = {
    "extend_2015": {"kind": "extend",  "start": 2015, "label": "Extend (2015->)", "color": "#3498db"},
    "extend_2018": {"kind": "extend",  "start": 2018, "label": "Extend (2018->)", "color": "#2ecc71"},
    "roll_2yr":    {"kind": "rolling", "years": 2,    "label": "Rolling 2-year",  "color": "#e67e22"},
    "roll_3yr":    {"kind": "rolling", "years": 3,    "label": "Rolling 3-year",  "color": "#e74c3c"},
}


def train_start_for(method_key: str, test_year: int) -> int:
    cfg = METHODS[method_key]
    return cfg["start"] if cfg["kind"] == "extend" else test_year - cfg["years"]


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    # ---- 1. OOF per method ----
    oof_all: dict[str, pd.DataFrame] = {}
    for k in METHODS:
        print(f"=== OOF :: {k} ===")
        rows = []
        for test_year in OOF_YEARS:
            ts = train_start_for(k, test_year)
            sub = walk_forward_models(
                MODEL_SPEC, oof_years=[test_year], train_start=ts, verbose=False,
            )
            sub["method"] = k; sub["train_start"] = ts; sub["test_year"] = test_year
            rows.append(sub)
            print(f"  fold {test_year}  ts={ts}  rows={len(sub)}")
        oof_all[k] = pd.concat(rows, ignore_index=True)

    fold_rows = []
    for k, df in oof_all.items():
        for test_year in OOF_YEARS:
            sub = df[df["season"] == test_year]
            ll = log_loss(sub["home_win"], np.clip(sub["pred_prob"], 1e-7, 1 - 1e-7))
            fold_rows.append({
                "method": k, "test_year": test_year,
                "train_start": int(sub["train_start"].iloc[0]),
                "log_loss": float(ll), "n_games": int(len(sub)),
            })
    fold_tbl = pd.DataFrame(fold_rows)
    save_table(fold_tbl, "training_windows_logloss_per_fold")

    # ---- 2. 2025 holdout ----
    holdout_preds = {}
    holdout_starts = {}
    for k, cfg in METHODS.items():
        ts = (cfg["start"] if cfg["kind"] == "extend" else HOLDOUT_YEAR - cfg["years"])
        holdout_starts[k] = ts
        df, _ = fit_holdout_models(
            MODEL_SPEC, holdout_year=HOLDOUT_YEAR, train_start=ts, verbose=False,
        )
        df["method"] = k; df["train_start"] = ts
        holdout_preds[k] = df
        sub = df[df["model"] == "xgb_with_elo"]
        ll = log_loss(sub["home_win"], np.clip(sub["pred_prob"], 1e-7, 1 - 1e-7))
        print(f"  holdout :: {k:<12s}  ts={ts}  n={len(sub)}  2025 LL={ll:.4f}")

    rows = []
    for k, cfg in METHODS.items():
        sub_oof = fold_tbl[fold_tbl["method"] == k]
        h_sub = holdout_preds[k][holdout_preds[k]["model"] == "xgb_with_elo"]
        rows.append({
            "method": k, "label": cfg["label"],
            "train_start_2025": holdout_starts[k],
            "oof_mean_log_loss": float(sub_oof["log_loss"].mean()),
            "oof_min_log_loss":  float(sub_oof["log_loss"].min()),
            "oof_max_log_loss":  float(sub_oof["log_loss"].max()),
            "holdout_log_loss_2025": float(log_loss(
                h_sub["home_win"], np.clip(h_sub["pred_prob"], 1e-7, 1 - 1e-7),
            )),
        })
    ll_summary = pd.DataFrame(rows)
    save_table(ll_summary, "training_windows_logloss_summary")
    print("\nLog-loss summary:")
    print(ll_summary.to_string(index=False))

    # ---- 3. Log-loss comparison figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for k, cfg in METHODS.items():
        sub = fold_tbl[fold_tbl["method"] == k].sort_values("test_year")
        ax.plot(sub["test_year"], sub["log_loss"], marker="o", linewidth=2,
                color=cfg["color"], label=cfg["label"])
    ax.set_xlabel("Test year (OOF fold)"); ax.set_ylabel("Log loss")
    ax.set_title("Per-Fold OOF Log Loss by Training Window", fontweight="bold")
    ax.set_xticks(OOF_YEARS); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    x = np.arange(len(METHODS)); w = 0.38
    labels = [cfg["label"] for cfg in METHODS.values()]
    colors = [cfg["color"] for cfg in METHODS.values()]
    ax.bar(x - w / 2, ll_summary["oof_mean_log_loss"], width=w,
           color=colors, alpha=0.55, edgecolor="white", label="Mean OOF (2020-2024)")
    ax.bar(x + w / 2, ll_summary["holdout_log_loss_2025"], width=w,
           color=colors, alpha=1.0, edgecolor="white", label="2025 holdout")
    for xi, (a, b) in enumerate(zip(ll_summary["oof_mean_log_loss"],
                                      ll_summary["holdout_log_loss_2025"])):
        ax.text(xi - w / 2, a + 0.001, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, b + 0.001, f"{b:.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Log loss")
    ax.set_title("Mean OOF vs 2025 Holdout Log Loss", fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "training_windows_logloss_comparison")
    plt.close(fig)

    # ---- 4. Trading on 2025 holdout ----
    test_2025 = (
        load_year(HOLDOUT_YEAR).dropna(subset=[LABEL_COL, "base_margin"])
        [["game_id","game_ts","game_date","home_team_id","away_team_id", LABEL_COL,"p_elo"]]
        .copy()
    )
    test_2025["game_ts"] = pd.to_datetime(test_2025["game_ts"], utc=True)
    test_2025["game_date"] = pd.to_datetime(test_2025["game_date"])

    trading_results, trade_frames = {}, {}
    for k, cfg in METHODS.items():
        h_sub = holdout_preds[k]
        h_sub = h_sub[h_sub["model"] == "xgb_with_elo"][["game_id", "pred_prob"]].rename(
            columns={"pred_prob": "p_full_model"},
        )
        signals = test_2025.merge(h_sub, on="game_id")
        idx = build_kalshi_trading_index(signals, pred_cols=("p_full_model", "p_elo"))
        ents = collect_entries(
            idx["ticker_info"], idx["pretip"], "p_full_model",
            BEST_EDGE_MIN, BEST_NORM_MIN, "half_life",
        )
        raw = run_kelly_sweep(ents, 0.5, idx["wt"], BANKROLL_REAL)
        if not raw:
            print(f"  trading :: {k}  no trades")
            continue
        tdf = add_trade_returns(pd.DataFrame(raw))
        eq  = equity_by_payout(tdf, bankroll_init=BANKROLL_REAL, ts_col="game_ts")
        fb_engine = float(tdf["bankroll"].iloc[-1])
        dd_engine = float((tdf["bankroll"].cummax() - tdf["bankroll"]).max())
        mean_log_ret = float(tdf["log_ret"].mean())
        std_log_ret  = float(tdf["log_ret"].std())
        sharpe = mean_log_ret / std_log_ret if std_log_ret > 0 else float("nan")
        trading_results[k] = {
            "method":            k,
            "label":             cfg["label"],
            "n_trades":          int(len(tdf)),
            "hit_rate":          float(tdf["won"].mean()),
            "total_return":      (fb_engine - BANKROLL_REAL) / BANKROLL_REAL,
            "final_bankroll":    fb_engine,
            "final_bankroll_payout_view": float(eq["display_bankroll"].iloc[-1]),
            "max_drawdown":      dd_engine,
            "mean_log_return":   mean_log_ret,
            "std_log_return":    std_log_ret,
            "sharpe_per_trade":  sharpe,
            "mean_fill_rate":    float(tdf["fill_pct"].mean()),
        }
        trade_frames[k] = (tdf, eq)
        print(f"  trading :: {k}  trades={len(tdf)}  ret={trading_results[k]['total_return']:+.2%}  "
              f"DD=${dd_engine:.0f}  sharpe={sharpe:+.3f}")

    trading_summary = pd.DataFrame(list(trading_results.values()))
    save_table(trading_summary, "training_windows_trading_summary")
    print("\nTrading summary:")
    print(trading_summary.to_string(index=False))

    # ---- 5. Equity curves figure ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, (tdf, eq) in trade_frames.items():
        cfg = METHODS[k]; r = trading_results[k]
        ax.step(eq["game_ts"], eq["display_bankroll"], where="post",
                color=cfg["color"], linewidth=1.7,
                label=f"{cfg['label']} ({r['total_return']:+.0%}, n={r['n_trades']})")
    ax.axhline(BANKROLL_REAL, color="gray", linestyle=":", alpha=0.6,
               label=f"Starting bankroll (${BANKROLL_REAL:,.0f})")
    ax.set_xlabel("Settlement date"); ax.set_ylabel("Bankroll ($)")
    ax.set_title("2025 Half-Kelly Sweep Equity Curves by Training Window",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_fig(fig, "training_windows_equity_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
