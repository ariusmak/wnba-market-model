"""Generate two correlation figures over the live half-Kelly trade set:

  1. trade_hit_rate_vs_edge_correlation.png
       Win indicator (0/1) vs. model-implied edge per trade, for Elo and the
       Full Model. Adds a rolling-mean smoother and reports Pearson +
       Spearman correlation per model and a logistic-fit smoother.

  2. trade_log_return_vs_edge_correlation.png
       Per-trade log return vs. edge, for both models, with a linear
       regression line per model and Pearson + Spearman correlation.

Both figures use the **live half-Kelly best config** trades
(edge_min = 0.05, norm_min = 0.25 — from
`trade_half_kelly_best_config_table.csv`), matching
`trade_hit_rate_by_edge_bucket.png` and `trade_log_return_by_edge_bucket.png`.

Outputs:
    organized/outputs/trade_hit_rate_vs_edge_correlation.png
    organized/outputs/trade_log_return_vs_edge_correlation.png
    organized/outputs/trade_edge_correlation_summary_table.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
PIPELINE_DIR = Path(__file__).resolve().parent
for d in (ANALYSIS_DIR, PIPELINE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from outputs import save_fig, save_table  # noqa: E402
from walkforward import fit_holdout_models  # noqa: E402
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_ideal, add_trade_returns, BANKROLL_INIT,
)
from final_model import load_year, LABEL_COL  # noqa: E402

MODEL_LABELS = {"elo": "Elo", "full_model": "Full Model"}
MODEL_COLORS = {"elo": "#3498db", "full_model": "#e74c3c"}
LIVE_EDGE_MIN = 0.05
LIVE_NORM_MIN = 0.25


def rolling_mean_by_edge(edge: np.ndarray, y: np.ndarray, frac: float = 0.30):
    """Smooth `y` against `edge` with a sliding window of `frac` of n. Returns
    (sorted_edge, sorted_smoothed_y)."""
    order = np.argsort(edge)
    e = edge[order]; v = y[order]
    n = len(e)
    if n == 0:
        return e, v
    win = max(int(round(frac * n)), 5)
    half = win // 2
    smoothed = np.empty(n)
    for i in range(n):
        lo = max(0, i - half); hi = min(n, i + half + 1)
        smoothed[i] = v[lo:hi].mean()
    return e, smoothed


def collect_trades_per_model(ticker_info, pretip):
    out = {}
    for model_name, model_col in [("elo", "p_elo"), ("full_model", "p_full_model")]:
        ents = collect_entries(
            ticker_info, pretip, model_col,
            LIVE_EDGE_MIN, LIVE_NORM_MIN, "half_life",
        )
        trades = run_kelly_ideal(ents, 0.5, BANKROLL_INIT)
        if not trades:
            out[model_name] = pd.DataFrame()
            continue
        out[model_name] = add_trade_returns(pd.DataFrame(trades))
    return out


def plot_hit_rate_vs_edge(trades_by_model: dict[str, pd.DataFrame]) -> tuple[plt.Figure, list[dict]]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    summary = []

    edge_grid = np.linspace(LIVE_EDGE_MIN, 0.55, 200).reshape(-1, 1)

    for model_name, tdf in trades_by_model.items():
        if tdf.empty:
            continue
        edge = tdf["edge"].values
        won  = tdf["won"].values.astype(float)
        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]

        # Pearson and Spearman correlations
        r_pearson,  p_pearson  = sstats.pearsonr(edge, won)
        r_spearman, p_spearman = sstats.spearmanr(edge, won)
        summary.append({
            "model":             model_name,
            "metric":            "hit_rate",
            "n":                 int(len(edge)),
            "pearson_r":         float(r_pearson),
            "pearson_p":         float(p_pearson),
            "spearman_r":        float(r_spearman),
            "spearman_p":        float(p_spearman),
        })

        # Jittered scatter so the 0/1 dots are visible
        jitter = (np.random.RandomState(42).rand(len(won)) - 0.5) * 0.06
        ax.scatter(edge, won + jitter, color=color, alpha=0.3, s=20,
                   edgecolors="none")

        # Logistic-fit smoother
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        lr.fit(edge.reshape(-1, 1), won.astype(int))
        proba = lr.predict_proba(edge_grid)[:, 1]
        ax.plot(edge_grid.ravel(), proba, color=color, linewidth=2.0,
                label=(f"{label}  n={len(edge)}  "
                       f"Pearson r={r_pearson:+.3f}, "
                       f"Spearman ρ={r_spearman:+.3f}"))

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Model-implied edge at entry (absolute)")
    ax.set_ylabel("Realized win indicator (jittered)  /  P(win)")
    ax.set_title("Hit Rate vs. Model-Implied Edge — Half-Kelly Best Config",
                 fontweight="bold")
    ax.set_ylim(-0.10, 1.10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, summary


def plot_log_return_vs_edge(trades_by_model: dict[str, pd.DataFrame]) -> tuple[plt.Figure, list[dict]]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    summary = []

    for model_name, tdf in trades_by_model.items():
        if tdf.empty:
            continue
        edge   = tdf["edge"].values
        logret = tdf["log_ret"].values
        color  = MODEL_COLORS[model_name]
        label  = MODEL_LABELS[model_name]

        r_pearson,  p_pearson  = sstats.pearsonr(edge, logret)
        r_spearman, p_spearman = sstats.spearmanr(edge, logret)
        summary.append({
            "model":             model_name,
            "metric":            "log_ret",
            "n":                 int(len(edge)),
            "pearson_r":         float(r_pearson),
            "pearson_p":         float(p_pearson),
            "spearman_r":        float(r_spearman),
            "spearman_p":        float(p_spearman),
        })

        ax.scatter(edge, logret, color=color, alpha=0.45, s=22, edgecolors="none")

        # OLS regression line (slope, intercept) over the trade set
        slope, intercept = np.polyfit(edge, logret, deg=1)
        x_line = np.linspace(edge.min(), edge.max(), 50)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2.0,
                label=(f"{label}  n={len(edge)}  "
                       f"slope={slope:+.3f}  "
                       f"Pearson r={r_pearson:+.3f}, "
                       f"Spearman ρ={r_spearman:+.3f}"))

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Model-implied edge at entry (absolute)")
    ax.set_ylabel("Per-trade log return  log(W$_{after}$ / W$_{before}$)")
    ax.set_title("Per-Trade Log Return vs. Model-Implied Edge — Half-Kelly Best Config",
                 fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig, summary


def main() -> None:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    print("Training holdout models …")
    preds_df, _ = fit_holdout_models(
        {"xgb_with_elo": {"type": "xgb", "use_bm": True}, "elo": {"type": "elo"}},
        holdout_year=2025, verbose=False,
    )
    model_w = (
        preds_df[preds_df["model"] == "xgb_with_elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_full_model"})
    )
    elo_w = (
        preds_df[preds_df["model"] == "elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_elo"})
    )
    test_df = (
        load_year(2025)
        .dropna(subset=[LABEL_COL, "base_margin"])
        [["game_id", "game_ts", "game_date", "home_team_id", "away_team_id", LABEL_COL]]
        .copy()
    )
    test_df["game_ts"]   = pd.to_datetime(test_df["game_ts"], utc=True)
    test_df["game_date"] = pd.to_datetime(test_df["game_date"])
    signals = test_df.merge(model_w, on="game_id").merge(elo_w, on="game_id")

    print("Building Kalshi trading index …")
    idx = build_kalshi_trading_index(signals)
    trades_by_model = collect_trades_per_model(idx["ticker_info"], idx["pretip"])

    fig_hit, hit_summary = plot_hit_rate_vs_edge(trades_by_model)
    save_fig(fig_hit, "trade_hit_rate_vs_edge_correlation"); plt.close(fig_hit)

    fig_lr, lr_summary = plot_log_return_vs_edge(trades_by_model)
    save_fig(fig_lr, "trade_log_return_vs_edge_correlation"); plt.close(fig_lr)

    summary_tbl = pd.DataFrame(hit_summary + lr_summary)
    save_table(summary_tbl, "trade_edge_correlation_summary_table")
    print(summary_tbl.to_string(index=False))


if __name__ == "__main__":
    main()
