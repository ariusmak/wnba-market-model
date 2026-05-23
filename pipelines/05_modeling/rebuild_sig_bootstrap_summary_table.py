"""Regenerate the half-Kelly significance table.

Outputs:
    organized/outputs/sig_bootstrap_summary_table.csv
    organized/outputs/sig_bootstrap_summary_table.png

Layout
------
Two per-model rows (Elo, Full Model):
    trades, mean_log_return, ci_lo, ci_hi, p_positive, p_value_one_sided,
    implied_terminal_wealth
Plus a "Full Model − Elo" row:
    mean_difference, ci_lo, ci_hi, p_fm_gt_elo, p_value_one_sided

Methodology
-----------
Same trade set and bootstrap conventions as `significance.ipynb`:
  - Live half-Kelly best config (edge_min = 0.05, norm_min = 0.25)
  - $100 starting bankroll, half-Kelly sizing
  - 10,000 bootstrap resamples, seed = 42
  - Per-mean and difference distributions are independent resamples (matching
    the existing `bootstrap_distribution` / `bootstrap_diff` helpers).

Null-centered one-sided p-values are added for each strategy and for the
difference. Per strategy: shift trade returns to mean zero, bootstrap, count
the share of resampled means that meet or exceed the observed mean. For the
difference: center the existing difference-bootstrap draws to mean zero,
count the share that meet or exceed the observed difference. Each is the
permutation-style p-value for a one-sided test (HA: mean > 0).
"""
from __future__ import annotations

import sys
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
from walkforward import fit_holdout_models  # noqa: E402
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_ideal, add_trade_returns, BANKROLL_INIT,
)
from eval_helpers import bootstrap_diff, bootstrap_distribution  # noqa: E402
from final_model import load_year, LABEL_COL  # noqa: E402

LIVE_EDGE_MIN = 0.05
LIVE_NORM_MIN = 0.25
N_BOOT = 10_000
SEED = 42
START = BANKROLL_INIT  # $100, matching the existing significance notebook


def _trades(ticker_info, pretip, model_col):
    ents = collect_entries(
        ticker_info, pretip, model_col,
        LIVE_EDGE_MIN, LIVE_NORM_MIN, "half_life",
    )
    tdf = add_trade_returns(pd.DataFrame(run_kelly_ideal(ents, 0.5, BANKROLL_INIT)))
    return tdf


def _fmt_p(p: float) -> str:
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def render_table_png(df: pd.DataFrame, *,
                      diff_row: dict, save_name: str) -> None:
    """Render the summary table as a publication-quality PNG."""
    plt.rcParams.update({"font.size": 11, "figure.dpi": 200})

    # --- per-model block ---
    body_cols = ["Trades", "Mean log return", "95% CI",
                 "P(mean > 0)", "One-sided p-value", "Implied terminal wealth"]
    body_rows = []
    for _, r in df.iterrows():
        body_rows.append([
            f"{int(r['trades'])}",
            f"{r['mean_log_ret']:+.4f}",
            f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
            f"{r['p_positive']:.3f}",
            _fmt_p(r["p_value_one_sided"]),
            f"${r['implied_terminal_wealth']:,.0f}",
        ])
    row_labels = [r["model"] for _, r in df.iterrows()]

    # --- difference block ---
    diff_cols = ["Mean difference", "95% CI on difference",
                 "P(FM > Elo)", "One-sided p-value"]
    diff_vals = [
        f"{diff_row['diff']:+.4f}",
        f"[{diff_row['ci_lo']:+.4f}, {diff_row['ci_hi']:+.4f}]",
        f"{diff_row['p_fm_gt_elo']:.3f}",
        _fmt_p(diff_row["p_value_one_sided"]),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 4.4))
    ax.axis("off")
    fig.suptitle("Bootstrap Comparison of Half-Kelly Per-Trade Log Returns",
                 fontweight="bold", fontsize=12, y=0.97)

    # Top table: per-model
    tbl1 = ax.table(
        cellText=body_rows,
        colLabels=body_cols,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.05, 0.45, 0.95, 0.4],
    )
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(10)
    for (row, col), cell in tbl1.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dfe6ed"); cell.set_text_props(weight="bold")
        if col == -1:
            cell.set_facecolor("#f2f5f8"); cell.set_text_props(weight="bold")

    # Bottom table: full-model − Elo
    tbl2 = ax.table(
        cellText=[diff_vals],
        colLabels=diff_cols,
        rowLabels=["Full Model − Elo"],
        cellLoc="center",
        rowLoc="center",
        bbox=[0.05, 0.10, 0.95, 0.22],
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(10)
    for (row, col), cell in tbl2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#fbe2dc"); cell.set_text_props(weight="bold")
        if col == -1:
            cell.set_facecolor("#f2f5f8"); cell.set_text_props(weight="bold")

    fig.text(0.5, 0.03,
             f"$100 starting bankroll · half-Kelly · "
             f"B = {N_BOOT:,} resamples · seed = {SEED} · "
             "one-sided null-centered bootstrap p-value (HA: mean > 0)",
             ha="center", fontsize=8.5, color="#555555")
    save_fig(fig, save_name)
    plt.close(fig)


def null_centered_pvalue(g: np.ndarray, *, n_boot: int = N_BOOT,
                          seed: int = SEED) -> tuple[float, float]:
    """One-sided null-centered bootstrap p-value for H0: mean(g) = 0.

    Returns (obs_mean, p_value).
        g_null   = g − mean(g)
        sample n_boot means from g_null with replacement
        p        = Pr(boot_mean_null >= obs_mean)
    """
    g = np.asarray(g)
    obs_mean = float(g.mean())
    g_null = g - obs_mean
    rng = np.random.default_rng(seed)
    n = len(g)
    boot_null = np.empty(n_boot)
    for i in range(n_boot):
        boot_null[i] = rng.choice(g_null, size=n, replace=True).mean()
    p_value = float((boot_null >= obs_mean).mean())
    return obs_mean, p_value


def null_centered_diff_pvalue(boot_diffs: np.ndarray, obs_diff: float) -> float:
    """One-sided null-centered p-value for H0: mean_FM − mean_Elo = 0.

    Centers the existing difference-bootstrap distribution to mean zero, then
    counts the share at or above the observed difference. This is the
    standard "shift the bootstrap to the null" construction.
    """
    centered = np.asarray(boot_diffs) - float(np.mean(boot_diffs))
    return float((centered >= obs_diff).mean())


def main() -> None:
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
    test_df["game_ts"] = pd.to_datetime(test_df["game_ts"], utc=True)
    signals = test_df.merge(model_w, on="game_id").merge(elo_w, on="game_id")

    print("Building Kalshi trading index …")
    idx = build_kalshi_trading_index(signals)
    ticker_info, pretip = idx["ticker_info"], idx["pretip"]

    elo_t = _trades(ticker_info, pretip, "p_elo")
    fm_t  = _trades(ticker_info, pretip, "p_full_model")

    lr_elo = elo_t["log_ret"].values
    lr_fm  = fm_t["log_ret"].values

    # Per-mean bootstrap distributions (independent resamples per model)
    boot_elo = bootstrap_distribution(lr_elo, n_boot=N_BOOT, statistic=np.mean, seed=SEED)
    boot_fm  = bootstrap_distribution(lr_fm,  n_boot=N_BOOT, statistic=np.mean, seed=SEED)

    # Difference bootstrap (matches existing significance.ipynb conventions)
    diff_result = bootstrap_diff(lr_fm, lr_elo, n_boot=N_BOOT, statistic=np.mean,
                                 ci=(2.5, 97.5), seed=SEED)

    # --- formal one-sided null-centered p-values ---
    _, p_value_elo = null_centered_pvalue(lr_elo, n_boot=N_BOOT, seed=SEED)
    _, p_value_fm  = null_centered_pvalue(lr_fm,  n_boot=N_BOOT, seed=SEED)
    p_value_diff = null_centered_diff_pvalue(diff_result["diffs"],
                                              obs_diff=diff_result["diff"])

    def _row(label, lr, boot, p_value):
        n = len(lr)
        mean = float(lr.mean())
        ci = np.percentile(boot, [2.5, 97.5])
        return {
            "model":                   label,
            "trades":                  int(n),
            "mean_log_ret":            mean,
            "ci_lo":                   float(ci[0]),
            "ci_hi":                   float(ci[1]),
            "p_positive":              float((boot > 0).mean()),
            "p_value_one_sided":       float(p_value),
            "implied_terminal_wealth": float(START * np.exp(mean * n)),
        }

    per_model = pd.DataFrame([
        _row("Elo",        lr_elo, boot_elo, p_value_elo),
        _row("Full Model", lr_fm,  boot_fm,  p_value_fm),
    ])

    diff_row = {
        "comparison":        "Full Model − Elo",
        "diff":              diff_result["diff"],
        "ci_lo":             diff_result["ci_lo"],
        "ci_hi":             diff_result["ci_hi"],
        "p_fm_gt_elo":       diff_result["p_a_gt_b"],
        "p_value_one_sided": float(p_value_diff),
    }

    # ----- save flat CSV -----
    csv_rows = [
        {**r, "block": "per_model"} for r in per_model.to_dict("records")
    ] + [
        {
            "block":                   "difference",
            "model":                   diff_row["comparison"],
            "trades":                  np.nan,
            "mean_log_ret":            diff_row["diff"],
            "ci_lo":                   diff_row["ci_lo"],
            "ci_hi":                   diff_row["ci_hi"],
            "p_positive":              np.nan,
            "p_value_one_sided":       diff_row["p_value_one_sided"],
            "implied_terminal_wealth": np.nan,
            "p_fm_gt_elo":             diff_row["p_fm_gt_elo"],
        }
    ]
    csv_df = pd.DataFrame(csv_rows)[
        ["block", "model", "trades", "mean_log_ret", "ci_lo", "ci_hi",
         "p_positive", "p_value_one_sided", "p_fm_gt_elo",
         "implied_terminal_wealth"]
    ]
    save_table(csv_df, "sig_bootstrap_summary_table")

    # ----- render PNG -----
    render_table_png(per_model, diff_row=diff_row,
                     save_name="sig_bootstrap_summary_table")

    print(csv_df.to_string(index=False))


if __name__ == "__main__":
    main()
