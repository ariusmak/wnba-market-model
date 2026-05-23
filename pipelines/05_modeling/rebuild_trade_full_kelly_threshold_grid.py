"""Generate the full-Kelly appendix grid-search table.

Writes:
    organized/outputs/trade_full_kelly_threshold_grid.csv
    organized/outputs/trade_full_kelly_threshold_grid.png

Reuses cached holdout predictions
(`organized/data/model_comparison/holdout_model_comparison_2025.csv`) — no
model retraining. Only the trading engine is rerun (Kelly fraction = 1.0)
so we can record `mean_entry_price` per cell, parallel to the fixed-risk
and half-Kelly grids.

Grid:
    model       ∈ {elo, full_model}
    edge_min    ∈ {0.05, 0.10, 0.15}
    norm_min    ∈ {0.0, 0.10, 0.20, 0.25}
    entry_window = 'half_life'

Selection rule: terminal bankroll (matches half-Kelly).
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
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_kelly_ideal, add_trade_returns, BANKROLL_INIT,
)
from final_model import load_year, LABEL_COL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

EDGE_MINS = [0.05, 0.10, 0.15]
NORM_MINS = [0.0, 0.10, 0.20, 0.25]
MODELS = [("elo", "p_elo"), ("full_model", "p_full_model")]
ENTRY_WINDOW = "half_life"
KELLY_FRACTION = 1.0  # full Kelly

NOTE = (
    "The selected configuration is the best-performing threshold pair under "
    "the criterion used in the main text. Full-Kelly configurations are "
    "selected by terminal bankroll, matching the half-Kelly rule. Full Kelly "
    "is over-aggressive: large drawdowns and bankroll wipeouts are typical "
    "outcomes of the same per-trade edge under this sizing."
)


def build_signals_and_index():
    holdout = pd.read_csv(HOLDOUT_CSV)
    model_w = (
        holdout[holdout["model"] == "xgb_with_elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_full_model"})
    )
    elo_w = (
        holdout[holdout["model"] == "elo"][["game_id", "pred_prob"]]
        .rename(columns={"pred_prob": "p_elo"})
    )
    test_df = (
        load_year(2025).dropna(subset=[LABEL_COL, "base_margin"])
        [["game_id", "game_ts", "game_date", "home_team_id", "away_team_id", LABEL_COL]]
        .copy()
    )
    test_df["game_ts"] = pd.to_datetime(test_df["game_ts"], utc=True)
    signals = test_df.merge(model_w, on="game_id").merge(elo_w, on="game_id")
    idx = build_kalshi_trading_index(signals)
    return signals, idx


def run_grid(ticker_info, pretip):
    rows = []
    for model_name, model_col in MODELS:
        for em in EDGE_MINS:
            for nm in NORM_MINS:
                ents = collect_entries(
                    ticker_info, pretip, model_col, em, nm, ENTRY_WINDOW,
                )
                ents_df = pd.DataFrame(ents)
                if ents_df.empty:
                    rows.append({
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades": 0, "hit_rate": np.nan,
                        "mean_kelly_fraction": np.nan,
                        "mean_log_return": np.nan,
                        "final_bankroll": float(BANKROLL_INIT),
                        "total_return": 0.0,
                        "max_drawdown": 0.0,
                        "mean_edge": np.nan,
                        "mean_entry_price": np.nan,
                    })
                    continue

                mean_entry_price = float(ents_df["entry_px"].mean())
                mean_edge        = float(ents_df["edge"].mean())

                fk = pd.DataFrame(run_kelly_ideal(ents, KELLY_FRACTION, BANKROLL_INIT))
                if len(fk):
                    fk = add_trade_returns(fk)
                    fb = float(fk["bankroll"].iloc[-1])
                    dd = float((fk["bankroll"].cummax() - fk["bankroll"]).max())
                    rows.append({
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades":              len(fk),
                        "hit_rate":            float(fk["won"].mean()),
                        "mean_kelly_fraction": float(fk["kelly_f"].mean()),
                        "mean_log_return":     float(fk["log_ret"].mean()),
                        "final_bankroll":      fb,
                        "total_return":        (fb - BANKROLL_INIT) / BANKROLL_INIT,
                        "max_drawdown":        dd,
                        "mean_edge":           mean_edge,
                        "mean_entry_price":    mean_entry_price,
                    })
                else:
                    rows.append({
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades": 0, "hit_rate": np.nan,
                        "mean_kelly_fraction": np.nan,
                        "mean_log_return": np.nan,
                        "final_bankroll": float(BANKROLL_INIT),
                        "total_return": 0.0,
                        "max_drawdown": 0.0,
                        "mean_edge": mean_edge,
                        "mean_entry_price": mean_entry_price,
                    })

    df = pd.DataFrame(rows).sort_values(["model", "edge_min", "norm_min"]).reset_index(drop=True)
    return df


def best_per_model(df: pd.DataFrame, metric: str = "final_bankroll"):
    out = {}
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].dropna(subset=[metric])
        if sub.empty:
            continue
        idx = sub[metric].idxmax()
        r = df.loc[idx]
        out[model_name] = (round(float(r["edge_min"]), 4),
                            round(float(r["norm_min"]), 4))
    return out


def round_full_kelly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("hit_rate", "total_return", "mean_kelly_fraction",
              "mean_edge", "mean_entry_price"):
        out[c] = out[c].round(3)
    out["mean_log_return"] = out["mean_log_return"].round(4)
    for c in ("final_bankroll", "max_drawdown"):
        out[c] = out[c].round(2)
    return out


def _fmt_cell(val, col):
    if pd.isna(val):
        return "—"
    if col == "trades":
        return f"{int(val)}"
    if col in ("final_bankroll", "max_drawdown"):
        return f"${val:,.2f}"
    if col == "mean_log_return":
        return f"{val:+.4f}"
    if col == "total_return":
        return f"{val:+.3f}"
    return f"{val:.3f}"


def render_png(df: pd.DataFrame, *, best_keys: dict, save_name: str) -> None:
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

    panels = [(m, df[df["model"] == m].reset_index(drop=True))
              for m in df["model"].unique()]

    rows_per_panel = max(len(s) for _, s in panels)
    fig_height = 1.6 + len(panels) * (rows_per_panel + 1) * 0.30
    fig, ax = plt.subplots(figsize=(14.0, fig_height))
    ax.axis("off")
    fig.suptitle("Full-Kelly Threshold Grid Search",
                 fontweight="bold", fontsize=12, y=0.985)

    cols = ["trades", "hit_rate", "mean_kelly_fraction", "mean_log_return",
            "final_bankroll", "total_return", "max_drawdown",
            "mean_edge", "mean_entry_price"]
    headers = ["Trades", "Hit rate", "Mean Kelly f", "Mean log ret",
               "Final bankroll", "Total return", "Max drawdown",
               "Mean edge", "Mean entry px"]

    y_cursor = 0.92
    panel_h = (0.92 - 0.10) / len(panels)
    for model, sub in panels:
        body = []
        for _, r in sub.iterrows():
            body.append([_fmt_cell(r[c], c) for c in cols])
        row_labels = [f"{r.edge_min:.2f} / {r.norm_min:.2f}" for r in sub.itertuples()]
        bb_top = y_cursor - 0.02
        bb_height = panel_h - 0.04
        bbox = [0.06, bb_top - bb_height, 0.94, bb_height]
        tbl = ax.table(
            cellText=body, colLabels=headers,
            rowLabels=row_labels,
            cellLoc="center", rowLoc="center", bbox=bbox,
        )
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        best = best_keys.get(model)
        for (rr, cc), cell in tbl.get_celld().items():
            if rr == 0:
                cell.set_facecolor("#dfe6ed"); cell.set_text_props(weight="bold")
            elif cc == -1:
                cell.set_facecolor("#f2f5f8"); cell.set_text_props(weight="bold")
            else:
                row_data = sub.iloc[rr - 1]
                if best and (round(float(row_data["edge_min"]), 4) == best[0]
                             and round(float(row_data["norm_min"]), 4) == best[1]):
                    cell.set_facecolor("#fff2cc")
                    cell.set_text_props(weight="bold")
        ax.text(0.06, bb_top + 0.005, f"{model.replace('_', ' ').title()}",
                fontweight="bold", fontsize=11)
        y_cursor -= panel_h

    fig.text(0.5, 0.045, NOTE, ha="center", fontsize=8.0,
             color="#444444", wrap=True)
    fig.text(0.5, 0.012, "Row labels: edge_min / norm_min · "
             "selected best row highlighted in yellow.",
             ha="center", fontsize=7.5, color="#666666")
    save_fig(fig, save_name)
    plt.close(fig)


def main() -> None:
    print("Loading cached holdout predictions ...")
    signals, idx = build_signals_and_index()
    print(f"  signals: {len(signals)} games  |  matched tickers: "
          f"{len(idx['ticker_info'])}  |  with candles: {len(idx['pretip'])}")

    print("Running full-Kelly threshold grid ...")
    grid = run_grid(idx["ticker_info"], idx["pretip"])
    rounded = round_full_kelly(grid)
    save_table(rounded, "trade_full_kelly_threshold_grid")

    best = best_per_model(grid, metric="final_bankroll")
    print(f"[ok] full-Kelly best by terminal bankroll: {best}")
    render_png(rounded, best_keys=best,
               save_name="trade_full_kelly_threshold_grid")

    # also print the best row per model for sanity
    for m in grid["model"].unique():
        sub = grid[grid["model"] == m].sort_values("final_bankroll", ascending=False)
        top = sub.head(3)[["model", "edge_min", "norm_min", "trades",
                            "final_bankroll", "total_return", "max_drawdown"]]
        print(f"\nTop 3 full-Kelly configs for {m}:")
        print(top.to_string(index=False))


if __name__ == "__main__":
    main()
