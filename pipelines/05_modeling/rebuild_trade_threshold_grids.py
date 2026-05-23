"""Generate appendix grid-search tables for trading thresholds.

Writes:
    organized/outputs/trade_fixed_risk_threshold_grid.csv
    organized/outputs/trade_fixed_risk_threshold_grid.png
    organized/outputs/trade_half_kelly_threshold_grid.csv
    organized/outputs/trade_half_kelly_threshold_grid.png

Reuses cached holdout predictions
(`organized/data/model_comparison/holdout_model_comparison_2025.csv`) — no
model retraining. Only the trading engines are rerun so we can record
`mean_entry_price` per cell, which is not in the existing grid CSVs.

Grid:
    model       ∈ {elo, full_model}
    edge_min    ∈ {0.05, 0.10, 0.15}
    norm_min    ∈ {0.0, 0.10, 0.20, 0.25}
    entry_window = 'half_life'

Selection rule:
    Fixed-risk : ROI per trade
    Half-Kelly : terminal bankroll
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

from outputs import OUTPUTS_DIR, save_fig, save_table  # noqa: E402
from markets import build_kalshi_trading_index  # noqa: E402
from trading import (  # noqa: E402
    collect_entries, run_fixed_risk, run_kelly_ideal, add_trade_returns,
    BANKROLL_INIT,
)
from final_model import load_year, LABEL_COL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = PROJECT_ROOT / "data" / "model_comparison" / "holdout_model_comparison_2025.csv"

EDGE_MINS = [0.05, 0.10, 0.15]
NORM_MINS = [0.0, 0.10, 0.20, 0.25]
MODELS = [("elo", "p_elo"), ("full_model", "p_full_model")]
ENTRY_WINDOW = "half_life"

# Validation: best rows expected by the main text
EXPECTED_BEST = {
    "fixed":      {"elo": (0.15, 0.25), "full_model": (0.15, 0.25)},
    "half_kelly": {"elo": (0.05, 0.25), "full_model": (0.05, 0.25)},
}

NOTE = (
    "The selected configuration is the best-performing threshold pair under "
    "the criterion used in the main text. Fixed-risk configurations are "
    "selected by ROI per trade; half-Kelly configurations are selected by "
    "terminal bankroll."
)


# --------------------------------------------------------------------------- #
# Build inputs                                                                #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Grid                                                                        #
# --------------------------------------------------------------------------- #

def run_grid(ticker_info, pretip):
    fixed_rows, hk_rows = [], []
    for model_name, model_col in MODELS:
        for em in EDGE_MINS:
            for nm in NORM_MINS:
                ents = collect_entries(
                    ticker_info, pretip, model_col, em, nm, ENTRY_WINDOW,
                )
                ents_df = pd.DataFrame(ents)
                if ents_df.empty:
                    base = {
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades": 0, "hit_rate": np.nan,
                        "mean_edge": np.nan, "mean_entry_price": np.nan,
                    }
                    fixed_rows.append({**base,
                        "total_pnl": 0.0, "total_fees": 0.0,
                        "roi_per_trade": np.nan})
                    hk_rows.append({**base,
                        "mean_kelly_fraction": np.nan,
                        "mean_log_return": np.nan,
                        "final_bankroll": float(BANKROLL_INIT),
                        "total_return": 0.0,
                        "max_drawdown": 0.0,
                    })
                    continue

                mean_entry_price = float(ents_df["entry_px"].mean())
                mean_edge        = float(ents_df["edge"].mean())

                # --- fixed risk ---
                fr = pd.DataFrame(run_fixed_risk(ents, 1.0, BANKROLL_INIT))
                if len(fr):
                    n = len(fr)
                    fixed_rows.append({
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades": n,
                        "hit_rate":         float(fr["won"].mean()),
                        "total_pnl":        float(fr["pnl"].sum()),
                        "total_fees":       float(fr["fee"].sum()),
                        "roi_per_trade":    float(fr["pnl"].sum() / n),
                        "mean_edge":        mean_edge,
                        "mean_entry_price": mean_entry_price,
                    })
                else:
                    fixed_rows.append({
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades": 0, "hit_rate": np.nan,
                        "total_pnl": 0.0, "total_fees": 0.0,
                        "roi_per_trade": np.nan,
                        "mean_edge": mean_edge,
                        "mean_entry_price": mean_entry_price,
                    })

                # --- half-Kelly ---
                hk = pd.DataFrame(run_kelly_ideal(ents, 0.5, BANKROLL_INIT))
                if len(hk):
                    hk = add_trade_returns(hk)
                    fb = float(hk["bankroll"].iloc[-1])
                    dd = float((hk["bankroll"].cummax() - hk["bankroll"]).max())
                    hk_rows.append({
                        "model": model_name, "edge_min": em, "norm_min": nm,
                        "trades":              len(hk),
                        "hit_rate":            float(hk["won"].mean()),
                        "mean_kelly_fraction": float(hk["kelly_f"].mean()),
                        "mean_log_return":     float(hk["log_ret"].mean()),
                        "final_bankroll":      fb,
                        "total_return":        (fb - BANKROLL_INIT) / BANKROLL_INIT,
                        "max_drawdown":        dd,
                        "mean_edge":           mean_edge,
                        "mean_entry_price":    mean_entry_price,
                    })
                else:
                    hk_rows.append({
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

    fixed_df = pd.DataFrame(fixed_rows).sort_values(["model", "edge_min", "norm_min"]).reset_index(drop=True)
    hk_df    = pd.DataFrame(hk_rows).sort_values(["model", "edge_min", "norm_min"]).reset_index(drop=True)
    return fixed_df, hk_df


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #

def best_per_model(df: pd.DataFrame, metric: str, ascending: bool = False):
    """Return dict {model: (edge_min, norm_min)} of the top row per model."""
    out = {}
    for model_name in df["model"].unique():
        sub = df[df["model"] == model_name].dropna(subset=[metric])
        if sub.empty:
            continue
        idx = sub[metric].idxmax() if not ascending else sub[metric].idxmin()
        r = df.loc[idx]
        out[model_name] = (float(r["edge_min"]), float(r["norm_min"]))
    return out


def validate(df: pd.DataFrame, metric: str, expected: dict, label: str):
    actual = best_per_model(df, metric)
    mismatches = []
    for model, want in expected.items():
        got = actual.get(model)
        if got != want:
            mismatches.append((model, got, want))
    if mismatches:
        print(f"[!] {label}: best-row mismatch — main text expects "
              f"{expected}, but got {actual}")
        print(f"    Top 5 by {metric}:")
        cols = ["model", "edge_min", "norm_min", metric]
        print(df.sort_values(metric, ascending=False).head(5)[cols].to_string(index=False))
    else:
        print(f"[ok] {label}: best rows match main text  -> {actual}")
    return actual


# --------------------------------------------------------------------------- #
# Rounding                                                                    #
# --------------------------------------------------------------------------- #

def round_fixed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("hit_rate", "roi_per_trade", "mean_edge", "mean_entry_price"):
        out[c] = out[c].round(3)
    for c in ("total_pnl", "total_fees"):
        out[c] = out[c].round(2)
    return out


def round_half_kelly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ("hit_rate", "total_return", "mean_kelly_fraction",
              "mean_edge", "mean_entry_price"):
        out[c] = out[c].round(3)
    out["mean_log_return"] = out["mean_log_return"].round(4)
    for c in ("final_bankroll", "max_drawdown"):
        out[c] = out[c].round(2)
    return out


# --------------------------------------------------------------------------- #
# PNG renderers                                                               #
# --------------------------------------------------------------------------- #

def _render(df: pd.DataFrame, *, columns: list[str], headers: list[str],
            best_keys: dict, save_name: str, title: str, footer: str,
            row_height: float = 0.30, fig_width: float = 12.0):
    """Render two model panels stacked vertically, highlighting the selected
    best row in each."""
    plt.rcParams.update({"font.size": 10, "figure.dpi": 200})

    panels = []
    for m in df["model"].unique():
        sub = df[df["model"] == m].reset_index(drop=True)
        panels.append((m, sub))

    rows_per_panel = max(len(s) for _, s in panels)
    fig_height = 1.6 + len(panels) * (rows_per_panel + 1) * row_height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    fig.suptitle(title, fontweight="bold", fontsize=12, y=0.985)

    # Place each panel vertically.
    panel_total = sum((len(s) + 2) for _, s in panels)
    y_cursor = 0.92
    panel_h = (0.92 - 0.10) / len(panels)

    def _fmt_cell(val, col):
        if pd.isna(val):
            return "—"
        if col == "trades":
            return f"{int(val)}"
        if col in ("total_pnl", "total_fees", "final_bankroll", "max_drawdown"):
            return f"${val:,.2f}"
        if col == "mean_log_return":
            return f"{val:+.4f}"
        if col in ("total_return", "roi_per_trade"):
            return f"{val:+.3f}"
        return f"{val:.3f}"

    for i, (model, sub) in enumerate(panels):
        body = []
        for _, r in sub.iterrows():
            body.append([_fmt_cell(r[c], c) for c in columns])
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

        # Style header / row-label / best row
        best = best_keys.get(model)
        for (rr, cc), cell in tbl.get_celld().items():
            if rr == 0:
                cell.set_facecolor("#dfe6ed"); cell.set_text_props(weight="bold")
            elif cc == -1:
                cell.set_facecolor("#f2f5f8"); cell.set_text_props(weight="bold")
            else:
                # rr is 1-indexed for body
                row_data = sub.iloc[rr - 1]
                if best and (round(float(row_data["edge_min"]), 4) == best[0]
                             and round(float(row_data["norm_min"]), 4) == best[1]):
                    cell.set_facecolor("#fff2cc")
                    cell.set_text_props(weight="bold")
        # Panel title above
        ax.text(0.06, bb_top + 0.005, f"{model.replace('_', ' ').title()}",
                fontweight="bold", fontsize=11)
        y_cursor -= panel_h

    fig.text(0.5, 0.03, footer, ha="center", fontsize=8.2,
             color="#444444", wrap=True)
    fig.text(0.5, 0.005, "Row labels: edge_min / norm_min · "
             "selected best row highlighted in yellow.",
             ha="center", fontsize=7.5, color="#666666")
    save_fig(fig, save_name)
    plt.close(fig)


def render_fixed(df: pd.DataFrame, best: dict) -> None:
    cols = ["trades", "hit_rate", "total_pnl", "total_fees",
            "roi_per_trade", "mean_edge", "mean_entry_price"]
    headers = ["Trades", "Hit rate", "Total PnL", "Total fees",
               "ROI / trade", "Mean edge", "Mean entry px"]
    _render(df, columns=cols, headers=headers, best_keys=best,
            save_name="trade_fixed_risk_threshold_grid",
            title="Fixed-Risk Threshold Grid Search",
            footer=NOTE, row_height=0.30)


def render_half_kelly(df: pd.DataFrame, best: dict) -> None:
    cols = ["trades", "hit_rate", "mean_kelly_fraction", "mean_log_return",
            "final_bankroll", "total_return", "max_drawdown",
            "mean_edge", "mean_entry_price"]
    headers = ["Trades", "Hit rate", "Mean Kelly f", "Mean log ret",
               "Final bankroll", "Total return", "Max drawdown",
               "Mean edge", "Mean entry px"]
    _render(df, columns=cols, headers=headers, best_keys=best,
            save_name="trade_half_kelly_threshold_grid",
            title="Half-Kelly Threshold Grid Search",
            footer=NOTE, row_height=0.30, fig_width=14.0)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Loading cached holdout predictions …")
    signals, idx = build_signals_and_index()
    print(f"  signals: {len(signals)} games  |  matched tickers: "
          f"{len(idx['ticker_info'])}  |  with candles: {len(idx['pretip'])}")

    print("Running threshold grid …")
    fixed_df, hk_df = run_grid(idx["ticker_info"], idx["pretip"])

    fixed_round = round_fixed(fixed_df)
    hk_round    = round_half_kelly(hk_df)

    save_table(fixed_round, "trade_fixed_risk_threshold_grid")
    save_table(hk_round,    "trade_half_kelly_threshold_grid")

    print()
    fixed_best = validate(fixed_df, "roi_per_trade",   EXPECTED_BEST["fixed"],      "fixed-risk")
    hk_best    = validate(hk_df,    "final_bankroll", EXPECTED_BEST["half_kelly"], "half-Kelly")

    render_fixed(fixed_round, best=fixed_best)
    render_half_kelly(hk_round, best=hk_best)


if __name__ == "__main__":
    main()
