"""Generate `feature_family_summary_stats.{csv,png}`.

Appendix-ready summary statistics of the 160-dim XGBoost feature matrix,
collapsed by feature family. Player families pool across home/away and the
seven player slots (14 columns); team-level families pool across home/away
(2 columns). Total represented columns = 9*14 + 5*2 + 6*2 + 6*2 = 160.

Scope: training feature matrix for the final XGBoost model — 2015–2024
gold tables, cold-start rows excluded (matching the model's actual training
input). Outputs land in `organized/outputs/`.
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
from final_model import (  # noqa: E402
    FEAT_COLS, N_PLAYERS, PLAYER_FEATS, FORM_FEATS, STYLE_FEATS, SCHED_FEATS,
    load_year,
)

TRAIN_YEARS = list(range(2015, 2025))  # the model's training range


# Pretty labels
PLAYER_LABELS = {
    "m_ewma_pre":                 "Player minutes EWMA",
    "q_pre":                      "Player quality per minute",
    "days_since_first_report_pre":"Days since first injury report",
    "days_since_last_dnp_pre":    "Days since last DNP",
    "consec_dnps_pre":            "Consecutive DNPs",
    "played_last_game_pre":       "Played last game",
    "minutes_last_game_pre":      "Minutes last game",
    "days_since_last_played_pre": "Days since last played",
    "injury_present_flag_pre":    "Injury present flag",
}
FORM_LABELS = {
    "net_rtg_ewma_pre": "Net rating EWMA",
    "efg_ewma_pre":     "eFG% EWMA",
    "tov_pct_ewma_pre": "Turnover% EWMA",
    "orb_pct_ewma_pre": "Offensive rebound% EWMA",
    "ftr_ewma_pre":     "Free throw rate EWMA",
}
STYLE_LABELS = {
    "off_3pa_rate_pre":    "Offensive 3PA rate",
    "def_3pa_allowed_pre": "Defensive 3PA allowed rate",
    "off_2pa_rate_pre":    "Offensive 2PA rate",
    "def_2pa_allowed_pre": "Defensive 2PA allowed rate",
    "off_tov_pct_pre":     "Offensive turnover%",
    "def_forced_tov_pre":  "Defensive forced turnover%",
}
SCHED_LABELS = {
    "days_rest_pre":            "Days rest",
    "is_b2b_pre":               "Back-to-back indicator",
    "games_last_4_days_pre":    "Games last 4 days",
    "games_last_7_days_pre":    "Games last 7 days",
    "travel_miles_pre":         "Travel miles",
    "timezone_shift_hours_pre": "Timezone shift",
}

BLOCK_ORDER = [
    "Player injury / availability",
    "Recent team form",
    "Team stylistic profile",
    "Rest / travel context",
]


def player_columns(suffix: str) -> list[str]:
    return [
        f"{side}_p{k}_{suffix}"
        for side in ("home", "away")
        for k in range(1, N_PLAYERS + 1)
    ]


def team_columns(suffix: str) -> list[str]:
    return [f"home_{suffix}", f"away_{suffix}"]


# --------------------------------------------------------------------------- #
# Build feature matrix                                                        #
# --------------------------------------------------------------------------- #

def load_feature_matrix() -> pd.DataFrame:
    frames = []
    for y in TRAIN_YEARS:
        df = load_year(y)
        present = [c for c in FEAT_COLS if c in df.columns]
        frames.append(df[present])
    X = pd.concat(frames, ignore_index=True)
    return X


# --------------------------------------------------------------------------- #
# Stack-and-summarize                                                         #
# --------------------------------------------------------------------------- #

def summarize_stack(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float).ravel()
    total = len(arr)
    miss = np.isnan(arr).sum()
    a = arr[~np.isnan(arr)]
    if a.size == 0:
        return {
            "mean": np.nan, "std": np.nan,
            "median": np.nan, "q1": np.nan, "q3": np.nan,
            "zero_rate": np.nan, "missing_rate": (miss / total if total else np.nan),
        }
    return {
        "mean":         float(a.mean()),
        "std":          float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median":       float(np.median(a)),
        "q1":           float(np.percentile(a, 25)),
        "q3":           float(np.percentile(a, 75)),
        "zero_rate":    float((a == 0).mean()),
        "missing_rate": float(miss / total),
    }


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #

def _fmt_num(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:.4f}" if abs(v) < 0.01 and v != 0 else f"{v:.3f}"


def _fmt_pct(v: float) -> str:
    return "—" if pd.isna(v) else f"{v * 100:.1f}%"


def _fmt_median_iqr(stats: dict) -> str:
    if pd.isna(stats["median"]):
        return "—"
    return f"{_fmt_num(stats['median'])} [{_fmt_num(stats['q1'])}, {_fmt_num(stats['q3'])}]"


# --------------------------------------------------------------------------- #
# Build table                                                                 #
# --------------------------------------------------------------------------- #

def build_summary(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    represented: list[str] = []

    # 1. Player block
    for suf, label in PLAYER_LABELS.items():
        cols = player_columns(suf)
        present = [c for c in cols if c in X.columns]
        represented.extend(present)
        stats = summarize_stack(X[present].to_numpy())
        rows.append({
            "Feature family":      label,
            "Block":               "Player injury / availability",
            "Columns represented": len(present),
            **stats,
        })

    # 2. Recent team form
    for suf, label in FORM_LABELS.items():
        cols = team_columns(suf)
        present = [c for c in cols if c in X.columns]
        represented.extend(present)
        stats = summarize_stack(X[present].to_numpy())
        rows.append({
            "Feature family":      label,
            "Block":               "Recent team form",
            "Columns represented": len(present),
            **stats,
        })

    # 3. Team stylistic profile
    for suf, label in STYLE_LABELS.items():
        cols = team_columns(suf)
        present = [c for c in cols if c in X.columns]
        represented.extend(present)
        stats = summarize_stack(X[present].to_numpy())
        rows.append({
            "Feature family":      label,
            "Block":               "Team stylistic profile",
            "Columns represented": len(present),
            **stats,
        })

    # 4. Rest / travel context
    for suf, label in SCHED_LABELS.items():
        cols = team_columns(suf)
        present = [c for c in cols if c in X.columns]
        represented.extend(present)
        stats = summarize_stack(X[present].to_numpy())
        rows.append({
            "Feature family":      label,
            "Block":               "Rest / travel context",
            "Columns represented": len(present),
            **stats,
        })

    df = pd.DataFrame(rows)
    df["Block"] = pd.Categorical(df["Block"], categories=BLOCK_ORDER, ordered=True)
    df = df.sort_values(["Block", "Feature family"], kind="stable").reset_index(drop=True)
    return df, represented


# --------------------------------------------------------------------------- #
# CSV / PNG outputs                                                           #
# --------------------------------------------------------------------------- #

def make_display_df(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Feature family":      stats_df["Feature family"],
        "Block":               stats_df["Block"].astype(str),
        "Columns represented": stats_df["Columns represented"].astype(int),
        "Mean":                stats_df["mean"].apply(_fmt_num),
        "Std. dev.":           stats_df["std"].apply(_fmt_num),
        "Median [Q1, Q3]":     stats_df.apply(_fmt_median_iqr, axis=1),
        "Zero rate":           stats_df["zero_rate"].apply(_fmt_pct),
        "Missing rate":        stats_df["missing_rate"].apply(_fmt_pct),
    })
    return out


def render_png(display_df: pd.DataFrame, *, save_name: str) -> None:
    plt.rcParams.update({"font.size": 9, "figure.dpi": 200})

    n_rows = len(display_df)
    fig_height = 1.6 + n_rows * 0.30
    fig, ax = plt.subplots(figsize=(13.0, fig_height))
    ax.axis("off")

    fig.suptitle("Summary Statistics by Feature Family", fontweight="bold",
                 fontsize=12, y=0.985)

    cell_text = display_df.values.tolist()
    headers = list(display_df.columns)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="left", rowLoc="center",
        bbox=[0.02, 0.10, 0.96, 0.82],
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)

    block_colors = {
        "Player injury / availability": "#eef4fa",
        "Recent team form":              "#eaf6ee",
        "Team stylistic profile":        "#fdf1e6",
        "Rest / travel context":         "#f3ecf8",
    }
    for (rr, cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_facecolor("#dfe6ed")
            cell.set_text_props(weight="bold")
        else:
            block = display_df.iloc[rr - 1]["Block"]
            cell.set_facecolor(block_colors.get(block, "#ffffff"))
            if cc == 1:  # block column
                cell.set_text_props(style="italic")
        if cc in (0, 1):
            cell.set_text_props(ha="left")
        else:
            cell.set_text_props(ha="center")

    note = (
        "Note. Player feature families collapse across seven player slots and "
        "both home/away teams, yielding 14 columns per family. Team-level "
        "families collapse across home and away versions, yielding two "
        "columns per family. For injury-window features, zero values include "
        "inactive injury windows because these features are set to zero when "
        "no active injury window is present."
    )
    fig.text(0.02, 0.045, note, ha="left", fontsize=8.0,
             color="#444444", wrap=True)
    fig.text(0.02, 0.012,
             f"Scope: 2015–2024 training feature matrix · "
             f"{TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} · "
             "cold-start rows excluded.",
             ha="left", fontsize=7.5, color="#666666")
    save_fig(fig, save_name)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main() -> None:
    print(f"Loading feature matrix for {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]} …")
    X = load_feature_matrix()
    print(f"  rows: {len(X):,}  |  columns present: {len(X.columns)}")

    stats_df, represented = build_summary(X)

    # --- validation ---
    expected = 9 * 14 + 5 * 2 + 6 * 2 + 6 * 2  # = 160
    total = int(stats_df["Columns represented"].sum())
    if total != expected:
        missing = sorted(set(FEAT_COLS) - set(represented))
        unmatched = sorted(set(represented) - set(FEAT_COLS))
        print(f"[!] columns represented = {total}, expected {expected}")
        if missing:
            print(f"    missing from any family: {missing}")
        if unmatched:
            print(f"    matched but not in FEAT_COLS: {unmatched}")
        raise SystemExit(1)
    print(f"[ok] columns represented = {total}/{expected}")

    display_df = make_display_df(stats_df)
    save_table(display_df, "feature_family_summary_stats")
    render_png(display_df, save_name="feature_family_summary_stats")
    print()
    print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
