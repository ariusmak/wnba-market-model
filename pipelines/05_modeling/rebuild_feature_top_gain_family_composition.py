"""Generate the feature-family composition figure + top-10 comparison table.

Reads cached normalized-gain importance from:
    organized/outputs/feature_importance_xgb_with_elo.csv
    organized/outputs/feature_importance_xgb_no_elo.csv

Produces:
    organized/outputs/feature_top_gain_family_composition.png
    organized/outputs/feature_top10_gain_comparison_table.csv

Family classification follows the paper's narrative grouping:

  Team strength       net_rtg_ewma, q_pre, m_ewma_pre
                      (treated as team-quality proxies, regardless of which
                      block they sit in structurally)
  Availability/injury played_last_game, days_since_last_played,
                      days_since_first_report, days_since_last_dnp,
                      injury_present_flag, consec_dnps, minutes_last_game
  Rest/schedule/travel days_rest, is_b2b, games_last_*_days, travel_miles,
                      timezone_shift
  Style               3pa_rate, 3pa_allowed, 2pa_rate, 2pa_allowed,
                      off_tov_pct, forced_tov
  Recent form         efg_ewma, tov_pct_ewma, orb_pct_ewma, ftr_ewma
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ANALYSIS_DIR = Path(__file__).resolve().parents[1].parent / "src" / "srwnba" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from outputs import OUTPUTS_DIR, save_fig, save_table  # noqa: E402

FAMILY_ORDER = [
    "Team strength",
    "Availability / injury",
    "Rest / schedule / travel",
    "Style",
    "Recent form",
    "Other",
]
FAMILY_COLORS = {
    "Team strength":            "#3498db",
    "Availability / injury":    "#e74c3c",
    "Rest / schedule / travel": "#9b59b6",
    "Style":                    "#e67e22",
    "Recent form":              "#2ecc71",
    "Other":                    "#95a5a6",
}


def classify_family(name: str) -> str:
    # Team-strength proxies
    if "net_rtg_ewma" in name:
        return "Team strength"
    if name.endswith("_q_pre"):
        return "Team strength"
    if name.endswith("_m_ewma_pre"):
        return "Team strength"
    # Availability / injury
    avail_tokens = (
        "played_last_game", "days_since_last_played", "days_since_first_report",
        "days_since_last_dnp", "injury_present_flag", "consec_dnps",
        "minutes_last_game",
    )
    if any(tok in name for tok in avail_tokens):
        return "Availability / injury"
    # Rest / schedule / travel
    sched_tokens = (
        "days_rest", "is_b2b", "games_last_4_days", "games_last_7_days",
        "travel_miles", "timezone_shift",
    )
    if any(tok in name for tok in sched_tokens):
        return "Rest / schedule / travel"
    # Style
    style_tokens = (
        "3pa_rate", "3pa_allowed", "2pa_rate", "2pa_allowed",
        "off_tov_pct", "forced_tov",
    )
    if any(tok in name for tok in style_tokens):
        return "Style"
    # Recent form (remaining form-block features)
    form_tokens = ("efg_ewma", "tov_pct_ewma", "orb_pct_ewma", "ftr_ewma")
    if any(tok in name for tok in form_tokens):
        return "Recent form"
    return "Other"


def top_n_with_family(path: Path, n: int) -> pd.DataFrame:
    fi = pd.read_csv(path)
    fi = fi.sort_values("gain_norm", ascending=False).reset_index(drop=True)
    fi = fi.head(n).copy()
    fi["family"] = fi["feature"].apply(classify_family)
    return fi


def family_shares(top: pd.DataFrame) -> pd.Series:
    """Share of the top-N gain that goes to each family (sums to 1)."""
    grouped = top.groupby("family")["gain_norm"].sum()
    total = grouped.sum()
    shares = grouped / total if total > 0 else grouped
    return shares.reindex(FAMILY_ORDER, fill_value=0.0)


def plot_composition(no_elo_top: pd.DataFrame, with_elo_top: pd.DataFrame) -> plt.Figure:
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "figure.dpi": 120})

    shares_no  = family_shares(no_elo_top)
    shares_yes = family_shares(with_elo_top)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    bar_labels = ["XGBoost without Elo", "XGBoost with Elo base margin"]
    y = [1, 0]  # display "no Elo" on top so visual flow reads top → bottom

    for i, shares in enumerate([shares_no, shares_yes]):
        left = 0.0
        for fam in FAMILY_ORDER:
            v = float(shares[fam])
            if v <= 0:
                continue
            ax.barh(y[i], v, left=left, color=FAMILY_COLORS[fam],
                    edgecolor="white", linewidth=1.2, height=0.55)
            if v >= 0.04:
                ax.text(left + v / 2, y[i], f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
            left += v

    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of top-30 normalized gain")
    ax.set_title("Composition of Top-Gain Features by Model Specification",
                 fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", alpha=0.25)

    legend_handles = [
        Patch(facecolor=FAMILY_COLORS[f], label=f)
        for f in FAMILY_ORDER if f != "Other"
        or (shares_no[f] > 0 or shares_yes[f] > 0)
    ]
    ax.legend(handles=legend_handles, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=min(6, len(legend_handles)), frameon=False)
    plt.tight_layout()
    return fig


def build_top10_table(no_elo_top: pd.DataFrame, with_elo_top: pd.DataFrame,
                      n: int = 10) -> pd.DataFrame:
    no_elo  = no_elo_top.head(n).reset_index(drop=True)
    with_elo = with_elo_top.head(n).reset_index(drop=True)
    return pd.DataFrame({
        "rank":                         range(1, n + 1),
        "xgb_no_elo_feature":           no_elo["feature"].values,
        "xgb_no_elo_family":            no_elo["family"].values,
        "xgb_no_elo_gain_norm":         no_elo["gain_norm"].round(4).values,
        "xgb_with_elo_feature":         with_elo["feature"].values,
        "xgb_with_elo_family":          with_elo["family"].values,
        "xgb_with_elo_gain_norm":       with_elo["gain_norm"].round(4).values,
    })


def main() -> None:
    no_elo_top   = top_n_with_family(OUTPUTS_DIR / "feature_importance_xgb_no_elo.csv",   30)
    with_elo_top = top_n_with_family(OUTPUTS_DIR / "feature_importance_xgb_with_elo.csv", 30)

    fig = plot_composition(no_elo_top, with_elo_top)
    save_fig(fig, "feature_top_gain_family_composition")
    plt.close(fig)

    table = build_top10_table(no_elo_top, with_elo_top, n=10)
    save_table(table, "feature_top10_gain_comparison_table")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
