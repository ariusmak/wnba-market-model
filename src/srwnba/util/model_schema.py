"""
Canonical production model-input schema.

Full feature-building tables may track every listed player and audit fields.
The XGBoost input table is intentionally narrower: metadata plus the 160
ordinary model features locked in config/final_hyperparams.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_config = Path(__file__).resolve().parents[3] / "config"
if str(_config) not in sys.path:
    sys.path.insert(0, str(_config))

from final_hyperparams import N_PLAYERS

PLAYER_FEATS = [
    "m_ewma_pre",
    "q_pre",
    "days_since_first_report_pre",
    "days_since_last_dnp_pre",
    "consec_dnps_pre",
    "played_last_game_pre",
    "minutes_last_game_pre",
    "days_since_last_played_pre",
    "injury_present_flag_pre",
]

FORM_FEATS = [
    "net_rtg_ewma_pre",
    "efg_ewma_pre",
    "tov_pct_ewma_pre",
    "orb_pct_ewma_pre",
    "ftr_ewma_pre",
]

STYLE_FEATS = [
    "off_3pa_rate_pre",
    "def_3pa_allowed_pre",
    "off_2pa_rate_pre",
    "def_2pa_allowed_pre",
    "off_tov_pct_pre",
    "def_forced_tov_pre",
]

SCHED_FEATS = [
    "days_rest_pre",
    "is_b2b_pre",
    "games_last_4_days_pre",
    "games_last_7_days_pre",
    "travel_miles_pre",
    "timezone_shift_hours_pre",
]

METADATA_COLS = [
    "game_id",
    "game_ts",
    "game_date",
    "season",
    "is_playoff",
    "home_team_id",
    "away_team_id",
    "home_franchise_id",
    "away_franchise_id",
    "home_elo_pre",
    "away_elo_pre",
    "p_elo",
    "base_margin",
    "home_win",
]

LABEL_COL = "home_win"
ELO_PROB_COL = "p_elo"


def build_feature_cols(n_players: int = N_PLAYERS) -> list[str]:
    cols: list[str] = []
    for side in ("home", "away"):
        for slot in range(1, n_players + 1):
            for feat in PLAYER_FEATS:
                cols.append(f"{side}_p{slot}_{feat}")
    for feat in FORM_FEATS + STYLE_FEATS + SCHED_FEATS:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")
    return cols


FEAT_COLS = build_feature_cols(N_PLAYERS)
GOLD_MODEL_INPUT_COLS = METADATA_COLS + FEAT_COLS
