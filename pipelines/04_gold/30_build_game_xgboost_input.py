"""
30_build_game_xgboost_input.py

Assembles the final one-row-per-game XGBoost input table per spec:
  data/spec_sheets/game_xgboost_input_spec.md

Inputs (all from data/silver_plus/ or data/silver/):
  elo_franchise_team_game_{year}_REGPST.csv
  game_franchise_recent_form_{year}_REGPST.csv
  game_franchise_style_profile_{year}_REGPST.csv
  game_team_schedule_context_{year}_REGPST.csv
  game_team_player_{year}_REGPST.csv     (from data/silver/)
  game_outcomes_{year}_REGPST.csv        (from data/silver/, for home_win target)

Outputs:
  data/gold/game_xgboost_input_{year}_REGPST.csv       (one per year)
  data/gold/game_xgboost_input_2015_2024_REGPST.csv    (combined training set)

Column order (per spec section 9):
  1. metadata block
  2. home player slots p1-p12 (debug + 9 model features each)
  3. away player slots p1-p12
  4. recent form block (home, away)
  5. style profile block (home, away)
  6. schedule/travel block (home, away)
"""
import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SLOTS = 12

PLAYER_MODEL_FEATURES = [
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

PLAYER_DEBUG_FEATURES = ["player_id", "player_name", "strength_pre"]

RECENT_FORM_FEATURES = [
    "net_rtg_ewma_pre",
    "efg_ewma_pre",
    "tov_pct_ewma_pre",
    "orb_pct_ewma_pre",
    "ftr_ewma_pre",
]

STYLE_FEATURES = [
    "off_3pa_rate_pre",
    "def_3pa_allowed_pre",
    "off_2pa_rate_pre",
    "def_2pa_allowed_pre",
    "off_tov_pct_pre",
    "def_forced_tov_pre",
]

SCHEDULE_FEATURES = [
    "days_rest_pre",
    "is_b2b_pre",
    "games_last_4_days_pre",
    "games_last_7_days_pre",
    "travel_miles_pre",
    "timezone_shift_hours_pre",
]

SCHEDULE_META = ["origin_city_pre", "current_city_pre"]

P_CLIP = 1e-6


def logit(p: float) -> float:
    p = max(P_CLIP, min(1.0 - P_CLIP, p))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_elo(year: int) -> pd.DataFrame:
    p = Path(f"data/silver_plus/elo_franchise_team_game_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    df["game_id"] = df["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    return df


def load_recent_form(year: int) -> pd.DataFrame:
    p = Path(f"data/silver_plus/game_franchise_recent_form_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p, usecols=["game_id", "team_id"] + RECENT_FORM_FEATURES)
    df["game_id"] = df["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    return df


def load_style(year: int) -> pd.DataFrame:
    p = Path(f"data/silver_plus/game_franchise_style_profile_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p, usecols=["game_id", "team_id"] + STYLE_FEATURES)
    df["game_id"] = df["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    return df


def load_schedule(year: int) -> pd.DataFrame:
    p = Path(f"data/silver_plus/game_team_schedule_context_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p, usecols=["game_id", "team_id"] + SCHEDULE_FEATURES + SCHEDULE_META)
    df["game_id"] = df["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    return df


def load_players(year: int) -> pd.DataFrame:
    p = Path(f"data/silver/game_team_player_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    cols = ["game_id", "team_id"] + PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES
    df = pd.read_csv(p, usecols=cols)
    df["game_id"] = df["game_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df["player_id"] = df["player_id"].astype(str)
    return df


def load_home_win(year: int) -> pd.DataFrame:
    p = Path(f"data/silver/game_outcomes_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p, usecols=["game_id", "home_win"])
    df["game_id"] = df["game_id"].astype(str)
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Player slot builder
# ---------------------------------------------------------------------------

def build_player_slots(players: pd.DataFrame, game_id: str, team_id: str) -> dict:
    """
    Sort players for one (game_id, team_id) by ranking rule and return flat slot dict.
    Missing slots (< 12 players) are NULL.
    """
    grp = players[(players["game_id"] == game_id) & (players["team_id"] == team_id)].copy()

    grp = grp.sort_values(
        ["strength_pre", "m_ewma_pre", "q_pre", "player_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )

    out = {}
    for slot in range(1, N_SLOTS + 1):
        prefix = f"p{slot}_"
        if slot <= len(grp):
            row = grp.iloc[slot - 1]
            for col in PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES:
                val = row[col]
                out[prefix + col] = None if pd.isna(val) else val
        else:
            for col in PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES:
                out[prefix + col] = None

    return out


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def main(year: int):
    elo = load_elo(year)
    form = load_recent_form(year)
    style = load_style(year)
    sched = load_schedule(year)
    players = load_players(year)
    outcomes = load_home_win(year)

    # Step 1: pivot Elo to one row per game
    home_elo = elo[elo["is_home"] == 1].set_index("game_id")
    away_elo = elo[elo["is_home"] == 0].set_index("game_id")

    game_ids = sorted(set(home_elo.index) & set(away_elo.index))
    print(f"  {year}: {len(game_ids)} games to assemble")

    # Pre-index feature tables for fast lookup
    form_idx = form.set_index(["game_id", "team_id"])
    style_idx = style.set_index(["game_id", "team_id"])
    sched_idx = sched.set_index(["game_id", "team_id"])
    outcomes_idx = outcomes.set_index("game_id")

    rows = []

    for gid in game_ids:
        h = home_elo.loc[gid]
        a = away_elo.loc[gid]

        home_tid = str(h["team_id"])
        away_tid = str(a["team_id"])
        home_fid = str(h["franchise_id"])
        away_fid = str(a["franchise_id"])

        p_elo = float(h["p_win_pre"])
        bm = logit(p_elo)

        # metadata
        row: dict = {
            "game_id": gid,
            "game_ts": h.get("scheduled"),
            "game_date": pd.to_datetime(h.get("scheduled"), utc=True, errors="coerce").date()
                         if pd.notna(h.get("scheduled")) else None,
            "season": int(h.get("season_year", year)),
            "is_playoff": int(h.get("is_playoff", 0)) if "is_playoff" in h.index else None,
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "home_franchise_id": home_fid,
            "away_franchise_id": away_fid,
            "home_elo_pre": float(h["elo_pre"]),
            "away_elo_pre": float(a["elo_pre"]),
            "p_elo": p_elo,
            "base_margin": bm,
            "home_win": int(outcomes_idx.loc[gid, "home_win"])
                        if gid in outcomes_idx.index and pd.notna(outcomes_idx.loc[gid, "home_win"])
                        else None,
        }

        # Steps 2-3: player slots (home then away)
        home_slots = build_player_slots(players, gid, home_tid)
        away_slots = build_player_slots(players, gid, away_tid)
        for k, v in home_slots.items():
            row[f"home_{k}"] = v
        for k, v in away_slots.items():
            row[f"away_{k}"] = v

        # Step 4: recent form
        def get_form(tid: str, feat: str):
            key = (gid, tid)
            return form_idx.loc[key, feat] if key in form_idx.index else None

        for f in RECENT_FORM_FEATURES:
            row[f"home_{f}"] = get_form(home_tid, f)
            row[f"away_{f}"] = get_form(away_tid, f)

        # Step 5: style profile
        def get_style(tid: str, feat: str):
            key = (gid, tid)
            return style_idx.loc[key, feat] if key in style_idx.index else None

        for f in STYLE_FEATURES:
            row[f"home_{f}"] = get_style(home_tid, f)
            row[f"away_{f}"] = get_style(away_tid, f)

        # Step 6: schedule/travel
        def get_sched(tid: str, feat: str):
            key = (gid, tid)
            return sched_idx.loc[key, feat] if key in sched_idx.index else None

        for f in SCHEDULE_FEATURES:
            row[f"home_{f}"] = get_sched(home_tid, f)
            row[f"away_{f}"] = get_sched(away_tid, f)

        # Schedule metadata (debug only)
        for f in SCHEDULE_META:
            row[f"home_{f}"] = get_sched(home_tid, f)
            row[f"away_{f}"] = get_sched(away_tid, f)

        rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["season", "game_ts", "game_id"], kind="stable").reset_index(drop=True)

    # ---------------------------------------------------------------------------
    # Enforce column order per spec section 8
    # ---------------------------------------------------------------------------
    metadata_cols = [
        "game_id", "game_ts", "game_date", "season", "is_playoff",
        "home_team_id", "away_team_id", "home_franchise_id", "away_franchise_id",
        "home_elo_pre", "away_elo_pre", "p_elo", "base_margin", "home_win",
    ]

    player_cols = []
    for side in ["home", "away"]:
        for slot in range(1, N_SLOTS + 1):
            for col in PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES:
                player_cols.append(f"{side}_p{slot}_{col}")

    form_cols = [f"{s}_{f}" for s in ["home", "away"] for f in RECENT_FORM_FEATURES]
    style_cols = [f"{s}_{f}" for s in ["home", "away"] for f in STYLE_FEATURES]
    sched_cols = [f"{s}_{f}" for s in ["home", "away"] for f in SCHEDULE_FEATURES]
    sched_meta_cols = [f"{s}_{f}" for s in ["home", "away"] for f in SCHEDULE_META]

    final_cols = metadata_cols + player_cols + form_cols + style_cols + sched_cols + sched_meta_cols
    out = out[[c for c in final_cols if c in out.columns]]

    out_dir = Path("data/gold")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"game_xgboost_input_{year}_REGPST.csv"
    out.to_csv(out_path, index=False)

    # Quick stats
    n_games = len(out)
    null_form = out["home_net_rtg_ewma_pre"].isna().sum()
    null_sched = out["home_days_rest_pre"].isna().sum()
    null_player = out["home_p1_m_ewma_pre"].isna().sum()
    null_target = out["home_win"].isna().sum()
    n_cols = len(out.columns)

    print(f"  wrote: {out_path}  rows={n_games}  cols={n_cols}")
    print(f"    null_form={null_form}  null_sched={null_sched}  "
          f"null_p1={null_player}  null_target={null_target}")

    return out


def main_range(start_year: int, end_year: int):
    Path("data/gold").mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for y in range(start_year, end_year + 1):
        print(f"=== {y} ===")
        df = main(y)
        all_dfs.append(df)

    # Combined 2015-2024 training set (exclude 2025 as future/live year)
    train_dfs = [df for df in all_dfs if int(df["season"].iloc[0]) <= 2024]
    combined = pd.concat(train_dfs, ignore_index=True)
    combined_path = Path("data/gold") / "game_xgboost_input_2015_2024_REGPST.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nwrote combined: {combined_path}  rows={len(combined)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()
    main_range(args.start_year, args.end_year)
