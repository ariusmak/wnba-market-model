import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def league_q_prev_season(year: int) -> float:
    """
    Option A: league-average prior quality from previous season:
      q_league_prev = total_EFF_prev / total_MIN_prev
    If prev season file doesn't exist (e.g., 2015), return 0.0.
    """
    prev_path = Path(f"data/silver/player_game_box_{year-1}_REGPST.csv")
    if year <= 2015 or not prev_path.exists():
        return 0.0

    df = pd.read_csv(prev_path)
    df["minutes"] = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0.0)
    df["eff"] = pd.to_numeric(df.get("eff"), errors="coerce").fillna(0.0)

    denom = float(df["minutes"].sum())
    if denom <= 0:
        return 0.0
    return float(df["eff"].sum() / denom)


def asof_join_player_state_strict(roster: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """
    Robust strict as-of join per player (avoids pandas by= sortedness issues):
      for each player_id:
        join latest state.asof_ts strictly < game_ts
    """
    out_parts = []

    # prep
    roster = roster.copy()
    roster["player_id"] = roster["player_id"].astype(str)
    roster["game_ts"] = pd.to_datetime(roster["game_ts"], utc=True, errors="coerce")

    state = state.copy()
    state["player_id"] = state["player_id"].astype(str)
    state["asof_ts"] = pd.to_datetime(state["asof_ts"], utc=True, errors="coerce")

    # de-dupe state PK just in case
    state = state.dropna(subset=["player_id", "asof_ts"]).drop_duplicates(
        subset=["player_id", "asof_ts"], keep="last"
    )

    # group dict for speed
    state_groups = {pid: g.sort_values("asof_ts", kind="stable") for pid, g in state.groupby("player_id", sort=False)}

    for pid, g in roster.groupby("player_id", sort=False):
        g2 = g.sort_values("game_ts", kind="stable").copy()
        sg = state_groups.get(pid)

        if sg is None or len(sg) == 0:
            # no state for this player -> keep rows with NaNs for state cols
            out_parts.append(g2)
            continue

        sg2 = sg.drop(columns=["player_id"])  # avoid player_id_x/player_id_y
        merged = pd.merge_asof(
            g2,
            sg2,
            left_on="game_ts",
            right_on="asof_ts",
            direction="backward",
            allow_exact_matches=False,  # STRICT: asof_ts < game_ts
        )
        out_parts.append(merged)

    return pd.concat(out_parts, ignore_index=True)


def main(year: int):
    # Inputs
    roster_path = Path(f"data/silver/game_availability_{year}_REGPST.csv")
    state_path = Path(f"data/silver/player_state_history_{year}.csv")
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")

    if not roster_path.exists():
        raise FileNotFoundError(roster_path)
    if not state_path.exists():
        raise FileNotFoundError(state_path)
    if not played_path.exists():
        raise FileNotFoundError(played_path)

    roster = pd.read_csv(roster_path)
    state = pd.read_csv(state_path)
    played = pd.read_csv(played_path)

    # Filter to played/closed games only (manifest) so we only build real games
    played_ids = set(played["game_id"].astype(str).unique())
    roster["game_id"] = roster["game_id"].astype(str)
    roster = roster[roster["game_id"].isin(played_ids)].copy()

    # Required base columns from roster (derived from game summary)
    # game_ts is the scheduled/tipoff timestamp
    roster["game_ts"] = pd.to_datetime(roster["scheduled"], utc=True, errors="coerce")
    roster["game_date"] = roster["game_ts"].dt.floor("D").dt.date.astype(str)

    # Context
    roster["season"] = year
    if "opponent_id" in roster.columns:
        roster["opponent_team_id"] = roster["opponent_id"]
    else:
        roster["opponent_team_id"] = roster.get("opponent_id")

    # is_home
    if "side" in roster.columns:
        roster["is_home"] = (roster["side"].astype(str).str.lower() == "home").astype(int)
    else:
        # fallback if side missing
        roster["is_home"] = 0

    # is_playoff
    if "season_type" in roster.columns:
        roster["is_playoff"] = (roster["season_type"].astype(str).str.upper() == "PST").astype(int)
    else:
        # fallback: join from played manifest which has season_type
        pm = played[["game_id", "season_type"]].drop_duplicates("game_id")
        roster = roster.merge(pm, on="game_id", how="left")
        roster["is_playoff"] = (roster["season_type"].astype(str).str.upper() == "PST").astype(int)

    # Identity
    roster["player_id"] = roster["player_id"].astype(str)
    roster["team_id"] = roster["team_id"].astype(str)
    roster["opponent_team_id"] = roster["opponent_team_id"].astype(str)

    roster["player_name"] = roster.get("player_name", roster.get("player_name", "")).astype(str)
    roster["listed_on_game_summary_flag"] = 1  # spec redundant flag :contentReference[oaicite:3]{index=3}

    # Realized same-game metadata
    roster["minutes_in_game"] = pd.to_numeric(roster.get("minutes"), errors="coerce").fillna(0.0)
    roster["played_in_game"] = (roster["minutes_in_game"] > 0).astype(int)
    roster["not_playing_reason"] = roster.get("not_playing_reason", "").fillna("").astype(str)
    roster["not_playing_description"] = roster.get("not_playing_description", "").fillna("").astype(str)

    # Dedup to PK
    roster = roster.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last").copy()

    # Prepare roster skeleton with required columns before join
    skeleton = roster[[
        "game_id",
        "game_ts",
        "game_date",
        "season",
        "team_id",
        "opponent_team_id",
        "is_home",
        "is_playoff",
        "player_id",
        "player_name",
        "listed_on_game_summary_flag",
        "minutes_in_game",
        "played_in_game",
        "not_playing_reason",
        "not_playing_description",
    ]].copy()

    # Prepare state columns (rename to *_pre after join)
    # We only need the required upstream columns :contentReference[oaicite:4]{index=4}
    state = state.copy()
    state["player_id"] = state["player_id"].astype(str)

    required_state_cols = [
        "player_id",
        "asof_ts",
        "m_ewma",
        "q",
        "strength",
        "days_since_first_report",
        "days_since_last_dnp",
        "consec_dnps",
        "played_last_game",
        "minutes_last_game",
        "days_since_last_played",
        "injury_present_flag",
    ]
    missing = [c for c in required_state_cols if c not in state.columns]
    if missing:
        raise ValueError(f"player_state_history_{year}.csv missing columns: {missing}")

    state = state[required_state_cols].copy()

    # Join strict as-of: state.asof_ts < game_ts :contentReference[oaicite:5]{index=5}
    joined = asof_join_player_state_strict(skeleton, state)

    # Rename joined state columns to *_pre and keep audit timestamp as state_asof_ts :contentReference[oaicite:6]{index=6}
    joined = joined.rename(columns={"asof_ts": "state_asof_ts"})
    rename_map = {
        "m_ewma": "m_ewma_pre",
        "q": "q_pre",
        "strength": "strength_pre",
        "days_since_first_report": "days_since_first_report_pre",
        "days_since_last_dnp": "days_since_last_dnp_pre",
        "consec_dnps": "consec_dnps_pre",
        "played_last_game": "played_last_game_pre",
        "minutes_last_game": "minutes_last_game_pre",
        "days_since_last_played": "days_since_last_played_pre",
        "injury_present_flag": "injury_present_flag_pre",
    }
    joined = joined.rename(columns=rename_map)

    # Default values if no upstream row exists :contentReference[oaicite:7]{index=7}
    q_default = league_q_prev_season(year)  # Option A
    no_state = joined["state_asof_ts"].isna()

    # Fill defaults (initialize)
    joined.loc[no_state, "m_ewma_pre"] = 0.0
    joined.loc[no_state, "q_pre"] = q_default
    joined.loc[no_state, "strength_pre"] = 0.0

    joined.loc[no_state, "days_since_first_report_pre"] = 0
    joined.loc[no_state, "days_since_last_dnp_pre"] = 0
    joined.loc[no_state, "consec_dnps_pre"] = 0
    joined.loc[no_state, "injury_present_flag_pre"] = 0

    joined.loc[no_state, "played_last_game_pre"] = 0
    joined.loc[no_state, "minutes_last_game_pre"] = 0.0
    joined.loc[no_state, "days_since_last_played_pre"] = 0

    # Ensure numeric dtypes
    for c in ["m_ewma_pre", "q_pre", "strength_pre", "minutes_last_game_pre"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0.0)
    for c in ["days_since_first_report_pre", "days_since_last_dnp_pre", "consec_dnps_pre",
              "played_last_game_pre", "days_since_last_played_pre", "injury_present_flag_pre"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0).astype(int)

    # Recompute strength_pre to guarantee equality strength_pre = m_ewma_pre * q_pre :contentReference[oaicite:8]{index=8}
    joined["strength_pre"] = joined["m_ewma_pre"] * joined["q_pre"]

    # Final schema order per spec :contentReference[oaicite:9]{index=9}
    out_cols = [
        "game_id",
        "game_ts",
        "game_date",
        "season",
        "team_id",
        "opponent_team_id",
        "is_home",
        "is_playoff",
        "player_id",
        "player_name",
        "listed_on_game_summary_flag",
        "state_asof_ts",
        "m_ewma_pre",
        "q_pre",
        "strength_pre",
        "days_since_first_report_pre",
        "days_since_last_dnp_pre",
        "consec_dnps_pre",
        "played_last_game_pre",
        "minutes_last_game_pre",
        "days_since_last_played_pre",
        "injury_present_flag_pre",
        "played_in_game",
        "minutes_in_game",
        "not_playing_reason",
        "not_playing_description",
    ]
    out = joined[out_cols].copy()

    # PK sanity (not enforced, but we can assert uniqueness)
    # One row per (game_id, team_id, player_id) :contentReference[oaicite:10]{index=10}
    dup = out.duplicated(subset=["game_id", "team_id", "player_id"]).sum()
    if dup != 0:
        raise ValueError(f"Duplicate PK rows found: {dup}")

    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/silver/game_team_player_{year}_REGPST.csv")
    out.to_csv(out_path, index=False)

    print(f"{year}: rows={len(out)} games={out['game_id'].nunique()} players={out['player_id'].nunique()}")
    print(f"q_default_prev_season={q_default:.6f}")
    print("wrote:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)