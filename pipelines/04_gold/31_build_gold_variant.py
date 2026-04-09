"""
31_build_gold_variant.py

Builds a variant gold XGBoost input dataset by recomputing the player-state layer
with a given set of feature-level hyperparams. Used to generate the 12 datasets
for Stage A tuning (h_M × L_inj = 4 × 3 = 12).

Feature hyperparams (Stage A grid):
  --h-m         Player minutes EWMA half-life (games)  ∈ {3, 5, 7, 10}   default 5
  --l-inj       Injury inclusion window (days)          ∈ {7, 14, 21}     default 14
  --tau         Player quality prior strength            ∈ {150}           default 150
  --h-team      Team recent-form half-life (games)       ∈ {7}             default 7
                (h_team != 7 requires recomputing franchise recent form;
                 the script handles it, but Stage A keeps h_team fixed at 7)

Output directory:
  data/gold/variants/hM{h_m}_Linj{l_inj}_tau{tau}_hT{h_team}/
    game_xgboost_input_{year}_REGPST.csv    (one per year)
    game_xgboost_input_2015_2024_REGPST.csv (combined training set)

Inputs reused from existing pipeline (unchanged across variants):
  data/silver_plus/elo_franchise_team_game_{year}_REGPST.csv
  data/silver_plus/game_franchise_style_profile_{year}_REGPST.csv
  data/silver_plus/game_team_schedule_context_{year}_REGPST.csv
  data/silver/game_outcomes_{year}_REGPST.csv

Inputs recomputed per variant:
  Player state (from data/silver/game_availability, player_game_box, injury files)
  → game_team_player  (in memory, not written to disk unless --write-silver is set)
  Recent form (from bronze, only if --h-team differs from existing silver_plus files)

Usage:
  # Single variant
  python 31_build_gold_variant.py --h-m 5 --l-inj 14

  # Full Stage A grid (12 combos)
  for hm in 3 5 7 10; do
    for linj in 7 14 21; do
      python 31_build_gold_variant.py --h-m $hm --l-inj $linj
    done
  done
"""
import argparse
import json
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
# Constants (locked feature lists — same as script 30)
# ---------------------------------------------------------------------------

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
N_SLOTS = 12  # always build 12 slots; N_players selection happens at XGB training time


def logit(p: float) -> float:
    p = max(P_CLIP, min(1.0 - P_CLIP, p))
    return math.log(p / (1.0 - p))


# ---------------------------------------------------------------------------
# Player state helpers (inlined from 21_build_player_state_history_year.py)
# ---------------------------------------------------------------------------

def _to_dt_floor_day_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.floor("D")


def _merge_asof_by(left, right, left_on, right_on, by, direction="backward",
                   allow_exact_matches=True):
    """Per-group merge_asof. Avoids pandas by= sortedness issues."""
    out_parts = []
    right_groups = {k: g.sort_values(right_on, kind="stable")
                    for k, g in right.groupby(by, sort=False)}
    for k, lg in left.groupby(by, sort=False):
        lg2 = lg.sort_values(left_on, kind="stable").copy()
        by_y = f"{by}_y"
        if by_y in lg2.columns:
            lg2 = lg2.drop(columns=[by_y])
        rg = right_groups.get(k)
        if rg is None or len(rg) == 0:
            out_parts.append(lg2)
            continue
        rg2 = rg.drop(columns=[by], errors="ignore").copy()
        merged = pd.merge_asof(
            lg2, rg2,
            left_on=left_on, right_on=right_on,
            direction=direction, allow_exact_matches=allow_exact_matches,
        )
        merged = merged.loc[:, ~merged.columns.duplicated()]
        out_parts.append(merged)
    out = pd.concat(out_parts, ignore_index=True)
    return out.loc[:, ~out.columns.duplicated()]


def _compute_prev_season_priors(prev_box: pd.DataFrame):
    tmp = prev_box.copy()
    tmp["minutes"] = pd.to_numeric(tmp["minutes"], errors="coerce").fillna(0.0)
    tmp["eff"] = pd.to_numeric(tmp["eff"], errors="coerce").fillna(0.0)
    g = tmp.groupby("player_id").agg(
        eff_sum=("eff", "sum"), min_sum=("minutes", "sum"), avg_min=("minutes", "mean")
    )
    q_prev = g["eff_sum"] / g["min_sum"].replace({0.0: np.nan})
    denom = float(g["min_sum"].sum())
    q_league = float(g["eff_sum"].sum() / denom) if denom > 0 else 0.0
    return q_prev, q_league, g["avg_min"]


def _build_injury_episode_summary(year: int) -> pd.DataFrame:
    upd_path = Path(f"data/silver/injury_updates_{year}_with_episode.csv")
    dnp_path = Path(f"data/silver/injury_dnp_evidence_{year}_with_episode.csv")
    empty = pd.DataFrame(columns=[
        "player_id", "episode_id",
        "first_report_date_current_window",
        "last_report_date", "last_dnp_date",
        "recent_injury_activity_date",
    ])
    if not upd_path.exists() or not dnp_path.exists():
        return empty
    upd = pd.read_csv(upd_path)
    dnp = pd.read_csv(dnp_path)
    for df in (upd, dnp):
        if "issue_class" in df.columns:
            df.drop(df[~df["issue_class"].isin(["injury_medical", "illness"])].index, inplace=True)
    upd["event_date_dt"] = _to_dt_floor_day_utc(upd["event_date"]) if "event_date" in upd.columns else pd.NaT
    dnp["game_date_dt"] = _to_dt_floor_day_utc(dnp["game_date"]) if "game_date" in dnp.columns else pd.NaT
    upd_ep = upd.dropna(subset=["episode_id"]).groupby(
        ["player_id", "episode_id"], as_index=False
    ).agg(first_report=("event_date_dt", "min"), last_report=("event_date_dt", "max"))
    dnp_ep = dnp.dropna(subset=["episode_id"]).groupby(
        ["player_id", "episode_id"], as_index=False
    ).agg(first_dnp=("game_date_dt", "min"), last_dnp=("game_date_dt", "max"))
    ep = pd.merge(upd_ep, dnp_ep, on=["player_id", "episode_id"], how="outer")
    ep["first_report_date_current_window"] = ep["first_report"]
    ep.loc[ep["first_report_date_current_window"].isna(), "first_report_date_current_window"] = ep["first_dnp"]
    ep["last_report_date"] = ep["last_report"]
    ep["last_dnp_date"] = ep["last_dnp"]
    ep["recent_injury_activity_date"] = ep[["last_report_date", "last_dnp_date"]].max(axis=1)
    ep = ep.dropna(subset=["recent_injury_activity_date"]).copy()
    return ep[["player_id", "episode_id", "first_report_date_current_window",
               "last_report_date", "last_dnp_date", "recent_injury_activity_date"]]


def _build_pg_features(year: int) -> pd.DataFrame:
    ava = pd.read_csv(f"data/silver/game_availability_{year}_REGPST.csv")
    played = pd.read_csv(f"data/silver/played_games_{year}_REGPST.csv")
    played_ids = set(played["game_id"].astype(str))
    ava["game_id"] = ava["game_id"].astype(str)
    ava = ava[ava["game_id"].isin(played_ids)].copy()
    ava["scheduled"] = pd.to_datetime(ava["scheduled"], utc=True, errors="coerce")
    ava["game_date"] = ava["scheduled"].dt.floor("D")
    ava["minutes"] = pd.to_numeric(ava["minutes"], errors="coerce").fillna(0.0)
    ava["not_playing_reason"] = ava["not_playing_reason"].fillna("").astype(str)
    ava["injury_dnp"] = (
        ava["not_playing_reason"].str.contains("Injury|Illness", case=False, regex=True)
        & (ava["minutes"] <= 0)
    )
    out = ava[["player_id", "player_name", "team_id", "game_id",
               "scheduled", "game_date", "minutes", "injury_dnp"]].copy()
    out["player_id"] = out["player_id"].astype(str)
    return out.sort_values(["player_id", "scheduled", "game_id"], kind="stable")


def build_player_state_history(
    year: int,
    lambda_m: float,
    tau_q: float,
    inj_window: int,
) -> pd.DataFrame:
    """
    Recompute player_state_history for `year` using the given hyperparams.
    Returns a DataFrame with the same schema as player_state_history_{year}.csv.
    """
    box_path = Path(f"data/silver/player_game_box_{year}_REGPST.csv")
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")
    if not box_path.exists():
        raise FileNotFoundError(box_path)
    if not played_path.exists():
        raise FileNotFoundError(played_path)

    box = pd.read_csv(box_path)
    played = pd.read_csv(played_path)
    box["player_id"] = box["player_id"].astype(str)
    box["scheduled"] = pd.to_datetime(box["scheduled"], utc=True, errors="coerce")
    box["game_date"] = box["scheduled"].dt.floor("D")
    box["minutes"] = pd.to_numeric(box["minutes"], errors="coerce").fillna(0.0)
    box["eff"] = pd.to_numeric(box["eff"], errors="coerce").fillna(0.0)

    played["scheduled_dt"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    season_start_dt = played["scheduled_dt"].min()
    season_end_dt = played["scheduled_dt"].max()
    season_start_day = season_start_dt.floor("D")
    season_end_day = season_end_dt.floor("D")
    season_start_seed_ts = season_start_day - pd.Timedelta(seconds=1)

    asof_dates = pd.date_range(season_start_day, season_end_day, freq="D", tz="UTC")
    asof_ts = asof_dates + pd.Timedelta(hours=23, minutes=59, seconds=59)

    pg = _build_pg_features(year)
    raw_ids = set(pg["player_id"].dropna().astype(str)) | set(box["player_id"].dropna().astype(str))
    players = pd.Index(sorted(pid for pid in raw_ids if isinstance(pid, str) and pid not in ("nan", "None", "")))

    # Priors from prev season
    prev_box_path = Path(f"data/silver/player_game_box_{year-1}_REGPST.csv")
    if (year - 1) >= 2015 and prev_box_path.exists():
        prev_box = pd.read_csv(prev_box_path)
        prev_box["player_id"] = prev_box["player_id"].astype(str)
        q_prev_by_player, q_league_prev, m_prev_avg_by_player = _compute_prev_season_priors(prev_box)
    else:
        q_prev_by_player = pd.Series(dtype=float)
        q_league_prev = 0.0
        m_prev_avg_by_player = pd.Series(dtype=float)

    m_seed = players.to_series().map(m_prev_avg_by_player).fillna(0.0)
    q_seed = players.to_series().map(q_prev_by_player).fillna(q_league_prev).clip(lower=0.0)
    q_prev = players.to_series().map(q_prev_by_player).fillna(q_league_prev).clip(lower=0.0)

    ep = _build_injury_episode_summary(year)
    if len(ep) > 0:
        ep = ep.copy()
        ep["player_id"] = ep["player_id"].astype(str)
        ep["recent_dt"] = _to_dt_floor_day_utc(ep["recent_injury_activity_date"])
        ep = ep.dropna(subset=["recent_dt"]).sort_values(["player_id", "recent_dt"], kind="stable")

    # Consecutive injury DNP per game
    pg2 = pg.copy()
    pg2["injury_dnp_int"] = pg2["injury_dnp"].astype(int)
    pg2 = pg2.sort_values(["player_id", "scheduled", "game_id"], kind="stable")
    pg2["consec_dnp_at_game"] = 0
    for pid, grp in pg2.groupby("player_id", sort=False):
        streak = 0
        out_streak = []
        for v in grp["injury_dnp_int"].tolist():
            streak = (streak + 1) if v == 1 else 0
            out_streak.append(streak)
        pg2.loc[grp.index, "consec_dnp_at_game"] = out_streak

    # Build panel
    panel = pd.MultiIndex.from_product(
        [players.tolist(), asof_ts], names=["player_id", "asof_ts"]
    ).to_frame(index=False)
    panel["player_id"] = panel["player_id"].astype(str)
    panel["asof_ts"] = pd.to_datetime(panel["asof_ts"], utc=True)
    panel["asof_date"] = panel["asof_ts"].dt.floor("D")
    panel["asof_date_dt"] = panel["asof_date"]

    # current_team_id
    team_hist = pg2[["player_id", "game_date", "team_id"]].dropna().copy()
    team_hist["game_date_dt"] = _to_dt_floor_day_utc(team_hist["game_date"])
    team_hist = team_hist.dropna(subset=["game_date_dt"])
    if len(team_hist) > 0:
        panel = _merge_asof_by(panel, team_hist[["player_id", "game_date_dt", "team_id"]],
                                "asof_date_dt", "game_date_dt", "player_id")
        panel = panel.rename(columns={"team_id": "current_team_id"})
    else:
        panel["current_team_id"] = pd.NA
    panel["current_team_id"] = panel.groupby("player_id")["current_team_id"].transform(lambda s: s.bfill())

    # last game info
    last_game = pg2[["player_id", "game_date", "minutes", "consec_dnp_at_game"]].copy()
    last_game["game_dt"] = _to_dt_floor_day_utc(last_game["game_date"])
    last_game = last_game.dropna(subset=["game_dt"])
    if len(last_game) > 0:
        panel = _merge_asof_by(panel,
                                last_game[["player_id", "game_dt", "minutes", "consec_dnp_at_game"]],
                                "asof_date_dt", "game_dt", "player_id")
    else:
        panel["minutes"] = np.nan
        panel["consec_dnp_at_game"] = np.nan
        panel["game_dt"] = pd.NaT
    panel["minutes_last_game"] = pd.to_numeric(panel["minutes"], errors="coerce").fillna(0.0)
    panel["played_last_game"] = (panel["minutes_last_game"] > 0).astype(int)
    panel["consec_dnps"] = pd.to_numeric(panel["consec_dnp_at_game"], errors="coerce").fillna(0).astype(int)
    panel["last_game_date"] = panel.get("game_dt", pd.NaT)
    panel = panel.drop(columns=["minutes", "consec_dnp_at_game", "game_dt"], errors="ignore")

    # last played date
    played_games = pg2[pg2["minutes"] > 0][["player_id", "game_date"]].copy()
    played_games["game_dt"] = _to_dt_floor_day_utc(played_games["game_date"])
    played_games = played_games.dropna(subset=["game_dt"])
    if len(played_games) > 0:
        panel = _merge_asof_by(panel, played_games[["player_id", "game_dt"]],
                                "asof_date_dt", "game_dt", "player_id")
        panel = panel.rename(columns={"game_dt": "last_played_date"})
    else:
        panel["last_played_date"] = pd.NaT
    # No fallback to prior-season last played: days_since_last_played resets to 0 at season start.
    panel["days_since_last_played"] = (
        (panel["asof_date_dt"]
         - pd.to_datetime(panel["last_played_date"], utc=True, errors="coerce").dt.floor("D"))
        .dt.days.fillna(0).astype(int)
    )

    # Cumulative eff/min for q computation
    box_day = box.groupby(["player_id", "game_date"], as_index=False).agg(
        eff_day=("eff", "sum"), min_day=("minutes", "sum")
    )
    box_day["game_dt"] = _to_dt_floor_day_utc(box_day["game_date"])
    box_day = box_day.dropna(subset=["game_dt"]).sort_values(["player_id", "game_dt"], kind="stable")
    box_day["eff_cum"] = box_day.groupby("player_id")["eff_day"].cumsum()
    box_day["min_cum"] = box_day.groupby("player_id")["min_day"].cumsum()
    if len(box_day) > 0:
        panel = _merge_asof_by(panel, box_day[["player_id", "game_dt", "eff_cum", "min_cum"]],
                                "asof_date_dt", "game_dt", "player_id")
    else:
        panel["eff_cum"] = np.nan
        panel["min_cum"] = np.nan
    panel["eff_cum"] = pd.to_numeric(panel["eff_cum"], errors="coerce").fillna(0.0)
    panel["min_cum"] = pd.to_numeric(panel["min_cum"], errors="coerce").fillna(0.0)
    panel = panel.drop(columns=["game_dt"], errors="ignore")

    # m_ewma with parameterized lambda_m
    m_game = pg2.groupby(["player_id", "game_date"], as_index=False).agg(
        minutes_game=("minutes", "sum")
    )
    m_game["game_dt"] = _to_dt_floor_day_utc(m_game["game_date"])
    m_game = m_game.dropna(subset=["game_dt"]).sort_values(["player_id", "game_dt"], kind="stable")
    m_game["m_ewma_at_game"] = np.nan
    for pid, grp in m_game.groupby("player_id", sort=False):
        prev = float(m_seed.get(pid, 0.0))
        outs = []
        for mval in grp["minutes_game"].tolist():
            prev = lambda_m * float(mval) + (1 - lambda_m) * prev
            outs.append(prev)
        m_game.loc[grp.index, "m_ewma_at_game"] = outs
    if len(m_game) > 0:
        panel = _merge_asof_by(panel, m_game[["player_id", "game_dt", "m_ewma_at_game"]],
                                "asof_date_dt", "game_dt", "player_id")
    else:
        panel["m_ewma_at_game"] = np.nan
    panel["m_ewma"] = pd.to_numeric(panel["m_ewma_at_game"], errors="coerce")
    panel["m_ewma"] = panel["m_ewma"].fillna(panel["player_id"].map(m_seed)).fillna(0.0)
    panel = panel.drop(columns=["m_ewma_at_game", "game_dt"], errors="ignore")

    # q with parameterized tau_q
    panel["q_prev"] = panel["player_id"].map(q_prev).fillna(q_league_prev)
    panel["q"] = (tau_q * panel["q_prev"] + panel["eff_cum"]) / (tau_q + panel["min_cum"])
    panel["q"] = panel["q"].clip(lower=0.0)
    panel["strength"] = panel["m_ewma"] * panel["q"]

    # Injury fields with parameterized inj_window
    if len(ep) > 0:
        ep_key = ep.copy()
        panel = _merge_asof_by(
            panel,
            ep_key[["player_id", "recent_dt", "episode_id",
                     "first_report_date_current_window", "last_report_date", "last_dnp_date"]],
            "asof_date_dt", "recent_dt", "player_id",
        )
    else:
        panel["episode_id"] = pd.NA
        panel["first_report_date_current_window"] = pd.NaT
        panel["last_report_date"] = pd.NaT
        panel["last_dnp_date"] = pd.NaT

    last_report_dt = _to_dt_floor_day_utc(panel["last_report_date"])
    last_dnp_dt = _to_dt_floor_day_utc(panel["last_dnp_date"])
    recent_dt = pd.concat([last_report_dt, last_dnp_dt], axis=1).max(axis=1)
    days_since_recent = (panel["asof_date_dt"] - recent_dt).dt.days
    panel["injury_present_flag"] = (recent_dt.notna() & (days_since_recent <= inj_window)).astype(int)

    first_report_dt = _to_dt_floor_day_utc(panel["first_report_date_current_window"])
    panel["days_since_first_report"] = (panel["asof_date_dt"] - first_report_dt).dt.days
    panel["days_since_last_dnp"] = (panel["asof_date_dt"] - last_dnp_dt).dt.days
    panel.loc[last_dnp_dt.isna(), "days_since_last_dnp"] = 0
    mask_no_inj = panel["injury_present_flag"] == 0
    panel.loc[mask_no_inj, "days_since_first_report"] = 0
    panel.loc[mask_no_inj, "days_since_last_dnp"] = 0
    panel.loc[mask_no_inj, "consec_dnps"] = 0
    panel["days_since_first_report"] = panel["days_since_first_report"].fillna(0).astype(int)
    panel["days_since_last_dnp"] = panel["days_since_last_dnp"].fillna(0).astype(int)

    # Build seed rows
    out_cols = [
        "player_id", "asof_ts", "current_team_id",
        "m_ewma", "q", "strength",
        "days_since_first_report", "days_since_last_dnp", "consec_dnps",
        "played_last_game", "minutes_last_game", "days_since_last_played",
        "injury_present_flag",
        "asof_date", "last_game_date", "last_report_date", "last_dnp_date",
        "first_report_date_current_window", "episode_id",
    ]
    seed = pd.DataFrame({
        "player_id": players.astype(str),
        "asof_ts": season_start_seed_ts,
        "asof_date": season_start_seed_ts.floor("D"),
    })
    seed["asof_ts"] = pd.to_datetime(seed["asof_ts"], utc=True)
    seed["asof_date_dt"] = pd.to_datetime(seed["asof_date"], utc=True)
    first_team = (
        pg2.dropna(subset=["team_id"])
        .sort_values(["player_id", "scheduled"], kind="stable")
        .groupby("player_id")["team_id"].first()
    )
    seed["current_team_id"] = seed["player_id"].map(first_team)
    seed["m_ewma"] = seed["player_id"].map(m_seed).fillna(0.0)
    seed["q"] = seed["player_id"].map(q_seed).fillna(q_league_prev).clip(lower=0.0)
    seed["strength"] = seed["m_ewma"] * seed["q"]
    for c in ["days_since_first_report", "days_since_last_dnp", "consec_dnps",
              "played_last_game", "days_since_last_played", "injury_present_flag"]:
        seed[c] = 0
    seed["minutes_last_game"] = 0.0
    seed["last_game_date"] = pd.NaT
    seed["last_report_date"] = pd.NaT
    seed["last_dnp_date"] = pd.NaT
    seed["first_report_date_current_window"] = pd.NaT
    seed["episode_id"] = pd.NA

    panel_out = panel[[c for c in out_cols if c in panel.columns]].copy()
    seed_out = seed[[c for c in out_cols if c in seed.columns]].copy()
    out = pd.concat([seed_out, panel_out], ignore_index=True)
    out = out.sort_values(["player_id", "asof_ts"], kind="stable")
    return out


# ---------------------------------------------------------------------------
# Game-team-player builder (inlined from 22_build_game_team_player_year.py)
# ---------------------------------------------------------------------------

def _asof_join_player_state_strict(roster: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    out_parts = []
    roster = roster.copy()
    roster["player_id"] = roster["player_id"].astype(str)
    roster["game_ts"] = pd.to_datetime(roster["game_ts"], utc=True, errors="coerce")
    state = state.copy()
    state["player_id"] = state["player_id"].astype(str)
    state["asof_ts"] = pd.to_datetime(state["asof_ts"], utc=True, errors="coerce")
    state = state.dropna(subset=["player_id", "asof_ts"]).drop_duplicates(
        subset=["player_id", "asof_ts"], keep="last"
    )
    state_groups = {pid: g.sort_values("asof_ts", kind="stable")
                    for pid, g in state.groupby("player_id", sort=False)}
    for pid, g in roster.groupby("player_id", sort=False):
        g2 = g.sort_values("game_ts", kind="stable").copy()
        sg = state_groups.get(pid)
        if sg is None or len(sg) == 0:
            out_parts.append(g2)
            continue
        sg2 = sg.drop(columns=["player_id"])
        merged = pd.merge_asof(
            g2, sg2,
            left_on="game_ts", right_on="asof_ts",
            direction="backward", allow_exact_matches=False,
        )
        out_parts.append(merged)
    return pd.concat(out_parts, ignore_index=True)


def build_game_team_player(year: int, state: pd.DataFrame) -> pd.DataFrame:
    """Build game_team_player for `year` using a pre-computed state DataFrame."""
    roster_path = Path(f"data/silver/game_availability_{year}_REGPST.csv")
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")
    if not roster_path.exists():
        raise FileNotFoundError(roster_path)
    if not played_path.exists():
        raise FileNotFoundError(played_path)

    roster = pd.read_csv(roster_path)
    played = pd.read_csv(played_path)
    played_ids = set(played["game_id"].astype(str))
    roster["game_id"] = roster["game_id"].astype(str)
    roster = roster[roster["game_id"].isin(played_ids)].copy()

    roster["game_ts"] = pd.to_datetime(roster["scheduled"], utc=True, errors="coerce")
    roster["game_date"] = roster["game_ts"].dt.floor("D").dt.date.astype(str)
    roster["season"] = year
    if "opponent_id" in roster.columns:
        roster["opponent_team_id"] = roster["opponent_id"]
    else:
        roster["opponent_team_id"] = roster.get("opponent_team_id", pd.NA)
    if "side" in roster.columns:
        roster["is_home"] = (roster["side"].astype(str).str.lower() == "home").astype(int)
    else:
        roster["is_home"] = 0
    if "season_type" in roster.columns:
        roster["is_playoff"] = (roster["season_type"].astype(str).str.upper() == "PST").astype(int)
    else:
        pm = played[["game_id", "season_type"]].drop_duplicates("game_id")
        roster = roster.merge(pm, on="game_id", how="left")
        roster["is_playoff"] = (roster["season_type"].astype(str).str.upper() == "PST").astype(int)

    roster["player_id"] = roster["player_id"].astype(str)
    roster["team_id"] = roster["team_id"].astype(str)
    roster["player_name"] = roster.get("player_name", "").astype(str)
    roster["listed_on_game_summary_flag"] = 1
    roster["minutes_in_game"] = pd.to_numeric(roster.get("minutes"), errors="coerce").fillna(0.0)
    roster["played_in_game"] = (roster["minutes_in_game"] > 0).astype(int)
    roster["not_playing_reason"] = roster.get("not_playing_reason", "").fillna("").astype(str)
    roster["not_playing_description"] = roster.get("not_playing_description", "").fillna("").astype(str)
    roster = roster.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")

    skeleton = roster[[
        "game_id", "game_ts", "game_date", "season", "team_id",
        "opponent_team_id", "is_home", "is_playoff",
        "player_id", "player_name", "listed_on_game_summary_flag",
        "minutes_in_game", "played_in_game",
        "not_playing_reason", "not_playing_description",
    ]].copy()

    req_state_cols = [
        "player_id", "asof_ts", "m_ewma", "q", "strength",
        "days_since_first_report", "days_since_last_dnp", "consec_dnps",
        "played_last_game", "minutes_last_game", "days_since_last_played",
        "injury_present_flag",
    ]
    state_slim = state[[c for c in req_state_cols if c in state.columns]].copy()

    joined = _asof_join_player_state_strict(skeleton, state_slim)
    joined = joined.rename(columns={"asof_ts": "state_asof_ts"})
    rename_map = {
        "m_ewma": "m_ewma_pre", "q": "q_pre", "strength": "strength_pre",
        "days_since_first_report": "days_since_first_report_pre",
        "days_since_last_dnp": "days_since_last_dnp_pre",
        "consec_dnps": "consec_dnps_pre",
        "played_last_game": "played_last_game_pre",
        "minutes_last_game": "minutes_last_game_pre",
        "days_since_last_played": "days_since_last_played_pre",
        "injury_present_flag": "injury_present_flag_pre",
    }
    joined = joined.rename(columns=rename_map)

    prev_box_path = Path(f"data/silver/player_game_box_{year-1}_REGPST.csv")
    if (year - 1) >= 2015 and prev_box_path.exists():
        prev_box = pd.read_csv(prev_box_path)
        prev_box["player_id"] = prev_box["player_id"].astype(str)
        prev_box["minutes"] = pd.to_numeric(prev_box["minutes"], errors="coerce").fillna(0.0)
        prev_box["eff"] = pd.to_numeric(prev_box["eff"], errors="coerce").fillna(0.0)
        g = prev_box.groupby("player_id").agg(eff_sum=("eff", "sum"), min_sum=("minutes", "sum"))
        q_league = float(g["eff_sum"].sum() / g["min_sum"].sum()) if g["min_sum"].sum() > 0 else 0.0
    else:
        q_league = 0.0

    no_state = joined["state_asof_ts"].isna()
    joined.loc[no_state, "m_ewma_pre"] = 0.0
    joined.loc[no_state, "q_pre"] = q_league
    joined.loc[no_state, "strength_pre"] = 0.0
    for c in ["days_since_first_report_pre", "days_since_last_dnp_pre", "consec_dnps_pre",
              "injury_present_flag_pre", "played_last_game_pre", "days_since_last_played_pre"]:
        joined.loc[no_state, c] = 0
    joined.loc[no_state, "minutes_last_game_pre"] = 0.0

    for c in ["m_ewma_pre", "q_pre", "strength_pre", "minutes_last_game_pre"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0.0)
    for c in ["days_since_first_report_pre", "days_since_last_dnp_pre", "consec_dnps_pre",
              "played_last_game_pre", "days_since_last_played_pre", "injury_present_flag_pre"]:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0).astype(int)
    joined["strength_pre"] = joined["m_ewma_pre"] * joined["q_pre"]

    dup = joined.duplicated(subset=["game_id", "team_id", "player_id"]).sum()
    if dup:
        raise ValueError(f"Duplicate PK in game_team_player year={year}: {dup}")
    return joined


# ---------------------------------------------------------------------------
# Recent-form builder (only used when h_team != 7)
# ---------------------------------------------------------------------------

def build_franchise_recent_form(year: int, lambda_team: float) -> pd.DataFrame:
    """
    Recompute franchise recent form with a custom EWMA half-life.
    Only needed when lambda_team != default (7-game half-life).
    Requires data/bronze/game_summary__*.json files.
    """
    try:
        from srwnba.util.franchise import load_franchise_map, map_team_to_franchise
    except ImportError:
        raise ImportError("srwnba package not found; cannot recompute recent form")

    played_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(played_path)

    played = pd.read_csv(played_path)
    played["game_id"] = played["game_id"].astype(str)
    played["game_ts"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    played["game_date"] = played["game_ts"].dt.date.astype(str)
    played["is_playoff"] = (played["season_type"].astype(str).str.upper() == "PST").astype(int)
    played = played.sort_values(["game_ts", "game_id"], kind="stable")

    map_df = load_franchise_map()
    game_ids = set(played["game_id"])

    def _ts_from_name(name):
        parts = name.split("__")
        return parts[2].replace(".json", "") if len(parts) >= 3 else ""

    best = {}
    for p in Path("data/bronze").glob("game_summary__*__*.json"):
        parts = p.name.split("__")
        if len(parts) < 3:
            continue
        gid = parts[1]
        if gid not in game_ids:
            continue
        ts = _ts_from_name(p.name)
        if gid not in best or ts > best[gid][0]:
            best[gid] = (ts, p)
    latest = {gid: v[1] for gid, v in best.items()}

    def _to_int(x):
        try:
            return 0 if x is None else int(x)
        except Exception:
            return 0

    def _extract_totals(block):
        s = (block or {}).get("statistics") or {}
        if s.get("total_turnovers") is not None:
            tov = _to_int(s.get("total_turnovers"))
        else:
            tov = _to_int(s.get("player_turnovers", 0)) + _to_int(s.get("team_turnovers", 0))
        return {
            "pts": _to_int(s.get("points") or (block or {}).get("points")),
            "fga": _to_int(s.get("field_goals_att")),
            "fgm": _to_int(s.get("field_goals_made")),
            "tpm": _to_int(s.get("three_points_made")),
            "fta": _to_int(s.get("free_throws_att")),
            "tov": tov,
            "orb": _to_int(s.get("offensive_rebounds")),
            "drb": _to_int(s.get("defensive_rebounds")),
        }

    def _safe_div(n, d):
        try:
            return 0.0 if not d else float(n) / float(d)
        except Exception:
            return 0.0

    def _metrics(t, o):
        poss = t["fga"] - t["orb"] + t["tov"] + 0.44 * t["fta"]
        poss_o = o["fga"] - o["orb"] + o["tov"] + 0.44 * o["fta"]
        ortg = 100.0 * _safe_div(t["pts"], poss)
        drtg = 100.0 * _safe_div(o["pts"], poss_o)
        return {
            "net_rtg": ortg - drtg,
            "efg": _safe_div(t["fgm"] + 0.5 * t["tpm"], t["fga"]),
            "tov_pct": _safe_div(t["tov"], poss),
            "orb_pct": _safe_div(t["orb"], t["orb"] + o["drb"]),
            "ftr": _safe_div(t["fta"], t["fga"]),
        }

    realized_rows = []
    meta = played.set_index("game_id").to_dict(orient="index")
    for gid in played["game_id"].tolist():
        if gid not in latest:
            continue
        d = json.loads(latest[gid].read_text(encoding="utf-8"))
        h_tot = _extract_totals(d.get("home") or {})
        a_tot = _extract_totals(d.get("away") or {})
        h_m = _metrics(h_tot, a_tot)
        a_m = _metrics(a_tot, h_tot)
        m = meta.get(gid, {})
        home_id = str(m.get("home_id"))
        away_id = str(m.get("away_id"))
        home_fid = map_team_to_franchise(home_id, year, map_df)
        away_fid = map_team_to_franchise(away_id, year, map_df)
        for tid, fid, opp_tid, opp_fid, is_home, metrics in [
            (home_id, home_fid, away_id, away_fid, 1, h_m),
            (away_id, away_fid, home_id, home_fid, 0, a_m),
        ]:
            realized_rows.append({
                "season": year, "game_id": gid,
                "game_ts": m.get("game_ts"), "game_date": m.get("game_date", ""),
                "team_id": tid, "franchise_id": fid,
                "opponent_team_id": opp_tid, "opponent_franchise_id": opp_fid,
                "is_home": is_home, "is_playoff": int(m.get("is_playoff", 0)),
                **{f"{k}_game": v for k, v in metrics.items()},
            })

    realized = pd.DataFrame(realized_rows).sort_values(["game_ts", "game_id", "is_home"], kind="stable")

    state: dict = {}
    out_rows = []
    for _, row in realized.iterrows():
        fid = str(row["franchise_id"])
        if fid not in state:
            state[fid] = {k: 0.0 for k in ["net_rtg_ewma", "efg_ewma", "tov_pct_ewma", "orb_pct_ewma", "ftr_ewma"]}
        st = state[fid]
        out_rows.append({
            "season": int(row["season"]), "game_id": row["game_id"],
            "game_ts": row["game_ts"], "game_date": row["game_date"],
            "team_id": str(row["team_id"]), "franchise_id": fid,
            "opponent_team_id": str(row["opponent_team_id"]),
            "opponent_franchise_id": str(row["opponent_franchise_id"]),
            "is_home": int(row["is_home"]), "is_playoff": int(row["is_playoff"]),
            "net_rtg_ewma_pre": st["net_rtg_ewma"],
            "efg_ewma_pre": st["efg_ewma"],
            "tov_pct_ewma_pre": st["tov_pct_ewma"],
            "orb_pct_ewma_pre": st["orb_pct_ewma"],
            "ftr_ewma_pre": st["ftr_ewma"],
        })
        for k in ["net_rtg", "efg", "tov_pct", "orb_pct", "ftr"]:
            st[f"{k}_ewma"] = lambda_team * float(row[f"{k}_game"]) + (1 - lambda_team) * st[f"{k}_ewma"]

    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Gold assembly (inlined from 30_build_game_xgboost_input.py)
# ---------------------------------------------------------------------------

def build_player_slots(players: pd.DataFrame, game_id: str, team_id: str) -> dict:
    grp = players[(players["game_id"] == game_id) & (players["team_id"] == team_id)].copy()
    grp = grp.sort_values(
        ["strength_pre", "m_ewma_pre", "q_pre", "player_id"],
        ascending=[False, False, False, True], kind="stable",
    )
    out = {}
    for slot in range(1, N_SLOTS + 1):
        prefix = f"p{slot}_"
        if slot <= len(grp):
            row = grp.iloc[slot - 1]
            for col in PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES:
                val = row.get(col)
                out[prefix + col] = None if (val is None or (isinstance(val, float) and math.isnan(val))) else val
        else:
            for col in PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES:
                out[prefix + col] = None
    return out


def assemble_gold(
    year: int,
    players: pd.DataFrame,
    form: pd.DataFrame,
    style: pd.DataFrame,
    sched: pd.DataFrame,
    elo: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    elo["game_id"] = elo["game_id"].astype(str)
    elo["team_id"] = elo["team_id"].astype(str)
    players["game_id"] = players["game_id"].astype(str)
    players["team_id"] = players["team_id"].astype(str)
    form["game_id"] = form["game_id"].astype(str)
    form["team_id"] = form["team_id"].astype(str)
    style["game_id"] = style["game_id"].astype(str)
    style["team_id"] = style["team_id"].astype(str)
    sched["game_id"] = sched["game_id"].astype(str)
    sched["team_id"] = sched["team_id"].astype(str)
    outcomes["game_id"] = outcomes["game_id"].astype(str)

    home_elo = elo[elo["is_home"] == 1].set_index("game_id")
    away_elo = elo[elo["is_home"] == 0].set_index("game_id")
    game_ids = sorted(set(home_elo.index) & set(away_elo.index))

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
        p_elo = float(h["p_win_pre"])
        bm = logit(p_elo)

        row: dict = {
            "game_id": gid,
            "game_ts": h.get("scheduled"),
            "game_date": (pd.to_datetime(h.get("scheduled"), utc=True, errors="coerce").date()
                          if pd.notna(h.get("scheduled")) else None),
            "season": int(h.get("season_year", year)),
            "is_playoff": int(h.get("is_playoff", 0)) if "is_playoff" in h.index else None,
            "home_team_id": home_tid,
            "away_team_id": away_tid,
            "home_franchise_id": str(h["franchise_id"]),
            "away_franchise_id": str(a["franchise_id"]),
            "home_elo_pre": float(h["elo_pre"]),
            "away_elo_pre": float(a["elo_pre"]),
            "p_elo": p_elo,
            "base_margin": bm,
            "home_win": (int(outcomes_idx.loc[gid, "home_win"])
                         if gid in outcomes_idx.index and pd.notna(outcomes_idx.loc[gid, "home_win"])
                         else None),
        }

        for k, v in build_player_slots(players, gid, home_tid).items():
            row[f"home_{k}"] = v
        for k, v in build_player_slots(players, gid, away_tid).items():
            row[f"away_{k}"] = v

        def get_feat(idx, tid, feat):
            key = (gid, tid)
            return idx.loc[key, feat] if key in idx.index else None

        for f in RECENT_FORM_FEATURES:
            row[f"home_{f}"] = get_feat(form_idx, home_tid, f)
            row[f"away_{f}"] = get_feat(form_idx, away_tid, f)
        for f in STYLE_FEATURES:
            row[f"home_{f}"] = get_feat(style_idx, home_tid, f)
            row[f"away_{f}"] = get_feat(style_idx, away_tid, f)
        for f in SCHEDULE_FEATURES + SCHEDULE_META:
            row[f"home_{f}"] = get_feat(sched_idx, home_tid, f)
            row[f"away_{f}"] = get_feat(sched_idx, away_tid, f)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["season", "game_ts", "game_id"], kind="stable").reset_index(drop=True)

    metadata_cols = [
        "game_id", "game_ts", "game_date", "season", "is_playoff",
        "home_team_id", "away_team_id", "home_franchise_id", "away_franchise_id",
        "home_elo_pre", "away_elo_pre", "p_elo", "base_margin", "home_win",
    ]
    player_cols = [
        f"{side}_p{slot}_{col}"
        for side in ("home", "away")
        for slot in range(1, N_SLOTS + 1)
        for col in PLAYER_DEBUG_FEATURES + PLAYER_MODEL_FEATURES
    ]
    form_cols = [f"{s}_{f}" for s in ("home", "away") for f in RECENT_FORM_FEATURES]
    style_cols = [f"{s}_{f}" for s in ("home", "away") for f in STYLE_FEATURES]
    sched_cols = [f"{s}_{f}" for s in ("home", "away") for f in SCHEDULE_FEATURES]
    sched_meta_cols = [f"{s}_{f}" for s in ("home", "away") for f in SCHEDULE_META]
    final_cols = metadata_cols + player_cols + form_cols + style_cols + sched_cols + sched_meta_cols
    return out[[c for c in final_cols if c in out.columns]]


# ---------------------------------------------------------------------------
# Per-year orchestration
# ---------------------------------------------------------------------------

DEFAULT_H_TEAM = 7


def build_year(
    year: int,
    lambda_m: float,
    tau_q: float,
    inj_window: int,
    lambda_team: float,
    h_team: int,
) -> pd.DataFrame:
    print(f"  [{year}] computing player state...")
    state = build_player_state_history(year, lambda_m=lambda_m, tau_q=tau_q, inj_window=inj_window)

    print(f"  [{year}] building game_team_player...")
    players = build_game_team_player(year, state)

    # Load fixed silver_plus tables (unchanged across player variants)
    elo_path = Path(f"data/silver_plus/elo_franchise_team_game_{year}_REGPST.csv")
    if not elo_path.exists():
        raise FileNotFoundError(elo_path)
    elo = pd.read_csv(elo_path)

    style_path = Path(f"data/silver_plus/game_franchise_style_profile_{year}_REGPST.csv")
    if not style_path.exists():
        raise FileNotFoundError(style_path)
    style = pd.read_csv(style_path, usecols=["game_id", "team_id"] + STYLE_FEATURES)

    sched_path = Path(f"data/silver_plus/game_team_schedule_context_{year}_REGPST.csv")
    if not sched_path.exists():
        raise FileNotFoundError(sched_path)
    sched = pd.read_csv(sched_path, usecols=["game_id", "team_id"] + SCHEDULE_FEATURES + SCHEDULE_META)

    outcomes_path = Path(f"data/silver/game_outcomes_{year}_REGPST.csv")
    if not outcomes_path.exists():
        raise FileNotFoundError(outcomes_path)
    outcomes = pd.read_csv(outcomes_path, usecols=["game_id", "home_win"])
    outcomes["home_win"] = pd.to_numeric(outcomes["home_win"], errors="coerce").astype("Int64")

    # Recent form: recompute if h_team differs from default, else load existing
    if abs(lambda_team - (1 - 2 ** (-1 / DEFAULT_H_TEAM))) > 1e-9:
        print(f"  [{year}] recomputing recent form (h_team={h_team})...")
        form = build_franchise_recent_form(year, lambda_team)
        form = form[["game_id", "team_id"] + RECENT_FORM_FEATURES]
    else:
        form_path = Path(f"data/silver_plus/game_franchise_recent_form_{year}_REGPST.csv")
        if not form_path.exists():
            raise FileNotFoundError(form_path)
        form = pd.read_csv(form_path, usecols=["game_id", "team_id"] + RECENT_FORM_FEATURES)

    print(f"  [{year}] assembling gold table...")
    df = assemble_gold(year, players, form, style, sched, elo, outcomes)
    print(f"  [{year}] rows={len(df)}  cols={len(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a variant gold XGBoost input dataset.")

    # Feature hyperparams
    ap.add_argument("--h-m", type=float, default=5,
                    help="Player minutes EWMA half-life in games {3, 5, 7, 10}")
    ap.add_argument("--l-inj", type=int, default=14,
                    help="Injury inclusion window in days {7, 14, 21}")
    ap.add_argument("--tau", type=float, default=150,
                    help="Player quality prior strength (tau_q)")
    ap.add_argument("--h-team", type=float, default=7,
                    help="Team recent-form EWMA half-life in games")

    # Year range
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)

    # Output
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override output directory (default: data/gold/variants/<tag>)")

    args = ap.parse_args()

    lambda_m = 1 - 2 ** (-1 / args.h_m)
    lambda_team = 1 - 2 ** (-1 / args.h_team)
    tag = f"hM{args.h_m:g}_Linj{args.l_inj}_tau{args.tau:g}_hT{args.h_team:g}"

    out_dir = args.out_dir if args.out_dir else Path("data/gold/variants") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Variant tag    : {tag}")
    print(f"lambda_m       : {lambda_m:.6f}  (h_m={args.h_m})")
    print(f"tau_q          : {args.tau}")
    print(f"inj_window     : {args.l_inj} days")
    print(f"lambda_team    : {lambda_team:.6f}  (h_team={args.h_team})")
    print(f"output dir     : {out_dir}")
    print("=" * 60)

    all_dfs = []
    for year in range(args.start_year, args.end_year + 1):
        print(f"\n=== {year} ===")
        df = build_year(
            year=year,
            lambda_m=lambda_m,
            tau_q=args.tau,
            inj_window=args.l_inj,
            lambda_team=lambda_team,
            h_team=args.h_team,
        )
        out_path = out_dir / f"game_xgboost_input_{year}_REGPST.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote: {out_path}")
        all_dfs.append(df)

    # Combined training set (2015–2024)
    train_dfs = [df for df in all_dfs if int(df["season"].iloc[0]) <= 2024]
    if train_dfs:
        combined = pd.concat(train_dfs, ignore_index=True)
        combined_path = out_dir / "game_xgboost_input_2015_2024_REGPST.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nwrote combined: {combined_path}  rows={len(combined)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
