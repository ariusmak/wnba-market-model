import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Locked params from spec
LAMBDA_M = 1 - 2 ** (-1 / 7)   # EWMA half-life 7 games (Stage 1 tuning winner)
TAU_Q = 150                    # q prior strength
INJ_WINDOW_DAYS = 14           # injury inclusion window


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def to_dt_floor_day_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce").dt.floor("D")


def merge_asof_by(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: str,
    right_on: str,
    by: str,
    direction: str = "backward",
    allow_exact_matches: bool = True,
    suffixes=("", "_y"),
) -> pd.DataFrame:
    """
    Robust as-of merge per `by` group.

    Critical fix:
      - Drop the `by` column from the RIGHT frame before merge_asof to prevent
        repeated creation of `{by}_y` across multiple merges (which creates duplicate columns).
      - Drop any duplicate columns after merging (safety belt).
    """
    out_parts = []

    right_sorted = right.sort_values([by, right_on], kind="stable")
    right_groups = {k: g.sort_values(right_on, kind="stable") for k, g in right_sorted.groupby(by, sort=False)}

    for k, lg in left.groupby(by, sort=False):
        lg2 = lg.sort_values(left_on, kind="stable").copy()

        # If a previous merge created a redundant '{by}_y', drop it now (defensive)
        by_y = f"{by}_y"
        if by_y in lg2.columns:
            lg2 = lg2.drop(columns=[by_y])

        rg = right_groups.get(k)
        if rg is None or len(rg) == 0:
            out_parts.append(lg2)
            continue

        # IMPORTANT: drop `by` from the right frame so we don't create player_id_y repeatedly
        rg2 = rg.copy()
        if by in rg2.columns:
            rg2 = rg2.drop(columns=[by])

        merged = pd.merge_asof(
            lg2,
            rg2,
            left_on=left_on,
            right_on=right_on,
            direction=direction,
            allow_exact_matches=allow_exact_matches,
            suffixes=suffixes,
        )

        # Defensive: ensure columns are unique
        merged = merged.loc[:, ~merged.columns.duplicated()]

        out_parts.append(merged)

    out = pd.concat(out_parts, ignore_index=True)
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def compute_prev_season_priors(prev_box: pd.DataFrame) -> tuple[pd.Series, float, pd.Series]:
    """
    Returns:
      q_prev_by_player: player_id -> q_prev (EFF_prev / MIN_prev)
      q_league_avg_prev: float
      m_prev_avg_by_player: player_id -> avg minutes per game (prev season), used to init m_ewma
    """
    tmp = prev_box.copy()
    tmp["minutes"] = pd.to_numeric(tmp["minutes"], errors="coerce").fillna(0.0)
    tmp["eff"] = pd.to_numeric(tmp["eff"], errors="coerce").fillna(0.0)

    g = tmp.groupby("player_id", as_index=True).agg(
        eff_sum=("eff", "sum"),
        min_sum=("minutes", "sum"),
        avg_min=("minutes", "mean"),
    )

    q_prev = g["eff_sum"] / g["min_sum"].replace({0.0: np.nan})
    denom = float(g["min_sum"].sum())
    q_league = float(g["eff_sum"].sum() / denom) if denom > 0 else 0.0
    m_prev_avg = g["avg_min"]
    return q_prev, q_league, m_prev_avg


def build_injury_episode_summary(year: int) -> pd.DataFrame:
    """
    Per-episode summary; Decision A: DNP substitutes for missing report.
    """
    upd_path = Path(f"data/silver/injury_updates_{year}_with_episode.csv")
    dnp_path = Path(f"data/silver/injury_dnp_evidence_{year}_with_episode.csv")

    if (not upd_path.exists()) or (not dnp_path.exists()):
        return pd.DataFrame(columns=[
            "player_id", "episode_id",
            "first_report_date_current_window",
            "last_report_date", "last_dnp_date",
            "recent_injury_activity_date"
        ])

    upd = pd.read_csv(upd_path)
    dnp = pd.read_csv(dnp_path)

    if "issue_class" in upd.columns:
        upd = upd[upd["issue_class"].isin(["injury_medical", "illness"])].copy()
    if "issue_class" in dnp.columns:
        dnp = dnp[dnp["issue_class"].isin(["injury_medical", "illness"])].copy()

    upd["event_date_dt"] = to_dt_floor_day_utc(upd["event_date"]) if "event_date" in upd.columns else pd.NaT
    dnp["game_date_dt"] = to_dt_floor_day_utc(dnp["game_date"]) if "game_date" in dnp.columns else pd.NaT

    upd_ep = upd.dropna(subset=["episode_id"]).groupby(["player_id", "episode_id"], as_index=False).agg(
        first_report=("event_date_dt", "min"),
        last_report=("event_date_dt", "max"),
    )
    dnp_ep = dnp.dropna(subset=["episode_id"]).groupby(["player_id", "episode_id"], as_index=False).agg(
        first_dnp=("game_date_dt", "min"),
        last_dnp=("game_date_dt", "max"),
    )

    ep = pd.merge(upd_ep, dnp_ep, on=["player_id", "episode_id"], how="outer")

    # Decision A: if no report exists, first DNP substitutes
    ep["first_report_date_current_window"] = ep["first_report"]
    ep.loc[ep["first_report_date_current_window"].isna(), "first_report_date_current_window"] = ep["first_dnp"]

    ep["last_report_date"] = ep["last_report"]
    ep["last_dnp_date"] = ep["last_dnp"]
    ep["recent_injury_activity_date"] = ep[["last_report_date", "last_dnp_date"]].max(axis=1)

    ep = ep.dropna(subset=["recent_injury_activity_date"]).copy()
    ep = ep.sort_values(["player_id", "recent_injury_activity_date"], kind="stable")
    return ep[[
        "player_id", "episode_id",
        "first_report_date_current_window",
        "last_report_date",
        "last_dnp_date",
        "recent_injury_activity_date"
    ]]


def build_player_game_features_for_year(year: int) -> pd.DataFrame:
    """
    Per-player per-game for played games only: minutes, injury_dnp flag, team_id.
    """
    ava = load_csv(f"data/silver/game_availability_{year}_REGPST.csv")
    played = load_csv(f"data/silver/played_games_{year}_REGPST.csv")
    played_ids = set(played["game_id"].astype(str).unique())

    ava["game_id"] = ava["game_id"].astype(str)
    ava = ava[ava["game_id"].isin(played_ids)].copy()

    ava["scheduled"] = pd.to_datetime(ava["scheduled"], utc=True, errors="coerce")
    ava["game_date"] = ava["scheduled"].dt.floor("D")
    ava["minutes"] = pd.to_numeric(ava["minutes"], errors="coerce").fillna(0.0)
    ava["not_playing_reason"] = ava["not_playing_reason"].fillna("").astype(str)

    ava["injury_dnp"] = ava["not_playing_reason"].str.contains("Injury|Illness", case=False, regex=True) & (ava["minutes"] <= 0)

    out = ava[[
        "player_id", "player_name", "team_id", "game_id",
        "scheduled", "game_date", "minutes", "injury_dnp"
    ]].copy()

    out["player_id"] = out["player_id"].astype(str)
    return out.sort_values(["player_id", "scheduled", "game_id"], kind="stable")


def main(year: int):
    box = load_csv(f"data/silver/player_game_box_{year}_REGPST.csv")
    played = load_csv(f"data/silver/played_games_{year}_REGPST.csv")

    played["scheduled_dt"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    season_start_dt = played["scheduled_dt"].min()
    season_end_dt = played["scheduled_dt"].max()

    season_start_day = season_start_dt.floor("D")
    season_end_day = season_end_dt.floor("D")

    # Seed timestamp strictly before first game timestamp:
    # midnight on first game date minus 1 second
    season_start_seed_ts = season_start_day - pd.Timedelta(seconds=1)

    # Daily asof_ts (end of each day) for season days (start at first game date)
    asof_dates = pd.date_range(season_start_day, season_end_day, freq="D", tz="UTC")
    asof_ts = asof_dates + pd.Timedelta(hours=23, minutes=59, seconds=59)

    pg = build_player_game_features_for_year(year)

    box["player_id"] = box["player_id"].astype(str)
    box["scheduled"] = pd.to_datetime(box["scheduled"], utc=True, errors="coerce")
    box["game_date"] = box["scheduled"].dt.floor("D")
    box["minutes"] = pd.to_numeric(box["minutes"], errors="coerce").fillna(0.0)
    box["eff"] = pd.to_numeric(box["eff"], errors="coerce").fillna(0.0)

    # Player universe for season (drop NaN/None player_ids before sorting)
    raw_ids = set(pg["player_id"].dropna().astype(str)) | set(box["player_id"].dropna().astype(str))
    players = pd.Index(sorted(pid for pid in raw_ids if isinstance(pid, str) and pid not in ("nan", "None", "")))

    # Priors from previous season
    if (year - 1) >= 2015 and Path(f"data/silver/player_game_box_{year-1}_REGPST.csv").exists():
        prev_box = load_csv(f"data/silver/player_game_box_{year-1}_REGPST.csv")
        prev_box["player_id"] = prev_box["player_id"].astype(str)
        q_prev_by_player, q_league_prev, m_prev_avg_by_player = compute_prev_season_priors(prev_box)
    else:
        q_prev_by_player = pd.Series(dtype=float)
        q_league_prev = 0.0
        m_prev_avg_by_player = pd.Series(dtype=float)

    # Seed values:
    m_seed = players.to_series().map(m_prev_avg_by_player).fillna(0.0)

    # q seed from prior-season eff/min else league avg prev season
    q_seed = players.to_series().map(q_prev_by_player).fillna(q_league_prev)
    q_seed = q_seed.clip(lower=0.0)

    # ALSO define q_prev for the daily q formula (tau-weighted prior)
    q_prev = players.to_series().map(q_prev_by_player).fillna(q_league_prev)
    q_prev = q_prev.clip(lower=0.0)

    strength_seed = m_seed * q_seed

    # Injury episode summary (seed row injury defaults = 0)
    ep = build_injury_episode_summary(year)
    if len(ep) > 0:
        ep = ep.copy()
        ep["player_id"] = ep["player_id"].astype(str)
        ep["recent_dt"] = pd.to_datetime(ep["recent_injury_activity_date"], utc=True, errors="coerce").dt.floor("D")
        ep = ep.dropna(subset=["recent_dt"]).sort_values(["player_id", "recent_dt"], kind="stable")

    # Consecutive injury DNP streak per game
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

    # -----------------------
    # Build SEED ROWS
    # -----------------------
    seed = pd.DataFrame({
        "player_id": players.astype(str),
        "asof_ts": season_start_seed_ts,
        "asof_date": season_start_seed_ts.floor("D"),
    })
    seed["asof_ts"] = pd.to_datetime(seed["asof_ts"], utc=True)
    seed["asof_date_dt"] = pd.to_datetime(seed["asof_date"], utc=True)

    # current_team_id for seed: first observed team in season (from pg2)
    first_team = (
        pg2.dropna(subset=["team_id"])
        .sort_values(["player_id", "scheduled"], kind="stable")
        .groupby("player_id")["team_id"]
        .first()
    )
    seed["current_team_id"] = seed["player_id"].map(first_team)

    seed["m_ewma"] = seed["player_id"].map(m_seed).fillna(0.0)
    seed["q"] = seed["player_id"].map(q_seed).fillna(q_league_prev).clip(lower=0.0)
    seed["strength"] = seed["m_ewma"] * seed["q"]

    # Seed recency / injury fields = 0
    seed["days_since_first_report"] = 0
    seed["days_since_last_dnp"] = 0
    seed["consec_dnps"] = 0
    seed["played_last_game"] = 0
    seed["minutes_last_game"] = 0.0
    seed["days_since_last_played"] = 0
    seed["injury_present_flag"] = 0

    # metadata columns
    seed["last_game_date"] = pd.NaT
    seed["last_report_date"] = pd.NaT
    seed["last_dnp_date"] = pd.NaT
    seed["first_report_date_current_window"] = pd.NaT
    seed["episode_id"] = pd.NA

    # -----------------------
    # Daily panel rows (player_id x asof_ts end-of-day)
    # -----------------------
    panel = pd.MultiIndex.from_product([players.tolist(), asof_ts], names=["player_id", "asof_ts"]).to_frame(index=False)
    panel["player_id"] = panel["player_id"].astype(str)
    panel["asof_ts"] = pd.to_datetime(panel["asof_ts"], utc=True)
    panel["asof_date"] = panel["asof_ts"].dt.floor("D")
    panel["asof_date_dt"] = panel["asof_date"]

    # current_team_id from last observed game assignment <= asof_date
    team_hist = pg2[["player_id", "game_date", "team_id"]].dropna().copy()
    team_hist["game_date_dt"] = pd.to_datetime(team_hist["game_date"], utc=True, errors="coerce").dt.floor("D")
    team_hist = team_hist.dropna(subset=["game_date_dt"]).copy()

    if len(team_hist) > 0:
        panel = merge_asof_by(
            panel,
            team_hist[["player_id", "game_date_dt", "team_id"]],
            left_on="asof_date_dt",
            right_on="game_date_dt",
            by="player_id",
            direction="backward",
        )
        panel = panel.rename(columns={"team_id": "current_team_id"})
    else:
        panel["current_team_id"] = pd.NA

    # fill early-season missing team ids by backfill within player
    panel["current_team_id"] = panel.groupby("player_id")["current_team_id"].transform(lambda s: s.bfill())

    # last game info (minutes + played_last_game + consec_dnps)
    last_game = pg2[["player_id", "game_date", "minutes", "consec_dnp_at_game"]].copy()
    last_game["game_dt"] = pd.to_datetime(last_game["game_date"], utc=True, errors="coerce").dt.floor("D")
    last_game = last_game.dropna(subset=["game_dt"]).copy()

    if len(last_game) > 0:
        panel = merge_asof_by(
            panel,
            last_game[["player_id", "game_dt", "minutes", "consec_dnp_at_game"]],
            left_on="asof_date_dt",
            right_on="game_dt",
            by="player_id",
            direction="backward",
        )
    else:
        panel["minutes"] = np.nan
        panel["consec_dnp_at_game"] = np.nan
        panel["game_dt"] = pd.NaT

    panel["minutes_last_game"] = pd.to_numeric(panel["minutes"], errors="coerce").fillna(0.0)
    panel["played_last_game"] = (panel["minutes_last_game"] > 0).astype(int)
    panel["consec_dnps"] = pd.to_numeric(panel["consec_dnp_at_game"], errors="coerce").fillna(0).astype(int)
    panel["last_game_date"] = panel["game_dt"]

    panel = panel.drop(columns=["minutes", "consec_dnp_at_game", "game_dt"])

    # last played date (minutes>0)
    played_games = pg2[pg2["minutes"] > 0][["player_id", "game_date"]].copy()
    played_games["game_dt"] = pd.to_datetime(played_games["game_date"], utc=True, errors="coerce").dt.floor("D")
    played_games = played_games.dropna(subset=["game_dt"]).copy()

    if len(played_games) > 0:
        panel = merge_asof_by(
            panel,
            played_games[["player_id", "game_dt"]],
            left_on="asof_date_dt",
            right_on="game_dt",
            by="player_id",
            direction="backward",
        )
        panel = panel.rename(columns={"game_dt": "last_played_date"})
    else:
        panel["last_played_date"] = pd.NaT

    # No fallback to prior-season last played: days_since_last_played resets to 0 at season start.
    panel["days_since_last_played"] = (
        (panel["asof_date_dt"] - pd.to_datetime(panel["last_played_date"], utc=True, errors="coerce").dt.floor("D"))
        .dt.days
    )
    panel["days_since_last_played"] = panel["days_since_last_played"].fillna(0).astype(int)

    # cumulative eff/min through asof_date
    box_day = box.groupby(["player_id", "game_date"], as_index=False).agg(
        eff_day=("eff", "sum"),
        min_day=("minutes", "sum"),
    )
    box_day["game_dt"] = pd.to_datetime(box_day["game_date"], utc=True, errors="coerce").dt.floor("D")
    box_day = box_day.dropna(subset=["game_dt"]).sort_values(["player_id", "game_dt"], kind="stable")
    box_day["eff_cum"] = box_day.groupby("player_id")["eff_day"].cumsum()
    box_day["min_cum"] = box_day.groupby("player_id")["min_day"].cumsum()

    if len(box_day) > 0:
        panel = merge_asof_by(
            panel,
            box_day[["player_id", "game_dt", "eff_cum", "min_cum"]],
            left_on="asof_date_dt",
            right_on="game_dt",
            by="player_id",
            direction="backward",
        )
    else:
        panel["eff_cum"] = np.nan
        panel["min_cum"] = np.nan

    panel["eff_cum"] = pd.to_numeric(panel["eff_cum"], errors="coerce").fillna(0.0)
    panel["min_cum"] = pd.to_numeric(panel["min_cum"], errors="coerce").fillna(0.0)
    panel = panel.drop(columns=["game_dt"], errors="ignore")

    # m_ewma updated on game days; merged to daily
    m_game = pg2.groupby(["player_id", "game_date"], as_index=False).agg(minutes_game=("minutes", "sum"))
    m_game["game_dt"] = pd.to_datetime(m_game["game_date"], utc=True, errors="coerce").dt.floor("D")
    m_game = m_game.dropna(subset=["game_dt"]).sort_values(["player_id", "game_dt"], kind="stable")

    m_game["m_ewma_at_game"] = np.nan
    for pid, grp in m_game.groupby("player_id", sort=False):
        prev = float(m_seed.get(pid, 0.0))
        outs = []
        for mval in grp["minutes_game"].tolist():
            prev = LAMBDA_M * float(mval) + (1 - LAMBDA_M) * prev
            outs.append(prev)
        m_game.loc[grp.index, "m_ewma_at_game"] = outs

    if len(m_game) > 0:
        panel = merge_asof_by(
            panel,
            m_game[["player_id", "game_dt", "m_ewma_at_game"]],
            left_on="asof_date_dt",
            right_on="game_dt",
            by="player_id",
            direction="backward",
        )
    else:
        panel["m_ewma_at_game"] = np.nan

    panel["m_ewma"] = pd.to_numeric(panel["m_ewma_at_game"], errors="coerce")
    panel["m_ewma"] = panel["m_ewma"].fillna(panel["player_id"].map(m_seed)).fillna(0.0)
    panel = panel.drop(columns=["m_ewma_at_game", "game_dt"], errors="ignore")

    # q and strength
    panel["q_prev"] = panel["player_id"].map(q_prev).fillna(q_league_prev)
    panel["q"] = (TAU_Q * panel["q_prev"] + panel["eff_cum"]) / (TAU_Q + panel["min_cum"])
    panel["q"] = panel["q"].clip(lower=0.0)
    panel["strength"] = panel["m_ewma"] * panel["q"]

    # injury fields and flag (robustly computed from last_report/last_dnp)
    if len(ep) > 0:
        ep_key = ep.copy()
        ep_key["recent_dt"] = pd.to_datetime(ep_key["recent_injury_activity_date"], utc=True, errors="coerce").dt.floor("D")
        ep_key = ep_key.dropna(subset=["recent_dt"]).copy()

        panel = merge_asof_by(
            panel,
            ep_key[[
                "player_id", "recent_dt",
                "episode_id",
                "first_report_date_current_window",
                "last_report_date",
                "last_dnp_date",
            ]],
            left_on="asof_date_dt",
            right_on="recent_dt",
            by="player_id",
            direction="backward",
        )
    else:
        panel["episode_id"] = pd.NA
        panel["first_report_date_current_window"] = pd.NaT
        panel["last_report_date"] = pd.NaT
        panel["last_dnp_date"] = pd.NaT

    last_report_dt = pd.to_datetime(panel["last_report_date"], utc=True, errors="coerce").dt.floor("D")
    last_dnp_dt = pd.to_datetime(panel["last_dnp_date"], utc=True, errors="coerce").dt.floor("D")
    recent_dt = pd.concat([last_report_dt, last_dnp_dt], axis=1).max(axis=1)
    days_since_recent = (panel["asof_date_dt"] - recent_dt).dt.days

    panel["injury_present_flag"] = (recent_dt.notna() & (days_since_recent <= INJ_WINDOW_DAYS)).astype(int)

    first_report_dt = pd.to_datetime(panel["first_report_date_current_window"], utc=True, errors="coerce").dt.floor("D")
    panel["days_since_first_report"] = (panel["asof_date_dt"] - first_report_dt).dt.days
    panel["days_since_last_dnp"] = (panel["asof_date_dt"] - last_dnp_dt).dt.days
    panel.loc[last_dnp_dt.isna(), "days_since_last_dnp"] = 0

    panel.loc[panel["injury_present_flag"] == 0, "days_since_first_report"] = 0
    panel.loc[panel["injury_present_flag"] == 0, "days_since_last_dnp"] = 0
    panel.loc[panel["injury_present_flag"] == 0, "consec_dnps"] = 0

    panel["days_since_first_report"] = panel["days_since_first_report"].fillna(0).astype(int)
    panel["days_since_last_dnp"] = panel["days_since_last_dnp"].fillna(0).astype(int)

    # -----------------------
    # Combine seed + daily and write
    # -----------------------
    out_cols = [
        "player_id",
        "asof_ts",
        "current_team_id",
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
        # metadata
        "asof_date",
        "last_game_date",
        "last_report_date",
        "last_dnp_date",
        "first_report_date_current_window",
        "episode_id",
    ]

    daily_out = panel[out_cols].copy()
    seed_out = seed[out_cols].copy()

    out = pd.concat([seed_out, daily_out], ignore_index=True)
    out = out.sort_values(["player_id", "asof_ts"], kind="stable")

    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/silver/player_state_history_{year}.csv")
    out.to_csv(out_path, index=False)

    print(f"{year}: rows={len(out)} players={out['player_id'].nunique()} asof_ts={out['asof_ts'].nunique()}")
    print("seed_ts:", season_start_seed_ts)
    print("wrote:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)