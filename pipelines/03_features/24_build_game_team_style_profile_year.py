import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_div(n, d) -> float:
    try:
        if d is None:
            return 0.0
        d = float(d)
        if d <= 0 or (isinstance(d, float) and np.isnan(d)):
            return 0.0
        return float(n) / d
    except Exception:
        return 0.0


def ts_from_name(name: str) -> str:
    # game_summary__{gid}__YYYYMMDDTHHMMSSZ.json
    parts = name.split("__")
    if len(parts) < 3:
        return ""
    return parts[2].replace(".json", "")


def pick_latest_game_summary_files(game_ids: set[str]) -> dict[str, Path]:
    best: dict[str, tuple[str, Path]] = {}
    for p in Path("data/bronze").glob("game_summary__*__*.json"):
        parts = p.name.split("__")
        if len(parts) < 3:
            continue
        gid = parts[1]
        if gid not in game_ids:
            continue
        ts = ts_from_name(p.name)
        if (gid not in best) or (ts > best[gid][0]):
            best[gid] = (ts, p)
    return {gid: v[1] for gid, v in best.items()}


def to_int(x) -> int:
    try:
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        return int(x)
    except Exception:
        return 0


def get_stat(stats: dict, *keys: str) -> int:
    for k in keys:
        if k in stats and stats[k] is not None:
            return to_int(stats[k])
    return 0


def extract_team_inputs(team_block: dict) -> dict:
    """
    Extract raw inputs required by spec:
      FGA, 3PA, FTA, TO
    """
    stats = (team_block or {}).get("statistics") or {}
    fga = get_stat(stats, "field_goals_att")
    tpa = get_stat(stats, "three_points_att")  # attempts
    fta = get_stat(stats, "free_throws_att")
    # Turnovers: prefer total_turnovers; fall back to player+team; then 'turnovers'
    if stats.get("total_turnovers") is not None:
        tov = to_int(stats.get("total_turnovers"))
    else:
        pt = stats.get("player_turnovers")
        tt = stats.get("team_turnovers")
        if pt is not None and tt is not None:
            tov = to_int(pt) + to_int(tt)  # equals total_turnovers when present
        else:
            tov = get_stat(stats, "turnovers")
    return {"FGA": fga, "3PA": tpa, "FTA": fta, "TO": tov}


def style_from_totals(t: dict) -> dict:
    """
    Compute the 6 style metrics from cumulative totals (season-to-date):
      off_3pa_rate = sum 3PA / sum FGA
      def_3pa_allowed = sum Opp3PA / sum OppFGA
      off_2pa_rate = sum (FGA-3PA) / sum FGA
      def_2pa_allowed = sum (OppFGA-Opp3PA) / sum OppFGA
      off_tov_pct = sum TO / sum (FGA + 0.44*FTA + TO)
      def_forced_tov = sum OppTO / sum (OppFGA + 0.44*OppFTA + OppTO)
    """
    sum_fga = t["FGA"]
    sum_3pa = t["3PA"]
    sum_fta = t["FTA"]
    sum_to = t["TO"]

    sum_opp_fga = t["OppFGA"]
    sum_opp_3pa = t["Opp3PA"]
    sum_opp_fta = t["OppFTA"]
    sum_opp_to = t["OppTO"]

    off_3pa = safe_div(sum_3pa, sum_fga)
    def_3pa = safe_div(sum_opp_3pa, sum_opp_fga)

    off_2pa = safe_div((sum_fga - sum_3pa), sum_fga)
    def_2pa = safe_div((sum_opp_fga - sum_opp_3pa), sum_opp_fga)

    off_tov = safe_div(sum_to, (sum_fga + 0.44 * sum_fta + sum_to))
    def_forced = safe_div(sum_opp_to, (sum_opp_fga + 0.44 * sum_opp_fta + sum_opp_to))

    return {
        "off_3pa_rate": off_3pa,
        "def_3pa_allowed": def_3pa,
        "off_2pa_rate": off_2pa,
        "def_2pa_allowed": def_2pa,
        "off_tov_pct": off_tov,
        "def_forced_tov": def_forced,
    }


def compute_team_finals_from_realized(realized: pd.DataFrame) -> pd.DataFrame:
    """
    realized has one row per (game_id, team_id) with raw inputs & opponent inputs.
    Returns team-year final metrics for each team.
    """
    g = realized.groupby("team_id", as_index=False).agg(
        FGA=("FGA", "sum"),
        _3PA=("3PA", "sum"),
        FTA=("FTA", "sum"),
        TO=("TO", "sum"),
        OppFGA=("OppFGA", "sum"),
        Opp3PA=("Opp3PA", "sum"),
        OppFTA=("OppFTA", "sum"),
        OppTO=("OppTO", "sum"),
    )
    # compute metrics
    finals_rows = []
    for _, r in g.iterrows():
        totals = {
            "FGA": float(r["FGA"]),
            "3PA": float(r["_3PA"]),
            "FTA": float(r["FTA"]),
            "TO": float(r["TO"]),
            "OppFGA": float(r["OppFGA"]),
            "Opp3PA": float(r["Opp3PA"]),
            "OppFTA": float(r["OppFTA"]),
            "OppTO": float(r["OppTO"]),
        }
        m = style_from_totals(totals)
        finals_rows.append({"team_id": str(r["team_id"]), **m})
    return pd.DataFrame(finals_rows)


def mean_league_constants_from_team_finals(team_finals: pd.DataFrame) -> dict:
    """
    League init constant vector computed as UNWEIGHTED mean across teams' final metrics.
    (Consistent for 2015 init and for expansion-team fallback in later years.)
    """
    out = {}
    for col in [
        "off_3pa_rate", "def_3pa_allowed", "off_2pa_rate",
        "def_2pa_allowed", "off_tov_pct", "def_forced_tov"
    ]:
        out[col] = float(team_finals[col].mean()) if col in team_finals.columns and len(team_finals) else 0.0
    return out


def main(year: int):
    played_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(f"Missing {played_path}")

    played = pd.read_csv(played_path)
    played["game_id"] = played["game_id"].astype(str)
    played["game_ts"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    played["game_date"] = played["game_ts"].dt.date.astype(str)
    played["is_playoff"] = (played["season_type"].astype(str).str.upper() == "PST").astype(int)
    played = played.sort_values(["game_ts", "game_id"], kind="stable")

    # Need home/away ids for context
    if "home_id" not in played.columns or "away_id" not in played.columns:
        raise ValueError("played_games_{year}_REGPST.csv must include home_id and away_id.")

    game_ids = set(played["game_id"].unique())
    latest = pick_latest_game_summary_files(game_ids)
    if len(latest) == 0:
        raise FileNotFoundError(f"No bronze game_summary files found for year={year}")

    # Build realized per (game_id, team_id) raw inputs + opponent inputs
    realized_rows = []
    missing_summaries = 0

    meta = played.set_index("game_id").to_dict(orient="index")

    for gid in played["game_id"].tolist():
        if gid not in latest:
            missing_summaries += 1
            continue

        d = json.loads(latest[gid].read_text(encoding="utf-8"))
        home_blk = d.get("home") or {}
        away_blk = d.get("away") or {}

        h = extract_team_inputs(home_blk)
        a = extract_team_inputs(away_blk)

        m = meta[gid]
        game_ts = m["game_ts"]
        game_date = m["game_date"]
        is_playoff = int(m["is_playoff"])
        home_id = str(m["home_id"])
        away_id = str(m["away_id"])

        # home row
        realized_rows.append({
            "game_id": gid,
            "game_ts": game_ts,
            "game_date": game_date,
            "season": year,
            "team_id": home_id,
            "opponent_team_id": away_id,
            "is_home": 1,
            "is_playoff": is_playoff,
            "FGA": h["FGA"],
            "3PA": h["3PA"],
            "FTA": h["FTA"],
            "TO": h["TO"],
            "OppFGA": a["FGA"],
            "Opp3PA": a["3PA"],
            "OppFTA": a["FTA"],
            "OppTO": a["TO"],
            "bronze_file": latest[gid].name,
        })
        # away row
        realized_rows.append({
            "game_id": gid,
            "game_ts": game_ts,
            "game_date": game_date,
            "season": year,
            "team_id": away_id,
            "opponent_team_id": home_id,
            "is_home": 0,
            "is_playoff": is_playoff,
            "FGA": a["FGA"],
            "3PA": a["3PA"],
            "FTA": a["FTA"],
            "TO": a["TO"],
            "OppFGA": h["FGA"],
            "Opp3PA": h["3PA"],
            "OppFTA": h["FTA"],
            "OppTO": h["TO"],
            "bronze_file": latest[gid].name,
        })

    realized = pd.DataFrame(realized_rows)
    realized["game_ts"] = pd.to_datetime(realized["game_ts"], utc=True, errors="coerce")
    realized = realized.sort_values(["game_ts", "game_id", "team_id"], kind="stable")

    # --------
    # Determine priors for game 1
    # --------
    silver_plus = Path("data/silver_plus")
    silver_plus.mkdir(parents=True, exist_ok=True)

    if year == 2015:
        # Option 1: compute league init constants from 2015 full season (two-pass)
        team_finals_2015 = compute_team_finals_from_realized(realized)
        league_init = mean_league_constants_from_team_finals(team_finals_2015)
        prev_team_final = None
        prev_league_avg = league_init
    else:
        prev_final_path = silver_plus / f"team_style_profile_final_{year-1}.csv"
        if not prev_final_path.exists():
            raise FileNotFoundError(
                f"Missing {prev_final_path}. Build {year-1} first (incremental order required)."
            )
        prev_final = pd.read_csv(prev_final_path)
        prev_final["team_id"] = prev_final["team_id"].astype(str)

        prev_team_final = prev_final.set_index("team_id").to_dict(orient="index")
        prev_league_avg = mean_league_constants_from_team_finals(prev_final)

        # For completeness: league_init is the prev-season league-average vector (Option A)
        league_init = prev_league_avg

    # --------
    # Build pregame style values per (game_id, team_id)
    # --------
    # Maintain running totals per team for current season
    state = {}  # team_id -> {totals, games_played, last_game_id, last_game_ts}

    out_rows = []

    for _, r in realized.iterrows():
        team_id = str(r["team_id"])
        game_id = r["game_id"]
        game_ts = r["game_ts"]

        if team_id not in state:
            state[team_id] = {
                "games_played": 0,
                "last_completed_game_id": pd.NA,
                "last_completed_game_ts": pd.NaT,
                "totals": {
                    "FGA": 0.0, "3PA": 0.0, "FTA": 0.0, "TO": 0.0,
                    "OppFGA": 0.0, "Opp3PA": 0.0, "OppFTA": 0.0, "OppTO": 0.0,
                }
            }

        st = state[team_id]
        gp = int(st["games_played"])

        if gp == 0:
            # game 1 of the season for this team
            if year == 2015:
                prior = league_init
                prior_source = "league_init"
            else:
                if prev_team_final is not None and team_id in prev_team_final:
                    prior = prev_team_final[team_id]
                    prior_source = "prev_season_final"
                else:
                    prior = prev_league_avg
                    prior_source = "league_init"

            off_3pa_rate_pre = float(prior.get("off_3pa_rate", 0.0))
            def_3pa_allowed_pre = float(prior.get("def_3pa_allowed", 0.0))
            off_2pa_rate_pre = float(prior.get("off_2pa_rate", 0.0))
            def_2pa_allowed_pre = float(prior.get("def_2pa_allowed", 0.0))
            off_tov_pct_pre = float(prior.get("off_tov_pct", 0.0))
            def_forced_tov_pre = float(prior.get("def_forced_tov", 0.0))
        else:
            # games 2+: pure season-to-date through previous game
            prior_source = "season_to_date"
            cur = style_from_totals(st["totals"])
            off_3pa_rate_pre = cur["off_3pa_rate"]
            def_3pa_allowed_pre = cur["def_3pa_allowed"]
            off_2pa_rate_pre = cur["off_2pa_rate"]
            def_2pa_allowed_pre = cur["def_2pa_allowed"]
            off_tov_pct_pre = cur["off_tov_pct"]
            def_forced_tov_pre = cur["def_forced_tov"]

        out_rows.append({
            # context
            "game_id": game_id,
            "game_ts": game_ts,
            "game_date": r["game_date"],
            "season": int(r["season"]),
            "team_id": team_id,
            "opponent_team_id": str(r["opponent_team_id"]),
            "is_home": int(r["is_home"]),
            "is_playoff": int(r["is_playoff"]),

            # features
            "off_3pa_rate_pre": off_3pa_rate_pre,
            "def_3pa_allowed_pre": def_3pa_allowed_pre,
            "off_2pa_rate_pre": off_2pa_rate_pre,
            "def_2pa_allowed_pre": def_2pa_allowed_pre,
            "off_tov_pct_pre": off_tov_pct_pre,
            "def_forced_tov_pre": def_forced_tov_pre,

            # audit/debug
            "games_played_before_game": gp,
            "prior_source": prior_source,
            "last_completed_game_id": st["last_completed_game_id"],
            "last_completed_game_ts": st["last_completed_game_ts"],
        })

        # Update running totals with this game's realized inputs (after writing pregame values)
        t = st["totals"]
        t["FGA"] += float(r["FGA"])
        t["3PA"] += float(r["3PA"])
        t["FTA"] += float(r["FTA"])
        t["TO"] += float(r["TO"])
        t["OppFGA"] += float(r["OppFGA"])
        t["Opp3PA"] += float(r["Opp3PA"])
        t["OppFTA"] += float(r["OppFTA"])
        t["OppTO"] += float(r["OppTO"])

        st["games_played"] = gp + 1
        st["last_completed_game_id"] = game_id
        st["last_completed_game_ts"] = game_ts

    out = pd.DataFrame(out_rows)
    out["game_ts"] = pd.to_datetime(out["game_ts"], utc=True, errors="coerce")
    out = out.sort_values(["game_ts", "game_id", "team_id"], kind="stable")

    out_path = silver_plus / f"game_team_style_profile_{year}_REGPST.csv"
    out.to_csv(out_path, index=False)

    # --------
    # Write final team style values for this year (for next season game-1 priors)
    # --------
    finals_df = compute_team_finals_from_realized(realized)
    finals_df.insert(0, "season", year)
    finals_path = silver_plus / f"team_style_profile_final_{year}.csv"
    finals_df.to_csv(finals_path, index=False)

    print(f"{year}: games={out['game_id'].nunique()} team_games={len(out)} teams={out['team_id'].nunique()}")
    print("missing_summaries:", missing_summaries)
    if year == 2015:
        print("league_init_constants_from_2015_finals:", league_init)
    print("wrote:", out_path)
    print("wrote:", finals_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)