import argparse
import json
from pathlib import Path
import math

import pandas as pd


# Spec: half-life 7 games
LAMBDA = 1 - 2 ** (-1 / 7)


def safe_div(n, d) -> float:
    try:
        if d is None or d == 0 or (isinstance(d, float) and pd.isna(d)) or d <= 0:
            return 0.0
        return float(n) / float(d)
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


def extract_team_totals(team_block: dict) -> dict:
    """
    Extract totals needed for four factors + possessions.
    Assumes Sportradar summary structure (team.statistics exists).
    """
    stats = (team_block or {}).get("statistics") or {}

    pts = stats.get("points") or (team_block or {}).get("points")
    fga = stats.get("field_goals_att")
    fgm = stats.get("field_goals_made")
    tpm = stats.get("three_points_made")
    fta = stats.get("free_throws_att")
    tov = stats.get("turnovers")
    orb = stats.get("offensive_rebounds")
    drb = stats.get("defensive_rebounds")

    # Convert to ints safely
    def to_int(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return 0
            return int(x)
        except Exception:
            return 0

    return {
        "pts": to_int(pts),
        "fga": to_int(fga),
        "fgm": to_int(fgm),
        "tpm": to_int(tpm),
        "fta": to_int(fta),
        "tov": to_int(tov),
        "orb": to_int(orb),
        "drb": to_int(drb),
    }


def compute_metrics(team: dict, opp: dict) -> dict:
    """
    Compute per spec:
      Poss = FGA - ORB + TO + 0.44*FTA
      ORtg = 100*PTS/Poss
      DRtg = 100*OppPTS/Poss_opp
      NetRtg = ORtg - DRtg
      eFG = (FGM + 0.5*3PM)/FGA
      TOV% = TO/Poss
      ORB% = ORB/(ORB + OppDRB)
      FTr = FTA/FGA
    Safe-div: denom<=0 => 0.
    """
    poss = team["fga"] - team["orb"] + team["tov"] + 0.44 * team["fta"]
    poss_opp = opp["fga"] - opp["orb"] + opp["tov"] + 0.44 * opp["fta"]

    ortg = 100.0 * safe_div(team["pts"], poss)
    drtg = 100.0 * safe_div(opp["pts"], poss_opp)
    net = ortg - drtg

    efg = safe_div(team["fgm"] + 0.5 * team["tpm"], team["fga"])
    tov_pct = safe_div(team["tov"], poss)
    orb_pct = safe_div(team["orb"], team["orb"] + opp["drb"])
    ftr = safe_div(team["fta"], team["fga"])

    return {
        "poss": float(poss) if poss is not None else 0.0,
        "poss_opp": float(poss_opp) if poss_opp is not None else 0.0,
        "ortg": ortg,
        "drtg": drtg,
        "net_rtg": net,
        "efg": efg,
        "tov_pct": tov_pct,
        "orb_pct": orb_pct,
        "ftr": ftr,
    }


def main(year: int):
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(f"Missing {played_path}. Build played manifest first.")

    played = pd.read_csv(played_path)
    played["game_id"] = played["game_id"].astype(str)
    played["game_ts"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    played["is_playoff"] = (played["season_type"].astype(str).str.upper() == "PST").astype(int)

    # Sort chronological
    played = played.sort_values(["game_ts", "game_id"], kind="stable")

    game_ids = set(played["game_id"].unique())
    latest = pick_latest_game_summary_files(game_ids)
    if len(latest) == 0:
        raise FileNotFoundError(f"No bronze game_summary files found for year={year}")

    # Precompute realized team-game stats for every game_id (home+away)
    realized_rows = []
    missing_summaries = 0

    # We'll need home/away team IDs from schedule manifest
    # played has home_id, away_id columns
    if "home_id" not in played.columns or "away_id" not in played.columns:
        raise ValueError("played_games manifest must include home_id and away_id columns.")

    meta = played.set_index("game_id").to_dict(orient="index")

    for gid in played["game_id"].tolist():
        if gid not in latest:
            missing_summaries += 1
            continue

        d = json.loads(latest[gid].read_text(encoding="utf-8"))
        home = d.get("home") or {}
        away = d.get("away") or {}

        h_tot = extract_team_totals(home)
        a_tot = extract_team_totals(away)

        h_metrics = compute_metrics(h_tot, a_tot)
        a_metrics = compute_metrics(a_tot, h_tot)

        # Meta
        m = meta.get(gid, {})
        game_ts = m.get("game_ts")
        is_playoff = int(m.get("is_playoff", 0))
        home_id = str(m.get("home_id"))
        away_id = str(m.get("away_id"))

        realized_rows.append({
            "season": year,
            "game_id": gid,
            "game_ts": game_ts,
            "team_id": home_id,
            "opponent_team_id": away_id,
            "is_home": 1,
            "is_playoff": is_playoff,
            **{f"{k}_game": v for k, v in h_metrics.items()},
            "bronze_file": latest[gid].name,
        })
        realized_rows.append({
            "season": year,
            "game_id": gid,
            "game_ts": game_ts,
            "team_id": away_id,
            "opponent_team_id": home_id,
            "is_home": 0,
            "is_playoff": is_playoff,
            **{f"{k}_game": v for k, v in a_metrics.items()},
            "bronze_file": latest[gid].name,
        })

    realized = pd.DataFrame(realized_rows)
    realized = realized.sort_values(["game_ts", "game_id", "is_home"], kind="stable")

    # EWMA state per team
    # per spec initialize to 0 for first game for that team
    state = {}  # team_id -> dict of ewmas and audit info

    out_rows = []

    for _, row in realized.iterrows():
        team_id = str(row["team_id"])
        game_id = row["game_id"]
        game_ts = row["game_ts"]

        if team_id not in state:
            state[team_id] = {
                "net_rtg_ewma": 0.0,
                "efg_ewma": 0.0,
                "tov_pct_ewma": 0.0,
                "orb_pct_ewma": 0.0,
                "ftr_ewma": 0.0,
                "last_completed_game_id": pd.NA,
                "last_completed_game_ts": pd.NaT,
                "net_rtg_last_game": 0.0,
                "efg_last_game": 0.0,
                "tov_pct_last_game": 0.0,
                "orb_pct_last_game": 0.0,
                "ftr_last_game": 0.0,
            }

        st = state[team_id]

        # Pre-game EWMA features (before updating with this game's realized stats)
        out_rows.append({
            "season": int(row["season"]),
            "game_id": game_id,
            "game_ts": game_ts,
            "team_id": team_id,
            "opponent_team_id": str(row["opponent_team_id"]),
            "is_home": int(row["is_home"]),
            "is_playoff": int(row["is_playoff"]),

            "net_rtg_ewma_pre": st["net_rtg_ewma"],
            "efg_ewma_pre": st["efg_ewma"],
            "tov_pct_ewma_pre": st["tov_pct_ewma"],
            "orb_pct_ewma_pre": st["orb_pct_ewma"],
            "ftr_ewma_pre": st["ftr_ewma"],

            # audit/debug
            "last_completed_game_id": st["last_completed_game_id"],
            "last_completed_game_ts": st["last_completed_game_ts"],
            "net_rtg_last_game": st["net_rtg_last_game"],
            "efg_last_game": st["efg_last_game"],
            "tov_pct_last_game": st["tov_pct_last_game"],
            "orb_pct_last_game": st["orb_pct_last_game"],
            "ftr_last_game": st["ftr_last_game"],

            # include realized game metrics (optional but helpful for QC)
            "net_rtg_game": row["net_rtg_game"],
            "efg_game": row["efg_game"],
            "tov_pct_game": row["tov_pct_game"],
            "orb_pct_game": row["orb_pct_game"],
            "ftr_game": row["ftr_game"],
            "poss_game": row["poss_game"],
            "poss_opp_game": row["poss_opp_game"],
            "ortg_game": row["ortg_game"],
            "drtg_game": row["drtg_game"],

            "bronze_file": row["bronze_file"],
        })

        # Update EWMA with realized stats
        st["net_rtg_ewma"] = LAMBDA * float(row["net_rtg_game"]) + (1 - LAMBDA) * st["net_rtg_ewma"]
        st["efg_ewma"] = LAMBDA * float(row["efg_game"]) + (1 - LAMBDA) * st["efg_ewma"]
        st["tov_pct_ewma"] = LAMBDA * float(row["tov_pct_game"]) + (1 - LAMBDA) * st["tov_pct_ewma"]
        st["orb_pct_ewma"] = LAMBDA * float(row["orb_pct_game"]) + (1 - LAMBDA) * st["orb_pct_ewma"]
        st["ftr_ewma"] = LAMBDA * float(row["ftr_game"]) + (1 - LAMBDA) * st["ftr_ewma"]

        # Update audit info with realized game results
        st["last_completed_game_id"] = game_id
        st["last_completed_game_ts"] = game_ts
        st["net_rtg_last_game"] = float(row["net_rtg_game"])
        st["efg_last_game"] = float(row["efg_game"])
        st["tov_pct_last_game"] = float(row["tov_pct_game"])
        st["orb_pct_last_game"] = float(row["orb_pct_game"])
        st["ftr_last_game"] = float(row["ftr_game"])

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["game_ts", "game_id", "team_id", "is_home"], kind="stable")

    # Ensure output directory exists
    out_dir = Path("data/silver_plus")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"game_team_recent_form_{year}_REGPST.csv"
    out.to_csv(out_path, index=False)

    print(f"{year}: games={out['game_id'].nunique()} team_games={len(out)} teams={out['team_id'].nunique()}")
    print("missing_summaries:", missing_summaries)
    print("lambda:", LAMBDA)
    print("wrote:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)