import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def ts_from_name(name: str) -> str:
    # game_summary__{gid}__YYYYMMDDTHHMMSSZ.json
    parts = name.split("__")
    if len(parts) < 3:
        return ""
    return parts[2].replace(".json", "")


def pick_latest_game_summary_files(game_ids: set[str]) -> dict[str, Path]:
    """
    Return {game_id: latest_path} across any duplicates in bronze.
    """
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
    return {gid: item[1] for gid, item in best.items()}


def parse_minutes_to_float(m):
    """
    Sportradar often uses 'MM:SS'. Sometimes it's numeric.
    Returns minutes as float or None.
    """
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return None
    if isinstance(m, (int, float)):
        return float(m)
    if isinstance(m, str):
        s = m.strip()
        if not s:
            return None
        if ":" in s:
            mm, ss = s.split(":")[:2]
            try:
                return float(int(mm) + int(ss) / 60.0)
            except Exception:
                return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def safe_int(x):
    try:
        if x is None:
            return 0
        if isinstance(x, float) and pd.isna(x):
            return 0
        return int(x)
    except Exception:
        return 0


def compute_eff(stats: dict) -> int:
    """
    EFF = PTS + REB + AST + STL + BLK - (FGA-FGM) - (FTA-FTM) - TO
    """
    pts = safe_int(stats.get("points"))
    reb = safe_int(stats.get("rebounds"))
    ast = safe_int(stats.get("assists"))
    stl = safe_int(stats.get("steals"))
    blk = safe_int(stats.get("blocks"))
    fga = safe_int(stats.get("field_goals_att"))
    fgm = safe_int(stats.get("field_goals_made"))
    fta = safe_int(stats.get("free_throws_att"))
    ftm = safe_int(stats.get("free_throws_made"))
    tov = safe_int(stats.get("turnovers"))
    return pts + reb + ast + stl + blk - (fga - fgm) - (fta - ftm) - tov


def main(year: int):
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(
            f"Missing {played_path}. Run notebooks/14_build_played_games_manifest_year.py --year {year} first."
        )

    played = pd.read_csv(played_path)
    # manifest contains REG+PST closed games only
    game_ids = set(played["game_id"].astype(str).unique())

    latest = pick_latest_game_summary_files(game_ids)
    if not latest:
        raise FileNotFoundError(f"No bronze game_summary files found for year={year} matching played manifest.")

    # Map manifest metadata (authoritative scheduled timestamps)
    meta = played.set_index("game_id").to_dict(orient="index")

    rows = []
    missing_players_sides = 0
    missing_summaries = 0

    for gid in sorted(game_ids):
        if gid not in latest:
            missing_summaries += 1
            continue

        summ = json.loads(latest[gid].read_text(encoding="utf-8"))
        m = meta.get(gid, {})
        scheduled = m.get("scheduled") or summ.get("scheduled")  # schedule timestamps for consistency
        season_type = m.get("season_type")

        for side in ["home", "away"]:
            team = summ.get(side) or {}
            opp_side = "away" if side == "home" else "home"
            opp = summ.get(opp_side) or {}

            team_id = team.get("id")
            opp_id = opp.get("id")

            players = team.get("players") or []
            if not players:
                missing_players_sides += 1
                continue

            for pl in players:
                st = pl.get("statistics") or {}
                minutes_raw = st.get("minutes")
                minutes = parse_minutes_to_float(minutes_raw)

                eff = compute_eff(st)

                rows.append(
                    {
                        "season_year": year,
                        "season_type": season_type,   # REG/PST
                        "game_id": gid,
                        "scheduled": scheduled,        # UTC from schedule
                        "team_id": team_id,
                        "opponent_id": opp_id,
                        "is_home": 1 if side == "home" else 0,
                        "player_id": pl.get("id"),
                        "player_name": pl.get("full_name"),
                        "played": pl.get("played"),
                        "active": pl.get("active"),
                        "starter": pl.get("starter"),
                        "minutes": minutes,
                        "minutes_raw": minutes_raw,

                        # Box components used for EFF (kept explicit for auditing)
                        "points": safe_int(st.get("points")),
                        "rebounds": safe_int(st.get("rebounds")),
                        "assists": safe_int(st.get("assists")),
                        "steals": safe_int(st.get("steals")),
                        "blocks": safe_int(st.get("blocks")),
                        "turnovers": safe_int(st.get("turnovers")),
                        "fga": safe_int(st.get("field_goals_att")),
                        "fgm": safe_int(st.get("field_goals_made")),
                        "fta": safe_int(st.get("free_throws_att")),
                        "ftm": safe_int(st.get("free_throws_made")),

                        "eff": eff,
                        "bronze_file": latest[gid].name,
                    }
                )

    df = pd.DataFrame(rows)
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df.sort_values(["scheduled", "game_id", "team_id", "player_id"], kind="stable")

    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out = Path(f"data/silver/player_game_box_{year}_REGPST.csv")
    df.to_csv(out, index=False)

    print(f"{year}: played_games={len(game_ids)} summaries_found={len(latest)} rows={len(df)}")
    print("missing_summaries:", missing_summaries)
    print("missing_players_sides:", missing_players_sides)
    print("wrote:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)