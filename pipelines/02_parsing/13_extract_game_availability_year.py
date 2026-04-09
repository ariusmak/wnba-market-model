import argparse
import json
from pathlib import Path

import pandas as pd


def load_latest_schedule(year: int, season_type: str) -> dict:
    season_type = season_type.upper()
    pattern = f"schedule_{year}_{season_type}__*.json"
    files = sorted(Path("data/bronze").glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Missing {pattern} in data/bronze. Run 00_backfill_schedule_year.py first."
        )
    return json.loads(files[-1].read_text(encoding="utf-8"))


def ts_from_name(name: str) -> str:
    # game_summary__{gid}__YYYYMMDDTHHMMSSZ.json
    parts = name.split("__")
    if len(parts) < 3:
        return ""
    return parts[2].replace(".json", "")


def pick_latest_game_summary_files(game_ids: set[str]) -> dict[str, Path]:
    """
    Return {game_id: latest_path} for game summaries found in bronze.
    Handles duplicates by choosing the latest timestamped file per game_id.
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
    if m is None:
        return None
    if isinstance(m, (int, float)):
        return float(m)
    if isinstance(m, str):
        s = m.strip()
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


def main(year: int):
    # Load both schedules and treat them as one continuous season
    reg = load_latest_schedule(year, "REG")
    pst = load_latest_schedule(year, "PST")

    games_reg = reg.get("games", []) or []
    games_pst = pst.get("games", []) or []

    # Build schedule maps and combined game id set
    sched_map = {}
    game_ids = set()

    def add_games(games, season_type):
        nonlocal game_ids, sched_map
        for g in games:
            gid = g.get("id")
            if not gid:
                continue
            game_ids.add(gid)
            sched_map[gid] = {
                "season_year": year,
                "season_type": season_type,
                "scheduled": g.get("scheduled"),
                "status": g.get("status"),
                "home_id": (g.get("home") or {}).get("id"),
                "away_id": (g.get("away") or {}).get("id"),
            }

    add_games(games_reg, "REG")
    add_games(games_pst, "PST")

    latest_files = pick_latest_game_summary_files(game_ids)
    if not latest_files:
        raise FileNotFoundError(
            f"No game_summary__<gid>__*.json files found in data/bronze for {year} REG/PST."
        )

    rows = []
    missing_players_games = 0

    for gid, path in latest_files.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = sched_map.get(gid, {})
        scheduled = meta.get("scheduled") or data.get("scheduled")
        status = meta.get("status") or data.get("status")
        season_type = meta.get("season_type")

        for side in ["home", "away"]:
            team = data.get(side) or {}
            team_id = team.get("id")
            opp_side = "away" if side == "home" else "home"
            opp_id = (data.get(opp_side) or {}).get("id")

            players = team.get("players") or []
            if not players:
                missing_players_games += 1
                continue

            for pl in players:
                st = pl.get("statistics") or {}
                minutes_raw = st.get("minutes")
                minutes = parse_minutes_to_float(minutes_raw)

                rows.append({
                    "season_year": year,
                    "season_type": season_type,   # REG or PST (kept for filtering later)
                    "game_id": gid,
                    "scheduled": scheduled,
                    "status": status,
                    "side": side,
                    "team_id": team_id,
                    "opponent_id": opp_id,
                    "player_id": pl.get("id"),
                    "player_name": pl.get("full_name"),
                    "played": pl.get("played"),
                    "active": pl.get("active"),
                    "starter": pl.get("starter"),
                    "on_court": pl.get("on_court"),
                    "minutes": minutes,
                    "minutes_raw": minutes_raw,
                    "not_playing_reason": pl.get("not_playing_reason"),
                    "not_playing_description": pl.get("not_playing_description"),
                    "bronze_file": path.name,
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["scheduled", "game_id", "team_id", "player_id"], kind="stable")

    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out = Path(f"data/silver/game_availability_{year}_REGPST.csv")
    df.to_csv(out, index=False)

    print(f"{year}: schedule_games REG={len(games_reg)} PST={len(games_pst)} combined={len(game_ids)}")
    print(f"summaries_found={len(latest_files)} rows={len(df)} games_missing_players={missing_players_games}")
    print("wrote:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)