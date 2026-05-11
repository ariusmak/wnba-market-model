import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd


def latest_daily_injury_files(year: int) -> list[Path]:
    latest: dict[str, tuple[str, Path]] = {}
    for path in Path("data/bronze").glob(f"daily_injuries__{year}-*__*.json"):
        parts = path.name.split("__")
        if len(parts) < 3:
            continue
        injury_date = parts[1]
        pulled = parts[2].removesuffix(".json")
        try:
            datetime.strptime(injury_date, "%Y-%m-%d")
            datetime.strptime(pulled, "%Y%m%dT%H%M%SZ")
        except ValueError:
            continue
        if injury_date not in latest or pulled > latest[injury_date][0]:
            latest[injury_date] = (pulled, path)
    return [item[1] for item in sorted(latest.values(), key=lambda item: item[1].name)]


def main(year: int):
    files = latest_daily_injury_files(year)
    if not files:
        raise FileNotFoundError(f"No daily_injuries__{year}-*.json files found in data/bronze")

    rows = []
    nonempty_days = 0

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        date = data.get("date") or path.name.split("__")[1]
        teams = data.get("teams") or []
        if teams:
            nonempty_days += 1

        for t in teams:
            team_id = t.get("id")
            team_name = (f"{t.get('market','')}".strip() + (" " if t.get("market") else "") + (t.get("name") or "")).strip()

            for p in (t.get("players") or []):
                player_id = p.get("id")
                player_name = p.get("full_name") or (f"{p.get('first_name','')} {p.get('last_name','')}".strip())

                for inj in (p.get("injuries") or []):
                    rows.append({
                        "asof_date": date,
                        "team_id": team_id,
                        "team_name": team_name,
                        "player_id": player_id,
                        "player_name": player_name,
                        "injury_id": inj.get("id"),
                        "desc": inj.get("desc"),
                        "status": inj.get("status"),
                        "comment": inj.get("comment"),
                        "start_date": inj.get("start_date"),
                        "update_date": inj.get("update_date"),
                        "bronze_file": path.name,
                    })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(
            subset=[
                "asof_date",
                "team_id",
                "player_id",
                "injury_id",
                "status",
                "start_date",
                "update_date",
            ],
            keep="last",
        )
    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/silver/injury_events_{year}.csv")
    df.to_csv(out_path, index=False)

    print("Daily files:", len(files))
    print("Non-empty days:", nonempty_days)
    print("Injury events rows:", len(df))
    print("Wrote:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)
