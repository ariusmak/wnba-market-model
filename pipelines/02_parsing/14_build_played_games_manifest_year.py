import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_latest(pattern: str) -> dict:
    files = sorted(Path("data/bronze").glob(pattern))
    if not files:
        raise FileNotFoundError(f"Missing {pattern} in data/bronze")
    return json.loads(files[-1].read_text(encoding="utf-8"))


def main(year: int):
    print("STEP 0: starting")
    print("cwd:", os.getcwd())

    # Prove we can write to data/silver
    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "MANIFEST_SCRIPT_RAN.marker.txt"
    marker.write_text(f"ran at {datetime.now(timezone.utc).isoformat()}Z\n", encoding="utf-8")
    print("STEP 1: wrote marker:", marker.resolve())

    reg = load_latest(f"schedule_{year}_REG__*.json")
    pst = load_latest(f"schedule_{year}_PST__*.json")
    print("STEP 2: loaded schedules")

    rows = []
    for season_type, sched in [("REG", reg), ("PST", pst)]:
        for g in (sched.get("games") or []):
            gid = g.get("id")
            if not gid:
                continue
            rows.append({
                "season_year": year,
                "season_type": season_type,
                "game_id": gid,
                "status": (g.get("status") or "").lower(),
                "scheduled": g.get("scheduled"),
                "title": g.get("title"),
                "home_id": (g.get("home") or {}).get("id"),
                "away_id": (g.get("away") or {}).get("id"),
                "home_name": (g.get("home") or {}).get("name"),
                "away_name": (g.get("away") or {}).get("name"),
            })

    df = pd.DataFrame(rows)
    print("STEP 3: built schedule df rows:", len(df))

    played = df[df["status"] == "closed"].copy()
    print("STEP 4: played(closed) rows:", len(played))

    out = out_dir / f"played_games_{year}_REGPST.csv"
    played.sort_values(["scheduled", "game_id"]).to_csv(out, index=False)
    print("STEP 5: wrote:", out.resolve())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)