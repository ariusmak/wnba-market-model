import argparse
import json
from pathlib import Path

import pandas as pd


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


def extract_points(summary: dict) -> tuple[int | None, int | None]:
    home = summary.get("home") or {}
    away = summary.get("away") or {}
    hp = home.get("points")
    ap = away.get("points")
    try:
        hp = int(hp) if hp is not None else None
    except Exception:
        hp = None
    try:
        ap = int(ap) if ap is not None else None
    except Exception:
        ap = None
    return hp, ap


def main(year: int):
    played_path = Path(f"data/silver/played_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(played_path)

    played = pd.read_csv(played_path)
    game_ids = set(played["game_id"].astype(str).unique())

    latest = pick_latest_game_summary_files(game_ids)
    if len(latest) == 0:
        raise FileNotFoundError(f"No bronze game_summary files found for year={year}")

    # Map manifest metadata
    meta = played.set_index("game_id").to_dict(orient="index")

    rows = []
    missing_summary = 0
    missing_points = 0

    for gid in sorted(game_ids):
        m = meta.get(gid, {})
        if gid not in latest:
            missing_summary += 1
            continue

        summary = json.loads(latest[gid].read_text(encoding="utf-8"))
        hp, ap = extract_points(summary)
        if hp is None or ap is None:
            missing_points += 1

        home_id = m.get("home_id") or (summary.get("home") or {}).get("id")
        away_id = m.get("away_id") or (summary.get("away") or {}).get("id")
        scheduled = m.get("scheduled") or summary.get("scheduled")
        season_type = m.get("season_type")

        if (hp is not None) and (ap is not None):
            home_win = 1 if hp > ap else 0
            mov = abs(hp - ap)
        else:
            home_win = None
            mov = None

        rows.append({
            "season_year": year,
            "season_type": season_type,   # REG/PST
            "game_id": gid,
            "scheduled": scheduled,
            "home_id": home_id,
            "away_id": away_id,
            "home_points": hp,
            "away_points": ap,
            "home_win": home_win,
            "mov": mov,
            "bronze_file": latest[gid].name,
        })

    df = pd.DataFrame(rows)
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df.sort_values(["scheduled", "game_id"], kind="stable")

    Path("data/silver").mkdir(parents=True, exist_ok=True)
    out = Path(f"data/silver/game_outcomes_{year}_REGPST.csv")
    df.to_csv(out, index=False)

    print(f"{year}: played_games={len(game_ids)} outcomes_rows={len(df)}")
    print("missing_summary_files:", missing_summary)
    print("missing_points:", missing_points)
    print("wrote:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)