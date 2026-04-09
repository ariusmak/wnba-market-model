import json
from pathlib import Path
import pandas as pd

bronze_dir = Path("data/bronze")
out_dir = Path("data/silver")
out_dir.mkdir(parents=True, exist_ok=True)

files = sorted(bronze_dir.glob("game_summary__*.json"))
if not files:
    raise FileNotFoundError("No game_summary__*.json files found in data/bronze")

def extract_ts(path: Path) -> str:
    # filename like: game_summary__{gid}__20260224T224955Z.json
    name = path.stem  # without .json
    parts = name.split("__")
    return parts[-1] if len(parts) >= 3 else ""

# First pass: pick the latest file per game_id
latest_by_game = {}  # game_id -> (ts, path)

for path in files:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue

    gid = data.get("id")
    if not gid:
        continue

    ts = extract_ts(path)
    if gid not in latest_by_game or ts > latest_by_game[gid][0]:
        latest_by_game[gid] = (ts, path)

chosen_paths = [v[1] for v in latest_by_game.values()]
chosen_paths = sorted(chosen_paths, key=lambda p: extract_ts(p))

print(f"Found files total: {len(files)}")
print(f"Unique games: {len(chosen_paths)} (keeping latest per game_id)")

games_rows = []
teams_rows = []

for path in chosen_paths:
    data = json.loads(path.read_text(encoding="utf-8"))

    game_id = data.get("id")
    season = data.get("season") or {}
    venue = data.get("venue") or {}

    games_rows.append({
        "game_id": game_id,
        "sr_id": data.get("sr_id"),
        "status": data.get("status"),
        "scheduled": data.get("scheduled"),
        "attendance": data.get("attendance"),
        "season_year": season.get("year"),
        "season_type": season.get("type"),
        "venue_id": venue.get("id"),
        "venue_name": venue.get("name"),
        "bronze_file": path.name,
    })

    for side in ["home", "away"]:
        t = data.get(side) or {}
        teams_rows.append({
            "game_id": game_id,
            "side": side,
            "team_id": t.get("id"),
            "team_name": t.get("name"),
            "market": t.get("market"),
            "alias": t.get("alias"),
            "points": t.get("points"),
        })

df_games = pd.DataFrame(games_rows).sort_values("scheduled", kind="stable")
df_teams = pd.DataFrame(teams_rows)
df_teams["side"] = pd.Categorical(df_teams["side"], categories=["home", "away"], ordered=True)
df_teams = df_teams.sort_values(["game_id", "side"], kind="stable")

games_path = out_dir / "games_2025_REG.csv"
teams_path = out_dir / "game_teams_2025_REG.csv"

df_games.to_csv(games_path, index=False)
df_teams.to_csv(teams_path, index=False)

print("Wrote:", games_path, "rows=", len(df_games))
print("Wrote:", teams_path, "rows=", len(df_teams))
print(df_games.head(3).to_string(index=False))
print(df_teams.head(4).to_string(index=False))