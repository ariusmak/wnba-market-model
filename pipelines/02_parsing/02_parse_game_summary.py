import json
from pathlib import Path
import pandas as pd

bronze_dir = Path("data/bronze")
files = sorted(bronze_dir.glob("game_summary__*.json"))
if not files:
    raise FileNotFoundError("No game_summary__*.json found in data/bronze")

path = files[-1]
data = json.loads(path.read_text(encoding="utf-8"))

# 1) Game-level row
game_row = {
    "game_id": data.get("id"),
    "sr_id": data.get("sr_id"),
    "status": data.get("status"),
    "scheduled": data.get("scheduled"),
    "attendance": data.get("attendance"),
    "season_year": (data.get("season") or {}).get("year"),
    "season_type": (data.get("season") or {}).get("type"),
    "venue_id": (data.get("venue") or {}).get("id"),
    "venue_name": (data.get("venue") or {}).get("name"),
}

# 2) Team rows (home + away)
team_rows = []
for side in ["home", "away"]:
    t = data.get(side) or {}
    team_rows.append({
        "game_id": data.get("id"),
        "side": side,
        "team_id": t.get("id"),
        "team_name": t.get("name"),
        "points": t.get("points"),
        "market": t.get("market"),  # sometimes present
        "alias": t.get("alias"),     # sometimes present
    })

df_game = pd.DataFrame([game_row])
df_teams = pd.DataFrame(team_rows)

print("Loaded:", path)
print("\nGAME:")
print(df_game.to_string(index=False))
print("\nTEAMS:")
print(df_teams.to_string(index=False))