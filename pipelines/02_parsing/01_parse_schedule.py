import json
from pathlib import Path
import pandas as pd

# Pick the newest schedule file automatically
bronze_dir = Path("data/bronze")
files = sorted(bronze_dir.glob("schedule_2025_REG__*.json"))
if not files:
    raise FileNotFoundError("No schedule_2025_REG__*.json found in data/bronze")

path = files[-1]
data = json.loads(path.read_text(encoding="utf-8"))

rows = []
for g in data.get("games", []):
    rows.append({
        "game_id": g.get("id"),
        "scheduled": g.get("scheduled"),
        "status": g.get("status"),
        "home_id": (g.get("home") or {}).get("id"),
        "away_id": (g.get("away") or {}).get("id"),
        "home_name": (g.get("home") or {}).get("name"),
        "away_name": (g.get("away") or {}).get("name"),
        "season_year": (data.get("season") or {}).get("year"),
        "season_type": (data.get("season") or {}).get("type"),
    })

df = pd.DataFrame(rows)
print("Loaded:", path)
print("Shape:", df.shape)
print(df.head(10).to_string(index=False))