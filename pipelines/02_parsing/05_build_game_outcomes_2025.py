import pandas as pd
from pathlib import Path

games = pd.read_csv(Path("data/silver/games_2025_REG.csv"))
teams = pd.read_csv(Path("data/silver/game_teams_2025_REG.csv"))

# Split home/away
home = teams[teams["side"] == "home"].copy()
away = teams[teams["side"] == "away"].copy()

home = home.rename(columns={
    "team_id": "home_id",
    "team_name": "home_name",
    "market": "home_market",
    "alias": "home_alias",
    "points": "home_points",
})
away = away.rename(columns={
    "team_id": "away_id",
    "team_name": "away_name",
    "market": "away_market",
    "alias": "away_alias",
    "points": "away_points",
})

out = games.merge(home[["game_id","home_id","home_name","home_market","home_alias","home_points"]], on="game_id", how="left")
out = out.merge(away[["game_id","away_id","away_name","away_market","away_alias","away_points"]], on="game_id", how="left")

# Basic outcome fields
out["home_win"] = (out["home_points"] > out["away_points"]).astype("Int64")
out["point_diff"] = out["home_points"] - out["away_points"]

# Keep a clean column order
cols = [
    "game_id","sr_id","season_year","season_type","scheduled","status",
    "home_id","home_name","away_id","away_name",
    "home_points","away_points","home_win","point_diff",
    "attendance","venue_name"
]
out = out[cols].sort_values("scheduled", kind="stable")

Path("data/gold").mkdir(parents=True, exist_ok=True)
out_path = Path("data/gold/game_outcomes_2025_REG.csv")
out.to_csv(out_path, index=False)

print("Wrote:", out_path, "rows=", len(out))
print(out.head(10).to_string(index=False))