"""
Polymarket WNBA Moneyline -> Sportradar game_id / team_id matcher
=================================================================
Fetches all 283 Polymarket WNBA moneyline markets and joins them to
Sportradar game_ids and team_ids from the silver layer.

Output: data/polymarket/wnba_2025_game_markets_matched.csv
        (same schema as data/kalshi/wnba_2025_game_markets_matched.csv)

Run with:
  /c/Users/arius/anaconda3/envs/kalshi-wnba/python.exe notebooks/polymarket/02_match_markets_to_sportradar.py
"""

import json
import os
import requests
import pandas as pd

GAMMA    = "https://gamma-api.polymarket.com"
WNBA_TAG = 100254
OUT_PATH = "data/polymarket/wnba_2025_game_markets_matched.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 1. Polymarket short name -> Sportradar full name + team_id
# ──────────────────────────────────────────────────────────────────────────────
# Derived from data/kalshi/wnba_2025_game_markets_matched.csv
TEAM_MAP = {
    "Aces":      ("Las Vegas",     "171b097d-01db-4ae8-9d56-035689402ec6"),
    "Dream":     ("Atlanta",       "5d70a9af-8c2b-4aec-9e68-9acc6ddb93e4"),
    "Fever":     ("Indiana",       "f073a15f-0486-4179-b0a3-dfd0294eb595"),
    "Liberty":   ("New York",      "08ed8274-e29f-4248-bc2e-83cc8ed18d75"),
    "Lynx":      ("Minnesota",     "6f017f37-be96-4bdc-b6d3-0a0429c72e89"),
    "Mercury":   ("Phoenix",       "0699edf3-5993-4182-b9b4-ec935cbd4fcc"),
    "Mystics":   ("Washington",    "5c0d47fe-8539-47b0-9f36-d0b3609ca89b"),
    "Sky":       ("Chicago",       "3c409388-ab73-4c7f-953d-3a71062240f6"),
    "Sparks":    ("Los Angeles",   "0a5ad38d-2fe3-43ba-894b-1ba3d5042ea9"),
    "Storm":     ("Seattle",       "d6a012ed-84aa-48d3-8265-2d3f3ff2199a"),
    "Sun":       ("Connecticut",   "a015b02d-845c-40c1-8ef4-844984f47e4d"),
    "Valkyries": ("Golden State",  "4f57ec40-0d35-4b59-bea0-9d040f0d2292"),
    "Wings":     ("Dallas",        "5f0b5caf-708b-4300-92f2-53b51d83ec06"),
}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Fetch all Polymarket WNBA moneyline markets
# ──────────────────────────────────────────────────────────────────────────────
print("Fetching Polymarket WNBA markets...")
all_markets = []
offset = 0
while True:
    r = requests.get(f"{GAMMA}/markets", params={"tag_id": WNBA_TAG, "limit": 100, "offset": offset})
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    all_markets.extend(batch)
    if len(batch) < 100:
        break
    offset += 100

moneyline = [m for m in all_markets if m.get("sportsMarketType") == "moneyline"]
print(f"  Total WNBA markets: {len(all_markets)}  |  moneyline: {len(moneyline)}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Load Sportradar games (unique game per row, keyed by game_ts)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading Sportradar 2025 silver data...")
sr = pd.read_csv(
    "data/silver/game_team_player_2025_REGPST.csv",
    usecols=["game_id", "game_ts", "game_date", "team_id", "opponent_team_id", "is_home"],
)
# One row per game: the home-team row gives us (home_team_id, away_team_id)
games = (
    sr[sr["is_home"] == 1][["game_id", "game_ts", "game_date", "team_id", "opponent_team_id"]]
    .drop_duplicates("game_id")
    .rename(columns={"team_id": "home_team_id", "opponent_team_id": "away_team_id"})
)
# Normalise timestamps: strip timezone suffix so we can string-match with Polymarket
games["game_ts_norm"] = games["game_ts"].str.replace(r"\+00.*$", "", regex=True).str.strip()
print(f"  Unique Sportradar games: {len(games)}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Build output rows
# ──────────────────────────────────────────────────────────────────────────────
rows = []
unmatched_ts  = []
unmatched_team = []

for m in moneyline:
    outcomes = json.loads(m.get("outcomes", "[]"))
    prices   = json.loads(m.get("outcomePrices", "[]"))
    clob_ids = json.loads(m.get("clobTokenIds", "[]"))

    if len(outcomes) != 2:
        continue

    short_a, short_b = outcomes[0], outcomes[1]

    # Resolve team names / IDs
    if short_a not in TEAM_MAP or short_b not in TEAM_MAP:
        unmatched_team.append((m.get("conditionId"), short_a, short_b))
        continue

    full_a, id_a = TEAM_MAP[short_a]
    full_b, id_b = TEAM_MAP[short_b]

    # Match Sportradar game by gameStartTime
    poly_ts = str(m.get("gameStartTime", "")).replace("+00", "").strip()
    sr_match = games[games["game_ts_norm"] == poly_ts]

    if sr_match.empty:
        # Fallback: match by date + team IDs (handles time-mismatch cases)
        poly_date = poly_ts[:10]
        sr_date_match = games[
            (games["game_date"].astype(str) == poly_date)
            & (
                ((games["home_team_id"] == id_a) & (games["away_team_id"] == id_b))
                | ((games["home_team_id"] == id_b) & (games["away_team_id"] == id_a))
            )
        ]
        if not sr_date_match.empty:
            game_id   = sr_date_match.iloc[0]["game_id"]
            game_date = sr_date_match.iloc[0]["game_date"]
        else:
            unmatched_ts.append((m.get("conditionId"), poly_ts, short_a, short_b))
            game_id   = None
            game_date = m.get("endDateIso")
    else:
        game_id   = sr_match.iloc[0]["game_id"]
        game_date = sr_match.iloc[0]["game_date"]

    # Derive winner (outcomePrices: "1" = winner, "0" = loser, else unresolved)
    if prices[0] == "1":
        winner = full_a
    elif prices[1] == "1":
        winner = full_b
    else:
        winner = None

    # Status
    if m.get("closed"):
        status = "finalized"
    elif m.get("active") and not m.get("closed"):
        status = "open"
    else:
        status = "unknown"

    rows.append({
        "condition_id":   m.get("conditionId"),
        "polymarket_slug": m.get("slug"),
        "game_id":        game_id,
        "game_date":      game_date,
        "open_time":      m.get("startDate"),
        "close_time":     m.get("closedTime"),
        "status":         status,
        "team_a":         full_a,
        "team_a_id":      id_a,
        "team_a_token_id": clob_ids[0] if clob_ids else None,
        "team_b":         full_b,
        "team_b_id":      id_b,
        "team_b_token_id": clob_ids[1] if clob_ids else None,
        "winner":         winner,
        "game_label":     None,
        "title":          m.get("question"),
    })

# ──────────────────────────────────────────────────────────────────────────────
# 5. Save
# ──────────────────────────────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
out_df.to_csv(OUT_PATH, index=False)

print(f"\nOutput rows: {len(out_df)}")
print(f"Saved to:    {OUT_PATH}")
print(f"\nUnmatched by timestamp ({len(unmatched_ts)}):")
for row in unmatched_ts[:10]:
    print(" ", row)
print(f"\nUnmatched by team name ({len(unmatched_team)}):")
for row in unmatched_team[:10]:
    print(" ", row)

print("\nSample output:")
print(out_df.head(5).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# 6. Match quality summary
# ──────────────────────────────────────────────────────────────────────────────
matched = out_df["game_id"].notna().sum()
print(f"\nMatch rate: {matched}/{len(out_df)} ({matched/len(out_df)*100:.1f}%) have a Sportradar game_id")
print(f"Status breakdown:\n{out_df['status'].value_counts().to_string()}")
