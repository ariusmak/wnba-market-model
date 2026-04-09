"""
Notebook 01: Pull WNBA 2025 Game Outcome Markets from Kalshi
============================================================
Fetches all markets in the KXWNBAGAME series, deduplicates by event,
and builds a clean game-level table with: event_ticker, game_date,
home_team, away_team, winning_team, and status.

Outputs:
  data/kalshi/wnba_2025_game_markets.csv  – game-level table
  data/kalshi/wnba_2025_markets_raw.json  – raw market data
"""

import json
import os
import requests
import pandas as pd
from datetime import datetime, timezone

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL    = "https://api.elections.kalshi.com/trade-api/v2"
SERIES      = "KXWNBAGAME"
OUTPUT_DIR  = "data/kalshi"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Step 1: Fetch all markets for KXWNBAGAME ─────────────────────────────────
def fetch_all_markets(series_ticker: str) -> list[dict]:
    markets, cursor = [], None
    while True:
        params = {"series_ticker": series_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(f"{BASE_URL}/markets", params=params)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break
    return markets


print("Fetching KXWNBAGAME markets from Kalshi...")
markets = fetch_all_markets(SERIES)
print(f"  Total markets fetched: {len(markets)}")

# Save raw
raw_path = os.path.join(OUTPUT_DIR, "wnba_2025_markets_raw.json")
with open(raw_path, "w", encoding="utf-8") as f:
    json.dump(markets, f, indent=2)
print(f"  Raw data saved: {raw_path}")


# ── Step 2: Build game-level table ───────────────────────────────────────────
# Each game event has 2 markets (one per team). Group by event_ticker and
# collect both teams. The winning team has result == "yes".

events: dict[str, dict] = {}

for m in markets:
    et = m["event_ticker"]
    team = m["yes_sub_title"]
    result = m.get("result", "")

    if et not in events:
        # Parse date from open_time (first available timestamp)
        # Prefer open_time as game date proxy
        open_ts = m.get("open_time", "")
        close_ts = m.get("close_time", "")
        ts_str = open_ts if open_ts else close_ts
        game_date = None
        if ts_str:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            game_date = dt.date().isoformat()

        events[et] = {
            "event_ticker": et,
            "title": m["title"],
            "game_date": game_date,
            "open_time": open_ts,
            "close_time": close_ts,
            "status": m["status"],
            "teams": [],
            "winner": None,
        }

    events[et]["teams"].append(team)
    if result == "yes":
        events[et]["winner"] = team

# Build DataFrame
rows = []
for et, ev in events.items():
    teams = ev["teams"]
    # title format: "Team A vs Team B (Game N) Winner?"
    title = ev["title"]
    # Parse home/away from title (first listed is home in Kalshi convention)
    # event_ticker format: KXWNBAGAME-25{MON}{DD}{TEAM1}{TEAM2}
    # The title gives "X vs Y", where X is listed first
    parts = title.replace(" Winner?", "").split(" vs ")
    if len(parts) == 2:
        team_a = parts[0].strip()
        # Strip playoff suffix like " (Game 3)"
        team_b = parts[1].split("(")[0].strip()
    else:
        team_a = teams[0] if len(teams) > 0 else ""
        team_b = teams[1] if len(teams) > 1 else ""

    # Extract game label (e.g., "Game 3") if present
    game_label = None
    if "(" in title and "Game" in title:
        import re
        match = re.search(r"\(Game (\d+)\)", title)
        if match:
            game_label = int(match.group(1))

    rows.append({
        "event_ticker": et,
        "game_date":    ev["game_date"],
        "open_time":    ev["open_time"],
        "close_time":   ev["close_time"],
        "status":       ev["status"],
        "team_a":       team_a,
        "team_b":       team_b,
        "winner":       ev["winner"],
        "game_label":   game_label,
        "title":        ev["title"],
    })

df = pd.DataFrame(rows).sort_values("game_date").reset_index(drop=True)

# ── Step 3: Save ──────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "wnba_2025_game_markets.csv")
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Saved: {out_path}")


# ── Step 4: Print summary ─────────────────────────────────────────────────────
print(f"\n=== WNBA 2025 Game Markets: {len(df)} unique games ===")
with pd.option_context("display.max_rows", None, "display.max_colwidth", 50, "display.width", 120):
    cols = ["event_ticker", "game_date", "team_a", "team_b", "winner", "status"]
    print(df[cols].to_string(index=False).encode("ascii", errors="replace").decode())

print(f"\nDate range: {df['game_date'].min()} -> {df['game_date'].max()}")
print(f"Status breakdown:\n{df['status'].value_counts().to_string()}")

teams_seen = pd.Series(df["team_a"].tolist() + df["team_b"].tolist()).value_counts()
print(f"\nGames per team:\n{teams_seen.to_string()}")
