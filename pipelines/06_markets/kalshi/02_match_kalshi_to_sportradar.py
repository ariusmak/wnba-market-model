"""
Notebook 02: Match Kalshi WNBA Markets to SportRadar Game/Team IDs
==================================================================
Joins wnba_2025_game_markets.csv to the SR silver data on
  - date (from event_ticker) ± 1 day
  - team pair (unordered)

Adds columns: game_id, team_a_id, team_b_id

Outputs:
  data/kalshi/wnba_2025_game_markets_matched.csv
"""

import json
import re
import pandas as pd
from datetime import date, timedelta
from dateutil.parser import parse as parse_dt

# ── Step 1: Build SR game lookup (game_id, game_date, home/away team_id) ────
df_sr = pd.read_csv(
    "data/silver/game_team_player_2025_REGPST.csv",
    usecols=["game_id", "game_date", "team_id", "opponent_team_id", "is_home"],
)
sr_games = (
    df_sr[df_sr["is_home"] == 1]
    [["game_id", "game_date", "team_id", "opponent_team_id"]]
    .drop_duplicates()
    .rename(columns={"team_id": "home_team_id", "opponent_team_id": "away_team_id"})
    .copy()
)
sr_games["game_date"] = pd.to_datetime(sr_games["game_date"]).dt.date

# Key: frozenset of (home_team_id, away_team_id) → list of (game_date, game_id)
sr_by_teams: dict[frozenset, list[tuple]] = {}
for _, row in sr_games.iterrows():
    key = frozenset([row["home_team_id"], row["away_team_id"]])
    sr_by_teams.setdefault(key, []).append((row["game_date"], row["game_id"]))

print(f"SR games loaded: {len(sr_games)}")


# ── Step 2: Build Kalshi team name → SR team_id map ──────────────────────────
with open("data/bronze/league_hierarchy__20260224T224041Z.json") as f:
    hierarchy = json.load(f)

# market (city) → team_id  (2025 active teams only)
sr_team_ids = {}
for conf in hierarchy["conferences"]:
    for team in conf.get("teams", []):
        sr_team_ids[team["market"]] = team["id"]

# Kalshi uses market names directly; verify coverage
KALSHI_TEAMS = [
    "New York", "Chicago", "Washington", "Atlanta", "Connecticut",
    "Indiana", "Phoenix", "Los Angeles", "Las Vegas", "Golden State",
    "Dallas", "Minnesota", "Seattle",
]
print("\nKalshi -> SR team_id map:")
for t in KALSHI_TEAMS:
    tid = sr_team_ids.get(t, "NOT FOUND")
    print(f"  {t:20s} -> {tid}")


# ── Step 3: Parse game date from event_ticker ─────────────────────────────────
MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def parse_ticker_date(event_ticker: str) -> date | None:
    """Extract game date from KXWNBAGAME-25{MON}{DD}... ticker."""
    # Strip prefix
    m = re.match(r"KXWNBAGAME-(\d{2})([A-Z]{3})(\d{2})", event_ticker)
    if not m:
        return None
    yy, mon_str, dd = m.group(1), m.group(2), m.group(3)
    month = MONTH_MAP.get(mon_str)
    if not month:
        return None
    return date(2000 + int(yy), month, int(dd))


# ── Step 4: Load Kalshi game table and match ──────────────────────────────────
kalshi = pd.read_csv("data/kalshi/wnba_2025_game_markets.csv")
print(f"\nKalshi games: {len(kalshi)}")

results = []
unmatched = []

for _, row in kalshi.iterrows():
    et = row["event_ticker"]
    ta, tb = row["team_a"], row["team_b"]

    # Resolve team_ids (null for All-Star or unknown teams)
    ta_id = sr_team_ids.get(ta)
    tb_id = sr_team_ids.get(tb)

    # Parse game date from ticker
    ticker_date = parse_ticker_date(et)

    # Find matching SR game — prefer closest date, break ties by preferring
    # SR game date >= ticker date (markets open before the game).
    game_id = None
    if ta_id and tb_id and ticker_date:
        key = frozenset([ta_id, tb_id])
        candidates = sr_by_teams.get(key, [])
        within = [
            (abs((gdate - ticker_date).days), (gdate - ticker_date).days, gdate, gid)
            for gdate, gid in candidates
            if abs((gdate - ticker_date).days) <= 1
        ]
        if within:
            # Primary: smallest distance. Tiebreak: prefer game on/after ticker date
            # (positive offset), then latest date.
            best = min(within, key=lambda x: (x[0], x[1] < 0, -x[2].toordinal()))
            game_id = best[3]

    if game_id is None and ta_id and tb_id:
        unmatched.append(et)

    results.append({
        **row.to_dict(),
        "game_id":   game_id,
        "team_a_id": ta_id,
        "team_b_id": tb_id,
    })

df_out = pd.DataFrame(results)

# ── Step 5: Report ────────────────────────────────────────────────────────────
matched   = df_out["game_id"].notna().sum()
null_ids  = (df_out["team_a_id"].isna() | df_out["team_b_id"].isna()).sum()
unmatched_ct = df_out["game_id"].isna().sum() - null_ids

print(f"\nMatch results:")
print(f"  Matched:              {matched}")
print(f"  No SR game_id found:  {unmatched_ct}  (known SR games with no Kalshi match)")
print(f"  No team_id (All-Star):{null_ids}")

if unmatched:
    print(f"\nUnmatched event_tickers ({len(unmatched)}):")
    for u in unmatched:
        row = df_out[df_out["event_ticker"] == u].iloc[0]
        print(f"  {u}  ({row['team_a']} vs {row['team_b']}, ticker_date={parse_ticker_date(u)})")

# ── Step 6: Save ──────────────────────────────────────────────────────────────
out_path = "data/kalshi/wnba_2025_game_markets_matched.csv"
col_order = [
    "event_ticker", "game_id", "game_date", "open_time", "close_time",
    "status", "team_a", "team_a_id", "team_b", "team_b_id",
    "winner", "game_label", "title",
]
df_out[col_order].to_csv(out_path, index=False, encoding="utf-8")
print(f"\nSaved: {out_path}")
print(df_out[["event_ticker", "game_date", "team_a", "team_b", "game_id"]].head(10).to_string(index=False))
