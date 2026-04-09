"""
29_build_game_team_schedule_context.py

Builds game_team_schedule_context per the spec at:
  data/spec_sheets/game_team_schedule_context_spec.md

Output per year:
  data/silver_plus/game_team_schedule_context_{year}_REGPST.csv

Inputs:
  data/silver/played_franchise_games_{year}_REGPST.csv
  data/bronze/game_summary__*.json  (for venue city + lat/lng)

Locked hyperparams:
  HOME_RESET_DAYS = 4   (rest >= 4 -> origin = home city)
"""
import argparse
import json
import math
from pathlib import Path
import sys
from collections import defaultdict

import pandas as pd

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from srwnba.util.franchise import load_franchise_map, map_team_to_franchise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOME_RESET_DAYS = 4  # rest >= 4 -> origin = home city

# Standard DST UTC offsets (WNBA season May-Oct; DST in effect for all US zones)
# Arizona (Phoenix) does NOT observe DST -> UTC-7 year-round.
CITY_TZ_OFFSET: dict[str, int] = {
    "Atlanta": -4,
    "Chicago": -5,
    "Connecticut": -4,   # Sun play in Uncasville, CT
    "Uncasville": -4,
    "Dallas": -5,
    "Fort Worth": -5,
    "Indianapolis": -4,
    "Las Vegas": -7,
    "Los Angeles": -7,
    "Los Angeles (LA)": -7,
    "Minneapolis": -5,
    "New York": -4,
    "Brooklyn": -4,
    "Newark": -4,
    "Phoenix": -7,       # Arizona no DST
    "San Antonio": -5,
    "Seattle": -7,
    "Washington": -4,
    "San Francisco": -7,
    "Oakland": -7,
    "Toronto": -4,
    "Anaheim": -7,
    "Rosemont": -5,      # Chicago suburb (Allstate Arena)
    "Hoffman Estates": -5,
    "Southaven": -5,     # Memphis suburb
    "Jacksonville": -4,
    "Columbia": -4,      # South Carolina
    "Arlington": -5,     # Dallas Wings (College Park Center, Arlington TX)
    "Paradise": -7,      # Las Vegas Aces (T-Mobile Arena, Paradise NV)
    "White Plains": -4,  # NY Liberty (Westchester County Center)
    "Tempe": -7,         # Phoenix suburb, Arizona no DST
    "Fairfax": -4,       # DC suburb, Virginia (ET)
    "Uncasville": -4,    # Connecticut Sun (Mohegan Sun Arena)
    "Everett": -7,       # Seattle suburb (PT)
    "College Park": -4,  # Atlanta Dream (Gateway Center Arena, College Park GA)
    "Tulsa": -5,         # CT
    "Oklahoma City": -5, # CT
    "Sacramento": -7,    # PT
    "San Jose": -7,      # PT
    "Long Beach": -7,    # PT
    "Portland": -7,      # PT
    "Salt Lake City": -6, # MT (Utah observes DST) -- summer = MDT = UTC-6
    "St. Paul": -5,      # Minnesota Lynx (CT)
    "Baltimore": -4,     # ET
    "Boston": -4,        # ET
    "Bradenton": -4,     # Florida (ET)
    "Vancouver": -7,     # BC Canada (PT)
}


# Static city coordinates (lat, lng) — used as fallback when venue.location is absent
# (older Sportradar summaries lack location). All WNBA cities covered.
CITY_COORDS: dict[str, tuple[float, float]] = {
    "Atlanta": (33.7490, -84.3880),
    "College Park": (33.6534, -84.4496),  # Atlanta Dream, Gateway Center Arena
    "Chicago": (41.8781, -87.6298),
    "Rosemont": (41.9872, -87.8648),
    "Hoffman Estates": (42.0631, -88.1481),
    "Connecticut": (41.4868, -72.1073),   # Uncasville / Mohegan Sun
    "Uncasville": (41.4868, -72.1073),
    "Dallas": (32.7767, -96.7970),
    "Arlington": (32.7357, -97.1081),     # College Park Center
    "Fort Worth": (32.7555, -97.3308),
    "Indianapolis": (39.7684, -86.1581),
    "Las Vegas": (36.1699, -115.1398),
    "Paradise": (36.0840, -115.1522),     # T-Mobile Arena
    "Los Angeles": (34.0522, -118.2437),
    "Anaheim": (33.8366, -117.9143),
    "Long Beach": (33.7701, -118.1937),
    "Minneapolis": (44.9778, -93.2650),
    "St. Paul": (44.9537, -93.0900),
    "New York": (40.7128, -74.0060),
    "Brooklyn": (40.6782, -73.9442),
    "White Plains": (41.0340, -73.7629),
    "Newark": (40.7357, -74.1724),
    "Phoenix": (33.4484, -112.0740),
    "Tempe": (33.4255, -111.9400),
    "San Antonio": (29.4241, -98.4936),
    "Seattle": (47.6062, -122.3321),
    "Everett": (47.9790, -122.2021),
    "Washington": (38.9072, -77.0369),
    "Fairfax": (38.8462, -77.3064),
    "Baltimore": (39.2904, -76.6122),
    "San Francisco": (37.7749, -122.4194),
    "Oakland": (37.8044, -122.2712),
    "San Jose": (37.3382, -121.8863),
    "Sacramento": (38.5816, -121.4944),
    "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207),
    "Boston": (42.3601, -71.0589),
    "Portland": (45.5051, -122.6750),
    "Salt Lake City": (40.7608, -111.8910),
    "Oklahoma City": (35.4676, -97.5164),
    "Tulsa": (36.1540, -95.9928),
    "Bradenton": (27.4989, -82.5748),
    "Jacksonville": (30.3322, -81.6557),
    "Columbia": (34.0007, -81.0348),
    "Southaven": (34.9898, -90.0126),
}


def city_coords(city: str) -> tuple[float, float] | None:
    """Return (lat, lng) for a city, or None if unknown."""
    if city in CITY_COORDS:
        return CITY_COORDS[city]
    for k, v in CITY_COORDS.items():
        if city.startswith(k) or k.startswith(city):
            return v
    return None


def tz_offset(city: str) -> int:
    """Return DST UTC offset (hours) for a WNBA city. Raises if unknown."""
    if city in CITY_TZ_OFFSET:
        return CITY_TZ_OFFSET[city]
    # Try prefix matching (e.g. "Las Vegas" matches "Las Vegas, NV")
    for k, v in CITY_TZ_OFFSET.items():
        if city.startswith(k) or k.startswith(city):
            return v
    raise ValueError(
        f"Unknown city timezone: '{city}'. Add it to CITY_TZ_OFFSET."
    )


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Bronze scan helpers
# ---------------------------------------------------------------------------

def ts_from_name(name: str) -> str:
    parts = name.split("__")
    return parts[2].replace(".json", "") if len(parts) >= 3 else ""


def pick_latest_game_summary_files(game_ids: set) -> dict:
    best: dict = {}
    for p in Path("data/bronze").glob("game_summary__*__*.json"):
        parts = p.name.split("__")
        if len(parts) < 3:
            continue
        gid = parts[1]
        if gid not in game_ids:
            continue
        ts = ts_from_name(p.name)
        if gid not in best or ts > best[gid][0]:
            best[gid] = (ts, p)
    return {gid: v[1] for gid, v in best.items()}


def extract_venue_and_markets(path: Path) -> dict:
    """Return venue city + resolved coords (JSON loc preferred, static fallback), plus team markets."""
    d = json.loads(path.read_text(encoding="utf-8"))
    venue = d.get("venue") or {}
    loc = venue.get("location") or {}
    home = d.get("home") or {}
    away = d.get("away") or {}
    vcity = venue.get("city")

    # Coordinates: JSON location preferred; fall back to static dict by city name
    if loc.get("lat") and loc.get("lng"):
        vlat = float(loc["lat"])
        vlng = float(loc["lng"])
    elif vcity:
        coords = city_coords(vcity)
        vlat, vlng = (coords[0], coords[1]) if coords else (None, None)
    else:
        vlat, vlng = None, None

    return {
        "venue_city": vcity,
        "venue_lat": vlat,
        "venue_lng": vlng,
        "home_id": str(home.get("id", "")),
        "home_market": home.get("market"),
        "away_id": str(away.get("id", "")),
        "away_market": away.get("market"),
    }


# ---------------------------------------------------------------------------
# Build venue + team-home maps across ALL years (scan bronze once)
# ---------------------------------------------------------------------------

def build_maps(all_game_ids: set) -> tuple[dict, dict]:
    """
    Returns:
      game_venue_map: {game_id: {"city", "lat", "lng"}}
      team_home_map:  {team_id: {"city", "lat", "lng"}}  (from home-game venues)
    """
    latest = pick_latest_game_summary_files(all_game_ids)

    game_venue_map: dict = {}
    # team_id -> list of (city, lat, lng) from their HOME games
    team_home_candidates: dict = defaultdict(list)

    for gid, path in latest.items():
        info = extract_venue_and_markets(path)
        city = info["venue_city"]
        lat = info["venue_lat"]
        lng = info["venue_lng"]

        if city and lat is not None and lng is not None:
            game_venue_map[gid] = {"city": city, "lat": lat, "lng": lng}

        # Home team's market -> home city candidate
        if info["home_id"] and city and lat is not None:
            team_home_candidates[info["home_id"]].append((city, lat, lng))

    # For each team, pick their most common home venue city
    from collections import Counter
    team_home_map: dict = {}
    for tid, candidates in team_home_candidates.items():
        city_counts = Counter(c[0] for c in candidates)
        home_city = city_counts.most_common(1)[0][0]
        # Prefer JSON lat/lng; fall back to static dict
        city_latlng = [(lat, lng) for (c, lat, lng) in candidates
                       if c == home_city and lat is not None]
        if city_latlng:
            avg_lat = sum(c[0] for c in city_latlng) / len(city_latlng)
            avg_lng = sum(c[1] for c in city_latlng) / len(city_latlng)
        else:
            coords = city_coords(home_city)
            avg_lat, avg_lng = (coords[0], coords[1]) if coords else (None, None)
        team_home_map[tid] = {"city": home_city, "lat": avg_lat, "lng": avg_lng}

    return game_venue_map, team_home_map


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(year: int):
    played_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(f"Missing {played_path}")

    played = pd.read_csv(played_path)
    played["game_id"] = played["game_id"].astype(str)
    played["home_id"] = played["home_id"].astype(str)
    played["away_id"] = played["away_id"].astype(str)

    # Game date in Central time (safe for all WNBA game start times)
    played["game_ts"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    played["game_date"] = (
        played["game_ts"]
        .dt.tz_convert("America/Chicago")
        .dt.date
    )
    played["is_playoff"] = (played["season_type"].astype(str).str.upper() == "PST").astype(int)
    played = played.sort_values(["game_ts", "game_id"], kind="stable").reset_index(drop=True)

    map_df = load_franchise_map()

    # Build venue/home maps from ALL years' bronze files (scan once per run)
    # We collect game_ids across all years for an efficient single scan
    all_game_ids = set(played["game_id"].tolist())
    game_venue_map, team_home_map = build_maps(all_game_ids)

    missing_venue = 0

    # Build per-team game list for this year: {team_id: [(game_date, game_id), ...]}
    team_games: dict = defaultdict(list)
    for _, row in played.iterrows():
        gid = row["game_id"]
        gdate = row["game_date"]
        team_games[row["home_id"]].append((gdate, gid))
        team_games[row["away_id"]].append((gdate, gid))
    for tid in team_games:
        team_games[tid].sort(key=lambda x: x[0])

    # Game date lookup
    game_date_of: dict = {row["game_id"]: row["game_date"] for _, row in played.iterrows()}

    def get_venue(gid: str, home_tid: str) -> dict:
        """Return venue dict for game; fall back to home team home venue."""
        if gid in game_venue_map:
            return game_venue_map[gid]
        # Fallback: home team home city
        if home_tid in team_home_map:
            missing_venue
            return team_home_map[home_tid]
        return {}

    # For each game, record home_id for venue fallback
    game_home_id: dict = {row["game_id"]: row["home_id"] for _, row in played.iterrows()}

    out_rows = []

    for _, row in played.iterrows():
        gid = row["game_id"]
        gdate = row["game_date"]
        game_ts = row["game_ts"]
        is_playoff = int(row["is_playoff"])

        home_tid = row["home_id"]
        away_tid = row["away_id"]

        venue = get_venue(gid, home_tid)
        current_city = venue.get("city")
        current_lat = venue.get("lat")
        current_lng = venue.get("lng")

        for (team_id, opp_id, is_home) in [(home_tid, away_tid, 1), (away_tid, home_tid, 0)]:
            franchise_id = map_team_to_franchise(team_id, year, map_df)
            opp_franchise_id = map_team_to_franchise(opp_id, year, map_df)

            # Chronological game sequence for this team
            seq = team_games[team_id]
            idx = next((i for i, (d, g) in enumerate(seq) if g == gid), None)

            # Previous game
            prev_game_id = None
            prev_game_ts = None
            prev_game_date = None
            prev_game_city = None
            prev_lat = None
            prev_lng = None

            if idx is not None and idx > 0:
                prev_date, prev_gid = seq[idx - 1]
                prev_game_id = prev_gid
                prev_game_date = prev_date
                # Get prev game timestamp
                prev_rows = played[played["game_id"] == prev_gid]
                if len(prev_rows):
                    prev_game_ts = prev_rows.iloc[0]["game_ts"]
                prev_venue = get_venue(prev_gid, game_home_id.get(prev_gid, team_id))
                prev_game_city = prev_venue.get("city")
                prev_lat = prev_venue.get("lat")
                prev_lng = prev_venue.get("lng")

            # Rest features
            if prev_game_date is not None:
                days_rest = (gdate - prev_game_date).days - 1
            else:
                days_rest = 0

            is_b2b = 1 if days_rest == 0 else 0

            # games in last N days (strictly before current game date)
            games_last_4 = sum(
                1 for (d, g) in seq
                if g != gid and (gdate - d).days >= 1 and (gdate - d).days <= 4
            )
            games_last_7 = sum(
                1 for (d, g) in seq
                if g != gid and (gdate - d).days >= 1 and (gdate - d).days <= 7
            )

            # Travel origin rule
            home_info = team_home_map.get(team_id, {})
            home_city = home_info.get("city")
            home_lat = home_info.get("lat")
            home_lng = home_info.get("lng")

            if prev_game_date is None or days_rest >= HOME_RESET_DAYS:
                # Season opener or long break -> origin = home city
                origin_city = home_city
                origin_lat = home_lat
                origin_lng = home_lng
                origin_rule_used = "home_city"
            else:
                # Recent game -> origin = previous game city
                origin_city = prev_game_city if prev_game_city else home_city
                origin_lat = prev_lat if prev_lat is not None else home_lat
                origin_lng = prev_lng if prev_lng is not None else home_lng
                origin_rule_used = "previous_game_city"

            # Resolve coordinates via static dict if still missing
            if current_lat is None and current_city:
                c = city_coords(current_city)
                if c:
                    current_lat, current_lng = c
            if origin_lat is None and origin_city:
                c = city_coords(origin_city)
                if c:
                    origin_lat, origin_lng = c

            # Travel distance
            if (origin_lat is not None and origin_lng is not None
                    and current_lat is not None and current_lng is not None):
                travel_miles = haversine_miles(origin_lat, origin_lng, current_lat, current_lng)
            else:
                travel_miles = None

            # Timezone shift
            try:
                tz_current = tz_offset(current_city) if current_city else None
                tz_origin = tz_offset(origin_city) if origin_city else None
                if tz_current is not None and tz_origin is not None:
                    tz_shift = tz_current - tz_origin
                else:
                    tz_shift = None
            except ValueError as e:
                print(f"  WARNING: {e}")
                tz_shift = None

            out_rows.append({
                # A. Game context
                "game_id": gid,
                "game_ts": game_ts,
                "game_date": str(gdate),
                "season": year,
                "team_id": team_id,
                "franchise_id": franchise_id,
                "opponent_team_id": opp_id,
                "opponent_franchise_id": opp_franchise_id,
                "is_home": is_home,
                "is_playoff": is_playoff,
                # B. Rest / schedule
                "days_rest_pre": days_rest,
                "is_b2b_pre": is_b2b,
                "games_last_4_days_pre": games_last_4,
                "games_last_7_days_pre": games_last_7,
                # C. Travel / timezone
                "origin_city_pre": origin_city,
                "current_city_pre": current_city,
                "travel_miles_pre": round(travel_miles, 1) if travel_miles is not None else None,
                "timezone_shift_hours_pre": tz_shift,
                # D. Audit
                "previous_game_id": prev_game_id,
                "previous_game_ts": prev_game_ts,
                "previous_game_city": prev_game_city,
                "home_city": home_city,
                "origin_rule_used": origin_rule_used,
            })

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["game_ts", "game_id", "is_home"], ascending=[True, True, False], kind="stable")

    out_dir = Path("data/silver_plus")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"game_team_schedule_context_{year}_REGPST.csv"
    out.to_csv(out_path, index=False)

    b2b_count = out["is_b2b_pre"].sum()
    avg_rest = out[out["days_rest_pre"] > 0]["days_rest_pre"].mean()
    avg_miles = out["travel_miles_pre"].mean()
    null_tz = out["timezone_shift_hours_pre"].isna().sum()

    print(f"{year}: team_games={len(out)}  b2b_legs={b2b_count}  "
          f"avg_rest(>0)={avg_rest:.1f}d  avg_travel={avg_miles:.0f}mi  "
          f"null_tz={null_tz}")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)
