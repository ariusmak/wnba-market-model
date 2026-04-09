"""
26_build_franchise_style_profile.py

Franchise-aware style profile builder. Outputs NEW files (does not overwrite old ones).

Outputs per year:
  data/silver_plus/game_franchise_style_profile_{year}_REGPST.csv
  data/silver_plus/franchise_style_profile_final_{year}.csv

Run sequentially 2015->2025 (year y uses y-1 finals for game-1 priors).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from srwnba.util.franchise import load_franchise_map, map_team_to_franchise


# ---------------------------------------------------------------------------
# helpers (identical logic to notebook 24 but keyed by franchise_id)
# ---------------------------------------------------------------------------

def safe_div(n, d) -> float:
    try:
        if d is None:
            return 0.0
        d = float(d)
        if d <= 0 or (isinstance(d, float) and np.isnan(d)):
            return 0.0
        return float(n) / d
    except Exception:
        return 0.0


def ts_from_name(name: str) -> str:
    parts = name.split("__")
    if len(parts) < 3:
        return ""
    return parts[2].replace(".json", "")


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
        if (gid not in best) or (ts > best[gid][0]):
            best[gid] = (ts, p)
    return {gid: v[1] for gid, v in best.items()}


def to_int(x) -> int:
    try:
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        return int(x)
    except Exception:
        return 0


def get_stat(stats: dict, *keys: str) -> int:
    for k in keys:
        if k in stats and stats[k] is not None:
            return to_int(stats[k])
    return 0


def extract_team_inputs(team_block: dict) -> dict:
    stats = (team_block or {}).get("statistics") or {}
    fga = get_stat(stats, "field_goals_att")
    tpa = get_stat(stats, "three_points_att")
    fta = get_stat(stats, "free_throws_att")
    if stats.get("total_turnovers") is not None:
        tov = to_int(stats.get("total_turnovers"))
    else:
        pt = stats.get("player_turnovers")
        tt = stats.get("team_turnovers")
        if pt is not None and tt is not None:
            tov = to_int(pt) + to_int(tt)
        else:
            tov = get_stat(stats, "turnovers")
    return {"FGA": fga, "3PA": tpa, "FTA": fta, "TO": tov}


def style_from_totals(t: dict) -> dict:
    return {
        "off_3pa_rate": safe_div(t["3PA"], t["FGA"]),
        "def_3pa_allowed": safe_div(t["Opp3PA"], t["OppFGA"]),
        "off_2pa_rate": safe_div(t["FGA"] - t["3PA"], t["FGA"]),
        "def_2pa_allowed": safe_div(t["OppFGA"] - t["Opp3PA"], t["OppFGA"]),
        "off_tov_pct": safe_div(t["TO"], t["FGA"] + 0.44 * t["FTA"] + t["TO"]),
        "def_forced_tov": safe_div(t["OppTO"], t["OppFGA"] + 0.44 * t["OppFTA"] + t["OppTO"]),
    }


def zero_totals() -> dict:
    return {"FGA": 0.0, "3PA": 0.0, "FTA": 0.0, "TO": 0.0,
            "OppFGA": 0.0, "Opp3PA": 0.0, "OppFTA": 0.0, "OppTO": 0.0}


def compute_franchise_finals(realized: pd.DataFrame) -> pd.DataFrame:
    g = realized.groupby("franchise_id", as_index=False).agg(
        FGA=("FGA", "sum"),
        _3PA=("3PA", "sum"),
        FTA=("FTA", "sum"),
        TO=("TO", "sum"),
        OppFGA=("OppFGA", "sum"),
        Opp3PA=("Opp3PA", "sum"),
        OppFTA=("OppFTA", "sum"),
        OppTO=("OppTO", "sum"),
    )
    rows = []
    for _, r in g.iterrows():
        totals = {
            "FGA": float(r["FGA"]), "3PA": float(r["_3PA"]),
            "FTA": float(r["FTA"]), "TO": float(r["TO"]),
            "OppFGA": float(r["OppFGA"]), "Opp3PA": float(r["Opp3PA"]),
            "OppFTA": float(r["OppFTA"]), "OppTO": float(r["OppTO"]),
        }
        rows.append({"franchise_id": str(r["franchise_id"]), **style_from_totals(totals)})
    return pd.DataFrame(rows)


def mean_league_constants(finals_df: pd.DataFrame) -> dict:
    cols = ["off_3pa_rate", "def_3pa_allowed", "off_2pa_rate",
            "def_2pa_allowed", "off_tov_pct", "def_forced_tov"]
    return {c: float(finals_df[c].mean()) if c in finals_df.columns and len(finals_df) else 0.0
            for c in cols}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(year: int):
    played_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(f"Missing {played_path}")

    played = pd.read_csv(played_path)
    played["game_id"] = played["game_id"].astype(str)
    played["game_ts"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    played["game_date"] = played["game_ts"].dt.date.astype(str)
    played["is_playoff"] = (played["season_type"].astype(str).str.upper() == "PST").astype(int)
    played = played.sort_values(["game_ts", "game_id"], kind="stable")

    if "home_id" not in played.columns or "away_id" not in played.columns:
        raise ValueError("played manifest must include home_id and away_id.")

    map_df = load_franchise_map()
    game_ids = set(played["game_id"].unique())
    latest = pick_latest_game_summary_files(game_ids)
    if len(latest) == 0:
        raise FileNotFoundError(f"No bronze game_summary files found for year={year}")

    # Build realized rows with raw inputs + franchise IDs
    realized_rows = []
    missing_summaries = 0
    meta = played.set_index("game_id").to_dict(orient="index")

    for gid in played["game_id"].tolist():
        if gid not in latest:
            missing_summaries += 1
            continue
        d = json.loads(latest[gid].read_text(encoding="utf-8"))
        home_blk = d.get("home") or {}
        away_blk = d.get("away") or {}
        h = extract_team_inputs(home_blk)
        a = extract_team_inputs(away_blk)
        m = meta[gid]
        game_ts = m["game_ts"]
        game_date = m["game_date"]
        is_playoff = int(m["is_playoff"])
        home_id = str(m["home_id"])
        away_id = str(m["away_id"])
        home_fid = map_team_to_franchise(home_id, year, map_df)
        away_fid = map_team_to_franchise(away_id, year, map_df)

        for (tid, fid, opp_tid, opp_fid, is_home, inp, opp_inp) in [
            (home_id, home_fid, away_id, away_fid, 1, h, a),
            (away_id, away_fid, home_id, home_fid, 0, a, h),
        ]:
            realized_rows.append({
                "game_id": gid, "game_ts": game_ts, "game_date": game_date,
                "season": year,
                "team_id": tid, "franchise_id": fid,
                "opponent_team_id": opp_tid, "opponent_franchise_id": opp_fid,
                "is_home": is_home, "is_playoff": is_playoff,
                "FGA": inp["FGA"], "3PA": inp["3PA"], "FTA": inp["FTA"], "TO": inp["TO"],
                "OppFGA": opp_inp["FGA"], "Opp3PA": opp_inp["3PA"],
                "OppFTA": opp_inp["FTA"], "OppTO": opp_inp["TO"],
                "bronze_file": latest[gid].name,
            })

    realized = pd.DataFrame(realized_rows)
    realized["game_ts"] = pd.to_datetime(realized["game_ts"], utc=True, errors="coerce")
    realized = realized.sort_values(["game_ts", "game_id", "franchise_id"], kind="stable")

    silver_plus = Path("data/silver_plus")
    silver_plus.mkdir(parents=True, exist_ok=True)

    # Determine game-1 priors (keyed by franchise_id)
    if year == 2015:
        finals_2015 = compute_franchise_finals(realized)
        league_init = mean_league_constants(finals_2015)
        prev_franchise_final: dict = {}
        prev_league_avg = league_init
    else:
        prev_final_path = silver_plus / f"franchise_style_profile_final_{year-1}.csv"
        if not prev_final_path.exists():
            raise FileNotFoundError(f"Missing {prev_final_path}. Build {year-1} first.")
        prev_final = pd.read_csv(prev_final_path)
        prev_final["franchise_id"] = prev_final["franchise_id"].astype(str)
        prev_franchise_final = prev_final.set_index("franchise_id").to_dict(orient="index")
        prev_league_avg = mean_league_constants(prev_final)
        league_init = prev_league_avg

    # Build pregame style values; EWMA state keyed by franchise_id
    state: dict = {}
    out_rows = []

    for _, r in realized.iterrows():
        fid = str(r["franchise_id"])
        tid = str(r["team_id"])
        game_id = r["game_id"]
        game_ts = r["game_ts"]

        if fid not in state:
            state[fid] = {
                "games_played": 0,
                "last_completed_game_id": pd.NA,
                "last_completed_game_ts": pd.NaT,
                "totals": zero_totals(),
            }

        st = state[fid]
        gp = int(st["games_played"])

        if gp == 0:
            if year == 2015:
                prior = league_init
                prior_source = "league_init"
            else:
                if fid in prev_franchise_final:
                    prior = prev_franchise_final[fid]
                    prior_source = "prev_season_final"
                else:
                    prior = prev_league_avg
                    prior_source = "league_init"
            m_pre = {
                "off_3pa_rate_pre": float(prior.get("off_3pa_rate", 0.0)),
                "def_3pa_allowed_pre": float(prior.get("def_3pa_allowed", 0.0)),
                "off_2pa_rate_pre": float(prior.get("off_2pa_rate", 0.0)),
                "def_2pa_allowed_pre": float(prior.get("def_2pa_allowed", 0.0)),
                "off_tov_pct_pre": float(prior.get("off_tov_pct", 0.0)),
                "def_forced_tov_pre": float(prior.get("def_forced_tov", 0.0)),
            }
        else:
            prior_source = "season_to_date"
            m_pre = {k + "_pre": v for k, v in style_from_totals(st["totals"]).items()}

        out_rows.append({
            "game_id": game_id, "game_ts": game_ts, "game_date": r["game_date"],
            "season": int(r["season"]),
            "team_id": tid, "franchise_id": fid,
            "opponent_team_id": str(r["opponent_team_id"]),
            "opponent_franchise_id": str(r["opponent_franchise_id"]),
            "is_home": int(r["is_home"]), "is_playoff": int(r["is_playoff"]),
            **m_pre,
            "games_played_before_game": gp,
            "prior_source": prior_source,
            "last_completed_game_id": st["last_completed_game_id"],
            "last_completed_game_ts": st["last_completed_game_ts"],
        })

        # Update franchise running totals
        t = st["totals"]
        t["FGA"] += float(r["FGA"]); t["3PA"] += float(r["3PA"])
        t["FTA"] += float(r["FTA"]); t["TO"] += float(r["TO"])
        t["OppFGA"] += float(r["OppFGA"]); t["Opp3PA"] += float(r["Opp3PA"])
        t["OppFTA"] += float(r["OppFTA"]); t["OppTO"] += float(r["OppTO"])
        st["games_played"] = gp + 1
        st["last_completed_game_id"] = game_id
        st["last_completed_game_ts"] = game_ts

    out = pd.DataFrame(out_rows)
    out["game_ts"] = pd.to_datetime(out["game_ts"], utc=True, errors="coerce")
    out = out.sort_values(["game_ts", "game_id", "franchise_id"], kind="stable")

    out_path = silver_plus / f"game_franchise_style_profile_{year}_REGPST.csv"
    out.to_csv(out_path, index=False)

    finals_df = compute_franchise_finals(realized)
    finals_df.insert(0, "season", year)
    finals_path = silver_plus / f"franchise_style_profile_final_{year}.csv"
    finals_df.to_csv(finals_path, index=False)

    n_franchises = out["franchise_id"].nunique()
    print(f"{year}: games={out['game_id'].nunique()} team_games={len(out)} "
          f"franchises={n_franchises} missing_summaries={missing_summaries}")
    print(f"  wrote: {out_path}")
    print(f"  wrote: {finals_path}")

    expected = 13 if year == 2025 else 12
    if n_franchises != expected:
        print(f"  WARNING: expected {expected} franchises, got {n_franchises}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)
