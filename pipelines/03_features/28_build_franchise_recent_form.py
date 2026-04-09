"""
28_build_franchise_recent_form.py

Franchise-aware recent form (EWMA) builder. Outputs NEW files (does not overwrite old ones).

Output per year:
  data/silver_plus/game_franchise_recent_form_{year}_REGPST.csv

EWMA state keyed by franchise_id. Half-life: 7 games.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from srwnba.util.franchise import load_franchise_map, map_team_to_franchise


LAMBDA = 1 - 2 ** (-1 / 7)


def safe_div(n, d) -> float:
    try:
        if d is None or d == 0 or (isinstance(d, float) and pd.isna(d)) or d <= 0:
            return 0.0
        return float(n) / float(d)
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


def extract_team_totals(team_block: dict) -> dict:
    stats = (team_block or {}).get("statistics") or {}

    def to_int(x):
        try:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return 0
            return int(x)
        except Exception:
            return 0

    if stats.get("total_turnovers") is not None:
        tov = to_int(stats.get("total_turnovers"))
    else:
        pt = stats.get("player_turnovers")
        tt = stats.get("team_turnovers")
        if pt is not None and tt is not None:
            tov = to_int(pt) + to_int(tt)
        else:
            tov = to_int(stats.get("turnovers"))

    return {
        "pts": to_int(stats.get("points") or (team_block or {}).get("points")),
        "fga": to_int(stats.get("field_goals_att")),
        "fgm": to_int(stats.get("field_goals_made")),
        "tpm": to_int(stats.get("three_points_made")),
        "fta": to_int(stats.get("free_throws_att")),
        "tov": tov,
        "orb": to_int(stats.get("offensive_rebounds")),
        "drb": to_int(stats.get("defensive_rebounds")),
    }


def compute_metrics(team: dict, opp: dict) -> dict:
    poss = team["fga"] - team["orb"] + team["tov"] + 0.44 * team["fta"]
    poss_opp = opp["fga"] - opp["orb"] + opp["tov"] + 0.44 * opp["fta"]
    ortg = 100.0 * safe_div(team["pts"], poss)
    drtg = 100.0 * safe_div(opp["pts"], poss_opp)
    return {
        "poss": float(poss),
        "poss_opp": float(poss_opp),
        "ortg": ortg,
        "drtg": drtg,
        "net_rtg": ortg - drtg,
        "efg": safe_div(team["fgm"] + 0.5 * team["tpm"], team["fga"]),
        "tov_pct": safe_div(team["tov"], poss),
        "orb_pct": safe_div(team["orb"], team["orb"] + opp["drb"]),
        "ftr": safe_div(team["fta"], team["fga"]),
    }


def main(year: int):
    played_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    if not played_path.exists():
        raise FileNotFoundError(f"Missing {played_path}.")

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

    realized_rows = []
    missing_summaries = 0
    meta = played.set_index("game_id").to_dict(orient="index")

    for gid in played["game_id"].tolist():
        if gid not in latest:
            missing_summaries += 1
            continue

        d = json.loads(latest[gid].read_text(encoding="utf-8"))
        home = d.get("home") or {}
        away = d.get("away") or {}

        h_tot = extract_team_totals(home)
        a_tot = extract_team_totals(away)
        h_m = compute_metrics(h_tot, a_tot)
        a_m = compute_metrics(a_tot, h_tot)

        m = meta.get(gid, {})
        game_ts = m.get("game_ts")
        game_date = m.get("game_date", "")
        is_playoff = int(m.get("is_playoff", 0))
        home_id = str(m.get("home_id"))
        away_id = str(m.get("away_id"))
        home_fid = map_team_to_franchise(home_id, year, map_df)
        away_fid = map_team_to_franchise(away_id, year, map_df)

        for (tid, fid, opp_tid, opp_fid, is_home, metrics) in [
            (home_id, home_fid, away_id, away_fid, 1, h_m),
            (away_id, away_fid, home_id, home_fid, 0, a_m),
        ]:
            realized_rows.append({
                "season": year, "game_id": gid, "game_ts": game_ts, "game_date": game_date,
                "team_id": tid, "franchise_id": fid,
                "opponent_team_id": opp_tid, "opponent_franchise_id": opp_fid,
                "is_home": is_home, "is_playoff": is_playoff,
                **{f"{k}_game": v for k, v in metrics.items()},
                "bronze_file": latest[gid].name,
            })

    realized = pd.DataFrame(realized_rows)
    realized = realized.sort_values(["game_ts", "game_id", "is_home"], kind="stable")

    # EWMA state keyed by franchise_id
    state: dict = {}
    out_rows = []

    for _, row in realized.iterrows():
        fid = str(row["franchise_id"])
        tid = str(row["team_id"])
        game_id = row["game_id"]
        game_ts = row["game_ts"]

        if fid not in state:
            state[fid] = {
                "net_rtg_ewma": 0.0, "efg_ewma": 0.0,
                "tov_pct_ewma": 0.0, "orb_pct_ewma": 0.0, "ftr_ewma": 0.0,
                "last_completed_game_id": pd.NA,
                "last_completed_game_ts": pd.NaT,
                "net_rtg_last_game": 0.0, "efg_last_game": 0.0,
                "tov_pct_last_game": 0.0, "orb_pct_last_game": 0.0, "ftr_last_game": 0.0,
            }

        st = state[fid]

        out_rows.append({
            "season": int(row["season"]),
            "game_id": game_id, "game_ts": game_ts, "game_date": row["game_date"],
            "team_id": tid, "franchise_id": fid,
            "opponent_team_id": str(row["opponent_team_id"]),
            "opponent_franchise_id": str(row["opponent_franchise_id"]),
            "is_home": int(row["is_home"]), "is_playoff": int(row["is_playoff"]),
            "net_rtg_ewma_pre": st["net_rtg_ewma"],
            "efg_ewma_pre": st["efg_ewma"],
            "tov_pct_ewma_pre": st["tov_pct_ewma"],
            "orb_pct_ewma_pre": st["orb_pct_ewma"],
            "ftr_ewma_pre": st["ftr_ewma"],
            "last_completed_game_id": st["last_completed_game_id"],
            "last_completed_game_ts": st["last_completed_game_ts"],
            "net_rtg_last_game": st["net_rtg_last_game"],
            "efg_last_game": st["efg_last_game"],
            "tov_pct_last_game": st["tov_pct_last_game"],
            "orb_pct_last_game": st["orb_pct_last_game"],
            "ftr_last_game": st["ftr_last_game"],
            "net_rtg_game": row["net_rtg_game"],
            "efg_game": row["efg_game"],
            "tov_pct_game": row["tov_pct_game"],
            "orb_pct_game": row["orb_pct_game"],
            "ftr_game": row["ftr_game"],
            "poss_game": row["poss_game"],
            "poss_opp_game": row["poss_opp_game"],
            "ortg_game": row["ortg_game"],
            "drtg_game": row["drtg_game"],
            "bronze_file": row["bronze_file"],
        })

        st["net_rtg_ewma"] = LAMBDA * float(row["net_rtg_game"]) + (1 - LAMBDA) * st["net_rtg_ewma"]
        st["efg_ewma"] = LAMBDA * float(row["efg_game"]) + (1 - LAMBDA) * st["efg_ewma"]
        st["tov_pct_ewma"] = LAMBDA * float(row["tov_pct_game"]) + (1 - LAMBDA) * st["tov_pct_ewma"]
        st["orb_pct_ewma"] = LAMBDA * float(row["orb_pct_game"]) + (1 - LAMBDA) * st["orb_pct_ewma"]
        st["ftr_ewma"] = LAMBDA * float(row["ftr_game"]) + (1 - LAMBDA) * st["ftr_ewma"]
        st["last_completed_game_id"] = game_id
        st["last_completed_game_ts"] = game_ts
        st["net_rtg_last_game"] = float(row["net_rtg_game"])
        st["efg_last_game"] = float(row["efg_game"])
        st["tov_pct_last_game"] = float(row["tov_pct_game"])
        st["orb_pct_last_game"] = float(row["orb_pct_game"])
        st["ftr_last_game"] = float(row["ftr_game"])

    out = pd.DataFrame(out_rows)
    out = out.sort_values(["game_ts", "game_id", "team_id", "is_home"], kind="stable")

    out_dir = Path("data/silver_plus")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"game_franchise_recent_form_{year}_REGPST.csv"
    out.to_csv(out_path, index=False)

    print(f"{year}: games={out['game_id'].nunique()} team_games={len(out)} "
          f"franchises={out['franchise_id'].nunique()} missing_summaries={missing_summaries}")
    print(f"  lambda={LAMBDA:.4f}")
    print(f"  wrote: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    main(args.year)
