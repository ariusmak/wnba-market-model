"""
27_build_franchise_elo.py

Franchise-aware Elo builder. Outputs NEW files (does not overwrite old ones).

Output per year:
  data/silver_plus/elo_franchise_team_game_{year}_REGPST.csv

Locked hyperparams: H=25, K=20, a=0.45, b=1.0
Carryover is keyed by franchise_id, so Stars->Aces continuity is automatic.

Run sequentially 2015->2025.
"""
import argparse
from pathlib import Path
import sys

import pandas as pd

_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from srwnba.util.elo import EloParams, apply_carryover, elo_prob, update_one_game
from srwnba.util.franchise import load_franchise_map, map_team_to_franchise


def load_outcomes(year: int) -> pd.DataFrame:
    p = Path(f"data/silver/game_outcomes_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}")
    df = pd.read_csv(p)
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df[df["home_win"].notna()].copy()
    df["home_win"] = df["home_win"].astype(int)
    df["mov"] = pd.to_numeric(df["mov"], errors="coerce")

    # Filter to franchise games only (same set as played_franchise_games manifest)
    pf_path = Path(f"data/silver/played_franchise_games_{year}_REGPST.csv")
    if pf_path.exists():
        pf = pd.read_csv(pf_path, usecols=["game_id"])
        franchise_gids = set(pf["game_id"].astype(str))
        before = len(df)
        df = df[df["game_id"].astype(str).isin(franchise_gids)].copy()
        dropped = before - len(df)
        if dropped:
            print(f"  [{year}] filtered {dropped} non-franchise game(s) from outcomes")

    return df.sort_values(["scheduled", "game_id"], kind="stable")


def main(start_year: int, end_year: int):
    params = EloParams(H=25, K=20, a=0.45, b=1.0)
    map_df = load_franchise_map()

    # ratings dict keyed by franchise_id
    ratings: dict = {}

    def get_r(fid: str) -> float:
        if fid not in ratings:
            ratings[fid] = params.mu
        return ratings[fid]

    out_dir = Path("data/silver_plus")
    out_dir.mkdir(parents=True, exist_ok=True)

    for y in range(start_year, end_year + 1):
        if y != start_year and len(ratings) > 0:
            ratings = apply_carryover(ratings, params)

        df = load_outcomes(y)
        rows = []

        for _, g in df.iterrows():
            gid = str(g["game_id"])
            sched = g["scheduled"]
            home_tid = str(g["home_id"])
            away_tid = str(g["away_id"])

            home_fid = map_team_to_franchise(home_tid, y, map_df)
            away_fid = map_team_to_franchise(away_tid, y, map_df)

            rH_pre = get_r(home_fid)
            rA_pre = get_r(away_fid)

            p_home = elo_prob(rH_pre, rA_pre, H=params.H, scale=params.scale)
            p_away = 1.0 - p_home

            home_win = int(g["home_win"])
            mov = int(g["mov"]) if pd.notna(g["mov"]) else None

            _, delta, rH_post, rA_post = update_one_game(
                rH_pre, rA_pre, home_win=home_win, mov=mov, params=params
            )
            ratings[home_fid] = rH_post
            ratings[away_fid] = rA_post

            rows.append({
                "season_year": y, "scheduled": sched, "game_id": gid,
                "team_id": home_tid, "franchise_id": home_fid,
                "opponent_team_id": away_tid, "opponent_franchise_id": away_fid,
                "is_home": 1,
                "elo_pre": rH_pre, "elo_post": rH_post, "p_win_pre": p_home,
            })
            rows.append({
                "season_year": y, "scheduled": sched, "game_id": gid,
                "team_id": away_tid, "franchise_id": away_fid,
                "opponent_team_id": home_tid, "opponent_franchise_id": home_fid,
                "is_home": 0,
                "elo_pre": rA_pre, "elo_post": rA_post, "p_win_pre": p_away,
            })

        season_df = pd.DataFrame(rows)
        season_df = season_df.sort_values(
            ["season_year", "scheduled", "game_id", "is_home"], kind="stable"
        )
        out_path = out_dir / f"elo_franchise_team_game_{y}_REGPST.csv"
        season_df.to_csv(out_path, index=False)

        n_fids = season_df["franchise_id"].nunique()
        print(f"{y}: games={season_df['game_id'].nunique()} "
              f"team_games={len(season_df)} franchises={n_fids}")
        print(f"  wrote: {out_path}")

        expected = 13 if y == 2025 else 12
        if n_fids != expected:
            print(f"  WARNING: expected {expected} franchises, got {n_fids}")

    # Write combined
    combined_rows = []
    for y in range(start_year, end_year + 1):
        p = out_dir / f"elo_franchise_team_game_{y}_REGPST.csv"
        combined_rows.append(pd.read_csv(p))
    combined = pd.concat(combined_rows, ignore_index=True)
    combined_path = out_dir / f"elo_franchise_team_game_{start_year}_{end_year}_REGPST.csv"
    combined.to_csv(combined_path, index=False)
    print(f"wrote combined: {combined_path} rows={len(combined)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()
    main(args.start_year, args.end_year)
