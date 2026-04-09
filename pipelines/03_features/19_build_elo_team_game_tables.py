import argparse
import sys
from pathlib import Path

# Allow importing srwnba when run from project root (e.g. python notebooks/19_...py)
_root = Path(__file__).resolve().parents[2]
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pandas as pd

from srwnba.util.elo import EloParams, apply_carryover, elo_prob, update_one_game


def load_outcomes(year: int) -> pd.DataFrame:
    p = Path(f"data/silver/game_outcomes_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run: python notebooks/17_build_game_outcomes_year.py --year {year}")
    df = pd.read_csv(p)
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df[df["home_win"].notna()].copy()
    df["home_win"] = df["home_win"].astype(int)
    df["mov"] = pd.to_numeric(df["mov"], errors="coerce")
    return df.sort_values(["scheduled", "game_id"], kind="stable")


def main(start_year: int, end_year: int):
    # Fixed hyperparams you selected
    params = EloParams(H=25, K=20, a=0.45, b=1.0)

    ratings: dict[str, float] = {}

    def get_r(team_id: str) -> float:
        if team_id not in ratings:
            ratings[team_id] = params.mu
        return ratings[team_id]

    all_rows = []

    for y in range(start_year, end_year + 1):
        # carryover into new season (skip first year)
        if y != start_year and len(ratings) > 0:
            ratings = apply_carryover(ratings, params)

        df = load_outcomes(y)

        for _, g in df.iterrows():
            gid = str(g["game_id"])
            sched = g["scheduled"]
            home = str(g["home_id"])
            away = str(g["away_id"])

            rH_pre = get_r(home)
            rA_pre = get_r(away)

            p_home = elo_prob(rH_pre, rA_pre, H=params.H, scale=params.scale)
            p_away = 1.0 - p_home

            home_win = int(g["home_win"])
            mov = int(g["mov"]) if pd.notna(g["mov"]) else None

            # Update ratings
            _, delta, rH_post, rA_post = update_one_game(
                rH_pre, rA_pre, home_win=home_win, mov=mov, params=params
            )
            ratings[home] = rH_post
            ratings[away] = rA_post

            # Two team-game rows
            all_rows.append({
                "season_year": y,
                "scheduled": sched,
                "game_id": gid,
                "team_id": home,
                "opponent_id": away,
                "is_home": 1,
                "elo_pre": rH_pre,
                "elo_post": rH_post,
                "p_win_pre": p_home,
                "points_for": int(g["home_points"]),
                "points_against": int(g["away_points"]),
                "win": home_win,
            })
            all_rows.append({
                "season_year": y,
                "scheduled": sched,
                "game_id": gid,
                "team_id": away,
                "opponent_id": home,
                "is_home": 0,
                "elo_pre": rA_pre,
                "elo_post": rA_post,
                "p_win_pre": p_away,
                "points_for": int(g["away_points"]),
                "points_against": int(g["home_points"]),
                "win": 1 - home_win,
            })

    out_df = pd.DataFrame(all_rows)
    out_df = out_df.sort_values(["season_year", "scheduled", "game_id", "is_home"], kind="stable")

    out_dir = Path("data/silver")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write per-season
    for y in range(start_year, end_year + 1):
        season_df = out_df[out_df["season_year"] == y].copy()
        out_path = out_dir / f"elo_team_game_{y}_REGPST.csv"
        season_df.to_csv(out_path, index=False)
        print("wrote", out_path, "rows", len(season_df))

    # Write combined
    combined_path = out_dir / f"elo_team_game_{start_year}_{end_year}_REGPST.csv"
    out_df.to_csv(combined_path, index=False)
    print("wrote", combined_path, "rows", len(out_df))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2025)
    args = ap.parse_args()
    main(args.start_year, args.end_year)