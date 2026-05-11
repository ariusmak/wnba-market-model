"""
11_extend_elo_to_year.py
========================

Extends the existing Elo state file (`elo_franchise_team_game_*.csv`)
into a new year without recomputing prior years. This avoids re-running
the multi-year Elo scripts (19, 27) which require silver/game_outcomes_*
for every year 2015..Y — files we don't have on disk.

What it does
------------
1. Reads `data/silver_plus/elo_franchise_team_game_{prev_year}_REGPST.csv`
2. Extracts each team's end-of-season Elo (last `elo_post` per team_id)
3. Applies the season carryover R_start = a*R_end + (1-a)*mu  (a=0.45)
4. Reads `data/silver/game_outcomes_{year}_REGPST.csv` (which the parse
   phase already built from real bronze game summaries — has real MOV)
5. Runs Elo updates for the new year
6. Writes `data/silver_plus/elo_franchise_team_game_{year}_REGPST.csv`
   with the same schema gold (script 30) reads.

Locked Elo hyperparameters from CLAUDE.md §4:
    H = 25   K = 20   a = 0.45   b = 1.0   mu = 1505

Usage:
    python pipelines/07_live/11_extend_elo_to_year.py --year 2026
    python pipelines/07_live/11_extend_elo_to_year.py --year 2026 --prev-year 2025 --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.util.elo import (  # noqa: E402
    EloParams,
    apply_carryover,
    elo_prob,
    update_one_game,
)


# Locked Elo hyperparameters from CLAUDE.md §4. Do NOT change without
# also revising the existing 2015-2025 Elo state on disk.
ELO_PARAMS = EloParams(H=25.0, K=20.0, a=0.45, b=1.0)


def _end_of_season_ratings(prev_year: int) -> Dict[str, float]:
    """For each team, take their LAST game's elo_post in the prior year."""
    p = REPO_ROOT / "data" / "silver_plus" / f"elo_franchise_team_game_{prev_year}_REGPST.csv"
    if not p.exists():
        raise SystemExit(f"Missing {p}")
    df = pd.read_csv(p)
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df["team_id"] = df["team_id"].astype(str)
    df = df.sort_values(["team_id", "scheduled", "game_id"], kind="stable")
    last = df.groupby("team_id", as_index=False).tail(1)
    return {row["team_id"]: float(row["elo_post"]) for _, row in last.iterrows()}


def _team_to_franchise_map(prev_year: int) -> Dict[str, str]:
    """Use the prior-year file as ground truth for team_id → franchise_id."""
    p = REPO_ROOT / "data" / "silver_plus" / f"elo_franchise_team_game_{prev_year}_REGPST.csv"
    df = pd.read_csv(p, usecols=["team_id", "franchise_id"])
    df = df.drop_duplicates(subset="team_id")
    return {str(r["team_id"]): str(r["franchise_id"]) for _, r in df.iterrows()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True,
                    help="New year to extend Elo into (e.g. 2026)")
    ap.add_argument("--prev-year", type=int, default=None,
                    help="Previous year's saved Elo file (default: --year - 1)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output file")
    args = ap.parse_args()

    prev_year = args.prev_year if args.prev_year is not None else args.year - 1

    out_path = REPO_ROOT / "data" / "silver_plus" / f"elo_franchise_team_game_{args.year}_REGPST.csv"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists — pass --force to overwrite.")

    # 1. Bootstrap from prior year end-of-season state + carryover
    print(f"[elo-extend] reading prior year {prev_year} end-of-season state")
    end_state = _end_of_season_ratings(prev_year)
    print(f"[elo-extend] {len(end_state)} teams in prior-year state")
    seed = apply_carryover(end_state, ELO_PARAMS)
    franchise_of = _team_to_franchise_map(prev_year)

    # 2. Read silver outcomes for the new year (built from real bronze)
    outcomes_path = REPO_ROOT / "data" / "silver" / f"game_outcomes_{args.year}_REGPST.csv"
    if not outcomes_path.exists():
        raise SystemExit(f"Missing {outcomes_path} — run parse phase first.")
    outcomes = pd.read_csv(outcomes_path)
    outcomes["scheduled"] = pd.to_datetime(outcomes["scheduled"], utc=True, errors="coerce")
    outcomes["home_id"] = outcomes["home_id"].astype(str)
    outcomes["away_id"] = outcomes["away_id"].astype(str)
    outcomes["game_id"] = outcomes["game_id"].astype(str)
    outcomes = outcomes.sort_values(["scheduled", "game_id"], kind="stable")
    print(f"[elo-extend] {args.year}: {len(outcomes)} closed games to process")

    # 3. Run Elo updates one game at a time, emitting two rows per game
    ratings: Dict[str, float] = dict(seed)

    def get_r(team_id: str) -> float:
        if team_id not in ratings:
            ratings[team_id] = ELO_PARAMS.mu  # expansion / unseen team
        return ratings[team_id]

    rows = []
    for _, g in outcomes.iterrows():
        home = str(g["home_id"])
        away = str(g["away_id"])
        rH_pre = get_r(home)
        rA_pre = get_r(away)
        p_home = elo_prob(rH_pre, rA_pre, H=ELO_PARAMS.H, scale=ELO_PARAMS.scale)

        home_win = g.get("home_win")
        mov = g.get("mov")
        if pd.isna(home_win) or pd.isna(mov):
            # Game not yet settled — emit pregame state, no update.
            rH_post, rA_post = rH_pre, rA_pre
        else:
            _, _, rH_post, rA_post = update_one_game(
                rH_pre, rA_pre, int(home_win), int(mov), ELO_PARAMS,
            )
            ratings[home] = rH_post
            ratings[away] = rA_post

        for tid, opp, is_home, r_pre, r_post in [
            (home, away, 1, rH_pre, rH_post),
            (away, home, 0, rA_pre, rA_post),
        ]:
            rows.append({
                "season_year": args.year,
                "scheduled": g["scheduled"],
                "game_id": g["game_id"],
                "team_id": tid,
                "franchise_id": franchise_of.get(tid, tid),
                "opponent_team_id": opp,
                "opponent_franchise_id": franchise_of.get(opp, opp),
                "is_home": is_home,
                "elo_pre": r_pre,
                "elo_post": r_post,
                "p_win_pre": p_home if is_home else (1.0 - p_home),
            })

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[elo-extend] wrote {out_path.relative_to(REPO_ROOT)}  rows={len(out)}")
    print(f"[elo-extend] post-update top 5 ratings:")
    final = pd.Series(ratings).sort_values(ascending=False).head(5)
    for tid, r in final.items():
        print(f"    {tid[:12]}...  {r:7.2f}")


if __name__ == "__main__":
    main()
