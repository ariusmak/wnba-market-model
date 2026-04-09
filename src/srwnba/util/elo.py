from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import pandas as pd


@dataclass(frozen=True)
class EloParams:
    # Tunable hyperparameters (your 4 knobs)
    H: float = 100.0     # home advantage (40–120)
    K: float = 20.0      # learning rate (10–30)
    a: float = 0.75      # season carryover (0.50–0.85)
    b: float = 0.80      # MOV exponent strength (0.6–1.0); b=0 disables MOV if you want

    # Fixed conventions (keep stable unless we explicitly change later)
    mu: float = 1505.0
    scale: float = 400.0
    mov_add: float = 3.0
    mov_den_base: float = 7.5
    mov_den_coef: float = 0.006
    use_mov: bool = True


def elo_prob(r_home: float, r_away: float, H: float, scale: float = 400.0) -> float:
    """p(home win) = 1 / (1 + 10^(-((R_home + H) - R_away)/scale))"""
    return 1.0 / (1.0 + 10.0 ** (-(((r_home + H) - r_away) / scale)))


def mov_multiplier(mov: int, d_win: float, params: EloParams) -> float:
    # b=0 => no MOV
    if params.b <= 0:
        return 1.0
    return ((mov + params.mov_add) ** params.b) / (params.mov_den_base + params.mov_den_coef * d_win)


def apply_carryover(prev_ratings: Dict[str, float], params: EloParams) -> Dict[str, float]:
    """
    Season carryover:
      R_start = a * R_end + (1-a) * mu
    """
    a = params.a
    mu = params.mu
    return {tid: a * r + (1.0 - a) * mu for tid, r in prev_ratings.items()}


def update_one_game(r_home, r_away, home_win, mov, params):
    p_home = elo_prob(r_home, r_away, H=params.H, scale=params.scale)
    s = float(home_win)

    d_win = abs((r_home + params.H) - r_away)

    mult = 1.0
    if mov is not None:
        mult = mov_multiplier(int(mov), d_win, params)

    delta = params.K * mult * (s - p_home)
    return p_home, delta, r_home + delta, r_away - delta


def run_elo_on_games(
    games: pd.DataFrame,
    params: EloParams,
    initial_ratings: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run Elo over a single season's games.

    Required columns in `games`:
      - game_id, scheduled, home_id, away_id, home_points, away_points
    Optional:
      - season_type
    """
    df = games.copy()
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df.sort_values(["scheduled", "game_id"], kind="stable")

    ratings: Dict[str, float] = dict(initial_ratings) if initial_ratings else {}

    def r(team_id: str) -> float:
        if team_id not in ratings:
            ratings[team_id] = params.mu
        return ratings[team_id]

    out = []

    for _, g in df.iterrows():
        gid = str(g["game_id"])
        home = str(g["home_id"])
        away = str(g["away_id"])
        rH_pre = r(home)
        rA_pre = r(away)

        hp = g.get("home_points")
        ap = g.get("away_points")

        pH = elo_prob(rH_pre, rA_pre, H=params.H, scale=params.scale)

        if pd.isna(hp) or pd.isna(ap):
            out.append({
                "game_id": gid,
                "scheduled": g.get("scheduled"),
                "season_type": g.get("season_type", None),
                "home_id": home,
                "away_id": away,
                "home_points": hp,
                "away_points": ap,
                "home_win": None,
                "mov": None,
                "r_home_pre": rH_pre,
                "r_away_pre": rA_pre,
                "p_home": pH,
                "delta": 0.0,
                "r_home_post": rH_pre,
                "r_away_post": rA_pre,
            })
            continue

        hp_i = int(hp)
        ap_i = int(ap)
        home_win = 1 if hp_i > ap_i else 0
        mov = abs(hp_i - ap_i)

        pH, delta, rH_post, rA_post = update_one_game(rH_pre, rA_pre, home_win, mov, params)

        ratings[home] = rH_post
        ratings[away] = rA_post

        out.append({
            "game_id": gid,
            "scheduled": g.get("scheduled"),
            "season_type": g.get("season_type", None),
            "home_id": home,
            "away_id": away,
            "home_points": hp_i,
            "away_points": ap_i,
            "home_win": home_win,
            "mov": mov,
            "r_home_pre": rH_pre,
            "r_away_pre": rA_pre,
            "p_home": pH,
            "delta": delta,
            "r_home_post": rH_post,
            "r_away_post": rA_post,
        })

    game_log = pd.DataFrame(out)

    final_ratings = (
        pd.DataFrame([{"team_id": tid, "elo": val} for tid, val in ratings.items()])
        .sort_values("elo", ascending=False)
        .reset_index(drop=True)
    )

    return game_log, final_ratings