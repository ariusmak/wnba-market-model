# notebooks/gridsearch_elo.py
#
# Gridsearch Elo hyperparameters over train years and report test performance.
# Designed to be imported by walk-forward scripts, and also runnable as a CLI.
#
# Usage (PowerShell example):
#   $env:PYTHONPATH="src;notebooks"
#   python notebooks/gridsearch_elo.py `
#     --train-years 2015,2016,2017,2018,2019 `
#     --test-years 2020 `
#     --H 35,40,43,45,50,55 `
#     --K 20 `
#     --a 0.7 `
#     --b 0.8 `
#     --out data/silver/grid_2015_2019_test2020.csv `
#     --verbose
#
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from srwnba.util.elo import EloParams, apply_carryover, update_one_game


# -------------------------
# Metrics
# -------------------------
def log_loss(y_true: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())


def brier(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y_true) ** 2).mean())


# -------------------------
# Parsing helpers (accept single values too)
# -------------------------
def parse_int_list(s: str) -> list[int]:
    # accepts "2015,2016,2017" or "2015"
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> list[float]:
    # accepts "40,60,80" or "43"
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# -------------------------
# Data loading
# -------------------------
def load_year(year: int) -> pd.DataFrame:
    p = Path(f"data/silver/game_outcomes_{year}_REGPST.csv")
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run: python notebooks/17_build_game_outcomes_year.py --year {year}")

    df = pd.read_csv(p)
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df[df["home_win"].notna()].copy()
    df["home_win"] = df["home_win"].astype(int)
    df = df.sort_values(["scheduled", "game_id"], kind="stable")
    return df


# -------------------------
# Elo runner (multi-year with carryover)
# -------------------------
def run_multiyear_elo_predict(
    years: list[int],
    params: EloParams,
    start_ratings: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Runs Elo over multiple years sequentially with carryover between seasons.

    Returns:
      preds dataframe with per-game p_home and outcome (home_win)
      ending ratings dict after last year
    """
    ratings: dict[str, float] = dict(start_ratings) if start_ratings else {}

    def get_r(tid: str) -> float:
        if tid not in ratings:
            ratings[tid] = params.mu
        return ratings[tid]

    all_rows: list[dict] = []

    for yi, y in enumerate(years):
        # Apply carryover at the start of each new season (except first) if we have ratings
        if yi > 0 and len(ratings) > 0:
            ratings = apply_carryover(ratings, params)

        dfy = load_year(y)

        for _, g in dfy.iterrows():
            home = str(g["home_id"])
            away = str(g["away_id"])

            rH = get_r(home)
            rA = get_r(away)

            home_win = int(g["home_win"])
            mov = int(g["mov"]) if not pd.isna(g["mov"]) else None

            pH, delta, rH_post, rA_post = update_one_game(rH, rA, home_win=home_win, mov=mov, params=params)

            ratings[home] = rH_post
            ratings[away] = rA_post

            all_rows.append(
                {
                    "season_year": y,
                    "game_id": g["game_id"],
                    "scheduled": g["scheduled"],
                    "home_id": home,
                    "away_id": away,
                    "home_win": home_win,
                    "p_home": pH,
                    "mov": mov,
                }
            )

    preds = pd.DataFrame(all_rows)
    return preds, ratings


def evaluate_years(preds: pd.DataFrame, years: list[int]) -> dict:
    df = preds[preds["season_year"].isin(years)].copy()
    y = df["home_win"].to_numpy(dtype=float)
    p = df["p_home"].to_numpy(dtype=float)

    return {
        "games": int(len(df)),
        "logloss": log_loss(y, p) if len(df) else float("nan"),
        "brier": brier(y, p) if len(df) else float("nan"),
        "home_win_rate": float(y.mean()) if len(df) else float("nan"),
        "avg_p_home": float(p.mean()) if len(df) else float("nan"),
    }


# -------------------------
# Gridsearch core (importable)
# -------------------------
def run_gridsearch(
    train_years: list[int],
    test_years: list[int],
    H_list: list[float],
    K_list: list[float],
    a_list: list[float],
    b_list: list[float],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Returns a DataFrame sorted by test_logloss then test_brier (ascending).

    Notes:
      - Any hyperparameter list may be length 1.
      - Elo is run once per parameter combo over union(train_years, test_years),
        then metrics are sliced for train and test.
    """
    combos = list(itertools.product(H_list, K_list, a_list, b_list))
    all_years = sorted(set(train_years + test_years))

    results = []

    for (H, K, a, b) in combos:
        params = EloParams(H=H, K=K, a=a, b=b)
        preds, _ = run_multiyear_elo_predict(all_years, params=params)

        train_m = evaluate_years(preds, train_years)
        test_m = evaluate_years(preds, test_years)

        if verbose:
            print(
                f"H={H:g} K={K:g} a={a:g} b={b:g} | "
                f"train_logloss={train_m['logloss']:.6f} train_brier={train_m['brier']:.6f} | "
                f"test_logloss={test_m['logloss']:.6f} test_brier={test_m['brier']:.6f}"
            )

        results.append(
            {
                "H": H,
                "K": K,
                "a": a,
                "b": b,
                "train_years": ",".join(map(str, train_years)),
                "test_years": ",".join(map(str, test_years)),
                "train_games": train_m["games"],
                "train_logloss": train_m["logloss"],
                "train_brier": train_m["brier"],
                "test_games": test_m["games"],
                "test_logloss": test_m["logloss"],
                "test_brier": test_m["brier"],
            }
        )

    res = pd.DataFrame(results)
    res = res.sort_values(["test_logloss", "test_brier"], kind="stable").reset_index(drop=True)
    return res


# -------------------------
# CLI wrapper
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", required=True, help="Comma-separated list, e.g. 2015,2016,2017")
    ap.add_argument("--test-years", required=True, help="Comma-separated list, e.g. 2020 or 2020,2021")
    ap.add_argument("--H", required=True, help="Comma-separated list of H values (home advantage)")
    ap.add_argument("--K", required=True, help="Comma-separated list of K values (learning rate)")
    ap.add_argument("--a", required=True, help="Comma-separated list of a values (season carryover)")
    ap.add_argument("--b", required=True, help="Comma-separated list of b values (MOV exponent)")
    ap.add_argument("--out", default="data/silver/elo_gridsearch_results.csv")
    ap.add_argument("--verbose", action="store_true", help="Print losses for every combo")
    args = ap.parse_args()

    train_years = parse_int_list(args.train_years)
    test_years = parse_int_list(args.test_years)
    H_list = parse_float_list(args.H)
    K_list = parse_float_list(args.K)
    a_list = parse_float_list(args.a)
    b_list = parse_float_list(args.b)

    print(f"Grid size: {len(list(itertools.product(H_list, K_list, a_list, b_list)))} combos")
    print(f"Train years: {train_years}")
    print(f"Test years: {test_years}")

    res = run_gridsearch(
        train_years=train_years,
        test_years=test_years,
        H_list=H_list,
        K_list=K_list,
        a_list=a_list,
        b_list=b_list,
        verbose=args.verbose,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    best = res.iloc[0].to_dict()
    print("\nBEST (by train_logloss, then test_logloss):")
    print(best)
    print("\nWrote:", str(out_path.resolve()))


if __name__ == "__main__":
    main()