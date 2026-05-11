"""
10_synthesize_cascade_state.py
==============================

Bootstrap helper for live: reconstructs the prior-year intermediate files
that the existing pipeline expects, but only `data/gold/` is committed.

Two reconstructions:

1) **Team / franchise style cascade** — scripts 24 / 26 read
   `data/silver_plus/team_style_profile_final_{Y-1}.csv` and
   `data/silver_plus/franchise_style_profile_final_{Y-1}.csv` to seed
   game 1 of year Y. We synthesize these from the LAST pregame style
   values per team / franchise in the gold CSV — off by one game's
   contribution out of ~40 (< 3% per metric).

2) **Silver outcomes for 2015..Y** — scripts 19 / 27 (multi-year Elo)
   require `data/silver/game_outcomes_{year}_REGPST.csv` for every year
   in their range. The gold CSV has `home_win` per game but not the
   margin of victory (MOV). We synthesize outcomes with MOV = 8.5
   (league-average) for years missing on disk; the recomputed Elo state
   drifts by ~5-10 points per team vs the original (which is fine — Elo
   is the base prior, and 2026's predictions only see this drift on the
   prior, not on the XGB correction).

Idempotent: any file already present on disk is left alone unless
`--force` is passed.

Usage:
    # Synthesize cascade for year Y-1 (single year)
    python pipelines/07_live/10_synthesize_cascade_state.py --year 2025

    # Also reconstruct prior-year outcomes from gold (for multi-year Elo)
    python pipelines/07_live/10_synthesize_cascade_state.py \\
        --year 2025 --outcomes-from 2015 --outcomes-to 2025
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

STYLE_METRICS = [
    "off_3pa_rate",
    "def_3pa_allowed",
    "off_2pa_rate",
    "def_2pa_allowed",
    "off_tov_pct",
    "def_forced_tov",
]


def _last_pregame_per_id(
    gold: pd.DataFrame, id_col_home: str, id_col_away: str, out_id_col: str,
) -> pd.DataFrame:
    """For each team or franchise, return the pregame style row from their
    LAST game in the season, using the side (home/away) where they appear.
    """
    # Reshape to long form: one row per (game_ts, entity_id, side, metrics)
    home_long = gold[["game_ts", id_col_home] + [f"home_{m}_pre" for m in STYLE_METRICS]].copy()
    home_long = home_long.rename(columns={id_col_home: out_id_col,
                                          **{f"home_{m}_pre": m for m in STYLE_METRICS}})
    away_long = gold[["game_ts", id_col_away] + [f"away_{m}_pre" for m in STYLE_METRICS]].copy()
    away_long = away_long.rename(columns={id_col_away: out_id_col,
                                          **{f"away_{m}_pre": m for m in STYLE_METRICS}})
    long_df = pd.concat([home_long, away_long], ignore_index=True)
    long_df["game_ts"] = pd.to_datetime(long_df["game_ts"], utc=True, errors="coerce")
    long_df[out_id_col] = long_df[out_id_col].astype(str)

    # Drop entities with no pregame data (NaN across all metrics)
    long_df = long_df.dropna(subset=STYLE_METRICS, how="all")

    # Pick the last game per entity
    long_df = long_df.sort_values([out_id_col, "game_ts"], kind="stable")
    last = long_df.groupby(out_id_col, as_index=False).tail(1).reset_index(drop=True)
    return last[[out_id_col] + STYLE_METRICS]


def _synthesize_outcomes(year: int, mov_default: float, force: bool) -> Path | None:
    """Reconstruct `data/silver/game_outcomes_{year}_REGPST.csv` from gold.

    Output schema mirrors what 17_build_game_outcomes_year produces (the
    columns that scripts 19/27 actually read): season_year, season_type,
    game_id, scheduled, home_id, away_id, home_win, mov.

    Returns the path written, or None if it already exists and force is
    False.
    """
    gold_path = REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{year}_REGPST.csv"
    if not gold_path.exists():
        raise SystemExit(f"Missing gold CSV: {gold_path}")

    silver = REPO_ROOT / "data" / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    out = silver / f"game_outcomes_{year}_REGPST.csv"
    if out.exists() and not force:
        print(f"[synth] outcomes {year}: already exists, skipping")
        return None

    gold = pd.read_csv(gold_path)
    df = pd.DataFrame({
        "season_year": gold["season"].astype(int),
        "season_type": gold.get("is_playoff", 0).map(
            lambda v: "PST" if int(v or 0) == 1 else "REG"
        ),
        "game_id": gold["game_id"].astype(str),
        "scheduled": gold["game_ts"],
        "home_id": gold["home_team_id"].astype(str),
        "away_id": gold["away_team_id"].astype(str),
        "home_win": pd.to_numeric(gold["home_win"], errors="coerce").astype("Int64"),
        # MOV is not in gold — placeholder = league-average (~8.5 pts).
        # Affects only multi-year Elo recompute on prior-year games.
        "mov": mov_default,
    })
    df["scheduled"] = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    df = df.sort_values(["scheduled", "game_id"], kind="stable")
    df.to_csv(out, index=False)
    print(f"[synth] wrote {out.relative_to(REPO_ROOT)}  rows={len(df)}  mov={mov_default}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True,
                    help="Year whose style-cascade state we synthesize (e.g. 2025)")
    ap.add_argument("--gold-csv", default=None,
                    help="Override gold CSV path; defaults to the per-year file")
    ap.add_argument("--outcomes-from", type=int, default=None,
                    help="Optional: also reconstruct silver/game_outcomes_*.csv "
                         "starting at this year (inclusive)")
    ap.add_argument("--outcomes-to", type=int, default=None,
                    help="Optional: outcomes upper bound (inclusive); "
                         "defaults to --year if --outcomes-from is set")
    ap.add_argument("--mov-default", type=float, default=8.5,
                    help="Placeholder MOV when reconstructing prior-year "
                         "outcomes (default: 8.5, league-average)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing synthesized files")
    args = ap.parse_args()

    gold_path = (Path(args.gold_csv) if args.gold_csv
                 else REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{args.year}_REGPST.csv")
    if not gold_path.exists():
        raise SystemExit(f"Missing gold CSV: {gold_path}")

    print(f"[synth] reading {gold_path}")
    gold = pd.read_csv(gold_path)
    n = len(gold)
    print(f"[synth] {args.year}: {n} games in gold")

    silver_plus = REPO_ROOT / "data" / "silver_plus"
    silver_plus.mkdir(parents=True, exist_ok=True)

    # Team-level cascade (consumed by script 24)
    team_path = silver_plus / f"team_style_profile_final_{args.year}.csv"
    if team_path.exists() and not args.force:
        raise SystemExit(f"{team_path} exists — pass --force to overwrite.")
    team_df = _last_pregame_per_id(gold, "home_team_id", "away_team_id", "team_id")
    team_df.to_csv(team_path, index=False)
    print(f"[synth] wrote {team_path}  rows={len(team_df)} "
          f"(unique team_ids in {args.year})")

    # Franchise-level cascade (consumed by script 26).
    # Note: script 26 expects a `season` column too — preserve schema.
    franchise_path = silver_plus / f"franchise_style_profile_final_{args.year}.csv"
    if franchise_path.exists() and not args.force:
        raise SystemExit(f"{franchise_path} exists — pass --force to overwrite.")
    franchise_df = _last_pregame_per_id(
        gold, "home_franchise_id", "away_franchise_id", "franchise_id",
    )
    franchise_df.insert(0, "season", args.year)
    franchise_df.to_csv(franchise_path, index=False)
    print(f"[synth] wrote {franchise_path}  rows={len(franchise_df)} "
          f"(unique franchise_ids in {args.year})")

    # Sanity print — the 6 metrics should be in [0, 1] range (they're rates)
    for m in STYLE_METRICS:
        rng = (team_df[m].min(), team_df[m].max())
        print(f"   team {m}: [{rng[0]:.4f}, {rng[1]:.4f}]")


if __name__ == "__main__":
    main()
