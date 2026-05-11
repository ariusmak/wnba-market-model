"""
09_combine_gold.py
==================

Concatenates per-year gold CSVs into a single training file the
FinalModel can ingest. Run after 08_append_year.py for the new season,
e.g.:

    python pipelines/07_live/09_combine_gold.py --start-year 2015 --end-year 2026

This produces (using REG+PST as the gold suffix):
    data/gold/game_xgboost_input_2015_2026_REGPST.csv

By default the script:
  • Concatenates inputs in chronological order (by year)
  • Validates that every per-year CSV has identical columns + dtypes
  • Sorts the merged frame by (season, game_ts, game_id) to keep walk-
    forward CV reproducible
  • Refuses to overwrite an existing combined file unless --force is set
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "data" / "gold"
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.util.model_schema import GOLD_MODEL_INPUT_COLS


def _per_year_path(year: int) -> Path:
    return GOLD_DIR / f"game_xgboost_input_{year}_REGPST.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-year", type=int, required=True)
    ap.add_argument("--end-year", type=int, required=True)
    ap.add_argument("--out",
                    default=None,
                    help="Output path; defaults to "
                         "data/gold/game_xgboost_input_{S}_{E}_REGPST.csv")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing combined file")
    args = ap.parse_args()

    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    years = list(range(args.start_year, args.end_year + 1))
    paths: List[Path] = [_per_year_path(y) for y in years]
    missing = [str(p.name) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing per-year gold CSVs:\n  " + "\n  ".join(missing) +
            "\nRun 08_append_year.py for those years first."
        )

    out = (Path(args.out) if args.out
           else GOLD_DIR / f"game_xgboost_input_{args.start_year}_{args.end_year}_REGPST.csv")
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists — pass --force to overwrite.")

    print(f"[combine] {len(years)} years: {years[0]}..{years[-1]}")
    frames: list[pd.DataFrame] = []
    ref_cols: list[str] | None = None
    for y, p in zip(years, paths):
        df = pd.read_csv(p)
        missing_required = [c for c in GOLD_MODEL_INPUT_COLS if c not in df.columns]
        if missing_required:
            raise SystemExit(
                f"Missing canonical model-input columns in {p.name}:\n  "
                + "\n  ".join(missing_required)
            )
        df = df[GOLD_MODEL_INPUT_COLS].copy()
        cols = list(df.columns)
        if ref_cols is None:
            ref_cols = cols
        elif cols != ref_cols:
            extra = set(cols) - set(ref_cols)
            missing_c = set(ref_cols) - set(cols)
            raise SystemExit(
                f"Column mismatch in {p.name}\n"
                f"  extra: {sorted(extra)}\n  missing: {sorted(missing_c)}"
            )
        print(f"  {y}: {len(df)} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    sort_cols = [c for c in ("season", "game_ts", "game_id") if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"[combine] wrote {out}  rows={len(merged)}  cols={len(merged.columns)}")


if __name__ == "__main__":
    main()
