"""
12_build_gold_year.py
=====================

Thin wrapper around `pipelines/04_gold/30_build_game_xgboost_input.py`
that builds the gold CSV for ONE year only. Script 30's `main_range`
also tries to write a combined `2015_2024_REGPST.csv` at the end and
crashes when the year being built is outside that range — this wrapper
just calls the per-year `main(year)` and skips the tail step.

Usage:
    python pipelines/07_live/12_build_gold_year.py --year 2026
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "04_gold"))

# Import the per-year `main` from script 30. The numeric prefix is fine
# in module names; we just need importlib because of the leading digit.
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "_gold_30",
    REPO_ROOT / "pipelines" / "04_gold" / "30_build_game_xgboost_input.py",
)
_gold30 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gold30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    args = ap.parse_args()
    print(f"=== {args.year} ===")
    _gold30.main(args.year)


if __name__ == "__main__":
    main()
