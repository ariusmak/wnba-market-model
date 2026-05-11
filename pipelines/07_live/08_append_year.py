"""
08_append_year.py
=================

End-to-end orchestrator that runs the existing year-suffix pipeline for
ONE year, in dependency order. Use this to:

  • Build silver / silver_plus / gold for a brand-new season (e.g. 2026)
  • Refresh a season mid-flight as new games complete (idempotent — each
    underlying script skips already-processed bronze)

What it runs (per year Y):
  1. Ingest  : bronze schedule (REG + PST) → game summaries → daily injuries
  2. Parse   : played-games manifest, outcomes, injury events/episodes,
               game availability, player game-box (canonical facts in data/silver/)
  3. Feature : daily player state remains in data/silver/; game-as-of feature
               families such as game_team_player, recent form, style profile,
               schedule context, franchise Elo, and franchise recent form go to
               data/silver_plus/
  4. Multiyear: re-runs Elo and franchise-Elo across [2015..Y]
  5. Gold    : data/gold/game_xgboost_input_{Y}_REGPST.csv

Multi-year scripts (19 elo_team_game_tables, 27 franchise_elo) take a
--start-year / --end-year and recompute across the whole range. Per-year
scripts (26 franchise_style_profile, 28 franchise_recent_form) cascade
from the previous year's saved state — that state must already exist on
disk for years < Y. The repo ships gold CSVs through 2025, so appending
2026 just needs the silver/silver_plus rebuild from bronze.

Usage:
    python pipelines/07_live/08_append_year.py --year 2026
    python pipelines/07_live/08_append_year.py --year 2026 --from-phase parse
    python pipelines/07_live/08_append_year.py --year 2026 --skip-ingest
    python pipelines/07_live/08_append_year.py --year 2026 --dry-run

Concatenate the gold CSVs into a fresh training file with
    python pipelines/07_live/09_combine_gold.py --start-year 2015 --end-year 2026
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]


def _child_env() -> dict:
    """Return a child env that puts repo `src/` on PYTHONPATH so the
    underlying scripts can `from srwnba...` without each one fixing sys.path."""
    env = os.environ.copy()
    src = str(REPO_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else (src + os.pathsep + existing)
    return env


# (script_relative_to_pipelines, args_template) — {Y}/{S}/{ACCESS} are placeholders
PHASES: dict[str, List[Tuple[str, List[str]]]] = {
    "ingest": [
        ("01_ingestion/00_backfill_schedule_year.py",
         ["--year", "{Y}", "--season-type", "REG", "--access-level", "{ACCESS}"]),
        ("01_ingestion/00_backfill_schedule_year.py",
         ["--year", "{Y}", "--season-type", "PST", "--access-level", "{ACCESS}"]),
        ("01_ingestion/12_backfill_game_summaries_year.py",
         ["--year", "{Y}", "--season-type", "REG", "--only-closed",
          "--access-level", "{ACCESS}", "--force-refresh-recent-days", "2"]),
        ("01_ingestion/12_backfill_game_summaries_year.py",
         ["--year", "{Y}", "--season-type", "PST", "--only-closed",
          "--access-level", "{ACCESS}", "--force-refresh-recent-days", "2"]),
        ("01_ingestion/10_backfill_daily_injuries_year.py",
         ["--year", "{Y}", "--access-level", "{ACCESS}"]),
    ],
    "parse": [
        # Parsers that turn bronze into silver. Order matters: the manifest +
        # outcomes feed everything downstream, so they go first.
        ("02_parsing/14_build_played_games_manifest_year.py", ["--year", "{Y}"]),
        ("02_parsing/17_build_game_outcomes_year.py", ["--year", "{Y}"]),
        ("02_parsing/11_extract_injury_events_year.py", ["--year", "{Y}"]),
        ("02_parsing/13_extract_game_availability_year.py", ["--year", "{Y}"]),
        ("02_parsing/15_build_augmented_injury_events_year.py", ["--year", "{Y}"]),
        ("02_parsing/16_build_injury_episodes_year.py", ["--year", "{Y}"]),
        ("02_parsing/20_build_player_game_box_year.py", ["--year", "{Y}"]),
    ],
    "feature": [
        # Per-year feature builders. 25 (played_franchise_games) is foundational
        # for 24/26/28 so it runs early. 21→22 is the player-slot chain. 23/24
        # are independent recent-form / style builders. 26+28 are franchise-level
        # rollups that need 24/23 + 25. 29 is rest/travel from bronze + outcomes.
        ("03_features/21_build_player_state_history_year.py",
         ["--year", "{Y}", "--through-date", "{THROUGH_DATE}"]),
        ("03_features/22_build_game_team_player_year.py", ["--year", "{Y}"]),
        ("03_features/25_build_played_franchise_games_year.py", ["--year", "{Y}"]),
        ("03_features/23_build_game_team_recent_form_year.py", ["--year", "{Y}"]),
        ("03_features/24_build_game_team_style_profile_year.py", ["--year", "{Y}"]),
        ("03_features/26_build_franchise_style_profile.py", ["--year", "{Y}"]),
        ("03_features/28_build_franchise_recent_form.py", ["--year", "{Y}"]),
        ("03_features/29_build_game_team_schedule_context.py", ["--year", "{Y}"]),
    ],
    "multiyear": [
        # Cross-year aggregations — run AFTER per-year features for Y exist.
        ("07_live/11_extend_elo_to_year.py", ["--year", "{Y}", "--force"]),
    ],
    "gold": [
        # Wrapper around 04_gold/30 that builds one year only and skips
        # script 30's `main_range` tail step (which assumes 2015-2024 is
        # in scope and crashes when called for a single newer year).
        ("07_live/12_build_gold_year.py", ["--year", "{Y}"]),
    ],
}

PHASE_ORDER = ["ingest", "parse", "feature", "multiyear", "gold"]


def _expand(args: List[str], year: int, access: str, start_year: int, through_date: str | None) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--through-date" and i + 1 < len(args) and args[i + 1] == "{THROUGH_DATE}" and not through_date:
            i += 2
            continue
        out.append(
            a.replace("{Y}", str(year))
             .replace("{S}", "REG")
             .replace("{ACCESS}", access)
             .replace("{START}", str(start_year))
             .replace("{THROUGH_DATE}", str(through_date or ""))
        )
        i += 1
    return out


def _run_step(script_rel: str, args: List[str], dry_run: bool) -> int:
    script_path = REPO_ROOT / "pipelines" / script_rel
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n  >> {script_rel} {' '.join(args)}")
    if dry_run:
        print("     [dry-run, skipped]")
        return 0
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_child_env(), check=False)
    except FileNotFoundError as exc:
        print(f"     MISSING SCRIPT: {script_path}  ({exc})")
        return 127
    dt = time.monotonic() - t0
    rc = proc.returncode
    status = "ok" if rc == 0 else f"rc={rc}"
    print(f"     {status}  ({dt:0.1f}s)")
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True,
                    help="Season year to (re)build, e.g. 2026")
    ap.add_argument("--access-level", default="trial",
                    help="Sportradar access tier (trial|production)")
    ap.add_argument("--start-year", type=int, default=2015,
                    help="Lower bound for multi-year aggregations (Elo, franchise)")
    ap.add_argument("--from-phase", choices=PHASE_ORDER, default="ingest",
                    help="Resume from a specific phase (skips earlier ones)")
    ap.add_argument("--to-phase", choices=PHASE_ORDER, default="gold",
                    help="Stop after this phase (inclusive)")
    ap.add_argument("--skip-ingest", action="store_true",
                    help="Equivalent to --from-phase parse")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run without executing")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Don't stop the pipeline on a non-zero exit code")
    ap.add_argument(
        "--player-state-through-date",
        default=None,
        help="Optional YYYY-MM-DD live inference date to carry player_state_history through.",
    )
    args = ap.parse_args()

    if args.skip_ingest and args.from_phase == "ingest":
        args.from_phase = "parse"

    start_idx = PHASE_ORDER.index(args.from_phase)
    end_idx = PHASE_ORDER.index(args.to_phase)
    if start_idx > end_idx:
        raise SystemExit(f"--from-phase {args.from_phase} > --to-phase {args.to_phase}")

    phases = PHASE_ORDER[start_idx:end_idx + 1]
    print(f"[append] year={args.year} phases={phases} dry_run={args.dry_run}")

    overall_start = time.monotonic()
    failures: list[str] = []
    for ph in phases:
        print(f"\n=== phase: {ph} ===")
        for script_rel, raw_args in PHASES[ph]:
            cmd_args = _expand(
                raw_args,
                args.year,
                args.access_level,
                args.start_year,
                args.player_state_through_date,
            )
            rc = _run_step(script_rel, cmd_args, args.dry_run)
            if rc != 0:
                failures.append(f"{script_rel} (rc={rc})")
                if not args.continue_on_error:
                    print(f"\n[append] STOP — {script_rel} failed (rc={rc}). "
                          f"Re-run with --from-phase {ph} after fixing, or pass "
                          f"--continue-on-error to skip past failures.")
                    sys.exit(rc)

    dt = time.monotonic() - overall_start
    print(f"\n[append] DONE year={args.year}  ({dt:0.1f}s total)")
    if failures:
        print(f"[append] {len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print(f"[append] gold: data/gold/game_xgboost_input_{args.year}_REGPST.csv")
    print(f"[append] next: python pipelines/07_live/09_combine_gold.py "
          f"--start-year {args.start_year} --end-year {args.year}")


if __name__ == "__main__":
    main()
