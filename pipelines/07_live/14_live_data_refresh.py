"""
Production live data refresh scheduler.

This is the small operational layer that connects:
  - fixed settled-history refreshes (02:30 ET and 09:00 ET),
  - market/game-time driven T-20 refresh triggers,
  - staged API-pull audit artifacts,
  - downstream validation and gold/training rebuilds.

The script is intentionally idempotent. It keeps a local state file so an
hourly automation can wake it up frequently without rerunning the same
scheduled refresh or T-20 refresh packet twice.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from srwnba.live.canonical.kalshi_mapping import (  # noqa: E402
    SportRadarGameRef,
    filter_open_wnba_moneyline_markets,
    filter_wnba_moneyline_markets,
    load_team_name_map,
    map_game_to_kalshi_markets,
    parse_datetime,
)
from utils.kalshi_authed_client import AuthedKalshiClient, KalshiAuthConfig  # noqa: E402

EASTERN = ZoneInfo("America/New_York")
RUN_ROOT = REPO_ROOT / "data" / "runs" / "live_refresh"
DAEMON_RUN_ROOT = REPO_ROOT / "data" / "runs" / "live_daemon"
STATE_PATH = RUN_ROOT / "scheduler_state.json"
DEFAULT_SERIES = ("KXWNBAGAME", "KXWNBAH")


@dataclass(frozen=True)
class DueJob:
    key: str
    kind: str
    reason: str
    scheduled_for_et: str | None = None
    game: dict[str, Any] | None = None
    mapping: dict[str, Any] | None = None


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        rec = {
            "ts_utc": utc_now().isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--access-level", default="trial")
    ap.add_argument(
        "--mode",
        choices=["due", "settled", "market"],
        default="due",
        help="due=run any scheduled job now due; settled=force settled-history refresh; market=check T-20 due games only.",
    )
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--today", default=None, help="Validation date override, YYYY-MM-DD.")
    ap.add_argument("--now", default=None, help="Clock override for tests, ISO timestamp.")
    ap.add_argument("--t20-window-minutes", type=int, default=75)
    ap.add_argument("--market-discovery-limit", type=int, default=100)
    ap.add_argument("--series-ticker", action="append")
    ap.add_argument("--team-name-map", default=str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"))
    ap.add_argument(
        "--market-snapshot-json",
        default=None,
        help="Use an already-cached Kalshi market snapshot instead of pulling markets inside this worker.",
    )
    ap.add_argument(
        "--daemon-snapshot-max-age-minutes",
        type=float,
        default=15.0,
        help="Maximum age for reusing the daemon's latest Kalshi market snapshot before trying direct API.",
    )
    ap.add_argument("--skip-market-api", action="store_true")
    ap.add_argument("--force", action="store_true", help="Ignore scheduler state for the requested mode.")
    ap.add_argument("--dry-run", action="store_true", help="Plan and log, but do not run refresh commands or call Kalshi.")
    args = ap.parse_args()

    now = parse_now(args.now)
    run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{args.mode}_{uuid4().hex[:8]}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = JsonlLogger(run_dir / "logs.jsonl")
    state = load_state()

    logger.write(
        "run_start",
        run_id=run_id,
        mode=args.mode,
        dry_run=args.dry_run,
        now_utc=now.astimezone(timezone.utc).isoformat(),
        now_et=now.astimezone(EASTERN).isoformat(),
    )

    bronze_before = snapshot_paths(REPO_ROOT / "data" / "bronze")
    bronze_runs_before = snapshot_paths(REPO_ROOT / "data" / "bronze_runs")

    market_snapshot: dict[str, Any] = {"skipped": True, "markets": []}
    if args.market_snapshot_json:
        market_snapshot = json.loads(Path(args.market_snapshot_json).read_text(encoding="utf-8"))
        market_snapshot = moneyline_only_snapshot(market_snapshot)
        write_json(run_dir / "kalshi_markets_snapshot.json", market_snapshot)
        logger.write(
            "kalshi_market_snapshot_loaded",
            path=args.market_snapshot_json,
            markets=len(market_snapshot.get("markets", []) or []),
            raw_markets=market_snapshot.get("raw_market_count"),
            filtered_out=market_snapshot.get("filtered_out_count"),
        )
    elif args.mode in {"due", "market"}:
        daemon_snapshot = load_latest_daemon_snapshot(max_age_minutes=args.daemon_snapshot_max_age_minutes)
        if daemon_snapshot is not None:
            market_snapshot = daemon_snapshot
            write_json(run_dir / "kalshi_markets_snapshot.json", market_snapshot)
            logger.write(
                "kalshi_market_snapshot_loaded_from_daemon",
                markets=len(market_snapshot.get("markets", []) or []),
                raw_markets=market_snapshot.get("raw_market_count"),
                filtered_out=market_snapshot.get("filtered_out_count"),
                pulled_at_utc=market_snapshot.get("pulled_at_utc"),
            )
        elif args.skip_market_api or args.dry_run:
            logger.write(
                "kalshi_market_snapshot_skipped",
                dry_run=args.dry_run,
                skip_market_api=args.skip_market_api,
                reason="no_fresh_daemon_snapshot",
            )
        else:
            try:
                market_snapshot = fetch_kalshi_markets(
                    series_tickers=tuple(args.series_ticker or DEFAULT_SERIES),
                    limit=args.market_discovery_limit,
                    out_path=run_dir / "kalshi_markets_snapshot.json",
                    logger=logger,
                )
            except Exception as exc:
                stale_snapshot = load_latest_daemon_snapshot(max_age_minutes=None)
                if stale_snapshot is not None:
                    market_snapshot = stale_snapshot
                    market_snapshot["warning"] = f"direct Kalshi API failed; using stale daemon snapshot: {exc}"
                    write_json(run_dir / "kalshi_markets_snapshot.json", market_snapshot)
                    logger.write(
                        "kalshi_market_snapshot_loaded_from_stale_daemon_after_api_failure",
                        error=str(exc),
                        markets=len(market_snapshot.get("markets", []) or []),
                        pulled_at_utc=market_snapshot.get("pulled_at_utc"),
                    )
                else:
                    market_snapshot = {
                        "skipped": False,
                        "error": str(exc),
                        "markets": [],
                        "pulled_at_utc": utc_now().isoformat(),
                    }
                    write_json(run_dir / "kalshi_markets_snapshot.json", market_snapshot)
                    logger.write("kalshi_market_snapshot_failed", error=str(exc))

    skip_market_mapping = args.mode == "settled" or not market_snapshot.get("markets")
    jobs = determine_due_jobs(
        year=args.year,
        mode=args.mode,
        now=now,
        state=state,
        force=args.force,
        t20_window_minutes=args.t20_window_minutes,
        markets=market_snapshot.get("markets", []),
        team_name_map_path=Path(args.team_name_map),
        skip_market_mapping=skip_market_mapping,
        logger=logger,
    )
    write_json(run_dir / "due_jobs.json", {"jobs": [job.__dict__ for job in jobs]})

    preflight_report = evaluate_t20_market_preflight(
        jobs,
        market_snapshot=market_snapshot,
        dry_run=args.dry_run,
    )
    write_json(run_dir / "preflight_report.json", preflight_report)
    if preflight_report["issues"]:
        logger.write("market_preflight_warning", issues=preflight_report["issues"])

    preflight_issue_job_keys = {
        err["job_key"]
        for err in preflight_report["issues"]
        if err.get("job_key")
    }
    runnable_jobs = list(jobs)
    should_refresh = any(job.kind in {"settled_history", "market_t20"} for job in runnable_jobs)
    command_results: list[dict[str, Any]] = []
    command_success = True

    market_snapshot_path = (
        run_dir / "kalshi_markets_snapshot.json"
        if (run_dir / "kalshi_markets_snapshot.json").exists()
        else Path(args.market_snapshot_json) if args.market_snapshot_json else None
    )

    if should_refresh:
        command_results = run_settled_refresh(
            year=args.year,
            access_level=args.access_level,
            start_year=args.start_year,
            today=args.today or now.astimezone(timezone.utc).date().isoformat(),
            run_dir=run_dir,
            logger=logger,
            dry_run=args.dry_run,
        )
        command_success = all(r["returncode"] == 0 for r in command_results)
        if command_success:
            packet_results = run_t20_prediction_packets(
                year=args.year,
                jobs=runnable_jobs,
                asof=now,
                train_csv=REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{args.start_year}_{args.year}_REGPST.csv",
                market_snapshot_path=market_snapshot_path,
                team_name_map_path=Path(args.team_name_map),
                run_dir=run_dir,
                logger=logger,
                dry_run=args.dry_run,
                start_idx=len(command_results) + 1,
            )
            command_results.extend(packet_results)
            command_success = all(r["returncode"] == 0 for r in command_results)
    else:
        logger.write("no_refresh_due")

    success = command_success
    if command_results:
        write_json(run_dir / "command_results.json", {"commands": command_results})

    bronze_after = snapshot_paths(REPO_ROOT / "data" / "bronze")
    bronze_runs_after = snapshot_paths(REPO_ROOT / "data" / "bronze_runs")
    bronze_delta = {
        "bronze_added": sorted(bronze_after - bronze_before),
        "bronze_runs_added": sorted(bronze_runs_after - bronze_runs_before),
    }
    write_json(run_dir / "bronze_delta.json", bronze_delta)

    validation_report = build_validation_report(args.year)
    write_json(run_dir / "validation_report.json", validation_report)
    promoted_outputs = {
        "gold_year": str(REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{args.year}_REGPST.csv"),
        "combined_training": str(REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{args.start_year}_{args.year}_REGPST.csv"),
        "live_features_dir": str(REPO_ROOT / "data" / "live_features"),
        "live_prediction_packets": [
            str(REPO_ROOT / "data" / "runs" / "live_games" / str((job.game or {}).get("game_id")) / "prediction_packet.json")
            for job in runnable_jobs
            if job.kind == "market_t20" and job.game
        ],
    }
    write_json(run_dir / "promoted_outputs.json", promoted_outputs)

    if command_success and not args.dry_run:
        mark_jobs_complete(state, runnable_jobs, run_id=run_id)
        save_state(state)

    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "dry_run": args.dry_run,
        "success": success,
        "started_at_utc": run_id.split("_", 1)[0],
        "now_utc": now.astimezone(timezone.utc).isoformat(),
        "now_et": now.astimezone(EASTERN).isoformat(),
        "jobs": [job.__dict__ for job in jobs],
        "runnable_jobs": [job.__dict__ for job in runnable_jobs],
        "market_preflight_issue_job_keys": sorted(preflight_issue_job_keys),
        "preflight_report_path": str(run_dir / "preflight_report.json"),
        "preflight_report": preflight_report,
        "market_preflight_ok": preflight_report["ok"],
        "command_results": command_results,
        "bronze_delta": bronze_delta,
        "validation_report_path": str(run_dir / "validation_report.json"),
        "promoted_outputs_path": str(run_dir / "promoted_outputs.json"),
        "market_snapshot_path": str(run_dir / "kalshi_markets_snapshot.json")
        if (run_dir / "kalshi_markets_snapshot.json").exists()
        else args.market_snapshot_json,
    }
    write_json(run_dir / "manifest.json", manifest)
    logger.write("run_end", success=success, run_dir=str(run_dir))

    print(f"[live-refresh] run_id={run_id}")
    print(f"[live-refresh] run_dir={run_dir}")
    print(f"[live-refresh] jobs={len(jobs)} refresh={should_refresh} success={success}")
    if not success:
        raise SystemExit(1)


def determine_due_jobs(
    *,
    year: int,
    mode: str,
    now: datetime,
    state: dict[str, Any],
    force: bool,
    t20_window_minutes: int,
    markets: list[dict[str, Any]],
    team_name_map_path: Path,
    skip_market_mapping: bool,
    logger: JsonlLogger,
) -> list[DueJob]:
    jobs: list[DueJob] = []
    now_et = now.astimezone(EASTERN)

    if mode in {"due", "settled"}:
        jobs.extend(settled_history_due_jobs(now_et, state, force=force or mode == "settled"))

    if mode in {"due", "market"}:
        games = load_latest_schedule_games(year)
        team_name_map = {}
        if not skip_market_mapping and team_name_map_path.exists():
            team_name_map = load_team_name_map(str(team_name_map_path))
        jobs.extend(
            market_t20_due_jobs(
                games=games,
                now=now,
                state=state,
                force=force and mode == "market",
                window_minutes=t20_window_minutes,
                markets=markets,
                team_name_map=team_name_map,
                skip_mapping=skip_market_mapping,
                logger=logger,
            )
        )

    logger.write("due_jobs_resolved", count=len(jobs), jobs=[job.__dict__ for job in jobs])
    return jobs


def settled_history_due_jobs(now_et: datetime, state: dict[str, Any], *, force: bool) -> list[DueJob]:
    jobs: list[DueJob] = []
    schedule = [
        ("main_0230", 2, 30),
        ("backup_0900", 9, 0),
    ]
    completed = state.setdefault("completed_jobs", {})
    today = now_et.date().isoformat()

    for label, hour, minute in schedule:
        scheduled = now_et.replace(hour=hour, minute=minute, second=0, microsecond=0)
        key = f"settled_history:{today}:{label}"
        if (force or now_et >= scheduled) and key not in completed:
            jobs.append(
                DueJob(
                    key=key,
                    kind="settled_history",
                    reason=label if not force else f"forced_{label}",
                    scheduled_for_et=scheduled.isoformat(),
                )
            )
    return jobs


def market_t20_due_jobs(
    *,
    games: list[dict[str, Any]],
    now: datetime,
    state: dict[str, Any],
    force: bool,
    window_minutes: int,
    markets: list[dict[str, Any]],
    team_name_map: dict[str, str],
    skip_mapping: bool,
    logger: JsonlLogger,
) -> list[DueJob]:
    jobs: list[DueJob] = []
    completed = state.setdefault("completed_jobs", {})
    now_utc = now.astimezone(timezone.utc)
    window = timedelta(minutes=window_minutes)

    for raw in games:
        scheduled_raw = raw.get("scheduled")
        game_id = raw.get("id")
        if not scheduled_raw or not game_id:
            continue
        scheduled = parse_datetime(str(scheduled_raw)).astimezone(timezone.utc)
        target = scheduled - timedelta(hours=20)
        due = force or (target <= now_utc <= target + window)
        key = f"market_t20:{game_id}"
        if not due or key in completed:
            continue

        game_ref = game_ref_from_schedule(raw)
        mapping_payload: dict[str, Any] | None = None
        if not skip_mapping:
            try:
                mapping = map_game_to_kalshi_markets(
                    game_ref,
                    markets,
                    require_open=True,
                    team_name_to_id=team_name_map,
                )
                mapping_payload = {
                    "confirmed": mapping.confirmed,
                    "event_ticker": mapping.event_ticker,
                    "candidate_count": mapping.candidate_count,
                    "diagnostics": mapping.diagnostics[:12],
                    "home_market": getattr(mapping.home_market, "ticker", ""),
                    "away_market": getattr(mapping.away_market, "ticker", ""),
                    "complement_market_confirmed": mapping.complement_market_confirmed,
                }
            except Exception as exc:
                mapping_payload = {"confirmed": False, "error": str(exc)}
                logger.write("market_mapping_failed", game_id=game_id, error=str(exc))

        jobs.append(
            DueJob(
                key=key,
                kind="market_t20",
                reason="t20_due",
                scheduled_for_et=target.astimezone(EASTERN).isoformat(),
                game={
                    "game_id": game_id,
                    "scheduled_utc": scheduled.isoformat(),
                    "t20_target_utc": target.isoformat(),
                    "home_team_id": game_ref.home_team_id,
                    "away_team_id": game_ref.away_team_id,
                    "home_team_name": game_ref.home_team_name,
                    "away_team_name": game_ref.away_team_name,
                },
                mapping=mapping_payload,
            )
        )
    return jobs


def evaluate_t20_market_preflight(
    jobs: list[DueJob],
    *,
    market_snapshot: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    snapshot_error = str(market_snapshot.get("error") or "").strip()
    snapshot_skipped = bool(market_snapshot.get("skipped"))

    for job in jobs:
        if job.kind != "market_t20":
            continue

        if dry_run:
            if job.mapping is None:
                warnings.append({
                    "job_key": job.key,
                    "code": "market_mapping_not_checked_dry_run",
                    "message": "Dry run did not check Kalshi mapping because no snapshot was supplied.",
                })
            continue

        if snapshot_error:
            issues.append({
                "job_key": job.key,
                "code": "kalshi_snapshot_failed",
                "message": snapshot_error,
            })
            continue

        if snapshot_skipped:
            issues.append({
                "job_key": job.key,
                "code": "kalshi_snapshot_skipped",
                "message": "T-20 market jobs require an active/open WNBA moneyline Kalshi snapshot.",
            })
            continue

        if job.mapping is None:
            issues.append({
                "job_key": job.key,
                "code": "market_mapping_not_checked",
                "message": "T-20 market job had no Kalshi mapping payload.",
            })
            continue

        if not job.mapping.get("confirmed"):
            issues.append({
                "job_key": job.key,
                "code": "market_mapping_not_confirmed",
                "message": "T-20 market job requires confirmed active/open WNBA moneyline mapping.",
                "mapping": job.mapping,
            })
            continue

        if not job.mapping.get("complement_market_confirmed"):
            issues.append({
                "job_key": job.key,
                "code": "complement_market_not_confirmed",
                "message": "T-20 market job requires two complementary team-wins contracts.",
                "mapping": job.mapping,
            })

    return {
        "ok": not issues,
        "dry_run": dry_run,
        "issues": issues,
        "warnings": warnings,
    }


def run_settled_refresh(
    *,
    year: int,
    access_level: str,
    start_year: int,
    today: str,
    run_dir: Path,
    logger: JsonlLogger,
    dry_run: bool,
) -> list[dict[str, Any]]:
    commands = [
        [
            sys.executable,
            "pipelines/07_live/08_append_year.py",
            "--year",
            str(year),
            "--to-phase",
            "feature",
            "--access-level",
            access_level,
            "--player-state-through-date",
            today,
        ],
        [
            sys.executable,
            "pipelines/07_live/11_extend_elo_to_year.py",
            "--year",
            str(year),
            "--force",
        ],
        [sys.executable, "pipelines/07_live/12_build_gold_year.py", "--year", str(year)],
        [
            sys.executable,
            "pipelines/07_live/09_combine_gold.py",
            "--start-year",
            str(start_year),
            "--end-year",
            str(year),
            "--force",
        ],
        [
            sys.executable,
            "pipelines/07_live/13_validate_production_artifacts.py",
            "--year",
            str(year),
            "--today",
            today,
            "--start-year",
            str(start_year),
        ],
    ]
    results: list[dict[str, Any]] = []
    for idx, cmd in enumerate(commands, start=1):
        results.append(run_command(idx, cmd, run_dir=run_dir, logger=logger, dry_run=dry_run))
        if results[-1]["returncode"] != 0:
            break
    write_json(run_dir / "command_results.json", {"commands": results})
    return results


def run_t20_prediction_packets(
    *,
    year: int,
    jobs: list[DueJob],
    asof: datetime,
    train_csv: Path,
    market_snapshot_path: Path | None,
    team_name_map_path: Path,
    run_dir: Path,
    logger: JsonlLogger,
    dry_run: bool,
    start_idx: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    packet_jobs = [job for job in jobs if job.kind == "market_t20" and job.game]
    for offset, job in enumerate(packet_jobs):
        game_id = str((job.game or {}).get("game_id") or "")
        if not game_id:
            continue
        cmd = [
            sys.executable,
            "pipelines/07_live/17_build_live_prediction_packet.py",
            "--year",
            str(year),
            "--game-id",
            game_id,
            "--asof",
            asof.astimezone(timezone.utc).isoformat(),
            "--train-csv",
            str(train_csv),
            "--feature-csv",
            str(REPO_ROOT / "data" / "live_features" / f"{game_id}.csv"),
            "--packet-json",
            str(REPO_ROOT / "data" / "runs" / "live_games" / game_id / "prediction_packet.json"),
            "--team-name-map",
            str(team_name_map_path),
        ]
        if market_snapshot_path is not None and market_snapshot_path.exists():
            cmd.extend(["--market-snapshot-json", str(market_snapshot_path)])
        results.append(
            run_command(start_idx + offset, cmd, run_dir=run_dir, logger=logger, dry_run=dry_run)
        )
        if results[-1]["returncode"] != 0:
            break
    if results:
        write_json(run_dir / "prediction_packet_results.json", {"commands": results})
    return results


def run_command(
    idx: int,
    cmd: list[str],
    *,
    run_dir: Path,
    logger: JsonlLogger,
    dry_run: bool,
) -> dict[str, Any]:
    log_dir = run_dir / "command_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    label = Path(cmd[1]).stem if len(cmd) > 1 else f"cmd_{idx}"
    log_path = log_dir / f"{idx:02d}_{label}.log"
    logger.write("command_start", idx=idx, cmd=cmd, log_path=str(log_path), dry_run=dry_run)
    t0 = time.monotonic()
    if dry_run:
        log_path.write_text("DRY RUN: " + " ".join(cmd) + "\n", encoding="utf-8")
        rc = 0
    else:
        env = os.environ.copy()
        src = str(SRC_ROOT)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(proc.stdout or "", encoding="utf-8")
        rc = proc.returncode
    duration = time.monotonic() - t0
    result = {
        "idx": idx,
        "cmd": cmd,
        "returncode": rc,
        "duration_s": round(duration, 3),
        "log_path": str(log_path),
    }
    logger.write("command_end", **result)
    return result


def fetch_kalshi_markets(
    *,
    series_tickers: Iterable[str],
    limit: int,
    out_path: Path,
    logger: JsonlLogger,
) -> dict[str, Any]:
    cfg = KalshiAuthConfig.from_env(REPO_ROOT / ".env")
    client = AuthedKalshiClient(cfg)
    markets: list[dict[str, Any]] = []
    pulls: list[dict[str, Any]] = []
    for series in series_tickers:
        logger.write("kalshi_api_pull_start", endpoint="list_markets", series_ticker=series, limit=limit)
        t0 = time.monotonic()
        items = client.list_markets(series_ticker=series, limit=limit)
        moneyline_items = list(filter_wnba_moneyline_markets(items))
        open_moneyline_items = list(filter_open_wnba_moneyline_markets(items))
        pull = {
            "endpoint": "list_markets",
            "series_ticker": series,
            "raw_count": len(items),
            "moneyline_count": len(moneyline_items),
            "open_moneyline_count": len(open_moneyline_items),
            "non_moneyline_filtered_out_count": len(items) - len(moneyline_items),
            "closed_moneyline_filtered_out_count": len(moneyline_items) - len(open_moneyline_items),
            "filtered_out_count": len(items) - len(open_moneyline_items),
            "duration_s": round(time.monotonic() - t0, 3),
        }
        logger.write("kalshi_api_pull_end", **pull)
        pulls.append(pull)
        markets.extend(open_moneyline_items)
    snapshot = {
        "pulled_at_utc": utc_now().isoformat(),
        "base_url": cfg.base_url,
        "trading_enabled": cfg.trading_enabled,
        "pulls": pulls,
        "raw_market_count": sum(int(p["raw_count"]) for p in pulls),
        "moneyline_market_count": sum(int(p["moneyline_count"]) for p in pulls),
        "open_moneyline_market_count": sum(int(p["open_moneyline_count"]) for p in pulls),
        "non_moneyline_filtered_out_count": sum(int(p["non_moneyline_filtered_out_count"]) for p in pulls),
        "closed_moneyline_filtered_out_count": sum(int(p["closed_moneyline_filtered_out_count"]) for p in pulls),
        "filtered_out_count": sum(int(p["filtered_out_count"]) for p in pulls),
        "moneyline_only": True,
        "open_markets_only": True,
        "markets": markets,
    }
    write_json(out_path, snapshot)
    return snapshot


def moneyline_only_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    raw_markets = list(snapshot.get("markets", []) or [])
    moneyline_markets = list(filter_wnba_moneyline_markets(raw_markets))
    markets = list(filter_open_wnba_moneyline_markets(raw_markets))
    out = dict(snapshot)
    out["raw_market_count"] = snapshot.get("raw_market_count", len(raw_markets))
    out["moneyline_market_count"] = snapshot.get("moneyline_market_count", len(moneyline_markets))
    out["open_moneyline_market_count"] = snapshot.get("open_moneyline_market_count", len(markets))
    out["non_moneyline_filtered_out_count"] = snapshot.get(
        "non_moneyline_filtered_out_count",
        len(raw_markets) - len(moneyline_markets),
    )
    out["closed_moneyline_filtered_out_count"] = snapshot.get(
        "closed_moneyline_filtered_out_count",
        len(moneyline_markets) - len(markets),
    )
    out["filtered_out_count"] = snapshot.get("filtered_out_count", len(raw_markets) - len(markets))
    out["moneyline_only"] = True
    out["open_markets_only"] = True
    out["markets"] = markets
    return out


def load_latest_daemon_snapshot(*, max_age_minutes: float | None) -> dict[str, Any] | None:
    candidates: list[Path] = []
    heartbeat_path = DAEMON_RUN_ROOT / "heartbeat_latest.json"
    if heartbeat_path.exists():
        try:
            heartbeat = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            raw = str(heartbeat.get("latest_market_snapshot_path") or "").strip()
            if raw:
                candidates.append(Path(raw))
        except json.JSONDecodeError:
            pass
    candidates.append(DAEMON_RUN_ROOT / "latest_market_snapshot.json")

    for path in candidates:
        if not path.exists():
            continue
        try:
            snapshot = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pulled_at_raw = snapshot.get("pulled_at_utc")
        if max_age_minutes is not None and pulled_at_raw:
            pulled_at = parse_datetime(pulled_at_raw)
            if pulled_at.tzinfo is None:
                pulled_at = pulled_at.replace(tzinfo=timezone.utc)
            age_minutes = (utc_now() - pulled_at.astimezone(timezone.utc)).total_seconds() / 60.0
            if age_minutes > max_age_minutes:
                continue
        out = moneyline_only_snapshot(snapshot)
        out["source_snapshot_path"] = str(path)
        out["source"] = "live_daemon"
        return out
    return None


def build_validation_report(year: int) -> dict[str, Any]:
    report: dict[str, Any] = {
        "built_at_utc": utc_now().isoformat(),
        "year": year,
        "files": {},
        "latest_bronze": {},
    }
    csvs = {
        "played": REPO_ROOT / "data" / "silver" / f"played_games_{year}_REGPST.csv",
        "outcomes": REPO_ROOT / "data" / "silver" / f"game_outcomes_{year}_REGPST.csv",
        "injury_events": REPO_ROOT / "data" / "silver" / f"injury_events_{year}.csv",
        "player_state": REPO_ROOT / "data" / "silver" / f"player_state_history_{year}.csv",
        "game_player": REPO_ROOT / "data" / "silver_plus" / f"game_team_player_{year}_REGPST.csv",
        "schedule_context": REPO_ROOT / "data" / "silver_plus" / f"game_team_schedule_context_{year}_REGPST.csv",
        "gold": REPO_ROOT / "data" / "gold" / f"game_xgboost_input_{year}_REGPST.csv",
    }
    for name, path in csvs.items():
        rec: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if path.exists():
            df = pd.read_csv(path)
            rec["rows"] = int(len(df))
            rec["cols"] = int(len(df.columns))
            if name == "gold":
                rec["null_cells"] = int(df.isna().sum().sum())
        report["files"][name] = rec

    bronze = REPO_ROOT / "data" / "bronze"
    for pattern_name, pattern in {
        "reg_schedule": f"schedule_{year}_REG__*.json",
        "pst_schedule": f"schedule_{year}_PST__*.json",
    }.items():
        files = sorted(bronze.glob(pattern))
        if files:
            data = json.loads(files[-1].read_text(encoding="utf-8"))
            games = data.get("games", []) or []
            report["latest_bronze"][pattern_name] = {
                "file": files[-1].name,
                "games": len(games),
                "closed_games": sum(1 for g in games if str(g.get("status", "")).lower() == "closed"),
            }
    injury_files = sorted(bronze.glob(f"daily_injuries__{year}-*__*.json"))
    report["latest_bronze"]["daily_injury_file_count"] = len(injury_files)
    report["latest_bronze"]["future_daily_injury_files"] = [
        p.name for p in injury_files if _injury_date_from_name(p.name) > utc_now().date().isoformat()
    ]
    return report


def mark_jobs_complete(state: dict[str, Any], jobs: list[DueJob], *, run_id: str) -> None:
    completed = state.setdefault("completed_jobs", {})
    completed_at = utc_now().isoformat()
    for job in jobs:
        completed[job.key] = {
            "run_id": run_id,
            "completed_at_utc": completed_at,
            "kind": job.kind,
            "reason": job.reason,
            "scheduled_for_et": job.scheduled_for_et,
            "game": job.game,
        }


def load_latest_schedule_games(year: int) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    bronze = REPO_ROOT / "data" / "bronze"
    for season_type in ("REG", "PST"):
        files = sorted(bronze.glob(f"schedule_{year}_{season_type}__*.json"))
        if not files:
            continue
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        games.extend(data.get("games", []) or [])
    return games


def game_ref_from_schedule(raw: dict[str, Any]) -> SportRadarGameRef:
    home = raw.get("home") or {}
    away = raw.get("away") or {}
    return SportRadarGameRef(
        game_id=str(raw.get("id") or ""),
        scheduled=parse_datetime(str(raw.get("scheduled"))),
        home_team_id=str(home.get("id") or ""),
        away_team_id=str(away.get("id") or ""),
        home_team_name=team_display_name(home),
        away_team_name=team_display_name(away),
    )


def team_display_name(team: dict[str, Any]) -> str:
    parts = [str(team.get("market") or "").strip(), str(team.get("name") or "").strip()]
    return " ".join(p for p in parts if p).strip()


def snapshot_paths(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {str(p.relative_to(REPO_ROOT)) for p in root.rglob("*") if p.is_file()}


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"completed_jobs": {}}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def parse_now(value: str | None) -> datetime:
    if not value:
        return utc_now()
    parsed = parse_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _injury_date_from_name(name: str) -> str:
    parts = name.split("__")
    return parts[1] if len(parts) >= 3 else ""


if __name__ == "__main__":
    main()
