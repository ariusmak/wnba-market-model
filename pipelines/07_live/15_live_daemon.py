"""
Always-on WNBA live daemon.

This is the Codex-independent heartbeat process for the live system. It:
  - polls Kalshi WNBA markets on a fixed cadence,
  - caches every market-list pull for audit,
  - detects new/updated/disappeared tickers,
  - writes daemon heartbeats and JSONL events,
  - invokes 14_live_data_refresh.py as the idempotent worker for fixed
    settled-history refreshes and T-20 game-time refreshes,
  - invokes 16_execution_supervisor.py to launch/supervise canonical
    per-game route entry loops.

The daemon itself does not place orders. Execution windows remain owned by
the canonical route entry loop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.kalshi_authed_client import AuthedKalshiClient, KalshiAuthConfig  # noqa: E402
from srwnba.live.canonical.kalshi_mapping import (  # noqa: E402
    filter_open_wnba_moneyline_markets,
    filter_wnba_moneyline_markets,
)

RUN_ROOT = REPO_ROOT / "data" / "runs" / "live_daemon"
SESSION_ROOT = RUN_ROOT / "sessions"
MARKET_PULL_ROOT = RUN_ROOT / "market_pulls"
STATE_PATH = RUN_ROOT / "daemon_state.json"
LOCK_PATH = RUN_ROOT / "live_daemon.lock"
HEALTH_LATEST_PATH = RUN_ROOT / "health_latest.json"
HEALTH_HISTORY_PATH = RUN_ROOT / "health.jsonl"
DEFAULT_SERIES = ("KXWNBAGAME", "KXWNBAH")


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
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--series-ticker", action="append")
    ap.add_argument("--market-discovery-limit", type=int, default=100)
    ap.add_argument("--market-poll-s", type=float, default=180.0)
    ap.add_argument("--worker-check-s", type=float, default=300.0)
    ap.add_argument("--execution-check-s", type=float, default=120.0)
    ap.add_argument("--heartbeat-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true", help="Run one loop iteration and exit.")
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--skip-market-api", action="store_true")
    ap.add_argument("--worker-dry-run", action="store_true")
    ap.add_argument("--disable-worker", action="store_true")
    ap.add_argument("--execution-dry-run", action="store_true",
                    help="Launch route loops with --dry-run for read-only execution testing.")
    ap.add_argument("--control-plane-mode",
                    choices=("local-only", "supabase-shadow", "supabase-live"),
                    default="local-only",
                    help="Remote control-plane mode passed to execution supervisor and route loops.")
    ap.add_argument("--control-plane-bot-id", default="wnba-live-daemon",
                    help="bot_heartbeat.bot_id prefix for worker status publishing.")
    ap.add_argument("--disable-execution-supervisor", action="store_true")
    ap.add_argument("--ignore-lock", action="store_true")
    args = ap.parse_args()

    run_id = f"{utc_now().strftime('%Y%m%dT%H%M%SZ')}_daemon_{uuid4().hex[:8]}"
    session_dir = SESSION_ROOT / run_id
    session_dir.mkdir(parents=True, exist_ok=False)
    logger = JsonlLogger(session_dir / "logs.jsonl")

    lock_owned = acquire_lock(run_id, ignore_lock=args.ignore_lock)
    state = load_state()
    state.setdefault("markets", {})
    state.setdefault("sessions", {})
    session_started_at = utc_now()
    state["sessions"][run_id] = {
        "pid": os.getpid(),
        "started_at_utc": session_started_at.isoformat(),
        "session_dir": str(session_dir),
    }
    save_state(state)

    stop = {"requested": False}

    def _request_stop(signum: int, _frame: Any) -> None:
        stop["requested"] = True
        logger.write("stop_requested", signal=signum)

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    logger.write(
        "daemon_start",
        run_id=run_id,
        pid=os.getpid(),
        year=args.year,
        market_poll_s=args.market_poll_s,
        worker_check_s=args.worker_check_s,
        execution_check_s=args.execution_check_s,
        heartbeat_s=args.heartbeat_s,
        skip_market_api=args.skip_market_api,
        worker_dry_run=args.worker_dry_run,
        disable_worker=args.disable_worker,
        execution_dry_run=args.execution_dry_run,
        control_plane_mode=args.control_plane_mode,
        disable_execution_supervisor=args.disable_execution_supervisor,
    )

    series_tickers = tuple(args.series_ticker or DEFAULT_SERIES)
    last_market_poll = 0.0
    last_worker_check = 0.0
    last_execution_check = 0.0
    last_heartbeat = 0.0
    latest_market_snapshot: Path | None = latest_snapshot_path(state)
    last_error: str | None = None
    iteration = 0

    try:
        while not stop["requested"]:
            iteration += 1
            now_mono = time.monotonic()
            market_changed_this_iter = False

            if args.skip_market_api:
                if latest_market_snapshot is None:
                    logger.write("market_poll_skipped", reason="skip_market_api")
                last_market_poll = now_mono
            elif due(now_mono, last_market_poll, args.market_poll_s):
                try:
                    poll_result = poll_kalshi_markets(
                        series_tickers=series_tickers,
                        limit=args.market_discovery_limit,
                        state=state,
                        logger=logger,
                    )
                    latest_market_snapshot = poll_result["path"]
                    market_changed_this_iter = poll_result["changed"]
                    last_error = None
                except Exception as exc:
                    last_error = str(exc)
                    state["consecutive_market_poll_failures"] = int(
                        state.get("consecutive_market_poll_failures", 0)
                    ) + 1
                    state["last_market_poll_result"] = {
                        "success": False,
                        "finished_at_utc": utc_now().isoformat(),
                        "error": last_error,
                    }
                    save_state(state)
                    logger.write("market_poll_failed", error=last_error)
                finally:
                    last_market_poll = now_mono

            worker_due = due(now_mono, last_worker_check, args.worker_check_s) or market_changed_this_iter
            if not args.disable_worker and worker_due:
                try:
                    result = run_refresh_worker(
                        year=args.year,
                        access_level=args.access_level,
                        start_year=args.start_year,
                        market_snapshot=latest_market_snapshot,
                        dry_run=args.worker_dry_run,
                        session_dir=session_dir,
                        logger=logger,
                    )
                    result["success"] = result["returncode"] == 0
                    state["last_worker_result"] = result
                    if result["success"]:
                        state["consecutive_worker_failures"] = 0
                    else:
                        state["consecutive_worker_failures"] = int(
                            state.get("consecutive_worker_failures", 0)
                        ) + 1
                    save_state(state)
                    last_error = None if result["returncode"] == 0 else f"worker rc={result['returncode']}"
                except Exception as exc:
                    last_error = str(exc)
                    state["consecutive_worker_failures"] = int(
                        state.get("consecutive_worker_failures", 0)
                    ) + 1
                    state["last_worker_result"] = {
                        "success": False,
                        "finished_at_utc": utc_now().isoformat(),
                        "error": last_error,
                    }
                    save_state(state)
                    logger.write("worker_failed", error=last_error)
                finally:
                    last_worker_check = now_mono

            execution_due = due(now_mono, last_execution_check, args.execution_check_s) or market_changed_this_iter
            if not args.disable_execution_supervisor and execution_due:
                try:
                    result = run_execution_supervisor(
                        year=args.year,
                        market_snapshot=latest_market_snapshot,
                        route_dry_run=args.execution_dry_run,
                        control_plane_mode=args.control_plane_mode,
                        control_plane_bot_id=args.control_plane_bot_id,
                        session_dir=session_dir,
                        logger=logger,
                    )
                    result["success"] = result["returncode"] == 0
                    state["last_execution_supervisor_result"] = result
                    if result["success"]:
                        state["consecutive_execution_supervisor_failures"] = 0
                    else:
                        state["consecutive_execution_supervisor_failures"] = int(
                            state.get("consecutive_execution_supervisor_failures", 0)
                        ) + 1
                    save_state(state)
                    last_error = None if result["returncode"] == 0 else f"execution supervisor rc={result['returncode']}"
                except Exception as exc:
                    last_error = str(exc)
                    state["consecutive_execution_supervisor_failures"] = int(
                        state.get("consecutive_execution_supervisor_failures", 0)
                    ) + 1
                    state["last_execution_supervisor_result"] = {
                        "success": False,
                        "finished_at_utc": utc_now().isoformat(),
                        "error": last_error,
                    }
                    save_state(state)
                    logger.write("execution_supervisor_failed", error=last_error)
                finally:
                    last_execution_check = now_mono

            if due(now_mono, last_heartbeat, args.heartbeat_s):
                write_heartbeat(
                    run_id=run_id,
                    session_dir=session_dir,
                    iteration=iteration,
                    state=state,
                    last_error=last_error,
                    market_poll_s=args.market_poll_s,
                    worker_check_s=args.worker_check_s,
                    execution_check_s=args.execution_check_s,
                    heartbeat_s=args.heartbeat_s,
                    skip_market_api=args.skip_market_api,
                    disable_worker=args.disable_worker,
                    disable_execution_supervisor=args.disable_execution_supervisor,
                    publish_global=lock_owned,
                    logger=logger,
                )
                last_heartbeat = now_mono

            if args.once:
                break
            if args.max_iterations is not None and iteration >= args.max_iterations:
                break
            time.sleep(1.0)
    finally:
        state = load_state()
        state.setdefault("sessions", {}).setdefault(run_id, {})
        state["sessions"][run_id]["stopped_at_utc"] = utc_now().isoformat()
        save_state(state)
        if lock_owned:
            release_lock(run_id)
        logger.write("daemon_stop", run_id=run_id, iteration=iteration)
        print(f"[live-daemon] run_id={run_id}")
        print(f"[live-daemon] session_dir={session_dir}")


def poll_kalshi_markets(
    *,
    series_tickers: Iterable[str],
    limit: int,
    state: dict[str, Any],
    logger: JsonlLogger,
) -> dict[str, Any]:
    cfg = KalshiAuthConfig.from_env(REPO_ROOT / ".env")
    client = AuthedKalshiClient(cfg)
    markets: list[dict[str, Any]] = []
    pulls: list[dict[str, Any]] = []
    pulled_at = utc_now()
    t0_total = time.monotonic()

    for series in series_tickers:
        logger.write("kalshi_market_pull_start", series_ticker=series, limit=limit)
        t0 = time.monotonic()
        items = client.list_markets(series_ticker=series, limit=limit)
        moneyline_items = list(filter_wnba_moneyline_markets(items))
        open_moneyline_items = list(filter_open_wnba_moneyline_markets(items))
        pull = {
            "series_ticker": series,
            "raw_count": len(items),
            "moneyline_count": len(moneyline_items),
            "open_moneyline_count": len(open_moneyline_items),
            "non_moneyline_filtered_out_count": len(items) - len(moneyline_items),
            "closed_moneyline_filtered_out_count": len(moneyline_items) - len(open_moneyline_items),
            "filtered_out_count": len(items) - len(open_moneyline_items),
            "duration_s": round(time.monotonic() - t0, 3),
        }
        logger.write("kalshi_market_pull_end", **pull)
        pulls.append(pull)
        markets.extend(open_moneyline_items)

    snapshot = {
        "pulled_at_utc": pulled_at.isoformat(),
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
    snapshot_path = MARKET_PULL_ROOT / pulled_at.strftime("%Y%m%d") / f"{pulled_at.strftime('%Y%m%dT%H%M%SZ')}_kalshi_markets.json"
    write_json(snapshot_path, snapshot)

    changes = update_market_state(state, markets, pulled_at=pulled_at)
    state["latest_market_snapshot_path"] = str(snapshot_path)
    state["latest_market_snapshot_at_utc"] = pulled_at.isoformat()
    state["latest_market_count"] = len(markets)
    state["latest_open_market_tickers"] = sorted(
        str(market.get("ticker") or market.get("market_ticker") or "").strip()
        for market in markets
        if str(market.get("ticker") or market.get("market_ticker") or "").strip()
    )
    state["last_market_poll_result"] = {
        "success": True,
        "started_at_utc": pulled_at.isoformat(),
        "finished_at_utc": utc_now().isoformat(),
        "duration_s": round(time.monotonic() - t0_total, 3),
        "snapshot_path": str(snapshot_path),
        "raw_market_count": snapshot["raw_market_count"],
        "moneyline_market_count": snapshot["moneyline_market_count"],
        "open_moneyline_market_count": snapshot["open_moneyline_market_count"],
        "filtered_out_count": snapshot["filtered_out_count"],
    }
    state["consecutive_market_poll_failures"] = 0
    save_state(state)
    write_json(RUN_ROOT / "latest_market_snapshot.json", snapshot)

    changed = bool(changes["new"] or changes["updated"] or changes["missing"])
    if changed:
        state["latest_market_change_at_utc"] = pulled_at.isoformat()
        write_json(
            MARKET_PULL_ROOT / pulled_at.strftime("%Y%m%d") / f"{pulled_at.strftime('%Y%m%dT%H%M%SZ')}_changes.json",
            changes,
        )
        logger.write(
            "market_changes_detected",
            new=len(changes["new"]),
            updated=len(changes["updated"]),
            missing=len(changes["missing"]),
            snapshot_path=str(snapshot_path),
        )
    else:
        logger.write("market_no_changes", count=len(markets), snapshot_path=str(snapshot_path))
    return {"path": snapshot_path, "changed": changed}


def update_market_state(
    state: dict[str, Any],
    markets: list[dict[str, Any]],
    *,
    pulled_at: datetime,
) -> dict[str, list[dict[str, Any]]]:
    tracked = state.setdefault("markets", {})
    seen: set[str] = set()
    changes: dict[str, list[dict[str, Any]]] = {"new": [], "updated": [], "missing": []}
    ts = pulled_at.isoformat()

    for market in markets:
        ticker = str(market.get("ticker") or market.get("market_ticker") or "").strip()
        if not ticker:
            continue
        seen.add(ticker)
        digest = market_digest(market)
        summary = market_summary(market)
        prev = tracked.get(ticker)
        if prev is None:
            tracked[ticker] = {
                **summary,
                "hash": digest,
                "first_seen_utc": ts,
                "last_seen_utc": ts,
                "last_changed_utc": ts,
                "seen_count": 1,
            }
            changes["new"].append({"ticker": ticker, **summary})
        else:
            prev["last_seen_utc"] = ts
            prev["seen_count"] = int(prev.get("seen_count", 0)) + 1
            if prev.get("hash") != digest:
                changes["updated"].append({
                    "ticker": ticker,
                    "old_hash": prev.get("hash"),
                    "new_hash": digest,
                    "old_status": prev.get("status"),
                    "new_status": summary.get("status"),
                    "old_title": prev.get("title"),
                    "new_title": summary.get("title"),
                })
                prev.update(summary)
                prev["hash"] = digest
                prev["last_changed_utc"] = ts

    for ticker, rec in tracked.items():
        if ticker not in seen and rec.get("last_missing_utc") != ts:
            if rec.get("last_seen_utc") != ts:
                rec["last_missing_utc"] = ts
                changes["missing"].append({
                    "ticker": ticker,
                    "last_seen_utc": rec.get("last_seen_utc"),
                    "status": rec.get("status"),
                    "event_ticker": rec.get("event_ticker"),
                })

    return changes


def run_refresh_worker(
    *,
    year: int,
    access_level: str,
    start_year: int,
    market_snapshot: Path | None,
    dry_run: bool,
    session_dir: Path,
    logger: JsonlLogger,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "pipelines/07_live/14_live_data_refresh.py",
        "--year",
        str(year),
        "--mode",
        "due",
        "--access-level",
        access_level,
        "--start-year",
        str(start_year),
    ]
    if market_snapshot is not None and market_snapshot.exists():
        cmd.extend(["--market-snapshot-json", str(market_snapshot)])
    else:
        cmd.append("--skip-market-api")
    if dry_run:
        cmd.append("--dry-run")

    worker_dir = session_dir / "worker_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    log_path = worker_dir / f"{started.strftime('%Y%m%dT%H%M%SZ')}_14_live_data_refresh.log"
    logger.write("worker_start", cmd=cmd, log_path=str(log_path))
    env = os.environ.copy()
    src = str(SRC_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.monotonic() - t0
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_s": round(duration, 3),
        "log_path": str(log_path),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": utc_now().isoformat(),
    }
    logger.write("worker_end", **result)
    return result


def run_execution_supervisor(
    *,
    year: int,
    market_snapshot: Path | None,
    route_dry_run: bool,
    control_plane_mode: str,
    control_plane_bot_id: str,
    session_dir: Path,
    logger: JsonlLogger,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "pipelines/07_live/16_execution_supervisor.py",
        "--year",
        str(year),
    ]
    if market_snapshot is not None and market_snapshot.exists():
        cmd.extend(["--market-snapshot-json", str(market_snapshot)])
    if route_dry_run:
        cmd.append("--route-dry-run")
    cmd.extend(["--control-plane-mode", control_plane_mode])
    cmd.extend(["--control-plane-bot-id", control_plane_bot_id])

    worker_dir = session_dir / "execution_supervisor_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    log_path = worker_dir / f"{started.strftime('%Y%m%dT%H%M%SZ')}_16_execution_supervisor.log"
    logger.write("execution_supervisor_start", cmd=cmd, log_path=str(log_path))
    env = os.environ.copy()
    src = str(SRC_ROOT)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.monotonic() - t0
    log_path.write_text(proc.stdout or "", encoding="utf-8")
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_s": round(duration, 3),
        "log_path": str(log_path),
        "started_at_utc": started.isoformat(),
        "finished_at_utc": utc_now().isoformat(),
    }
    logger.write("execution_supervisor_end", **result)
    return result


def write_heartbeat(
    *,
    run_id: str,
    session_dir: Path,
    iteration: int,
    state: dict[str, Any],
    last_error: str | None,
    market_poll_s: float,
    worker_check_s: float,
    execution_check_s: float,
    heartbeat_s: float,
    skip_market_api: bool,
    disable_worker: bool,
    disable_execution_supervisor: bool,
    publish_global: bool,
    logger: JsonlLogger,
) -> None:
    health = build_health(
        run_id=run_id,
        state=state,
        last_error=last_error,
        market_poll_s=market_poll_s,
        worker_check_s=worker_check_s,
        execution_check_s=execution_check_s,
        heartbeat_s=heartbeat_s,
        skip_market_api=skip_market_api,
        disable_worker=disable_worker,
        disable_execution_supervisor=disable_execution_supervisor,
        owns_lock=publish_global,
    )
    write_health(session_dir, health, publish_global=publish_global)
    heartbeat = {
        "ts_utc": utc_now().isoformat(),
        "run_id": run_id,
        "pid": os.getpid(),
        "iteration": iteration,
        "health_status": health["status"],
        "health_path": str(HEALTH_LATEST_PATH),
        "health_summary": health["summary"],
        "tracked_markets": len(state.get("latest_open_market_tickers", [])),
        "tracked_market_history": len(state.get("markets", {})),
        "latest_market_snapshot_path": state.get("latest_market_snapshot_path"),
        "latest_market_count": state.get("latest_market_count"),
        "last_worker_result": state.get("last_worker_result"),
        "last_execution_supervisor_result": state.get("last_execution_supervisor_result"),
        "last_error": last_error,
    }
    if publish_global:
        write_json(RUN_ROOT / "heartbeat_latest.json", heartbeat)
        with (RUN_ROOT / "heartbeats.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(heartbeat, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
    with (session_dir / "heartbeats.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(heartbeat, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
    logger.write("heartbeat", **heartbeat)


def build_health(
    *,
    run_id: str,
    state: dict[str, Any],
    last_error: str | None,
    market_poll_s: float,
    worker_check_s: float,
    execution_check_s: float,
    heartbeat_s: float,
    skip_market_api: bool,
    disable_worker: bool,
    disable_execution_supervisor: bool,
    owns_lock: bool,
) -> dict[str, Any]:
    now = utc_now()
    checks: dict[str, dict[str, Any]] = {}

    def add_check(name: str, status: str, message: str, **details: Any) -> None:
        checks[name] = {"status": status, "message": message, **details}

    session = state.get("sessions", {}).get(run_id, {})
    started_at = parse_iso_datetime(session.get("started_at_utc"))
    uptime_s = round((now - started_at).total_seconds(), 3) if started_at else None

    add_check("process_running", "ok", "daemon process is executing", pid=os.getpid())
    add_lock_check(checks, run_id, owns_lock=owns_lock)

    latest_snapshot_path = state.get("latest_market_snapshot_path")
    latest_snapshot_at = parse_iso_datetime(state.get("latest_market_snapshot_at_utc"))
    latest_snapshot_age_s = round((now - latest_snapshot_at).total_seconds(), 3) if latest_snapshot_at else None
    if skip_market_api:
        add_check("market_polling", "ok", "market API polling disabled by flag")
    else:
        consecutive_market_failures = int(state.get("consecutive_market_poll_failures", 0))
        last_poll = state.get("last_market_poll_result") or {}
        if consecutive_market_failures >= 3:
            add_check(
                "market_polling",
                "fail",
                "three or more consecutive Kalshi market poll failures",
                consecutive_failures=consecutive_market_failures,
                last_poll=last_poll,
            )
        elif consecutive_market_failures > 0:
            add_check(
                "market_polling",
                "warn",
                "recent Kalshi market poll failure",
                consecutive_failures=consecutive_market_failures,
                last_poll=last_poll,
            )
        else:
            add_check("market_polling", "ok", "latest Kalshi market poll succeeded", last_poll=last_poll)

        if not latest_snapshot_path:
            add_check("market_snapshot_freshness", "fail", "no latest market snapshot path recorded")
        elif not Path(str(latest_snapshot_path)).exists():
            add_check(
                "market_snapshot_freshness",
                "fail",
                "latest market snapshot path does not exist",
                latest_market_snapshot_path=latest_snapshot_path,
            )
        elif latest_snapshot_age_s is None:
            add_check("market_snapshot_freshness", "warn", "latest market snapshot timestamp missing")
        else:
            warn_after = max((2.0 * market_poll_s) + 30.0, 300.0)
            fail_after = max((4.0 * market_poll_s) + 60.0, 900.0)
            if latest_snapshot_age_s > fail_after:
                add_check(
                    "market_snapshot_freshness",
                    "fail",
                    "latest market snapshot is stale",
                    age_s=latest_snapshot_age_s,
                    fail_after_s=fail_after,
                    latest_market_snapshot_path=latest_snapshot_path,
                )
            elif latest_snapshot_age_s > warn_after:
                add_check(
                    "market_snapshot_freshness",
                    "warn",
                    "latest market snapshot is older than expected",
                    age_s=latest_snapshot_age_s,
                    warn_after_s=warn_after,
                    latest_market_snapshot_path=latest_snapshot_path,
                )
            else:
                add_check(
                    "market_snapshot_freshness",
                    "ok",
                    "latest market snapshot is fresh",
                    age_s=latest_snapshot_age_s,
                    latest_market_snapshot_path=latest_snapshot_path,
                )

    latest_market_count = state.get("latest_market_count")
    if latest_market_count is None:
        add_check("active_market_count", "warn", "latest active/open market count missing")
    elif int(latest_market_count) == 0:
        add_check(
            "active_market_count",
            "warn",
            "latest active/open WNBA moneyline market count is zero",
            latest_market_count=latest_market_count,
        )
    else:
        add_check(
            "active_market_count",
            "ok",
            "active/open WNBA moneyline markets present",
            latest_market_count=latest_market_count,
        )

    if disable_worker:
        add_check("worker", "ok", "refresh worker disabled by flag")
    else:
        consecutive_worker_failures = int(state.get("consecutive_worker_failures", 0))
        last_worker = state.get("last_worker_result") or {}
        last_worker_finished_at = parse_iso_datetime(last_worker.get("finished_at_utc"))
        worker_age_s = round((now - last_worker_finished_at).total_seconds(), 3) if last_worker_finished_at else None
        if consecutive_worker_failures >= 3:
            add_check(
                "worker",
                "fail",
                "three or more consecutive refresh worker failures",
                consecutive_failures=consecutive_worker_failures,
                last_worker_result=last_worker,
            )
        elif consecutive_worker_failures > 0 or (last_worker and not last_worker.get("success", True)):
            add_check(
                "worker",
                "warn",
                "latest refresh worker run failed",
                consecutive_failures=consecutive_worker_failures,
                last_worker_result=last_worker,
            )
        elif not last_worker:
            add_check("worker", "warn", "refresh worker has not run yet")
        else:
            add_check(
                "worker",
                "ok",
                "latest refresh worker run succeeded",
                age_s=worker_age_s,
                last_worker_result=last_worker,
            )

        if worker_age_s is not None:
            warn_after = max((2.0 * worker_check_s) + 60.0, 900.0)
            fail_after = max((4.0 * worker_check_s) + 300.0, 1800.0)
            if worker_age_s > fail_after:
                add_check(
                    "worker_freshness",
                    "fail",
                    "refresh worker has not completed recently",
                    age_s=worker_age_s,
                    fail_after_s=fail_after,
                )
            elif worker_age_s > warn_after:
                add_check(
                    "worker_freshness",
                    "warn",
                    "refresh worker completion is older than expected",
                    age_s=worker_age_s,
                    warn_after_s=warn_after,
                )
            else:
                add_check("worker_freshness", "ok", "refresh worker completion is recent", age_s=worker_age_s)

    if disable_execution_supervisor:
        add_check("execution_supervisor", "ok", "execution supervisor disabled by flag")
    else:
        consecutive_exec_failures = int(state.get("consecutive_execution_supervisor_failures", 0))
        last_exec = state.get("last_execution_supervisor_result") or {}
        last_exec_finished_at = parse_iso_datetime(last_exec.get("finished_at_utc"))
        exec_age_s = round((now - last_exec_finished_at).total_seconds(), 3) if last_exec_finished_at else None
        if consecutive_exec_failures >= 3:
            add_check(
                "execution_supervisor",
                "fail",
                "three or more consecutive execution-supervisor failures",
                consecutive_failures=consecutive_exec_failures,
                last_execution_supervisor_result=last_exec,
            )
        elif consecutive_exec_failures > 0 or (last_exec and not last_exec.get("success", True)):
            add_check(
                "execution_supervisor",
                "warn",
                "latest execution-supervisor run failed",
                consecutive_failures=consecutive_exec_failures,
                last_execution_supervisor_result=last_exec,
            )
        elif not last_exec:
            add_check("execution_supervisor", "warn", "execution supervisor has not run yet")
        else:
            add_check(
                "execution_supervisor",
                "ok",
                "latest execution-supervisor run succeeded",
                age_s=exec_age_s,
                last_execution_supervisor_result=last_exec,
            )

        if exec_age_s is not None:
            warn_after = max((2.0 * execution_check_s) + 60.0, 420.0)
            fail_after = max((4.0 * execution_check_s) + 180.0, 900.0)
            if exec_age_s > fail_after:
                add_check(
                    "execution_supervisor_freshness",
                    "fail",
                    "execution supervisor has not completed recently",
                    age_s=exec_age_s,
                    fail_after_s=fail_after,
                )
            elif exec_age_s > warn_after:
                add_check(
                    "execution_supervisor_freshness",
                    "warn",
                    "execution supervisor completion is older than expected",
                    age_s=exec_age_s,
                    warn_after_s=warn_after,
                )
            else:
                add_check(
                    "execution_supervisor_freshness",
                    "ok",
                    "execution supervisor completion is recent",
                    age_s=exec_age_s,
                )

    if last_error:
        add_check("last_error", "warn", "daemon has a current last_error", error=last_error)
    else:
        add_check("last_error", "ok", "no current daemon error")

    failed = sum(1 for check in checks.values() if check["status"] == "fail")
    warned = sum(1 for check in checks.values() if check["status"] == "warn")
    status = "failed" if failed else ("degraded" if warned else "ok")
    return {
        "ts_utc": now.isoformat(),
        "run_id": run_id,
        "pid": os.getpid(),
        "status": status,
        "summary": {
            "failed_checks": failed,
            "warning_checks": warned,
            "ok_checks": sum(1 for check in checks.values() if check["status"] == "ok"),
        },
        "metrics": {
            "uptime_s": uptime_s,
            "heartbeat_interval_s": heartbeat_s,
            "market_poll_interval_s": market_poll_s,
            "worker_check_interval_s": worker_check_s,
            "execution_check_interval_s": execution_check_s,
            "latest_market_count": state.get("latest_market_count"),
            "tracked_markets": len(state.get("latest_open_market_tickers", [])),
            "tracked_market_history": len(state.get("markets", {})),
            "latest_market_snapshot_path": latest_snapshot_path,
            "latest_market_snapshot_age_s": latest_snapshot_age_s,
            "consecutive_market_poll_failures": int(state.get("consecutive_market_poll_failures", 0)),
            "consecutive_worker_failures": int(state.get("consecutive_worker_failures", 0)),
            "consecutive_execution_supervisor_failures": int(
                state.get("consecutive_execution_supervisor_failures", 0)
            ),
        },
        "checks": checks,
    }


def write_health(session_dir: Path, health: dict[str, Any], *, publish_global: bool) -> None:
    if publish_global:
        write_json(HEALTH_LATEST_PATH, health)
        with HEALTH_HISTORY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(health, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
    with (session_dir / "health.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(health, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def add_lock_check(checks: dict[str, dict[str, Any]], run_id: str, *, owns_lock: bool) -> None:
    if not owns_lock:
        checks["lock"] = {
            "status": "ok",
            "message": "lock check skipped for non-publishing diagnostic run",
            "path": str(LOCK_PATH),
        }
        return
    if not LOCK_PATH.exists():
        checks["lock"] = {"status": "fail", "message": "daemon lock file is missing", "path": str(LOCK_PATH)}
        return
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks["lock"] = {"status": "fail", "message": "daemon lock file is not valid JSON", "error": str(exc)}
        return

    lock_pid = int(lock.get("pid", 0) or 0)
    lock_run_id = str(lock.get("run_id") or "")
    if lock_pid != os.getpid() or lock_run_id != run_id:
        checks["lock"] = {
            "status": "fail",
            "message": "daemon lock does not match this process",
            "lock_pid": lock_pid,
            "pid": os.getpid(),
            "lock_run_id": lock_run_id,
            "run_id": run_id,
        }
        return
    checks["lock"] = {"status": "ok", "message": "daemon lock matches this process", "path": str(LOCK_PATH)}


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def market_digest(market: dict[str, Any]) -> str:
    body = json.dumps(market, ensure_ascii=False, sort_keys=True, default=json_default)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def market_summary(market: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": market.get("ticker") or market.get("market_ticker"),
        "event_ticker": market.get("event_ticker"),
        "status": market.get("status"),
        "title": market.get("title"),
        "subtitle": market.get("subtitle"),
        "open_time": market.get("open_time"),
        "close_time": market.get("close_time"),
        "expiration_time": market.get("expiration_time"),
        "yes_bid": market.get("yes_bid"),
        "yes_ask": market.get("yes_ask"),
        "no_bid": market.get("no_bid"),
        "no_ask": market.get("no_ask"),
        "last_price": market.get("last_price"),
        "volume": market.get("volume"),
        "open_interest": market.get("open_interest"),
    }


def due(now_mono: float, last_mono: float, interval_s: float) -> bool:
    return last_mono <= 0.0 or (now_mono - last_mono) >= interval_s


def acquire_lock(run_id: str, *, ignore_lock: bool) -> bool:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if ignore_lock:
        return False
    if LOCK_PATH.exists() and not ignore_lock:
        try:
            existing = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            pid = int(existing.get("pid", 0))
            if pid and pid_is_running(pid):
                raise RuntimeError(
                    f"live daemon already appears to be running: pid={pid}, lock={LOCK_PATH}"
                )
        except json.JSONDecodeError:
            pass
    LOCK_PATH.write_text(
        json.dumps(
            {"run_id": run_id, "pid": os.getpid(), "created_at_utc": utc_now().isoformat()},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return True


def release_lock(run_id: str) -> None:
    if not LOCK_PATH.exists():
        return
    try:
        data = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if data.get("run_id") == run_id:
        LOCK_PATH.unlink(missing_ok=True)


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def latest_snapshot_path(state: dict[str, Any]) -> Path | None:
    raw = state.get("latest_market_snapshot_path")
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


if __name__ == "__main__":
    main()
