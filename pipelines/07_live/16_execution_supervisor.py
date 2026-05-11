"""
16_execution_supervisor.py
==========================

Start and supervise canonical per-game route loops.

The supervisor never replans orders. It only:
  - finds upcoming games with gold feature rows,
  - confirms a moneyline market mapping exists in the latest Kalshi snapshot,
  - honors local operator controls,
  - prevents duplicate route-loop processes for the same game,
  - launches one 05_run_route_entry_loop.py process per eligible game.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

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
    load_team_name_map,
    map_game_to_kalshi_markets,
    parse_datetime,
)
from srwnba.live.canonical.operator_control import resolve_operator_decision  # noqa: E402
from srwnba.live.canonical.process_lock import read_game_lock_status  # noqa: E402
from srwnba.live.control_plane import (  # noqa: E402
    CONTROL_PLANE_MODES,
    ControlPlaneBridge,
    merge_control_decision,
)

RUN_ROOT = REPO_ROOT / "data" / "runs" / "live_execution"
STATE_PATH = RUN_ROOT / "execution_state.json"
LATEST_SUMMARY_PATH = RUN_ROOT / "execution_supervisor_latest.json"
LOG_ROOT = RUN_ROOT / "worker_logs"
GAME_ROOT = REPO_ROOT / "data" / "runs" / "live_games"
GAME_LOCK_ROOT = RUN_ROOT / "game_locks"
DEFAULT_MARKET_SNAPSHOT = REPO_ROOT / "data" / "runs" / "live_daemon" / "latest_market_snapshot.json"
ROUTE_ENTRYPOINT = REPO_ROOT / "pipelines" / "07_live" / "canonical" / "05_run_route_entry_loop.py"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--market-snapshot-json", default=str(DEFAULT_MARKET_SNAPSHOT))
    ap.add_argument("--live-feature-dir", default=str(REPO_ROOT / "data" / "live_features"))
    ap.add_argument("--max-lead-hours", type=float, default=24.0)
    ap.add_argument("--min-lead-hours", type=float, default=0.0)
    ap.add_argument("--route-dry-run", action="store_true",
                    help="Launch route loops in --dry-run mode.")
    ap.add_argument("--plan-only", action="store_true",
                    help="Evaluate launch decisions but do not spawn processes.")
    ap.add_argument("--poll-interval-s", type=float, default=0.0)
    ap.add_argument("--operator-control-path", default=None)
    ap.add_argument("--control-plane-mode", choices=CONTROL_PLANE_MODES, default="local-only")
    ap.add_argument("--control-plane-bot-id", default="wnba-execution-supervisor")
    ap.add_argument("--team-name-map",
                    default=str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"))
    ap.add_argument("--market-discovery-limit", type=int, default=100)
    ap.add_argument("--series-ticker", action="append")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    state = load_state()
    state.setdefault("games", {})
    summary = {
        "ts_utc": now.isoformat(),
        "year": args.year,
        "plan_only": args.plan_only,
        "route_dry_run": args.route_dry_run,
        "control_plane_mode": args.control_plane_mode,
        "market_snapshot_json": args.market_snapshot_json,
        "live_feature_dir": args.live_feature_dir,
        "considered": 0,
        "already_running": [],
        "launched": [],
        "skipped": [],
        "stale_processes": [],
        "errors": [],
    }

    refresh_process_status(state, summary)
    markets = load_markets(Path(args.market_snapshot_json))
    team_name_to_id = load_team_name_map(args.team_name_map)
    schedule = load_latest_schedule(args.year)
    live_feature_dir = Path(args.live_feature_dir)
    control_plane = ControlPlaneBridge(mode=args.control_plane_mode, bot_id=args.control_plane_bot_id)

    for raw in sorted(schedule.values(), key=lambda item: (str(item.get("scheduled") or ""), str(item.get("id") or ""))):
        game = game_ref_from_schedule(raw)
        if game is None:
            continue
        lead_hours = (game.scheduled - now).total_seconds() / 3600.0
        if lead_hours <= args.min_lead_hours or lead_hours > args.max_lead_hours:
            continue
        summary["considered"] += 1

        feature_csv = live_feature_dir / f"{game.game_id}.csv"
        packet_json = GAME_ROOT / game.game_id / "prediction_packet.json"
        if not feature_csv.exists():
            summary["skipped"].append({
                "game_id": game.game_id,
                "reason": "live_feature_row_missing",
                "feature_csv": str(feature_csv),
                "lead_hours": lead_hours,
            })
            continue
        if not packet_json.exists():
            summary["skipped"].append({
                "game_id": game.game_id,
                "reason": "prediction_packet_missing",
                "prediction_packet": str(packet_json),
                "lead_hours": lead_hours,
            })
            continue

        running = state.get("games", {}).get(game.game_id, {})
        if running.get("status") == "running" and pid_is_running(int(running.get("pid") or 0)):
            summary["already_running"].append({
                "game_id": game.game_id,
                "pid": running.get("pid"),
                "lead_hours": lead_hours,
            })
            continue
        lock_status = read_game_lock_status(GAME_LOCK_ROOT / f"{safe_name(game.game_id)}.lock")
        if lock_status.locked and lock_status.running:
            summary["already_running"].append({
                "game_id": game.game_id,
                "pid": lock_status.pid,
                "lead_hours": lead_hours,
                "source": "process_lock",
                "lock_path": str(lock_status.path),
            })
            continue

        decision = resolve_operator_decision(
            game.game_id,
            global_control_path=Path(args.operator_control_path) if args.operator_control_path else None,
        )
        remote_snapshot = control_plane.read_controls(game.game_id)
        effective_decision = merge_control_decision(
            game_id=game.game_id,
            mode=args.control_plane_mode,
            local_decision=decision,
            remote_snapshot=remote_snapshot,
            publish_failure_count=control_plane.consecutive_publish_failures,
        )
        if not effective_decision.trade_allowed:
            summary["skipped"].append({
                "game_id": game.game_id,
                "reason": effective_decision.reason,
                "lead_hours": lead_hours,
                **decision.to_log_payload(),
                **effective_decision.to_log_payload(),
            })
            continue

        try:
            mapping = map_game_to_kalshi_markets(
                game,
                markets,
                require_open=True,
                team_name_to_id=team_name_to_id,
            )
        except Exception as exc:
            summary["skipped"].append({
                "game_id": game.game_id,
                "reason": "mapping_exception",
                "error": repr(exc),
                "lead_hours": lead_hours,
            })
            continue
        if not mapping.confirmed:
            summary["skipped"].append({
                "game_id": game.game_id,
                "reason": "market_mapping_not_confirmed",
                "lead_hours": lead_hours,
                "diagnostics": list(mapping.diagnostics),
            })
            continue

        cmd = route_loop_cmd(
            game=game,
            feature_csv=feature_csv,
            args=args,
        )
        if args.plan_only:
            summary["launched"].append({
                "game_id": game.game_id,
                "planned_only": True,
                "cmd": cmd,
                "lead_hours": lead_hours,
                "event_ticker": mapping.event_ticker,
            })
            continue
        proc_info = launch_route_loop(game_id=game.game_id, cmd=cmd, lead_hours=lead_hours)
        state["games"][game.game_id] = proc_info
        summary["launched"].append({
            "game_id": game.game_id,
            "pid": proc_info["pid"],
            "cmd": cmd,
            "lead_hours": lead_hours,
            "event_ticker": mapping.event_ticker,
            "ledger_dir": proc_info["ledger_dir"],
            "stdout_log_path": proc_info["stdout_log_path"],
        })

    finish(state, summary)


def route_loop_cmd(*, game: SportRadarGameRef, feature_csv: Path, args: argparse.Namespace) -> list[str]:
    tipoff_ts = int(game.scheduled.timestamp())
    completed_games_csv = REPO_ROOT / "data" / "silver" / f"played_games_{args.year}_REGPST.csv"
    ledger_dir = GAME_ROOT / game.game_id
    cmd = [
        sys.executable,
        str(ROUTE_ENTRYPOINT),
        "--game-id", game.game_id,
        "--scheduled", game.scheduled.isoformat(),
        "--home-team-id", game.home_team_id,
        "--away-team-id", game.away_team_id,
        "--home-team-name", game.home_team_name,
        "--away-team-name", game.away_team_name,
        "--tipoff-ts", str(tipoff_ts),
        "--feature-csv", str(feature_csv),
        "--completed-games-csv", str(completed_games_csv),
        "--ledger-dir", str(ledger_dir),
        "--poll-interval-s", str(args.poll_interval_s),
        "--team-name-map", str(args.team_name_map),
        "--market-discovery-limit", str(args.market_discovery_limit),
        "--control-plane-mode", str(args.control_plane_mode),
        "--control-plane-bot-id", str(args.control_plane_bot_id),
    ]
    for series in args.series_ticker or ["KXWNBAGAME", "KXWNBAH"]:
        cmd.extend(["--series-ticker", str(series)])
    if args.operator_control_path:
        cmd.extend(["--operator-control-path", str(args.operator_control_path)])
    if args.route_dry_run:
        cmd.append("--dry-run")
    return cmd


def launch_route_loop(*, game_id: str, cmd: list[str], lead_hours: float) -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    GAME_ROOT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    log_path = LOG_ROOT / f"{started.strftime('%Y%m%dT%H%M%SZ')}_{safe_name(game_id)}.log"
    stdout = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing else str(SRC_ROOT) + os.pathsep + existing
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {
        "game_id": game_id,
        "pid": proc.pid,
        "status": "running",
        "started_at_utc": started.isoformat(),
        "lead_hours_at_start": lead_hours,
        "cmd": cmd,
        "ledger_dir": str(GAME_ROOT / game_id),
        "stdout_log_path": str(log_path),
    }


def refresh_process_status(state: dict[str, Any], summary: dict[str, Any]) -> None:
    for game_id, rec in state.get("games", {}).items():
        if rec.get("status") != "running":
            continue
        pid = int(rec.get("pid") or 0)
        if pid and pid_is_running(pid):
            continue
        rec["status"] = "stopped"
        rec["stopped_seen_at_utc"] = datetime.now(timezone.utc).isoformat()
        summary["stale_processes"].append({"game_id": game_id, "pid": pid})


def load_markets(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, Mapping):
        markets = data.get("markets") or []
        if isinstance(markets, list):
            return filter_open_wnba_moneyline_markets(
                item for item in markets if isinstance(item, Mapping)
            )
    return []


def load_latest_schedule(year: int) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    paths = sorted((REPO_ROOT / "data" / "bronze").glob(f"schedule_{year}_*__*.json"))
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        games = data.get("games") if isinstance(data, Mapping) else None
        if not isinstance(games, list):
            continue
        for game in games:
            if isinstance(game, Mapping) and game.get("id"):
                rows[str(game["id"])] = game
    return rows


def game_ref_from_row(row: pd.Series, schedule: Mapping[str, Mapping[str, Any]]) -> Optional[SportRadarGameRef]:
    game_id = str(row.get("game_id") or "").strip()
    if not game_id:
        return None
    raw_ts = row.get("game_ts") or row.get("scheduled")
    if not raw_ts:
        return None
    scheduled = parse_datetime(str(raw_ts))
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=timezone.utc)
    else:
        scheduled = scheduled.astimezone(timezone.utc)
    sched = schedule.get(game_id) or {}
    home = sched.get("home") if isinstance(sched.get("home"), Mapping) else {}
    away = sched.get("away") if isinstance(sched.get("away"), Mapping) else {}
    home_id = str(row.get("home_team_id") or row.get("home_id") or home.get("id") or "").strip()
    away_id = str(row.get("away_team_id") or row.get("away_id") or away.get("id") or "").strip()
    if not home_id or not away_id:
        return None
    return SportRadarGameRef(
        game_id=game_id,
        scheduled=scheduled,
        home_team_id=home_id,
        away_team_id=away_id,
        home_team_name=str(home.get("name") or row.get("home_team") or ""),
        away_team_name=str(away.get("name") or row.get("away_team") or ""),
    )


def game_ref_from_schedule(raw: Mapping[str, Any]) -> Optional[SportRadarGameRef]:
    game_id = str(raw.get("id") or "").strip()
    scheduled_raw = raw.get("scheduled")
    home = raw.get("home") if isinstance(raw.get("home"), Mapping) else {}
    away = raw.get("away") if isinstance(raw.get("away"), Mapping) else {}
    home_id = str(home.get("id") or "").strip()
    away_id = str(away.get("id") or "").strip()
    if not game_id or not scheduled_raw or not home_id or not away_id:
        return None
    scheduled = parse_datetime(str(scheduled_raw))
    if scheduled.tzinfo is None:
        scheduled = scheduled.replace(tzinfo=timezone.utc)
    else:
        scheduled = scheduled.astimezone(timezone.utc)
    return SportRadarGameRef(
        game_id=game_id,
        scheduled=scheduled,
        home_team_id=home_id,
        away_team_id=away_id,
        home_team_name=team_display_name(home),
        away_team_name=team_display_name(away),
    )


def team_display_name(team: Mapping[str, Any]) -> str:
    parts = [str(team.get("market") or "").strip(), str(team.get("name") or "").strip()]
    return " ".join(part for part in parts if part).strip()


def write_feature_csv(game_id: str, feature_row: pd.DataFrame) -> Path:
    path = GAME_ROOT / game_id / "feature_row.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_row.to_csv(path, index=False)
    return path


def finish(state: dict[str, Any], summary: dict[str, Any]) -> None:
    summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["last_supervisor_summary"] = summary
    write_json(STATE_PATH, state)
    write_json(LATEST_SUMMARY_PATH, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str), encoding="utf-8")


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


if __name__ == "__main__":
    main()
