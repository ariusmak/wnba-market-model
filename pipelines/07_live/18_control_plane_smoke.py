"""
Supabase control-plane smoke test.

This script is intentionally separate from Streamlit. It verifies the remote
control-plane tables the dashboard and worker share, and can optionally insert
a no-op audit command plus wait for daemon command acknowledgment.

Default mode is read-only:

    python pipelines/07_live/18_control_plane_smoke.py

Full rehearsal once the daemon is running in supabase-shadow:

    python pipelines/07_live/18_control_plane_smoke.py ^
        --write-noop-command ^
        --require-daemon-ack ^
        --ack-timeout-s 180
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

try:  # pragma: no cover - depends on deployment environment.
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOT_ID = "wnba-live-daemon"

EXPECTED_TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "control_state": (
        "id",
        "trading_enabled",
        "kill_switch_active",
        "allow_new_entries",
        "allow_ioc_orders",
        "allow_passive_orders",
        "allow_burst_mode",
        "mode",
        "max_market_exposure_pct",
        "shadow_mode_enabled",
        "updated_at",
        "updated_by",
        "reason",
    ),
    "market_controls": (
        "game_id",
        "market_status",
        "pause_active",
        "cancel_entry",
        "block_new_entries",
        "cancel_passive_orders",
        "force_conservative",
        "updated_at",
        "updated_by",
        "reason",
    ),
    "control_commands": (
        "command_id",
        "command_type",
        "scope",
        "game_id",
        "payload",
        "requested_by",
        "requested_via",
        "auth_status",
        "received_at",
        "applied_at",
        "status",
        "reason",
    ),
    "live_market_snapshots": (
        "game_id",
        "updated_at",
        "home_team",
        "away_team",
        "selected_team",
        "opponent_team",
        "tipoff_ts",
        "phase",
        "trading_status",
        "model_prob",
        "model_prob_t20",
        "model_prob_latest_pre_t8",
        "model_prob_change_t20_to_t8",
        "model_prob_changed_t20_to_t8",
        "market_prob",
        "abs_edge",
        "norm_edge",
        "q_exec_all_in_price",
        "filled_position_dollars",
        "filled_contracts",
        "reserved_open_order_dollars",
        "remaining_position_dollars",
        "visible_depth_cap_dollars",
        "recent_volume_cap_dollars",
        "cold_start_cap_dollars",
        "rolling_liquidity_cap_dollars",
        "cumulative_cap_remaining_dollars",
        "cash_limited_mode",
        "cash_priority_rank",
        "last_action",
        "last_reject_reason",
        "market_data_ts",
        "model_snapshot_ts",
        "injury_data_ts",
        "orderbook_ts",
    ),
    "route_snapshots": (
        "id",
        "game_id",
        "route_name",
        "market_ticker",
        "outcome_side",
        "q_exec_all_in_price",
        "best_bid_price",
        "best_ask_price",
        "visible_depth_to_qmax_dollars",
        "route_rolling_cap_dollars",
        "route_cumulative_cap_remaining_dollars",
        "chosen",
        "updated_at",
    ),
    "order_events": (
        "event_id",
        "game_id",
        "market_ticker",
        "route_name",
        "order_id",
        "event_type",
        "order_mode",
        "outcome_side",
        "price",
        "contracts",
        "cost_dollars",
        "lead_hours",
        "reason",
        "raw_payload",
        "created_at",
    ),
    "bot_heartbeat": (
        "bot_id",
        "status",
        "last_seen_at",
        "last_control_seen_at",
        "current_mode",
        "kalshi_connected",
        "market_data_connected",
        "database_connected",
        "open_orders_count",
        "open_positions_count",
        "last_error",
    ),
    "equity_curve": (
        "ts",
        "equity_dollars",
        "cash_dollars",
        "open_position_value_dollars",
        "realized_pnl_dollars",
        "drawdown_dollars",
    ),
    "closed_market_summaries": (
        "game_id",
        "game_date",
        "home_team",
        "away_team",
        "selected_team",
        "status",
        "did_enter",
        "pnl_dollars",
        "settled_at",
    ),
    "system_alerts": (
        "alert_id",
        "severity",
        "alert_type",
        "game_id",
        "message",
        "payload",
        "created_at",
        "acknowledged",
    ),
}


@dataclass
class SmokeReport:
    ok: bool = True
    checks: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def pass_check(self, name: str, message: str, **details: Any) -> None:
        self.checks.append({"status": "pass", "name": name, "message": message, **details})

    def warn(self, name: str, message: str, **details: Any) -> None:
        self.warnings.append(message)
        self.checks.append({"status": "warn", "name": name, "message": message, **details})

    def fail(self, name: str, message: str, **details: Any) -> None:
        self.ok = False
        self.failures.append(message)
        self.checks.append({"status": "fail", "name": name, "message": message, **details})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot-id", default=DEFAULT_BOT_ID)
    ap.add_argument("--heartbeat-max-age-s", type=float, default=180.0)
    ap.add_argument("--ack-timeout-s", type=float, default=0.0)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--require-daemon-ack", action="store_true")
    ap.add_argument("--write-noop-command", action="store_true")
    ap.add_argument("--requested-by", default="control_plane_smoke")
    ap.add_argument("--json", action="store_true", help="Print only JSON output.")
    args = ap.parse_args()

    report = SmokeReport()
    try:
        client = make_client()
        report.pass_check("supabase_client", "Supabase client created")
    except Exception as exc:
        report.fail("supabase_client", str(exc))
        print_report(report, json_only=args.json)
        return 1

    check_required_columns(client, report)
    control_state = read_control_state(client, report)
    if control_state:
        report.details["control_state_updated_at"] = control_state.get("updated_at")
        check_control_state_values(control_state, report)

    if args.write_noop_command and control_state:
        write_noop_command(
            client,
            report,
            requested_by=args.requested_by,
            control_state=control_state,
        )

    if args.require_daemon_ack:
        wait_for_daemon_ack(
            client,
            report,
            bot_id=args.bot_id,
            control_state=control_state or {},
            heartbeat_max_age_s=args.heartbeat_max_age_s,
            timeout_s=max(0.0, args.ack_timeout_s),
            poll_s=max(1.0, args.poll_s),
        )
    else:
        check_daemon_heartbeat(
            client,
            report,
            bot_id=args.bot_id,
            control_state=control_state or {},
            heartbeat_max_age_s=args.heartbeat_max_age_s,
            required=False,
        )

    print_report(report, json_only=args.json)
    return 0 if report.ok else 1


def check_required_columns(client: Any, report: SmokeReport) -> None:
    for table, columns in EXPECTED_TABLE_COLUMNS.items():
        try:
            client.table(table).select(",".join(columns)).limit(1).execute()
        except Exception as exc:
            report.fail(
                f"schema:{table}",
                f"{table} missing or missing required column(s): {exc!r}",
                columns=list(columns),
            )
            continue
        report.pass_check(f"schema:{table}", "required columns are selectable", columns=list(columns))


def read_control_state(client: Any, report: SmokeReport) -> Mapping[str, Any] | None:
    try:
        response = client.table("control_state").select("*").eq("id", "global").limit(1).execute()
        rows = getattr(response, "data", None) or []
    except Exception as exc:
        report.fail("control_state", f"failed to read global control_state: {exc!r}")
        return None
    if not rows:
        report.fail("control_state", "control_state row id='global' is missing")
        return None
    row = rows[0]
    report.pass_check(
        "control_state",
        "global control_state row is present",
        mode=row.get("mode"),
        trading_enabled=row.get("trading_enabled"),
        kill_switch_active=row.get("kill_switch_active"),
        updated_at=row.get("updated_at"),
    )
    return row


def check_control_state_values(control_state: Mapping[str, Any], report: SmokeReport) -> None:
    mode = str(control_state.get("mode") or "").lower()
    if mode not in {"normal", "conservative", "paused", "killed", "shadow"}:
        report.fail("control_state_mode", f"unexpected control_state.mode={mode!r}")
    else:
        report.pass_check("control_state_mode", "control_state.mode is valid", mode=mode)

    exposure = _float_or_none(control_state.get("max_market_exposure_pct"))
    if exposure is None or not (0.0 <= exposure <= 1.0):
        report.fail(
            "control_state_exposure",
            f"max_market_exposure_pct must be in [0, 1], got {control_state.get('max_market_exposure_pct')!r}",
        )
    else:
        report.pass_check("control_state_exposure", "max_market_exposure_pct is valid", value=exposure)


def write_noop_command(
    client: Any,
    report: SmokeReport,
    *,
    requested_by: str,
    control_state: Mapping[str, Any],
) -> None:
    run_id = f"smoke_{utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    payload = {
        "smoke_run_id": run_id,
        "noop": True,
        "expected_control_state_mutation": False,
        "control_state_updated_at_before": control_state.get("updated_at"),
    }
    row = {
        "command_type": "SMOKE_TEST_NOOP",
        "scope": "global",
        "game_id": None,
        "payload": payload,
        "requested_by": requested_by,
        "requested_via": "control_plane_smoke",
        "auth_status": "authorized",
        "status": "applied",
        "reason": "No-op control-plane smoke test; does not mutate control_state.",
        "applied_at": utc_now().isoformat(),
    }
    try:
        response = client.table("control_commands").insert(row).execute()
        rows = getattr(response, "data", None) or []
    except Exception as exc:
        report.fail("noop_command", f"failed to insert no-op control command: {exc!r}")
        return
    command_id = rows[0].get("command_id") if rows else None
    report.pass_check(
        "noop_command",
        "inserted no-op command audit row",
        command_id=command_id,
        smoke_run_id=run_id,
    )
    report.details["noop_command_id"] = command_id
    report.details["smoke_run_id"] = run_id


def wait_for_daemon_ack(
    client: Any,
    report: SmokeReport,
    *,
    bot_id: str,
    control_state: Mapping[str, Any],
    heartbeat_max_age_s: float,
    timeout_s: float,
    poll_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        before_fail_count = len(report.failures)
        before_warn_count = len(report.warnings)
        check_daemon_heartbeat(
            client,
            report,
            bot_id=bot_id,
            control_state=control_state,
            heartbeat_max_age_s=heartbeat_max_age_s,
            required=True,
        )
        latest = report.checks[-1] if report.checks else {}
        if latest.get("name") == "daemon_ack" and latest.get("status") == "pass":
            return
        if timeout_s <= 0.0 or time.monotonic() >= deadline:
            return
        del report.checks[-1:]
        del report.failures[before_fail_count:]
        del report.warnings[before_warn_count:]
        report.ok = not report.failures
        time.sleep(poll_s)


def check_daemon_heartbeat(
    client: Any,
    report: SmokeReport,
    *,
    bot_id: str,
    control_state: Mapping[str, Any],
    heartbeat_max_age_s: float,
    required: bool,
) -> None:
    try:
        response = client.table("bot_heartbeat").select("*").eq("bot_id", bot_id).limit(1).execute()
        rows = getattr(response, "data", None) or []
    except Exception as exc:
        report.fail("daemon_ack", f"failed to read bot_heartbeat for {bot_id}: {exc!r}")
        return
    if not rows:
        msg = f"bot_heartbeat row for {bot_id!r} is missing"
        if required:
            report.fail("daemon_ack", msg)
        else:
            report.warn("daemon_ack", msg)
        return
    heartbeat = rows[0]
    last_seen = parse_ts(heartbeat.get("last_seen_at"))
    last_control_seen = parse_ts(heartbeat.get("last_control_seen_at"))
    control_updated = parse_ts(control_state.get("updated_at"))
    now = utc_now()
    age_s = (now - last_seen).total_seconds() if last_seen else None
    details = {
        "bot_id": bot_id,
        "status": heartbeat.get("status"),
        "current_mode": heartbeat.get("current_mode"),
        "last_seen_at": heartbeat.get("last_seen_at"),
        "last_control_seen_at": heartbeat.get("last_control_seen_at"),
        "control_updated_at": control_state.get("updated_at"),
        "age_s": age_s,
    }
    if last_seen is None:
        report.fail("daemon_ack", "daemon heartbeat last_seen_at is missing or invalid", **details)
        return
    if age_s is not None and age_s > heartbeat_max_age_s:
        msg = f"daemon heartbeat is stale: age_s={age_s:.1f}, max={heartbeat_max_age_s:.1f}"
        if required:
            report.fail("daemon_ack", msg, **details)
        else:
            report.warn("daemon_ack", msg, **details)
        return
    if control_updated is not None and (last_control_seen is None or last_control_seen < control_updated):
        msg = "daemon has not acknowledged current control_state.updated_at"
        if required:
            report.fail("daemon_ack", msg, **details)
        else:
            report.warn("daemon_ack", msg, **details)
        return
    report.pass_check("daemon_ack", "daemon heartbeat is fresh and controls are acknowledged", **details)


def make_client() -> Any:
    if create_client is None:
        raise RuntimeError("supabase package is not installed")
    url = secret("SUPABASE_URL")
    key = secret("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    return create_client(url, key)


def secret(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    secrets_path = REPO_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            data = tomllib.loads(secrets_path.read_text(encoding="utf-8-sig"))
            return str(data.get(name) or "")
        except Exception:
            return ""
    return ""


def print_report(report: SmokeReport, *, json_only: bool) -> None:
    payload = {
        "ok": report.ok,
        "failures": report.failures,
        "warnings": report.warnings,
        "details": report.details,
        "checks": report.checks,
    }
    if json_only:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    status = "OK" if report.ok else "FAILED"
    print(f"[control-plane-smoke] {status}")
    for check in report.checks:
        print(f"  - {check['status'].upper()}: {check['name']} - {check['message']}")
    if report.failures:
        print("[control-plane-smoke] failures:")
        for failure in report.failures:
            print(f"  - {failure}")
    if report.warnings:
        print("[control-plane-smoke] warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")


def parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
