from __future__ import annotations

import os
import tomllib
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import streamlit as st

try:
    from supabase import Client, create_client
except ImportError:  # pragma: no cover - exercised only before deps install.
    Client = Any  # type: ignore[misc,assignment]
    create_client = None


class ControlPlaneConfigError(RuntimeError):
    """Raised when the dashboard is missing required Supabase config."""


@lru_cache(maxsize=1)
def _local_streamlit_secrets() -> dict[str, Any]:
    path = os.path.join(".streamlit", "secrets.toml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8-sig") as handle:
        return tomllib.loads(handle.read())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secret(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    if value:
        return str(value)
    value = _local_streamlit_secrets().get(name, default)
    return str(value or default)


def dashboard_password_configured() -> bool:
    return bool(_secret("DASHBOARD_PASSWORD"))


def dashboard_password_matches(password: str) -> bool:
    configured = _secret("DASHBOARD_PASSWORD")
    return bool(configured) and password == configured


def supabase_configured() -> bool:
    return bool(_secret("SUPABASE_URL") and _secret("SUPABASE_SERVICE_ROLE_KEY"))


def config_snapshot() -> dict[str, Any]:
    url = _secret("SUPABASE_URL")
    return {
        "supabase_configured": supabase_configured(),
        "supabase_url_host": url.replace("https://", "").split("/")[0] if url else "",
        "dashboard_password_configured": dashboard_password_configured(),
        "twilio_configured": bool(
            _secret("TWILIO_ACCOUNT_SID")
            and _secret("TWILIO_AUTH_TOKEN")
            and _secret("TWILIO_FROM_NUMBER")
            and _secret("ALLOWED_PHONE_NUMBER")
            and _secret("SMS_PIN")
        ),
    }


@st.cache_resource
def get_supabase() -> Client:
    if create_client is None:
        raise ControlPlaneConfigError("Install the supabase package to use connected mode.")
    url = _secret("SUPABASE_URL")
    service_key = _secret("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not service_key:
        raise ControlPlaneConfigError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")
    return create_client(url, service_key)


def _data(response: Any) -> Any:
    return getattr(response, "data", None)


def read_control_state() -> dict[str, Any] | None:
    res = (
        get_supabase()
        .table("control_state")
        .select("*")
        .eq("id", "global")
        .single()
        .execute()
    )
    return _data(res)


def list_live_market_snapshots(limit: int = 25) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("live_market_snapshots")
        .select("*")
        .order("tipoff_ts", desc=False)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def list_route_snapshots(game_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    query = get_supabase().table("route_snapshots").select("*")
    if game_id:
        query = query.eq("game_id", game_id)
    res = query.order("updated_at", desc=True).limit(limit).execute()
    return _data(res) or []


def list_order_events(game_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    query = get_supabase().table("order_events").select("*")
    if game_id:
        query = query.eq("game_id", game_id)
    res = query.order("created_at", desc=True).limit(limit).execute()
    return _data(res) or []


def list_control_commands(limit: int = 100) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("control_commands")
        .select("*")
        .order("received_at", desc=True)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def list_market_controls(limit: int = 100) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("market_controls")
        .select("*")
        .order("updated_at", desc=True)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def list_closed_market_summaries(limit: int = 100) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("closed_market_summaries")
        .select("*")
        .order("game_date", desc=True)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def list_equity_curve(limit: int = 500) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("equity_curve")
        .select("*")
        .order("ts", desc=False)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def read_latest_equity_snapshot() -> dict[str, Any] | None:
    res = (
        get_supabase()
        .table("equity_curve")
        .select("*")
        .order("ts", desc=True)
        .limit(1)
        .execute()
    )
    rows = _data(res) or []
    return rows[0] if rows else None


def list_bot_heartbeat(limit: int = 20) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("bot_heartbeat")
        .select("*")
        .order("last_seen_at", desc=True)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def list_system_alerts(limit: int = 100) -> list[dict[str, Any]]:
    res = (
        get_supabase()
        .table("system_alerts")
        .select("*")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return _data(res) or []


def _insert_command(
    command_type: str,
    scope: str,
    requested_by: str,
    reason: str,
    game_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> str | None:
    res = (
        get_supabase()
        .table("control_commands")
        .insert(
            {
                "command_type": command_type,
                "scope": scope,
                "game_id": game_id,
                "payload": payload or {},
                "requested_by": requested_by,
                "requested_via": "streamlit_dashboard",
                "auth_status": "authorized",
                "status": "received",
                "reason": reason,
            }
        )
        .execute()
    )
    rows = _data(res) or []
    return rows[0].get("command_id") if rows else None


def _mark_command(command_id: str | None, status: str, reason: str | None = None) -> None:
    if not command_id:
        return
    updates: dict[str, Any] = {"status": status}
    if status == "applied":
        updates["applied_at"] = _utc_now_iso()
    if reason is not None:
        updates["reason"] = reason
    (
        get_supabase()
        .table("control_commands")
        .update(updates)
        .eq("command_id", command_id)
        .execute()
    )


GLOBAL_COMMAND_UPDATES: dict[str, dict[str, Any]] = {
    "KILL_BOT": {
        "kill_switch_active": True,
        "trading_enabled": False,
        "allow_new_entries": False,
        "allow_ioc_orders": False,
        "allow_passive_orders": False,
        "allow_burst_mode": False,
        "mode": "killed",
    },
    "LAUNCH_BOT": {
        "kill_switch_active": False,
        "trading_enabled": True,
        "allow_new_entries": True,
        "allow_ioc_orders": True,
        "allow_passive_orders": True,
        "allow_burst_mode": True,
        "mode": "normal",
        "max_market_exposure_pct": 0.15,
    },
    "RESUME_ALL": {
        "kill_switch_active": False,
        "trading_enabled": True,
        "allow_new_entries": True,
        "allow_ioc_orders": True,
        "allow_passive_orders": True,
        "allow_burst_mode": True,
        "mode": "normal",
    },
    "PAUSE_ALL_NEW_ENTRIES": {
        "allow_new_entries": False,
        "mode": "paused",
    },
    "CONSERVATIVE_MODE": {
        "mode": "conservative",
        "max_market_exposure_pct": 0.12,
        "allow_burst_mode": False,
    },
    "NORMAL_RISK_MODE": {
        "mode": "normal",
        "max_market_exposure_pct": 0.15,
        "allow_burst_mode": True,
    },
    "CANCEL_ALL_PASSIVES": {
        "allow_passive_orders": False,
    },
    "ENABLE_PASSIVES": {
        "allow_passive_orders": True,
    },
}


MARKET_COMMAND_UPDATES: dict[str, dict[str, Any]] = {
    "PAUSE_MARKET": {
        "pause_active": True,
        "market_status": "paused",
    },
    "UNPAUSE_MARKET": {
        "pause_active": False,
        "market_status": "normal",
    },
    "CANCEL_ENTRY": {
        "cancel_entry": True,
        "block_new_entries": True,
        "market_status": "cancelled",
    },
    "CANCEL_MARKET_PASSIVES": {
        "cancel_passive_orders": True,
    },
    "BLOCK_GAME": {
        "block_new_entries": True,
        "market_status": "blocked",
    },
    "FORCE_CONSERVATIVE_MARKET": {
        "force_conservative": True,
        "market_status": "force_conservative",
    },
    "CLEAR_MARKET_CONTROLS": {
        "pause_active": False,
        "cancel_entry": False,
        "block_new_entries": False,
        "cancel_passive_orders": False,
        "force_conservative": False,
        "market_status": "normal",
    },
}


def apply_global_command(
    command_type: str,
    reason: str,
    updated_by: str = "arius",
) -> Any:
    if command_type not in GLOBAL_COMMAND_UPDATES:
        raise ValueError(f"Unknown global command: {command_type}")
    updates = {
        **GLOBAL_COMMAND_UPDATES[command_type],
        "updated_at": _utc_now_iso(),
        "updated_by": updated_by,
        "reason": reason,
    }
    command_id = _insert_command(command_type, "global", updated_by, reason, payload=updates)
    try:
        res = (
            get_supabase()
            .table("control_state")
            .update(updates)
            .eq("id", "global")
            .execute()
        )
        _mark_command(command_id, "applied", reason)
        return res
    except Exception as exc:
        _mark_command(command_id, "failed", f"{reason} | {exc}")
        raise


def apply_market_command(
    game_id: str,
    command_type: str,
    reason: str,
    updated_by: str = "arius",
) -> Any:
    if command_type not in MARKET_COMMAND_UPDATES:
        raise ValueError(f"Unknown market command: {command_type}")
    updates = {
        **MARKET_COMMAND_UPDATES[command_type],
        "updated_at": _utc_now_iso(),
        "updated_by": updated_by,
        "reason": reason,
    }
    command_id = _insert_command(
        command_type,
        "market",
        updated_by,
        reason,
        game_id=game_id,
        payload=updates,
    )
    try:
        res = (
            get_supabase()
            .table("market_controls")
            .upsert({"game_id": game_id, **updates}, on_conflict="game_id")
            .execute()
        )
        _mark_command(command_id, "applied", reason)
        return res
    except Exception as exc:
        _mark_command(command_id, "failed", f"{reason} | {exc}")
        raise
