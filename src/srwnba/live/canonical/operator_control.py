"""Local operator controls consumed by live execution.

The default posture is intentionally permissive: if no control file exists,
games are eligible for normal trading. Operators must explicitly disable
auto-trading globally or abort a game to block execution.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
CONTROL_ROOT = REPO_ROOT / "data" / "runs" / "live_control"
GLOBAL_CONTROL_PATH = CONTROL_ROOT / "operator_control.json"
GAME_OVERRIDE_ROOT = CONTROL_ROOT / "game_overrides"

VALID_RISK_MODES = {"normal", "conservative", "kill"}
VALID_GAME_DECISIONS = {"default", "proceed", "abort"}


@dataclass(frozen=True)
class OperatorDecision:
    game_id: str
    trade_allowed: bool
    reason: str
    auto_trade_enabled: bool
    risk_mode: str
    game_decision: str
    global_control_path: str
    game_override_path: str
    note: str = ""

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "operator_trade_allowed": self.trade_allowed,
            "operator_reason": self.reason,
            "operator_auto_trade_enabled": self.auto_trade_enabled,
            "operator_risk_mode": self.risk_mode,
            "operator_game_decision": self.game_decision,
            "operator_global_control_path": self.global_control_path,
            "operator_game_override_path": self.game_override_path,
            "operator_note": self.note,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_global_control() -> dict[str, Any]:
    return {
        "schema_version": "operator_control_v1",
        "auto_trade_enabled": True,
        "risk_mode": "normal",
        "reason": "Default: trade all eligible games unless explicitly aborted.",
        "updated_at_utc": None,
        "updated_by": None,
    }


def load_global_control(path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(path) if path else GLOBAL_CONTROL_PATH
    data = dict(default_global_control())
    if path.exists():
        raw = _read_json(path)
        if isinstance(raw, Mapping):
            data.update(raw)
    data["auto_trade_enabled"] = _boolish(data.get("auto_trade_enabled"), default=True)
    risk_mode = str(data.get("risk_mode") or "normal").strip().lower()
    data["risk_mode"] = risk_mode if risk_mode in VALID_RISK_MODES else "normal"
    return data


def save_global_control(
    *,
    auto_trade_enabled: bool,
    risk_mode: str,
    reason: str,
    updated_by: str = "webapp",
    path: Optional[Path] = None,
) -> dict[str, Any]:
    path = Path(path) if path else GLOBAL_CONTROL_PATH
    risk_mode = str(risk_mode or "normal").strip().lower()
    if risk_mode not in VALID_RISK_MODES:
        raise ValueError(f"risk_mode must be one of {sorted(VALID_RISK_MODES)}, got {risk_mode!r}")
    payload = {
        **default_global_control(),
        "auto_trade_enabled": bool(auto_trade_enabled),
        "risk_mode": risk_mode,
        "reason": reason,
        "updated_at_utc": utc_now_iso(),
        "updated_by": updated_by,
        "pid": os.getpid(),
    }
    _write_json_atomic(path, payload)
    return payload


def game_override_path(game_id: str, root: Optional[Path] = None) -> Path:
    safe_game_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(game_id))
    base = Path(root) if root else GAME_OVERRIDE_ROOT
    return base / f"{safe_game_id}.json"


def load_game_override(game_id: str, path: Optional[Path] = None) -> dict[str, Any]:
    path = Path(path) if path else game_override_path(game_id)
    data: dict[str, Any] = {
        "schema_version": "operator_game_override_v1",
        "game_id": str(game_id),
        "decision": "default",
        "reason": "",
        "updated_at_utc": None,
        "updated_by": None,
    }
    if path.exists():
        raw = _read_json(path)
        if isinstance(raw, Mapping):
            data.update(raw)
    decision = str(data.get("decision") or "default").strip().lower()
    data["decision"] = decision if decision in VALID_GAME_DECISIONS else "default"
    return data


def save_game_override(
    *,
    game_id: str,
    decision: str,
    reason: str,
    p_home_override: Optional[float] = None,
    updated_by: str = "webapp",
    path: Optional[Path] = None,
) -> dict[str, Any]:
    decision = str(decision or "default").strip().lower()
    if decision not in VALID_GAME_DECISIONS:
        raise ValueError(f"decision must be one of {sorted(VALID_GAME_DECISIONS)}, got {decision!r}")
    if p_home_override is not None:
        raise ValueError("Probability overrides are not supported by the locked production spec")
    payload = {
        "schema_version": "operator_game_override_v1",
        "game_id": str(game_id),
        "decision": decision,
        "reason": reason,
        "updated_at_utc": utc_now_iso(),
        "updated_by": updated_by,
        "pid": os.getpid(),
    }
    path = Path(path) if path else game_override_path(game_id)
    _write_json_atomic(path, payload)
    return payload


def resolve_operator_decision(
    game_id: str,
    *,
    global_control_path: Optional[Path] = None,
    game_override_path_: Optional[Path] = None,
) -> OperatorDecision:
    control_path = Path(global_control_path) if global_control_path else GLOBAL_CONTROL_PATH
    override_path = Path(game_override_path_) if game_override_path_ else game_override_path(game_id)
    global_control = load_global_control(control_path)
    override = load_game_override(game_id, override_path)

    auto_enabled = bool(global_control.get("auto_trade_enabled"))
    risk_mode = str(global_control.get("risk_mode") or "normal").lower()
    game_decision = str(override.get("decision") or "default").lower()
    note = str(override.get("reason") or global_control.get("reason") or "")

    if risk_mode == "kill":
        allowed = False
        reason = "operator_risk_mode_kill"
    elif not auto_enabled:
        allowed = False
        reason = "operator_auto_trade_disabled"
    elif game_decision == "abort":
        allowed = False
        reason = "operator_game_aborted"
    else:
        allowed = True
        reason = "operator_allowed"

    return OperatorDecision(
        game_id=str(game_id),
        trade_allowed=allowed,
        reason=reason,
        auto_trade_enabled=auto_enabled,
        risk_mode=risk_mode,
        game_decision=game_decision,
        global_control_path=str(control_path),
        game_override_path=str(override_path),
        note=note,
    )


def _boolish(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    return default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)
