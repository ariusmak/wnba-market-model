"""Per-game production audit ledger for canonical live execution."""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


LEDGER_SCHEMA_VERSION = "live_game_ledger_v1"


ORDER_EVENTS = {
    "dry_order",
    "order_submitted",
    "order_error",
    "passive_reserved",
    "passive_cancelled",
    "passive_cancel_error",
    "passive_status_error",
    "passive_completed",
}

FILL_EVENTS = {
    "fill",
    "passive_fill_reconciled",
}

ERROR_EVENTS = {
    "order_error",
    "portfolio_sizing_error",
    "poll_error",
    "trade_volume_error",
    "passive_cancel_error",
    "passive_status_error",
}

APPEND_FILES = {
    "market_snapshot": "market_snapshots.jsonl",
    "route_quote": "route_quotes.jsonl",
    "route_capacity": "route_capacities.jsonl",
    "execution_plan": "execution_plans.jsonl",
    "signal_state": "signal_state.jsonl",
    "portfolio_sizing": "portfolio_sizing.jsonl",
    "trade_volume_snapshot": "trade_volume.jsonl",
    "trade_volume_error": "trade_volume.jsonl",
    "cash_coordination": "cash_priority.jsonl",
}

REQUIRED_REVIEW_JSONL_FILES = (
    "events.jsonl",
    "market_snapshots.jsonl",
    "route_quotes.jsonl",
    "portfolio_sizing.jsonl",
    "execution_plans.jsonl",
    "orders.jsonl",
    "fills.jsonl",
    "positions.jsonl",
    "errors.jsonl",
)


@dataclass
class GameLedger:
    """Append-only local audit packet for one game.

    Game-level JSONL files are intentionally append-only so restarts for the
    same game preserve chronology. `summary.json` is the latest materialized
    view for quick checks.
    """

    game_id: str
    root_dir: Path
    raw_log_path: Optional[Path] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: _new_run_id("route"))

    _event_counts: Dict[str, int] = field(default_factory=dict, init=False)
    _route_candidates: List[Dict[str, Any]] = field(default_factory=list, init=False)
    _latest_quotes_by_route: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    _latest_market_snapshot_by_route: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)
    _latest_prediction: Optional[Dict[str, Any]] = field(default=None, init=False)
    _latest_mapping: Optional[Dict[str, Any]] = field(default=None, init=False)
    _latest_expansion_gate: Optional[Dict[str, Any]] = field(default=None, init=False)
    _latest_portfolio_sizing: Optional[Dict[str, Any]] = field(default=None, init=False)
    _latest_signal: Optional[Dict[str, Any]] = field(default=None, init=False)
    _latest_plan: Optional[Dict[str, Any]] = field(default=None, init=False)
    _latest_position: Dict[str, Any] = field(default_factory=dict, init=False)
    _last_event: Optional[Dict[str, Any]] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._touch_required_review_files()
        manifest = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "game_id": self.game_id,
            "run_id": self.run_id,
            "created_ts_ms": int(time.time() * 1000),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "ledger_dir": str(self.root_dir),
            "session_dir": str(self.session_dir),
            "raw_log_path": str(self.raw_log_path) if self.raw_log_path else None,
            "metadata": dict(self.metadata or {}),
        }
        self._write_json(self.session_dir / "manifest.json", manifest)
        self._write_json(self.root_dir / "manifest.json", manifest)
        self._write_summary()

    @property
    def session_dir(self) -> Path:
        return self.root_dir / "sessions" / self.run_id

    def _touch_required_review_files(self) -> None:
        for base_dir in (self.root_dir, self.session_dir):
            for filename in REQUIRED_REVIEW_JSONL_FILES:
                (base_dir / filename).touch(exist_ok=True)

    def write_event(self, event: Mapping[str, Any]) -> Dict[str, Any]:
        payload = _jsonable({
            "schema_version": LEDGER_SCHEMA_VERSION,
            "game_id": self.game_id,
            "run_id": self.run_id,
            **dict(event),
        })
        evt = str(payload.get("evt") or "")
        self._event_counts[evt] = self._event_counts.get(evt, 0) + 1
        self._last_event = {
            "evt": evt,
            "ts_ms": payload.get("ts_ms"),
            "run_id": self.run_id,
        }

        self._append_jsonl(self.root_dir / "events.jsonl", payload)
        self._append_jsonl(self.session_dir / "events.jsonl", payload)
        self._dispatch_event(payload)
        self._write_summary()
        return payload

    def _dispatch_event(self, payload: Dict[str, Any]) -> None:
        evt = str(payload.get("evt") or "")
        filename = APPEND_FILES.get(evt)
        if filename:
            self._append_jsonl(self.root_dir / filename, payload)
            self._append_jsonl(self.session_dir / filename, payload)

        if evt in ORDER_EVENTS:
            self._append_jsonl(self.root_dir / "orders.jsonl", payload)
            self._append_jsonl(self.session_dir / "orders.jsonl", payload)
        if evt in FILL_EVENTS:
            self._append_jsonl(self.root_dir / "fills.jsonl", payload)
            self._append_jsonl(self.session_dir / "fills.jsonl", payload)
        if evt in ERROR_EVENTS:
            self._append_jsonl(self.root_dir / "errors.jsonl", payload)
            self._append_jsonl(self.session_dir / "errors.jsonl", payload)

        if evt == "route_loop_start":
            self._latest_prediction = payload
            packet = self._prediction_packet_payload(payload)
            self._write_json(self.root_dir / "prediction_packet.json", packet)
            self._write_json(self.session_dir / "prediction_packet.json", packet)
        elif evt == "mapping":
            self._latest_mapping = payload
            self._write_mapping()
        elif evt == "route_candidate":
            self._route_candidates.append(payload)
            self._write_mapping()
        elif evt == "expansion_gate":
            self._latest_expansion_gate = payload
            self._write_json(self.root_dir / "expansion_gate.json", payload)
            self._write_json(self.session_dir / "expansion_gate.json", payload)
        elif evt == "portfolio_sizing":
            self._latest_portfolio_sizing = payload
        elif evt == "market_snapshot":
            route_id = str(payload.get("route_id") or payload.get("market_ticker") or "")
            if route_id:
                self._latest_market_snapshot_by_route[route_id] = payload
        elif evt == "route_quote":
            route_id = str(payload.get("route_id") or payload.get("market_ticker") or "")
            if route_id:
                self._latest_quotes_by_route[route_id] = payload
        elif evt == "signal_state":
            self._latest_signal = payload
        elif evt == "execution_plan":
            self._latest_plan = payload
            self._update_position_from_plan(payload)
        elif evt in FILL_EVENTS:
            self._update_position_from_fill(payload)
        elif evt in ORDER_EVENTS:
            self._update_position_from_order(payload)

    def _write_mapping(self) -> None:
        packet = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "game_id": self.game_id,
            "run_id": self.run_id,
            "mapping": self._latest_mapping,
            "route_candidates": self._route_candidates,
        }
        self._write_json(self.root_dir / "market_mapping.json", packet)
        self._write_json(self.session_dir / "market_mapping.json", packet)

    def _prediction_packet_payload(self, route_loop_payload: Mapping[str, Any]) -> Dict[str, Any]:
        path = self.root_dir / "prediction_packet.json"
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        if existing.get("schema_version") == "live_prediction_packet_v1":
            merged = dict(existing)
            merged["route_loop_start"] = dict(route_loop_payload)
            merged["route_loop_prediction"] = {
                key: route_loop_payload.get(key)
                for key in (
                    "p_home",
                    "p_away",
                    "p_selected",
                    "p_elo",
                    "p_raw",
                    "selected_team_id",
                    "selected_side_label",
                    "model_best_round",
                    "model_best_round_source",
                )
            }
            return merged
        return dict(route_loop_payload)

    def _write_summary(self) -> None:
        summary = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "game_id": self.game_id,
            "run_id": self.run_id,
            "updated_ts_ms": int(time.time() * 1000),
            "ledger_dir": str(self.root_dir),
            "session_dir": str(self.session_dir),
            "raw_log_path": str(self.raw_log_path) if self.raw_log_path else None,
            "event_counts_this_run": dict(sorted(self._event_counts.items())),
            "last_event": self._last_event,
            "prediction": self._compact_prediction(),
            "mapping_confirmed": (
                self._latest_mapping.get("confirmed") if self._latest_mapping else None
            ),
            "event_ticker": (
                self._latest_mapping.get("event_ticker") if self._latest_mapping else None
            ),
            "expansion_gate": self._latest_expansion_gate,
            "latest_portfolio_sizing": self._latest_portfolio_sizing,
            "latest_signal": self._latest_signal,
            "latest_execution_plan": self._compact_plan(),
            "latest_quotes_by_route": self._latest_quotes_by_route,
            "latest_market_snapshot_by_route": self._compact_market_snapshots(),
            "position": self._latest_position,
        }
        self._write_json(self.root_dir / "summary.json", summary)
        self._write_json(self.session_dir / "summary.json", summary)

    def _compact_prediction(self) -> Optional[Dict[str, Any]]:
        if not self._latest_prediction:
            return None
        keys = {
            "ts_ms",
            "game_id",
            "event_ticker",
            "p_home",
            "p_away",
            "p_selected",
            "p_elo",
            "p_raw",
            "selected_team_id",
            "selected_side_label",
            "tipoff_ts_s",
            "dry_run",
            "model_best_round",
            "model_best_round_source",
        }
        return {k: v for k, v in self._latest_prediction.items() if k in keys}

    def _compact_plan(self) -> Optional[Dict[str, Any]]:
        if not self._latest_plan:
            return None
        keys = {
            "ts_ms",
            "selected_team_id",
            "p_selected",
            "target_position_dollars",
            "filled_position_dollars",
            "reserved_position_dollars",
            "remaining_position_dollars",
            "allowed_child_dollars",
            "q_max_cents",
            "lead_hours",
            "timing_window",
            "signal_class",
            "binding_cap",
            "cash_limited_mode",
            "cash_priority_rule",
            "cash_priority_rank",
            "cash_priority_score",
            "expected_log_growth_next_child",
            "cash_priority_candidate_child_dollars",
            "q_current_position",
            "q_avg_after_child",
            "skipped_due_to_cash",
            "route_capacity_sum_dollars",
            "global_cumulative_remaining_dollars",
            "decision",
            "reject_reason",
            "orders",
        }
        return {k: v for k, v in self._latest_plan.items() if k in keys}

    def _compact_market_snapshots(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for route_id, snapshot in self._latest_market_snapshot_by_route.items():
            out[route_id] = {
                "ts_ms": snapshot.get("ts_ms"),
                "market_ticker": snapshot.get("market_ticker"),
                "route_type": snapshot.get("route_type"),
                "side": snapshot.get("side"),
            }
        return out

    def _update_position_from_plan(self, payload: Mapping[str, Any]) -> None:
        for key in (
            "target_position_dollars",
            "filled_position_dollars",
            "reserved_position_dollars",
            "remaining_position_dollars",
        ):
            if key in payload:
                self._latest_position[key] = payload.get(key)
        self._append_position_snapshot("execution_plan", payload)

    def _update_position_from_fill(self, payload: Mapping[str, Any]) -> None:
        for key in (
            "cumulative_filled_cost_dollars",
            "filled_contracts_by_route",
            "filled_cost_by_route",
            "total_reserved_cost_dollars",
        ):
            if key in payload:
                self._latest_position[key] = payload.get(key)
        self._append_position_snapshot(str(payload.get("evt") or "fill"), payload)

    def _update_position_from_order(self, payload: Mapping[str, Any]) -> None:
        if "total_reserved_cost_dollars" in payload:
            self._latest_position["reserved_position_dollars"] = payload.get(
                "total_reserved_cost_dollars"
            )
            self._append_position_snapshot(str(payload.get("evt") or "order"), payload)

    def _append_position_snapshot(self, source_evt: str, payload: Mapping[str, Any]) -> None:
        snapshot = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "game_id": self.game_id,
            "run_id": self.run_id,
            "ts_ms": payload.get("ts_ms") or int(time.time() * 1000),
            "source_evt": source_evt,
            "position": dict(self._latest_position),
        }
        self._append_jsonl(self.root_dir / "positions.jsonl", snapshot)
        self._append_jsonl(self.session_dir / "positions.jsonl", snapshot)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(payload), sort_keys=True, allow_nan=False) + "\n")

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)


def _new_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{prefix}_{uuid.uuid4().hex[:8]}"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)
