"""Worker-side Supabase control-plane bridge.

The dashboard writes remote operator controls to Supabase, but the local
trading worker remains the only process that may touch Kalshi. This module is
intentionally free of Streamlit imports so canonical execution can read remote
controls and publish status without depending on dashboard runtime code.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from srwnba.live.canonical.execution import ExecutionConfig, PlannedChildOrder
from srwnba.live.canonical.operator_control import OperatorDecision

try:  # pragma: no cover - depends on deployment environment.
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTROL_PLANE_MODES = ("local-only", "supabase-shadow", "supabase-live")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class RemoteControlSnapshot:
    mode: str
    read_ok: bool
    configured: bool
    database_connected: bool
    read_at_utc: str
    control_state: Optional[Mapping[str, Any]] = None
    market_control: Optional[Mapping[str, Any]] = None
    error: str = ""

    @property
    def control_updated_at(self) -> Optional[str]:
        state = self.control_state or {}
        value = state.get("updated_at")
        return str(value) if value else None


@dataclass(frozen=True)
class EffectiveControlDecision:
    game_id: str
    mode: str
    trade_allowed: bool
    reason: str
    risk_mode: str
    shadow_mode_enabled: bool
    allow_ioc_orders: bool
    allow_passive_orders: bool
    allow_burst_mode: bool
    max_market_exposure_pct: Optional[float]
    local_decision: OperatorDecision
    remote_snapshot: RemoteControlSnapshot
    publish_healthy: bool = True
    publish_failure_count: int = 0

    def to_log_payload(self) -> dict[str, Any]:
        state = dict(self.remote_snapshot.control_state or {})
        market = dict(self.remote_snapshot.market_control or {})
        return {
            "effective_trade_allowed": self.trade_allowed,
            "effective_control_reason": self.reason,
            "effective_risk_mode": self.risk_mode,
            "effective_shadow_mode_enabled": self.shadow_mode_enabled,
            "effective_allow_ioc_orders": self.allow_ioc_orders,
            "effective_allow_passive_orders": self.allow_passive_orders,
            "effective_allow_burst_mode": self.allow_burst_mode,
            "effective_max_market_exposure_pct": self.max_market_exposure_pct,
            "control_plane_mode": self.mode,
            "control_plane_read_ok": self.remote_snapshot.read_ok,
            "control_plane_configured": self.remote_snapshot.configured,
            "control_plane_database_connected": self.remote_snapshot.database_connected,
            "control_plane_read_at_utc": self.remote_snapshot.read_at_utc,
            "control_plane_control_updated_at": self.remote_snapshot.control_updated_at,
            "control_plane_error": self.remote_snapshot.error,
            "control_plane_publish_healthy": self.publish_healthy,
            "control_plane_publish_failure_count": self.publish_failure_count,
            "control_plane_global_mode": state.get("mode"),
            "control_plane_kill_switch_active": state.get("kill_switch_active"),
            "control_plane_trading_enabled": state.get("trading_enabled"),
            "control_plane_allow_new_entries": state.get("allow_new_entries"),
            "control_plane_allow_ioc_orders": state.get("allow_ioc_orders"),
            "control_plane_allow_passive_orders": state.get("allow_passive_orders"),
            "control_plane_allow_burst_mode": state.get("allow_burst_mode"),
            "control_plane_shadow_mode_enabled": state.get("shadow_mode_enabled"),
            "control_plane_market_status": market.get("market_status"),
            "control_plane_market_pause_active": market.get("pause_active"),
            "control_plane_market_block_new_entries": market.get("block_new_entries"),
            "control_plane_market_cancel_entry": market.get("cancel_entry"),
            "control_plane_market_cancel_passive_orders": market.get("cancel_passive_orders"),
            "control_plane_market_force_conservative": market.get("force_conservative"),
            "local_operator_trade_allowed": self.local_decision.trade_allowed,
            "local_operator_reason": self.local_decision.reason,
            "local_operator_auto_trade_enabled": self.local_decision.auto_trade_enabled,
            "local_operator_risk_mode": self.local_decision.risk_mode,
            "local_operator_game_decision": self.local_decision.game_decision,
        }

    def block_reason_for_order(self, order: PlannedChildOrder) -> str:
        if not self.trade_allowed:
            return self.reason
        if self.shadow_mode_enabled:
            return "control_plane_shadow_mode"
        if order.order_mode == "passive_probe" and not self.allow_passive_orders:
            return "control_plane_passive_orders_disabled"
        if order.order_mode == "burst_ioc" and not self.allow_burst_mode:
            return "control_plane_burst_disabled"
        if order.order_mode != "passive_probe" and not self.allow_ioc_orders:
            return "control_plane_ioc_orders_disabled"
        return ""


class ControlPlaneBridge:
    def __init__(self, *, mode: str = "local-only", bot_id: str = "wnba-route-worker") -> None:
        if mode not in CONTROL_PLANE_MODES:
            raise ValueError(f"control plane mode must be one of {CONTROL_PLANE_MODES}, got {mode!r}")
        self.mode = mode
        self.bot_id = bot_id
        self._client: Any = None
        self._init_error = ""
        self.consecutive_publish_failures = 0
        self.last_publish_error = ""
        if self.enabled:
            self._client = self._create_client()

    @property
    def enabled(self) -> bool:
        return self.mode != "local-only"

    @property
    def live_mode(self) -> bool:
        return self.mode == "supabase-live"

    @property
    def shadow_mode(self) -> bool:
        return self.mode == "supabase-shadow"

    @property
    def configured(self) -> bool:
        return self._client is not None

    def read_controls(self, game_id: str) -> RemoteControlSnapshot:
        if not self.enabled:
            return RemoteControlSnapshot(
                mode=self.mode,
                read_ok=True,
                configured=False,
                database_connected=False,
                read_at_utc=utc_now_iso(),
            )
        if self._client is None:
            return RemoteControlSnapshot(
                mode=self.mode,
                read_ok=False,
                configured=False,
                database_connected=False,
                read_at_utc=utc_now_iso(),
                error=self._init_error or "control_plane_not_configured",
            )
        try:
            control_state = self._select_one("control_state", eq={"id": "global"})
            if not control_state:
                raise RuntimeError("control_state row id='global' not found")
            market_control = self._select_one("market_controls", eq={"game_id": str(game_id)})
            return RemoteControlSnapshot(
                mode=self.mode,
                read_ok=True,
                configured=True,
                database_connected=True,
                read_at_utc=utc_now_iso(),
                control_state=control_state,
                market_control=market_control,
            )
        except Exception as exc:
            return RemoteControlSnapshot(
                mode=self.mode,
                read_ok=False,
                configured=True,
                database_connected=False,
                read_at_utc=utc_now_iso(),
                error=repr(exc),
            )

    def publish_heartbeat(
        self,
        *,
        decision: EffectiveControlDecision,
        status: str,
        kalshi_connected: bool,
        market_data_connected: bool,
        open_orders_count: int,
        open_positions_count: int,
        last_error: str = "",
    ) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            current_mode = decision.risk_mode
            if not decision.trade_allowed:
                current_mode = decision.reason
            elif decision.shadow_mode_enabled:
                current_mode = "shadow"
            row = {
                "bot_id": self.bot_id,
                "status": status,
                "last_seen_at": utc_now_iso(),
                "last_control_seen_at": decision.remote_snapshot.control_updated_at,
                "current_mode": current_mode,
                "kalshi_connected": bool(kalshi_connected),
                "market_data_connected": bool(market_data_connected),
                "database_connected": bool(decision.remote_snapshot.database_connected),
                "open_orders_count": int(open_orders_count),
                "open_positions_count": int(open_positions_count),
                "last_error": last_error or decision.remote_snapshot.error or self.last_publish_error or None,
            }
            self._safe_upsert("bot_heartbeat", row, on_conflict="bot_id")
            self.consecutive_publish_failures = 0
            self.last_publish_error = ""
        except Exception as exc:
            self.consecutive_publish_failures += 1
            self.last_publish_error = repr(exc)

    def publish_event(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            evt = str(event.get("evt") or "")
            if evt == "execution_plan":
                self._publish_live_market_snapshot(event, context)
            elif evt in {"route_quote", "route_capacity"}:
                self._publish_route_snapshot(event, context)
            elif evt in {
                "dry_order",
                "order_skipped",
                "order_submitted",
                "fill",
                "passive_fill_reconciled",
                "passive_cancelled",
                "passive_cancel_error",
                "passive_status_error",
                "order_error",
            }:
                self._publish_order_event(event, context)
            elif evt == "portfolio_sizing":
                self._publish_equity_snapshot(event, context)
            elif evt in {"poll_error", "position_reconciliation_error", "portfolio_sizing_error"}:
                self._publish_alert(event, context, severity="warning")
            self.consecutive_publish_failures = 0
            self.last_publish_error = ""
        except Exception as exc:
            self.consecutive_publish_failures += 1
            self.last_publish_error = repr(exc)

    def _publish_live_market_snapshot(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        decision = str(event.get("decision") or "")
        reject_reason = str(event.get("reject_reason") or "")
        row = {
            "game_id": str(event.get("game_id") or context.get("game_id")),
            "updated_at": utc_now_iso(),
            "home_team": context.get("home_team_name"),
            "away_team": context.get("away_team_name"),
            "selected_team": context.get("selected_team_name"),
            "opponent_team": context.get("opponent_team_name"),
            "tipoff_ts": context.get("tipoff_ts_utc"),
            "phase": _phase_from_decision(decision, reject_reason),
            "trading_status": reject_reason or decision or None,
            "model_prob": _prob(event.get("p_selected")),
            "model_prob_t20": _prob(event.get("model_prob_t20_selected")),
            "model_prob_latest_pre_t8": _prob(event.get("model_prob_latest_pre_t8_selected")),
            "model_prob_change_t20_to_t8": _num(event.get("model_prob_change_t20_to_t8_selected")),
            "model_prob_changed_t20_to_t8": bool(event.get("model_prob_changed_t20_to_t8") or False),
            "model_prob_last_refresh_at": event.get("model_prob_last_refresh_at_utc"),
            "model_probability_update_count": _int(event.get("model_probability_update_count")),
            "market_prob": _prob(event.get("market_prob")),
            "abs_edge": _num(event.get("abs_edge")),
            "norm_edge": _num(event.get("norm_edge")),
            "q_max_price": _prob(event.get("q_max_price")),
            "q_exec_all_in_price": _prob(event.get("q_exec_all_in_price")),
            "bankroll_for_sizing_dollars": _num(context.get("bankroll")),
            "available_cash_after_buffer_dollars": _num(context.get("available_cash_after_buffer_dollars")),
            "target_position_now_dollars": _num(event.get("target_position_dollars")),
            "filled_position_dollars": _num(event.get("filled_position_dollars")),
            "filled_contracts": _int(event.get("filled_contracts")),
            "reserved_open_order_dollars": _num(event.get("reserved_position_dollars")),
            "remaining_position_dollars": _num(event.get("remaining_position_dollars")),
            "visible_depth_cap_dollars": _num(event.get("visible_depth_cap_dollars")),
            "recent_volume_cap_dollars": _num(event.get("recent_volume_cap_dollars")),
            "cold_start_cap_dollars": _num(event.get("cold_start_cap_dollars")),
            "rolling_liquidity_cap_dollars": _num(event.get("rolling_liquidity_cap_dollars")),
            "cumulative_cap_remaining_dollars": _num(event.get("cumulative_cap_remaining_dollars")),
            "allowed_to_try_now_dollars": _num(event.get("allowed_child_dollars")),
            "next_child_order_dollars": _num(event.get("allowed_child_dollars")),
            "cash_limited_mode": bool(event.get("cash_limited_mode") or False),
            "cash_priority_rule": event.get("cash_priority_rule"),
            "cash_priority_rank": event.get("cash_priority_rank"),
            "cash_priority_score": _num(event.get("cash_priority_score")),
            "expected_log_growth_next_child": _num(event.get("expected_log_growth_next_child")),
            "cash_priority_candidate_child_dollars": _num(event.get("cash_priority_candidate_child_dollars")),
            "q_current_position": _prob(event.get("q_current_position")),
            "q_avg_after_child": _prob(event.get("q_avg_after_child")),
            "skipped_due_to_cash": bool(event.get("skipped_due_to_cash") or False),
            "target_position_binder": event.get("binding_cap"),
            "execution_binder": event.get("execution_binder"),
            "last_action": decision,
            "last_reject_reason": reject_reason or None,
            "number_of_order_attempts": len(event.get("orders") or []),
            "model_snapshot_ts": event.get("model_prob_last_refresh_at_utc") or utc_now_iso(),
            "injury_data_ts": event.get("model_prob_last_refresh_at_utc"),
        }
        self._safe_upsert("live_market_snapshots", row, on_conflict="game_id")

    def _publish_route_snapshot(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        route_type = str(event.get("route_type") or "")
        market_ticker = str(event.get("market_ticker") or "")
        side = str(event.get("side") or "").lower()
        if route_type not in {"BUY_YES_SELECTED", "BUY_NO_OPPONENT"} or side not in {"yes", "no"} or not market_ticker:
            return
        row = {
            "game_id": str(event.get("game_id") or context.get("game_id")),
            "route_name": route_type,
            "market_ticker": market_ticker,
            "outcome_side": side,
            "q_exec_all_in_price": _cents_to_prob(event.get("all_in_avg_price_cents")),
            "best_bid_price": _cents_to_prob(event.get("best_bid_cents")),
            "best_ask_price": _cents_to_prob(event.get("best_ask_cents")),
            "spread_ticks": _int(event.get("spread_ticks")),
            "visible_depth_to_qmax_dollars": _num(event.get("visible_cost_dollars_at_qmax")),
            "recent_qualifying_volume_3h_dollars": _num(event.get("recent_qualifying_volume_dollars")),
            "route_rolling_cap_dollars": _num(event.get("route_capacity_dollars")),
            "route_cumulative_cap_remaining_dollars": _num(event.get("cumulative_cap_remaining_dollars")),
            "chosen": False,
            "route_decision_reason": event.get("reject_reason"),
            "updated_at": utc_now_iso(),
        }
        self._safe_upsert(
            "route_snapshots",
            row,
            on_conflict="game_id,route_name,market_ticker,outcome_side",
        )

    def _publish_order_event(self, event: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        evt = str(event.get("evt") or "")
        event_type = {
            "dry_order": "skipped",
            "order_skipped": "skipped",
            "order_submitted": "submitted",
            "fill": "filled",
            "passive_fill_reconciled": "partial_fill",
            "passive_cancelled": "cancelled",
            "passive_cancel_error": "rejected",
            "passive_status_error": "rejected",
            "order_error": "rejected",
        }.get(evt)
        if not event_type:
            return
        row = {
            "game_id": str(event.get("game_id") or context.get("game_id")),
            "market_ticker": event.get("market_ticker"),
            "route_name": event.get("route_type"),
            "order_id": event.get("order_id") or event.get("client_order_id"),
            "event_type": event_type,
            "order_mode": _order_mode(event),
            "outcome_side": str(event.get("side")).lower() if event.get("side") else None,
            "price": _cents_to_prob(event.get("limit_price_cents")),
            "contracts": _int(event.get("requested_count") or event.get("count") or event.get("filled") or event.get("filled_delta")),
            "cost_dollars": _cost_dollars(event),
            "lead_hours": context.get("lead_hours"),
            "reason": event.get("reason") or event.get("err") or event.get("skip_reason"),
            "raw_payload": dict(event),
            "created_at": utc_now_iso(),
        }
        self._safe_insert("order_events", row)

    def _publish_equity_snapshot(self, event: Mapping[str, Any], _context: Mapping[str, Any]) -> None:
        equity = _num(event.get("kalshi_portfolio_value_dollars")) or _num(event.get("sizing_bankroll_dollars"))
        if equity is None:
            return
        row = {
            "ts": utc_now_iso(),
            "equity_dollars": equity,
            "cash_dollars": _num(event.get("kalshi_cash_dollars")) or _num(event.get("available_cash_dollars")),
            "open_position_value_dollars": None,
            "realized_pnl_dollars": None,
            "drawdown_dollars": None,
        }
        self._safe_upsert("equity_curve", row, on_conflict="ts")

    def _publish_alert(self, event: Mapping[str, Any], context: Mapping[str, Any], *, severity: str) -> None:
        row = {
            "severity": severity,
            "alert_type": str(event.get("evt") or "worker_event"),
            "game_id": str(event.get("game_id") or context.get("game_id")),
            "message": str(event.get("err") or event.get("reason") or event.get("evt") or "worker event"),
            "payload": dict(event),
            "created_at": utc_now_iso(),
        }
        self._safe_insert("system_alerts", row)

    def _select_one(self, table: str, *, eq: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        query = self._client.table(table).select("*")
        for key, value in eq.items():
            query = query.eq(key, value)
        response = query.limit(1).execute()
        data = getattr(response, "data", None) or []
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, Mapping):
            return data
        return None

    def _safe_upsert(self, table: str, row: Mapping[str, Any], *, on_conflict: str) -> None:
        self._client.table(table).upsert(dict(row), on_conflict=on_conflict).execute()

    def _safe_insert(self, table: str, row: Mapping[str, Any]) -> None:
        self._client.table(table).insert(dict(row)).execute()

    def _create_client(self) -> Any:
        if create_client is None:
            self._init_error = "supabase package is not installed"
            return None
        url = _secret("SUPABASE_URL")
        key = _secret("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            self._init_error = "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required"
            return None
        try:
            return create_client(url, key)
        except Exception as exc:
            self._init_error = repr(exc)
            return None


def merge_control_decision(
    *,
    game_id: str,
    mode: str,
    local_decision: OperatorDecision,
    remote_snapshot: RemoteControlSnapshot,
    publish_failure_count: int = 0,
) -> EffectiveControlDecision:
    trade_allowed = bool(local_decision.trade_allowed)
    reason = local_decision.reason
    risk_mode = str(local_decision.risk_mode or "normal").lower()
    shadow = mode == "supabase-shadow"
    allow_ioc = True
    allow_passive = True
    allow_burst = True
    max_exposure: Optional[float] = None

    if mode != "local-only":
        if not remote_snapshot.read_ok:
            trade_allowed = False
            reason = "control_plane_read_failed"
        else:
            state = remote_snapshot.control_state or {}
            market = remote_snapshot.market_control or {}
            if _bool(state.get("shadow_mode_enabled"), default=False) or str(state.get("mode") or "").lower() == "shadow":
                shadow = True
            allow_ioc = _bool(state.get("allow_ioc_orders"), default=True)
            allow_passive = _bool(state.get("allow_passive_orders"), default=True)
            allow_burst = _bool(state.get("allow_burst_mode"), default=True)
            cap = _fraction(state.get("max_market_exposure_pct"))
            if cap is not None:
                max_exposure = cap
            global_mode = str(state.get("mode") or "normal").lower()
            if global_mode == "conservative":
                risk_mode = "conservative"
            if (
                _bool(state.get("kill_switch_active"), default=False)
                or not _bool(state.get("trading_enabled"), default=False)
                or not _bool(state.get("allow_new_entries"), default=True)
                or global_mode in {"paused", "killed"}
            ):
                trade_allowed = False
                reason = f"control_plane_global_{global_mode or 'blocked'}"
            if market:
                market_status = str(market.get("market_status") or "normal").lower()
                if _bool(market.get("force_conservative"), default=False) or market_status == "force_conservative":
                    risk_mode = "conservative"
                if _bool(market.get("cancel_passive_orders"), default=False):
                    allow_passive = False
                if (
                    _bool(market.get("pause_active"), default=False)
                    or _bool(market.get("block_new_entries"), default=False)
                    or _bool(market.get("cancel_entry"), default=False)
                    or market_status in {"paused", "cancelled", "blocked"}
                ):
                    trade_allowed = False
                    if _bool(market.get("cancel_entry"), default=False) or market_status == "cancelled":
                        market_reason = "cancelled"
                    elif _bool(market.get("block_new_entries"), default=False) or market_status == "blocked":
                        market_reason = "blocked"
                    elif _bool(market.get("pause_active"), default=False) or market_status == "paused":
                        market_reason = "paused"
                    else:
                        market_reason = market_status or "blocked"
                    reason = f"control_plane_market_{market_reason}"

    publish_healthy = publish_failure_count < 3
    if mode == "supabase-live" and not publish_healthy:
        trade_allowed = False
        reason = "control_plane_publish_unhealthy"

    return EffectiveControlDecision(
        game_id=str(game_id),
        mode=mode,
        trade_allowed=trade_allowed,
        reason=reason,
        risk_mode=risk_mode,
        shadow_mode_enabled=shadow,
        allow_ioc_orders=allow_ioc,
        allow_passive_orders=allow_passive,
        allow_burst_mode=allow_burst,
        max_market_exposure_pct=max_exposure,
        local_decision=local_decision,
        remote_snapshot=remote_snapshot,
        publish_healthy=publish_healthy,
        publish_failure_count=publish_failure_count,
    )


def apply_effective_config(cfg: ExecutionConfig, decision: EffectiveControlDecision) -> ExecutionConfig:
    max_exposure = cfg.max_market_exposure_pct
    if decision.max_market_exposure_pct is not None:
        max_exposure = min(max_exposure, decision.max_market_exposure_pct)
    return replace(
        cfg,
        max_market_exposure_pct=max_exposure,
        passive_enabled=cfg.passive_enabled and decision.allow_passive_orders,
        burst_enabled=cfg.burst_enabled and decision.allow_burst_mode,
    )


def _secret(name: str) -> str:
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


def _bool(value: Any, *, default: bool) -> bool:
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


def _fraction(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= val <= 1.0:
        return val
    return None


def _num(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prob(value: Any) -> Optional[float]:
    val = _num(value)
    if val is None:
        return None
    return max(0.0, min(1.0, val))


def _cents_to_prob(value: Any) -> Optional[float]:
    val = _num(value)
    if val is None:
        return None
    return _prob(val / 100.0)


def _cost_dollars(event: Mapping[str, Any]) -> Optional[float]:
    if event.get("cost_cents") is not None:
        cents = _num(event.get("cost_cents"))
        return cents / 100.0 if cents is not None else None
    return _num(event.get("cost_delta_dollars") or event.get("max_cost_dollars"))


def _order_mode(event: Mapping[str, Any]) -> Optional[str]:
    mode = str(event.get("order_mode") or "").lower()
    if mode in {"passive", "passive_probe"}:
        return "passive"
    if mode == "burst_ioc":
        return "burst_ioc"
    if mode == "shadow" or event.get("evt") == "order_skipped":
        return "shadow"
    if event.get("evt") == "passive_cancelled":
        return "cancel"
    if mode:
        return "ioc"
    return None


def _phase_from_decision(decision: str, reject_reason: str) -> str:
    if reject_reason.startswith("control_plane") or reject_reason.startswith("operator_"):
        return "BLOCKED"
    if decision in {"normal_ioc", "burst_ioc", "passive_probe"}:
        return "ACTIVE"
    if decision == "monitor_only":
        return "MONITOR"
    if decision == "no_trade":
        return "PRE_QUALIFIED_ONLY"
    return "BLOCKED" if "blocked" in decision else "MONITOR"
