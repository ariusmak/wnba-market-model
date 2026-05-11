"""
Canonical-route entry loop for WNBA Kalshi execution.

This loop trades one canonical exposure, "selected team wins", through the
confirmed equivalent routes:

    - BUY YES on the selected team's market
    - BUY NO on the opponent team's market

It wires FinalModel probability, Kalshi route mapping, the locked v1.2
planner, passive-order reservation/cancellation, IOC submission, expansion
team gating, operational brakes, and JSONL audit logging.
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pandas as pd

from ...util.final_model import FinalModel
from ..common import estimate_kalshi_fee_dollars
from .execution import (
    ExecutionConfig,
    ExecutionPlan,
    PlannedChildOrder,
    RouteQuote,
    evaluate_route_quote,
    submit_planned_child_order,
)
from .cash_coordinator import clear_cash_candidate, coordinate_cash_for_plan
from .expansion_gate import evaluate_expansion_team_gate
from .game_ledger import GameLedger
from .kalshi_mapping import (
    KalshiGameMapping,
    RouteCandidate,
    SportRadarGameRef,
    build_equivalent_routes,
    filter_open_wnba_moneyline_markets,
    load_team_name_map,
    map_game_to_kalshi_markets,
)
from .operator_control import OperatorDecision, resolve_operator_decision
from .portfolio import PortfolioSizingSnapshot, resolve_portfolio_sizing
from .reconciliation import reconcile_exchange_routes, recover_open_route_orders
from .v1_2 import (
    BrakeState,
    PlannerRuntimeState,
    SignalMemory,
    VolumeSnapshot,
    plan_v1_2_orders,
    timing_state,
    update_signal_memory,
)

log = logging.getLogger("srwnba.live.canonical.route_entry_loop")

DEFAULT_SERIES_TICKERS = ("KXWNBAGAME", "KXWNBAH")
CASH_COORDINATOR_DIR = Path(__file__).resolve().parents[4] / "data" / "runs" / "live_execution" / "cash_priority"


@dataclass(frozen=True)
class RouteEntryContext:
    game: SportRadarGameRef
    tipoff_ts_s: int
    feature_row: pd.DataFrame
    team_name_map_path: Path
    series_tickers: Sequence[str] = DEFAULT_SERIES_TICKERS
    market_discovery_limit: int = 100
    completed_games_by_team: Mapping[str, int] = field(default_factory=dict)


@dataclass
class CanonicalExposureState:
    selected_team_id: str
    selected_team_name: str = ""
    opponent_team_id: str = ""
    opponent_team_name: str = ""
    filled_cost_dollars: float = 0.0
    filled_contracts_by_route: Dict[str, int] = field(default_factory=dict)
    filled_cost_by_route: Dict[str, float] = field(default_factory=dict)
    reserved_cost_dollars: float = 0.0
    total_fee_dollars: float = 0.0
    orders: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ActivePassiveOrder:
    order_id: str
    client_order_id: str
    route_id: str
    market_ticker: str
    side: str
    count: int
    limit_price_cents: int
    reserved_cost_dollars: float
    created_ts_s: float
    reprices: int = 0
    filled_count_seen: int = 0
    fill_cost_seen_dollars: float = 0.0


class RouteEntryLoop:
    def __init__(
        self,
        *,
        predictor: FinalModel,
        client: Any,
        ctx: RouteEntryContext,
        cfg: Optional[ExecutionConfig] = None,
        log_path: Optional[Path] = None,
        poll_interval_s: float = 0.0,
        dry_run: bool = False,
        available_cash_dollars: Optional[float] = None,
        markets: Optional[Sequence[Mapping[str, Any]]] = None,
        ledger: Optional[GameLedger] = None,
        follow_kalshi_wealth: bool = False,
        initial_portfolio_sizing: Optional[PortfolioSizingSnapshot] = None,
        sizing_bankroll_override_dollars: Optional[float] = None,
        available_cash_override_dollars: Optional[float] = None,
        portfolio_refresh_interval_s: float = 300.0,
        operator_global_control_path: Optional[Path] = None,
        operator_game_override_path: Optional[Path] = None,
        position_reconcile_interval_s: float = 300.0,
        startup_reconciliation: bool = True,
    ) -> None:
        self.predictor = predictor
        self.client = client
        self.ctx = ctx
        self.cfg = cfg or ExecutionConfig()
        self.log_path = Path(log_path) if log_path else None
        self.ledger = ledger
        self.poll_interval_s = float(poll_interval_s)
        self.dry_run = bool(dry_run)
        self.available_cash_dollars = available_cash_dollars
        self.follow_kalshi_wealth = bool(follow_kalshi_wealth)
        self.sizing_bankroll_override_dollars = sizing_bankroll_override_dollars
        self.available_cash_override_dollars = available_cash_override_dollars
        self.portfolio_refresh_interval_s = max(0.0, float(portfolio_refresh_interval_s))
        self.portfolio_sizing: Optional[PortfolioSizingSnapshot] = None
        self.last_portfolio_refresh_s: Optional[float] = None
        self.operator_global_control_path = operator_global_control_path
        self.operator_game_override_path = operator_game_override_path
        self.operator_decision: OperatorDecision = resolve_operator_decision(
            self.ctx.game.game_id,
            global_control_path=self.operator_global_control_path,
            game_override_path_=self.operator_game_override_path,
        )
        self.position_reconcile_interval_s = max(0.0, float(position_reconcile_interval_s))
        self.startup_reconciliation = bool(startup_reconciliation)
        self.last_position_reconcile_s: Optional[float] = None
        self.position_mismatch_dollars = 0.0
        self.execution_block_reason: Optional[str] = None
        self.client_order_prefix = f"wnba-route-{self.ctx.game.game_id}-"
        self.signal = SignalMemory()
        self.active_passive: Optional[ActivePassiveOrder] = None
        self.order_reject_timestamps_s: List[float] = []
        self.api_error_timestamps_s: List[float] = []
        self.last_ioc_order_ts_s: Optional[float] = None
        self.last_burst_order_ts_s: Optional[float] = None
        self.burst_orders_last_5min: List[tuple[float, float]] = []
        self.previous_combined_visible_cost_dollars = 0.0
        self.combined_visible_cost_after_last_order_dollars = 0.0
        self.passive_cooldown_until_s: Optional[float] = None
        self.passive_reprices_current_episode = 0
        self.expansion_gate = evaluate_expansion_team_gate(
            home_team_id=ctx.game.home_team_id,
            away_team_id=ctx.game.away_team_id,
            completed_games_by_team=ctx.completed_games_by_team,
        )
        if initial_portfolio_sizing is not None:
            self._apply_portfolio_sizing(initial_portfolio_sizing)
            self.last_portfolio_refresh_s = time.time()
        elif self.follow_kalshi_wealth or self.sizing_bankroll_override_dollars is not None:
            self._refresh_portfolio_sizing(force=True, emit=False)

        self.prediction_result = predictor.predict(ctx.feature_row)
        self.p_home_model = float(self.prediction_result["p_home"][0])
        self.p_home = self.p_home_model
        self.p_away = 1.0 - self.p_home
        if self.p_home >= self.p_away:
            self.selected_team_id = ctx.game.home_team_id
            self.p_selected = self.p_home
            self.selected_side_label = "home"
        else:
            self.selected_team_id = ctx.game.away_team_id
            self.p_selected = self.p_away
            self.selected_side_label = "away"

        self.markets = (
            list(filter_open_wnba_moneyline_markets(markets))
            if markets is not None
            else self._discover_markets()
        )
        team_name_to_id = load_team_name_map(str(ctx.team_name_map_path))
        self.mapping = map_game_to_kalshi_markets(
            ctx.game,
            self.markets,
            require_open=True,
            team_name_to_id=team_name_to_id,
        )
        if not self.mapping.confirmed:
            raise RuntimeError(f"Kalshi mapping not confirmed: {self.mapping.diagnostics}")

        self.routes = build_equivalent_routes(self.mapping, self.selected_team_id)
        self.state = CanonicalExposureState(
            selected_team_id=self.selected_team_id,
            selected_team_name=self.routes[0].selected_team_name,
            opponent_team_id=self.routes[0].opponent_team_id,
            opponent_team_name=self.routes[0].opponent_team_name,
        )

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.startup_reconciliation and not self.dry_run:
            self._recover_open_orders_at_startup()
            self._reconcile_exchange_position(source="startup", seed=True)

    def run(self, wall_clock_now_s=None) -> None:
        now_fn = wall_clock_now_s or (lambda: time.time())
        self._write_event({
            "evt": "route_loop_start",
            "game_id": self.ctx.game.game_id,
            "event_ticker": self.mapping.event_ticker,
            "p_home": self.p_home,
            "p_away": self.p_away,
            "p_selected": self.p_selected,
            "p_home_model": self.p_home_model,
            "p_raw": _prediction_value(self.prediction_result, "p_raw"),
            "p_elo": _prediction_value(self.prediction_result, "p_elo"),
            "selected_team_id": self.selected_team_id,
            "selected_side_label": self.selected_side_label,
            "tipoff_ts_s": self.ctx.tipoff_ts_s,
            "dry_run": self.dry_run,
            "model_best_round": getattr(self.predictor, "best_round", None),
            "model_best_round_source": getattr(self.predictor, "best_round_source", None),
            **self.operator_decision.to_log_payload(),
            "portfolio_sizing": (
                self.portfolio_sizing.to_log_payload() if self.portfolio_sizing else None
            ),
            "feature_row": _feature_row_payload(self.ctx.feature_row),
            **self.expansion_gate.to_log_payload(),
        })
        if self.portfolio_sizing is not None:
            self._write_portfolio_sizing_event(source="startup")
        self._log_mapping()
        self._log_expansion_gate()

        while True:
            now = now_fn()
            if now >= self.ctx.tipoff_ts_s:
                clear_cash_candidate(game_id=self.ctx.game.game_id, coordinator_dir=CASH_COORDINATOR_DIR)
                self._write_event({"evt": "tipoff_stop", "ts_s": now})
                return
            try:
                self._poll_once(now_s=now)
            except Exception as exc:
                log.exception("[route_entry_loop] poll failed: %s", exc)
                self.api_error_timestamps_s.append(time.time())
                self._write_event({"evt": "poll_error", "err": repr(exc)})
            time.sleep(self._sleep_s(now))

    def _discover_markets(self) -> List[Mapping[str, Any]]:
        markets: List[Mapping[str, Any]] = []
        for series in self.ctx.series_tickers:
            markets.extend(
                self.client.list_markets(
                    series_ticker=series,
                    limit=self.ctx.market_discovery_limit,
                )
            )
        return list(filter_open_wnba_moneyline_markets(markets))

    def _sleep_s(self, now_s: float) -> float:
        if self.poll_interval_s < 0:
            return 0.0
        timing = timing_state(now_s, float(self.ctx.tipoff_ts_s), self.cfg)
        if self.poll_interval_s == 0:
            return timing.poll_interval_s
        return min(self.poll_interval_s, timing.poll_interval_s)

    def _poll_once(self, now_s: Optional[float] = None) -> None:
        now = float(now_s if now_s is not None else time.time())
        self._refresh_portfolio_sizing(now_s=now)
        self._refresh_operator_decision()
        self._reconcile_exchange_position(now_s=now, source="runtime", seed=False)
        quotes: List[RouteQuote] = []
        for route in self.routes:
            payload = self.client.get_orderbook(route.market_ticker)
            self._write_market_snapshot(route, payload, now_s=now)
            quote = evaluate_route_quote(
                route,
                payload,
                p_selected=self.p_selected,
                cfg=self.cfg,
            )
            quotes.append(quote)
            self._write_route_quote(quote)

        self.signal = update_signal_memory(
            self.signal,
            quotes,
            now_s=now,
            tipoff_ts_s=float(self.ctx.tipoff_ts_s),
        )
        self._write_signal_state(now, quotes)
        self._refresh_active_passive(now)
        self._expire_or_cancel_passive_if_needed(now, quotes)

        if not self.operator_decision.trade_allowed:
            clear_cash_candidate(game_id=self.ctx.game.game_id, coordinator_dir=CASH_COORDINATOR_DIR)
            self._write_blocked_plan(quotes, self.operator_decision.reason)
            return
        if self.execution_block_reason:
            clear_cash_candidate(game_id=self.ctx.game.game_id, coordinator_dir=CASH_COORDINATOR_DIR)
            self._write_blocked_plan(quotes, self.execution_block_reason)
            return
        if not self.expansion_gate.allowed:
            clear_cash_candidate(game_id=self.ctx.game.game_id, coordinator_dir=CASH_COORDINATOR_DIR)
            self._write_expansion_blocked_plan(quotes)
            return

        runtime = PlannerRuntimeState(
            now_s=now,
            tipoff_ts_s=float(self.ctx.tipoff_ts_s),
            signal=self.signal,
            filled_position_dollars=self.state.filled_cost_dollars,
            reserved_position_dollars=self.state.reserved_cost_dollars,
            filled_cost_by_route=self.state.filled_cost_by_route,
            current_position_q_all_in=self._current_position_q_all_in(),
            available_cash_dollars=self.available_cash_dollars,
            volume=self._volume_snapshot(now, quotes),
            last_ioc_order_ts_s=self.last_ioc_order_ts_s,
            last_burst_order_ts_s=self.last_burst_order_ts_s,
            previous_combined_visible_cost_dollars=self.previous_combined_visible_cost_dollars,
            combined_visible_cost_after_last_order_dollars=self.combined_visible_cost_after_last_order_dollars,
            burst_orders_last_5min=tuple(self.burst_orders_last_5min),
            passive_order_live=self.active_passive is not None,
            passive_cooldown_until_s=self.passive_cooldown_until_s,
            brake_state=BrakeState(
                order_reject_timestamps_s=tuple(self.order_reject_timestamps_s),
                api_error_timestamps_s=tuple(self.api_error_timestamps_s),
                position_mismatch_dollars=self.position_mismatch_dollars,
                conservative_mode=self.operator_decision.risk_mode == "conservative",
            ),
        )
        plan = plan_v1_2_orders(
            selected_team_id=self.selected_team_id,
            p_selected=self.p_selected,
            route_quotes=quotes,
            cfg=self.cfg,
            runtime=runtime,
        )
        if plan.orders and not self.dry_run:
            coordination = coordinate_cash_for_plan(
                game_id=self.ctx.game.game_id,
                plan=plan,
                cfg=self.cfg,
                coordinator_dir=CASH_COORDINATOR_DIR,
                available_cash_after_buffer=self._available_cash_after_buffer(),
                filled_position_dollars=self.state.filled_cost_dollars,
                reserved_position_dollars=self.state.reserved_cost_dollars,
                current_position_q=self._current_position_q_all_in(),
            )
            plan = coordination.plan
            self._write_event({"evt": "cash_coordination", **dict(coordination.payload)})
        elif not plan.orders:
            clear_cash_candidate(
                game_id=self.ctx.game.game_id,
                coordinator_dir=CASH_COORDINATOR_DIR,
            )
        for quote in plan.route_quotes:
            self._write_route_capacity(quote)
        self._write_execution_plan(plan)
        self.previous_combined_visible_cost_dollars = sum(
            quote.visible_cost_dollars_at_qmax for quote in plan.route_quotes if quote.eligible
        )
        if not plan.orders:
            return
        if self.active_passive is not None and plan.decision != "passive_probe":
            self._cancel_active_passive("ioc_priority_cancel", now_s=now)
            if self.active_passive is not None:
                return
        for order in plan.orders:
            self._execute(order, now_s=now)

    def _available_cash_after_buffer(self) -> float:
        if self.available_cash_dollars is None:
            return self.cfg.bankroll
        return max(0.0, self.available_cash_dollars - self.cfg.cash_buffer_pct * self.cfg.bankroll)

    def _execute(self, order: PlannedChildOrder, *, now_s: Optional[float] = None) -> None:
        now = float(now_s if now_s is not None else time.time())
        client_order_id = (
            f"wnba-route-{self.ctx.game.game_id}-{order.route_type.lower()}-"
            f"{uuid.uuid4().hex[:10]}"
        )
        if self.dry_run:
            self._write_event({
                "evt": "dry_order",
                "order_mode": order.order_mode,
                "client_order_id": client_order_id,
                "route_id": order.route_id,
                "market_ticker": order.market_ticker,
                "route_type": order.route_type,
                "action": order.action,
                "side": order.side,
                "count": order.count,
                "limit_price_cents": order.limit_price_cents,
                "max_cost_dollars": order.max_cost_dollars,
                "q_max_cents": order.q_max_cents,
                "time_in_force": order.time_in_force,
                "post_only": order.post_only,
            })
            if order.order_mode == "passive_probe":
                self._reserve_passive(
                    order=order,
                    client_order_id=client_order_id,
                    order_id="dry-passive",
                    now_s=now,
                )
            return

        try:
            resp = submit_planned_child_order(
                self.client,
                order,
                client_order_id=client_order_id,
            )
        except Exception as exc:
            log.exception("[route_entry_loop] order submit failed: %s", exc)
            self.order_reject_timestamps_s.append(time.time())
            self._write_event({
                "evt": "order_error",
                "order_mode": order.order_mode,
                "client_order_id": client_order_id,
                "route_id": order.route_id,
                "market_ticker": order.market_ticker,
                "route_type": order.route_type,
                "side": order.side,
                "count": order.count,
                "limit_price_cents": order.limit_price_cents,
                "err": repr(exc),
            })
            return

        order_resp = (resp or {}).get("order") or {}
        filled = _order_fill_count(order_resp)
        fill_cost_dollars = _order_fill_cost_dollars(
            order_resp,
            fallback_count=filled,
            fallback_price_cents=order.limit_price_cents,
        )
        fill_cost_cents = int(round(fill_cost_dollars * 100.0))
        if order.order_mode == "passive_probe":
            remaining = _intish(
                order_resp.get("remaining_count")
                or order_resp.get("remaining_count_fp")
                or max(order.count - filled, 0)
                or 0
            )
            if remaining > 0:
                self._reserve_passive(
                    order=order,
                    client_order_id=client_order_id,
                    order_id=str(order_resp.get("order_id") or ""),
                    now_s=now,
                    remaining_count=remaining,
                    filled_count_seen=filled,
                    fill_cost_seen_dollars=fill_cost_dollars,
                )
        else:
            self.last_ioc_order_ts_s = now
            self.combined_visible_cost_after_last_order_dollars = self.previous_combined_visible_cost_dollars
            if order.order_mode == "burst_ioc":
                self.last_burst_order_ts_s = self.last_ioc_order_ts_s
                self.burst_orders_last_5min.append((self.last_ioc_order_ts_s, order.max_cost_dollars))
        self.state.filled_cost_dollars += fill_cost_dollars
        self.state.filled_contracts_by_route[order.route_id] = (
            self.state.filled_contracts_by_route.get(order.route_id, 0) + filled
        )
        self.state.filled_cost_by_route[order.route_id] = (
            self.state.filled_cost_by_route.get(order.route_id, 0.0) + fill_cost_dollars
        )
        self.state.orders.append({
            "client_order_id": client_order_id,
            "order_id": order_resp.get("order_id"),
            "status": order_resp.get("status"),
            "route_id": order.route_id,
            "market_ticker": order.market_ticker,
            "route_type": order.route_type,
            "side": order.side,
            "requested_count": order.count,
            "limit_price_cents": order.limit_price_cents,
            "response": order_resp,
        })
        self._write_event({
            "evt": "order_submitted",
            "order_mode": order.order_mode,
            "client_order_id": client_order_id,
            "order_id": order_resp.get("order_id"),
            "route_id": order.route_id,
            "market_ticker": order.market_ticker,
            "route_type": order.route_type,
            "side": order.side,
            "requested_count": order.count,
            "limit_price_cents": order.limit_price_cents,
            "status": order_resp.get("status"),
            "post_only": order.post_only,
            "time_in_force": order.time_in_force,
        })
        self._write_event({
            "evt": "fill",
            "order_mode": order.order_mode,
            "order_id": order_resp.get("order_id"),
            "route_id": order.route_id,
            "market_ticker": order.market_ticker,
            "route_type": order.route_type,
            "side": order.side,
            "filled": filled,
            "cost_cents": fill_cost_cents,
            "cumulative_filled_cost_dollars": self.state.filled_cost_dollars,
            "filled_contracts_by_route": dict(self.state.filled_contracts_by_route),
            "filled_cost_by_route": dict(self.state.filled_cost_by_route),
        })

    def _log_mapping(self) -> None:
        self._write_event({
            "evt": "mapping",
            "game_id": self.ctx.game.game_id,
            "event_ticker": self.mapping.event_ticker,
            "confirmed": self.mapping.confirmed,
            "side_mapping_confirmed": self.mapping.side_mapping_confirmed,
            "complement_market_confirmed": self.mapping.complement_market_confirmed,
            "settlement_mapping_confirmed": self.mapping.settlement_mapping_confirmed,
            "candidate_count": self.mapping.candidate_count,
            "home_market": _market_payload(self.mapping.home_market),
            "away_market": _market_payload(self.mapping.away_market),
            "diagnostics": list(self.mapping.diagnostics),
        })
        for route in self.routes:
            self._write_event({
                "evt": "route_candidate",
                "route_id": route.route_id,
                "canonical_exposure": route.canonical_exposure,
                "selected_team_id": route.selected_team_id,
                "opponent_team_id": route.opponent_team_id,
                "market_ticker": route.market_ticker,
                "event_ticker": route.event_ticker,
                "route_type": route.route_type,
                "action": route.action,
                "side": route.side,
                "market_yes_team_id": route.market_yes_team_id,
                "market_yes_team_name": route.market_yes_team_name,
                "side_mapping_confirmed": route.side_mapping_confirmed,
                "complement_market_confirmed": route.complement_market_confirmed,
                "settlement_mapping_confirmed": route.settlement_mapping_confirmed,
            })

    def _log_expansion_gate(self) -> None:
        self._write_event({
            "evt": "expansion_gate",
            "game_id": self.ctx.game.game_id,
            "home_team_id": self.ctx.game.home_team_id,
            "away_team_id": self.ctx.game.away_team_id,
            **self.expansion_gate.to_log_payload(),
        })

    def _write_signal_state(self, now_s: float, quotes: Sequence[RouteQuote]) -> None:
        eligible = [quote for quote in quotes if quote.eligible]
        best = min(eligible, key=lambda quote: quote.all_in_avg_price_cents) if eligible else None
        self._write_event({
            "evt": "signal_state",
            "game_id": self.ctx.game.game_id,
            "now_s": now_s,
            "currently_qualified": bool(eligible),
            "best_qualified_route_id": best.route.route_id if best else None,
            "best_qualified_all_in_avg_price_cents": best.all_in_avg_price_cents if best else None,
            "best_qualified_edge": best.edge if best else None,
            **self.signal.to_log_payload(),
        })

    def _write_route_quote(self, quote: RouteQuote) -> None:
        self._write_event({
            "evt": "route_quote",
            "route_id": quote.route.route_id,
            "market_ticker": quote.route.market_ticker,
            "route_type": quote.route.route_type,
            "side": quote.route.side,
            "q_max_cents": quote.q_max_cents,
            "best_bid_cents": quote.best_bid_cents,
            "best_bid_size": quote.best_bid_size,
            "best_ask_cents": quote.best_ask_cents,
            "best_ask_size": quote.best_ask_size,
            "spread_ticks": quote.spread_ticks,
            "fillable_contracts_at_qmax": quote.fillable_contracts_at_qmax,
            "raw_avg_price_cents": quote.raw_avg_price_cents,
            "all_in_avg_price_cents": quote.all_in_avg_price_cents,
            "limit_price_cents": quote.limit_price_cents,
            "visible_cost_dollars_at_qmax": quote.visible_cost_dollars_at_qmax,
            "visible_depth_cap_dollars": quote.visible_depth_cap_dollars,
            "recent_qualifying_volume_dollars": quote.recent_qualifying_volume_dollars,
            "recent_qualifying_volume_cap_dollars": quote.recent_qualifying_volume_cap_dollars,
            "cumulative_qualifying_volume_dollars": quote.cumulative_qualifying_volume_dollars,
            "cumulative_cap_remaining_dollars": quote.cumulative_cap_remaining_dollars,
            "cold_start_cap_dollars": quote.cold_start_cap_dollars,
            "route_capacity_dollars": quote.route_capacity_dollars,
            "edge": quote.edge,
            "norm_edge": quote.norm_edge,
            "eligible": quote.eligible,
            "reject_reason": quote.reject_reason,
        })

    def _write_market_snapshot(
        self,
        route: RouteCandidate,
        payload: Mapping[str, Any],
        *,
        now_s: float,
    ) -> None:
        self._write_event({
            "evt": "market_snapshot",
            "game_id": self.ctx.game.game_id,
            "now_s": now_s,
            "route_id": route.route_id,
            "market_ticker": route.market_ticker,
            "event_ticker": route.event_ticker,
            "route_type": route.route_type,
            "action": route.action,
            "side": route.side,
            "canonical_exposure": route.canonical_exposure,
            "selected_team_id": route.selected_team_id,
            "opponent_team_id": route.opponent_team_id,
            "raw_orderbook_payload": payload,
        })

    def _write_route_capacity(self, quote: RouteQuote) -> None:
        self._write_event({
            "evt": "route_capacity",
            "route_id": quote.route.route_id,
            "market_ticker": quote.route.market_ticker,
            "route_type": quote.route.route_type,
            "q_max_cents": quote.q_max_cents,
            "eligible": quote.eligible,
            "visible_cost_dollars_at_qmax": quote.visible_cost_dollars_at_qmax,
            "visible_depth_cap_dollars": quote.visible_depth_cap_dollars,
            "recent_qualifying_volume_dollars": quote.recent_qualifying_volume_dollars,
            "recent_qualifying_volume_cap_dollars": quote.recent_qualifying_volume_cap_dollars,
            "cumulative_qualifying_volume_dollars": quote.cumulative_qualifying_volume_dollars,
            "cumulative_cap_remaining_dollars": quote.cumulative_cap_remaining_dollars,
            "cold_start_cap_dollars": quote.cold_start_cap_dollars,
            "route_capacity_dollars": quote.route_capacity_dollars,
        })

    def _write_execution_plan(self, plan: ExecutionPlan) -> None:
        self._write_event({
            "evt": "execution_plan",
            "selected_team_id": plan.selected_team_id,
            "p_selected": plan.p_selected,
            "operator_trade_allowed": self.operator_decision.trade_allowed,
            "operator_reason": self.operator_decision.reason,
            "operator_risk_mode": self.operator_decision.risk_mode,
            "position_mismatch_dollars": self.position_mismatch_dollars,
            "target_position_dollars": plan.target_position_dollars,
            "filled_position_dollars": self.state.filled_cost_dollars,
            "reserved_position_dollars": self.state.reserved_cost_dollars,
            "remaining_position_dollars": plan.remaining_position_dollars,
            "allowed_child_dollars": plan.allowed_child_dollars,
            "q_max_cents": plan.q_max_cents,
            "lead_hours": plan.lead_hours,
            "timing_window": plan.timing_window,
            "signal_class": plan.signal_class,
            "binding_cap": plan.binding_cap,
            "cash_limited_mode": plan.cash_limited_mode,
            "cash_priority_rule": plan.cash_priority_rule,
            "cash_priority_rank": plan.cash_priority_rank,
            "cash_priority_score": plan.cash_priority_score,
            "expected_log_growth_next_child": plan.expected_log_growth_next_child,
            "cash_priority_candidate_child_dollars": plan.cash_priority_candidate_child_dollars,
            "q_current_position": plan.q_current_position,
            "q_avg_after_child": plan.q_avg_after_child,
            "skipped_due_to_cash": plan.skipped_due_to_cash,
            "route_capacity_sum_dollars": plan.route_capacity_sum_dollars,
            "global_cumulative_remaining_dollars": plan.global_cumulative_remaining_dollars,
            "decision": plan.decision,
            "reject_reason": plan.reject_reason,
            "orders": [
                {
                    "route_id": order.route_id,
                    "market_ticker": order.market_ticker,
                    "route_type": order.route_type,
                    "action": order.action,
                    "side": order.side,
                    "order_mode": order.order_mode,
                    "count": order.count,
                    "limit_price_cents": order.limit_price_cents,
                    "max_cost_dollars": order.max_cost_dollars,
                    "expected_all_in_avg_price_cents": order.expected_all_in_avg_price_cents,
                    "time_in_force": order.time_in_force,
                    "post_only": order.post_only,
                }
                for order in plan.orders
            ],
        })

    def _write_blocked_plan(self, quotes: Sequence[RouteQuote], reason: str) -> None:
        q_max_cents = quotes[0].q_max_cents if quotes else 0
        self._write_event({
            "evt": "execution_plan",
            "selected_team_id": self.selected_team_id,
            "p_selected": self.p_selected,
            "operator_trade_allowed": self.operator_decision.trade_allowed,
            "operator_reason": self.operator_decision.reason,
            "operator_risk_mode": self.operator_decision.risk_mode,
            "position_mismatch_dollars": self.position_mismatch_dollars,
            "target_position_dollars": 0.0,
            "filled_position_dollars": self.state.filled_cost_dollars,
            "reserved_position_dollars": self.state.reserved_cost_dollars,
            "remaining_position_dollars": 0.0,
            "allowed_child_dollars": 0.0,
            "q_max_cents": q_max_cents,
            "cash_limited_mode": False,
            "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
            "cash_priority_rank": None,
            "cash_priority_score": None,
            "expected_log_growth_next_child": None,
            "cash_priority_candidate_child_dollars": 0.0,
            "q_current_position": self._current_position_q_all_in(),
            "q_avg_after_child": None,
            "skipped_due_to_cash": False,
            "decision": "blocked_operator" if reason.startswith("operator_") else "blocked_brake",
            "reject_reason": reason,
            "orders": [],
        })

    def _reserve_passive(
        self,
        *,
        order: PlannedChildOrder,
        client_order_id: str,
        order_id: str,
        now_s: float,
        remaining_count: Optional[int] = None,
        filled_count_seen: int = 0,
        fill_cost_seen_dollars: float = 0.0,
    ) -> None:
        count = remaining_count if remaining_count is not None else order.count
        reserved = count * order.limit_price_cents / 100.0
        self.active_passive = ActivePassiveOrder(
            order_id=order_id,
            client_order_id=client_order_id,
            route_id=order.route_id,
            market_ticker=order.market_ticker,
            side=order.side,
            count=count,
            limit_price_cents=order.limit_price_cents,
            reserved_cost_dollars=reserved,
            created_ts_s=now_s,
            reprices=self.passive_reprices_current_episode,
            filled_count_seen=filled_count_seen,
            fill_cost_seen_dollars=fill_cost_seen_dollars,
        )
        self.state.reserved_cost_dollars += reserved
        self._write_event({
            "evt": "passive_reserved",
            "order_id": order_id,
            "client_order_id": client_order_id,
            "route_id": order.route_id,
            "market_ticker": order.market_ticker,
            "side": order.side,
            "count": count,
            "limit_price_cents": order.limit_price_cents,
            "reserved_cost_dollars": reserved,
            "total_reserved_cost_dollars": self.state.reserved_cost_dollars,
            "reprices": self.passive_reprices_current_episode,
            "filled_count_seen": filled_count_seen,
            "fill_cost_seen_dollars": fill_cost_seen_dollars,
        })

    def _refresh_active_passive(self, now_s: float) -> None:
        active = self.active_passive
        if active is None or self.dry_run or not active.order_id or not hasattr(self.client, "get_order"):
            return
        try:
            resp = self.client.get_order(active.order_id)
        except Exception as exc:
            self.api_error_timestamps_s.append(now_s)
            self._write_event({
                "evt": "passive_status_error",
                "order_id": active.order_id,
                "client_order_id": active.client_order_id,
                "err": repr(exc),
            })
            return
        self._apply_passive_order_snapshot(resp, source="passive_status")

    def _expire_or_cancel_passive_if_needed(self, now_s: float, quotes: Sequence[RouteQuote]) -> None:
        if self.active_passive is None:
            return
        timing = timing_state(now_s, float(self.ctx.tipoff_ts_s), self.cfg)
        by_route = {quote.route.route_id: quote for quote in quotes}
        quote = by_route.get(self.active_passive.route_id)
        if timing.lead_hours <= self.cfg.no_new_entry_hours_before_tip:
            self._cancel_active_passive("cancel_passive_at_T8", now_s=now_s)
            self.passive_reprices_current_episode = 0
        elif now_s - self.active_passive.created_ts_s >= _passive_timeout_s(timing, self.cfg):
            self._cancel_active_passive("cancel_passive_timeout", now_s=now_s)
            self.passive_reprices_current_episode = 0
        elif quote and quote.q_max_cents < self.active_passive.limit_price_cents:
            self._cancel_active_passive("cancel_passive_above_qmax", now_s=now_s)
            self.passive_reprices_current_episode = 0
        elif quote:
            desired_price = _passive_reprice_tick(quote, self.cfg)
            if desired_price and desired_price > self.active_passive.limit_price_cents:
                if self.active_passive.reprices < self.cfg.max_upward_reprices_per_passive_episode:
                    self.passive_reprices_current_episode = self.active_passive.reprices + 1
                    self._cancel_active_passive(
                        "cancel_passive_for_upward_reprice",
                        now_s=now_s,
                        start_cooldown=False,
                    )
                else:
                    self._cancel_active_passive("cancel_passive_reprice_limit", now_s=now_s)
                    self.passive_reprices_current_episode = 0

    def _cancel_active_passive(
        self,
        reason: str,
        *,
        now_s: Optional[float] = None,
        start_cooldown: bool = True,
    ) -> None:
        now = float(now_s if now_s is not None else time.time())
        active = self.active_passive
        if active is None:
            return
        resp = None
        if not self.dry_run and active.order_id:
            try:
                resp = self.client.cancel_order(active.order_id)
            except Exception as exc:
                self.order_reject_timestamps_s.append(now)
                self._write_event({
                    "evt": "passive_cancel_error",
                    "order_id": active.order_id,
                    "client_order_id": active.client_order_id,
                    "reason": reason,
                    "err": repr(exc),
                })
                return
        if resp:
            self._apply_passive_order_snapshot(resp, source="passive_cancel")
            active = self.active_passive
            if active is None:
                return
        self.state.reserved_cost_dollars = max(
            0.0,
            self.state.reserved_cost_dollars - active.reserved_cost_dollars,
        )
        self.active_passive = None
        if start_cooldown:
            self.passive_cooldown_until_s = now + self.cfg.passive_episode_cooldown_s
        self._write_event({
            "evt": "passive_cancelled",
            "order_id": active.order_id,
            "client_order_id": active.client_order_id,
            "route_id": active.route_id,
            "market_ticker": active.market_ticker,
            "reason": reason,
            "released_reserved_cost_dollars": active.reserved_cost_dollars,
            "total_reserved_cost_dollars": self.state.reserved_cost_dollars,
            "reprices": active.reprices,
            "cooldown_until_s": self.passive_cooldown_until_s,
        })

    def _apply_passive_order_snapshot(self, resp: Mapping[str, Any], *, source: str) -> None:
        active = self.active_passive
        if active is None:
            return
        order_resp = (resp or {}).get("order") or resp or {}
        filled_total = _order_fill_count(order_resp)
        cost_total = _order_fill_cost_dollars(
            order_resp,
            fallback_count=filled_total,
            fallback_price_cents=active.limit_price_cents,
        )
        filled_delta = max(0, filled_total - active.filled_count_seen)
        cost_delta = max(0.0, cost_total - active.fill_cost_seen_dollars)
        if filled_delta > 0 or cost_delta > 0:
            if cost_delta <= 0 and filled_delta > 0:
                cost_delta = filled_delta * active.limit_price_cents / 100.0
            active.filled_count_seen = filled_total
            active.fill_cost_seen_dollars = cost_total
            active.reserved_cost_dollars = max(0.0, active.reserved_cost_dollars - cost_delta)
            self.state.filled_cost_dollars += cost_delta
            self.state.reserved_cost_dollars = max(0.0, self.state.reserved_cost_dollars - cost_delta)
            self.state.filled_contracts_by_route[active.route_id] = (
                self.state.filled_contracts_by_route.get(active.route_id, 0) + filled_delta
            )
            self.state.filled_cost_by_route[active.route_id] = (
                self.state.filled_cost_by_route.get(active.route_id, 0.0) + cost_delta
            )
            self._write_event({
                "evt": "passive_fill_reconciled",
                "source": source,
                "order_id": active.order_id,
                "client_order_id": active.client_order_id,
                "route_id": active.route_id,
                "market_ticker": active.market_ticker,
                "filled_delta": filled_delta,
                "cost_delta_dollars": cost_delta,
                "filled_total": filled_total,
                "cost_total_dollars": cost_total,
                "cumulative_filled_cost_dollars": self.state.filled_cost_dollars,
                "total_reserved_cost_dollars": self.state.reserved_cost_dollars,
            })
        remaining = _order_remaining_count(order_resp)
        status = str(order_resp.get("status") or "").strip().lower()
        if remaining == 0 and (filled_total > 0 or status in {"canceled", "cancelled", "executed"}):
            self.state.reserved_cost_dollars = max(
                0.0,
                self.state.reserved_cost_dollars - active.reserved_cost_dollars,
            )
            self._write_event({
                "evt": "passive_completed",
                "source": source,
                "order_id": active.order_id,
                "client_order_id": active.client_order_id,
                "route_id": active.route_id,
                "market_ticker": active.market_ticker,
                "released_reserved_cost_dollars": active.reserved_cost_dollars,
                "total_reserved_cost_dollars": self.state.reserved_cost_dollars,
            })
            self.active_passive = None
            self.passive_reprices_current_episode = 0

    def _volume_snapshot(self, now_s: float, quotes: Sequence[RouteQuote]) -> VolumeSnapshot:
        if not hasattr(self.client, "get_trades"):
            return VolumeSnapshot()
        recent_by_route: Dict[str, float] = {}
        cumulative_by_route: Dict[str, float] = {}
        min_ts = int(max(0.0, min(
            now_s - self.cfg.recent_volume_window_hours * 3600.0,
            float(self.ctx.tipoff_ts_s) - self.cfg.monitor_start_hours_before_tip * 3600.0,
        )))
        max_ts = int(max(0.0, now_s))
        recent_cutoff = now_s - self.cfg.recent_volume_window_hours * 3600.0
        for quote in quotes:
            route_id = quote.route.route_id
            try:
                trades = self.client.get_trades(
                    ticker=quote.route.market_ticker,
                    min_ts=min_ts,
                    max_ts=max_ts,
                    limit=1000,
                )
            except Exception as exc:
                self.api_error_timestamps_s.append(now_s)
                self._write_event({
                    "evt": "trade_volume_error",
                    "route_id": route_id,
                    "market_ticker": quote.route.market_ticker,
                    "err": repr(exc),
                })
                continue
            recent = 0.0
            cumulative = 0.0
            for trade in trades:
                price_cents = _trade_price_for_route(quote.route, trade)
                if price_cents <= 0 or price_cents > quote.q_max_cents:
                    continue
                count = _trade_count(trade)
                if count <= 0:
                    continue
                cost = count * price_cents / 100.0
                cumulative += cost
                trade_ts = _trade_created_ts_s(trade)
                if trade_ts is not None and trade_ts >= recent_cutoff:
                    recent += cost
            recent_by_route[route_id] = recent
            cumulative_by_route[route_id] = cumulative
            self._write_event({
                "evt": "trade_volume_snapshot",
                "route_id": route_id,
                "market_ticker": quote.route.market_ticker,
                "min_ts": min_ts,
                "max_ts": max_ts,
                "recent_cutoff_ts": recent_cutoff,
                "recent_qualifying_volume_dollars": recent,
                "cumulative_qualifying_volume_dollars": cumulative,
            })
        return VolumeSnapshot(
            recent_qualifying_by_route=recent_by_route,
            cumulative_qualifying_by_route=cumulative_by_route,
        )

    def _write_expansion_blocked_plan(self, quotes: Sequence[RouteQuote]) -> None:
        q_max_cents = quotes[0].q_max_cents if quotes else 0
        self._write_event({
            "evt": "execution_plan",
            "selected_team_id": self.selected_team_id,
            "p_selected": self.p_selected,
            "operator_trade_allowed": self.operator_decision.trade_allowed,
            "operator_reason": self.operator_decision.reason,
            "operator_risk_mode": self.operator_decision.risk_mode,
            "position_mismatch_dollars": self.position_mismatch_dollars,
            "target_position_dollars": 0.0,
            "filled_position_dollars": self.state.filled_cost_dollars,
            "reserved_position_dollars": self.state.reserved_cost_dollars,
            "remaining_position_dollars": 0.0,
            "allowed_child_dollars": 0.0,
            "q_max_cents": q_max_cents,
            "cash_limited_mode": False,
            "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
            "cash_priority_rank": None,
            "cash_priority_score": None,
            "expected_log_growth_next_child": None,
            "cash_priority_candidate_child_dollars": 0.0,
            "q_current_position": self._current_position_q_all_in(),
            "q_avg_after_child": None,
            "skipped_due_to_cash": False,
            "decision": "no_trade",
            "reject_reason": self.expansion_gate.reason,
            **self.expansion_gate.to_log_payload(),
            "orders": [],
        })

    def _current_position_q_all_in(self) -> Optional[float]:
        filled_contracts = sum(int(v) for v in self.state.filled_contracts_by_route.values())
        filled_cost = max(0.0, self.state.filled_cost_dollars)

        reserved_contracts = 0
        reserved_cost = 0.0
        if self.active_passive is not None:
            reserved_contracts = max(0, self.active_passive.count - self.active_passive.filled_count_seen)
            reserved_cost = max(0.0, self.active_passive.reserved_cost_dollars)

        total_contracts = filled_contracts + reserved_contracts
        raw_cost = filled_cost + reserved_cost
        if total_contracts <= 0 or raw_cost <= 0.0:
            return None

        raw_avg_price_cents = int(round(100.0 * raw_cost / total_contracts))
        fee_dollars = estimate_kalshi_fee_dollars(
            total_contracts,
            max(1, min(99, raw_avg_price_cents)),
            self.cfg.fee_rate,
        )
        all_in_cost = raw_cost + fee_dollars
        return max(0.0001, min(0.9999, all_in_cost / total_contracts))

    def _refresh_operator_decision(self) -> None:
        previous = self.operator_decision
        self.operator_decision = resolve_operator_decision(
            self.ctx.game.game_id,
            global_control_path=self.operator_global_control_path,
            game_override_path_=self.operator_game_override_path,
        )
        if self.operator_decision != previous:
            payload = self.operator_decision.to_log_payload()
            payload["evt"] = "operator_control_refresh"
            self._write_event(payload)

    def _recover_open_orders_at_startup(self) -> None:
        if not (hasattr(self.client, "get_orders") and hasattr(self.client, "cancel_order")):
            self._write_event({
                "evt": "startup_open_order_recovery",
                "cancelled_orders": [],
                "unknown_open_orders": [],
                "cancel_errors": [],
                "blocked": False,
                "skipped": True,
                "skip_reason": "client_missing_order_methods",
            })
            return
        try:
            recovery = recover_open_route_orders(
                self.client,
                self.routes,
                client_order_prefix=self.client_order_prefix,
            )
        except Exception as exc:
            self.execution_block_reason = "startup_open_order_recovery_failed"
            self.api_error_timestamps_s.append(time.time())
            self._write_event({
                "evt": "startup_open_order_recovery_error",
                "err": repr(exc),
                "execution_block_reason": self.execution_block_reason,
            })
            return
        self._write_event({
            "evt": "startup_open_order_recovery",
            **recovery.to_log_payload(),
        })
        if recovery.blocked:
            self.execution_block_reason = "unknown_or_uncancelled_open_order_on_route_ticker"

    def _reconcile_exchange_position(
        self,
        *,
        source: str,
        now_s: Optional[float] = None,
        seed: bool = False,
    ) -> None:
        if self.dry_run:
            return
        if not (hasattr(self.client, "get_fills") and hasattr(self.client, "get_positions")):
            return
        now = float(now_s if now_s is not None else time.time())
        if (
            not seed
            and self.last_position_reconcile_s is not None
            and now - self.last_position_reconcile_s < self.position_reconcile_interval_s
        ):
            return
        try:
            rec = reconcile_exchange_routes(self.client, self.routes)
        except Exception as exc:
            self.api_error_timestamps_s.append(now)
            self._write_event({
                "evt": "position_reconciliation_error",
                "source": source,
                "err": repr(exc),
            })
            return
        local_before = dict(self.state.filled_cost_by_route)
        if seed:
            self.state.filled_contracts_by_route = {
                str(k): int(v) for k, v in rec.filled_contracts_by_route.items()
            }
            self.state.filled_cost_by_route = {
                str(k): float(v) for k, v in rec.filled_cost_by_route.items()
            }
            self.state.filled_cost_dollars = float(rec.filled_cost_dollars)
            self.position_mismatch_dollars = 0.0
        else:
            self.position_mismatch_dollars = rec.mismatch_dollars(self.state.filled_cost_by_route)
        self.last_position_reconcile_s = now
        payload = rec.to_log_payload()
        payload.update({
            "evt": "position_reconciliation",
            "source": source,
            "seeded_local_state": seed,
            "local_filled_cost_by_route_before": local_before,
            "local_filled_cost_by_route_after": dict(self.state.filled_cost_by_route),
            "position_mismatch_dollars": self.position_mismatch_dollars,
        })
        self._write_event(payload)

    def _write_event(self, payload: Dict[str, Any]) -> None:
        payload = {
            "ts_ms": int(time.time() * 1000),
            "game_id": self.ctx.game.game_id,
            **payload,
        }
        if self.ledger is not None:
            payload = self.ledger.write_event(payload)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        if payload.get("evt") in {
            "route_loop_start",
            "operator_control_refresh",
            "mapping",
            "expansion_gate",
            "startup_open_order_recovery",
            "startup_open_order_recovery_error",
            "position_reconciliation_error",
            "portfolio_sizing",
            "portfolio_sizing_error",
            "signal_state",
            "execution_plan",
            "dry_order",
            "passive_reserved",
            "passive_cancelled",
            "passive_cancel_error",
            "passive_status_error",
            "passive_fill_reconciled",
            "passive_completed",
            "trade_volume_error",
            "trade_volume_snapshot",
            "route_capacity",
            "order_submitted",
            "fill",
            "order_error",
            "tipoff_stop",
        }:
            log.info("[event] %s", json.dumps(payload, default=str))

    def _refresh_portfolio_sizing(
        self,
        *,
        now_s: Optional[float] = None,
        force: bool = False,
        emit: bool = True,
    ) -> None:
        if not (
            self.follow_kalshi_wealth
            or self.sizing_bankroll_override_dollars is not None
            or self.available_cash_override_dollars is not None
        ):
            return
        now = float(now_s if now_s is not None else time.time())
        if (
            not force
            and self.last_portfolio_refresh_s is not None
            and now - self.last_portfolio_refresh_s < self.portfolio_refresh_interval_s
        ):
            return
        try:
            snapshot = resolve_portfolio_sizing(
                self.client,
                sizing_bankroll_override_dollars=self.sizing_bankroll_override_dollars,
                available_cash_override_dollars=self.available_cash_override_dollars,
            )
        except Exception as exc:
            self.api_error_timestamps_s.append(now)
            if self.portfolio_sizing is None:
                raise
            if emit:
                self._write_event({
                    "evt": "portfolio_sizing_error",
                    "err": repr(exc),
                    "using_last_sizing_bankroll_dollars": self.cfg.bankroll,
                    "using_last_available_cash_dollars": self.available_cash_dollars,
                })
            return
        self._apply_portfolio_sizing(snapshot)
        self.last_portfolio_refresh_s = now
        if emit:
            self._write_portfolio_sizing_event(source="refresh")

    def _apply_portfolio_sizing(self, snapshot: PortfolioSizingSnapshot) -> None:
        self.portfolio_sizing = snapshot
        self.cfg = replace(self.cfg, bankroll=snapshot.sizing_bankroll_dollars)
        self.available_cash_dollars = snapshot.available_cash_dollars

    def _write_portfolio_sizing_event(self, *, source: str) -> None:
        if self.portfolio_sizing is None:
            return
        self._write_event({
            "evt": "portfolio_sizing",
            "source": source,
            **self.portfolio_sizing.to_log_payload(),
        })


def _market_payload(market: Any) -> Dict[str, Any]:
    return {
        "ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "yes_team_id": market.yes_team_id,
        "yes_team_id_source": market.yes_team_id_source,
        "custom_strike_team_id": market.custom_strike_team_id,
        "yes_team_name": market.yes_team_name,
        "title": market.title,
        "status": market.status,
    }


def _prediction_value(result: Mapping[str, Any], key: str) -> Optional[float]:
    values = result.get(key)
    if not values:
        return None
    try:
        value = values[0]
    except Exception:
        value = values
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _feature_row_payload(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {str(k): _json_scalar(v) for k, v in row.items()}


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return _json_scalar(value.item())
        except Exception:
            pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _intish(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _order_fill_count(order_resp: Mapping[str, Any]) -> int:
    for key in ("filled_count", "fill_count", "fill_count_fp", "filled_count_fp"):
        value = order_resp.get(key)
        count = _intish(value)
        if count > 0:
            return count
    return 0


def _order_remaining_count(order_resp: Mapping[str, Any]) -> Optional[int]:
    for key in ("remaining_count", "remaining_count_fp"):
        if key in order_resp and order_resp.get(key) is not None:
            return max(0, _intish(order_resp.get(key)))
    return None


def _order_fill_cost_dollars(
    order_resp: Mapping[str, Any],
    *,
    fallback_count: int,
    fallback_price_cents: int,
) -> float:
    cents = (
        _number(order_resp.get("taker_fill_cost"))
        + _number(order_resp.get("maker_fill_cost"))
    )
    if cents > 0:
        return cents / 100.0
    dollars = (
        _number(order_resp.get("taker_fill_cost_dollars"))
        + _number(order_resp.get("maker_fill_cost_dollars"))
    )
    if dollars > 0:
        return dollars
    return max(0.0, fallback_count * fallback_price_cents / 100.0)


def _trade_count(trade: Mapping[str, Any]) -> int:
    for key in ("count", "count_fp"):
        count = _intish(trade.get(key))
        if count > 0:
            return count
    return 0


def _trade_price_for_route(route: Any, trade: Mapping[str, Any]) -> int:
    if route.route_type == "BUY_YES_SELECTED":
        return _trade_price_cents(trade, cents_key="yes_price", dollars_key="yes_price_dollars")
    if route.route_type == "BUY_NO_OPPONENT":
        return _trade_price_cents(trade, cents_key="no_price", dollars_key="no_price_dollars")
    return 0


def _trade_price_cents(trade: Mapping[str, Any], *, cents_key: str, dollars_key: str) -> int:
    if trade.get(cents_key) is not None:
        value = _number(trade.get(cents_key))
        if 0 < value <= 1:
            return int(round(value * 100.0))
        return int(round(value))
    if trade.get(dollars_key) is not None:
        return int(round(_number(trade.get(dollars_key)) * 100.0))
    return 0


def _trade_created_ts_s(trade: Mapping[str, Any]) -> Optional[float]:
    for key in ("created_ts", "created_time", "timestamp", "ts"):
        value = trade.get(key)
        if value is None:
            continue
        numeric = _number(value)
        if numeric > 0:
            return numeric
        try:
            return float(pd.Timestamp(value).timestamp())
        except Exception:
            continue
    return None


def _passive_timeout_s(timing, cfg: ExecutionConfig) -> float:
    return cfg.passive_timeout_t17_to_t12_s if timing.lead_hours > 12 else cfg.passive_timeout_t12_to_t8_s


def _passive_reprice_tick(quote: RouteQuote, cfg: ExecutionConfig) -> Optional[int]:
    if (
        quote.best_bid_cents <= 0
        or quote.best_ask_cents <= 0
        or quote.spread_ticks < cfg.min_spread_for_passive_ticks
        or quote.q_max_cents <= 1
    ):
        return None
    midpoint = math.floor((quote.best_bid_cents + quote.best_ask_cents) / 2)
    price = min(quote.best_bid_cents + 1, midpoint, quote.q_max_cents - 1)
    if quote.best_bid_cents < price < quote.best_ask_cents:
        return price
    return None
