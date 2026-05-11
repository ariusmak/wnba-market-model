"""
Locked v1.2 execution planner.

This module is side-effect-free. It applies timing, signal memory,
liquidity, child-slicing, passive-probe, burst, and operational-brake
rules to route quotes, then returns planned child orders.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .cash_priority import PRIORITY_RULE, marginal_expected_log_growth_per_dollar
from .execution import (
    ExecutionConfig,
    ExecutionPlan,
    PlannedChildOrder,
    RouteQuote,
    max_all_in_price_cents,
    target_position_dollars,
)

CONSERVATIVE_MAX_MARKET_EXPOSURE_PCT = 0.12


@dataclass
class SignalMemory:
    first_qualified_ts_s: Optional[float] = None
    first_qualified_lead_hours: Optional[float] = None
    first_qualified_route_id: Optional[str] = None
    first_qualified_price_all_in_cents: Optional[float] = None
    first_qualified_edge: Optional[float] = None
    last_qualified_ts_s: Optional[float] = None
    num_qualifying_snapshots: int = 0

    @property
    def signal_class(self) -> str:
        if self.first_qualified_lead_hours is None:
            return "unqualified"
        if self.first_qualified_lead_hours < 8.0:
            return "late_only"
        return "early_stable"

    def to_log_payload(self) -> Dict[str, object]:
        return {
            "first_qualified_ts_s": self.first_qualified_ts_s,
            "first_qualified_lead_hours": self.first_qualified_lead_hours,
            "first_qualified_route_id": self.first_qualified_route_id,
            "first_qualified_price_all_in_cents": self.first_qualified_price_all_in_cents,
            "first_qualified_edge": self.first_qualified_edge,
            "last_qualified_ts_s": self.last_qualified_ts_s,
            "num_qualifying_snapshots": self.num_qualifying_snapshots,
            "signal_class": self.signal_class,
        }


@dataclass(frozen=True)
class TimingState:
    lead_hours: float
    window: str
    poll_interval_s: float
    orders_allowed: bool
    ioc_allowed: bool
    passive_allowed: bool
    prequalified_only: bool
    reject_reason: str = ""


@dataclass(frozen=True)
class VolumeSnapshot:
    recent_qualifying_by_route: Mapping[str, float] = field(default_factory=dict)
    cumulative_qualifying_by_route: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BrakeState:
    order_reject_timestamps_s: Tuple[float, ...] = ()
    api_error_timestamps_s: Tuple[float, ...] = ()
    position_mismatch_dollars: float = 0.0
    conservative_mode: bool = False


@dataclass(frozen=True)
class PlannerRuntimeState:
    now_s: float
    tipoff_ts_s: float
    signal: SignalMemory
    filled_position_dollars: float = 0.0
    reserved_position_dollars: float = 0.0
    filled_cost_by_route: Mapping[str, float] = field(default_factory=dict)
    current_position_q_all_in: Optional[float] = None
    available_cash_dollars: Optional[float] = None
    volume: VolumeSnapshot = field(default_factory=VolumeSnapshot)
    last_ioc_order_ts_s: Optional[float] = None
    last_burst_order_ts_s: Optional[float] = None
    previous_combined_visible_cost_dollars: float = 0.0
    combined_visible_cost_after_last_order_dollars: float = 0.0
    burst_orders_last_5min: Tuple[Tuple[float, float], ...] = ()
    passive_order_live: bool = False
    passive_cooldown_until_s: Optional[float] = None
    brake_state: BrakeState = field(default_factory=BrakeState)


def timing_state(now_s: float, tipoff_ts_s: float, cfg: ExecutionConfig) -> TimingState:
    lead_hours = max(0.0, (tipoff_ts_s - now_s) / 3600.0)
    if lead_hours > cfg.monitor_start_hours_before_tip:
        return TimingState(
            lead_hours, "before_monitor", cfg.poll_t24_to_t17_s,
            False, False, False, False, "before_monitor_window",
        )
    if lead_hours > cfg.trade_start_hours_before_tip:
        return TimingState(
            lead_hours, "monitor_T24_to_T17", cfg.poll_t24_to_t17_s,
            False, False, False, False, "monitor_only_before_T17",
        )
    if lead_hours > 12.0:
        return TimingState(
            lead_hours, "main_T17_to_T12", cfg.poll_t17_to_t12_s,
            True, True, True, False,
        )
    if lead_hours > cfg.no_new_entry_hours_before_tip:
        return TimingState(
            lead_hours, "main_T12_to_T8", cfg.poll_t12_to_t8_s,
            True, True, True, False,
        )
    if lead_hours > cfg.no_orders_hours_before_tip:
        return TimingState(
            lead_hours, "prequalified_T8_to_T4", cfg.poll_t8_to_t4_s,
            True, True, False, True,
        )
    return TimingState(
        lead_hours, "monitor_T4_to_tip", cfg.poll_t4_to_tip_s,
        False, False, False, False, "after_hard_no_add_cutoff",
    )


def update_signal_memory(
    signal: SignalMemory,
    route_quotes: Sequence[RouteQuote],
    *,
    now_s: float,
    tipoff_ts_s: float,
) -> SignalMemory:
    eligible = [quote for quote in route_quotes if quote.eligible]
    if not eligible:
        return signal
    lead_hours = max(0.0, (tipoff_ts_s - now_s) / 3600.0)
    best = min(eligible, key=lambda quote: quote.all_in_avg_price_cents)
    signal.last_qualified_ts_s = now_s
    signal.num_qualifying_snapshots += 1
    if signal.first_qualified_ts_s is None:
        signal.first_qualified_ts_s = now_s
        signal.first_qualified_lead_hours = lead_hours
        signal.first_qualified_route_id = best.route.route_id
        signal.first_qualified_price_all_in_cents = best.all_in_avg_price_cents
        signal.first_qualified_edge = best.edge
    return signal


def plan_v1_2_orders(
    *,
    selected_team_id: str,
    p_selected: float,
    route_quotes: Sequence[RouteQuote],
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
) -> ExecutionPlan:
    cfg = _config_for_runtime(cfg, runtime)
    cfg.validate()
    timing = timing_state(runtime.now_s, runtime.tipoff_ts_s, cfg)
    q_max_cents = max_all_in_price_cents(p_selected, cfg)
    enriched = tuple(_apply_liquidity_caps(route_quotes, cfg, runtime))
    eligible = [quote for quote in enriched if quote.eligible and quote.route_capacity_dollars > 0]
    route_capacity_sum = sum(quote.route_capacity_dollars for quote in eligible)
    global_cumulative_remaining = _global_cumulative_remaining(enriched, cfg, runtime)

    if timing.reject_reason and not timing.orders_allowed:
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            decision="monitor_only", reject_reason=timing.reject_reason,
            timing=timing, signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    brake_reason = _enforced_brake_reason(runtime, cfg)
    if brake_reason:
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            decision="blocked_brake", reject_reason=brake_reason,
            timing=timing, signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    if runtime.signal.first_qualified_lead_hours is None:
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            decision="no_trade", reject_reason="never_qualified",
            timing=timing, signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    if runtime.signal.first_qualified_lead_hours < cfg.no_new_entry_hours_before_tip:
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            decision="no_trade", reject_reason="late_only_signal",
            timing=timing, signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    if timing.prequalified_only and runtime.signal.first_qualified_lead_hours < cfg.no_new_entry_hours_before_tip:
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            decision="no_trade", reject_reason="no_prequalified_ticket",
            timing=timing, signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    if not eligible:
        passive = _plan_passive_probe(
            selected_team_id=selected_team_id,
            p_selected=p_selected,
            route_quotes=enriched,
            cfg=cfg,
            runtime=runtime,
            timing=timing,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )
        if passive is not None:
            return passive
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            decision="wait", reject_reason="no_eligible_ioc_routes",
            timing=timing, signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    best = min(eligible, key=lambda quote: quote.all_in_avg_price_cents)
    target = target_position_dollars(
        p_selected=p_selected,
        q_all_in=best.all_in_avg_price_cents / 100.0,
        cfg=cfg,
        available_cash_dollars=runtime.available_cash_dollars,
    )
    remaining = max(0.0, target - runtime.filled_position_dollars - runtime.reserved_position_dollars)
    available_cash_after_buffer = _available_cash_after_buffer(runtime, cfg)
    allowed_to_try = max(
        0.0,
        min(remaining, route_capacity_sum, global_cumulative_remaining, available_cash_after_buffer),
    )
    binding_cap = _binding_cap(
        remaining,
        route_capacity_sum,
        global_cumulative_remaining,
        available_cash_after_buffer,
    )
    priority_fields = _cash_priority_fields(
        p_selected=p_selected,
        best_quote=best,
        cfg=cfg,
        runtime=runtime,
        candidate_child_size=allowed_to_try,
        binding_cap=binding_cap,
    )
    if allowed_to_try < cfg.min_child_order_dollars and not _cleanup_allowed(remaining, runtime, cfg):
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            target_position_dollars=target,
            remaining_position_dollars=remaining,
            allowed_child_dollars=allowed_to_try,
            decision="wait",
            reject_reason="allowed_to_try_below_min_child",
            timing=timing,
            signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
            binding_cap=binding_cap,
            **priority_fields,
        )

    if not timing.ioc_allowed:
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            target_position_dollars=target,
            remaining_position_dollars=remaining,
            allowed_child_dollars=allowed_to_try,
            decision="monitor_only",
            reject_reason=timing.reject_reason or "ioc_not_allowed_in_window",
            timing=timing,
            signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
            binding_cap=binding_cap,
            **priority_fields,
        )

    mode = "normal_ioc"
    child_size = _normal_child_size(remaining, allowed_to_try, available_cash_after_buffer, timing, cfg, runtime)
    if _burst_triggered(enriched, runtime, cfg) and _burst_debounce_ok(runtime, cfg):
        mode = "burst_ioc"
        child_size = _burst_child_size(remaining, allowed_to_try, child_size, available_cash_after_buffer, cfg)
    elif timing.prequalified_only:
        mode = "completion_ioc"
        child_size = _completion_child_size(remaining, allowed_to_try, available_cash_after_buffer, timing, cfg, runtime)
    elif not _normal_debounce_ok(runtime, cfg):
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            target_position_dollars=target,
            remaining_position_dollars=remaining,
            allowed_child_dollars=0.0,
            decision="wait",
            reject_reason="ioc_debounce_active",
            timing=timing,
            signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
        )

    child_size = min(child_size, allowed_to_try)
    if mode == "burst_ioc":
        child_size = min(child_size, _burst_room_last_5min(runtime, cfg))
    priority_fields = _cash_priority_fields(
        p_selected=p_selected,
        best_quote=best,
        cfg=cfg,
        runtime=runtime,
        candidate_child_size=child_size,
        binding_cap=binding_cap,
    )
    if child_size < cfg.min_child_order_dollars and not _cleanup_allowed(remaining, runtime, cfg):
        return _plan(
            selected_team_id, p_selected, q_max_cents, enriched,
            target_position_dollars=target,
            remaining_position_dollars=remaining,
            allowed_child_dollars=child_size,
            decision="wait",
            reject_reason="child_size_below_min",
            timing=timing,
            signal=runtime.signal,
            route_capacity_sum=route_capacity_sum,
            global_cumulative_remaining=global_cumulative_remaining,
            binding_cap=binding_cap,
            **priority_fields,
        )

    orders = _allocate_orders(eligible, child_size, cfg, order_mode=mode)
    return _plan(
        selected_team_id, p_selected, q_max_cents, enriched,
        target_position_dollars=target,
        remaining_position_dollars=remaining,
        allowed_child_dollars=child_size,
        decision=mode if orders else "wait",
        reject_reason="" if orders else "no_contracts_after_rounding",
        orders=orders,
        timing=timing,
        signal=runtime.signal,
        route_capacity_sum=route_capacity_sum,
        global_cumulative_remaining=global_cumulative_remaining,
        binding_cap=binding_cap,
        **priority_fields,
    )


def _apply_liquidity_caps(
    route_quotes: Sequence[RouteQuote],
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
) -> Iterable[RouteQuote]:
    for quote in route_quotes:
        route_id = quote.route.route_id
        recent = float(runtime.volume.recent_qualifying_by_route.get(route_id, 0.0))
        cumulative = float(runtime.volume.cumulative_qualifying_by_route.get(route_id, 0.0))
        visible_cap = cfg.max_visible_depth_participation * quote.visible_cost_dollars_at_qmax
        recent_cap = cfg.max_recent_qualifying_volume_participation * recent
        cold_start_cap = min(
            cfg.cold_start_bankroll_cap * cfg.bankroll,
            cfg.cold_start_visible_depth_participation * quote.visible_cost_dollars_at_qmax,
        )
        rolling = min(visible_cap, max(recent_cap, cold_start_cap))
        cumulative_cap = max(
            cfg.max_cumulative_qualifying_volume_share * cumulative,
            cold_start_cap,
        )
        filled_route = float(runtime.filled_cost_by_route.get(route_id, 0.0))
        cumulative_remaining = max(0.0, cumulative_cap - filled_route)
        capacity = max(0.0, min(rolling, cumulative_remaining))
        yield RouteQuote(
            **{
                **quote.__dict__,
                "visible_depth_cap_dollars": visible_cap,
                "recent_qualifying_volume_dollars": recent,
                "recent_qualifying_volume_cap_dollars": recent_cap,
                "cumulative_qualifying_volume_dollars": cumulative,
                "cumulative_cap_remaining_dollars": cumulative_remaining,
                "cold_start_cap_dollars": cold_start_cap,
                "route_capacity_dollars": capacity,
            }
        )


def _global_cumulative_remaining(
    route_quotes: Sequence[RouteQuote],
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
) -> float:
    cumulative = sum(quote.cumulative_qualifying_volume_dollars for quote in route_quotes)
    cold_start = max((quote.cold_start_cap_dollars for quote in route_quotes), default=0.0)
    cap = max(cfg.max_cumulative_qualifying_volume_share * cumulative, cold_start)
    return max(0.0, cap - runtime.filled_position_dollars)


def _plan_passive_probe(
    *,
    selected_team_id: str,
    p_selected: float,
    route_quotes: Sequence[RouteQuote],
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
    timing: TimingState,
    route_capacity_sum: float,
    global_cumulative_remaining: float,
) -> Optional[ExecutionPlan]:
    if not cfg.passive_enabled or not timing.passive_allowed or runtime.passive_order_live:
        return None
    if runtime.passive_cooldown_until_s and runtime.now_s < runtime.passive_cooldown_until_s:
        return None
    if not _passive_debounce_ok(runtime, cfg):
        return None

    candidates = [
        quote for quote in route_quotes
        if quote.best_bid_cents > 0
        and quote.best_ask_cents > 0
        and quote.spread_ticks >= cfg.min_spread_for_passive_ticks
        and quote.q_max_cents > 1
    ]
    passive_orders = []
    allowed = max(cfg.min_child_order_dollars, min(
        route_capacity_sum or max((q.cold_start_cap_dollars for q in route_quotes), default=0.0),
        global_cumulative_remaining,
        _available_cash_after_buffer(runtime, cfg),
    ))
    child_size = min(
        cfg.passive_child_fraction_of_allowed * allowed,
        cfg.max_passive_child_order_pct * cfg.bankroll,
        _available_cash_after_buffer(runtime, cfg),
    )
    if child_size < cfg.min_child_order_dollars:
        return None
    for quote in sorted(candidates, key=lambda q: q.spread_ticks, reverse=True):
        midpoint = math.floor((quote.best_bid_cents + quote.best_ask_cents) / 2)
        price = min(quote.best_bid_cents + 1, midpoint, quote.q_max_cents - 1)
        if not (quote.best_bid_cents < price < quote.best_ask_cents):
            continue
        count = int(math.floor(child_size / (price / 100.0)))
        if count <= 0:
            continue
        passive_orders.append(PlannedChildOrder(
            route_id=quote.route.route_id,
            market_ticker=quote.route.market_ticker,
            route_type=quote.route.route_type,
            action=quote.route.action,
            side=quote.route.side,
            count=count,
            limit_price_cents=price,
            max_cost_dollars=count * price / 100.0,
            expected_all_in_avg_price_cents=float(price),
            q_max_cents=quote.q_max_cents,
            order_mode="passive_probe",
            time_in_force="good_till_canceled",
            post_only=True,
            expiration_ts=int(min(runtime.tipoff_ts_s - cfg.no_new_entry_hours_before_tip * 3600, runtime.now_s + _passive_timeout_s(timing, cfg))),
        ))
        break
    if not passive_orders:
        return None
    return _plan(
        selected_team_id,
        p_selected,
        max_all_in_price_cents(p_selected, cfg),
        route_quotes,
        allowed_child_dollars=passive_orders[0].max_cost_dollars,
        decision="passive_probe",
        orders=tuple(passive_orders),
        timing=timing,
        signal=runtime.signal,
        route_capacity_sum=route_capacity_sum,
        global_cumulative_remaining=global_cumulative_remaining,
    )


def _allocate_orders(
    eligible: Sequence[RouteQuote],
    child_size_dollars: float,
    cfg: ExecutionConfig,
    *,
    order_mode: str,
) -> Tuple[PlannedChildOrder, ...]:
    ordered = sorted(eligible, key=lambda quote: quote.all_in_avg_price_cents)
    tied = [
        quote for quote in ordered
        if quote.all_in_avg_price_cents - ordered[0].all_in_avg_price_cents
        <= cfg.route_price_tie_threshold_ticks
    ]
    if len(tied) > 1:
        capacity_sum = sum(quote.route_capacity_dollars for quote in tied)
        allocations = [
            (quote, child_size_dollars * (quote.route_capacity_dollars / capacity_sum))
            for quote in tied
            if capacity_sum > 0
        ]
    else:
        remaining_for_alloc = child_size_dollars
        allocations = []
        for quote in ordered:
            if remaining_for_alloc <= 0:
                break
            dollars = min(remaining_for_alloc, quote.route_capacity_dollars)
            allocations.append((quote, dollars))
            remaining_for_alloc -= dollars

    orders = []
    for quote, dollars in allocations:
        if dollars < cfg.min_child_order_dollars:
            continue
        price_dollars = quote.limit_price_cents / 100.0
        count = min(
            int(math.floor(dollars / price_dollars)) if price_dollars > 0 else 0,
            quote.fillable_contracts_at_qmax,
        )
        if count <= 0:
            continue
        max_cost = count * price_dollars
        orders.append(PlannedChildOrder(
            route_id=quote.route.route_id,
            market_ticker=quote.route.market_ticker,
            route_type=quote.route.route_type,
            action=quote.route.action,
            side=quote.route.side,
            count=count,
            limit_price_cents=quote.limit_price_cents,
            max_cost_dollars=max_cost,
            expected_all_in_avg_price_cents=quote.all_in_avg_price_cents,
            q_max_cents=quote.q_max_cents,
            order_mode=order_mode,
            time_in_force="immediate_or_cancel",
            post_only=False,
        ))
    return tuple(orders)


def _normal_child_size(
    remaining: float,
    allowed: float,
    cash: float,
    timing: TimingState,
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
) -> float:
    opportunities = _effective_remaining_opportunities(timing, cfg, runtime)
    urgency = cfg.urgency_multiplier_t17_to_t12 if timing.lead_hours > 12 else cfg.urgency_multiplier_t12_to_t8
    base_slice = remaining / max(1, opportunities)
    return min(remaining, allowed, base_slice * urgency, cfg.normal_max_ioc_child_order_pct * cfg.bankroll, cash)


def _completion_child_size(
    remaining: float,
    allowed: float,
    cash: float,
    timing: TimingState,
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
) -> float:
    opportunities = _effective_remaining_opportunities(timing, cfg, runtime)
    base_slice = remaining / max(1, opportunities)
    return min(remaining, allowed, base_slice * cfg.urgency_multiplier_t8_to_t4, cfg.completion_max_ioc_child_order_pct * cfg.bankroll, cash)


def _burst_child_size(
    remaining: float,
    allowed: float,
    normal_child: float,
    cash: float,
    cfg: ExecutionConfig,
) -> float:
    return min(remaining, allowed, cfg.burst_max_ioc_child_order_pct * cfg.bankroll, 2.5 * normal_child, cash)


def _burst_triggered(route_quotes: Sequence[RouteQuote], runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> bool:
    if not cfg.burst_enabled:
        return False
    combined_now = sum(quote.visible_cost_dollars_at_qmax for quote in route_quotes if quote.eligible)
    if combined_now < cfg.burst_min_visible_depth_pct_bankroll * cfg.bankroll:
        return False
    if (
        runtime.previous_combined_visible_cost_dollars <= 0
        and runtime.combined_visible_cost_after_last_order_dollars <= 0
    ):
        return False
    if runtime.previous_combined_visible_cost_dollars > 0 and combined_now < cfg.burst_depth_multiplier_trigger * runtime.previous_combined_visible_cost_dollars:
        return False
    refresh_threshold = max(
        25.0,
        0.005 * cfg.bankroll,
        0.10 * runtime.combined_visible_cost_after_last_order_dollars,
    )
    return combined_now > runtime.combined_visible_cost_after_last_order_dollars + refresh_threshold


def _normal_debounce_ok(runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> bool:
    return runtime.last_ioc_order_ts_s is None or runtime.now_s - runtime.last_ioc_order_ts_s >= cfg.normal_min_time_between_ioc_sweeps_s


def _burst_debounce_ok(runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> bool:
    if runtime.last_burst_order_ts_s is not None and runtime.now_s - runtime.last_burst_order_ts_s < cfg.burst_min_time_between_ioc_sweeps_s:
        return False
    recent = [(ts, cost) for ts, cost in runtime.burst_orders_last_5min if runtime.now_s - ts <= 300]
    if len(recent) >= cfg.max_burst_orders_per_5min:
        return False
    return _burst_room_last_5min(runtime, cfg) >= cfg.min_child_order_dollars


def _burst_room_last_5min(runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> float:
    used = sum(cost for ts, cost in runtime.burst_orders_last_5min if runtime.now_s - ts <= 300)
    return max(0.0, cfg.max_burst_total_per_5min_pct * cfg.bankroll - used)


def _passive_debounce_ok(runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> bool:
    # Passive orders are represented as reserved exposure. If no live passive
    # exists, allow a new episode; repricing cadence is handled by the loop.
    return not runtime.passive_order_live


def _passive_timeout_s(timing: TimingState, cfg: ExecutionConfig) -> float:
    return cfg.passive_timeout_t17_to_t12_s if timing.lead_hours > 12 else cfg.passive_timeout_t12_to_t8_s


def _effective_remaining_opportunities(timing: TimingState, cfg: ExecutionConfig, runtime: PlannerRuntimeState) -> int:
    cutoff = runtime.tipoff_ts_s - cfg.no_orders_hours_before_tip * 3600.0
    seconds_left = max(0.0, cutoff - runtime.now_s)
    expected = max(1, int(math.floor(seconds_left / max(1.0, timing.poll_interval_s))))
    return min(expected, cfg.max_effective_remaining_opportunities)


def _available_cash_after_buffer(runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> float:
    if runtime.available_cash_dollars is None:
        return cfg.bankroll
    return max(0.0, runtime.available_cash_dollars - cfg.cash_buffer_pct * cfg.bankroll)


def _cleanup_allowed(remaining: float, runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> bool:
    return 0.0 < remaining < cfg.min_child_order_dollars and runtime.filled_position_dollars > 0


def _binding_cap(remaining: float, route_cap: float, global_cap: float, cash: float) -> str:
    values = {
        "remaining_position": remaining,
        "route_capacity_sum": route_cap,
        "global_cumulative_liquidity": global_cap,
        "available_cash_after_buffer": cash,
    }
    return min(values, key=values.get)


def _cash_priority_fields(
    *,
    p_selected: float,
    best_quote: RouteQuote,
    cfg: ExecutionConfig,
    runtime: PlannerRuntimeState,
    candidate_child_size: float,
    binding_cap: str,
) -> Dict[str, object]:
    fields: Dict[str, object] = {
        "cash_limited_mode": binding_cap == "available_cash_after_buffer",
        "cash_priority_rule": PRIORITY_RULE,
        "cash_priority_candidate_child_dollars": max(0.0, candidate_child_size),
        "skipped_due_to_cash": False,
    }
    if candidate_child_size <= 0.0:
        return fields

    current_cost = max(0.0, runtime.filled_position_dollars + runtime.reserved_position_dollars)
    q_child = best_quote.all_in_avg_price_cents / 100.0
    q_current = runtime.current_position_q_all_in if current_cost > 0.0 else q_child
    try:
        marginal = marginal_expected_log_growth_per_dollar(
            p=p_selected,
            q_child=q_child,
            bankroll_for_sizing=cfg.bankroll,
            current_cost_dollars=current_cost,
            child_cost_dollars=candidate_child_size,
            q_current_position=q_current,
        )
    except ValueError:
        return fields

    fields.update(
        cash_priority_score=marginal.priority_score,
        expected_log_growth_next_child=marginal.expected_log_growth_next_child,
        q_current_position=marginal.q_current_position,
        q_avg_after_child=marginal.q_avg_after_child,
    )
    return fields


def _enforced_brake_reason(runtime: PlannerRuntimeState, cfg: ExecutionConfig) -> str:
    order_rejects = [ts for ts in runtime.brake_state.order_reject_timestamps_s if runtime.now_s - ts <= 3600]
    if len(order_rejects) >= cfg.order_reject_brake_threshold_per_hour:
        return "order_reject_brake_enforced"
    api_errors = [ts for ts in runtime.brake_state.api_error_timestamps_s if runtime.now_s - ts <= 600]
    if len(api_errors) >= cfg.api_error_brake_threshold_per_10min:
        return "api_error_brake_enforced"
    if runtime.brake_state.position_mismatch_dollars > 10.0:
        return "position_mismatch_brake_enforced"
    return ""


def _config_for_runtime(cfg: ExecutionConfig, runtime: PlannerRuntimeState) -> ExecutionConfig:
    if not runtime.brake_state.conservative_mode:
        return cfg
    return replace(
        cfg,
        max_market_exposure_pct=min(
            cfg.max_market_exposure_pct,
            CONSERVATIVE_MAX_MARKET_EXPOSURE_PCT,
        ),
        burst_enabled=False,
    )


def _plan(
    selected_team_id: str,
    p_selected: float,
    q_max_cents: int,
    route_quotes: Sequence[RouteQuote],
    *,
    target_position_dollars: float = 0.0,
    remaining_position_dollars: float = 0.0,
    allowed_child_dollars: float = 0.0,
    decision: str,
    reject_reason: str = "",
    orders: Sequence[PlannedChildOrder] = (),
    timing: TimingState,
    signal: SignalMemory,
    route_capacity_sum: float,
    global_cumulative_remaining: float,
    binding_cap: str = "",
    cash_limited_mode: bool = False,
    cash_priority_rule: str = "",
    cash_priority_rank: Optional[int] = None,
    cash_priority_score: Optional[float] = None,
    expected_log_growth_next_child: Optional[float] = None,
    cash_priority_candidate_child_dollars: float = 0.0,
    q_current_position: Optional[float] = None,
    q_avg_after_child: Optional[float] = None,
    skipped_due_to_cash: bool = False,
) -> ExecutionPlan:
    return ExecutionPlan(
        selected_team_id=selected_team_id,
        p_selected=p_selected,
        target_position_dollars=target_position_dollars,
        remaining_position_dollars=remaining_position_dollars,
        allowed_child_dollars=allowed_child_dollars,
        q_max_cents=q_max_cents,
        route_quotes=tuple(route_quotes),
        orders=tuple(orders),
        decision=decision,
        reject_reason=reject_reason,
        timing_window=timing.window,
        lead_hours=timing.lead_hours,
        signal_class=signal.signal_class,
        binding_cap=binding_cap,
        route_capacity_sum_dollars=route_capacity_sum,
        global_cumulative_remaining_dollars=global_cumulative_remaining,
        cash_limited_mode=cash_limited_mode,
        cash_priority_rule=cash_priority_rule,
        cash_priority_rank=cash_priority_rank,
        cash_priority_score=cash_priority_score,
        expected_log_growth_next_child=expected_log_growth_next_child,
        cash_priority_candidate_child_dollars=cash_priority_candidate_child_dollars,
        q_current_position=q_current_position,
        q_avg_after_child=q_avg_after_child,
        skipped_due_to_cash=skipped_due_to_cash,
    )
