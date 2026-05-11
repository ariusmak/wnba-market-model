"""
Execution router foundation for WNBA Kalshi live trading.

This is the first production-oriented slice of the v1.2 executor. It is
intentionally side-effect-free: it consumes confirmed route candidates and
orderbook payloads, then returns planned IOC child orders. Submission remains
outside this module and is still protected by the Kalshi write latch.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from ..common import OrderbookLevel, estimate_kalshi_fee_dollars, yes_asks_from_no_bids
from .kalshi_mapping import RouteCandidate


@dataclass(frozen=True)
class ExecutionConfig:
    edge_min: float = 0.05
    norm_edge_min: float = 0.25
    kelly_fraction: float = 0.50
    bankroll: float = 5000.0
    fee_rate: float = 0.07
    max_market_exposure_pct: float = 0.15
    cash_buffer_pct: float = 0.02
    max_visible_depth_participation: float = 0.25
    recent_volume_window_hours: float = 3.0
    max_recent_qualifying_volume_participation: float = 0.15
    cold_start_bankroll_cap: float = 0.01
    cold_start_visible_depth_participation: float = 0.15
    max_cumulative_qualifying_volume_share: float = 0.30
    normal_max_ioc_child_order_pct: float = 0.025
    completion_max_ioc_child_order_pct: float = 0.030
    burst_max_ioc_child_order_pct: float = 0.050
    passive_child_fraction_of_allowed: float = 0.25
    max_passive_child_order_pct: float = 0.010
    min_child_order_dollars: float = 25.0
    route_price_tie_threshold_ticks: int = 1
    monitor_start_hours_before_tip: float = 24.0
    trade_start_hours_before_tip: float = 17.0
    no_new_entry_hours_before_tip: float = 8.0
    no_orders_hours_before_tip: float = 4.0
    poll_t24_to_t17_s: float = 15.0 * 60.0
    poll_t17_to_t12_s: float = 5.0 * 60.0
    poll_t12_to_t8_s: float = 2.0 * 60.0
    poll_t8_to_t4_s: float = 5.0 * 60.0
    poll_t4_to_tip_s: float = 15.0 * 60.0
    normal_min_time_between_ioc_sweeps_s: float = 60.0
    burst_min_time_between_ioc_sweeps_s: float = 15.0
    min_time_between_passive_updates_s: float = 5.0 * 60.0
    passive_enabled: bool = True
    min_spread_for_passive_ticks: int = 2
    passive_timeout_t17_to_t12_s: float = 15.0 * 60.0
    passive_timeout_t12_to_t8_s: float = 10.0 * 60.0
    max_upward_reprices_per_passive_episode: int = 2
    passive_episode_cooldown_s: float = 10.0 * 60.0
    burst_enabled: bool = True
    burst_depth_multiplier_trigger: float = 2.0
    burst_min_visible_depth_pct_bankroll: float = 0.03
    max_burst_orders_per_5min: int = 3
    max_burst_total_per_5min_pct: float = 0.07
    urgency_multiplier_t17_to_t12: float = 1.0
    urgency_multiplier_t12_to_t8: float = 1.5
    urgency_multiplier_t8_to_t4: float = 2.0
    max_effective_remaining_opportunities: int = 12
    order_reject_brake_threshold_per_hour: int = 5
    api_error_brake_threshold_per_10min: int = 10

    def validate(self) -> None:
        assert 0.0 < self.edge_min < 0.5, self.edge_min
        assert 0.0 < self.norm_edge_min < 5.0, self.norm_edge_min
        assert 0.0 < self.kelly_fraction <= 1.0, self.kelly_fraction
        assert self.bankroll > 0, self.bankroll
        assert 0.0 < self.max_market_exposure_pct <= 1.0, self.max_market_exposure_pct
        assert 0.0 <= self.cash_buffer_pct < 1.0, self.cash_buffer_pct
        assert 0.0 < self.max_visible_depth_participation <= 1.0
        assert 0.0 <= self.max_recent_qualifying_volume_participation <= 1.0
        assert 0.0 <= self.cold_start_bankroll_cap <= 1.0
        assert 0.0 <= self.cold_start_visible_depth_participation <= 1.0
        assert 0.0 <= self.max_cumulative_qualifying_volume_share <= 1.0
        assert 0.0 < self.normal_max_ioc_child_order_pct <= 1.0
        assert 0.0 < self.completion_max_ioc_child_order_pct <= 1.0
        assert 0.0 < self.burst_max_ioc_child_order_pct <= 1.0
        assert 0.0 <= self.passive_child_fraction_of_allowed <= 1.0
        assert 0.0 < self.max_passive_child_order_pct <= 1.0
        assert self.min_child_order_dollars >= 0


@dataclass(frozen=True)
class RouteQuote:
    route: RouteCandidate
    ts_ms: int
    q_max_cents: int
    best_bid_cents: int
    best_bid_size: int
    best_ask_cents: int
    best_ask_size: int
    spread_ticks: int
    fillable_contracts_at_qmax: int
    raw_avg_price_cents: float
    all_in_avg_price_cents: float
    limit_price_cents: int
    visible_cost_dollars_at_qmax: float
    visible_depth_cap_dollars: float
    recent_qualifying_volume_dollars: float
    recent_qualifying_volume_cap_dollars: float
    cumulative_qualifying_volume_dollars: float
    cumulative_cap_remaining_dollars: float
    cold_start_cap_dollars: float
    route_capacity_dollars: float
    edge: float
    norm_edge: float
    eligible: bool
    reject_reason: str = ""


@dataclass(frozen=True)
class PlannedChildOrder:
    route_id: str
    market_ticker: str
    route_type: str
    action: str
    side: str
    count: int
    limit_price_cents: int
    max_cost_dollars: float
    expected_all_in_avg_price_cents: float
    q_max_cents: int
    order_mode: str = "normal_ioc"
    time_in_force: str = "immediate_or_cancel"
    post_only: bool = False
    expiration_ts: Optional[int] = None

    @property
    def yes_price_cents(self) -> Optional[int]:
        return self.limit_price_cents if self.side == "yes" else None

    @property
    def no_price_cents(self) -> Optional[int]:
        return self.limit_price_cents if self.side == "no" else None


@dataclass(frozen=True)
class ExecutionPlan:
    selected_team_id: str
    p_selected: float
    target_position_dollars: float
    remaining_position_dollars: float
    allowed_child_dollars: float
    q_max_cents: int
    route_quotes: Tuple[RouteQuote, ...]
    orders: Tuple[PlannedChildOrder, ...]
    decision: str
    reject_reason: str = ""
    timing_window: str = ""
    lead_hours: float = 0.0
    signal_class: str = ""
    binding_cap: str = ""
    route_capacity_sum_dollars: float = 0.0
    global_cumulative_remaining_dollars: float = 0.0
    cash_limited_mode: bool = False
    cash_priority_rule: str = ""
    cash_priority_rank: Optional[int] = None
    cash_priority_score: Optional[float] = None
    expected_log_growth_next_child: Optional[float] = None
    cash_priority_candidate_child_dollars: float = 0.0
    q_current_position: Optional[float] = None
    q_avg_after_child: Optional[float] = None
    skipped_due_to_cash: bool = False


def order_kwargs_from_plan(order: PlannedChildOrder, client_order_id: str) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "ticker": order.market_ticker,
        "action": order.action,
        "side": order.side,
        "count": order.count,
        "order_type": "limit",
        "client_order_id": client_order_id,
    }
    if order.time_in_force:
        body["time_in_force"] = order.time_in_force
    if order.side == "yes":
        body["yes_price_cents"] = order.limit_price_cents
    elif order.side == "no":
        body["no_price_cents"] = order.limit_price_cents
    else:
        raise ValueError(f"unsupported planned order side {order.side}")
    if order.post_only:
        body["post_only"] = True
    if order.expiration_ts is not None:
        body["expiration_ts"] = order.expiration_ts
    return body


def submit_planned_child_order(
    client: Any,
    order: PlannedChildOrder,
    *,
    client_order_id: str,
) -> Dict[str, Any]:
    """Submit one planned child order through a Kalshi-like client.

    The real client still enforces `KALSHI_TRADING_ENABLED=true` before any
    write request can leave the process.
    """
    return client.create_order(**order_kwargs_from_plan(order, client_order_id))


def max_all_in_price_cents(p_selected: float, cfg: ExecutionConfig) -> int:
    q = min(p_selected - cfg.edge_min, p_selected / (1.0 + cfg.norm_edge_min))
    return max(0, int(math.floor(q * 100.0)))


def no_asks_from_yes_bids(yes_bids: Sequence[Sequence[Any]]) -> Tuple[OrderbookLevel, ...]:
    levels: List[OrderbookLevel] = []
    for row in yes_bids or []:
        if not row or len(row) < 2:
            continue
        yes_price = _price_to_cents(row[0])
        size = int(float(row[1]))
        if size <= 0 or not (1 <= yes_price <= 99):
            continue
        levels.append(OrderbookLevel(price_cents=100 - yes_price, size=size))
    levels.sort(key=lambda level: level.price_cents)
    return _collapse_levels(levels)


def executable_levels_for_route(
    route: RouteCandidate,
    orderbook_payload: Mapping[str, Any],
) -> Tuple[OrderbookLevel, ...]:
    yes_bids, no_bids = _extract_bid_lists(orderbook_payload)
    if route.route_type == "BUY_YES_SELECTED":
        return yes_asks_from_no_bids(_normalize_book_rows(no_bids))
    if route.route_type == "BUY_NO_OPPONENT":
        return no_asks_from_yes_bids(yes_bids)
    raise ValueError(f"unknown route_type {route.route_type}")


def bid_levels_for_route(
    route: RouteCandidate,
    orderbook_payload: Mapping[str, Any],
) -> Tuple[OrderbookLevel, ...]:
    yes_bids, no_bids = _extract_bid_lists(orderbook_payload)
    rows = yes_bids if route.route_type == "BUY_YES_SELECTED" else no_bids
    levels: List[OrderbookLevel] = []
    for row in rows or []:
        if not row or len(row) < 2:
            continue
        price = _price_to_cents(row[0])
        size = int(float(row[1]))
        if size <= 0 or not (1 <= price <= 99):
            continue
        levels.append(OrderbookLevel(price_cents=price, size=size))
    levels.sort(key=lambda level: level.price_cents, reverse=True)
    return tuple(levels)


def evaluate_route_quote(
    route: RouteCandidate,
    orderbook_payload: Mapping[str, Any],
    *,
    p_selected: float,
    cfg: ExecutionConfig,
    ts_ms: Optional[int] = None,
) -> RouteQuote:
    cfg.validate()
    q_max_cents = max_all_in_price_cents(p_selected, cfg)
    ts = ts_ms if ts_ms is not None else int(time.time() * 1000)

    if not route.confirmed:
        return _rejected_quote(route, ts, q_max_cents, "unconfirmed_route")
    if q_max_cents < 1:
        return _rejected_quote(route, ts, q_max_cents, "q_max_below_1c")

    bid_levels = bid_levels_for_route(route, orderbook_payload)
    best_bid_cents = bid_levels[0].price_cents if bid_levels else 0
    best_bid_size = bid_levels[0].size if bid_levels else 0
    levels = executable_levels_for_route(route, orderbook_payload)
    if not levels:
        return _rejected_quote(
            route,
            ts,
            q_max_cents,
            "empty_executable_book",
            best_bid_cents=best_bid_cents,
            best_bid_size=best_bid_size,
        )

    best = levels[0]
    spread_ticks = max(0, best.price_cents - best_bid_cents) if best_bid_cents else 0
    prefix = _fillable_prefix_under_all_in_cap(levels, q_max_cents, cfg)
    if prefix is None:
        fee_cents = _fee_cents(best.size, best.price_cents, cfg)
        all_in = best.price_cents + (fee_cents / max(1, best.size))
        return RouteQuote(
            route=route,
            ts_ms=ts,
            q_max_cents=q_max_cents,
            best_bid_cents=best_bid_cents,
            best_bid_size=best_bid_size,
            best_ask_cents=best.price_cents,
            best_ask_size=best.size,
            spread_ticks=spread_ticks,
            fillable_contracts_at_qmax=0,
            raw_avg_price_cents=0.0,
            all_in_avg_price_cents=all_in,
            limit_price_cents=0,
            visible_cost_dollars_at_qmax=0.0,
            visible_depth_cap_dollars=0.0,
            recent_qualifying_volume_dollars=0.0,
            recent_qualifying_volume_cap_dollars=0.0,
            cumulative_qualifying_volume_dollars=0.0,
            cumulative_cap_remaining_dollars=0.0,
            cold_start_cap_dollars=0.0,
            route_capacity_dollars=0.0,
            edge=p_selected - (all_in / 100.0),
            norm_edge=((p_selected - (all_in / 100.0)) / (all_in / 100.0)) if all_in > 0 else 0.0,
            eligible=False,
            reject_reason=f"best_all_in {all_in:.2f}c > q_max {q_max_cents}c",
        )

    count, raw_cost_cents, limit_price_cents = prefix
    raw_avg = raw_cost_cents / count
    fee_cents = _fee_cents(count, round(raw_avg), cfg)
    all_in_avg = (raw_cost_cents + fee_cents) / count
    q_all_in = all_in_avg / 100.0
    edge = p_selected - q_all_in
    norm_edge = edge / q_all_in if q_all_in > 0 else 0.0
    visible_cost = raw_cost_cents / 100.0
    visible_depth_cap = cfg.max_visible_depth_participation * visible_cost
    cold_start_cap = min(
        cfg.cold_start_bankroll_cap * cfg.bankroll,
        cfg.cold_start_visible_depth_participation * visible_cost,
    )
    eligible = edge >= cfg.edge_min and norm_edge >= cfg.norm_edge_min
    return RouteQuote(
        route=route,
        ts_ms=ts,
        q_max_cents=q_max_cents,
        best_bid_cents=best_bid_cents,
        best_bid_size=best_bid_size,
        best_ask_cents=best.price_cents,
        best_ask_size=best.size,
        spread_ticks=spread_ticks,
        fillable_contracts_at_qmax=count,
        raw_avg_price_cents=raw_avg,
        all_in_avg_price_cents=all_in_avg,
        limit_price_cents=limit_price_cents,
        visible_cost_dollars_at_qmax=visible_cost,
        visible_depth_cap_dollars=visible_depth_cap,
        recent_qualifying_volume_dollars=0.0,
        recent_qualifying_volume_cap_dollars=0.0,
        cumulative_qualifying_volume_dollars=0.0,
        cumulative_cap_remaining_dollars=cold_start_cap,
        cold_start_cap_dollars=cold_start_cap,
        route_capacity_dollars=min(visible_depth_cap, cold_start_cap),
        edge=edge,
        norm_edge=norm_edge,
        eligible=eligible,
        reject_reason="" if eligible else "edge_threshold_not_met",
    )


def plan_normal_ioc_orders(
    *,
    selected_team_id: str,
    p_selected: float,
    route_quotes: Sequence[RouteQuote],
    cfg: ExecutionConfig,
    available_cash_dollars: Optional[float] = None,
    filled_position_dollars: float = 0.0,
    reserved_position_dollars: float = 0.0,
) -> ExecutionPlan:
    cfg.validate()
    q_max_cents = max_all_in_price_cents(p_selected, cfg)
    eligible = [quote for quote in route_quotes if quote.eligible and quote.route_capacity_dollars > 0]
    if not eligible:
        return ExecutionPlan(
            selected_team_id=selected_team_id,
            p_selected=p_selected,
            target_position_dollars=0.0,
            remaining_position_dollars=0.0,
            allowed_child_dollars=0.0,
            q_max_cents=q_max_cents,
            route_quotes=tuple(route_quotes),
            orders=(),
            decision="no_trade",
            reject_reason="no_eligible_routes",
        )

    best = min(eligible, key=lambda quote: quote.all_in_avg_price_cents)
    target = target_position_dollars(
        p_selected=p_selected,
        q_all_in=best.all_in_avg_price_cents / 100.0,
        cfg=cfg,
        available_cash_dollars=available_cash_dollars,
    )
    remaining = max(0.0, target - filled_position_dollars - reserved_position_dollars)
    if remaining < cfg.min_child_order_dollars:
        return ExecutionPlan(
            selected_team_id=selected_team_id,
            p_selected=p_selected,
            target_position_dollars=target,
            remaining_position_dollars=remaining,
            allowed_child_dollars=0.0,
            q_max_cents=q_max_cents,
            route_quotes=tuple(route_quotes),
            orders=(),
            decision="wait",
            reject_reason="remaining_below_min_child",
        )

    route_capacity_sum = sum(quote.route_capacity_dollars for quote in eligible)
    child_cap = cfg.normal_max_ioc_child_order_pct * cfg.bankroll
    allowed_child = min(remaining, route_capacity_sum, child_cap)
    if allowed_child < cfg.min_child_order_dollars:
        return ExecutionPlan(
            selected_team_id=selected_team_id,
            p_selected=p_selected,
            target_position_dollars=target,
            remaining_position_dollars=remaining,
            allowed_child_dollars=allowed_child,
            q_max_cents=q_max_cents,
            route_quotes=tuple(route_quotes),
            orders=(),
            decision="wait",
            reject_reason="allowed_child_below_min",
        )

    orders = _allocate_child_orders(eligible, allowed_child, cfg)
    return ExecutionPlan(
        selected_team_id=selected_team_id,
        p_selected=p_selected,
        target_position_dollars=target,
        remaining_position_dollars=remaining,
        allowed_child_dollars=allowed_child,
        q_max_cents=q_max_cents,
        route_quotes=tuple(route_quotes),
        orders=tuple(orders),
        decision="normal_ioc" if orders else "wait",
        reject_reason="" if orders else "no_contracts_after_rounding",
    )


def target_position_dollars(
    *,
    p_selected: float,
    q_all_in: float,
    cfg: ExecutionConfig,
    available_cash_dollars: Optional[float],
) -> float:
    if q_all_in <= 0.0 or q_all_in >= p_selected:
        return 0.0
    f_star = (p_selected - q_all_in) / (1.0 - q_all_in)
    half_kelly = cfg.kelly_fraction * f_star * cfg.bankroll
    market_cap = cfg.max_market_exposure_pct * cfg.bankroll
    if available_cash_dollars is None:
        cash_cap = market_cap
    else:
        cash_cap = max(0.0, available_cash_dollars - (cfg.cash_buffer_pct * cfg.bankroll))
    return max(0.0, min(half_kelly, market_cap, cash_cap))


def _allocate_child_orders(
    eligible: Sequence[RouteQuote],
    allowed_child_dollars: float,
    cfg: ExecutionConfig,
) -> List[PlannedChildOrder]:
    ordered = sorted(eligible, key=lambda quote: quote.all_in_avg_price_cents)
    tied = [
        quote for quote in ordered
        if quote.all_in_avg_price_cents - ordered[0].all_in_avg_price_cents
        <= cfg.route_price_tie_threshold_ticks
    ]
    if len(tied) > 1:
        capacity_sum = sum(quote.route_capacity_dollars for quote in tied)
        allocations = [
            (quote, allowed_child_dollars * (quote.route_capacity_dollars / capacity_sum))
            for quote in tied
            if capacity_sum > 0
        ]
    else:
        remaining = allowed_child_dollars
        allocations = []
        for quote in ordered:
            if remaining <= 0:
                break
            dollars = min(remaining, quote.route_capacity_dollars)
            allocations.append((quote, dollars))
            remaining -= dollars

    orders: List[PlannedChildOrder] = []
    for quote, dollars in allocations:
        if dollars < cfg.min_child_order_dollars:
            continue
        price_dollars = quote.limit_price_cents / 100.0
        count = int(math.floor(dollars / price_dollars)) if price_dollars > 0 else 0
        count = min(count, quote.fillable_contracts_at_qmax)
        if count <= 0:
            continue
        max_cost = count * price_dollars
        orders.append(
            PlannedChildOrder(
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
            )
        )
    return orders


def _extract_bid_lists(payload: Mapping[str, Any]) -> Tuple[List[List[Any]], List[List[Any]]]:
    book = payload.get("orderbook") or {}
    if book:
        return list(book.get("yes") or []), list(book.get("no") or [])
    book_fp = payload.get("orderbook_fp") or {}
    return list(book_fp.get("yes") or book_fp.get("yes_dollars") or []), list(
        book_fp.get("no") or book_fp.get("no_dollars") or []
    )


def _normalize_book_rows(rows: Sequence[Sequence[Any]]) -> List[List[int]]:
    out: List[List[int]] = []
    for row in rows or []:
        if not row or len(row) < 2:
            continue
        out.append([_price_to_cents(row[0]), int(float(row[1]))])
    return out


def _price_to_cents(value: Any) -> int:
    if isinstance(value, str) and "." in value:
        return int(round(float(value) * 100))
    numeric = float(value)
    if 0 < numeric <= 1:
        return int(round(numeric * 100))
    return int(round(numeric))


def _collapse_levels(levels: Sequence[OrderbookLevel]) -> Tuple[OrderbookLevel, ...]:
    collapsed: List[OrderbookLevel] = []
    for level in sorted(levels, key=lambda item: item.price_cents):
        if collapsed and collapsed[-1].price_cents == level.price_cents:
            collapsed[-1] = OrderbookLevel(level.price_cents, collapsed[-1].size + level.size)
        else:
            collapsed.append(level)
    return tuple(collapsed)


def _fee_cents(count: int, price_cents: int, cfg: ExecutionConfig) -> int:
    return int(round(estimate_kalshi_fee_dollars(count, price_cents, cfg.fee_rate) * 100))


def _fillable_prefix_under_all_in_cap(
    levels: Sequence[OrderbookLevel],
    q_max_cents: int,
    cfg: ExecutionConfig,
) -> Optional[Tuple[int, int, int]]:
    count = 0
    raw_cost_cents = 0
    limit_price_cents = 0
    best: Optional[Tuple[int, int, int]] = None
    for level in levels:
        count += level.size
        raw_cost_cents += level.size * level.price_cents
        limit_price_cents = level.price_cents
        raw_avg = raw_cost_cents / count
        fee_cents = _fee_cents(count, round(raw_avg), cfg)
        all_in_avg = (raw_cost_cents + fee_cents) / count
        if all_in_avg <= q_max_cents:
            best = (count, raw_cost_cents, limit_price_cents)
        else:
            break
    return best


def _rejected_quote(
    route: RouteCandidate,
    ts_ms: int,
    q_max_cents: int,
    reason: str,
    *,
    best_bid_cents: int = 0,
    best_bid_size: int = 0,
) -> RouteQuote:
    return RouteQuote(
        route=route,
        ts_ms=ts_ms,
        q_max_cents=q_max_cents,
        best_bid_cents=best_bid_cents,
        best_bid_size=best_bid_size,
        best_ask_cents=0,
        best_ask_size=0,
        spread_ticks=0,
        fillable_contracts_at_qmax=0,
        raw_avg_price_cents=0.0,
        all_in_avg_price_cents=0.0,
        limit_price_cents=0,
        visible_cost_dollars_at_qmax=0.0,
        visible_depth_cap_dollars=0.0,
        recent_qualifying_volume_dollars=0.0,
        recent_qualifying_volume_cap_dollars=0.0,
        cumulative_qualifying_volume_dollars=0.0,
        cumulative_cap_remaining_dollars=0.0,
        cold_start_cap_dollars=0.0,
        route_capacity_dollars=0.0,
        edge=0.0,
        norm_edge=0.0,
        eligible=False,
        reject_reason=reason,
    )
