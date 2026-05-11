"""Kalshi reconciliation helpers for canonical-route execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .kalshi_mapping import RouteCandidate

OPEN_ORDER_STATUSES = {
    "open",
    "resting",
    "pending",
    "unfilled",
    "partially_filled",
    "partially-filled",
    "active",
}


@dataclass(frozen=True)
class RoutePosition:
    route_id: str
    market_ticker: str
    route_type: str
    side: str
    filled_contracts: int = 0
    filled_cost_dollars: float = 0.0
    raw_fill_count: int = 0
    raw_position_contracts: Optional[int] = None
    raw_fills: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    raw_positions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "market_ticker": self.market_ticker,
            "route_type": self.route_type,
            "side": self.side,
            "filled_contracts": self.filled_contracts,
            "filled_cost_dollars": self.filled_cost_dollars,
            "raw_fill_count": self.raw_fill_count,
            "raw_position_contracts": self.raw_position_contracts,
            "raw_fills": list(self.raw_fills),
            "raw_positions": list(self.raw_positions),
        }


@dataclass(frozen=True)
class ExchangeReconciliation:
    routes: Sequence[RoutePosition]
    filled_contracts_by_route: Mapping[str, int]
    filled_cost_by_route: Mapping[str, float]
    filled_cost_dollars: float
    raw_open_orders: Sequence[Mapping[str, Any]] = field(default_factory=tuple)

    def mismatch_dollars(self, local_cost_by_route: Mapping[str, float]) -> float:
        all_route_ids = set(self.filled_cost_by_route) | set(local_cost_by_route)
        return sum(
            abs(float(self.filled_cost_by_route.get(route_id, 0.0)) - float(local_cost_by_route.get(route_id, 0.0)))
            for route_id in all_route_ids
        )

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "filled_cost_dollars": self.filled_cost_dollars,
            "filled_contracts_by_route": dict(self.filled_contracts_by_route),
            "filled_cost_by_route": dict(self.filled_cost_by_route),
            "routes": [route.to_log_payload() for route in self.routes],
            "raw_open_orders": list(self.raw_open_orders),
        }


@dataclass(frozen=True)
class StartupOpenOrderRecovery:
    cancelled_orders: Sequence[Mapping[str, Any]]
    unknown_open_orders: Sequence[Mapping[str, Any]]
    cancel_errors: Sequence[Mapping[str, Any]]

    @property
    def blocked(self) -> bool:
        return bool(self.unknown_open_orders or self.cancel_errors)

    def to_log_payload(self) -> dict[str, Any]:
        return {
            "cancelled_orders": list(self.cancelled_orders),
            "unknown_open_orders": list(self.unknown_open_orders),
            "cancel_errors": list(self.cancel_errors),
            "blocked": self.blocked,
        }


def reconcile_exchange_routes(
    client: Any,
    routes: Sequence[RouteCandidate],
    *,
    fills_limit: int = 1000,
) -> ExchangeReconciliation:
    route_positions: list[RoutePosition] = []
    filled_contracts_by_route: dict[str, int] = {}
    filled_cost_by_route: dict[str, float] = {}
    raw_open_orders: list[Mapping[str, Any]] = []

    for route in routes:
        fills = _safe_list(client.get_fills(ticker=route.market_ticker, limit=fills_limit))
        positions = _safe_list(client.get_positions(ticker=route.market_ticker, limit=100))
        open_orders = _open_orders_for_ticker(client, route.market_ticker)
        raw_open_orders.extend(open_orders)

        contracts = 0
        cost = 0.0
        for fill in fills:
            delta = _fill_contract_delta(fill, route.side)
            if delta == 0:
                continue
            price_cents = _fill_price_cents(fill, route.side)
            if price_cents <= 0:
                continue
            contracts += delta
            cost += delta * price_cents / 100.0
        contracts = max(0, contracts)
        cost = max(0.0, cost)
        pos_contracts = _position_contracts_for_route(positions, route.side)

        route_positions.append(
            RoutePosition(
                route_id=route.route_id,
                market_ticker=route.market_ticker,
                route_type=route.route_type,
                side=route.side,
                filled_contracts=contracts,
                filled_cost_dollars=cost,
                raw_fill_count=len(fills),
                raw_position_contracts=pos_contracts,
                raw_fills=tuple(fills),
                raw_positions=tuple(positions),
            )
        )
        filled_contracts_by_route[route.route_id] = contracts
        filled_cost_by_route[route.route_id] = cost

    return ExchangeReconciliation(
        routes=tuple(route_positions),
        filled_contracts_by_route=filled_contracts_by_route,
        filled_cost_by_route=filled_cost_by_route,
        filled_cost_dollars=sum(filled_cost_by_route.values()),
        raw_open_orders=tuple(raw_open_orders),
    )


def recover_open_route_orders(
    client: Any,
    routes: Sequence[RouteCandidate],
    *,
    client_order_prefix: str,
) -> StartupOpenOrderRecovery:
    cancelled: list[Mapping[str, Any]] = []
    unknown: list[Mapping[str, Any]] = []
    errors: list[Mapping[str, Any]] = []

    for route in routes:
        for order in _open_orders_for_ticker(client, route.market_ticker):
            client_order_id = str(order.get("client_order_id") or "").strip()
            order_id = str(order.get("order_id") or order.get("id") or "").strip()
            record = {
                "route_id": route.route_id,
                "market_ticker": route.market_ticker,
                "order_id": order_id,
                "client_order_id": client_order_id,
                "status": order.get("status"),
                "raw_order": order,
            }
            if not client_order_id.startswith(client_order_prefix):
                unknown.append(record)
                continue
            if not order_id:
                errors.append({**record, "error": "missing_order_id"})
                continue
            try:
                resp = client.cancel_order(order_id)
            except Exception as exc:  # pragma: no cover - exercised against live API
                errors.append({**record, "error": repr(exc)})
            else:
                cancelled.append({**record, "cancel_response": resp})

    return StartupOpenOrderRecovery(
        cancelled_orders=tuple(cancelled),
        unknown_open_orders=tuple(unknown),
        cancel_errors=tuple(errors),
    )


def _open_orders_for_ticker(client: Any, ticker: str) -> list[Mapping[str, Any]]:
    merged: dict[str, Mapping[str, Any]] = {}
    for kwargs in ({"status": "open"}, {}):
        try:
            orders = _safe_list(client.get_orders(ticker=ticker, limit=100, **kwargs))
        except Exception:
            continue
        for idx, order in enumerate(orders):
            key = str(order.get("order_id") or order.get("id") or order.get("client_order_id") or idx)
            merged[key] = order
    return [order for order in merged.values() if _is_open_order(order)]


def _is_open_order(order: Mapping[str, Any]) -> bool:
    status = str(order.get("status") or "").strip().lower()
    if not status:
        return True
    return status in OPEN_ORDER_STATUSES


def _fill_contract_delta(fill: Mapping[str, Any], route_side: str) -> int:
    action = str(fill.get("action") or fill.get("trade_type") or "buy").strip().lower()
    side = str(fill.get("side") or fill.get("taker_side") or route_side).strip().lower()
    if side and side not in {route_side, "both"}:
        return 0
    count = _intish(
        fill.get("count")
        or fill.get("count_fp")
        or fill.get("fill_count")
        or fill.get("filled_count")
        or 0
    )
    if count <= 0:
        return 0
    return -count if action == "sell" else count


def _fill_price_cents(fill: Mapping[str, Any], route_side: str) -> int:
    if route_side == "yes":
        return _price_cents(fill, ("yes_price", "yes_price_cents"), ("yes_price_dollars",))
    return _price_cents(fill, ("no_price", "no_price_cents"), ("no_price_dollars",))


def _price_cents(
    payload: Mapping[str, Any],
    cents_keys: Iterable[str],
    dollar_keys: Iterable[str],
) -> int:
    for key in cents_keys:
        if payload.get(key) is not None:
            value = _floatish(payload.get(key))
            if 0.0 < value <= 1.0:
                return int(round(value * 100.0))
            return int(round(value))
    for key in dollar_keys:
        if payload.get(key) is not None:
            return int(round(_floatish(payload.get(key)) * 100.0))
    if payload.get("price") is not None:
        value = _floatish(payload.get("price"))
        if 0.0 < value <= 1.0:
            return int(round(value * 100.0))
        return int(round(value))
    return 0


def _position_contracts_for_route(
    positions: Sequence[Mapping[str, Any]],
    route_side: str,
) -> Optional[int]:
    if not positions:
        return None
    total = 0
    saw_specific = False
    for pos in positions:
        if route_side == "yes":
            for key in ("yes_count", "yes_position", "yes_contracts"):
                if pos.get(key) is not None:
                    total += max(0, _intish(pos.get(key)))
                    saw_specific = True
            if not saw_specific and pos.get("position") is not None:
                total += max(0, _intish(pos.get("position")))
        else:
            for key in ("no_count", "no_position", "no_contracts"):
                if pos.get(key) is not None:
                    total += max(0, _intish(pos.get(key)))
                    saw_specific = True
            if not saw_specific and pos.get("position") is not None:
                total += max(0, -_intish(pos.get("position")))
    return total


def _safe_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("orders", "fills", "market_positions", "positions"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, Mapping)]
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _intish(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0


def _floatish(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
