"""File-backed cross-game cash coordinator for canonical route loops."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .cash_priority import CashPriorityTicket, allocate_cash_greedily
from .execution import ExecutionConfig, ExecutionPlan, PlannedChildOrder, RouteQuote


CASH_COORDINATOR_SCHEMA_VERSION = "cash_coordinator_v1"


@dataclass(frozen=True)
class CashCoordinationResult:
    plan: ExecutionPlan
    payload: Mapping[str, Any]


def coordinate_cash_for_plan(
    *,
    game_id: str,
    plan: ExecutionPlan,
    cfg: ExecutionConfig,
    coordinator_dir: Path,
    available_cash_after_buffer: float,
    filled_position_dollars: float,
    reserved_position_dollars: float,
    current_position_q: Optional[float],
    wait_s: float = 1.0,
    freshness_s: float = 120.0,
) -> CashCoordinationResult:
    """Publish this game's candidate and apply the global cash allocation."""
    coordinator_dir = Path(coordinator_dir)
    candidate_dir = coordinator_dir / "candidates"
    lock_dir = coordinator_dir / "locks"
    now_s = time.time()

    if not plan.orders:
        clear_cash_candidate(game_id=game_id, coordinator_dir=coordinator_dir)
        return CashCoordinationResult(plan=plan, payload={"coordinated": False, "reason": "no_orders"})

    best = _best_quote(plan.route_quotes)
    if best is None:
        clear_cash_candidate(game_id=game_id, coordinator_dir=coordinator_dir)
        return CashCoordinationResult(plan=plan, payload={"coordinated": False, "reason": "no_eligible_quote"})

    planned_child_cost = sum(order.max_cost_dollars for order in plan.orders)
    candidate = _candidate_payload(
        game_id=game_id,
        plan=plan,
        cfg=cfg,
        best=best,
        planned_child_cost=planned_child_cost,
        available_cash_after_buffer=available_cash_after_buffer,
        filled_position_dollars=filled_position_dollars,
        reserved_position_dollars=reserved_position_dollars,
        current_position_q=current_position_q,
        now_s=now_s,
    )
    with _CoordinatorLock(lock_dir):
        candidate_dir.mkdir(parents=True, exist_ok=True)
        _write_json(candidate_dir / f"{_safe_name(game_id)}.json", candidate)

    if wait_s > 0:
        time.sleep(wait_s)

    with _CoordinatorLock(lock_dir):
        candidates = _load_fresh_candidates(candidate_dir, now_s=time.time(), freshness_s=freshness_s)
        tickets = [_ticket_from_candidate(item) for item in candidates]
        allocations = allocate_cash_greedily(
            tickets,
            available_cash_after_buffer=available_cash_after_buffer,
        )
        allocation_by_id = {item.ticket_id: item for item in allocations}
        allocation = allocation_by_id.get(game_id)
        allocation_payload = {
            "schema_version": CASH_COORDINATOR_SCHEMA_VERSION,
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "available_cash_after_buffer": available_cash_after_buffer,
            "freshness_s": freshness_s,
            "candidate_count": len(candidates),
            "allocations": [item.__dict__ for item in allocations],
        }
        _write_json(coordinator_dir / "latest_allocation.json", allocation_payload)

    allocated = 0.0 if allocation is None else float(allocation.allocated_child_size_dollars)
    adjusted_plan = _apply_allocation_to_plan(
        plan=plan,
        allocated_child_dollars=allocated,
        cfg=cfg,
        allocation=allocation,
    )
    payload = {
        "coordinated": True,
        "game_id": game_id,
        "candidate_child_dollars": planned_child_cost,
        "allocated_child_dollars": allocated,
        "available_cash_after_buffer": available_cash_after_buffer,
        "candidate_count": len(candidates),
        "cash_priority_rank": getattr(allocation, "cash_priority_rank", None),
        "cash_priority_score": getattr(allocation, "priority_score", None),
        "expected_log_growth_next_child": getattr(allocation, "expected_log_growth_next_child", None),
        "skipped_due_to_cash": allocation is None or bool(getattr(allocation, "skipped_due_to_cash", False)),
    }
    return CashCoordinationResult(plan=adjusted_plan, payload=payload)


def clear_cash_candidate(*, game_id: str, coordinator_dir: Path) -> None:
    path = Path(coordinator_dir) / "candidates" / f"{_safe_name(game_id)}.json"
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _candidate_payload(
    *,
    game_id: str,
    plan: ExecutionPlan,
    cfg: ExecutionConfig,
    best: RouteQuote,
    planned_child_cost: float,
    available_cash_after_buffer: float,
    filled_position_dollars: float,
    reserved_position_dollars: float,
    current_position_q: Optional[float],
    now_s: float,
) -> dict[str, Any]:
    return {
        "schema_version": CASH_COORDINATOR_SCHEMA_VERSION,
        "ticket_id": str(game_id),
        "updated_ts_s": now_s,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligible": bool(plan.orders),
        "model_prob": plan.p_selected,
        "q_exec_all_in": best.all_in_avg_price_cents / 100.0,
        "q_max": plan.q_max_cents / 100.0,
        "bankroll_for_sizing": cfg.bankroll,
        "allowed_to_try_now": plan.allowed_child_dollars,
        "child_cap": planned_child_cost,
        "filled_position_cost": filled_position_dollars,
        "reserved_open_order_cost": reserved_position_dollars,
        "current_position_q": current_position_q,
        "min_child_order_dollars": cfg.min_child_order_dollars,
        "absolute_edge": best.edge,
        "normalized_edge": best.norm_edge,
        "first_qualified_ts_s": None,
        "executable_liquidity_dollars": plan.route_capacity_sum_dollars,
        "route_slippage": max(0.0, best.all_in_avg_price_cents - best.best_ask_cents),
        "available_cash_after_buffer_seen": available_cash_after_buffer,
    }


def _ticket_from_candidate(item: Mapping[str, Any]) -> CashPriorityTicket:
    return CashPriorityTicket(
        ticket_id=str(item.get("ticket_id") or ""),
        eligible=bool(item.get("eligible")),
        model_prob=_float(item.get("model_prob")),
        q_exec_all_in=_float(item.get("q_exec_all_in")),
        q_max=_float(item.get("q_max")),
        bankroll_for_sizing=max(_float(item.get("bankroll_for_sizing")), 1e-9),
        allowed_to_try_now=max(0.0, _float(item.get("allowed_to_try_now"))),
        child_cap=max(0.0, _float(item.get("child_cap"))),
        filled_position_cost=max(0.0, _float(item.get("filled_position_cost"))),
        reserved_open_order_cost=max(0.0, _float(item.get("reserved_open_order_cost"))),
        current_position_q=_optional_float(item.get("current_position_q")),
        min_child_order_dollars=max(0.0, _float(item.get("min_child_order_dollars"))),
        absolute_edge=_float(item.get("absolute_edge")),
        normalized_edge=_float(item.get("normalized_edge")),
        first_qualified_ts_s=_optional_float(item.get("first_qualified_ts_s")),
        executable_liquidity_dollars=max(0.0, _float(item.get("executable_liquidity_dollars"))),
        route_slippage=max(0.0, _float(item.get("route_slippage"))),
    )


def _apply_allocation_to_plan(
    *,
    plan: ExecutionPlan,
    allocated_child_dollars: float,
    cfg: ExecutionConfig,
    allocation: Any,
) -> ExecutionPlan:
    allocated = max(0.0, float(allocated_child_dollars))
    original_cost = sum(order.max_cost_dollars for order in plan.orders)
    if allocated + 1e-9 >= original_cost:
        return dc_replace(
            plan,
            cash_limited_mode=True,
            cash_priority_rule="marginal_expected_log_growth_per_dollar",
            cash_priority_rank=getattr(allocation, "cash_priority_rank", None),
            cash_priority_score=getattr(allocation, "priority_score", None),
            expected_log_growth_next_child=getattr(allocation, "expected_log_growth_next_child", None),
            cash_priority_candidate_child_dollars=getattr(allocation, "candidate_child_size_dollars", 0.0),
            skipped_due_to_cash=False,
        )
    if allocated < cfg.min_child_order_dollars:
        return dc_replace(
            plan,
            orders=(),
            allowed_child_dollars=0.0,
            decision="wait",
            reject_reason="cash_priority_no_allocation",
            cash_limited_mode=True,
            cash_priority_rule="marginal_expected_log_growth_per_dollar",
            cash_priority_rank=getattr(allocation, "cash_priority_rank", None),
            cash_priority_score=getattr(allocation, "priority_score", None),
            expected_log_growth_next_child=getattr(allocation, "expected_log_growth_next_child", None),
            cash_priority_candidate_child_dollars=getattr(allocation, "candidate_child_size_dollars", 0.0),
            skipped_due_to_cash=True,
        )
    orders = _trim_orders_to_cash(plan.orders, allocated, cfg)
    total = sum(order.max_cost_dollars for order in orders)
    return dc_replace(
        plan,
        orders=orders,
        allowed_child_dollars=total,
        decision=plan.decision if orders else "wait",
        reject_reason="" if orders else "cash_priority_allocation_below_contract_rounding",
        cash_limited_mode=True,
        cash_priority_rule="marginal_expected_log_growth_per_dollar",
        cash_priority_rank=getattr(allocation, "cash_priority_rank", None),
        cash_priority_score=getattr(allocation, "priority_score", None),
        expected_log_growth_next_child=getattr(allocation, "expected_log_growth_next_child", None),
        cash_priority_candidate_child_dollars=getattr(allocation, "candidate_child_size_dollars", 0.0),
        skipped_due_to_cash=False if orders else True,
    )


def _trim_orders_to_cash(
    orders: Sequence[PlannedChildOrder],
    allocated_child_dollars: float,
    cfg: ExecutionConfig,
) -> tuple[PlannedChildOrder, ...]:
    remaining = max(0.0, allocated_child_dollars)
    out: list[PlannedChildOrder] = []
    for order in orders:
        if remaining < cfg.min_child_order_dollars:
            break
        price = order.limit_price_cents / 100.0
        if price <= 0:
            continue
        dollars = min(order.max_cost_dollars, remaining)
        count = min(order.count, int(math.floor(dollars / price)))
        if count <= 0:
            continue
        max_cost = count * price
        if max_cost < cfg.min_child_order_dollars:
            continue
        out.append(dc_replace(order, count=count, max_cost_dollars=max_cost))
        remaining -= max_cost
    return tuple(out)


def _best_quote(route_quotes: Sequence[RouteQuote]) -> Optional[RouteQuote]:
    eligible = [quote for quote in route_quotes if quote.eligible and quote.route_capacity_dollars > 0]
    if not eligible:
        return None
    return min(eligible, key=lambda quote: quote.all_in_avg_price_cents)


def _load_fresh_candidates(candidate_dir: Path, *, now_s: float, freshness_s: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not candidate_dir.exists():
        return out
    for path in candidate_dir.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        age = now_s - _float(item.get("updated_ts_s"))
        if age <= freshness_s:
            out.append(item)
    return out


class _CoordinatorLock:
    def __init__(self, lock_dir: Path, *, timeout_s: float = 5.0) -> None:
        self.lock_dir = Path(lock_dir)
        self.timeout_s = timeout_s
        self.path = self.lock_dir / "cash_allocator.lock"
        self.acquired = False

    def __enter__(self) -> "_CoordinatorLock":
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self.path.mkdir()
                self.acquired = True
                (self.path / "owner.json").write_text(
                    json.dumps({"pid": os.getpid(), "ts_s": time.time()}) + "\n",
                    encoding="utf-8",
                )
                return self
            except FileExistsError:
                if time.time() > deadline:
                    _clear_stale_lock(self.path, stale_s=self.timeout_s)
                    deadline = time.time() + self.timeout_s
                time.sleep(0.05)

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.acquired:
            return
        try:
            (self.path / "owner.json").unlink(missing_ok=True)
            self.path.rmdir()
        finally:
            self.acquired = False


def _clear_stale_lock(path: Path, *, stale_s: float) -> None:
    owner = path / "owner.json"
    try:
        payload = json.loads(owner.read_text(encoding="utf-8"))
        if time.time() - _float(payload.get("ts_s")) < stale_s:
            return
    except Exception:
        pass
    try:
        owner.unlink(missing_ok=True)
        path.rmdir()
    except OSError:
        pass


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            pass
    return str(value)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    out = _float(value)
    return out if math.isfinite(out) else None


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
