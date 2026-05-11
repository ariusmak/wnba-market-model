"""Cash-scarcity priority math for canonical live tickets.

The functions in this module are side-effect-free. They rank otherwise
eligible child orders when available cash, rather than per-market sizing or
liquidity, is the scarce resource.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Iterable, Optional, Sequence, Tuple


PRIORITY_RULE = "marginal_expected_log_growth_per_dollar"
CLOSE_PRIORITY_RELATIVE_TOLERANCE = 0.05


@dataclass(frozen=True)
class MarginalLogGrowth:
    current_cost_dollars: float
    child_cost_dollars: float
    new_cost_dollars: float
    q_current_position: float
    q_child: float
    q_avg_after_child: float
    expected_log_growth_current: float
    expected_log_growth_after_child: float
    expected_log_growth_next_child: float
    priority_score: float


@dataclass(frozen=True)
class CashPriorityTicket:
    ticket_id: str
    eligible: bool
    model_prob: float
    q_exec_all_in: float
    q_max: float
    bankroll_for_sizing: float
    allowed_to_try_now: float
    child_cap: float
    filled_position_cost: float = 0.0
    reserved_open_order_cost: float = 0.0
    current_position_q: Optional[float] = None
    min_child_order_dollars: float = 25.0
    absolute_edge: float = 0.0
    normalized_edge: float = 0.0
    first_qualified_ts_s: Optional[float] = None
    executable_liquidity_dollars: float = 0.0
    route_slippage: float = 0.0

    @property
    def current_cost(self) -> float:
        return max(0.0, self.filled_position_cost + self.reserved_open_order_cost)


@dataclass(frozen=True)
class ScoredCashPriorityTicket:
    ticket: CashPriorityTicket
    candidate_child_size_dollars: float
    marginal: MarginalLogGrowth
    cash_priority_rank: Optional[int] = None

    @property
    def priority_score(self) -> float:
        return self.marginal.priority_score


@dataclass(frozen=True)
class CashPriorityAllocation:
    ticket_id: str
    cash_priority_rank: int
    priority_score: float
    expected_log_growth_next_child: float
    candidate_child_size_dollars: float
    allocated_child_size_dollars: float
    skipped_due_to_cash: bool
    skip_reason: str = ""


def expected_log_wealth(
    position_cost_dollars: float,
    *,
    p: float,
    q: float,
    bankroll_for_sizing: float,
) -> float:
    """Expected log growth for a YES-equivalent binary position.

    `q` must be the fee-adjusted executable/average price of the position.
    """
    cost = max(0.0, float(position_cost_dollars))
    if cost == 0.0:
        return 0.0
    _validate_probability(p, "p")
    _validate_probability(q, "q")
    if bankroll_for_sizing <= 0:
        raise ValueError("bankroll_for_sizing must be positive")

    f = cost / bankroll_for_sizing
    if f >= 1.0:
        return float("-inf")
    win_return_on_cost = (1.0 - q) / q
    return (
        p * math.log(1.0 + f * win_return_on_cost)
        + (1.0 - p) * math.log(1.0 - f)
    )


def average_q_after_child(
    *,
    current_cost_dollars: float,
    q_current_position: Optional[float],
    child_cost_dollars: float,
    q_child: float,
) -> float:
    """Return contract-weighted average q after adding a child order."""
    child_cost = max(0.0, float(child_cost_dollars))
    current_cost = max(0.0, float(current_cost_dollars))
    _validate_probability(q_child, "q_child")
    if child_cost == 0.0 and current_cost == 0.0:
        return q_child
    if current_cost == 0.0:
        return q_child
    if q_current_position is None:
        raise ValueError("q_current_position is required when current_cost_dollars > 0")
    _validate_probability(q_current_position, "q_current_position")

    current_contracts = current_cost / q_current_position
    child_contracts = child_cost / q_child if child_cost > 0.0 else 0.0
    total_contracts = current_contracts + child_contracts
    if total_contracts <= 0.0:
        return q_child
    return (current_cost + child_cost) / total_contracts


def marginal_expected_log_growth_per_dollar(
    *,
    p: float,
    q_child: float,
    bankroll_for_sizing: float,
    current_cost_dollars: float,
    child_cost_dollars: float,
    q_current_position: Optional[float] = None,
    q_avg_after_child: Optional[float] = None,
) -> MarginalLogGrowth:
    """Exact incremental expected log-growth score for one child order."""
    child_cost = max(0.0, float(child_cost_dollars))
    current_cost = max(0.0, float(current_cost_dollars))
    _validate_probability(p, "p")
    _validate_probability(q_child, "q_child")
    if bankroll_for_sizing <= 0:
        raise ValueError("bankroll_for_sizing must be positive")

    if current_cost == 0.0:
        q_current = q_child if q_current_position is None else q_current_position
    elif q_current_position is None:
        raise ValueError("q_current_position is required when current_cost_dollars > 0")
    else:
        q_current = q_current_position
    _validate_probability(q_current, "q_current_position")

    q_after = (
        q_avg_after_child
        if q_avg_after_child is not None
        else average_q_after_child(
            current_cost_dollars=current_cost,
            q_current_position=q_current,
            child_cost_dollars=child_cost,
            q_child=q_child,
        )
    )
    _validate_probability(q_after, "q_avg_after_child")

    current_growth = expected_log_wealth(
        current_cost,
        p=p,
        q=q_current,
        bankroll_for_sizing=bankroll_for_sizing,
    )
    after_growth = expected_log_wealth(
        current_cost + child_cost,
        p=p,
        q=q_after,
        bankroll_for_sizing=bankroll_for_sizing,
    )
    incremental = after_growth - current_growth
    score = incremental / child_cost if child_cost > 0.0 else 0.0
    return MarginalLogGrowth(
        current_cost_dollars=current_cost,
        child_cost_dollars=child_cost,
        new_cost_dollars=current_cost + child_cost,
        q_current_position=q_current,
        q_child=q_child,
        q_avg_after_child=q_after,
        expected_log_growth_current=current_growth,
        expected_log_growth_after_child=after_growth,
        expected_log_growth_next_child=incremental,
        priority_score=score,
    )


def score_cash_priority_ticket(
    ticket: CashPriorityTicket,
    *,
    available_cash_after_buffer: float,
) -> Optional[ScoredCashPriorityTicket]:
    """Score one eligible ticket using the exact marginal log-growth rule."""
    if (
        not ticket.eligible
        or ticket.allowed_to_try_now <= 0.0
        or ticket.q_exec_all_in > ticket.q_max
    ):
        return None
    candidate_child = min(
        ticket.allowed_to_try_now,
        ticket.child_cap,
        max(0.0, available_cash_after_buffer),
    )
    if candidate_child < ticket.min_child_order_dollars:
        return None

    marginal = marginal_expected_log_growth_per_dollar(
        p=ticket.model_prob,
        q_child=ticket.q_exec_all_in,
        bankroll_for_sizing=ticket.bankroll_for_sizing,
        current_cost_dollars=ticket.current_cost,
        child_cost_dollars=candidate_child,
        q_current_position=ticket.current_position_q,
    )
    if not math.isfinite(marginal.priority_score):
        return None
    return ScoredCashPriorityTicket(
        ticket=ticket,
        candidate_child_size_dollars=candidate_child,
        marginal=marginal,
    )


def rank_cash_limited_tickets(
    tickets: Iterable[CashPriorityTicket],
    *,
    available_cash_after_buffer: float,
) -> Tuple[ScoredCashPriorityTicket, ...]:
    scored = [
        scored
        for ticket in tickets
        if (scored := score_cash_priority_ticket(
            ticket,
            available_cash_after_buffer=available_cash_after_buffer,
        ))
        is not None
    ]
    ranked = sorted(scored, key=cmp_to_key(_compare_scored_tickets))
    return tuple(
        ScoredCashPriorityTicket(
            ticket=item.ticket,
            candidate_child_size_dollars=item.candidate_child_size_dollars,
            marginal=item.marginal,
            cash_priority_rank=idx,
        )
        for idx, item in enumerate(ranked, start=1)
    )


def allocate_cash_greedily(
    tickets: Sequence[CashPriorityTicket],
    *,
    available_cash_after_buffer: float,
) -> Tuple[CashPriorityAllocation, ...]:
    """Rank tickets, then greedily allocate scarce cash in priority order."""
    remaining_cash = max(0.0, float(available_cash_after_buffer))
    allocations = []
    ranked = rank_cash_limited_tickets(
        tickets,
        available_cash_after_buffer=available_cash_after_buffer,
    )
    for scored in ranked:
        ticket = scored.ticket
        child_size = min(ticket.allowed_to_try_now, ticket.child_cap, remaining_cash)
        if child_size >= ticket.min_child_order_dollars:
            allocations.append(
                CashPriorityAllocation(
                    ticket_id=ticket.ticket_id,
                    cash_priority_rank=scored.cash_priority_rank or 0,
                    priority_score=scored.priority_score,
                    expected_log_growth_next_child=scored.marginal.expected_log_growth_next_child,
                    candidate_child_size_dollars=scored.candidate_child_size_dollars,
                    allocated_child_size_dollars=child_size,
                    skipped_due_to_cash=False,
                )
            )
            remaining_cash -= child_size
            continue

        allocations.append(
            CashPriorityAllocation(
                ticket_id=ticket.ticket_id,
                cash_priority_rank=scored.cash_priority_rank or 0,
                priority_score=scored.priority_score,
                expected_log_growth_next_child=scored.marginal.expected_log_growth_next_child,
                candidate_child_size_dollars=scored.candidate_child_size_dollars,
                allocated_child_size_dollars=0.0,
                skipped_due_to_cash=True,
                skip_reason="cash_remaining_below_min_child",
            )
        )
    return tuple(allocations)


def _compare_scored_tickets(a: ScoredCashPriorityTicket, b: ScoredCashPriorityTicket) -> int:
    if not _priority_scores_close(a.priority_score, b.priority_score):
        return -1 if a.priority_score > b.priority_score else 1

    tie_breakers = (
        (a.ticket.absolute_edge, b.ticket.absolute_edge, True),
        (a.ticket.normalized_edge, b.ticket.normalized_edge, True),
        (_early_ts_value(a.ticket.first_qualified_ts_s), _early_ts_value(b.ticket.first_qualified_ts_s), False),
        (a.ticket.executable_liquidity_dollars, b.ticket.executable_liquidity_dollars, True),
        (a.ticket.route_slippage, b.ticket.route_slippage, False),
    )
    for left, right, higher_is_better in tie_breakers:
        if left == right:
            continue
        if higher_is_better:
            return -1 if left > right else 1
        return -1 if left < right else 1
    return -1 if a.ticket.ticket_id < b.ticket.ticket_id else (1 if a.ticket.ticket_id > b.ticket.ticket_id else 0)


def _priority_scores_close(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / scale < CLOSE_PRIORITY_RELATIVE_TOLERANCE


def _early_ts_value(value: Optional[float]) -> float:
    return float(value) if value is not None else float("inf")


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
