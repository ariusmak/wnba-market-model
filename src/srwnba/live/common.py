"""
Shared Kalshi orderbook and fee primitives.

Keep this file neutral: both the legacy YES-only loop and the canonical
route executor can depend on it without creating a legacy/canonical import
cycle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class OrderbookLevel:
    price_cents: int
    size: int

    def __post_init__(self) -> None:
        assert 1 <= self.price_cents <= 99, self.price_cents
        assert self.size >= 0, self.size


def yes_asks_from_no_bids(no_bids: List[List[int]]) -> Tuple[OrderbookLevel, ...]:
    """Convert Kalshi NO bids into executable YES ask levels."""
    levels: List[OrderbookLevel] = []
    for row in no_bids or []:
        if not row or len(row) < 2:
            continue
        n_price, size = int(row[0]), int(row[1])
        if size <= 0 or not (1 <= n_price <= 99):
            continue
        levels.append(OrderbookLevel(price_cents=100 - n_price, size=size))
    levels.sort(key=lambda level: level.price_cents)

    collapsed: List[OrderbookLevel] = []
    for level in levels:
        if collapsed and collapsed[-1].price_cents == level.price_cents:
            collapsed[-1] = OrderbookLevel(
                level.price_cents,
                collapsed[-1].size + level.size,
            )
        else:
            collapsed.append(level)
    return tuple(collapsed)


def estimate_kalshi_fee_dollars(
    count: int,
    price_cents: int,
    fee_rate: float = 0.07,
) -> float:
    """Kalshi taker fee estimate in dollars."""
    if count <= 0:
        return 0.0
    num = fee_rate * count * price_cents * (100 - price_cents)
    scaled = num / 100.0
    fee_cents = math.ceil(scaled - 1e-9)
    return fee_cents / 100.0
