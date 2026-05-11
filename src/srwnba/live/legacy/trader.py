"""
src/srwnba/live/legacy/trader.py
─────────────────────────
Pure trading-logic layer for the live WNBA system.

Everything in this module is side-effect-free: given a model probability,
an orderbook snapshot, and current position state, it decides whether to
trade and returns a `SweepPlan` that the entry loop executes through the
authed Kalshi client.

The policy reproduces the frozen legacy backtest configuration:

    edge_min       = 0.05     (absolute-edge threshold)
    norm_edge_min  = 0.25     (edge / entry_price threshold)
    kelly_fraction = 0.5      (half-Kelly stake)
    bankroll       = $5000
    hold_to_settle = True

Key equations
-------------
Max price we're willing to pay (limit-IOC cap). Both the absolute and the
normalised edge thresholds must hold at fill price q:
    edge           = p_side - q           ≥ edge_min
    normalised_edge= (p_side - q) / q     ≥ norm_edge_min
Solving each for q:
    q ≤ p_side - edge_min
    q ≤ p_side / (1 + norm_edge_min)
So
    max_price = min(p_side - edge_min, p_side / (1 + norm_edge_min))
For (0.05, 0.25) the norm-edge constraint binds whenever p_side > 0.25,
which is essentially every qualifying trade (we always take the higher-
probability side so p_side > 0.5).

Half-Kelly on a $1 payoff at entry price q with model prob p:
    f*     = (p - q) / (1 - q)           (standard Kelly for binary bet)
    wager$ = (f* / 2) · bankroll
    target_contracts = wager$ / q

On Kalshi, each YES contract costs q dollars and pays $1 at settlement, so
`target_contracts` is the share count we'd want if liquidity allowed.

Sweep-plan sizing
-----------------
We walk the ask side of the book up to `max_price` and collect total
available size. The final order count is
    count = min(target_contracts - already_held, available_size)

The entry loop submits a **limit-IOC** buy at `max_price_cents` for `count`
contracts; any unfilled size simply drops (no resting order).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────
# Config + data classes
# ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TradeConfig:
    """Frozen trading parameters. Defaults reproduce the 2025 backtest."""
    edge_min: float = 0.05
    norm_edge_min: float = 0.25
    kelly_fraction: float = 0.5
    bankroll: float = 5000.0
    fee_rate: float = 0.07            # Kalshi taker fee coefficient
    min_contracts: int = 1            # smallest lot we'll bother submitting

    def validate(self) -> None:
        assert 0.0 < self.edge_min < 0.5, self.edge_min
        assert 0.0 < self.norm_edge_min < 5.0, self.norm_edge_min
        assert 0.0 < self.kelly_fraction <= 1.0, self.kelly_fraction
        assert self.bankroll > 0, self.bankroll
        assert self.fee_rate >= 0, self.fee_rate


@dataclass(frozen=True)
class MarketSide:
    """One side of a WNBA moneyline market (home YES or away YES)."""
    ticker: str                  # e.g. "KXWNBAH-26MAY12LASEA-LAS"
    side_label: str              # "home" | "away" — for logging only
    p_model: float               # our model's probability for this outcome

    def __post_init__(self) -> None:
        assert 0.0 < self.p_model < 1.0, self.p_model


@dataclass(frozen=True)
class OrderbookLevel:
    price_cents: int             # integer 1..99
    size: int                    # resting contracts at this level

    def __post_init__(self) -> None:
        assert 1 <= self.price_cents <= 99, self.price_cents
        assert self.size >= 0, self.size


@dataclass(frozen=True)
class OrderbookSnapshot:
    """Ask-side depth we'd lift to buy YES.

    Kalshi's /orderbook returns both `yes` and `no` as lists of
    [price_cents, size] pairs representing *bids* on each side. To buy YES
    we lift resting NO bids (since a buy YES at price p is economically
    equivalent to an offer of NO at 100-p). The entry loop converts the
    raw NO-bid book into ask levels for YES and hands us this struct.
    """
    ticker: str
    ts_ms: int
    yes_asks: Tuple[OrderbookLevel, ...] = field(default_factory=tuple)

    def best_ask_cents(self) -> Optional[int]:
        return self.yes_asks[0].price_cents if self.yes_asks else None


@dataclass(frozen=True)
class SweepPlan:
    """What the entry loop will send to Kalshi."""
    ticker: str
    side_label: str
    action: str = "buy"
    side: str = "yes"
    count: int = 0
    max_price_cents: int = 0
    p_model: float = 0.0
    best_ask_cents: int = 0
    expected_fill_cents: float = 0.0   # size-weighted avg over fillable levels
    target_contracts: int = 0          # half-Kelly target before book cap
    fillable_size: int = 0             # size ≤ max_price at the time of snap
    skip: bool = False
    skip_reason: str = ""

    @classmethod
    def skipped(cls, ticker: str, side_label: str, p_model: float,
                best_ask_cents: int, reason: str) -> "SweepPlan":
        return cls(
            ticker=ticker, side_label=side_label, count=0,
            p_model=p_model, best_ask_cents=best_ask_cents,
            skip=True, skip_reason=reason,
        )


# ──────────────────────────────────────────────────────────────────────────
# Core planner
# ──────────────────────────────────────────────────────────────────────────

def max_price_dollars(p_side: float, cfg: TradeConfig) -> float:
    """Highest fill price for which both edge thresholds hold.

    Binds on norm_edge_min for p_side > norm_edge_min (nearly always).
    """
    abs_cap  = p_side - cfg.edge_min
    norm_cap = p_side / (1.0 + cfg.norm_edge_min)
    return min(abs_cap, norm_cap)


def half_kelly_wager(p: float, q: float, cfg: TradeConfig) -> float:
    """Half-Kelly stake in dollars at model prob p and fill price q."""
    if q >= p:
        return 0.0
    f_star = (p - q) / (1.0 - q)
    return max(0.0, cfg.kelly_fraction * f_star * cfg.bankroll)


def walk_book_to_cap(
    yes_asks: Tuple[OrderbookLevel, ...],
    max_price_cents: int,
) -> Tuple[int, float]:
    """Sum available size at ask levels ≤ max_price_cents.

    Returns (total_size, size_weighted_avg_price_cents). Weighted-avg is
    0.0 if no fillable size.
    """
    total = 0
    cost_cents = 0
    for lvl in yes_asks:
        if lvl.price_cents > max_price_cents:
            break
        total += lvl.size
        cost_cents += lvl.size * lvl.price_cents
    if total == 0:
        return 0, 0.0
    return total, cost_cents / total


def plan_sweep(
    side: MarketSide,
    book: OrderbookSnapshot,
    already_held: int,
    cfg: TradeConfig,
) -> SweepPlan:
    """Decide whether to sweep a single market side, and for how much.

    Returns a `SweepPlan` — callers should check `.skip` before firing.

    Parameters
    ----------
    side : the market/ticker and our model's probability for that outcome.
    book : ask-side depth snapshot for the same ticker.
    already_held : contracts we've already filled on this side (so we cap
        total exposure at the half-Kelly target across repeated sweeps).
    cfg  : trade parameters (defaults = frozen backtest config).
    """
    cfg.validate()
    best_ask_cents = book.best_ask_cents() or 0

    if not book.yes_asks:
        return SweepPlan.skipped(
            book.ticker, side.side_label, side.p_model,
            best_ask_cents, "empty_book",
        )

    p = side.p_model
    mp_dollars = max_price_dollars(p, cfg)
    if mp_dollars <= 0.0:
        return SweepPlan.skipped(
            book.ticker, side.side_label, p, best_ask_cents,
            f"no_edge p={p:.4f}",
        )

    max_price_cents = int(math.floor(mp_dollars * 100))
    if max_price_cents < 1:
        return SweepPlan.skipped(
            book.ticker, side.side_label, p, best_ask_cents,
            f"max_price_below_1c p={p:.4f}",
        )

    if best_ask_cents > max_price_cents:
        return SweepPlan.skipped(
            book.ticker, side.side_label, p, best_ask_cents,
            f"best_ask {best_ask_cents} > cap {max_price_cents}",
        )

    fillable, avg_cents = walk_book_to_cap(book.yes_asks, max_price_cents)
    if fillable <= 0:
        return SweepPlan.skipped(
            book.ticker, side.side_label, p, best_ask_cents,
            "no_fillable_size",
        )

    q = best_ask_cents / 100.0
    wager = half_kelly_wager(p, q, cfg)
    target_contracts = int(math.floor(wager / q)) if q > 0 else 0
    remaining = max(0, target_contracts - already_held)

    if remaining < cfg.min_contracts:
        return SweepPlan.skipped(
            book.ticker, side.side_label, p, best_ask_cents,
            f"already_at_target held={already_held} target={target_contracts}",
        )

    count = min(remaining, fillable)
    if count < cfg.min_contracts:
        return SweepPlan.skipped(
            book.ticker, side.side_label, p, best_ask_cents,
            f"fillable_below_min fillable={fillable}",
        )

    return SweepPlan(
        ticker=book.ticker,
        side_label=side.side_label,
        count=count,
        max_price_cents=max_price_cents,
        p_model=p,
        best_ask_cents=best_ask_cents,
        expected_fill_cents=avg_cents,
        target_contracts=target_contracts,
        fillable_size=fillable,
        skip=False,
        skip_reason="",
    )


# ──────────────────────────────────────────────────────────────────────────
# Book conversion helpers
# ──────────────────────────────────────────────────────────────────────────

def yes_asks_from_no_bids(no_bids: List[List[int]]) -> Tuple[OrderbookLevel, ...]:
    """Convert Kalshi's NO-bid depth list into ascending YES-ask levels.

    Kalshi returns orderbook.no as [[price_cents, size], ...] representing
    resting NO bids. A NO bid at price n is equivalent to a YES offer at
    (100 - n). We dedupe, sort ascending by YES-ask price, and ignore any
    crossed/invalid entries.
    """
    levels: List[OrderbookLevel] = []
    for row in no_bids or []:
        if not row or len(row) < 2:
            continue
        n_price, size = int(row[0]), int(row[1])
        if size <= 0 or not (1 <= n_price <= 99):
            continue
        yes_price = 100 - n_price
        levels.append(OrderbookLevel(price_cents=yes_price, size=size))
    levels.sort(key=lambda lv: lv.price_cents)
    # Collapse duplicates (shouldn't happen for Kalshi, but be safe).
    collapsed: List[OrderbookLevel] = []
    for lv in levels:
        if collapsed and collapsed[-1].price_cents == lv.price_cents:
            collapsed[-1] = OrderbookLevel(lv.price_cents, collapsed[-1].size + lv.size)
        else:
            collapsed.append(lv)
    return tuple(collapsed)


def snapshot_from_kalshi(ticker: str, payload: dict, ts_ms: int) -> OrderbookSnapshot:
    """Build an ask-side snapshot from the raw /orderbook response.

    Kalshi payload shape (authed): {"orderbook": {"yes": [...], "no": [...]}}
    To buy YES we need NO-side bids (the mechanical offers against us).
    """
    book = payload.get("orderbook") or {}
    yes_asks = yes_asks_from_no_bids(book.get("no") or [])
    return OrderbookSnapshot(ticker=ticker, ts_ms=ts_ms, yes_asks=yes_asks)


# ──────────────────────────────────────────────────────────────────────────
# Fee estimate (for logging/telemetry only — does not alter sizing)
# ──────────────────────────────────────────────────────────────────────────

def estimate_kalshi_fee_dollars(count: int, price_cents: int, fee_rate: float = 0.07) -> float:
    """Kalshi published taker fee: ceil(fee_rate · n · p · (1-p) · 100) / 100 dollars.

    We compute in integer-cent space first to avoid float drift — e.g. the
    naive expression ``0.07 · 100 · 0.5 · 0.5 · 100`` evaluates to
    175.0000000000…3 in IEEE754 and would round up to 176¢ = $1.76 instead
    of the correct $1.75.
    """
    if count <= 0:
        return 0.0
    # fee_rate is normally 0.07 → 7 per 100. Keep arbitrary precision by
    # routing through Decimal only for the final ceil.
    num = fee_rate * count * price_cents * (100 - price_cents)
    # fee (cents) = ceil(num / 100), treating tiny float drift (<1e-6) as zero.
    scaled = num / 100.0
    eps = 1e-9
    fee_cents = math.ceil(scaled - eps)
    return fee_cents / 100.0
