"""
03_trader_unit_test.py
======================

Pure-logic sanity checks for src/srwnba/live/legacy/trader.py. No network, no
Kalshi credentials, no saved artifacts required — just verifies the
planner produces the sizes / max_price / skip reasons we expect under
hand-crafted scenarios.

Run from repo root:
    python pipelines/07_live/legacy/03_trader_unit_test.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.legacy.trader import (  # noqa: E402
    MarketSide,
    OrderbookLevel,
    OrderbookSnapshot,
    TradeConfig,
    estimate_kalshi_fee_dollars,
    half_kelly_wager,
    max_price_dollars,
    plan_sweep,
    yes_asks_from_no_bids,
)


def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def test_max_price_norm_binds() -> None:
    cfg = TradeConfig()
    p = 0.65
    mp = max_price_dollars(p, cfg)
    expect_norm = p / 1.25
    expect_abs = p - 0.05
    assert _approx(mp, min(expect_abs, expect_norm))
    # For p > 0.25, norm constraint binds:
    assert _approx(mp, expect_norm), (mp, expect_norm)
    print(f"  max_price(p=0.65) = {mp:.4f}  norm_cap={expect_norm:.4f}  abs_cap={expect_abs:.4f}  [norm binds OK]")


def test_max_price_abs_binds_when_prob_tiny() -> None:
    # For p < 0.25 the absolute cap is tighter (would only happen on short side)
    cfg = TradeConfig()
    p = 0.20
    mp = max_price_dollars(p, cfg)
    assert _approx(mp, p - 0.05)
    print(f"  max_price(p=0.20) = {mp:.4f}  [abs binds OK]")


def test_half_kelly_math() -> None:
    cfg = TradeConfig(bankroll=5000.0, kelly_fraction=0.5)
    p, q = 0.65, 0.55
    f_star = (p - q) / (1 - q)
    expected = 0.5 * f_star * 5000.0
    got = half_kelly_wager(p, q, cfg)
    assert _approx(got, expected), (got, expected)
    print(f"  half_kelly(p=0.65,q=0.55) = ${got:.2f}  (f*={f_star:.4f})")


def test_plan_sweep_happy_path() -> None:
    cfg = TradeConfig()
    side = MarketSide(ticker="KXWNBAH-TEST-HOME", side_label="home", p_model=0.65)
    # Ask side deep enough: 100 @ 52¢, 300 @ 54¢, plenty at higher prices.
    asks = (
        OrderbookLevel(52, 100),
        OrderbookLevel(53,  50),
        OrderbookLevel(54, 300),
        OrderbookLevel(56, 400),
    )
    book = OrderbookSnapshot(ticker=side.ticker, ts_ms=0, yes_asks=asks)
    plan = plan_sweep(side, book, already_held=0, cfg=cfg)
    assert not plan.skip, plan.skip_reason
    # max_price: min(0.60, 0.52) = 0.52 dollars → 52 cents
    assert plan.max_price_cents == 52, plan.max_price_cents
    # Only the first level (52¢) is ≤ cap → fillable = 100
    assert plan.fillable_size == 100, plan.fillable_size
    # Kelly sizing at best_ask=52¢: f* = (0.65-0.52)/(1-0.52) ≈ 0.2708
    # wager = 0.5 * 0.2708 * 5000 ≈ $677.08 → target = 677.08/0.52 ≈ 1302 contracts
    f_star = (0.65 - 0.52) / (1 - 0.52)
    expected_target = math.floor(0.5 * f_star * 5000.0 / 0.52)
    assert plan.target_contracts == expected_target, (plan.target_contracts, expected_target)
    # Book-capped count = min(target, fillable) = 100
    assert plan.count == 100, plan.count
    print(f"  happy_path: count={plan.count} cap={plan.max_price_cents}¢ "
          f"target={plan.target_contracts} fillable={plan.fillable_size}")


def test_plan_sweep_skips_when_no_edge() -> None:
    cfg = TradeConfig()
    side = MarketSide(ticker="T", side_label="home", p_model=0.55)
    # best ask 50¢ — p - 0.05 = 0.50, norm cap = 0.44. Cap = 0.44 → 44¢.
    # 50¢ > 44¢ → skip.
    asks = (OrderbookLevel(50, 200),)
    book = OrderbookSnapshot(ticker="T", ts_ms=0, yes_asks=asks)
    plan = plan_sweep(side, book, already_held=0, cfg=cfg)
    assert plan.skip, "should skip"
    assert "cap" in plan.skip_reason, plan.skip_reason
    print(f"  skip-no-edge: reason='{plan.skip_reason}'")


def test_plan_sweep_already_at_target() -> None:
    cfg = TradeConfig()
    side = MarketSide(ticker="T", side_label="home", p_model=0.70)
    asks = (OrderbookLevel(50, 5000),)
    book = OrderbookSnapshot(ticker="T", ts_ms=0, yes_asks=asks)
    # target at 50¢ ≈ 2000 contracts (tiny float rounding gives 1999)
    plan = plan_sweep(side, book, already_held=5000, cfg=cfg)
    assert plan.skip, "should skip when over target"
    assert "already_at_target" in plan.skip_reason
    print(f"  skip-over-target: reason='{plan.skip_reason}'")


def test_plan_sweep_remaining_after_partial_fill() -> None:
    cfg = TradeConfig()
    side = MarketSide(ticker="T", side_label="home", p_model=0.70)
    asks = (OrderbookLevel(50, 5000),)
    book = OrderbookSnapshot(ticker="T", ts_ms=0, yes_asks=asks)
    # Compute the planner's target using the same float path it takes
    target = math.floor(half_kelly_wager(0.70, 0.50, cfg) / 0.50)
    expected_count = target - 500
    plan = plan_sweep(side, book, already_held=500, cfg=cfg)
    assert not plan.skip, plan.skip_reason
    assert plan.count == expected_count, (plan.count, expected_count)
    print(f"  remaining-after-partial: count={plan.count} "
          f"(target={plan.target_contracts}, held=500)")


def test_plan_sweep_empty_book() -> None:
    cfg = TradeConfig()
    side = MarketSide(ticker="T", side_label="home", p_model=0.70)
    book = OrderbookSnapshot(ticker="T", ts_ms=0, yes_asks=())
    plan = plan_sweep(side, book, already_held=0, cfg=cfg)
    assert plan.skip and plan.skip_reason == "empty_book"
    print(f"  skip-empty-book OK")


def test_yes_asks_from_no_bids() -> None:
    # NO bids at 40¢ (size 100) and 45¢ (size 50)  → YES asks at 60¢, 55¢
    asks = yes_asks_from_no_bids([[40, 100], [45, 50]])
    assert [(lv.price_cents, lv.size) for lv in asks] == [(55, 50), (60, 100)]
    print(f"  yes_asks_from_no_bids: {[(lv.price_cents, lv.size) for lv in asks]}")


def test_fee_estimate() -> None:
    # Kalshi's published: ceil(0.07 · 100 · 0.5 · 0.5 · 100) / 100 = ceil(175)/100 = 1.75
    fee = estimate_kalshi_fee_dollars(count=100, price_cents=50)
    assert _approx(fee, 1.75), fee
    # At 30¢: ceil(0.07 · 100 · 0.3 · 0.7 · 100)/100 = ceil(147)/100 = 1.47
    assert _approx(estimate_kalshi_fee_dollars(100, 30), 1.47)
    print(f"  fee(100 @ 50¢) = ${fee:.2f}")


def main() -> None:
    print("[trader] unit tests")
    tests = [
        test_max_price_norm_binds,
        test_max_price_abs_binds_when_prob_tiny,
        test_half_kelly_math,
        test_plan_sweep_happy_path,
        test_plan_sweep_skips_when_no_edge,
        test_plan_sweep_already_at_target,
        test_plan_sweep_remaining_after_partial_fill,
        test_plan_sweep_empty_book,
        test_yes_asks_from_no_bids,
        test_fee_estimate,
    ]
    for t in tests:
        t()
    print(f"[trader] {len(tests)} tests passed")


if __name__ == "__main__":
    main()
