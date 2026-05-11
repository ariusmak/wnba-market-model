"""
Unit checks for Kalshi market mapping and route-level execution planning.

These tests are intentionally runnable without pytest:

    python pipelines/07_live/canonical/02_mapping_execution_unit_test.py

They do not call Kalshi and they do not submit orders.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.canonical.execution import (  # noqa: E402
    ExecutionConfig,
    PlannedChildOrder,
    evaluate_route_quote,
    no_asks_from_yes_bids,
    order_kwargs_from_plan,
    plan_normal_ioc_orders,
    submit_planned_child_order,
)
from srwnba.live.canonical.cash_priority import (  # noqa: E402
    CashPriorityTicket,
    allocate_cash_greedily,
    average_q_after_child,
    expected_log_wealth,
    marginal_expected_log_growth_per_dollar,
    rank_cash_limited_tickets,
)
from srwnba.live.canonical.cash_coordinator import coordinate_cash_for_plan  # noqa: E402
from srwnba.live.canonical.expansion_gate import (  # noqa: E402
    TRUE_EXPANSION_TEAMS_2026,
    evaluate_expansion_team_gate,
)
from srwnba.live.canonical.kalshi_mapping import (  # noqa: E402
    SportRadarGameRef,
    build_equivalent_routes,
    extract_custom_strike,
    filter_open_wnba_moneyline_markets,
    filter_wnba_moneyline_markets,
    is_open_wnba_moneyline_market,
    is_wnba_moneyline_market,
    load_team_name_map,
    map_game_to_kalshi_markets,
    parse_event_date,
)
from srwnba.live.canonical.operator_control import OperatorDecision  # noqa: E402
from srwnba.live.canonical.route_entry_loop import RouteEntryContext, RouteEntryLoop  # noqa: E402
from srwnba.live.canonical.portfolio import resolve_portfolio_sizing  # noqa: E402
from srwnba.live.canonical.process_lock import GameProcessLock, read_game_lock_status  # noqa: E402
from srwnba.live.control_plane import RemoteControlSnapshot, merge_control_decision  # noqa: E402
from srwnba.live.canonical.v1_2 import (  # noqa: E402
    BrakeState,
    PlannerRuntimeState,
    SignalMemory,
    plan_v1_2_orders,
    timing_state,
    update_signal_memory,
)


TORONTO_ID = "4e4f726e-a015-4306-91a7-28e8576c7868"
PORTLAND_ID = "d54283cc-c5ec-4dbd-bb61-166f217e3864"


def _load_first_matched_game() -> tuple[dict, list[dict]]:
    matched_path = REPO_ROOT / "data" / "kalshi" / "wnba_2025_game_markets_matched.csv"
    markets_path = REPO_ROOT / "data" / "kalshi" / "kalshi_markets.csv"
    with matched_path.open(newline="", encoding="utf-8") as f:
        matched_rows = list(csv.DictReader(f))
    row = next(r for r in matched_rows if r.get("game_id") and r.get("team_a_id") and r.get("team_b_id"))
    event_ticker = row["event_ticker"]
    with markets_path.open(newline="", encoding="utf-8") as f:
        markets = [r for r in csv.DictReader(f) if r.get("event_ticker") == event_ticker]
    if len(markets) != 2:
        raise AssertionError(f"expected two markets for {event_ticker}, got {len(markets)}")
    return row, markets


def test_custom_strike_parser() -> None:
    parsed = extract_custom_strike({"custom_strike": "{'basketball_team': 'abc'}"})
    assert parsed["basketball_team"] == "abc"
    parsed = extract_custom_strike({"custom_strike": '{"basketball_team": "xyz"}'})
    assert parsed["basketball_team"] == "xyz"
    print("  custom_strike parser OK")


def test_event_date_parser() -> None:
    assert parse_event_date("KXWNBAGAME-26MAY10PHXGS").isoformat() == "2026-05-10"
    print("  event date parser OK")


def test_moneyline_market_filter() -> None:
    moneyline = {
        "ticker": "KXWNBAGAME-26MAY10PHXGS-PHX",
        "event_ticker": "KXWNBAGAME-26MAY10PHXGS",
        "series_ticker": "KXWNBAGAME",
        "title": "Phoenix vs Golden State winner?",
        "market_type": "binary",
        "rules_primary": (
            "If Phoenix wins the Phoenix vs Golden State women's professional basketball game "
            "originally scheduled for May 10, 2026, then the market resolves to Yes."
        ),
        "custom_strike": {"basketball_team": "a007957c-46fa-4d50-82f8-48dd4da02ba6"},
    }
    spread = {
        **moneyline,
        "ticker": "KXWNBAGAME-26MAY10PHXGS-PHXSPREAD",
        "title": "Phoenix +4.5?",
        "rules_primary": "If Phoenix loses by fewer than 5 points, then the market resolves to Yes.",
    }
    finalized = {**moneyline, "status": "finalized"}
    active = {**moneyline, "status": "active"}
    assert is_wnba_moneyline_market(moneyline)
    assert not is_wnba_moneyline_market(spread)
    assert filter_wnba_moneyline_markets([moneyline, spread]) == [moneyline]
    assert is_open_wnba_moneyline_market(active)
    assert not is_open_wnba_moneyline_market(finalized)
    assert filter_open_wnba_moneyline_markets([active, finalized, spread]) == [active]
    print("  moneyline market filter OK")


class FakeBalanceClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def get_balance(self) -> dict:
        return dict(self.payload)


def test_portfolio_sizing_resolution() -> None:
    snap = resolve_portfolio_sizing(FakeBalanceClient({
        "balance": 425_000,
        "portfolio_value": 512_345,
    }))
    assert snap.kalshi_cash_dollars == 4250.0
    assert snap.kalshi_portfolio_value_dollars == 5123.45
    assert snap.sizing_bankroll_dollars == 5123.45
    assert snap.sizing_bankroll_source == "kalshi_portfolio_value"
    assert snap.available_cash_dollars == 4250.0
    assert snap.available_cash_source == "kalshi_balance"

    fallback = resolve_portfolio_sizing(FakeBalanceClient({"balance": 425_000}))
    assert fallback.sizing_bankroll_dollars == 4250.0
    assert fallback.sizing_bankroll_source == "kalshi_balance_fallback"

    override = resolve_portfolio_sizing(
        FakeBalanceClient({"balance": 425_000, "portfolio_value": 512_345}),
        sizing_bankroll_override_dollars=2500.0,
        available_cash_override_dollars=1000.0,
    )
    assert override.sizing_bankroll_dollars == 2500.0
    assert override.sizing_bankroll_source == "override"
    assert override.available_cash_dollars == 1000.0
    assert override.available_cash_source == "override"
    print("  portfolio sizing resolution OK")


def test_game_process_lock_blocks_duplicate_acquire() -> None:
    with TemporaryDirectory() as tmp:
        first = GameProcessLock(game_id="unit/game", lock_dir=Path(tmp))
        first.acquire()
        status = read_game_lock_status(first.path)
        assert status.locked and status.running and status.pid is not None
        duplicate_blocked = False
        try:
            GameProcessLock(game_id="unit/game", lock_dir=Path(tmp)).acquire()
        except RuntimeError:
            duplicate_blocked = True
        assert duplicate_blocked
        first.release()
        assert not read_game_lock_status(first.path).locked
    print("  game process lock duplicate guard OK")


def test_expansion_team_name_map_aliases() -> None:
    with (REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv").open(
        newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row["sportradar_team_id"] for row in rows)
    duplicates = {team_id: count for team_id, count in counts.items() if count > 1}
    assert not duplicates, duplicates
    assert len(rows) == 15, f"expected one row per 2026 WNBA team, got {len(rows)}"

    team_name_to_id = load_team_name_map(str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"))
    assert team_name_to_id["toronto"] == TORONTO_ID
    assert team_name_to_id["toronto tempo"] == TORONTO_ID
    assert team_name_to_id["portland"] == PORTLAND_ID
    assert team_name_to_id["portland fire"] == PORTLAND_ID
    print("  expansion team name-map aliases OK")


def test_expansion_gate() -> None:
    washington_id = "5c0d47fe-8539-47b0-9f36-d0b3609ca89b"
    assert TORONTO_ID in TRUE_EXPANSION_TEAMS_2026
    assert PORTLAND_ID in TRUE_EXPANSION_TEAMS_2026

    result = evaluate_expansion_team_gate(
        home_team_id=washington_id,
        away_team_id="0699edf3-5993-4182-b9b4-ec935cbd4fcc",
        completed_games_by_team={},
    )
    assert result.allowed and not result.applies

    result = evaluate_expansion_team_gate(
        home_team_id=TORONTO_ID,
        away_team_id=washington_id,
        completed_games_by_team={TORONTO_ID: 13},
    )
    assert not result.allowed
    assert "blocked_expansion_team_under_14_completed_games" in result.reason

    result = evaluate_expansion_team_gate(
        home_team_id=TORONTO_ID,
        away_team_id=PORTLAND_ID,
        completed_games_by_team={TORONTO_ID: 14, PORTLAND_ID: 14},
    )
    assert result.allowed and result.applies
    assert result.reason == "expansion_team_gate_passed"
    print("  expansion-team gate OK")


def test_historical_mapping_and_routes() -> None:
    row, markets = _load_first_matched_game()
    scheduled = datetime.fromisoformat(row["game_date"] + "T12:00:00+00:00")
    game = SportRadarGameRef(
        game_id=row["game_id"],
        scheduled=scheduled,
        home_team_id=row["team_a_id"],
        away_team_id=row["team_b_id"],
        home_team_name=row["team_a"],
        away_team_name=row["team_b"],
    )
    team_name_to_id = load_team_name_map(str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"))
    mapping = map_game_to_kalshi_markets(game, markets, team_name_to_id=team_name_to_id)
    assert mapping.confirmed, mapping.diagnostics
    assert mapping.event_ticker == row["event_ticker"]

    routes = build_equivalent_routes(mapping, selected_team_id=row["team_a_id"])
    assert len(routes) == 2
    assert routes[0].route_type == "BUY_YES_SELECTED"
    assert routes[0].side == "yes"
    assert routes[0].market_yes_team_id == row["team_a_id"]
    assert routes[1].route_type == "BUY_NO_OPPONENT"
    assert routes[1].side == "no"
    assert routes[1].market_yes_team_id == row["team_b_id"]
    print(f"  mapping/routes OK for {row['event_ticker']}")
    return routes


def test_route_book_conversion() -> None:
    levels = no_asks_from_yes_bids([[48, 100], [47, 50]])
    assert [(level.price_cents, level.size) for level in levels] == [(52, 100), (53, 50)]
    print("  BUY_NO book conversion OK")


def test_execution_plan() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    # BUY YES selected sees YES asks from NO bids: 55c.
    # BUY NO opponent sees NO asks from YES bids: 52c.
    books = {
        routes[0].market_ticker: {"orderbook": {"yes": [[41, 1000]], "no": [[45, 1000]]}},
        routes[1].market_ticker: {"orderbook": {"yes": [[48, 1000]], "no": [[40, 1000]]}},
    }
    quotes = [
        evaluate_route_quote(route, books[route.market_ticker], p_selected=0.72, cfg=cfg, ts_ms=1)
        for route in routes
    ]
    assert all(q.eligible for q in quotes), [(q.route.route_type, q.reject_reason) for q in quotes]
    plan = plan_normal_ioc_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        available_cash_dollars=5000.0,
    )
    assert plan.decision == "normal_ioc", plan.reject_reason
    assert plan.orders, plan
    first = plan.orders[0]
    assert first.side in {"yes", "no"}
    assert first.limit_price_cents <= plan.q_max_cents
    assert first.max_cost_dollars <= cfg.normal_max_ioc_child_order_pct * cfg.bankroll
    print(
        "  execution plan OK "
        f"decision={plan.decision} orders={len(plan.orders)} "
        f"first={first.route_type} {first.side} {first.count}@{first.limit_price_cents}c"
    )


def test_high_price_rejection() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quote = evaluate_route_quote(
        routes[0],
        {"orderbook": {"yes": [], "no": [[35, 1000]]}},
        p_selected=0.72,
        cfg=cfg,
        ts_ms=1,
    )
    assert not quote.eligible
    assert "q_max" in quote.reject_reason
    print("  high all-in price rejection OK")


class FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def create_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"order": {"order_id": "fake", "status": "executed"}}


def test_order_kwargs_and_submit_bridge() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quote = evaluate_route_quote(
        routes[1],
        {"orderbook": {"yes": [[48, 1000]], "no": [[40, 1000]]}},
        p_selected=0.72,
        cfg=cfg,
        ts_ms=1,
    )
    plan = plan_normal_ioc_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=[quote],
        cfg=cfg,
        available_cash_dollars=5000.0,
    )
    assert plan.orders
    order = plan.orders[0]
    kwargs = order_kwargs_from_plan(order, "unit-client-order-id")
    assert kwargs["side"] == "no"
    assert kwargs["no_price_cents"] == order.limit_price_cents
    assert kwargs["time_in_force"] == "immediate_or_cancel"
    assert "yes_price_cents" not in kwargs

    fake = FakeClient()
    resp = submit_planned_child_order(fake, order, client_order_id="unit-client-order-id")
    assert resp["order"]["order_id"] == "fake"
    assert fake.calls[0] == kwargs
    print("  planned order submit bridge OK")


def test_cash_priority_math_and_allocator() -> None:
    growth = expected_log_wealth(
        100.0,
        p=0.62,
        q=0.50,
        bankroll_for_sizing=5000.0,
    )
    expected = 0.62 * math.log(1.0 + 0.02 * 1.0) + 0.38 * math.log(1.0 - 0.02)
    assert abs(growth - expected) < 1e-12

    first = marginal_expected_log_growth_per_dollar(
        p=0.62,
        q_child=0.50,
        bankroll_for_sizing=5000.0,
        current_cost_dollars=0.0,
        child_cost_dollars=100.0,
    )
    later = marginal_expected_log_growth_per_dollar(
        p=0.62,
        q_child=0.50,
        bankroll_for_sizing=5000.0,
        current_cost_dollars=700.0,
        q_current_position=0.50,
        child_cost_dollars=100.0,
    )
    assert first.priority_score > later.priority_score

    q_after = average_q_after_child(
        current_cost_dollars=100.0,
        q_current_position=0.50,
        child_cost_dollars=100.0,
        q_child=0.40,
    )
    assert 0.44 < q_after < 0.45

    tickets = [
        CashPriorityTicket(
            ticket_id="higher_edge",
            eligible=True,
            model_prob=0.68,
            q_exec_all_in=0.50,
            q_max=0.52,
            bankroll_for_sizing=5000.0,
            allowed_to_try_now=100.0,
            child_cap=100.0,
            absolute_edge=0.18,
            normalized_edge=0.36,
            first_qualified_ts_s=20.0,
            executable_liquidity_dollars=100.0,
            route_slippage=2.0,
        ),
        CashPriorityTicket(
            ticket_id="lower_edge",
            eligible=True,
            model_prob=0.64,
            q_exec_all_in=0.50,
            q_max=0.52,
            bankroll_for_sizing=5000.0,
            allowed_to_try_now=100.0,
            child_cap=100.0,
            absolute_edge=0.14,
            normalized_edge=0.28,
            first_qualified_ts_s=10.0,
            executable_liquidity_dollars=500.0,
            route_slippage=1.0,
        ),
    ]
    ranked = rank_cash_limited_tickets(tickets, available_cash_after_buffer=120.0)
    assert [item.ticket.ticket_id for item in ranked] == ["higher_edge", "lower_edge"]
    allocated = allocate_cash_greedily(tickets, available_cash_after_buffer=120.0)
    assert allocated[0].ticket_id == "higher_edge"
    assert allocated[0].allocated_child_size_dollars == 100.0
    assert allocated[1].skipped_due_to_cash
    print("  cash priority math/allocator OK")


def test_cash_coordinator_preserves_first_qualification_ts() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quotes = [
        evaluate_route_quote(
            routes[0],
            {"orderbook": {"yes": [[41, 1000]], "no": [[45, 1000]]}},
            p_selected=0.72,
            cfg=cfg,
            ts_ms=1,
        )
    ]
    signal = update_signal_memory(SignalMemory(), quotes, now_s=123.0, tipoff_ts_s=13 * 3600)
    plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=123.0,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
        ),
    )
    assert plan.orders
    with TemporaryDirectory() as tmp:
        result = coordinate_cash_for_plan(
            game_id="cash-tie-break-game",
            plan=plan,
            cfg=cfg,
            coordinator_dir=Path(tmp),
            available_cash_after_buffer=1000.0,
            filled_position_dollars=0.0,
            reserved_position_dollars=0.0,
            current_position_q=None,
            first_qualified_ts_s=signal.first_qualified_ts_s,
            wait_s=0.0,
        )
        assert result.plan.orders
        candidate_path = Path(tmp) / "candidates" / "cash-tie-break-game.json"
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        assert candidate["first_qualified_ts_s"] == 123.0
    print("  cash coordinator first-qualified tie-break field OK")


def test_v1_2_timing_and_signal_memory() -> None:
    cfg = ExecutionConfig(bankroll=5000.0)
    assert timing_state(0, 20 * 3600, cfg).window == "monitor_T24_to_T17"
    assert timing_state(0, 13 * 3600, cfg).window == "main_T17_to_T12"
    assert timing_state(0, 6 * 3600, cfg).window == "prequalified_T8_to_T4"
    assert timing_state(0, 3 * 3600, cfg).window == "monitor_T4_to_tip"

    routes = test_historical_mapping_and_routes()
    quotes = [
        evaluate_route_quote(
            routes[0],
            {"orderbook": {"yes": [[41, 1000]], "no": [[45, 1000]]}},
            p_selected=0.72,
            cfg=cfg,
            ts_ms=1,
        )
    ]
    signal = update_signal_memory(SignalMemory(), quotes, now_s=0, tipoff_ts_s=13 * 3600)
    assert signal.first_qualified_lead_hours == 13
    plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=0,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
        ),
    )
    assert plan.decision in {"normal_ioc", "burst_ioc"}, plan
    assert plan.orders
    assert plan.timing_window == "main_T17_to_T12"
    assert plan.cash_priority_rule == "marginal_expected_log_growth_per_dollar"
    assert plan.cash_priority_score is not None
    assert plan.expected_log_growth_next_child is not None
    print("  v1.2 timing/signal/order planning OK")


def test_v1_2_late_only_rejection() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quotes = [
        evaluate_route_quote(
            routes[0],
            {"orderbook": {"yes": [[41, 1000]], "no": [[45, 1000]]}},
            p_selected=0.72,
            cfg=cfg,
            ts_ms=1,
        )
    ]
    signal = update_signal_memory(SignalMemory(), quotes, now_s=0, tipoff_ts_s=6 * 3600)
    plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=0,
            tipoff_ts_s=6 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
        ),
    )
    assert plan.decision == "no_trade"
    assert plan.reject_reason == "late_only_signal"
    print("  v1.2 late-only rejection OK")


def test_v1_2_burst_and_brake_controls() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quotes = [
        evaluate_route_quote(
            routes[0],
            {"orderbook": {"yes": [[60, 2000]], "no": [[60, 2000]]}},
            p_selected=0.72,
            cfg=cfg,
            ts_ms=1,
        )
    ]
    signal = update_signal_memory(SignalMemory(), quotes, now_s=0, tipoff_ts_s=13 * 3600)
    first_plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=0,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
        ),
    )
    assert first_plan.decision == "normal_ioc", first_plan.decision

    burst_plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=120,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
            previous_combined_visible_cost_dollars=300.0,
            combined_visible_cost_after_last_order_dollars=100.0,
        ),
    )
    assert burst_plan.decision == "burst_ioc", burst_plan.decision

    blocked = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=120,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
            brake_state=BrakeState(order_reject_timestamps_s=(1, 2, 3, 4, 5)),
        ),
    )
    assert blocked.decision == "blocked_brake"
    assert blocked.reject_reason == "order_reject_brake_enforced"
    print("  v1.2 burst/brake controls OK")


def test_v1_2_conservative_mode() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quotes = [
        evaluate_route_quote(
            routes[0],
            {"orderbook": {"yes": [[60, 5000]], "no": [[60, 5000]]}},
            p_selected=0.92,
            cfg=cfg,
            ts_ms=1,
        )
    ]
    signal = update_signal_memory(SignalMemory(), quotes, now_s=0, tipoff_ts_s=13 * 3600)
    plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.92,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=120,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
            previous_combined_visible_cost_dollars=300.0,
            combined_visible_cost_after_last_order_dollars=100.0,
            brake_state=BrakeState(conservative_mode=True),
        ),
    )
    assert plan.target_position_dollars <= 0.12 * cfg.bankroll + 1e-9, plan.target_position_dollars
    assert plan.decision != "burst_ioc", plan.decision
    print("  v1.2 conservative mode OK")


def test_v1_2_splits_tied_routes() -> None:
    routes = test_historical_mapping_and_routes()
    cfg = ExecutionConfig(bankroll=5000.0)
    quotes = [
        evaluate_route_quote(
            routes[0],
            {"orderbook": {"yes": [[48, 1000]], "no": [[48, 1000]]}},
            p_selected=0.72,
            cfg=cfg,
            ts_ms=1,
        ),
        evaluate_route_quote(
            routes[1],
            {"orderbook": {"yes": [[48, 1000]], "no": [[48, 1000]]}},
            p_selected=0.72,
            cfg=cfg,
            ts_ms=1,
        ),
    ]
    assert all(quote.eligible for quote in quotes)
    signal = update_signal_memory(SignalMemory(), quotes, now_s=0, tipoff_ts_s=13 * 3600)
    plan = plan_v1_2_orders(
        selected_team_id=routes[0].selected_team_id,
        p_selected=0.72,
        route_quotes=quotes,
        cfg=cfg,
        runtime=PlannerRuntimeState(
            now_s=0,
            tipoff_ts_s=13 * 3600,
            signal=signal,
            available_cash_dollars=5000.0,
        ),
    )
    assert len(plan.orders) == 2, [(order.route_type, order.max_cost_dollars) for order in plan.orders]
    assert {order.route_type for order in plan.orders} == {"BUY_YES_SELECTED", "BUY_NO_OPPONENT"}
    print("  v1.2 tied-route split OK")


class FakeExpansionRouteClient:
    def __init__(self, books):
        self.books = books
        self.calls = []

    def get_orderbook(self, ticker: str):
        return {"orderbook": self.books[ticker]}

    def get_balance(self):
        return {"balance": 400_000, "portfolio_value": 600_000}

    def create_order(self, **kwargs):
        self.calls.append(kwargs)
        return {"order": {"order_id": "should-not-submit", "status": "executed"}}


class FakePredictor:
    def predict(self, _df):
        return {"p_home": [0.72]}


def test_route_loop_blocks_expansion_team_under_gate() -> None:
    washington_id = "5c0d47fe-8539-47b0-9f36-d0b3609ca89b"
    scheduled = datetime(2026, 5, 8, 23, 30, tzinfo=timezone.utc)
    event_ticker = "KXWNBAGAME-26MAY08TORWAS"
    toronto_ticker = f"{event_ticker}-TOR"
    washington_ticker = f"{event_ticker}-WAS"
    markets = [
        {
            "ticker": toronto_ticker,
            "event_ticker": event_ticker,
            "title": "Toronto vs Washington winner?",
            "yes_sub_title": "Toronto",
            "status": "active",
            "market_type": "binary",
            "rules_primary": (
                "If Toronto wins the Toronto vs Washington women's professional basketball game "
                "originally scheduled for May 8, 2026, then the market resolves to Yes."
            ),
            "custom_strike": {"basketball_team": "diagnostic-toronto"},
        },
        {
            "ticker": washington_ticker,
            "event_ticker": event_ticker,
            "title": "Toronto vs Washington winner?",
            "yes_sub_title": "Washington",
            "status": "active",
            "market_type": "binary",
            "rules_primary": (
                "If Washington wins the Toronto vs Washington women's professional basketball game "
                "originally scheduled for May 8, 2026, then the market resolves to Yes."
            ),
            "custom_strike": {"basketball_team": "diagnostic-washington"},
        },
    ]
    fake = FakeExpansionRouteClient({
        toronto_ticker: {"yes": [[50, 1000]], "no": [[50, 1000]]},
        washington_ticker: {"yes": [[50, 1000]], "no": [[50, 1000]]},
    })
    log_path = REPO_ROOT / "data" / "live_logs" / "unit_expansion_gate.jsonl"
    if log_path.exists():
        log_path.unlink()

    ctx = RouteEntryContext(
        game=SportRadarGameRef(
            game_id="unit-expansion-gate",
            scheduled=scheduled,
            home_team_id=TORONTO_ID,
            away_team_id=washington_id,
            home_team_name="Toronto Tempo",
            away_team_name="Washington Mystics",
        ),
        tipoff_ts_s=1,
        feature_row=[],
        team_name_map_path=REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv",
        completed_games_by_team={TORONTO_ID: 13},
    )
    loop = RouteEntryLoop(
        predictor=FakePredictor(),
        client=fake,
        ctx=ctx,
        cfg=ExecutionConfig(bankroll=5000.0),
        log_path=log_path,
        poll_interval_s=0.0,
        dry_run=False,
        markets=markets,
        follow_kalshi_wealth=True,
    )
    assert loop.cfg.bankroll == 6000.0
    assert loop.available_cash_dollars == 4000.0
    loop._poll_once()

    assert fake.calls == []
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    plan = next(event for event in events if event["evt"] == "execution_plan")
    assert plan["decision"] == "no_trade"
    assert plan["expansion_team_gate_passed"] is False
    assert "blocked_expansion_team_under_14_completed_games" in plan["reject_reason"]
    print("  route loop expansion gate block OK")


def test_control_plane_merge_blocks_and_shadows() -> None:
    local = OperatorDecision(
        game_id="control-plane-unit",
        trade_allowed=True,
        reason="operator_allowed",
        auto_trade_enabled=True,
        risk_mode="normal",
        game_decision="default",
        global_control_path="local-control.json",
        game_override_path="local-override.json",
    )
    blocked_remote = RemoteControlSnapshot(
        mode="supabase-live",
        read_ok=True,
        configured=True,
        database_connected=True,
        read_at_utc="2026-05-11T00:00:00+00:00",
        control_state={
            "trading_enabled": True,
            "kill_switch_active": False,
            "allow_new_entries": True,
            "allow_ioc_orders": True,
            "allow_passive_orders": True,
            "allow_burst_mode": True,
            "mode": "normal",
        },
        market_control={
            "market_status": "blocked",
            "block_new_entries": True,
        },
    )
    decision = merge_control_decision(
        game_id="control-plane-unit",
        mode="supabase-live",
        local_decision=local,
        remote_snapshot=blocked_remote,
    )
    assert decision.trade_allowed is False
    assert decision.reason == "control_plane_market_blocked"

    read_failed = merge_control_decision(
        game_id="control-plane-unit",
        mode="supabase-live",
        local_decision=local,
        remote_snapshot=RemoteControlSnapshot(
            mode="supabase-live",
            read_ok=False,
            configured=True,
            database_connected=False,
            read_at_utc="2026-05-11T00:00:00+00:00",
            error="unit read failure",
        ),
    )
    assert read_failed.trade_allowed is False
    assert read_failed.reason == "control_plane_read_failed"

    publish_failed = merge_control_decision(
        game_id="control-plane-unit",
        mode="supabase-live",
        local_decision=local,
        remote_snapshot=RemoteControlSnapshot(
            mode="supabase-live",
            read_ok=True,
            configured=True,
            database_connected=True,
            read_at_utc="2026-05-11T00:00:00+00:00",
            control_state={
                "trading_enabled": True,
                "kill_switch_active": False,
                "allow_new_entries": True,
                "mode": "normal",
            },
        ),
        publish_failure_count=3,
    )
    assert publish_failed.trade_allowed is False
    assert publish_failed.reason == "control_plane_publish_unhealthy"

    shadow_remote = RemoteControlSnapshot(
        mode="supabase-shadow",
        read_ok=True,
        configured=True,
        database_connected=True,
        read_at_utc="2026-05-11T00:00:00+00:00",
        control_state={
            "trading_enabled": True,
            "kill_switch_active": False,
            "allow_new_entries": True,
            "allow_ioc_orders": True,
            "allow_passive_orders": True,
            "allow_burst_mode": True,
            "shadow_mode_enabled": False,
            "mode": "normal",
        },
        market_control={},
    )
    shadow = merge_control_decision(
        game_id="control-plane-unit",
        mode="supabase-shadow",
        local_decision=local,
        remote_snapshot=shadow_remote,
    )
    assert shadow.trade_allowed is True
    assert shadow.shadow_mode_enabled is True
    order = PlannedChildOrder(
        route_id="r1",
        market_ticker="KXUNIT",
        route_type="BUY_YES_SELECTED",
        action="buy",
        side="yes",
        count=1,
        limit_price_cents=40,
        max_cost_dollars=0.40,
        expected_all_in_avg_price_cents=40.0,
        q_max_cents=40,
    )
    assert shadow.block_reason_for_order(order) == "control_plane_shadow_mode"
    print("  control-plane merge/shadow gates OK")


def main() -> None:
    print("[mapping/execution] unit tests")
    test_custom_strike_parser()
    test_event_date_parser()
    test_moneyline_market_filter()
    test_portfolio_sizing_resolution()
    test_game_process_lock_blocks_duplicate_acquire()
    test_expansion_team_name_map_aliases()
    test_expansion_gate()
    test_historical_mapping_and_routes()
    test_route_book_conversion()
    test_execution_plan()
    test_high_price_rejection()
    test_order_kwargs_and_submit_bridge()
    test_cash_priority_math_and_allocator()
    test_cash_coordinator_preserves_first_qualification_ts()
    test_v1_2_timing_and_signal_memory()
    test_v1_2_late_only_rejection()
    test_v1_2_burst_and_brake_controls()
    test_v1_2_conservative_mode()
    test_v1_2_splits_tied_routes()
    test_route_loop_blocks_expansion_team_under_gate()
    test_control_plane_merge_blocks_and_shadows()
    print("[mapping/execution] OK")


if __name__ == "__main__":
    main()
