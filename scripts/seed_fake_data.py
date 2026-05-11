from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supabase_io import get_supabase


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def fake_rows(now: datetime) -> dict[str, list[dict]]:
    games = [
        {
            "game_id": "FAKE-ATL-NYL-2026-05-12",
            "home_team": "Atlanta Dream",
            "away_team": "New York Liberty",
            "selected_team": "New York Liberty",
            "opponent_team": "Atlanta Dream",
            "tipoff_ts": iso(now + timedelta(hours=16)),
            "time_to_tipoff_minutes": 960,
            "phase": "ACTIVE",
            "trading_status": "eligible",
            "expansion_gate_status": "clear",
            "first_qualified_lead_hours": 16.0,
            "model_prob": 0.66,
            "market_prob": 0.53,
            "abs_edge": 0.13,
            "norm_edge": 0.245,
            "q_max_price": 0.528,
            "q_exec_all_in_price": 0.525,
            "bankroll_for_sizing_dollars": 5000,
            "available_cash_after_buffer_dollars": 4400,
            "half_kelly_target_dollars": 690,
            "portfolio_cap_dollars": 750,
            "cash_cap_dollars": 4400,
            "target_position_now_dollars": 690,
            "filled_position_dollars": 140,
            "filled_contracts": 265,
            "reserved_open_order_dollars": 0,
            "remaining_position_dollars": 550,
            "visible_depth_cap_dollars": 320,
            "recent_volume_cap_dollars": 500,
            "cold_start_cap_dollars": 250,
            "rolling_liquidity_cap_dollars": 320,
            "cumulative_cap_remaining_dollars": 220,
            "allowed_to_try_now_dollars": 220,
            "next_child_order_dollars": 75,
            "target_position_binder": "half_kelly",
            "execution_binder": "cumulative",
            "cash_limited_mode": True,
            "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
            "cash_priority_rank": 2,
            "cash_priority_rank_total": 2,
            "cash_priority_score": 0.000074,
            "expected_log_growth_next_child": 0.00555,
            "cash_priority_candidate_child_dollars": 75,
            "cash_priority_allocated_child_dollars": 0,
            "skipped_due_to_cash": True,
            "q_current_position": 0.528,
            "q_avg_after_child": 0.526,
            "number_of_fills": 2,
            "number_of_order_attempts": 3,
            "number_of_trades_made": 2,
            "average_fill_price": 0.528,
            "vwap_lead_hours": 15.2,
            "last_action": "shadow_ioc_planned",
            "last_reject_reason": None,
            "last_fill_ts": iso(now - timedelta(minutes=8)),
            "last_order_ts": iso(now - timedelta(minutes=5)),
            "market_data_ts": iso(now - timedelta(seconds=20)),
            "model_snapshot_ts": iso(now - timedelta(minutes=2)),
            "injury_data_ts": iso(now - timedelta(hours=1)),
            "orderbook_ts": iso(now - timedelta(seconds=15)),
        },
        {
            "game_id": "FAKE-LVA-PHX-2026-05-12",
            "home_team": "Las Vegas Aces",
            "away_team": "Phoenix Mercury",
            "selected_team": "Las Vegas Aces",
            "opponent_team": "Phoenix Mercury",
            "tipoff_ts": iso(now + timedelta(hours=22)),
            "time_to_tipoff_minutes": 1320,
            "phase": "PASSIVE_ELIGIBLE",
            "trading_status": "no_edge",
            "expansion_gate_status": "clear",
            "model_prob": 0.61,
            "market_prob": 0.59,
            "abs_edge": 0.02,
            "norm_edge": 0.034,
            "q_max_price": 0.488,
            "q_exec_all_in_price": 0.594,
            "bankroll_for_sizing_dollars": 5000,
            "available_cash_after_buffer_dollars": 4400,
            "half_kelly_target_dollars": 99,
            "portfolio_cap_dollars": 750,
            "cash_cap_dollars": 4400,
            "target_position_now_dollars": 99,
            "filled_position_dollars": 0,
            "filled_contracts": 0,
            "reserved_open_order_dollars": 0,
            "remaining_position_dollars": 99,
            "visible_depth_cap_dollars": 0,
            "recent_volume_cap_dollars": 60,
            "cold_start_cap_dollars": 250,
            "rolling_liquidity_cap_dollars": 0,
            "cumulative_cap_remaining_dollars": 0,
            "allowed_to_try_now_dollars": 0,
            "next_child_order_dollars": 0,
            "target_position_binder": "half_kelly",
            "execution_binder": "none",
            "cash_limited_mode": False,
            "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
            "cash_priority_rank": None,
            "cash_priority_rank_total": 2,
            "cash_priority_score": None,
            "expected_log_growth_next_child": None,
            "cash_priority_candidate_child_dollars": 0,
            "cash_priority_allocated_child_dollars": 0,
            "skipped_due_to_cash": False,
            "q_current_position": None,
            "q_avg_after_child": None,
            "last_action": "skip_no_edge",
            "last_reject_reason": "edge_failed",
            "market_data_ts": iso(now - timedelta(seconds=30)),
            "model_snapshot_ts": iso(now - timedelta(minutes=2)),
            "injury_data_ts": iso(now - timedelta(hours=1)),
            "orderbook_ts": iso(now - timedelta(seconds=28)),
        },
        {
            "game_id": "FAKE-TOR-SEA-2026-05-13",
            "home_team": "Toronto Tempo",
            "away_team": "Seattle Storm",
            "selected_team": "Seattle Storm",
            "opponent_team": "Toronto Tempo",
            "tipoff_ts": iso(now + timedelta(hours=31)),
            "time_to_tipoff_minutes": 1860,
            "phase": "BLOCKED",
            "trading_status": "expansion_gate",
            "expansion_gate_status": "blocked_14_prior_games_required",
            "model_prob": 0.58,
            "market_prob": 0.45,
            "abs_edge": 0.13,
            "norm_edge": 0.289,
            "q_max_price": 0.464,
            "q_exec_all_in_price": 0.456,
            "bankroll_for_sizing_dollars": 5000,
            "available_cash_after_buffer_dollars": 4400,
            "half_kelly_target_dollars": 598,
            "portfolio_cap_dollars": 750,
            "cash_cap_dollars": 4400,
            "target_position_now_dollars": 598,
            "filled_position_dollars": 0,
            "filled_contracts": 0,
            "reserved_open_order_dollars": 0,
            "remaining_position_dollars": 598,
            "visible_depth_cap_dollars": 210,
            "recent_volume_cap_dollars": 280,
            "cold_start_cap_dollars": 0,
            "rolling_liquidity_cap_dollars": 0,
            "cumulative_cap_remaining_dollars": 0,
            "allowed_to_try_now_dollars": 0,
            "next_child_order_dollars": 0,
            "target_position_binder": "half_kelly",
            "execution_binder": "cold_start",
            "cash_limited_mode": False,
            "cash_priority_rule": "marginal_expected_log_growth_per_dollar",
            "cash_priority_rank": None,
            "cash_priority_rank_total": 2,
            "cash_priority_score": None,
            "expected_log_growth_next_child": None,
            "cash_priority_candidate_child_dollars": 0,
            "cash_priority_allocated_child_dollars": 0,
            "skipped_due_to_cash": False,
            "q_current_position": None,
            "q_avg_after_child": None,
            "last_action": "skip_expansion_gate",
            "last_reject_reason": "expansion_gate",
            "market_data_ts": iso(now - timedelta(seconds=40)),
            "model_snapshot_ts": iso(now - timedelta(minutes=3)),
            "injury_data_ts": iso(now - timedelta(hours=1)),
            "orderbook_ts": iso(now - timedelta(seconds=35)),
        },
    ]

    for game in games:
        game.setdefault("number_of_fills", 0)
        game.setdefault("number_of_order_attempts", 0)
        game.setdefault("number_of_trades_made", 0)

    routes = []
    for game in games:
        game_id = game["game_id"]
        selected = game["selected_team"].upper().replace(" ", "")
        opponent = game["opponent_team"].upper().replace(" ", "")
        routes.extend(
            [
                {
                    "game_id": game_id,
                    "route_name": "BUY_YES_SELECTED",
                    "market_ticker": f"KXWNBA-FAKE-{selected}",
                    "outcome_side": "yes",
                    "q_exec_all_in_price": game["q_exec_all_in_price"],
                    "best_bid_price": max(0, game["market_prob"] - 0.02),
                    "best_ask_price": game["market_prob"],
                    "spread_ticks": 2,
                    "visible_depth_to_qmax_dollars": game["visible_depth_cap_dollars"],
                    "recent_qualifying_volume_3h_dollars": game["recent_volume_cap_dollars"],
                    "route_rolling_cap_dollars": game["rolling_liquidity_cap_dollars"],
                    "route_cumulative_cap_remaining_dollars": game["cumulative_cap_remaining_dollars"],
                    "chosen": game["trading_status"] == "eligible",
                    "route_decision_reason": game["last_action"],
                    "updated_at": iso(now),
                },
                {
                    "game_id": game_id,
                    "route_name": "BUY_NO_OPPONENT",
                    "market_ticker": f"KXWNBA-FAKE-{opponent}",
                    "outcome_side": "no",
                    "q_exec_all_in_price": min(0.99, game["q_exec_all_in_price"] + 0.01),
                    "best_bid_price": max(0, game["market_prob"] - 0.03),
                    "best_ask_price": min(0.99, game["market_prob"] + 0.01),
                    "spread_ticks": 3,
                    "visible_depth_to_qmax_dollars": max(0, game["visible_depth_cap_dollars"] - 60),
                    "recent_qualifying_volume_3h_dollars": max(0, game["recent_volume_cap_dollars"] - 40),
                    "route_rolling_cap_dollars": max(0, game["rolling_liquidity_cap_dollars"] - 50),
                    "route_cumulative_cap_remaining_dollars": max(0, game["cumulative_cap_remaining_dollars"] - 50),
                    "chosen": False,
                    "route_decision_reason": "higher_all_in_price",
                    "updated_at": iso(now),
                },
            ]
        )

    return {
        "control_state": [
            {
                "id": "global",
                "trading_enabled": False,
                "kill_switch_active": False,
                "allow_new_entries": True,
                "allow_ioc_orders": True,
                "allow_passive_orders": True,
                "allow_burst_mode": True,
                "mode": "shadow",
                "max_market_exposure_pct": 0.15,
                "shadow_mode_enabled": True,
                "updated_at": iso(now),
                "updated_by": "seed_fake_data",
                "reason": "Seeded fake control-plane demo state",
            }
        ],
        "live_market_snapshots": games,
        "route_snapshots": routes,
        "bot_heartbeat": [
            {
                "bot_id": "fake-worker",
                "status": "shadow",
                "last_seen_at": iso(now),
                "last_control_seen_at": iso(now),
                "current_mode": "shadow",
                "kalshi_connected": False,
                "market_data_connected": True,
                "database_connected": True,
                "open_orders_count": 0,
                "open_positions_count": 1,
                "last_error": None,
            }
        ],
        "order_events": [
            {
                "game_id": "FAKE-ATL-NYL-2026-05-12",
                "market_ticker": "KXWNBA-FAKE-NEWYORKLIBERTY",
                "route_name": "BUY_YES_SELECTED",
                "order_id": "fake-shadow-001",
                "event_type": "submitted",
                "order_mode": "shadow",
                "outcome_side": "yes",
                "price": 0.528,
                "contracts": 140,
                "cost_dollars": 73.92,
                "lead_hours": 16.0,
                "reason": "fake shadow order for dashboard QA",
                "raw_payload": {"fake": True},
                "created_at": iso(now - timedelta(minutes=5)),
            },
            {
                "game_id": "FAKE-TOR-SEA-2026-05-13",
                "market_ticker": "KXWNBA-FAKE-SEATTLESTORM",
                "route_name": "BUY_YES_SELECTED",
                "event_type": "skipped",
                "order_mode": "shadow",
                "outcome_side": "yes",
                "price": 0.456,
                "contracts": 0,
                "cost_dollars": 0,
                "lead_hours": 31.0,
                "reason": "expansion_gate",
                "raw_payload": {"fake": True},
                "created_at": iso(now - timedelta(minutes=4)),
            },
        ],
        "equity_curve": [
            {
                "ts": iso(now),
                "equity_dollars": 5000.0,
                "cash_dollars": 4860.0,
                "open_position_value_dollars": 140.0,
                "realized_pnl_dollars": 0.0,
                "drawdown_dollars": 0.0,
                "total_markets_observed": len(games),
                "entered_markets": 1,
            }
        ],
        "system_alerts": [
            {
                "severity": "info",
                "alert_type": "fake_seed",
                "message": "Fake control-plane data seeded for dashboard QA.",
                "payload": {"fake": True},
                "created_at": iso(now),
            }
        ],
    }


def upsert_rows(table: str, rows: list[dict], on_conflict: str | None = None) -> None:
    if not rows:
        return
    query = get_supabase().table(table).upsert(rows, on_conflict=on_conflict) if on_conflict else get_supabase().table(table).upsert(rows)
    query.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed fake Supabase rows for the Streamlit control plane.")
    parser.add_argument("--now", help="UTC ISO timestamp override for deterministic demos.")
    args = parser.parse_args()

    now = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    rows_by_table = fake_rows(now)
    upsert_rows("control_state", rows_by_table["control_state"], on_conflict="id")
    upsert_rows("live_market_snapshots", rows_by_table["live_market_snapshots"], on_conflict="game_id")
    upsert_rows(
        "route_snapshots",
        rows_by_table["route_snapshots"],
        on_conflict="game_id,route_name,market_ticker,outcome_side",
    )
    upsert_rows("bot_heartbeat", rows_by_table["bot_heartbeat"], on_conflict="bot_id")
    upsert_rows("order_events", rows_by_table["order_events"])
    upsert_rows("equity_curve", rows_by_table["equity_curve"], on_conflict="ts")
    upsert_rows("system_alerts", rows_by_table["system_alerts"])
    print("Seeded fake control-plane data.")


if __name__ == "__main__":
    main()
