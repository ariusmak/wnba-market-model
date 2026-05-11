"""
Read-only Kalshi route-mapping and execution-planning smoke test.

Given one Sportradar game reference and a selected-team probability, this
script fetches active WNBA Kalshi markets, confirms the two-market mapping,
builds BUY YES selected / BUY NO opponent routes, fetches current books,
and prints the planned IOC child orders. It never submits or cancels orders.

Usage:
    python pipelines/07_live/canonical/02_smoke_kalshi_routes.py \
        --game-id <sr_game_id> \
        --scheduled 2026-05-10T22:00:00+00:00 \
        --home-team-id <sr_home_uuid> \
        --away-team-id <sr_away_uuid> \
        --selected-team-id <sr_selected_uuid> \
        --p-selected 0.64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.canonical.execution import (  # noqa: E402
    ExecutionConfig,
    evaluate_route_quote,
    plan_normal_ioc_orders,
)
from srwnba.live.canonical.kalshi_mapping import (  # noqa: E402
    SportRadarGameRef,
    build_equivalent_routes,
    load_team_name_map,
    map_game_to_kalshi_markets,
    parse_datetime,
)
from srwnba.live.canonical.portfolio import resolve_portfolio_sizing  # noqa: E402
from utils.kalshi_authed_client import AuthedKalshiClient, KalshiAuthConfig  # noqa: E402


DEFAULT_SERIES = ("KXWNBAGAME", "KXWNBAH")


def _fetch_markets(
    client: AuthedKalshiClient,
    series_tickers: list[str],
    limit: int,
    status: str | None,
) -> list[dict]:
    markets: list[dict] = []
    for series in series_tickers:
        markets.extend(client.list_markets(series_ticker=series, status=status, limit=limit))
    return markets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--scheduled", required=True,
                        help="ISO scheduled timestamp, preferably with timezone")
    parser.add_argument("--home-team-id", required=True)
    parser.add_argument("--away-team-id", required=True)
    parser.add_argument("--home-team-name", default="")
    parser.add_argument("--away-team-name", default="")
    parser.add_argument("--selected-team-id", required=True)
    parser.add_argument("--p-selected", type=float, required=True)
    parser.add_argument("--available-cash-override", "--available-cash", dest="available_cash_override", type=float, default=None)
    parser.add_argument("--sizing-bankroll-override", "--bankroll", dest="sizing_bankroll_override", type=float, default=None)
    parser.add_argument("--series-ticker", action="append")
    parser.add_argument("--status", default=None,
                        help="Optional Kalshi status filter. Default omits status because live markets report 'active' but the list API rejects status=active.")
    parser.add_argument("--markets-limit", type=int, default=100)
    args = parser.parse_args()

    cfg = KalshiAuthConfig.from_env(REPO_ROOT / ".env")
    client = AuthedKalshiClient(cfg)
    portfolio_sizing = resolve_portfolio_sizing(
        client,
        sizing_bankroll_override_dollars=args.sizing_bankroll_override,
        available_cash_override_dollars=args.available_cash_override,
    )
    exec_cfg = ExecutionConfig(bankroll=portfolio_sizing.sizing_bankroll_dollars)
    team_name_map = load_team_name_map(str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"))
    game = SportRadarGameRef(
        game_id=args.game_id,
        scheduled=parse_datetime(args.scheduled),
        home_team_id=args.home_team_id,
        away_team_id=args.away_team_id,
        home_team_name=args.home_team_name,
        away_team_name=args.away_team_name,
    )

    print("Kalshi route smoke")
    print(f"  base_url: {cfg.base_url}")
    print(f"  trading_enabled: {cfg.trading_enabled}")
    print(f"  game_id: {game.game_id}")
    print(f"  selected_team_id: {args.selected_team_id}")
    print(f"  p_selected: {args.p_selected:.4f}")
    print(
        f"  sizing_bankroll: ${portfolio_sizing.sizing_bankroll_dollars:,.2f} "
        f"source={portfolio_sizing.sizing_bankroll_source}"
    )
    if portfolio_sizing.available_cash_dollars is not None:
        print(
            f"  available_cash: ${portfolio_sizing.available_cash_dollars:,.2f} "
            f"source={portfolio_sizing.available_cash_source}"
        )

    series_tickers = args.series_ticker or list(DEFAULT_SERIES)
    markets = _fetch_markets(client, series_tickers, args.markets_limit, args.status)
    print(f"  fetched_active_markets: {len(markets)}")

    mapping = map_game_to_kalshi_markets(
        game,
        markets,
        require_open=True,
        team_name_to_id=team_name_map,
    )
    print("Mapping")
    print(f"  confirmed: {mapping.confirmed}")
    print(f"  event_ticker: {mapping.event_ticker or 'none'}")
    print(f"  candidate_count: {mapping.candidate_count}")
    for diagnostic in mapping.diagnostics[:8]:
        print(f"  diagnostic: {diagnostic}")
    if not mapping.confirmed:
        raise SystemExit("Mapping is not confirmed; refusing to plan routes.")

    routes = build_equivalent_routes(mapping, args.selected_team_id)
    quotes = []
    print("Routes")
    for route in routes:
        book = client.get_orderbook(route.market_ticker)
        quote = evaluate_route_quote(
            route,
            book,
            p_selected=args.p_selected,
            cfg=exec_cfg,
        )
        quotes.append(quote)
        print(
            f"  {route.route_type} {route.side.upper()} {route.market_ticker}: "
            f"best={quote.best_ask_cents}c all_in_avg={quote.all_in_avg_price_cents:.2f}c "
            f"qmax={quote.q_max_cents}c fillable={quote.fillable_contracts_at_qmax} "
            f"capacity=${quote.route_capacity_dollars:.2f} eligible={quote.eligible} "
            f"reject={quote.reject_reason or '-'}"
        )

    plan = plan_normal_ioc_orders(
        selected_team_id=args.selected_team_id,
        p_selected=args.p_selected,
        route_quotes=quotes,
        cfg=exec_cfg,
        available_cash_dollars=portfolio_sizing.available_cash_dollars,
    )
    print("Plan")
    print(f"  decision: {plan.decision}")
    print(f"  reject_reason: {plan.reject_reason or '-'}")
    print(f"  target_position: ${plan.target_position_dollars:.2f}")
    print(f"  remaining_position: ${plan.remaining_position_dollars:.2f}")
    print(f"  allowed_child: ${plan.allowed_child_dollars:.2f}")
    for order in plan.orders:
        print(
            f"  order: {order.route_type} {order.side.upper()} {order.market_ticker} "
            f"{order.count}@{order.limit_price_cents}c max_cost=${order.max_cost_dollars:.2f}"
        )
    print("OK")


if __name__ == "__main__":
    main()
