"""
01_smoke_kalshi_market_data.py
==============================

Read-only Kalshi market-data smoke test for WNBA live trading.

The script uses the active `.env` Kalshi credentials, lists WNBA markets
by series ticker when no explicit market ticker is supplied, then fetches
one market and its current orderbook. It never submits or cancels orders.

Usage:
    python pipelines/07_live/01_smoke_kalshi_market_data.py
    python pipelines/07_live/01_smoke_kalshi_market_data.py --ticker KX...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.common import yes_asks_from_no_bids  # noqa: E402
from utils.kalshi_authed_client import AuthedKalshiClient, KalshiAuthConfig  # noqa: E402


DEFAULT_WNBA_SERIES = ("KXWNBAH", "KXWNBAGAME")


def _best_yes_bid(yes_bids: List[List[int]]) -> tuple[int, int]:
    if not yes_bids:
        return 0, 0
    price, size = max(yes_bids, key=lambda row: int(row[0]))
    return int(price), int(size)


def _summarize_orderbook(payload: Dict[str, Any]) -> Dict[str, Any]:
    book = (payload or {}).get("orderbook") or {}
    yes_bids = book.get("yes") or []
    no_bids = book.get("no") or []
    yes_asks = yes_asks_from_no_bids(no_bids)
    best_bid_c, best_bid_size = _best_yes_bid(yes_bids)
    best_ask_c = yes_asks[0].price_cents if yes_asks else 0
    best_ask_size = yes_asks[0].size if yes_asks else 0
    return {
        "yes_bid_levels": len(yes_bids),
        "no_bid_levels": len(no_bids),
        "best_yes_bid_c": best_bid_c,
        "best_yes_bid_size": best_bid_size,
        "best_yes_ask_c": best_ask_c,
        "best_yes_ask_size": best_ask_size,
        "yes_ask_top5_size": sum(level.size for level in yes_asks[:5]),
    }


def _market_label(market: Dict[str, Any]) -> str:
    pieces = [
        market.get("ticker"),
        market.get("status"),
        market.get("title") or market.get("subtitle") or market.get("event_ticker"),
    ]
    return " | ".join(str(piece) for piece in pieces if piece)


def _find_market(
    client: AuthedKalshiClient,
    series_tickers: List[str],
    status: Optional[str],
    limit: int,
) -> tuple[Optional[str], List[Dict[str, Any]], str]:
    for active_status in [status, None] if status else [None]:
        for series in series_tickers:
            markets = client.list_markets(
                series_ticker=series,
                status=active_status,
                limit=limit,
            )
            if markets:
                ticker = markets[0].get("ticker")
                return ticker, markets, series
    return None, [], ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", help="Specific Kalshi market ticker to smoke-test")
    parser.add_argument("--series-ticker", action="append",
                        help="WNBA series ticker to search; repeatable")
    parser.add_argument("--status", default="open",
                        help="Market status filter for discovery; default=open")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--depth", type=int, default=None)
    args = parser.parse_args()

    cfg = KalshiAuthConfig.from_env(REPO_ROOT / ".env")
    client = AuthedKalshiClient(cfg)

    print("Kalshi WNBA market-data smoke")
    print(f"  base_url: {cfg.base_url}")
    print(f"  trading_enabled: {cfg.trading_enabled}")
    print(f"  access_key: {'set' if cfg.access_key else 'missing'}")

    markets: List[Dict[str, Any]] = []
    selected_ticker = args.ticker
    selected_series = ""
    if not selected_ticker:
        series_tickers = args.series_ticker or list(DEFAULT_WNBA_SERIES)
        selected_ticker, markets, selected_series = _find_market(
            client=client,
            series_tickers=series_tickers,
            status=args.status or None,
            limit=args.limit,
        )
        print(f"  discovery_series: {selected_series or 'none'}")
        print(f"  discovered_markets: {len(markets)}")
        for market in markets[:5]:
            print(f"    - {_market_label(market)}")

    if not selected_ticker:
        raise SystemExit("No WNBA market found. Pass --ticker for a specific market.")

    market_payload = client.get_market(selected_ticker)
    market = market_payload.get("market") or market_payload
    print("Selected market")
    print(f"  ticker: {selected_ticker}")
    print(f"  label: {_market_label(market)}")

    book_payload = client.get_orderbook(selected_ticker, depth=args.depth)
    summary = _summarize_orderbook(book_payload)
    print("Orderbook")
    print(f"  ts_ms: {int(time.time() * 1000)}")
    print(f"  yes_bid_levels: {summary['yes_bid_levels']}")
    print(f"  no_bid_levels: {summary['no_bid_levels']}")
    print(f"  best_yes_bid: {summary['best_yes_bid_c']}c x {summary['best_yes_bid_size']}")
    print(f"  best_yes_ask: {summary['best_yes_ask_c']}c x {summary['best_yes_ask_size']}")
    print(f"  yes_ask_top5_size: {summary['yes_ask_top5_size']}")
    print("OK")


if __name__ == "__main__":
    main()
