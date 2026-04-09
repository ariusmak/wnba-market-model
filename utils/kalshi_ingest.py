"""
utils/kalshi_ingest.py
──────────────────────
Functions to fetch raw Kalshi API data for a set of market tickers.

Routing logic
─────────────
Kalshi partitions data into a live tier and a historical tier.  The
boundary is determined by GET /historical/cutoff.  All 2025 WNBA markets
(finalized by October 2025) fall in the historical tier as of March 2026.

  Markets / candles  →  GET /historical/markets  &  GET /historical/markets/{ticker}/candlesticks
  Trades             →  GET /historical/trades

The live fallbacks (GET /markets, GET /markets/trades) are also provided
for completeness but are not needed for the 2025 backtest data.

Order books
───────────
GET /markets/{ticker}/orderbook returns the CURRENT live book only.
Historical orderbook snapshots are NOT available through the Kalshi API.
This function is NOT implemented; see kalshi_api_schema.md for details.

Pagination
──────────
All list endpoints use cursor-based pagination.  Pass the cursor returned
in each response back as the `cursor` query param.  An empty or absent
cursor means the last page.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from utils.kalshi_client import KalshiClient, KalshiConfig

# ── Max items per page (API limit is 1000) ────────────────────────────────────
PAGE_SIZE = 1000

# ── Max candlestick window per request (undocumented; use 30-day chunks) ──────
CANDLE_CHUNK_DAYS = 30


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _paginate(
    client: KalshiClient,
    path: str,
    result_key: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch all pages for a cursor-paginated list endpoint.

    Parameters
    ----------
    path       : API path, e.g. "/historical/markets"
    result_key : key in the JSON response that holds the list, e.g. "markets"
    params     : base query params (do not include cursor or limit here)
    verbose    : print page count progress

    Returns
    -------
    Flat list of all result objects across all pages.
    """
    params = dict(params or {})
    params["limit"] = PAGE_SIZE
    results: List[Dict] = []
    cursor: Optional[str] = None
    page = 0

    while True:
        if cursor:
            params["cursor"] = cursor
        elif "cursor" in params:
            del params["cursor"]

        data = client.get_json(path, params)
        batch = data.get(result_key, [])
        results.extend(batch)
        cursor = data.get("cursor") or None
        page += 1

        if verbose:
            print(f"    page {page}: got {len(batch)} {result_key}, total={len(results)}")

        if not cursor or not batch:
            break

    return results


def _ts(dt: datetime) -> int:
    """datetime → Unix timestamp (int)."""
    return int(dt.timestamp())


# ─────────────────────────────────────────────────────────────────────────────
# Cutoff
# ─────────────────────────────────────────────────────────────────────────────

def get_cutoff(client: KalshiClient) -> Dict[str, str]:
    """
    GET /historical/cutoff

    Returns dict with keys:
      market_settled_ts  : ISO datetime string
      trades_created_ts  : ISO datetime string
      orders_updated_ts  : ISO datetime string

    Markets settled BEFORE market_settled_ts must be fetched from
    /historical/markets.  Trades filled BEFORE trades_created_ts must
    come from /historical/trades.
    """
    return client.get_json("/historical/cutoff")


# ─────────────────────────────────────────────────────────────────────────────
# Markets
# ─────────────────────────────────────────────────────────────────────────────

def fetch_historical_markets(
    client: KalshiClient,
    tickers: List[str],
    *,
    batch_size: int = 200,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /historical/markets  — fetch metadata for settled markets.

    The endpoint does NOT support series_ticker filtering.  We filter
    by providing explicit tickers (comma-separated, max ~200 per call).

    Parameters
    ----------
    tickers    : list of market tickers to fetch
    batch_size : how many tickers per API call (no documented max)
    verbose    : print progress

    Returns
    -------
    List of raw market dicts.
    """
    all_markets: List[Dict] = []
    for i in range(0, len(tickers), batch_size):
        chunk = tickers[i : i + batch_size]
        if verbose:
            print(f"  historical markets batch {i // batch_size + 1}: {len(chunk)} tickers")
        batch = _paginate(
            client,
            "/historical/markets",
            "markets",
            params={"tickers": ",".join(chunk)},
            verbose=verbose,
        )
        all_markets.extend(batch)
    return all_markets


def fetch_live_markets(
    client: KalshiClient,
    series_ticker: str,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /markets  — fetch markets from the live tier by series_ticker.
    Use for markets NOT yet in the historical tier.
    """
    return _paginate(
        client,
        "/markets",
        "markets",
        params={"series_ticker": series_ticker},
        verbose=verbose,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Candlesticks
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_timerange(
    start_ts: int,
    end_ts: int,
    chunk_days: int = CANDLE_CHUNK_DAYS,
) -> Iterator[Tuple[int, int]]:
    """Yield (chunk_start, chunk_end) tuples covering [start_ts, end_ts]."""
    chunk_s = chunk_days * 86_400
    cur = start_ts
    while cur < end_ts:
        yield cur, min(cur + chunk_s, end_ts)
        cur += chunk_s


def fetch_historical_candles(
    client: KalshiClient,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /historical/markets/{ticker}/candlesticks

    Fetches historical OHLC candlesticks for a single market.
    Chunks the time range to avoid potential server-side limits.

    Parameters
    ----------
    ticker          : market ticker, e.g. "KXWNBAGAME-25MAY22INDATL-IND"
    start_ts        : Unix timestamp (inclusive lower bound)
    end_ts          : Unix timestamp (inclusive upper bound)
    period_interval : 1 (minute), 60 (hour), or 1440 (day)

    Returns
    -------
    List of raw candlestick dicts with end_period_ts de-duplicated
    (chunk overlap protection).
    """
    seen: set[int] = set()
    candles: List[Dict] = []
    path = f"/historical/markets/{ticker}/candlesticks"

    for chunk_start, chunk_end in _chunk_timerange(start_ts, end_ts):
        if verbose:
            print(f"    candles {ticker}: [{chunk_start}, {chunk_end}]")
        data = client.get_json(path, {
            "start_ts": chunk_start,
            "end_ts": chunk_end,
            "period_interval": period_interval,
        })
        for c in data.get("candlesticks", []):
            ts = c.get("end_period_ts")
            if ts not in seen:
                seen.add(ts)
                c["ticker"] = ticker  # embed ticker for downstream use
                candles.append(c)

    return candles


def fetch_live_candles(
    client: KalshiClient,
    series_ticker: str,
    ticker: str,
    start_ts: int,
    end_ts: int,
    period_interval: int = 1,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /series/{series_ticker}/markets/{ticker}/candlesticks

    Live-tier candlesticks for markets not yet in the historical tier.
    """
    seen: set[int] = set()
    candles: List[Dict] = []
    path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"

    for chunk_start, chunk_end in _chunk_timerange(start_ts, end_ts):
        if verbose:
            print(f"    candles {ticker}: [{chunk_start}, {chunk_end}]")
        data = client.get_json(path, {
            "start_ts": chunk_start,
            "end_ts": chunk_end,
            "period_interval": period_interval,
        })
        for c in data.get("candlesticks", []):
            ts = c.get("end_period_ts")
            if ts not in seen:
                seen.add(ts)
                c["ticker"] = ticker
                candles.append(c)

    return candles


# ─────────────────────────────────────────────────────────────────────────────
# Trades
# ─────────────────────────────────────────────────────────────────────────────

def fetch_historical_trades(
    client: KalshiClient,
    ticker: str,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /historical/trades

    Fetches trades from the historical tier for a single market ticker.

    Parameters
    ----------
    ticker : market ticker
    min_ts : Unix timestamp lower bound (inclusive)
    max_ts : Unix timestamp upper bound (inclusive)
    """
    params: Dict[str, Any] = {"ticker": ticker}
    if min_ts is not None:
        params["min_ts"] = min_ts
    if max_ts is not None:
        params["max_ts"] = max_ts
    return _paginate(client, "/historical/trades", "trades", params=params, verbose=verbose)


def fetch_live_trades(
    client: KalshiClient,
    ticker: str,
    min_ts: Optional[int] = None,
    max_ts: Optional[int] = None,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /markets/trades

    Fetches trades from the live tier for a single market ticker.
    """
    params: Dict[str, Any] = {"ticker": ticker}
    if min_ts is not None:
        params["min_ts"] = min_ts
    if max_ts is not None:
        params["max_ts"] = max_ts
    return _paginate(client, "/markets/trades", "trades", params=params, verbose=verbose)


def fetch_all_trades(
    client: KalshiClient,
    ticker: str,
    market_open_time: datetime,
    market_close_time: datetime,
    cutoff_ts: Optional[int] = None,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Smart-route trades fetch: historical tier if pre-cutoff, live otherwise.

    For the 2025 WNBA backtest all markets are pre-cutoff so this always
    hits the historical endpoint.  cutoff_ts=None forces historical.
    """
    min_ts = _ts(market_open_time)
    max_ts = _ts(market_close_time)

    if cutoff_ts is None or min_ts < cutoff_ts:
        return fetch_historical_trades(client, ticker, min_ts, max_ts, verbose=verbose)
    else:
        return fetch_live_trades(client, ticker, min_ts, max_ts, verbose=verbose)
