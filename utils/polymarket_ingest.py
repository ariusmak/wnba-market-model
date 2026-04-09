"""
utils/polymarket_ingest.py
──────────────────────────
Functions to fetch raw Polymarket API data for WNBA markets.

API architecture
────────────────
  Gamma API  https://gamma-api.polymarket.com    Market/event discovery, metadata
  CLOB API   https://clob.polymarket.com          Price history, live order book, quotes
  Data API   https://data-api.polymarket.com      Public trade history (no auth required)

WNBA tag: tag_id = 100254

Key behavioral notes (confirmed)
──────────────────────────────────────────────────────────────────────────────
1. /prices-history is TOKEN-level (not market-level).
   Pass one token_id at a time.  Yes and No tokens must be fetched separately.
   The `market` param is the CLOB token ID (not the condition ID).

2. /prices-history returns empty data ({history: []}) for resolved/closed
   markets when using interval=max or interval=all.
   Workaround: use explicit startTs/endTs with ≤15-day chunks.
   Even with explicit timestamps, very old resolved markets may return empty.
   Fall back to Data API /trades if price history is empty.

3. Historical order book snapshots are NOT available via any REST endpoint.
   Only live (current) snapshots are available via CLOB /book.
   This module only collects live snapshots; no historical OB reconstruction.

4. CLOB /trades requires L2 authentication.  Public trade history is fetched
   from Data API /trades (no auth required).

5. Sports markets have clearBookOnStart=true: all resting limit orders are
   cancelled at game start time, but live trading continues after.

6. No resolvedAt / settlementTs field in Gamma schema.
   Use endDate as the settlement timestamp proxy.

7. Data API /trades offset max = 10,000.  For markets with >10,000 trades,
   chunk by time.  WNBA game markets are short-lived; this is rarely an issue.

Pagination
──────────
  Gamma /events and /markets:  offset-based (limit + offset).
  Data API /trades:             offset-based (limit + offset).
  CLOB endpoints:               no pagination (single-object or list param).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from utils.polymarket_client import PolymarketClient, PolymarketConfig

# ── Constants ─────────────────────────────────────────────────────────────────
WNBA_TAG_ID = 100254

GAMMA_PAGE_SIZE = 100    # Gamma /markets and /events max per page
DATA_PAGE_SIZE  = 1_000  # Data API /trades max per page (documented max 10,000)
DATA_MAX_OFFSET = 10_000 # Data API hard offset ceiling per query window

# /prices-history: chunk size to avoid empty-response issue on resolved markets
PRICE_HISTORY_CHUNK_DAYS = 15


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gamma_paginate(
    client: PolymarketClient,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Offset-paginate a Gamma list endpoint.
    Stops when the batch is smaller than GAMMA_PAGE_SIZE.
    """
    params = dict(params or {})
    params["limit"] = GAMMA_PAGE_SIZE
    results: List[Dict] = []
    offset = 0
    page = 0

    while True:
        params["offset"] = offset
        batch = client.gamma_get(path, params)
        if not isinstance(batch, list):
            break
        results.extend(batch)
        page += 1
        if verbose:
            print(f"    Gamma {path} page {page}: {len(batch)} rows, total={len(results)}")
        if len(batch) < GAMMA_PAGE_SIZE:
            break
        offset += GAMMA_PAGE_SIZE

    return results


def _ts_to_unix(dt: datetime) -> int:
    """datetime → Unix timestamp (int seconds, UTC)."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _chunk_timerange(
    start_ts: int,
    end_ts: int,
    chunk_days: int = PRICE_HISTORY_CHUNK_DAYS,
) -> Iterator[Tuple[int, int]]:
    """Yield (chunk_start, chunk_end) tuples covering [start_ts, end_ts]."""
    chunk_s = chunk_days * 86_400
    cur = start_ts
    while cur < end_ts:
        yield cur, min(cur + chunk_s, end_ts)
        cur += chunk_s


# ─────────────────────────────────────────────────────────────────────────────
# Events  (Gamma API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_events(
    client: PolymarketClient,
    tag_id: int = WNBA_TAG_ID,
    *,
    include_closed: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /events?tag_id=<tag_id>

    Returns all WNBA events (each contains nested child markets).
    Set include_closed=False to fetch only currently active events.

    Endpoint: https://gamma-api.polymarket.com/events
    Params  : tag_id, active (optional), closed (optional), limit, offset
    Auth    : None
    Notes   : active=true + closed=false to restrict to live events.
              include_closed=True (default) retrieves full historical set.
    """
    params: Dict[str, Any] = {"tag_id": tag_id}
    if not include_closed:
        params["active"] = "true"
        params["closed"] = "false"
    return _gamma_paginate(client, "/events", params, verbose=verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Markets  (Gamma API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_markets(
    client: PolymarketClient,
    tag_id: int = WNBA_TAG_ID,
    *,
    include_closed: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /markets?tag_id=<tag_id>

    Returns all WNBA markets (flat list; no nesting).
    Includes both moneyline (individual game) and prop/futures markets.

    Endpoint : https://gamma-api.polymarket.com/markets
    Params   : tag_id, limit, offset
    Auth     : None
    Known quirks:
      - Adding 'order' or 'sports_market_types' params causes HTTP 422.
        Filter client-side instead.
      - include_closed has no effect here; all markets are returned regardless.
        The include_closed parameter is accepted for API parity but ignored.
    """
    params: Dict[str, Any] = {"tag_id": tag_id}
    return _gamma_paginate(client, "/markets", params, verbose=verbose)


def fetch_single_market(
    client: PolymarketClient,
    condition_id: str,
) -> Optional[Dict[str, Any]]:
    """
    GET /markets?condition_ids=<condition_id>

    Fetch full metadata for one market by its condition ID.
    Returns None if not found.
    """
    result = client.gamma_get("/markets", {"condition_ids": condition_id})
    if isinstance(result, list) and result:
        return result[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Price history  (CLOB API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices_history(
    client: PolymarketClient,
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 1,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /prices-history?market=<token_id>&startTs=X&endTs=Y&fidelity=F

    Returns 1-minute (or coarser) price history for a single outcome token.

    IMPORTANT:
      - 'market' param is the CLOB TOKEN ID (from clobTokenIds), not conditionId.
      - Yes and No tokens must be fetched SEPARATELY.
      - interval=max/all fails on resolved markets → use explicit startTs/endTs.
      - Chunks the time range into ≤15-day windows to avoid empty responses.
      - Returns empty list for very old resolved markets with no data.

    Endpoint : https://clob.polymarket.com/prices-history
    Params   : market (token_id), startTs, endTs, fidelity
    Auth     : None
    Response : {"history": [{"t": <unix_seconds>, "p": <float 0-1>}, ...]}
    Fidelity : minutes per data point. 1 = 1-min candles, 60 = hourly, etc.
    """
    seen: set = set()
    points: List[Dict] = []

    for chunk_start, chunk_end in _chunk_timerange(start_ts, end_ts):
        if verbose:
            print(f"    prices-history token={token_id[:12]}... [{chunk_start},{chunk_end}]")
        resp = client.clob_get("/prices-history", {
            "market":   token_id,
            "startTs":  chunk_start,
            "endTs":    chunk_end,
            "fidelity": fidelity,
        })
        for pt in resp.get("history", []):
            t = pt.get("t")
            if t not in seen:
                seen.add(t)
                pt["token_id"] = token_id
                points.append(pt)

    return points


# ─────────────────────────────────────────────────────────────────────────────
# Live order book snapshot  (CLOB API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_orderbook_snapshot(
    client: PolymarketClient,
    token_id: str,
) -> Optional[Dict[str, Any]]:
    """
    GET /book?token_id=<token_id>

    Returns the CURRENT live order book for a single outcome token.

    IMPORTANT: Historical order book snapshots are NOT available via any
    Polymarket REST endpoint.  This function returns only the live/current
    state.  The returned dict includes a 'timestamp' field (Unix seconds)
    that marks when the snapshot was taken.

    Endpoint : https://clob.polymarket.com/book
    Params   : token_id
    Auth     : None
    Response : OrderBookSummary (see polymarket_ingest_spec.md for full schema)
    """
    resp = client.clob_get("/book", {"token_id": token_id})
    if not resp or "bids" not in resp:
        return None
    resp["token_id"] = token_id
    resp["snapshot_ts"] = int(time.time())
    return resp


def fetch_orderbook_snapshots_bulk(
    client: PolymarketClient,
    token_ids: List[str],
) -> List[Dict[str, Any]]:
    """
    POST /books (bulk)  — fetch live order books for multiple tokens at once.

    Endpoint : https://clob.polymarket.com/books
    Body     : array of {"token_id": "..."}
    Auth     : None
    Note     : Still returns only live snapshots, not historical.
    """
    import requests as _req
    snap_ts = int(time.time())
    resp = _req.post(
        f"{client.cfg.clob_url}/books",
        json=[{"token_id": tid} for tid in token_ids],
        timeout=client.cfg.timeout_s,
    )
    resp.raise_for_status()
    results = resp.json()
    for i, book in enumerate(results):
        if i < len(token_ids):
            book["token_id"] = token_ids[i]
        book["snapshot_ts"] = snap_ts
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Best bid / ask / midpoint / spread  (CLOB API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_best_quotes(
    client: PolymarketClient,
    token_id: str,
) -> Dict[str, Any]:
    """
    Fetch best bid, best ask, midpoint, spread, last trade price, and tick size
    for a single outcome token.  Makes 3 parallel calls then assembles result.

    Endpoints used:
      GET /price?token_id=&side=BUY  →  best ask  (price to buy Yes)
      GET /price?token_id=&side=SELL →  best bid  (price to sell Yes)
      GET /midpoint?token_id=
      GET /spread?token_id=
      GET /last-trade-price?token_id=
      GET /tick-size?token_id=
    Auth: None
    """
    snap_ts = int(time.time())
    result: Dict[str, Any] = {"token_id": token_id, "snapshot_ts": snap_ts}

    def _safe(path: str, params: Dict) -> Any:
        try:
            return client.clob_get(path, params)
        except Exception:
            return {}

    ask_resp        = _safe("/price",            {"token_id": token_id, "side": "BUY"})
    bid_resp        = _safe("/price",            {"token_id": token_id, "side": "SELL"})
    mid_resp        = _safe("/midpoint",         {"token_id": token_id})
    spread_resp     = _safe("/spread",           {"token_id": token_id})
    last_resp       = _safe("/last-trade-price", {"token_id": token_id})
    tick_resp       = _safe("/tick-size",        {"token_id": token_id})

    result["best_ask"]          = ask_resp.get("price")
    result["best_bid"]          = bid_resp.get("price")
    result["midpoint"]          = mid_resp.get("mid")
    result["spread"]            = spread_resp.get("spread")
    result["last_trade_price"]  = last_resp.get("price")
    result["tick_size"]         = tick_resp.get("minimum_tick_size")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Public trade history  (Data API)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_trades(
    client: PolymarketClient,
    condition_id: str,
    *,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /trades?market=<condition_id>&limit=N&offset=N

    Returns all public trades for a single market (condition ID).
    No authentication required.

    Endpoint : https://data-api.polymarket.com/trades
    Params   : market (condition_id), limit, offset, takerOnly
    Auth     : None
    Paginate : offset-based; hard ceiling of 10,000 records per query window.
               Use start_ts/end_ts chunking for very high-volume markets.
    Notes    :
      - This is the correct public source for trade history.
      - CLOB /trades requires L2 auth and is NOT used here.
      - takerOnly=False returns all trades (maker + taker fills).
      - Response includes: asset (token_id), conditionId, side, price, size,
        timestamp, outcome, transactionHash.
    Rate limit: 200 req/10 s.
    """
    all_trades: List[Dict] = []
    offset = 0

    while True:
        params: Dict[str, Any] = {
            "market":    condition_id,
            "limit":     DATA_PAGE_SIZE,
            "offset":    offset,
            "takerOnly": "false",  # return all fills, not just taker side
        }
        batch = client.data_get("/trades", params)
        if not isinstance(batch, list) or not batch:
            break
        all_trades.extend(batch)
        if verbose:
            print(f"    trades market={condition_id[:12]}... offset={offset}: {len(batch)} rows")
        if len(batch) < DATA_PAGE_SIZE:
            break
        offset += DATA_PAGE_SIZE
        if offset >= DATA_MAX_OFFSET:
            # Hard ceiling reached for this query window.
            # Log a warning; time-chunking would be needed for higher volumes.
            if verbose:
                print(f"    WARNING: hit Data API offset ceiling ({DATA_MAX_OFFSET}) "
                      f"for market {condition_id[:16]}... Trades may be incomplete.")
            break

    return all_trades


def fetch_trades_by_token(
    client: PolymarketClient,
    token_id: str,
    *,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    GET /trades?asset=<token_id>

    Alternative to fetch_trades() when you have only a token ID.
    Useful as a fallback for reconstructing price history on resolved markets
    when /prices-history returns empty data.

    Note: 'asset' param targets a single outcome token; use this to pull
    trades for Yes or No side separately.
    """
    all_trades: List[Dict] = []
    offset = 0

    while True:
        params: Dict[str, Any] = {
            "asset":     token_id,
            "limit":     DATA_PAGE_SIZE,
            "offset":    offset,
            "takerOnly": "false",
        }
        batch = client.data_get("/trades", params)
        if not isinstance(batch, list) or not batch:
            break
        all_trades.extend(batch)
        if verbose:
            print(f"    trades token={token_id[:12]}... offset={offset}: {len(batch)} rows")
        if len(batch) < DATA_PAGE_SIZE:
            break
        offset += DATA_PAGE_SIZE
        if offset >= DATA_MAX_OFFSET:
            break

    return all_trades


# ─────────────────────────────────────────────────────────────────────────────
# Settlements / resolved outcome  (derived from Gamma market metadata)
# ─────────────────────────────────────────────────────────────────────────────

def extract_settlements(
    markets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract settlement information from raw Gamma market metadata.

    There is NO dedicated settlement endpoint in Polymarket.
    Settlement data is embedded in the market metadata object:
      - closed=True        → market is resolved
      - outcomePrices      → "1" for the winning outcome, "0" for the loser
      - endDate / endDateIso → resolution timestamp proxy (no resolvedAt field)

    Parameters
    ----------
    markets : list of raw market dicts from fetch_markets()

    Returns
    -------
    List of settlement dicts (one per resolved market).
    """
    settlements = []
    for m in markets:
        if not m.get("closed"):
            continue
        import json as _json
        outcomes = m.get("outcomes", "[]")
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        prices = m.get("outcomePrices", "[]")
        if isinstance(prices, str):
            prices = _json.loads(prices)

        winning_outcome = None
        for outcome, price in zip(outcomes, prices):
            try:
                if float(price) == 1.0:
                    winning_outcome = outcome
                    break
            except (ValueError, TypeError):
                pass

        settlements.append({
            "condition_id":    m.get("conditionId"),
            "market_id":       m.get("id"),
            "resolved_flag":   m.get("closed", False),
            "winning_outcome": winning_outcome,
            "settlement_ts":   m.get("closedTime") or m.get("endDate"),
            "final_status":    "finalized" if m.get("closed") else "open",
            "outcomes":        outcomes,
            "outcome_prices":  prices,
        })

    return settlements
