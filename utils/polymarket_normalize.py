"""
utils/polymarket_normalize.py
─────────────────────────────
Convert raw Polymarket API response dicts into clean pandas DataFrames.

All prices stored as float64 (0.0–1.0 probability scale).
All timestamps stored as timezone-aware pandas Timestamps (UTC).

API field type notes
─────────────────────
  Prices / probabilities : float or string "0.74"  → float64 [0.0, 1.0]
  Sizes                  : float or string "100.0"  → float64
  Date-time strings      : ISO 8601 with Z or offset → pd.Timestamp (UTC)
  Unix timestamps        : int (seconds)             → pd.Timestamp (UTC)
  JSON arrays in strings : '["Yes","No"]'            → parsed, then stored as
                           pipe-delimited string or list depending on column

Table outputs
─────────────
  polymarket_events
  polymarket_markets
  polymarket_tokens
  polymarket_prices_history
  polymarket_best_quotes
  polymarket_orderbooks_live_snapshots   (live snapshot only; historical N/A)
  polymarket_trades
  polymarket_settlements
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _fl(val: Any) -> Optional[float]:
    """Parse price/size to float, or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _ts_iso(val: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse ISO 8601 datetime string → pd.Timestamp (UTC), or None."""
    if not val:
        return None
    try:
        return pd.Timestamp(val, tz="UTC")
    except Exception:
        return None


def _ts_unix(val: Any) -> Optional[pd.Timestamp]:
    """Convert Unix int/str timestamp (seconds) → pd.Timestamp (UTC), or None."""
    if val is None:
        return None
    try:
        return pd.Timestamp(int(val), unit="s", tz="UTC")
    except Exception:
        return None


def _parse_json_list(val: Any) -> List[Any]:
    """Parse a JSON-encoded list string or return the list as-is."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def _pipe_join(val: Any) -> Optional[str]:
    """Render a list (or JSON list string) as a pipe-delimited string."""
    items = _parse_json_list(val)
    if not items:
        return None
    return "|".join(str(x) for x in items)


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_events
# ─────────────────────────────────────────────────────────────────────────────

EVENTS_COLS = [
    "event_id",
    "event_slug",
    "event_ticker",
    "event_title",
    "tags",           # pipe-delimited tag IDs
    "start_ts",
    "end_ts",
    "active_flag",
    "closed_flag",
    "archived_flag",
    "neg_risk_flag",
    "volume",
    "liquidity",
    "open_interest",
]


def normalize_events(raw_events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw Gamma event dicts into the polymarket_events table.

    Source endpoint: GET /events?tag_id=100254
    Each event may contain a nested 'markets' list (ignored here; use
    normalize_markets() on the flat /markets response for market rows).

    Parameters
    ----------
    raw_events : list of event dicts from fetch_events()

    Returns
    -------
    pd.DataFrame with columns defined in EVENTS_COLS, sorted by event_id.
    """
    rows = []
    for ev in raw_events:
        tags_raw = ev.get("tags") or []
        if isinstance(tags_raw, str):
            # Tags can arrive as a comma-delimited string of tag IDs
            tags_raw = [t.strip() for t in tags_raw.split(",") if t.strip()]
        rows.append({
            "event_id":      ev.get("id"),
            "event_slug":    ev.get("slug"),
            "event_ticker":  ev.get("ticker"),
            "event_title":   ev.get("title"),
            "tags":          "|".join(str(t) for t in tags_raw) if tags_raw else None,
            "start_ts":      _ts_iso(ev.get("startDate")),
            "end_ts":        _ts_iso(ev.get("endDate")),
            "active_flag":   bool(ev.get("active")),
            "closed_flag":   bool(ev.get("closed")),
            "archived_flag": bool(ev.get("archived")),
            "neg_risk_flag": bool(ev.get("negRisk")),
            "volume":        _fl(ev.get("volume")),
            "liquidity":     _fl(ev.get("liquidity")),
            "open_interest": _fl(ev.get("openInterest")),
        })

    df = pd.DataFrame(rows, columns=EVENTS_COLS)
    df = df.sort_values("event_id").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_markets
# ─────────────────────────────────────────────────────────────────────────────

MARKETS_COLS = [
    "market_id",          # Gamma integer ID
    "condition_id",       # 0x hex CTF condition ID (primary join key)
    "market_slug",
    "question",
    "description",        # market rules / resolution source (may be empty)
    "resolved_by",        # resolution authority string
    "active_flag",
    "closed_flag",
    "archived_flag",
    "accepting_orders_flag",
    "enable_order_book_flag",
    "clear_book_on_start",   # True for sports markets: book wiped at game start
    "neg_risk_flag",
    "sports_market_type",    # 'moneyline' | None
    "game_id",               # sports game identifier
    "start_ts",              # market open time
    "end_ts",                # market close / resolution deadline
    "game_start_ts",         # scheduled game tip-off (sports markets only)
    "closed_ts",             # actual close/resolution time (closedTime field)
    "outcomes",              # pipe-delimited outcome labels  e.g. "Aces|Liberty"
    "outcome_prices",        # pipe-delimited implied probs   e.g. "0.45|0.55"
    "clob_token_ids",        # pipe-delimited CLOB token IDs (Yes|No order)
    "volume",
    "liquidity",
    "volume_24hr",
    "last_trade_price",
    "best_bid",
    "best_ask",
]


def normalize_markets(raw_markets: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw Gamma market dicts into the polymarket_markets table.

    Source endpoint: GET /markets?tag_id=100254
    Includes ALL market types (moneyline + props/futures).
    Filter on sports_market_type == 'moneyline' for game-winner contracts.

    Pregame trading window field:
      Use game_start_ts as the end of the pregame window, NOT end_ts.
      clear_book_on_start=True means all limit orders are cancelled at
      game_start_ts.  Live (in-game) trading continues after that point.

    Parameters
    ----------
    raw_markets : list of market dicts from fetch_markets()

    Returns
    -------
    pd.DataFrame with columns defined in MARKETS_COLS,
    sorted by (condition_id).
    """
    rows = []
    for m in raw_markets:
        rows.append({
            "market_id":              m.get("id"),
            "condition_id":           m.get("conditionId"),
            "market_slug":            m.get("slug"),
            "question":               m.get("question"),
            "description":            m.get("description") or m.get("resolutionSource"),
            "resolved_by":            m.get("resolvedBy"),
            "active_flag":            bool(m.get("active")),
            "closed_flag":            bool(m.get("closed")),
            "archived_flag":          bool(m.get("archived")),
            "accepting_orders_flag":  bool(m.get("acceptingOrders")),
            "enable_order_book_flag": bool(m.get("enableOrderBook")),
            "clear_book_on_start":    bool(m.get("clearBookOnStart")),
            "neg_risk_flag":          bool(m.get("negRisk")),
            "sports_market_type":     m.get("sportsMarketType"),
            "game_id":                m.get("gameId"),
            "start_ts":               _ts_iso(m.get("startDate")),
            "end_ts":                 _ts_iso(m.get("endDate")),
            "game_start_ts":          _ts_iso(m.get("gameStartTime")),
            "closed_ts":              _ts_iso(
                                          str(m.get("closedTime"))
                                          if m.get("closedTime") else None
                                      ),
            "outcomes":               _pipe_join(m.get("outcomes")),
            "outcome_prices":         _pipe_join(m.get("outcomePrices")),
            "clob_token_ids":         _pipe_join(m.get("clobTokenIds")),
            "volume":                 _fl(m.get("volumeNum") or m.get("volume")),
            "liquidity":              _fl(m.get("liquidityNum") or m.get("liquidity")),
            "volume_24hr":            _fl(m.get("volume24hr")),
            "last_trade_price":       _fl(m.get("lastTradePrice")),
            "best_bid":               _fl(m.get("bestBid")),
            "best_ask":               _fl(m.get("bestAsk")),
        })

    df = pd.DataFrame(rows, columns=MARKETS_COLS)
    df = df.sort_values("condition_id").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_tokens
# ─────────────────────────────────────────────────────────────────────────────

TOKENS_COLS = [
    "condition_id",    # parent market
    "market_id",       # parent Gamma market ID
    "token_id",        # CLOB token ID / asset ID
    "outcome_label",   # e.g. "Aces", "Liberty", "Yes", "No"
    "token_index",     # 0 = first outcome (conventionally Yes/Team A), 1 = second
    "outcome_price",   # implied probability at time of metadata fetch
]


def normalize_tokens(raw_markets: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Expand raw market dicts into one row per outcome token.

    This is the key table linking condition_id (Gamma) to token_id (CLOB).
    Use token_id when calling /prices-history, /book, /price, etc.

    Source: clobTokenIds and outcomes arrays embedded in each market dict.
    token_index=0 is conventionally the first outcome (Yes or Team A).
    token_index=1 is conventionally the second outcome (No or Team B).

    Parameters
    ----------
    raw_markets : list of market dicts from fetch_markets()

    Returns
    -------
    pd.DataFrame with columns defined in TOKENS_COLS,
    sorted by (condition_id, token_index).
    """
    rows = []
    for m in raw_markets:
        condition_id = m.get("conditionId")
        market_id    = m.get("id")
        token_ids    = _parse_json_list(m.get("clobTokenIds"))
        outcomes     = _parse_json_list(m.get("outcomes"))
        prices       = _parse_json_list(m.get("outcomePrices"))

        for i, token_id in enumerate(token_ids):
            rows.append({
                "condition_id":  condition_id,
                "market_id":     market_id,
                "token_id":      token_id,
                "outcome_label": outcomes[i] if i < len(outcomes) else None,
                "token_index":   i,
                "outcome_price": _fl(prices[i]) if i < len(prices) else None,
            })

    df = pd.DataFrame(rows, columns=TOKENS_COLS)
    df = df.sort_values(["condition_id", "token_index"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_prices_history
# ─────────────────────────────────────────────────────────────────────────────

PRICES_HISTORY_COLS = [
    "token_id",
    "condition_id",    # joinable via polymarket_tokens
    "ts",              # pd.Timestamp (UTC) — bucket start
    "price",           # float [0.0, 1.0]
]


def normalize_prices_history(
    raw_points: List[Dict[str, Any]],
    token_to_condition: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize raw price history points from /prices-history into the
    polymarket_prices_history table.

    Parameters
    ----------
    raw_points         : list of dicts with keys: token_id, t, p
                         (as returned by fetch_prices_history() with
                          token_id embedded by the ingest function)
    token_to_condition : optional dict mapping token_id → condition_id
                         (from polymarket_tokens). Used to populate
                         the condition_id column for easy joins.

    Returns
    -------
    pd.DataFrame with columns defined in PRICES_HISTORY_COLS,
    sorted by (token_id, ts).
    """
    token_to_condition = token_to_condition or {}
    rows = []
    for pt in raw_points:
        token_id = pt.get("token_id")
        rows.append({
            "token_id":     token_id,
            "condition_id": token_to_condition.get(token_id),
            "ts":           _ts_unix(pt.get("t")),
            "price":        _fl(pt.get("p")),
        })

    df = pd.DataFrame(rows, columns=PRICES_HISTORY_COLS)
    df = df.sort_values(["token_id", "ts"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_best_quotes  (live snapshots only)
# ─────────────────────────────────────────────────────────────────────────────

BEST_QUOTES_COLS = [
    "token_id",
    "condition_id",
    "snapshot_ts",
    "best_bid",
    "best_ask",
    "midpoint",
    "spread",
    "last_trade_price",
    "tick_size",
]


def normalize_best_quotes(
    raw_quotes: List[Dict[str, Any]],
    token_to_condition: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize raw quote dicts from fetch_best_quotes() into the
    polymarket_best_quotes table.

    Note: These are LIVE snapshots only.
    Historical bid/ask data is not available through the Polymarket API.

    Parameters
    ----------
    raw_quotes         : list of quote dicts from fetch_best_quotes()
    token_to_condition : optional token_id → condition_id mapping

    Returns
    -------
    pd.DataFrame with columns defined in BEST_QUOTES_COLS,
    sorted by (token_id, snapshot_ts).
    """
    token_to_condition = token_to_condition or {}
    rows = []
    for q in raw_quotes:
        token_id = q.get("token_id")
        rows.append({
            "token_id":         token_id,
            "condition_id":     token_to_condition.get(token_id),
            "snapshot_ts":      _ts_unix(q.get("snapshot_ts")),
            "best_bid":         _fl(q.get("best_bid")),
            "best_ask":         _fl(q.get("best_ask")),
            "midpoint":         _fl(q.get("midpoint")),
            "spread":           _fl(q.get("spread")),
            "last_trade_price": _fl(q.get("last_trade_price")),
            "tick_size":        _fl(q.get("tick_size")),
        })

    df = pd.DataFrame(rows, columns=BEST_QUOTES_COLS)
    df = df.sort_values(["token_id", "snapshot_ts"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_orderbooks_live_snapshots  (live only — historical N/A)
# ─────────────────────────────────────────────────────────────────────────────

ORDERBOOK_COLS = [
    "token_id",
    "condition_id",
    "snapshot_ts",
    "side",           # "bid" | "ask"
    "price",
    "size",
    "level",          # 1 = best, 2 = second-best, etc.
    "tick_size",
    "neg_risk",
]


def normalize_orderbook_snapshots(
    raw_books: List[Dict[str, Any]],
    token_to_condition: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Normalize raw order book snapshots (from fetch_orderbook_snapshot or
    fetch_orderbook_snapshots_bulk) into the
    polymarket_orderbooks_live_snapshots table.

    IMPORTANT: Historical order book snapshots are NOT available via any
    Polymarket REST endpoint.  This table contains only live snapshots
    collected at the time of ingestion.  The table name is intentionally
    'polymarket_orderbooks_live_snapshots' to make this explicit.

    One row per price level per side per snapshot.

    Parameters
    ----------
    raw_books          : list of OrderBookSummary dicts from CLOB /book
    token_to_condition : optional token_id → condition_id mapping

    Returns
    -------
    pd.DataFrame with columns defined in ORDERBOOK_COLS,
    sorted by (token_id, snapshot_ts, side, level).
    """
    token_to_condition = token_to_condition or {}
    rows = []
    for book in raw_books:
        token_id    = book.get("token_id") or book.get("asset_id")
        snap_ts     = book.get("snapshot_ts") or book.get("timestamp")
        condition_id = token_to_condition.get(token_id) or book.get("market")
        tick_size   = _fl(book.get("tick_size"))
        neg_risk    = bool(book.get("neg_risk"))

        for side_key, side_label in [("bids", "bid"), ("asks", "ask")]:
            for level, entry in enumerate(book.get(side_key, []), start=1):
                rows.append({
                    "token_id":    token_id,
                    "condition_id": condition_id,
                    "snapshot_ts": _ts_unix(snap_ts),
                    "side":        side_label,
                    "price":       _fl(entry.get("price")),
                    "size":        _fl(entry.get("size")),
                    "level":       level,
                    "tick_size":   tick_size,
                    "neg_risk":    neg_risk,
                })

    df = pd.DataFrame(rows, columns=ORDERBOOK_COLS)
    if not df.empty:
        df = df.sort_values(["token_id", "snapshot_ts", "side", "level"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_trades
# ─────────────────────────────────────────────────────────────────────────────

TRADES_COLS = [
    "token_id",        # CLOB asset ID (outcome token)
    "condition_id",    # market condition ID
    "trade_ts",        # pd.Timestamp (UTC)
    "price",           # float [0.0, 1.0]
    "size",            # float (USDC notional)
    "side",            # "BUY" | "SELL" (taker side)
    "outcome",         # outcome label e.g. "Yes", "Aces"
    "outcome_index",   # 0 or 1
    "tx_hash",         # on-chain transaction hash
]


def normalize_trades(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw trade dicts from Data API /trades into the
    polymarket_trades table.

    Source endpoint: GET https://data-api.polymarket.com/trades
    Response fields used: asset, conditionId, timestamp, price, size,
                          side, outcome, outcomeIndex, transactionHash.

    Parameters
    ----------
    raw_trades : list of trade dicts from fetch_trades() or
                 fetch_trades_by_token()

    Returns
    -------
    pd.DataFrame with columns defined in TRADES_COLS,
    sorted by (condition_id, trade_ts).
    """
    rows = []
    for t in raw_trades:
        rows.append({
            "token_id":     t.get("asset"),
            "condition_id": t.get("conditionId"),
            "trade_ts":     _ts_unix(t.get("timestamp")),
            "price":        _fl(t.get("price")),
            "size":         _fl(t.get("size")),
            "side":         t.get("side"),
            "outcome":      t.get("outcome"),
            "outcome_index":t.get("outcomeIndex"),
            "tx_hash":      t.get("transactionHash"),
        })

    df = pd.DataFrame(rows, columns=TRADES_COLS)
    if not df.empty:
        df = df.sort_values(["condition_id", "trade_ts"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# polymarket_settlements
# ─────────────────────────────────────────────────────────────────────────────

SETTLEMENTS_COLS = [
    "condition_id",
    "market_id",
    "resolved_flag",
    "winning_outcome",
    "settlement_ts",    # closedTime if available, else endDate (proxy)
    "final_status",     # "finalized" | "open"
    "outcomes",         # pipe-delimited outcome labels
    "outcome_prices",   # pipe-delimited final prices ("1" for winner)
]


def normalize_settlements(raw_settlements: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw settlement dicts from extract_settlements() into the
    polymarket_settlements table.

    There is NO dedicated settlement endpoint in Polymarket.
    Settlement data is derived from Gamma market metadata:
      - closed=True         → resolved
      - outcomePrices = "1" → winning outcome
      - closedTime          → settlement timestamp (use endDate if absent)

    Note: resolvedAt / settlementTs do NOT exist in the Gamma schema.
    settlement_ts is best-effort using closedTime → endDate as fallback.

    Parameters
    ----------
    raw_settlements : list of settlement dicts from extract_settlements()

    Returns
    -------
    pd.DataFrame with columns defined in SETTLEMENTS_COLS,
    sorted by condition_id.
    """
    rows = []
    for s in raw_settlements:
        outcomes_list = s.get("outcomes", [])
        prices_list   = s.get("outcome_prices", [])
        rows.append({
            "condition_id":    s.get("condition_id"),
            "market_id":       s.get("market_id"),
            "resolved_flag":   bool(s.get("resolved_flag")),
            "winning_outcome": s.get("winning_outcome"),
            "settlement_ts":   _ts_iso(str(s.get("settlement_ts"))
                                       if s.get("settlement_ts") else None),
            "final_status":    s.get("final_status"),
            "outcomes":        "|".join(str(o) for o in outcomes_list) if outcomes_list else None,
            "outcome_prices":  "|".join(str(p) for p in prices_list)   if prices_list  else None,
        })

    df = pd.DataFrame(rows, columns=SETTLEMENTS_COLS)
    df = df.sort_values("condition_id").reset_index(drop=True)
    return df
