"""
utils/kalshi_normalize.py
─────────────────────────
Convert raw Kalshi API response dicts into clean pandas DataFrames.

All dollar amounts stored as float64 (converted from FixedPointDollars strings).
All contract counts stored as float64 (converted from FixedPointCount strings).
All timestamps stored as timezone-aware pandas Timestamps (UTC).

Data type notes
───────────────
  FixedPointDollars : string with ≤6 decimal places, e.g. "0.560000"
                      → float64 (dollars, [0.0, 1.0] for binary markets)
  FixedPointCount   : string with 2 decimal places, e.g. "100.00"
                      → float64 (number of contracts)
  date-time strings : ISO 8601 with Z or offset suffix
                      → pd.Timestamp (UTC)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


# ── helpers ───────────────────────────────────────────────────────────────────

def _fp(val: Optional[str]) -> Optional[float]:
    """Parse FixedPointDollars / FixedPointCount string → float, or None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _ts(val: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse ISO datetime string → pd.Timestamp (UTC), or None."""
    if not val:
        return None
    try:
        return pd.Timestamp(val, tz="UTC")
    except Exception:
        return None


def _unix_to_ts(val: Optional[int]) -> Optional[pd.Timestamp]:
    """Convert Unix integer timestamp → pd.Timestamp (UTC), or None."""
    if val is None:
        return None
    try:
        return pd.Timestamp(val, unit="s", tz="UTC")
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# kalshi_markets
# ─────────────────────────────────────────────────────────────────────────────

MARKETS_COLS = [
    "market_ticker",
    "event_ticker",
    "series_ticker",          # derived: prefix before first '-' in event_ticker
    "title",
    "yes_sub_title",
    "no_sub_title",
    "status",
    "market_type",
    "open_time",
    "close_time",
    "expected_expiration_time",
    "settlement_ts",
    "rules_primary",
    "rules_secondary",
    "expiration_value",       # text description of what triggers settlement
    "can_close_early",
    "early_close_condition",
    "result",                 # "yes" | "no" | "" for unsettled
    "settlement_value_dollars",
    "last_price_dollars",
    "volume_fp",
    "open_interest_fp",
    "custom_strike",          # JSON blob (team identifier for sports markets)
]


def normalize_markets(raw_markets: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize a list of raw market dicts (from /markets or /historical/markets)
    into the kalshi_markets table.

    Parameters
    ----------
    raw_markets : list of market dicts from the API

    Returns
    -------
    pd.DataFrame with columns defined in MARKETS_COLS
    """
    rows = []
    for m in raw_markets:
        et = m.get("event_ticker", "")
        # Derive series_ticker: everything before the second segment.
        # e.g. "KXWNBAGAME-25MAY22INDATL" → "KXWNBAGAME"
        series_ticker = et.split("-")[0] if et else None

        rows.append({
            "market_ticker":            m.get("ticker"),
            "event_ticker":             et,
            "series_ticker":            series_ticker,
            "title":                    m.get("title"),
            "yes_sub_title":            m.get("yes_sub_title"),
            "no_sub_title":             m.get("no_sub_title"),
            "status":                   m.get("status"),
            "market_type":              m.get("market_type"),
            "open_time":                _ts(m.get("open_time")),
            "close_time":               _ts(m.get("close_time")),
            "expected_expiration_time": _ts(m.get("expected_expiration_time")),
            "settlement_ts":            _ts(m.get("settlement_ts")),
            "rules_primary":            m.get("rules_primary"),
            "rules_secondary":          m.get("rules_secondary"),
            "expiration_value":         m.get("expiration_value"),
            "can_close_early":          m.get("can_close_early"),
            "early_close_condition":    m.get("early_close_condition"),
            "result":                   m.get("result", ""),
            "settlement_value_dollars": _fp(m.get("settlement_value_dollars")),
            "last_price_dollars":       _fp(m.get("last_price_dollars")),
            "volume_fp":                _fp(m.get("volume_fp")),
            "open_interest_fp":         _fp(m.get("open_interest_fp")),
            "custom_strike":            str(m.get("custom_strike")) if m.get("custom_strike") else None,
        })

    df = pd.DataFrame(rows, columns=MARKETS_COLS)
    df = df.sort_values(["event_ticker", "market_ticker"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# kalshi_settlements
# ─────────────────────────────────────────────────────────────────────────────

SETTLEMENTS_COLS = [
    "market_ticker",
    "event_ticker",
    "yes_sub_title",          # team that this "yes" position represents
    "status",
    "result",                 # "yes" | "no"
    "settlement_value_dollars",
    "settlement_ts",
    "close_time",
    "expiration_value",       # settled-to team name (from API)
]


def normalize_settlements(raw_markets: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract settlement information from raw market metadata.

    Only includes markets that have a non-empty result field (settled).

    Returns
    -------
    pd.DataFrame with columns defined in SETTLEMENTS_COLS
    """
    rows = []
    for m in raw_markets:
        result = m.get("result", "")
        if not result:
            continue
        rows.append({
            "market_ticker":            m.get("ticker"),
            "event_ticker":             m.get("event_ticker"),
            "yes_sub_title":            m.get("yes_sub_title"),
            "status":                   m.get("status"),
            "result":                   result,
            "settlement_value_dollars": _fp(m.get("settlement_value_dollars")),
            "settlement_ts":            _ts(m.get("settlement_ts")),
            "close_time":               _ts(m.get("close_time")),
            "expiration_value":         m.get("expiration_value"),
        })

    df = pd.DataFrame(rows, columns=SETTLEMENTS_COLS)
    df = df.sort_values(["event_ticker", "market_ticker"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# kalshi_candles_1m
# ─────────────────────────────────────────────────────────────────────────────

CANDLES_COLS = [
    "market_ticker",
    "end_period_ts",
    # YES bid OHLC
    "yes_bid_open",
    "yes_bid_high",
    "yes_bid_low",
    "yes_bid_close",
    # YES ask OHLC
    "yes_ask_open",
    "yes_ask_high",
    "yes_ask_low",
    "yes_ask_close",
    # Trade price OHLC + extras
    "price_open",
    "price_high",
    "price_low",
    "price_close",
    "price_mean",
    "price_previous",
    # Volume / OI
    "volume",
    "open_interest",
]


def _ohlc(sub: Optional[Dict], prefix: str) -> Dict[str, Optional[float]]:
    """
    Extract open/high/low/close from a BidAskDistributionHistorical dict.
    Keys in the dict: open, high, low, close (with _dollars suffix).
    """
    if not sub:
        return {f"{prefix}_open": None, f"{prefix}_high": None,
                f"{prefix}_low": None, f"{prefix}_close": None}
    return {
        f"{prefix}_open":  _fp(sub.get("open_dollars") or sub.get("open")),
        f"{prefix}_high":  _fp(sub.get("high_dollars") or sub.get("high")),
        f"{prefix}_low":   _fp(sub.get("low_dollars")  or sub.get("low")),
        f"{prefix}_close": _fp(sub.get("close_dollars") or sub.get("close")),
    }


def normalize_candles(raw_candles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw candlestick dicts into the kalshi_candles_1m table.

    Each dict must have a "ticker" key (embedded by the ingest functions).

    Parameters
    ----------
    raw_candles : list of candlestick dicts (mixed tickers is fine)

    Returns
    -------
    pd.DataFrame with columns defined in CANDLES_COLS, sorted by
    (market_ticker, end_period_ts).
    """
    rows = []
    for c in raw_candles:
        price = c.get("price") or {}
        row = {
            "market_ticker":  c.get("ticker"),
            "end_period_ts":  _unix_to_ts(c.get("end_period_ts")),
        }
        row.update(_ohlc(c.get("yes_bid"), "yes_bid"))
        row.update(_ohlc(c.get("yes_ask"), "yes_ask"))

        row["price_open"]     = _fp(price.get("open_dollars")     or price.get("open"))
        row["price_high"]     = _fp(price.get("high_dollars")     or price.get("high"))
        row["price_low"]      = _fp(price.get("low_dollars")      or price.get("low"))
        row["price_close"]    = _fp(price.get("close_dollars")    or price.get("close"))
        row["price_mean"]     = _fp(price.get("mean_dollars")     or price.get("mean"))
        row["price_previous"] = _fp(price.get("previous_dollars") or price.get("previous"))

        # Both historical and live responses use slightly different keys
        row["volume"]        = _fp(c.get("volume_fp") or c.get("volume"))
        row["open_interest"] = _fp(c.get("open_interest_fp") or c.get("open_interest"))

        rows.append(row)

    df = pd.DataFrame(rows, columns=CANDLES_COLS)
    df = df.sort_values(["market_ticker", "end_period_ts"]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# kalshi_trades
# ─────────────────────────────────────────────────────────────────────────────

TRADES_COLS = [
    "trade_id",
    "market_ticker",
    "trade_ts",
    "yes_price",
    "no_price",
    "count",          # number of contracts traded
    "taker_side",     # "yes" | "no"
]


def normalize_trades(raw_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize raw trade dicts into the kalshi_trades table.

    Parameters
    ----------
    raw_trades : list of trade dicts from /historical/trades or /markets/trades

    Returns
    -------
    pd.DataFrame with columns defined in TRADES_COLS, sorted by
    (market_ticker, trade_ts).
    """
    rows = []
    for t in raw_trades:
        rows.append({
            "trade_id":     t.get("trade_id"),
            "market_ticker": t.get("ticker"),
            "trade_ts":     _ts(t.get("created_time")),
            "yes_price":    _fp(t.get("yes_price_dollars")),
            "no_price":     _fp(t.get("no_price_dollars")),
            "count":        _fp(t.get("count_fp")),
            "taker_side":   t.get("taker_side"),
        })

    df = pd.DataFrame(rows, columns=TRADES_COLS)
    df = df.sort_values(["market_ticker", "trade_ts"]).reset_index(drop=True)
    return df
