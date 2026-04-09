"""
notebooks/polymarket/03_run_ingest.py
──────────────────────────────────────
Master ingestion script for Polymarket WNBA 2025 data.

Produces normalized output tables matching Kalshi layer naming conventions:
  data/polymarket/polymarket_events.csv
  data/polymarket/polymarket_markets.csv
  data/polymarket/polymarket_tokens.csv
  data/polymarket/polymarket_settlements.csv
  data/polymarket/polymarket_trades.csv           (--trades flag)
  data/polymarket/polymarket_prices_history.csv   (--prices flag)
  data/polymarket/polymarket_best_quotes.csv      (--quotes flag, live only)
  data/polymarket/polymarket_orderbooks_live_snapshots.csv (--books flag, live only)

Usage
─────
  # Fast (default): events, markets, tokens, settlements only
  python notebooks/polymarket/03_run_ingest.py

  # With trade history (slow: ~283 API calls)
  python notebooks/polymarket/03_run_ingest.py --trades

  # With price history (very slow: ~566 API calls, one per token)
  python notebooks/polymarket/03_run_ingest.py --prices

  # With live order book snapshots and best quotes (instantaneous)
  python notebooks/polymarket/03_run_ingest.py --books --quotes

  # Everything
  python notebooks/polymarket/03_run_ingest.py --trades --prices --books --quotes

  # Dry run (no writes)
  python notebooks/polymarket/03_run_ingest.py --trades --dry-run

  # Refresh market metadata even if already cached
  python notebooks/polymarket/03_run_ingest.py --markets

Run with:
  /c/Users/arius/anaconda3/envs/kalshi-wnba/python.exe notebooks/polymarket/03_run_ingest.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure project root is on sys.path when running from notebooks/ subdirectory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.polymarket_client import PolymarketClient, PolymarketConfig
from utils.polymarket_ingest import (
    WNBA_TAG_ID,
    extract_settlements,
    fetch_best_quotes,
    fetch_events,
    fetch_markets,
    fetch_orderbook_snapshot,
    fetch_prices_history,
    fetch_trades,
)
from utils.polymarket_normalize import (
    normalize_events,
    normalize_markets,
    normalize_tokens,
    normalize_prices_history,
    normalize_best_quotes,
    normalize_orderbook_snapshots,
    normalize_trades,
    normalize_settlements,
)

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR      = Path("data/polymarket")
RAW_CACHE    = OUT_DIR / "polymarket_markets_raw.json"

# Only include markets with this sportsMarketType for per-market operations
# (prices, trades, books).  None/null markets are props/futures — excluded
# from heavy per-market pulls to keep runtime manageable.
TARGET_MARKET_TYPE = "moneyline"

# Price history parameters
PRICE_FIDELITY = 1     # minutes per bucket (1 = finest available)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(df: pd.DataFrame, name: str, dry_run: bool) -> None:
    """Save DataFrame to CSV (or parquet if pyarrow is available)."""
    if dry_run:
        print(f"  [dry-run] would write {name}: {len(df)} rows")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = OUT_DIR / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  saved {name}.parquet ({len(df)} rows)")
    except ImportError:
        path = OUT_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"  saved {name}.csv ({len(df)} rows)")


def _load_or_fetch_raw_markets(
    client: PolymarketClient,
    *,
    refresh: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Load raw markets from cache, or fetch from API if missing/stale."""
    if RAW_CACHE.exists() and not refresh:
        print(f"  loading raw markets from cache: {RAW_CACHE}")
        with open(RAW_CACHE) as f:
            return json.load(f)

    print(f"  fetching all WNBA markets from Gamma API (tag_id={WNBA_TAG_ID})...")
    markets = fetch_markets(client, WNBA_TAG_ID, verbose=verbose)
    print(f"  fetched {len(markets)} markets")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_CACHE, "w") as f:
        json.dump(markets, f)
    print(f"  cached raw markets to {RAW_CACHE}")
    return markets


def _build_token_map(tokens_df: pd.DataFrame) -> Dict[str, str]:
    """Build token_id → condition_id dict from tokens DataFrame."""
    return dict(zip(tokens_df["token_id"], tokens_df["condition_id"]))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest Polymarket WNBA 2025 data"
    )
    parser.add_argument("--markets", action="store_true",
                        help="Force refresh of market metadata cache")
    parser.add_argument("--trades", action="store_true",
                        help="Fetch public trade history (Data API, ~283 calls)")
    parser.add_argument("--prices", action="store_true",
                        help="Fetch price history (CLOB API, ~566 calls, slow)")
    parser.add_argument("--quotes", action="store_true",
                        help="Fetch live best bid/ask quotes (CLOB, live snapshot only)")
    parser.add_argument("--books", action="store_true",
                        help="Fetch live order book snapshots (CLOB, live snapshot only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run without writing any files")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    dry_run = args.dry_run
    verbose = args.verbose

    client = PolymarketClient(PolymarketConfig())
    errors: List[str] = []

    # ── Step 0: Raw markets cache / refresh ───────────────────────────────────
    print("\n[0] Loading / fetching market metadata...")
    raw_markets = _load_or_fetch_raw_markets(
        client, refresh=args.markets, verbose=verbose
    )

    # ── Step 1: Events ────────────────────────────────────────────────────────
    print("\n[1] Fetching events...")
    raw_events = fetch_events(client, WNBA_TAG_ID, include_closed=True, verbose=verbose)
    print(f"    {len(raw_events)} events")
    events_df = normalize_events(raw_events)
    _save(events_df, "polymarket_events", dry_run)

    # ── Step 2: Markets ───────────────────────────────────────────────────────
    print("\n[2] Normalizing markets...")
    markets_df = normalize_markets(raw_markets)
    _save(markets_df, "polymarket_markets", dry_run)

    moneyline_df = markets_df[markets_df["sports_market_type"] == TARGET_MARKET_TYPE].copy()
    print(f"    total markets: {len(markets_df)}  |  moneyline: {len(moneyline_df)}")

    # ── Step 3: Tokens ────────────────────────────────────────────────────────
    print("\n[3] Extracting tokens...")
    tokens_df = normalize_tokens(raw_markets)
    _save(tokens_df, "polymarket_tokens", dry_run)
    print(f"    {len(tokens_df)} tokens (2 per market)")
    token_map = _build_token_map(tokens_df)

    # ── Step 4: Settlements ───────────────────────────────────────────────────
    print("\n[4] Extracting settlements...")
    raw_settlements = extract_settlements(raw_markets)
    settlements_df  = normalize_settlements(raw_settlements)
    _save(settlements_df, "polymarket_settlements", dry_run)
    print(f"    {len(settlements_df)} resolved markets")

    # ── Step 5: Trades (optional) ─────────────────────────────────────────────
    if args.trades:
        print(f"\n[5] Fetching public trade history "
              f"({len(moneyline_df)} moneyline markets)...")
        all_raw_trades: List[Dict[str, Any]] = []
        for i, row in enumerate(moneyline_df.itertuples()):
            try:
                raw = fetch_trades(client, row.condition_id, verbose=verbose)
                all_raw_trades.extend(raw)
                if verbose or (i + 1) % 25 == 0:
                    print(f"  [{i+1}/{len(moneyline_df)}] {row.condition_id[:16]}..."
                          f"  +{len(raw)} trades  total={len(all_raw_trades)}")
            except Exception as exc:
                msg = f"trades {row.condition_id}: {exc}"
                errors.append(msg)
                print(f"  ERROR: {msg}")

        trades_df = normalize_trades(all_raw_trades)
        _save(trades_df, "polymarket_trades", dry_run)
        print(f"    total trades: {len(trades_df)}")
    else:
        print("\n[5] Trades skipped (pass --trades to fetch)")

    # ── Step 6: Price history (optional) ──────────────────────────────────────
    if args.prices:
        # Build token list from moneyline markets only
        ml_tokens = tokens_df[
            tokens_df["condition_id"].isin(moneyline_df["condition_id"])
        ].copy()
        print(f"\n[6] Fetching price history "
              f"({len(ml_tokens)} tokens from {len(moneyline_df)} moneyline markets)...")
        print("    Note: /prices-history may return empty for resolved markets.")
        print("    Chunks each token into 15-day windows to maximise coverage.")

        all_raw_prices: List[Dict[str, Any]] = []
        empty_count = 0

        for i, row in enumerate(ml_tokens.itertuples()):
            # Use market start/end times as the fetch window
            mkt = markets_df[markets_df["condition_id"] == row.condition_id]
            if mkt.empty:
                continue
            mkt = mkt.iloc[0]

            start_ts = int(mkt["start_ts"].timestamp()) if pd.notna(mkt["start_ts"]) else None
            end_ts   = int(mkt["end_ts"].timestamp())   if pd.notna(mkt["end_ts"])   else None

            if start_ts is None or end_ts is None:
                continue

            try:
                pts = fetch_prices_history(
                    client, row.token_id, start_ts, end_ts,
                    fidelity=PRICE_FIDELITY, verbose=verbose
                )
                if pts:
                    all_raw_prices.extend(pts)
                else:
                    empty_count += 1
                if verbose or (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(ml_tokens)}] token={row.token_id[:12]}..."
                          f"  +{len(pts)} pts  empty_so_far={empty_count}")
            except Exception as exc:
                msg = f"prices_history {row.token_id[:16]}: {exc}"
                errors.append(msg)
                print(f"  ERROR: {msg}")

        prices_df = normalize_prices_history(all_raw_prices, token_map)
        _save(prices_df, "polymarket_prices_history", dry_run)
        print(f"    total price points: {len(prices_df)}  "
              f"empty token responses: {empty_count}/{len(ml_tokens)}")
    else:
        print("\n[6] Price history skipped (pass --prices to fetch)")

    # ── Step 7: Live best quotes (optional) ───────────────────────────────────
    if args.quotes:
        # Filter to ACTIVE non-closed moneyline tokens only — quotes are meaningless
        # for resolved markets (the market is done trading)
        active_tokens = tokens_df[
            tokens_df["condition_id"].isin(
                moneyline_df[
                    moneyline_df["active_flag"] & ~moneyline_df["closed_flag"]
                ]["condition_id"]
            )
        ].copy()
        print(f"\n[7] Fetching live best quotes ({len(active_tokens)} active tokens)...")
        all_raw_quotes: List[Dict[str, Any]] = []
        for i, row in enumerate(active_tokens.itertuples()):
            try:
                q = fetch_best_quotes(client, row.token_id)
                all_raw_quotes.append(q)
                if verbose or (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(active_tokens)}] {row.token_id[:12]}..."
                          f"  bid={q.get('best_bid')} ask={q.get('best_ask')}")
            except Exception as exc:
                msg = f"quotes {row.token_id[:16]}: {exc}"
                errors.append(msg)
                print(f"  ERROR: {msg}")

        quotes_df = normalize_best_quotes(all_raw_quotes, token_map)
        _save(quotes_df, "polymarket_best_quotes", dry_run)
        print(f"    total quote snapshots: {len(quotes_df)}")
    else:
        print("\n[7] Best quotes skipped (pass --quotes to fetch, live only)")

    # ── Step 8: Live order book snapshots (optional) ──────────────────────────
    if args.books:
        active_tokens = tokens_df[
            tokens_df["condition_id"].isin(
                moneyline_df[
                    moneyline_df["active_flag"] & ~moneyline_df["closed_flag"]
                ]["condition_id"]
            )
        ].copy()
        print(f"\n[8] Fetching live order book snapshots "
              f"({len(active_tokens)} active tokens)...")
        print("    Note: only live/current snapshots are available. "
              "Historical OB data does NOT exist in the Polymarket API.")
        all_raw_books: List[Dict[str, Any]] = []
        for i, row in enumerate(active_tokens.itertuples()):
            try:
                book = fetch_orderbook_snapshot(client, row.token_id)
                if book:
                    all_raw_books.append(book)
                if verbose or (i + 1) % 20 == 0:
                    bids = len(book.get("bids", [])) if book else 0
                    asks = len(book.get("asks", [])) if book else 0
                    print(f"  [{i+1}/{len(active_tokens)}] {row.token_id[:12]}..."
                          f"  bids={bids} asks={asks}")
            except Exception as exc:
                msg = f"orderbook {row.token_id[:16]}: {exc}"
                errors.append(msg)
                print(f"  ERROR: {msg}")

        books_df = normalize_orderbook_snapshots(all_raw_books, token_map)
        _save(books_df, "polymarket_orderbooks_live_snapshots", dry_run)
        print(f"    total book levels: {len(books_df)}")
    else:
        print("\n[8] Order book snapshots skipped (pass --books to fetch, live only)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INGEST COMPLETE")
    print("=" * 60)
    print(f"  events:      {len(events_df)}")
    print(f"  markets:     {len(markets_df)}  (moneyline: {len(moneyline_df)})")
    print(f"  tokens:      {len(tokens_df)}")
    print(f"  settlements: {len(settlements_df)}")

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")
        if len(errors) > 20:
            print(f"    ... and {len(errors) - 20} more")
    else:
        print("\n  No errors.")


if __name__ == "__main__":
    main()
