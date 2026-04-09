"""
Notebook 03: Ingest All Kalshi WNBA 2025 Data
==============================================
Pulls and stores all required Kalshi datasets for the v1 backtest:

  1. kalshi_markets        – market metadata for all KXWNBAGAME contracts
  2. kalshi_candles_1m     – 1-minute OHLC per market
  3. kalshi_trades         – every historical trade per market
  4. kalshi_settlements    – final result / settlement data per contract

  kalshi_orderbook_snapshots: NOT AVAILABLE
    The Kalshi API only exposes the current live order book (auth required).
    Historical orderbook snapshots cannot be retrieved. See schema doc.

Outputs (all in data/kalshi/):
  kalshi_markets.parquet
  kalshi_candles_1m.parquet
  kalshi_trades.parquet
  kalshi_settlements.parquet

Run from repo root:
  python notebooks/kalshi/03_ingest_all_data.py [--dry-run] [--candles] [--trades]

Flags:
  --dry-run  : fetch and normalize data but do not write files
  --candles  : include candlestick ingestion (slow; ~596 API calls)
  --trades   : include trade ingestion (fast; 1 call per market)
  --markets  : refresh market metadata (otherwise uses cached raw JSON)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Make utils importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.kalshi_client import KalshiClient, KalshiConfig
from utils.kalshi_ingest import (
    fetch_historical_candles,
    fetch_historical_markets,
    fetch_historical_trades,
    fetch_live_candles,
    fetch_live_trades,
    get_cutoff,
)
from utils.kalshi_normalize import (
    normalize_candles,
    normalize_markets,
    normalize_settlements,
    normalize_trades,
)

# ── Config ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = Path("data/kalshi")
RAW_MARKETS  = OUTPUT_DIR / "wnba_2025_markets_raw.json"
SERIES       = "KXWNBAGAME"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── CLI args ───────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kalshi WNBA 2025 data ingestion")
    p.add_argument("--dry-run",  action="store_true", help="Do not write output files")
    p.add_argument("--markets",  action="store_true", help="Re-fetch market metadata from API")
    p.add_argument("--candles",  action="store_true", help="Fetch 1-min candlesticks")
    p.add_argument("--trades",   action="store_true", help="Fetch historical trades")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _save(df: pd.DataFrame, name: str, dry_run: bool) -> None:
    # Try parquet; fall back to CSV if pyarrow is not installed.
    try:
        import pyarrow  # noqa: F401
        path = OUTPUT_DIR / f"{name}.parquet"
        fmt = "parquet"
    except ImportError:
        path = OUTPUT_DIR / f"{name}.csv"
        fmt = "csv"

    if dry_run:
        print(f"  [dry-run] would save {len(df):,} rows -> {path}")
        return

    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")

    size_kb = path.stat().st_size / 1024
    print(f"  saved {len(df):,} rows -> {path}  ({size_kb:.0f} KB)")


def _load_raw_markets() -> list:
    if not RAW_MARKETS.exists():
        raise FileNotFoundError(
            f"{RAW_MARKETS} not found. Run notebook 01 first, or pass --markets."
        )
    with open(RAW_MARKETS, encoding="utf-8") as f:
        return json.load(f)


# ── Step 1: Market metadata ────────────────────────────────────────────────────
def ingest_markets(client: KalshiClient, refresh: bool, dry_run: bool) -> pd.DataFrame:
    print("\n[1] Market metadata")

    if refresh:
        print("  fetching from API (historical tier)...")
        # We need tickers; load from the existing raw file for the ticker list.
        if RAW_MARKETS.exists():
            existing = _load_raw_markets()
            tickers = [m["ticker"] for m in existing]
        else:
            raise RuntimeError(
                "Cannot refresh without an existing ticker list. "
                "Run notebook 01 first to populate wnba_2025_markets_raw.json."
            )
        raw = fetch_historical_markets(client, tickers, verbose=True)
        # Save new raw JSON
        out_raw = OUTPUT_DIR / "wnba_2025_markets_raw_refreshed.json"
        if not dry_run:
            with open(out_raw, "w", encoding="utf-8") as f:
                json.dump(raw, f, indent=2)
            print(f"  raw JSON saved → {out_raw}")
    else:
        print(f"  using cached {RAW_MARKETS}")
        raw = _load_raw_markets()

    df = normalize_markets(raw)
    print(f"  {len(df):,} market rows, {df['event_ticker'].nunique()} unique events")
    _save(df, "kalshi_markets", dry_run)
    return df


# ── Step 2: Settlements ────────────────────────────────────────────────────────
def ingest_settlements(raw_markets: list, dry_run: bool) -> pd.DataFrame:
    print("\n[2] Settlements")
    df = normalize_settlements(raw_markets)
    settled = df[df["result"].isin(["yes", "no"])]
    print(f"  {len(df):,} settled contracts ({df['result'].value_counts().to_dict()})")
    _save(df, "kalshi_settlements", dry_run)
    return df


# ── Step 3: Candlesticks (1-minute) ───────────────────────────────────────────
def ingest_candles(
    client: KalshiClient,
    markets_df: pd.DataFrame,
    dry_run: bool,
) -> pd.DataFrame:
    print("\n[3] 1-minute candlesticks")

    # Only ingest markets that have open_time and close_time
    valid = markets_df[
        markets_df["open_time"].notna() & markets_df["close_time"].notna()
    ].copy()
    print(f"  {len(valid):,} markets with valid time windows")

    all_candles = []
    errors = []
    t0 = time.monotonic()

    for i, row in valid.iterrows():
        ticker    = row["market_ticker"]
        open_ts   = int(row["open_time"].timestamp())
        close_ts  = int(row["close_time"].timestamp())
        progress  = f"[{valid.index.get_loc(i)+1}/{len(valid)}]"

        try:
            candles = fetch_live_candles(
                client, SERIES, ticker, open_ts, close_ts, period_interval=1
            )
            all_candles.extend(candles)
            elapsed = time.monotonic() - t0
            print(f"  {progress} {ticker}: {len(candles):,} candles  "
                  f"(total={len(all_candles):,}, elapsed={elapsed:.0f}s)")
        except Exception as exc:
            errors.append((ticker, str(exc)))
            print(f"  {progress} {ticker}: ERROR — {exc}")

    if errors:
        print(f"\n  {len(errors)} errors:")
        for t, e in errors:
            print(f"    {t}: {e}")

    if not all_candles:
        print("  no candles fetched")
        return pd.DataFrame()

    df = normalize_candles(all_candles)
    print(f"\n  {len(df):,} total candle rows across {df['market_ticker'].nunique()} markets")
    _save(df, "kalshi_candles_1m", dry_run)
    return df


# ── Step 4: Trades ────────────────────────────────────────────────────────────
def ingest_trades(
    client: KalshiClient,
    markets_df: pd.DataFrame,
    dry_run: bool,
) -> pd.DataFrame:
    print("\n[4] Historical trades")

    valid = markets_df[
        markets_df["open_time"].notna() & markets_df["close_time"].notna()
    ].copy()
    print(f"  {len(valid):,} markets to pull trades for")

    all_trades = []
    errors = []
    t0 = time.monotonic()

    for i, row in valid.iterrows():
        ticker   = row["market_ticker"]
        min_ts   = int(row["open_time"].timestamp())
        max_ts   = int(row["close_time"].timestamp())
        progress = f"[{valid.index.get_loc(i)+1}/{len(valid)}]"

        try:
            trades = fetch_live_trades(client, ticker, min_ts, max_ts)
            all_trades.extend(trades)
            elapsed = time.monotonic() - t0
            print(f"  {progress} {ticker}: {len(trades):,} trades  "
                  f"(total={len(all_trades):,}, elapsed={elapsed:.0f}s)")
        except Exception as exc:
            errors.append((ticker, str(exc)))
            print(f"  {progress} {ticker}: ERROR — {exc}")

    if errors:
        print(f"\n  {len(errors)} errors:")
        for t, e in errors:
            print(f"    {t}: {e}")

    if not all_trades:
        print("  no trades fetched")
        return pd.DataFrame()

    df = normalize_trades(all_trades)
    print(f"\n  {len(df):,} total trade rows across {df['market_ticker'].nunique()} markets")
    _save(df, "kalshi_trades", dry_run)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    run_candles = args.candles
    run_trades  = args.trades

    if not run_candles and not run_trades and not args.markets:
        # Default: run markets + settlements only (fast)
        print("No data flags set — running markets + settlements only.")
        print("Add --candles and/or --trades to fetch time-series data.")

    client = KalshiClient(KalshiConfig())

    # ── Cutoff (informational) ────────────────────────────────────────────────
    print("\n[0] Historical cutoff")
    try:
        cutoff = get_cutoff(client)
        print(f"  market_settled_ts : {cutoff.get('market_settled_ts')}")
        print(f"  trades_created_ts : {cutoff.get('trades_created_ts')}")
    except Exception as exc:
        print(f"  WARNING: could not fetch cutoff — {exc}")

    # ── Markets ───────────────────────────────────────────────────────────────
    raw_markets = _load_raw_markets()
    markets_df  = ingest_markets(client, refresh=args.markets, dry_run=args.dry_run)

    # ── Settlements ───────────────────────────────────────────────────────────
    ingest_settlements(raw_markets, dry_run=args.dry_run)

    # ── Candlesticks ──────────────────────────────────────────────────────────
    if run_candles:
        ingest_candles(client, markets_df, dry_run=args.dry_run)
    else:
        print("\n[3] Candlesticks: skipped (pass --candles to enable)")

    # ── Trades ────────────────────────────────────────────────────────────────
    if run_trades:
        ingest_trades(client, markets_df, dry_run=args.dry_run)
    else:
        print("\n[4] Trades: skipped (pass --trades to enable)")

    print("\nDone.")


if __name__ == "__main__":
    main()
