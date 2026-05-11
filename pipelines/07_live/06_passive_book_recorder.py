"""
06_passive_book_recorder.py
===========================

Polls Kalshi orderbook snapshots for one or more tickers on a fixed
cadence and appends each snapshot to a Parquet/CSV log. No orders are
ever submitted — this is telemetry only.

Two uses:
  1. Pre-live calibration: record WNBA moneyline books at offseason /
     exhibition games to see realistic depth distribution vs what the
     backtest assumed.
  2. End-to-end auth smoke test: first run once credentials are in .env
     will surface any signing/endpoint issues without financial risk.

Usage (from repo root):
    python pipelines/07_live/06_passive_book_recorder.py \
        --ticker KXWNBAH-26MAY15ATLIND-IND \
        --ticker KXWNBAH-26MAY15ATLIND-ATL \
        --duration-s 300 --interval-s 5

Output CSV columns:
    ts_ms, ticker, best_yes_ask_c, best_yes_ask_size,
    best_no_ask_c, best_no_ask_size,
    yes_top5_size, no_top5_size, yes_bids_json, no_bids_json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.kalshi_authed_client import AuthedKalshiClient  # noqa: E402
from srwnba.live.common import yes_asks_from_no_bids  # noqa: E402


CSV_COLUMNS = [
    "ts_ms", "ticker",
    "best_yes_ask_c", "best_yes_ask_size",
    "best_yes_bid_c", "best_yes_bid_size",
    "yes_top5_size", "no_top5_size",
    "yes_bids_json", "no_bids_json",
]


def summarize_book(
    ticker: str, payload: Dict[str, Any], ts_ms: int,
) -> Dict[str, Any]:
    book = (payload or {}).get("orderbook") or {}
    yes_bids: List[List[int]] = book.get("yes") or []
    no_bids: List[List[int]] = book.get("no") or []

    # Buying YES lifts NO bids; buying NO lifts YES bids.
    yes_asks = yes_asks_from_no_bids(no_bids)
    # Best YES bid = highest yes_bids entry
    best_yes_bid_c = 0
    best_yes_bid_size = 0
    if yes_bids:
        yes_bids_sorted = sorted(yes_bids, key=lambda r: r[0], reverse=True)
        best_yes_bid_c, best_yes_bid_size = int(yes_bids_sorted[0][0]), int(yes_bids_sorted[0][1])

    best_yes_ask_c = yes_asks[0].price_cents if yes_asks else 0
    best_yes_ask_size = yes_asks[0].size if yes_asks else 0

    yes_top5_size = sum(lv.size for lv in yes_asks[:5])
    no_top5_size  = sum(int(r[1]) for r in sorted(yes_bids, key=lambda r: r[0], reverse=True)[:5])

    return {
        "ts_ms": ts_ms,
        "ticker": ticker,
        "best_yes_ask_c": best_yes_ask_c,
        "best_yes_ask_size": best_yes_ask_size,
        "best_yes_bid_c": best_yes_bid_c,
        "best_yes_bid_size": best_yes_bid_size,
        "yes_top5_size": yes_top5_size,
        "no_top5_size": no_top5_size,
        "yes_bids_json": json.dumps(yes_bids),
        "no_bids_json":  json.dumps(no_bids),
    }


_stop = {"flag": False}

def _install_sigint() -> None:
    def handler(signum, frame):  # noqa: ARG001
        _stop["flag"] = True
    try:
        signal.signal(signal.SIGINT, handler)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", action="append", required=True,
                    help="Repeatable — one --ticker per market to record")
    ap.add_argument("--duration-s", type=int, default=600,
                    help="Stop after N seconds; 0 = run until Ctrl-C")
    ap.add_argument("--interval-s", type=float, default=5.0,
                    help="Seconds between full-sweep polls of all tickers")
    ap.add_argument("--out",
                    default=str(REPO_ROOT / "data" / "live_logs" / "books.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("passive_book")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()

    client = AuthedKalshiClient()
    log.info("polling %d tickers every %.1fs → %s", len(args.ticker),
             args.interval_s, out_path)
    _install_sigint()

    deadline = time.time() + args.duration_s if args.duration_s > 0 else None
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        polls = 0
        while not _stop["flag"]:
            if deadline and time.time() >= deadline:
                break
            for t in args.ticker:
                try:
                    payload = client.get_orderbook(t)
                except Exception as exc:
                    log.warning("get_orderbook(%s) failed: %s", t, exc)
                    continue
                row = summarize_book(t, payload, int(time.time() * 1000))
                writer.writerow(row)
                f.flush()
                log.info("  %s best_ask=%d¢ sz=%d  best_bid=%d¢ sz=%d  top5_ask=%d top5_bid=%d",
                         t, row["best_yes_ask_c"], row["best_yes_ask_size"],
                         row["best_yes_bid_c"], row["best_yes_bid_size"],
                         row["yes_top5_size"], row["no_top5_size"])
            polls += 1
            time.sleep(args.interval_s)

    log.info("done — %d poll rounds written to %s", polls, out_path)


if __name__ == "__main__":
    main()
