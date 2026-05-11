"""
04_entry_loop_dryrun.py
=======================

End-to-end dry-run of EntryLoop with a fake Kalshi client and a 2025 gold
row. No network, no live orders — verifies that:

  1. FinalModel trains on the 2015-2024 CSV and scores a real feature row
  2. EntryLoop polls both sides, plans sweeps via trader.plan_sweep
  3. Orders are "submitted" to the fake client and state updates
  4. The event log is a well-formed JSONL stream

Run from repo root:
    python pipelines/07_live/legacy/04_entry_loop_dryrun.py

The fake client uses a deterministic book (deep YES asks at prices under
our model's max_price cap) so we should see at least one sweep per side
over ~5 poll iterations, capped at the half-Kelly target.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.legacy.entry_loop import EntryLoop, GameContext  # noqa: E402
from srwnba.live.legacy.trader import TradeConfig  # noqa: E402
from srwnba.util.final_model import FinalModel, _cold_start_mask  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Fake Kalshi client (no network)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class FakeKalshi:
    """Minimal stand-in for AuthedKalshiClient for dry-run tests.

    - get_orderbook returns a deterministic book (passed in per ticker).
    - create_order fills partially based on the book and shrinks depth
      so repeated polls eventually hit the target cap.
    """
    books: Dict[str, Dict[str, List[List[int]]]] = field(default_factory=dict)
    orders_log: List[Dict[str, Any]] = field(default_factory=list)

    def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict[str, Any]:
        book = self.books.get(ticker, {"yes": [], "no": []})
        # Return a shallow copy so mutation inside EntryLoop can't bleed
        return {"orderbook": {"yes": list(book.get("yes", [])),
                              "no": [list(lv) for lv in book.get("no", [])]}}

    def create_order(
        self, *, ticker: str, action: str, side: str, count: int,
        order_type: str, client_order_id: str,
        yes_price_cents: Optional[int] = None,
        no_price_cents: Optional[int] = None,
        time_in_force: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        assert action == "buy" and side == "yes"
        assert order_type == "limit" and time_in_force == "IOC"
        assert yes_price_cents is not None

        # Walk our NO-bid "book" and fill from the cheapest YES equivalents.
        book = self.books.setdefault(ticker, {"yes": [], "no": []})
        no_bids: List[List[int]] = book.get("no", [])
        # Sort so lowest YES-ask (highest NO-bid) consumed first
        no_bids.sort(key=lambda row: row[0], reverse=True)
        remaining = count
        filled = 0
        fill_cost_cents = 0
        new_bids: List[List[int]] = []
        for row in no_bids:
            n_price, size = row[0], row[1]
            yes_price = 100 - n_price
            if remaining <= 0 or yes_price > yes_price_cents:
                new_bids.append(row)
                continue
            take = min(size, remaining)
            filled += take
            fill_cost_cents += take * yes_price
            remaining -= take
            left = size - take
            if left > 0:
                new_bids.append([n_price, left])
        book["no"] = new_bids

        order_id = f"fake-{uuid.uuid4().hex[:8]}"
        resp = {"order": {
            "order_id": order_id,
            "status": "executed",
            "client_order_id": client_order_id,
            "filled_count": filled,
            "taker_fill_cost": fill_cost_cents,
        }}
        self.orders_log.append({"ticker": ticker, "count": count,
                                "yes_price_cents": yes_price_cents,
                                "filled": filled, "cost_cents": fill_cost_cents})
        return resp


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",
                    default=str(REPO_ROOT / "data" / "gold" / "game_xgboost_input_2015_2024_REGPST.csv"))
    ap.add_argument("--holdout-csv",
                    default=str(REPO_ROOT / "data" / "gold" / "game_xgboost_input_2025_REGPST.csv"))
    ap.add_argument("--polls", type=int, default=4,
                    help="Number of poll iterations to run before stopping")
    ap.add_argument("--log-path",
                    default=str(REPO_ROOT / "data" / "live_logs" / "dryrun_events.jsonl"))
    args = ap.parse_args()

    # 1. Train FinalModel and pick a real pregame row from 2025
    print("[dryrun] training FinalModel...")
    predictor = FinalModel(args.train_csv)
    df = pd.read_csv(args.holdout_csv)
    df = df[_cold_start_mask(df)].reset_index(drop=True)
    row = df.iloc[[0]].copy()
    p_home = predictor.predict_single(row)
    print(f"[dryrun] row game_id={row['game_id'].iloc[0]} p_home={p_home:.4f}")

    # 2. Build a fake book. Deep asks on each side — Kalshi exposes this
    #    as NO bids, with YES ask = 100 - NO bid.
    home_ticker = "KXWNBAH-TEST-HOME"
    away_ticker = "KXWNBAH-TEST-AWAY"
    fake = FakeKalshi(books={
        # Home book: NO bids at 48 (size 500) and 46 (size 800) → YES asks at 52 and 54
        home_ticker: {"no": [[48, 500], [46, 800]]},
        # Away book: NO bids at 60 (size 200), 58 (size 400) → YES asks at 40, 42
        away_ticker: {"no": [[60, 200], [58, 400]]},
    })

    # 3. GameContext with tipoff far in future; we'll fake the clock to
    #    stop after N polls.
    log_path = Path(args.log_path)
    if log_path.exists():
        log_path.unlink()

    ctx = GameContext(
        game_id=str(row["game_id"].iloc[0]),
        home_ticker=home_ticker,
        away_ticker=away_ticker,
        tipoff_ts_s=int(time.time()) + 10_000,
        feature_row=row,
    )
    loop = EntryLoop(
        predictor=predictor,
        client=fake,
        ctx=ctx,
        cfg=TradeConfig(),
        log_path=log_path,
        poll_interval_s=0.0,
        dry_run=False,  # hit the fake create_order so we exercise fill logic
    )

    # 4. Simulate a virtual clock. We'll tick once per poll and stop early.
    counter = {"i": 0}
    stop_after = args.polls

    def fake_now() -> float:
        counter["i"] += 1
        if counter["i"] > stop_after:
            return ctx.tipoff_ts_s + 1  # past tipoff → loop exits
        return ctx.tipoff_ts_s - 1000.0

    loop.run(wall_clock_now_s=fake_now)

    # 5. Report
    print(f"[dryrun] final home filled={loop.home.filled_contracts} "
          f"cost_cents={loop.home.total_cost_cents} fee=${loop.home.total_fee_dollars:.2f}")
    print(f"[dryrun] final away filled={loop.away.filled_contracts} "
          f"cost_cents={loop.away.total_cost_cents} fee=${loop.away.total_fee_dollars:.2f}")
    print(f"[dryrun] fake orders submitted: {len(fake.orders_log)}")
    for i, ord_ in enumerate(fake.orders_log):
        print(f"    #{i+1} {ord_}")

    # 6. Verify event log is parseable JSONL
    assert log_path.exists(), log_path
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    print(f"[dryrun] event log rows: {len(events)}")
    kinds: Dict[str, int] = {}
    for ev in events:
        kinds[ev["evt"]] = kinds.get(ev["evt"], 0) + 1
    print(f"[dryrun] event kinds: {kinds}")

    # Sanity: we should see at least one plan and at least one fill
    assert kinds.get("plan", 0) >= stop_after * 2  # two sides per poll
    assert (loop.home.filled_contracts + loop.away.filled_contracts) > 0, \
        "expected at least one side to fill under the fake deep book"
    print("[dryrun] OK")


if __name__ == "__main__":
    main()
