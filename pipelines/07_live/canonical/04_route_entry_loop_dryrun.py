"""
04_route_entry_loop_dryrun.py
=============================

End-to-end dry-run of RouteEntryLoop with a fake Kalshi client.

No network, no live orders. This verifies:
  1. FinalModel scores a real feature row
  2. Kalshi market mapping confirms a two-market event
  3. Equivalent routes are built for selected-team-wins exposure
  4. Execution planning can allocate to BUY YES or BUY NO routes
  5. Planned orders are submitted to the fake client and fills update
     canonical exposure state
  6. JSONL audit log is parseable
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.canonical.execution import ExecutionConfig  # noqa: E402
from srwnba.live.canonical.game_ledger import GameLedger  # noqa: E402
from srwnba.live.canonical.kalshi_mapping import SportRadarGameRef  # noqa: E402
from srwnba.live.canonical.route_entry_loop import RouteEntryContext, RouteEntryLoop  # noqa: E402
from srwnba.util.final_model import FinalModel, _cold_start_mask  # noqa: E402


@dataclass
class FakeRouteKalshi:
    markets: List[Dict[str, Any]]
    books: Dict[str, Dict[str, List[List[int]]]]
    orders_log: List[Dict[str, Any]] = field(default_factory=list)

    def list_markets(
        self,
        *,
        series_ticker: Optional[str] = None,
        event_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        out = []
        for market in self.markets:
            if series_ticker and not market["event_ticker"].startswith(series_ticker):
                continue
            if event_ticker and market["event_ticker"] != event_ticker:
                continue
            if status and market.get("status") != status:
                continue
            out.append(market)
        return out[:limit]

    def get_orderbook(self, ticker: str, depth: Optional[int] = None) -> Dict[str, Any]:
        book = self.books.get(ticker, {"yes": [], "no": []})
        return {
            "orderbook": {
                "yes": [list(level) for level in book.get("yes", [])],
                "no": [list(level) for level in book.get("no", [])],
            }
        }

    def create_order(
        self,
        *,
        ticker: str,
        action: str,
        side: str,
        count: int,
        order_type: str,
        client_order_id: str,
        yes_price_cents: Optional[int] = None,
        no_price_cents: Optional[int] = None,
        time_in_force: Optional[str] = None,
        expiration_ts: Optional[int] = None,
    ) -> Dict[str, Any]:
        assert action == "buy"
        assert side in {"yes", "no"}
        assert order_type == "limit" and time_in_force in {"IOC", "immediate_or_cancel"}
        book = self.books.setdefault(ticker, {"yes": [], "no": []})

        if side == "yes":
            assert yes_price_cents is not None
            filled, cost = _fill_buy_yes(book, count, yes_price_cents)
            limit_price = yes_price_cents
        else:
            assert no_price_cents is not None
            filled, cost = _fill_buy_no(book, count, no_price_cents)
            limit_price = no_price_cents

        order_id = f"fake-route-{uuid.uuid4().hex[:8]}"
        resp = {"order": {
            "order_id": order_id,
            "status": "executed",
            "client_order_id": client_order_id,
            "filled_count": filled,
            "taker_fill_cost": cost,
        }}
        self.orders_log.append({
            "ticker": ticker,
            "side": side,
            "count": count,
            "limit_price_cents": limit_price,
            "filled": filled,
            "cost_cents": cost,
        })
        return resp


def _fill_buy_yes(book: Dict[str, List[List[int]]], count: int, limit_yes_cents: int) -> tuple[int, int]:
    no_bids = book.get("no", [])
    no_bids.sort(key=lambda row: row[0], reverse=True)
    remaining = count
    filled = 0
    cost = 0
    kept: List[List[int]] = []
    for n_price, size in no_bids:
        yes_price = 100 - n_price
        if remaining <= 0 or yes_price > limit_yes_cents:
            kept.append([n_price, size])
            continue
        take = min(size, remaining)
        filled += take
        cost += take * yes_price
        remaining -= take
        if size > take:
            kept.append([n_price, size - take])
    book["no"] = kept
    return filled, cost


def _fill_buy_no(book: Dict[str, List[List[int]]], count: int, limit_no_cents: int) -> tuple[int, int]:
    yes_bids = book.get("yes", [])
    yes_bids.sort(key=lambda row: row[0], reverse=True)
    remaining = count
    filled = 0
    cost = 0
    kept: List[List[int]] = []
    for y_price, size in yes_bids:
        no_price = 100 - y_price
        if remaining <= 0 or no_price > limit_no_cents:
            kept.append([y_price, size])
            continue
        take = min(size, remaining)
        filled += take
        cost += take * no_price
        remaining -= take
        if size > take:
            kept.append([y_price, size - take])
    book["yes"] = kept
    return filled, cost


def _team_name_by_id() -> Dict[str, str]:
    path = REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return {row["sportradar_team_id"]: row["kalshi_team_name"] for row in csv.DictReader(f)}


def _fake_event_ticker_for_date(dt) -> str:
    mon = dt.strftime("%b").upper()
    return f"KXWNBAGAME-{dt.year % 100:02d}{mon}{dt.day:02d}ROUTE"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv",
                    default=str(REPO_ROOT / "data" / "gold" / "game_xgboost_input_2015_2024_REGPST.csv"))
    ap.add_argument("--holdout-csv",
                    default=str(REPO_ROOT / "data" / "gold" / "game_xgboost_input_2025_REGPST.csv"))
    ap.add_argument("--polls", type=int, default=4)
    ap.add_argument("--log-path",
                    default=str(REPO_ROOT / "data" / "live_logs" / "route_dryrun_events.jsonl"))
    ap.add_argument("--ledger-dir", default=None,
                    help="Optional per-game ledger directory for verifying audit packet writes.")
    args = ap.parse_args()

    print("[route-dryrun] training FinalModel...")
    predictor = FinalModel(args.train_csv)
    df = pd.read_csv(args.holdout_csv)
    df = df[_cold_start_mask(df)].reset_index(drop=True)
    names = _team_name_by_id()

    valid_idx = None
    for i, row in df.iterrows():
        if row["home_team_id"] in names and row["away_team_id"] in names:
            valid_idx = i
            break
    if valid_idx is None:
        raise RuntimeError("no holdout row has both team IDs in kalshi_team_name_map")
    row = df.iloc[[valid_idx]].copy()
    p_home = predictor.predict_single(row)
    p_away = 1.0 - p_home
    home_id = str(row["home_team_id"].iloc[0])
    away_id = str(row["away_team_id"].iloc[0])
    home_name = names[home_id]
    away_name = names[away_id]
    selected = home_name if p_home >= p_away else away_name
    print(
        f"[route-dryrun] row game_id={row['game_id'].iloc[0]} "
        f"{away_name}@{home_name} p_home={p_home:.4f} selected={selected}"
    )

    scheduled = pd.to_datetime(row["game_ts"].iloc[0]).to_pydatetime()
    event_ticker = _fake_event_ticker_for_date(scheduled)
    home_ticker = f"{event_ticker}-HOM"
    away_ticker = f"{event_ticker}-AWY"
    markets = [
        {
            "ticker": home_ticker,
            "event_ticker": event_ticker,
            "title": f"{home_name} vs {away_name} winner?",
            "yes_sub_title": home_name,
            "status": "active",
            "market_type": "binary",
            "rules_primary": (
                f"If {home_name} wins the {home_name} vs {away_name} women's professional basketball game "
                "originally scheduled for May 10, 2026, then the market resolves to Yes."
            ),
            "custom_strike": {"basketball_team": "diagnostic-home"},
        },
        {
            "ticker": away_ticker,
            "event_ticker": event_ticker,
            "title": f"{home_name} vs {away_name} winner?",
            "yes_sub_title": away_name,
            "status": "active",
            "market_type": "binary",
            "rules_primary": (
                f"If {away_name} wins the {home_name} vs {away_name} women's professional basketball game "
                "originally scheduled for May 10, 2026, then the market resolves to Yes."
            ),
            "custom_strike": {"basketball_team": "diagnostic-away"},
        },
    ]
    fake = FakeRouteKalshi(
        markets=markets,
        books={
            home_ticker: {"yes": [[60, 2000]], "no": [[60, 2000]]},
            away_ticker: {"yes": [[60, 2000]], "no": [[60, 2000]]},
        },
    )
    log_path = Path(args.log_path)
    if log_path.exists():
        log_path.unlink()

    game = SportRadarGameRef(
        game_id=str(row["game_id"].iloc[0]),
        scheduled=scheduled,
        home_team_id=home_id,
        away_team_id=away_id,
        home_team_name=home_name,
        away_team_name=away_name,
    )
    ctx = RouteEntryContext(
        game=game,
        tipoff_ts_s=int(time.time()) + 10_000,
        feature_row=row,
        team_name_map_path=REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv",
    )
    cfg = ExecutionConfig(bankroll=5000.0)
    ledger = None
    if args.ledger_dir:
        ledger = GameLedger(
            game_id=str(row["game_id"].iloc[0]),
            root_dir=Path(args.ledger_dir),
            raw_log_path=log_path,
            metadata={
                "entrypoint": str(Path(__file__).resolve()),
                "dry_run_script": True,
                "feature_source": args.holdout_csv,
                "train_csv": args.train_csv,
                "polls": args.polls,
                "execution_config": asdict(cfg),
            },
        )

    loop = RouteEntryLoop(
        predictor=predictor,
        client=fake,
        ctx=ctx,
        cfg=cfg,
        log_path=log_path,
        poll_interval_s=-1.0,
        dry_run=False,
        markets=markets,
        ledger=ledger,
    )

    counter = {"i": 0}

    def fake_now() -> float:
        counter["i"] += 1
        if counter["i"] > args.polls:
            return ctx.tipoff_ts_s + 1
        return ctx.tipoff_ts_s - 13.0 * 3600.0

    loop.run(wall_clock_now_s=fake_now)

    print(f"[route-dryrun] selected_team_id={loop.selected_team_id} p_selected={loop.p_selected:.4f}")
    print(f"[route-dryrun] event={loop.mapping.event_ticker}")
    print(f"[route-dryrun] filled_cost=${loop.state.filled_cost_dollars:.2f}")
    print(f"[route-dryrun] filled_contracts_by_route={loop.state.filled_contracts_by_route}")
    print(f"[route-dryrun] fake orders submitted: {len(fake.orders_log)}")
    for i, order in enumerate(fake.orders_log, start=1):
        print(f"    #{i} {order}")

    assert log_path.exists(), log_path
    events = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    kinds: Dict[str, int] = {}
    for event in events:
        kinds[event["evt"]] = kinds.get(event["evt"], 0) + 1
    print(f"[route-dryrun] event log rows: {len(events)}")
    print(f"[route-dryrun] event kinds: {kinds}")

    assert kinds.get("mapping", 0) == 1
    assert kinds.get("route_candidate", 0) == 2
    assert kinds.get("route_quote", 0) >= args.polls * 2
    assert kinds.get("execution_plan", 0) >= args.polls
    assert kinds.get("order_submitted", 0) >= 1
    assert kinds.get("fill", 0) >= 1
    if ledger is not None:
        assert (ledger.root_dir / "prediction_packet.json").exists()
        assert (ledger.root_dir / "market_mapping.json").exists()
        assert (ledger.root_dir / "market_snapshots.jsonl").exists()
        assert (ledger.root_dir / "execution_plans.jsonl").exists()
        assert (ledger.root_dir / "orders.jsonl").exists()
        assert (ledger.root_dir / "fills.jsonl").exists()
        assert (ledger.root_dir / "positions.jsonl").exists()
        assert (ledger.root_dir / "summary.json").exists()
        print(f"[route-dryrun] ledger: {ledger.root_dir}")
    assert loop.state.filled_cost_dollars > 0
    print("[route-dryrun] OK")


if __name__ == "__main__":
    main()
