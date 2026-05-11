"""
05_run_entry_loop.py
====================

Production entrypoint for the WNBA live entry loop. Runs one game:
trains the FinalModel, polls home+away orderbooks continuously from now
until tipoff, sweeps taker IOC orders when edge thresholds are met, and
caps exposure per side at the half-Kelly target.

Usage (from repo root):
    python pipelines/07_live/legacy/05_run_entry_loop.py \
        --game-id WNBA-20260515-ATL-IND \
        --home-ticker KXWNBAH-26MAY15ATLIND-IND \
        --away-ticker KXWNBAH-26MAY15ATLIND-ATL \
        --tipoff-ts 1747346400 \
        --feature-csv data/live_features/WNBA-20260515-ATL-IND.csv \
        --train-csv  data/gold/game_xgboost_input_2015_2025_REGPST.csv

Requires `KALSHI_ACCESS_KEY` and `KALSHI_PRIVATE_KEY_PATH` in env/.env.
Pass `--dry-run` to log plans without submitting orders.
Live order submission also requires `KALSHI_TRADING_ENABLED=true`.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.legacy.entry_loop import EntryLoop, GameContext  # noqa: E402
from srwnba.live.legacy.trader import TradeConfig  # noqa: E402
from srwnba.util.final_model import FinalModel  # noqa: E402
from utils.kalshi_authed_client import AuthedKalshiClient  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True,
                    help="Opaque game ID used in logs and client_order_id")
    ap.add_argument("--home-ticker", required=True,
                    help="Kalshi YES ticker for the home team winning")
    ap.add_argument("--away-ticker", required=True,
                    help="Kalshi YES ticker for the away team winning")
    ap.add_argument("--tipoff-ts", type=int, required=True,
                    help="Tipoff unix seconds — loop exits at this time")
    ap.add_argument("--feature-csv", required=True,
                    help="Single-row CSV in gold-table schema for this game")
    ap.add_argument("--train-csv", required=True,
                    help="Training CSV for FinalModel (e.g. 2015-2025 gold)")
    ap.add_argument("--bankroll", type=float, default=5000.0)
    ap.add_argument("--edge-min", type=float, default=0.05)
    ap.add_argument("--norm-edge-min", type=float, default=0.25)
    ap.add_argument("--kelly-fraction", type=float, default=0.5)
    ap.add_argument("--poll-interval-s", type=float, default=5.0)
    ap.add_argument("--log-path",
                    default=None,
                    help="Event log JSONL; defaults to data/live_logs/<game-id>.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute plans and log but never submit orders")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("srwnba.live.cli")

    # ── Train model (FinalModel retrains on demand from the CSV)
    log.info("training FinalModel from %s", args.train_csv)
    predictor = FinalModel(args.train_csv)

    # ── Load feature row
    feat_path = Path(args.feature_csv)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)
    feat_df = pd.read_csv(feat_path)
    if len(feat_df) != 1:
        raise ValueError(f"feature-csv must be exactly one row, got {len(feat_df)}")

    # ── Build client
    client = AuthedKalshiClient()
    log.info("authed client ready (base_url=%s)", client.cfg.base_url)
    if not args.dry_run and not client.cfg.trading_enabled:
        raise RuntimeError(
            "Live order submission is blocked because KALSHI_TRADING_ENABLED "
            "is not exactly true. Pass --dry-run for read-only planning, or "
            "set KALSHI_TRADING_ENABLED=true only when intentionally enabling live orders."
        )

    # ── Config + context
    cfg = TradeConfig(
        edge_min=args.edge_min,
        norm_edge_min=args.norm_edge_min,
        kelly_fraction=args.kelly_fraction,
        bankroll=args.bankroll,
    )
    cfg.validate()

    log_path = (
        Path(args.log_path)
        if args.log_path
        else REPO_ROOT / "data" / "live_logs" / f"{args.game_id}.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = GameContext(
        game_id=args.game_id,
        home_ticker=args.home_ticker,
        away_ticker=args.away_ticker,
        tipoff_ts_s=args.tipoff_ts,
        feature_row=feat_df,
    )

    tip_remaining_s = max(0, args.tipoff_ts - int(time.time()))
    log.info("game=%s home=%s away=%s tipoff=%d (in %ds) dry_run=%s",
             args.game_id, args.home_ticker, args.away_ticker,
             args.tipoff_ts, tip_remaining_s, args.dry_run)
    if tip_remaining_s == 0:
        log.warning("tipoff already passed — loop will exit immediately")

    loop = EntryLoop(
        predictor=predictor,
        client=client,
        ctx=ctx,
        cfg=cfg,
        log_path=log_path,
        poll_interval_s=args.poll_interval_s,
        dry_run=args.dry_run,
    )
    log.info("p_home=%.4f p_away=%.4f", loop.p_home, loop.p_away)

    try:
        loop.run()
    except KeyboardInterrupt:
        log.warning("interrupted — not submitting further orders")

    log.info("DONE home filled=%d cost_cents=%d fees=$%.2f",
             loop.home.filled_contracts, loop.home.total_cost_cents,
             loop.home.total_fee_dollars)
    log.info("DONE away filled=%d cost_cents=%d fees=$%.2f",
             loop.away.filled_contracts, loop.away.total_cost_cents,
             loop.away.total_fee_dollars)
    log.info("event log: %s", log_path)


if __name__ == "__main__":
    main()
