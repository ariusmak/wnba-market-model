"""
05_run_route_entry_loop.py
==========================

Production-shaped canonical-route entry loop for one WNBA game.

This entrypoint maps a Sportradar game to Kalshi winner markets, chooses
the model's selected team, plans selected-team-wins exposure across
BUY YES selected / BUY NO opponent routes, and writes an audit JSONL log.

Use `--dry-run` for read-only planning. Live order submission additionally
requires `KALSHI_TRADING_ENABLED=true` in `.env`.
"""
from __future__ import annotations

import argparse
import atexit
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.canonical.execution import ExecutionConfig  # noqa: E402
from srwnba.live.canonical.game_ledger import GameLedger  # noqa: E402
from srwnba.live.canonical.kalshi_mapping import SportRadarGameRef, parse_datetime  # noqa: E402
from srwnba.live.canonical.portfolio import resolve_portfolio_sizing  # noqa: E402
from srwnba.live.canonical.process_lock import GameProcessLock  # noqa: E402
from srwnba.live.canonical.route_entry_loop import RouteEntryContext, RouteEntryLoop  # noqa: E402
from srwnba.live.control_plane import CONTROL_PLANE_MODES  # noqa: E402
from srwnba.util.final_model import FinalModel  # noqa: E402
from utils.kalshi_authed_client import AuthedKalshiClient  # noqa: E402


PRODUCTION_TRAIN_CSV = REPO_ROOT / "data" / "gold" / "game_xgboost_input_2015_2026_REGPST.csv"
ROUTE_LOCK_DIR = REPO_ROOT / "data" / "runs" / "live_execution" / "game_locks"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--scheduled", required=True,
                    help="ISO scheduled timestamp for matching/audit")
    ap.add_argument("--home-team-id", required=True)
    ap.add_argument("--away-team-id", required=True)
    ap.add_argument("--home-team-name", default="")
    ap.add_argument("--away-team-name", default="")
    ap.add_argument("--tipoff-ts", type=int, required=True)
    ap.add_argument("--feature-csv", required=True,
                    help="Single-row CSV in gold-table schema for this game")
    ap.add_argument("--train-csv", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--sizing-bankroll-override",
        "--bankroll",
        dest="sizing_bankroll_override",
        type=float,
        default=None,
        help=(
            "Optional imaginary bankroll used for sizing/caps instead of Kalshi "
            "portfolio_value. --bankroll is retained as a deprecated alias."
        ),
    )
    ap.add_argument(
        "--available-cash-override",
        "--available-cash",
        dest="available_cash_override",
        type=float,
        default=None,
        help=(
            "Optional imaginary available-cash cap. By default order feasibility "
            "uses Kalshi balance."
        ),
    )
    ap.add_argument(
        "--portfolio-refresh-interval-s",
        type=float,
        default=300.0,
        help="How often the loop refreshes Kalshi balance/portfolio_value while running.",
    )
    ap.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.0,
        help=(
            "0 uses locked v1.2 cadence by timing window; positive values cap "
            "that cadence for diagnostics; negative disables sleeping for tests."
        ),
    )
    ap.add_argument("--series-ticker", action="append")
    ap.add_argument("--market-discovery-limit", type=int, default=100)
    ap.add_argument("--team-name-map",
                    default=str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"))
    ap.add_argument("--completed-games-csv", default=None,
                    help=("Optional played-games CSV used to count completed prior games "
                          "for the expansion-team trading gate. Defaults to "
                          "data/silver/played_games_<scheduled_year>_REGPST.csv when present."))
    ap.add_argument("--log-path", default=None)
    ap.add_argument("--ledger-dir", default=None,
                    help=("Per-game audit ledger directory; defaults to "
                          "data/runs/live_games/<game-id>"))
    ap.add_argument("--operator-control-path", default=None,
                    help=("Global operator-control JSON path. Defaults to "
                          "data/runs/live_control/operator_control.json."))
    ap.add_argument("--operator-override-path", default=None,
                    help=("Per-game override JSON path. Defaults to "
                          "data/runs/live_control/game_overrides/<game-id>.json."))
    ap.add_argument("--control-plane-mode", choices=CONTROL_PLANE_MODES, default="local-only",
                    help="Remote control-plane enforcement mode.")
    ap.add_argument("--control-plane-bot-id", default="wnba-route-worker",
                    help="bot_heartbeat.bot_id used when publishing worker status.")
    ap.add_argument("--position-reconcile-interval-s", type=float, default=300.0,
                    help="How often to reconcile route fills/positions against Kalshi.")
    ap.add_argument("--skip-startup-reconciliation", action="store_true",
                    help="Skip startup open-order and position/fill reconciliation.")
    ap.add_argument("--no-ledger", action="store_true",
                    help="Disable the structured per-game audit ledger.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan and log but do not submit orders")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log = logging.getLogger("srwnba.live.route_cli")
    process_lock = GameProcessLock(
        game_id=args.game_id,
        lock_dir=ROUTE_LOCK_DIR,
        metadata={
            "entrypoint": str(Path(__file__).resolve()),
            "dry_run": args.dry_run,
            "feature_csv": args.feature_csv,
            "scheduled": args.scheduled,
        },
    )
    process_lock.acquire()
    atexit.register(process_lock.release)

    if args.train_csv:
        log.warning(
            "ignoring deprecated --train-csv=%s; production route loop is hardcoded to %s",
            args.train_csv,
            PRODUCTION_TRAIN_CSV,
        )
    if not PRODUCTION_TRAIN_CSV.exists():
        raise FileNotFoundError(
            f"Production training CSV not found: {PRODUCTION_TRAIN_CSV}. "
            "Run pipelines/07_live/14_live_data_refresh.py or "
            "pipelines/07_live/09_combine_gold.py --start-year 2015 --end-year 2026 --force first."
        )

    log.info("training FinalModel from locked production CSV %s", PRODUCTION_TRAIN_CSV)
    predictor = FinalModel(PRODUCTION_TRAIN_CSV)

    feature_csv = Path(args.feature_csv)
    feat_df = pd.read_csv(feature_csv)
    if len(feat_df) != 1:
        raise ValueError(f"feature-csv must be exactly one row, got {len(feat_df)}")

    client = AuthedKalshiClient()
    log.info(
        "authed client ready (base_url=%s trading_enabled=%s)",
        client.cfg.base_url,
        client.cfg.trading_enabled,
    )
    if not args.dry_run and not client.cfg.trading_enabled:
        raise RuntimeError(
            "Live order submission is blocked because KALSHI_TRADING_ENABLED "
            "is not exactly true. Pass --dry-run for read-only route planning, "
            "or set KALSHI_TRADING_ENABLED=true only when intentionally enabling live orders."
        )
    portfolio_sizing = resolve_portfolio_sizing(
        client,
        sizing_bankroll_override_dollars=args.sizing_bankroll_override,
        available_cash_override_dollars=args.available_cash_override,
    )
    log.info(
        "portfolio sizing bankroll=$%.2f source=%s kalshi_portfolio=%s kalshi_cash=%s available_cash=%s",
        portfolio_sizing.sizing_bankroll_dollars,
        portfolio_sizing.sizing_bankroll_source,
        _money_or_unknown(portfolio_sizing.kalshi_portfolio_value_dollars),
        _money_or_unknown(portfolio_sizing.kalshi_cash_dollars),
        _money_or_unknown(portfolio_sizing.available_cash_dollars),
    )

    game = SportRadarGameRef(
        game_id=args.game_id,
        scheduled=parse_datetime(args.scheduled),
        home_team_id=args.home_team_id,
        away_team_id=args.away_team_id,
        home_team_name=args.home_team_name,
        away_team_name=args.away_team_name,
    )
    completed_games_by_team = _load_completed_games_by_team(
        csv_path=args.completed_games_csv,
        scheduled=game.scheduled,
    )
    ctx = RouteEntryContext(
        game=game,
        tipoff_ts_s=args.tipoff_ts,
        feature_row=feat_df,
        team_name_map_path=Path(args.team_name_map),
        series_tickers=tuple(args.series_ticker or ["KXWNBAGAME", "KXWNBAH"]),
        market_discovery_limit=args.market_discovery_limit,
        completed_games_by_team=completed_games_by_team,
    )
    cfg = ExecutionConfig(bankroll=portfolio_sizing.sizing_bankroll_dollars)
    log_path = (
        Path(args.log_path)
        if args.log_path
        else REPO_ROOT / "data" / "live_logs" / f"{args.game_id}.route.jsonl"
    )
    ledger = None
    if not args.no_ledger:
        ledger_dir = (
            Path(args.ledger_dir)
            if args.ledger_dir
            else REPO_ROOT / "data" / "runs" / "live_games" / args.game_id
        )
        ledger = GameLedger(
            game_id=args.game_id,
            root_dir=ledger_dir,
            raw_log_path=log_path,
            metadata={
                "entrypoint": str(Path(__file__).resolve()),
                "scheduled": args.scheduled,
                "home_team_id": args.home_team_id,
                "away_team_id": args.away_team_id,
                "home_team_name": args.home_team_name,
                "away_team_name": args.away_team_name,
                "tipoff_ts_s": args.tipoff_ts,
                "feature_csv": str(feature_csv),
                "train_csv": str(PRODUCTION_TRAIN_CSV),
                "completed_games_csv": args.completed_games_csv,
                "team_name_map": args.team_name_map,
                "series_tickers": args.series_ticker or ["KXWNBAGAME", "KXWNBAH"],
                "market_discovery_limit": args.market_discovery_limit,
                "poll_interval_s": args.poll_interval_s,
                "dry_run": args.dry_run,
                "kalshi_portfolio_sizing": portfolio_sizing.to_log_payload(),
                "available_cash_dollars": portfolio_sizing.available_cash_dollars,
                "sizing_bankroll_override_dollars": args.sizing_bankroll_override,
                "available_cash_override_dollars": args.available_cash_override,
                "portfolio_refresh_interval_s": args.portfolio_refresh_interval_s,
                "operator_control_path": args.operator_control_path,
                "operator_override_path": args.operator_override_path,
                "control_plane_mode": args.control_plane_mode,
                "control_plane_bot_id": args.control_plane_bot_id,
                "position_reconcile_interval_s": args.position_reconcile_interval_s,
                "startup_reconciliation": not args.skip_startup_reconciliation,
                "execution_config": asdict(cfg),
            },
        )

    tip_remaining_s = max(0, args.tipoff_ts - int(time.time()))
    log.info("game=%s tipoff=%d in=%ds dry_run=%s",
             args.game_id, args.tipoff_ts, tip_remaining_s, args.dry_run)
    loop = RouteEntryLoop(
        predictor=predictor,
        client=client,
        ctx=ctx,
        cfg=cfg,
        log_path=log_path,
        poll_interval_s=args.poll_interval_s,
        dry_run=args.dry_run,
        available_cash_dollars=portfolio_sizing.available_cash_dollars,
        follow_kalshi_wealth=True,
        initial_portfolio_sizing=portfolio_sizing,
        sizing_bankroll_override_dollars=args.sizing_bankroll_override,
        available_cash_override_dollars=args.available_cash_override,
        portfolio_refresh_interval_s=args.portfolio_refresh_interval_s,
        operator_global_control_path=Path(args.operator_control_path) if args.operator_control_path else None,
        operator_game_override_path=Path(args.operator_override_path) if args.operator_override_path else None,
        control_plane_mode=args.control_plane_mode,
        control_plane_bot_id=args.control_plane_bot_id,
        position_reconcile_interval_s=args.position_reconcile_interval_s,
        startup_reconciliation=not args.skip_startup_reconciliation,
        ledger=ledger,
    )
    log.info(
        "selected=%s p_selected=%.4f event=%s expansion_gate=%s routes=%s",
        loop.selected_team_id,
        loop.p_selected,
        loop.mapping.event_ticker,
        loop.expansion_gate.reason,
        [r.route_type + ":" + r.market_ticker for r in loop.routes],
    )

    try:
        loop.run()
    except KeyboardInterrupt:
        log.warning("interrupted - not submitting further orders")

    log.info(
        "DONE selected=%s filled_cost=$%.2f routes=%s log=%s",
        loop.selected_team_id,
        loop.state.filled_cost_dollars,
        loop.state.filled_contracts_by_route,
        log_path,
    )
    if ledger is not None:
        log.info("ledger=%s summary=%s", ledger.root_dir, ledger.root_dir / "summary.json")
    process_lock.release()


def _load_completed_games_by_team(
    *,
    csv_path: Optional[str],
    scheduled,
) -> Dict[str, int]:
    path = Path(csv_path) if csv_path else _default_played_games_path(scheduled)
    if path is None or not path.exists():
        return {}

    df = pd.read_csv(path)
    if "scheduled" not in df.columns or "home_id" not in df.columns or "away_id" not in df.columns:
        raise ValueError(
            f"completed-games CSV must include scheduled, home_id, and away_id columns: {path}"
        )

    game_ts = pd.Timestamp(scheduled)
    if game_ts.tzinfo is None:
        game_ts = game_ts.tz_localize("UTC")
    else:
        game_ts = game_ts.tz_convert("UTC")

    sched = pd.to_datetime(df["scheduled"], utc=True, errors="coerce")
    prior = df[sched < game_ts].copy()
    if "status" in prior.columns:
        prior = prior[prior["status"].astype(str).str.lower().isin({"closed", "complete", "completed"})]

    counts: Dict[str, int] = {}
    for _, row in prior.iterrows():
        for col in ("home_id", "away_id"):
            team_id = str(row.get(col) or "").strip()
            if team_id:
                counts[team_id] = counts.get(team_id, 0) + 1
    return counts


def _default_played_games_path(scheduled) -> Optional[Path]:
    try:
        year = int(pd.Timestamp(scheduled).year)
    except Exception:
        return None
    return REPO_ROOT / "data" / "silver" / f"played_games_{year}_REGPST.csv"


def _money_or_unknown(value: Optional[float]) -> str:
    return "unknown" if value is None else f"${value:,.2f}"


if __name__ == "__main__":
    main()
