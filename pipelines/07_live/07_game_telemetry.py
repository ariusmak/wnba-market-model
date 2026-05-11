"""
07_game_telemetry.py
====================

Canonical post-game or mid-game telemetry for one live game ledger.

Reads data/runs/live_games/<game_id>/events.jsonl by default, summarizes
prediction/mapping/plans/orders/fills, and can reconcile the canonical
routes against Kalshi positions/fills.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from srwnba.live.canonical.kalshi_mapping import RouteCandidate  # noqa: E402
from srwnba.live.canonical.reconciliation import reconcile_exchange_routes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", default=None,
                    help="Game id under data/runs/live_games/<game-id>.")
    ap.add_argument("--ledger-dir", default=None,
                    help="Explicit per-game ledger directory.")
    ap.add_argument("--log", default=None,
                    help="Explicit JSONL event path; retained for legacy logs.")
    ap.add_argument("--reconcile", action="store_true",
                    help="Cross-check live Kalshi positions/fills.")
    ap.add_argument("--settled", choices=["selected", "opponent", "home", "away"], default=None,
                    help="Settlement winner expressed relative to the model-selected team.")
    ap.add_argument("--json-out", default=None,
                    help="Optional path for the full machine-readable summary.")
    args = ap.parse_args()

    event_path = resolve_event_path(args)
    events = parse_events(event_path)
    summary = summarize_canonical(events)
    if args.reconcile:
        summary["kalshi_reconciliation"] = reconcile_with_kalshi(summary)
    if args.settled:
        summary["settlement_pnl"] = compute_settlement_pnl(summary, args.settled)
    if args.json_out:
        write_json(Path(args.json_out), summary)
    print_summary(summary)


def resolve_event_path(args: argparse.Namespace) -> Path:
    if args.log:
        return Path(args.log)
    if args.ledger_dir:
        return Path(args.ledger_dir) / "events.jsonl"
    if args.game_id:
        return REPO_ROOT / "data" / "runs" / "live_games" / args.game_id / "events.jsonl"
    raise SystemExit("Provide --game-id, --ledger-dir, or --log.")


def parse_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def summarize_canonical(events: list[dict[str, Any]]) -> dict[str, Any]:
    start = first_event(events, "route_loop_start") or first_event(events, "start") or {}
    mapping = first_event(events, "mapping") or {}
    route_candidates = [e for e in events if e.get("evt") == "route_candidate"]
    plans = [e for e in events if e.get("evt") == "execution_plan"]
    orders = [e for e in events if e.get("evt") in {"order_submitted", "dry_order"}]
    fills = [e for e in events if e.get("evt") in {"fill", "passive_fill_reconciled"}]
    quotes = [e for e in events if e.get("evt") == "route_quote"]
    errors = [e for e in events if e.get("evt") in {"order_error", "poll_error", "position_reconciliation_error"}]
    reconciliations = [e for e in events if e.get("evt") == "position_reconciliation"]

    route_summary: dict[str, dict[str, Any]] = {}
    for route in route_candidates:
        route_summary[route["route_id"]] = {
            "route_id": route.get("route_id"),
            "route_type": route.get("route_type"),
            "market_ticker": route.get("market_ticker"),
            "side": route.get("side"),
            "orders": 0,
            "filled_contracts": 0,
            "filled_cost_dollars": 0.0,
            "latest_quote": None,
        }
    for quote in quotes:
        rec = route_summary.setdefault(str(quote.get("route_id")), {"route_id": quote.get("route_id")})
        rec["latest_quote"] = quote
    for order in orders:
        route_id = str(order.get("route_id") or "")
        rec = route_summary.setdefault(route_id, {"route_id": route_id})
        rec["orders"] = int(rec.get("orders") or 0) + 1
    for fill in fills:
        route_id = str(fill.get("route_id") or "")
        rec = route_summary.setdefault(route_id, {"route_id": route_id})
        filled = int(fill.get("filled") or fill.get("filled_delta") or 0)
        cost = float(fill.get("cost_cents") or 0) / 100.0
        cost += float(fill.get("cost_delta_dollars") or 0.0)
        rec["filled_contracts"] = int(rec.get("filled_contracts") or 0) + filled
        rec["filled_cost_dollars"] = float(rec.get("filled_cost_dollars") or 0.0) + cost

    latest_plan = plans[-1] if plans else {}
    decision_counts = Counter(str(plan.get("decision") or "-") for plan in plans)
    reject_counts = Counter(str(plan.get("reject_reason") or "-") for plan in plans if plan.get("reject_reason"))
    total_filled_contracts = sum(int(rec.get("filled_contracts") or 0) for rec in route_summary.values())
    total_filled_cost = sum(float(rec.get("filled_cost_dollars") or 0.0) for rec in route_summary.values())
    fees = sum(float(fill.get("fee_dollars") or 0.0) for fill in fills)

    return {
        "game_id": start.get("game_id") or latest_plan.get("game_id"),
        "event_path_game_id": start.get("game_id"),
        "event_ticker": mapping.get("event_ticker") or start.get("event_ticker"),
        "p_home": start.get("p_home"),
        "p_away": start.get("p_away"),
        "p_selected": start.get("p_selected") or latest_plan.get("p_selected"),
        "selected_team_id": start.get("selected_team_id") or latest_plan.get("selected_team_id"),
        "selected_side_label": start.get("selected_side_label"),
        "tipoff_ts_s": start.get("tipoff_ts_s"),
        "dry_run": start.get("dry_run"),
        "mapping_confirmed": mapping.get("confirmed"),
        "operator": {
            "trade_allowed": latest_plan.get("operator_trade_allowed", start.get("operator_trade_allowed")),
            "reason": latest_plan.get("operator_reason", start.get("operator_reason")),
            "risk_mode": latest_plan.get("operator_risk_mode", start.get("operator_risk_mode")),
        },
        "latest_plan": latest_plan,
        "plan_count": len(plans),
        "decision_counts": dict(decision_counts),
        "reject_counts": dict(reject_counts),
        "order_count": len(orders),
        "fill_event_count": len(fills),
        "error_count": len(errors),
        "position_mismatch_dollars": latest_plan.get("position_mismatch_dollars"),
        "latest_reconciliation": reconciliations[-1] if reconciliations else None,
        "total_filled_contracts": total_filled_contracts,
        "total_filled_cost_dollars": total_filled_cost,
        "total_fee_dollars": fees,
        "avg_fill_price_cents": (100.0 * total_filled_cost / total_filled_contracts) if total_filled_contracts else None,
        "routes": list(route_summary.values()),
        "raw_route_candidates": route_candidates,
    }


def reconcile_with_kalshi(summary: Mapping[str, Any]) -> dict[str, Any]:
    from utils.kalshi_authed_client import AuthedKalshiClient  # type: ignore

    routes = []
    for raw in summary.get("raw_route_candidates") or []:
        routes.append(
            RouteCandidate(
                route_id=str(raw.get("route_id") or ""),
                canonical_exposure=str(raw.get("canonical_exposure") or "selected_team_wins"),
                selected_team_id=str(raw.get("selected_team_id") or ""),
                opponent_team_id=str(raw.get("opponent_team_id") or ""),
                selected_team_name=str(raw.get("selected_team_name") or ""),
                opponent_team_name=str(raw.get("opponent_team_name") or ""),
                market_ticker=str(raw.get("market_ticker") or ""),
                event_ticker=str(raw.get("event_ticker") or ""),
                route_type=str(raw.get("route_type") or ""),
                action=str(raw.get("action") or "buy"),
                side=str(raw.get("side") or ""),
                market_yes_team_id=str(raw.get("market_yes_team_id") or ""),
                market_yes_team_name=str(raw.get("market_yes_team_name") or ""),
                side_mapping_confirmed=bool(raw.get("side_mapping_confirmed")),
                complement_market_confirmed=bool(raw.get("complement_market_confirmed")),
                settlement_mapping_confirmed=bool(raw.get("settlement_mapping_confirmed")),
            )
        )
    client = AuthedKalshiClient()
    rec = reconcile_exchange_routes(client, routes)
    local_cost_by_route = {
        str(route.get("route_id")): float(route.get("filled_cost_dollars") or 0.0)
        for route in summary.get("routes") or []
    }
    return {
        **rec.to_log_payload(),
        "mismatch_dollars_vs_ledger": rec.mismatch_dollars(local_cost_by_route),
    }


def compute_settlement_pnl(summary: Mapping[str, Any], settled: str) -> dict[str, Any]:
    contracts = int(summary.get("total_filled_contracts") or 0)
    cost = float(summary.get("total_filled_cost_dollars") or 0.0)
    fees = float(summary.get("total_fee_dollars") or 0.0)
    selected_side = str(summary.get("selected_side_label") or "").lower()
    if settled in {"selected", "opponent"}:
        selected_won = settled == "selected"
    elif settled in {"home", "away"} and selected_side in {"home", "away"}:
        selected_won = settled == selected_side
    else:
        selected_won = False
    payout = float(contracts) if selected_won else 0.0
    return {
        "settled": settled,
        "selected_won": selected_won,
        "contracts": contracts,
        "cost_dollars": cost,
        "fees_dollars": fees,
        "payout_dollars": payout,
        "net_pnl_dollars": payout - cost - fees,
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print(f"== game {summary.get('game_id')} ==")
    print(
        f"  event={summary.get('event_ticker')} selected={summary.get('selected_team_id')} "
        f"p_selected={fmt_prob(summary.get('p_selected'))} dry_run={summary.get('dry_run')}"
    )
    op = summary.get("operator") or {}
    print(f"  operator: allowed={op.get('trade_allowed')} risk={op.get('risk_mode')} reason={op.get('reason')}")
    print(
        f"  plans={summary.get('plan_count')} orders={summary.get('order_count')} "
        f"fills={summary.get('fill_event_count')} errors={summary.get('error_count')}"
    )
    print(f"  decisions={summary.get('decision_counts')} rejects={summary.get('reject_counts')}")
    print(
        f"  filled={summary.get('total_filled_contracts')} contracts "
        f"cost=${float(summary.get('total_filled_cost_dollars') or 0.0):.2f} "
        f"avg={fmt_cents(summary.get('avg_fill_price_cents'))} "
        f"position_mismatch=${float(summary.get('position_mismatch_dollars') or 0.0):.2f}"
    )
    for route in summary.get("routes") or []:
        print(
            f"  [{route.get('route_id')}] {route.get('route_type')} {route.get('market_ticker')} "
            f"orders={route.get('orders')} filled={route.get('filled_contracts')} "
            f"cost=${float(route.get('filled_cost_dollars') or 0.0):.2f}"
        )
    if summary.get("kalshi_reconciliation"):
        rec = summary["kalshi_reconciliation"]
        print(f"  kalshi mismatch=${float(rec.get('mismatch_dollars_vs_ledger') or 0.0):.2f}")
    if summary.get("settlement_pnl"):
        pnl = summary["settlement_pnl"]
        print(f"  settled={pnl['settled']} selected_won={pnl['selected_won']} net=${pnl['net_pnl_dollars']:.2f}")


def first_event(events: Iterable[Mapping[str, Any]], evt: str) -> Optional[dict[str, Any]]:
    return next((dict(event) for event in events if event.get("evt") == evt), None)


def fmt_prob(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def fmt_cents(value: Any) -> str:
    return "-" if value is None else f"{float(value):.2f}c"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
