"""
17_build_live_prediction_packet.py
==================================

Build the official pre-tipoff live feature row and prediction packet for one
scheduled WNBA game. This is intentionally separate from the historical gold
builder, which remains closed-game/label oriented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from srwnba.live.canonical.kalshi_mapping import (  # noqa: E402
    load_team_name_map,
    map_game_to_kalshi_markets,
    parse_datetime,
)
from srwnba.live.canonical.live_features import (  # noqa: E402
    build_live_feature_row,
    write_live_feature_artifacts,
)
from srwnba.util.final_model import FinalModel  # noqa: E402


PRODUCTION_TRAIN_CSV = REPO_ROOT / "data" / "gold" / "game_xgboost_input_2015_2026_REGPST.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--asof", default=None, help="As-of timestamp, ISO. Defaults to now UTC.")
    ap.add_argument("--train-csv", default=str(PRODUCTION_TRAIN_CSV))
    ap.add_argument("--feature-csv", default=None)
    ap.add_argument("--packet-json", default=None)
    ap.add_argument("--market-snapshot-json", default=None)
    ap.add_argument(
        "--team-name-map",
        default=str(REPO_ROOT / "data" / "config" / "kalshi_team_name_map.csv"),
    )
    args = ap.parse_args()

    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(train_csv)

    asof_ts = parse_datetime(args.asof) if args.asof else parse_datetime(_now_utc_iso())
    result = build_live_feature_row(
        game_id=args.game_id,
        year=args.year,
        asof_ts=asof_ts,
        repo_root=REPO_ROOT,
    )
    model = FinalModel(train_csv)
    prediction = model.predict(result.feature_row)

    mapping = _mapping_payload(
        result.game,
        snapshot_json=Path(args.market_snapshot_json) if args.market_snapshot_json else None,
        team_name_map_path=Path(args.team_name_map),
    )
    feature_csv = Path(args.feature_csv) if args.feature_csv else (
        REPO_ROOT / "data" / "live_features" / f"{args.game_id}.csv"
    )
    packet_json = Path(args.packet_json) if args.packet_json else (
        REPO_ROOT / "data" / "runs" / "live_games" / args.game_id / "prediction_packet.json"
    )
    packet = write_live_feature_artifacts(
        result=result,
        feature_csv=feature_csv,
        packet_json=packet_json,
        prediction=prediction,
        train_csv=train_csv,
        model_best_round=getattr(model, "best_round", None),
        model_best_round_source=getattr(model, "best_round_source", None),
        mapping=mapping,
    )
    print(json.dumps({
        "game_id": args.game_id,
        "feature_csv": str(feature_csv),
        "packet_json": str(packet_json),
        "p_home": packet["p_home"],
        "p_elo": packet["p_elo"],
        "warnings": packet["source_report"].get("warnings", []),
        "mapping_confirmed": mapping.get("confirmed") if mapping else None,
    }, indent=2, sort_keys=True))


def _mapping_payload(
    game,
    *,
    snapshot_json: Optional[Path],
    team_name_map_path: Path,
) -> dict[str, Any]:
    if snapshot_json is None or not snapshot_json.exists():
        return {"checked": False, "reason": "no_market_snapshot"}
    data = json.loads(snapshot_json.read_text(encoding="utf-8"))
    markets = data.get("markets") if isinstance(data, Mapping) else []
    if not isinstance(markets, list):
        markets = []
    team_name_map = load_team_name_map(str(team_name_map_path)) if team_name_map_path.exists() else {}
    try:
        mapping = map_game_to_kalshi_markets(
            game,
            markets,
            require_open=True,
            team_name_to_id=team_name_map,
        )
    except Exception as exc:
        return {"checked": True, "confirmed": False, "error": repr(exc)}
    return {
        "checked": True,
        "confirmed": mapping.confirmed,
        "event_ticker": mapping.event_ticker,
        "candidate_count": mapping.candidate_count,
        "diagnostics": list(mapping.diagnostics),
        "home_market": getattr(mapping.home_market, "ticker", ""),
        "away_market": getattr(mapping.away_market, "ticker", ""),
        "side_mapping_confirmed": mapping.side_mapping_confirmed,
        "complement_market_confirmed": mapping.complement_market_confirmed,
        "settlement_mapping_confirmed": mapping.settlement_mapping_confirmed,
    }


def _now_utc_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
