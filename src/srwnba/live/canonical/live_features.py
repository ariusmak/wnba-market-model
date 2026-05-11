"""Live pre-tipoff model input and prediction packet builder.

The historical gold builder is intentionally closed-game only because it joins
labels and same-game official box-score rosters. This module builds the
separate production live artifact: one strict gold-schema row for a scheduled
game using only state available at an explicit as-of timestamp.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ...util.elo import EloParams, apply_carryover, elo_prob
from ...util.franchise import load_franchise_map, map_team_to_franchise
from ...util.model_schema import (
    FORM_FEATS,
    GOLD_MODEL_INPUT_COLS,
    PLAYER_FEATS,
    SCHED_FEATS,
    STYLE_FEATS,
)
from .kalshi_mapping import SportRadarGameRef, parse_datetime


LIVE_PACKET_SCHEMA_VERSION = "live_prediction_packet_v1"
RECENT_FORM_LAMBDA = 1 - 2 ** (-1 / 7)
P_CLIP = 1e-6
CENTRAL = ZoneInfo("America/Chicago")


@dataclass(frozen=True)
class LiveFeatureSourceReport:
    asof_ts_utc: str
    schedule_files: tuple[str, ...] = ()
    player_state_path: str = ""
    elo_path: str = ""
    recent_form_path: str = ""
    style_path: str = ""
    prev_style_path: str = ""
    warnings: tuple[str, ...] = ()
    player_counts: Mapping[str, int] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "asof_ts_utc": self.asof_ts_utc,
            "schedule_files": list(self.schedule_files),
            "player_state_path": self.player_state_path,
            "elo_path": self.elo_path,
            "recent_form_path": self.recent_form_path,
            "style_path": self.style_path,
            "prev_style_path": self.prev_style_path,
            "warnings": list(self.warnings),
            "player_counts": dict(self.player_counts),
        }


@dataclass(frozen=True)
class LiveFeatureResult:
    game: SportRadarGameRef
    feature_row: pd.DataFrame
    source_report: LiveFeatureSourceReport


def build_live_feature_row(
    *,
    game_id: str,
    year: int,
    asof_ts: datetime,
    repo_root: Path,
) -> LiveFeatureResult:
    """Build one strict gold-schema row for a scheduled game."""
    repo_root = Path(repo_root)
    asof_ts = _utc(asof_ts)
    schedule_games, schedule_files = _load_schedule_games(repo_root, year)
    raw_game = schedule_games.get(str(game_id))
    if raw_game is None:
        raise KeyError(f"game_id {game_id} not found in latest {year} schedule")

    game = _game_ref_from_schedule(raw_game)
    franchise_map = load_franchise_map(str(repo_root / "data" / "config" / "franchise_map.csv"))
    home_fid = map_team_to_franchise(game.home_team_id, year, franchise_map)
    away_fid = map_team_to_franchise(game.away_team_id, year, franchise_map)
    warnings: list[str] = []

    elo_features, elo_path = _live_elo_features(
        repo_root=repo_root,
        year=year,
        asof_ts=asof_ts,
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
        home_franchise_id=home_fid,
        away_franchise_id=away_fid,
    )
    player_features, player_counts, player_state_path = _live_player_features(
        repo_root=repo_root,
        year=year,
        asof_ts=asof_ts,
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
    )
    for side, count in player_counts.items():
        if count < 7:
            warnings.append(f"{side}_player_pool_below_7:{count}")

    form_features, recent_form_path = _live_recent_form_features(
        repo_root=repo_root,
        year=year,
        asof_ts=asof_ts,
        home_franchise_id=home_fid,
        away_franchise_id=away_fid,
    )
    style_features, style_path, prev_style_path = _live_style_features(
        repo_root=repo_root,
        year=year,
        asof_ts=asof_ts,
        home_franchise_id=home_fid,
        away_franchise_id=away_fid,
        franchise_map=franchise_map,
    )
    schedule_features = _live_schedule_features(
        raw_game=raw_game,
        schedule_games=schedule_games,
        year=year,
        home_team_id=game.home_team_id,
        away_team_id=game.away_team_id,
    )

    scheduled = _utc(game.scheduled)
    p_elo = float(elo_features["p_elo"])
    row: dict[str, Any] = {
        "game_id": game.game_id,
        "game_ts": scheduled.isoformat(),
        "game_date": scheduled.date().isoformat(),
        "season": year,
        "is_playoff": int(str(raw_game.get("season_type") or "").upper() == "PST"),
        "home_team_id": game.home_team_id,
        "away_team_id": game.away_team_id,
        "home_franchise_id": home_fid,
        "away_franchise_id": away_fid,
        "home_elo_pre": float(elo_features["home_elo_pre"]),
        "away_elo_pre": float(elo_features["away_elo_pre"]),
        "p_elo": p_elo,
        "base_margin": _logit(p_elo),
        "home_win": None,
    }
    row.update(player_features)
    row.update(form_features)
    row.update(style_features)
    row.update(schedule_features)

    missing = [col for col in GOLD_MODEL_INPUT_COLS if col not in row]
    if missing:
        raise ValueError(f"live feature row missing columns: {missing}")
    feature_row = pd.DataFrame([{col: row.get(col) for col in GOLD_MODEL_INPUT_COLS}])

    source_report = LiveFeatureSourceReport(
        asof_ts_utc=asof_ts.isoformat(),
        schedule_files=tuple(str(path) for path in schedule_files),
        player_state_path=str(player_state_path),
        elo_path=str(elo_path),
        recent_form_path=str(recent_form_path),
        style_path=str(style_path),
        prev_style_path=str(prev_style_path) if prev_style_path else "",
        warnings=tuple(warnings),
        player_counts=player_counts,
    )
    return LiveFeatureResult(game=game, feature_row=feature_row, source_report=source_report)


def write_live_feature_artifacts(
    *,
    result: LiveFeatureResult,
    feature_csv: Path,
    packet_json: Path,
    prediction: Mapping[str, Any],
    train_csv: Path,
    model_best_round: Optional[int],
    model_best_round_source: Optional[str] = None,
    mapping: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    feature_csv = Path(feature_csv)
    packet_json = Path(packet_json)
    feature_csv.parent.mkdir(parents=True, exist_ok=True)
    packet_json.parent.mkdir(parents=True, exist_ok=True)
    result.feature_row.to_csv(feature_csv, index=False)

    p_home = float(prediction["p_home"][0])
    p_elo = float(result.feature_row["p_elo"].iloc[0])
    selected_side = "home" if p_home >= 0.5 else "away"
    selected_team_id = (
        result.game.home_team_id if selected_side == "home" else result.game.away_team_id
    )
    selected_team_name = (
        result.game.home_team_name if selected_side == "home" else result.game.away_team_name
    )
    packet = {
        "schema_version": LIVE_PACKET_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "created_ts_ms": int(time.time() * 1000),
        "game_id": result.game.game_id,
        "scheduled_utc": _utc(result.game.scheduled).isoformat(),
        "home_team_id": result.game.home_team_id,
        "away_team_id": result.game.away_team_id,
        "home_team_name": result.game.home_team_name,
        "away_team_name": result.game.away_team_name,
        "p_home": p_home,
        "p_away": 1.0 - p_home,
        "p_raw": float(prediction["p_raw"][0]),
        "p_elo": p_elo,
        "base_margin": float(result.feature_row["base_margin"].iloc[0]),
        "selected_side_label": selected_side,
        "selected_team_id": selected_team_id,
        "selected_team_name": selected_team_name,
        "feature_csv": str(feature_csv),
        "train_csv": str(train_csv),
        "model_best_round": model_best_round,
        "model_best_round_source": model_best_round_source,
        "source_report": result.source_report.to_json(),
        "mapping": dict(mapping or {}),
    }
    _write_json(packet_json, packet)
    return packet


def _load_schedule_games(repo_root: Path, year: int) -> tuple[dict[str, dict[str, Any]], tuple[Path, ...]]:
    out: dict[str, dict[str, Any]] = {}
    files: list[Path] = []
    for season_type in ("REG", "PST"):
        paths = sorted((repo_root / "data" / "bronze").glob(f"schedule_{year}_{season_type}__*.json"))
        if not paths:
            continue
        path = paths[-1]
        files.append(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw in data.get("games", []) or []:
            gid = str(raw.get("id") or "")
            if not gid:
                continue
            rec = dict(raw)
            rec["season_type"] = season_type
            out[gid] = rec
    if not out:
        raise FileNotFoundError(f"no latest schedule files found for {year}")
    return out, tuple(files)


def _game_ref_from_schedule(raw: Mapping[str, Any]) -> SportRadarGameRef:
    home = raw.get("home") or {}
    away = raw.get("away") or {}
    return SportRadarGameRef(
        game_id=str(raw.get("id") or ""),
        scheduled=_utc(parse_datetime(str(raw.get("scheduled")))),
        home_team_id=str(home.get("id") or ""),
        away_team_id=str(away.get("id") or ""),
        home_team_name=_team_display_name(home),
        away_team_name=_team_display_name(away),
    )


def _team_display_name(team: Mapping[str, Any]) -> str:
    parts = [str(team.get("market") or "").strip(), str(team.get("name") or "").strip()]
    return " ".join(part for part in parts if part).strip()


def _live_elo_features(
    *,
    repo_root: Path,
    year: int,
    asof_ts: datetime,
    home_team_id: str,
    away_team_id: str,
    home_franchise_id: str,
    away_franchise_id: str,
) -> tuple[dict[str, float], Path]:
    params = EloParams(H=25, K=20, a=0.45, b=1.0)
    cur_path = repo_root / "data" / "silver_plus" / f"elo_franchise_team_game_{year}_REGPST.csv"
    prev_path = repo_root / "data" / "silver_plus" / f"elo_franchise_team_game_{year-1}_REGPST.csv"
    ratings: dict[str, float] = {}

    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev["scheduled"] = pd.to_datetime(prev["scheduled"], utc=True, errors="coerce")
        prev = prev.sort_values(["scheduled", "game_id"], kind="stable")
        final_prev = prev.groupby("franchise_id", as_index=False).tail(1)
        ratings = {
            str(row["franchise_id"]): float(row["elo_post"])
            for _, row in final_prev.iterrows()
            if pd.notna(row.get("elo_post"))
        }
        ratings = apply_carryover(ratings, params)

    if cur_path.exists():
        cur = pd.read_csv(cur_path)
        cur["scheduled"] = pd.to_datetime(cur["scheduled"], utc=True, errors="coerce")
        cur = cur[cur["scheduled"] < asof_ts].sort_values(["scheduled", "game_id"], kind="stable")
        if len(cur):
            final_cur = cur.groupby("franchise_id", as_index=False).tail(1)
            for _, row in final_cur.iterrows():
                if pd.notna(row.get("elo_post")):
                    ratings[str(row["franchise_id"])] = float(row["elo_post"])

    home_elo = float(ratings.get(home_franchise_id, params.mu))
    away_elo = float(ratings.get(away_franchise_id, params.mu))
    return {
        "home_elo_pre": home_elo,
        "away_elo_pre": away_elo,
        "p_elo": elo_prob(home_elo, away_elo, H=params.H, scale=params.scale),
    }, cur_path


def _live_player_features(
    *,
    repo_root: Path,
    year: int,
    asof_ts: datetime,
    home_team_id: str,
    away_team_id: str,
) -> tuple[dict[str, Any], dict[str, int], Path]:
    path = repo_root / "data" / "silver" / f"player_state_history_{year}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    state = pd.read_csv(path)
    state["player_id"] = state["player_id"].astype(str)
    state["asof_ts"] = pd.to_datetime(state["asof_ts"], utc=True, errors="coerce")
    state = state[state["asof_ts"] <= asof_ts].copy()
    if len(state):
        state = (
            state.sort_values(["player_id", "asof_ts"], kind="stable")
            .groupby("player_id", as_index=False)
            .tail(1)
        )
    features: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for side, team_id in (("home", home_team_id), ("away", away_team_id)):
        if "current_team_id" in state.columns:
            current_team_id = state["current_team_id"].astype(str)
        else:
            current_team_id = pd.Series("", index=state.index)
        pool = state[current_team_id == str(team_id)].copy()
        counts[side] = int(len(pool))
        pool["strength"] = pd.to_numeric(pool.get("strength"), errors="coerce").fillna(0.0)
        pool["m_ewma"] = pd.to_numeric(pool.get("m_ewma"), errors="coerce").fillna(0.0)
        pool["q"] = pd.to_numeric(pool.get("q"), errors="coerce").fillna(0.0)
        pool = pool.sort_values(
            ["strength", "m_ewma", "q", "player_id"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        for slot in range(1, 8):
            if slot <= len(pool):
                row = pool.iloc[slot - 1]
                for feat in PLAYER_FEATS:
                    raw = feat.removesuffix("_pre")
                    features[f"{side}_p{slot}_{feat}"] = _none_if_nan(row.get(raw))
            else:
                for feat in PLAYER_FEATS:
                    features[f"{side}_p{slot}_{feat}"] = None
    return features, counts, path


def _live_recent_form_features(
    *,
    repo_root: Path,
    year: int,
    asof_ts: datetime,
    home_franchise_id: str,
    away_franchise_id: str,
) -> tuple[dict[str, float], Path]:
    path = repo_root / "data" / "silver_plus" / f"game_franchise_recent_form_{year}_REGPST.csv"
    state: dict[str, dict[str, float]] = {}
    if path.exists():
        df = pd.read_csv(path)
        df["game_ts"] = pd.to_datetime(df["game_ts"], utc=True, errors="coerce")
        df = df[df["game_ts"] < asof_ts].sort_values(["game_ts", "game_id"], kind="stable")
        for _, row in df.iterrows():
            fid = str(row.get("franchise_id"))
            state[fid] = {}
            for feat in FORM_FEATS:
                game_value = _to_float(row.get(feat.replace("_ewma_pre", "_game"), 0.0))
                pre_value = _to_float(row.get(feat, 0.0))
                state[fid][feat] = RECENT_FORM_LAMBDA * game_value + (1.0 - RECENT_FORM_LAMBDA) * pre_value

    out: dict[str, float] = {}
    for side, fid in (("home", home_franchise_id), ("away", away_franchise_id)):
        st = state.get(fid, {})
        for feat in FORM_FEATS:
            out[f"{side}_{feat}"] = float(st.get(feat, 0.0))
    return out, path


def _live_style_features(
    *,
    repo_root: Path,
    year: int,
    asof_ts: datetime,
    home_franchise_id: str,
    away_franchise_id: str,
    franchise_map: pd.DataFrame,
) -> tuple[dict[str, float], Path, Optional[Path]]:
    style_path = repo_root / "data" / "silver_plus" / f"game_franchise_style_profile_{year}_REGPST.csv"
    prev_path = repo_root / "data" / "silver_plus" / f"franchise_style_profile_final_{year-1}.csv"
    prev_by_fid: dict[str, Mapping[str, Any]] = {}
    league_avg = {feat.removesuffix("_pre"): 0.0 for feat in STYLE_FEATS}
    if prev_path.exists():
        prev = pd.read_csv(prev_path)
        prev["franchise_id"] = prev["franchise_id"].astype(str)
        prev_by_fid = prev.set_index("franchise_id").to_dict(orient="index")
        for feat in league_avg:
            if feat in prev.columns and len(prev):
                league_avg[feat] = float(pd.to_numeric(prev[feat], errors="coerce").mean())

    totals_by_fid = _current_style_totals(repo_root, year, asof_ts, franchise_map)
    out: dict[str, float] = {}
    for side, fid in (("home", home_franchise_id), ("away", away_franchise_id)):
        if fid in totals_by_fid:
            metrics = _style_from_totals(totals_by_fid[fid])
        elif fid in prev_by_fid:
            metrics = {
                feat.removesuffix("_pre"): float(prev_by_fid[fid].get(feat.removesuffix("_pre"), 0.0))
                for feat in STYLE_FEATS
            }
        else:
            metrics = dict(league_avg)
        for feat in STYLE_FEATS:
            out[f"{side}_{feat}"] = float(metrics.get(feat.removesuffix("_pre"), 0.0))
    return out, style_path, prev_path if prev_path.exists() else None


def _current_style_totals(
    repo_root: Path,
    year: int,
    asof_ts: datetime,
    franchise_map: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    played_path = repo_root / "data" / "silver" / f"played_franchise_games_{year}_REGPST.csv"
    if not played_path.exists():
        return {}
    played = pd.read_csv(played_path)
    played["scheduled"] = pd.to_datetime(played["scheduled"], utc=True, errors="coerce")
    played = played[played["scheduled"] < asof_ts].sort_values(["scheduled", "game_id"], kind="stable")
    latest = _pick_latest_game_summary_files(repo_root, set(played["game_id"].astype(str)))
    totals: dict[str, dict[str, float]] = {}
    for _, game in played.iterrows():
        gid = str(game["game_id"])
        path = latest.get(gid)
        if path is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        home = data.get("home") or {}
        away = data.get("away") or {}
        h = _extract_style_inputs(home)
        a = _extract_style_inputs(away)
        for team_id, opp_id, inputs, opp_inputs in (
            (str(game["home_id"]), str(game["away_id"]), h, a),
            (str(game["away_id"]), str(game["home_id"]), a, h),
        ):
            fid = map_team_to_franchise(team_id, year, franchise_map)
            bucket = totals.setdefault(fid, _zero_style_totals())
            bucket["FGA"] += inputs["FGA"]
            bucket["3PA"] += inputs["3PA"]
            bucket["FTA"] += inputs["FTA"]
            bucket["TO"] += inputs["TO"]
            bucket["OppFGA"] += opp_inputs["FGA"]
            bucket["Opp3PA"] += opp_inputs["3PA"]
            bucket["OppFTA"] += opp_inputs["FTA"]
            bucket["OppTO"] += opp_inputs["TO"]
    return totals


def _live_schedule_features(
    *,
    raw_game: Mapping[str, Any],
    schedule_games: Mapping[str, Mapping[str, Any]],
    year: int,
    home_team_id: str,
    away_team_id: str,
) -> dict[str, Any]:
    team_home = _team_home_map(schedule_games)
    game_ts = _utc(parse_datetime(str(raw_game.get("scheduled"))))
    game_date = game_ts.astimezone(CENTRAL).date()
    current_venue = _venue_info(raw_game)
    out: dict[str, Any] = {}
    for side, team_id in (("home", home_team_id), ("away", away_team_id)):
        seq = _team_schedule_sequence(schedule_games, team_id)
        prev = None
        games_last_4 = 0
        games_last_7 = 0
        for item in seq:
            item_ts = _utc(parse_datetime(str(item.get("scheduled"))))
            if item_ts >= game_ts:
                continue
            item_date = item_ts.astimezone(CENTRAL).date()
            delta_days = (game_date - item_date).days
            if 1 <= delta_days <= 4:
                games_last_4 += 1
            if 1 <= delta_days <= 7:
                games_last_7 += 1
            if prev is None or item_ts > _utc(parse_datetime(str(prev.get("scheduled")))):
                prev = item

        if prev is not None:
            prev_ts = _utc(parse_datetime(str(prev.get("scheduled"))))
            prev_date = prev_ts.astimezone(CENTRAL).date()
            days_rest = (game_date - prev_date).days - 1
            is_b2b = 1 if days_rest == 0 else 0
        else:
            days_rest = 0
            is_b2b = 0

        home_info = team_home.get(team_id, {})
        if prev is None or days_rest >= 4:
            origin = home_info
        else:
            origin = _venue_info(prev) or home_info
        travel = _haversine_miles(
            origin.get("lat"),
            origin.get("lng"),
            current_venue.get("lat"),
            current_venue.get("lng"),
        )
        tz_shift = _timezone_shift_hours(origin.get("timezone"), current_venue.get("timezone"), game_ts)
        out[f"{side}_days_rest_pre"] = days_rest
        out[f"{side}_is_b2b_pre"] = is_b2b
        out[f"{side}_games_last_4_days_pre"] = games_last_4
        out[f"{side}_games_last_7_days_pre"] = games_last_7
        out[f"{side}_travel_miles_pre"] = round(travel, 1) if travel is not None else None
        out[f"{side}_timezone_shift_hours_pre"] = tz_shift
    return out


def _team_schedule_sequence(
    schedule_games: Mapping[str, Mapping[str, Any]],
    team_id: str,
) -> list[Mapping[str, Any]]:
    out = []
    for raw in schedule_games.values():
        home = raw.get("home") or {}
        away = raw.get("away") or {}
        if str(home.get("id") or "") == team_id or str(away.get("id") or "") == team_id:
            out.append(raw)
    return sorted(out, key=lambda raw: (str(raw.get("scheduled") or ""), str(raw.get("id") or "")))


def _team_home_map(schedule_games: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in sorted(schedule_games.values(), key=lambda x: str(x.get("scheduled") or "")):
        home = raw.get("home") or {}
        tid = str(home.get("id") or "")
        venue = _venue_info(raw)
        if tid and venue and tid not in out:
            out[tid] = venue
    return out


def _venue_info(raw: Mapping[str, Any]) -> dict[str, Any]:
    venue = raw.get("venue") or {}
    loc = venue.get("location") or {}
    tzs = raw.get("time_zones") or {}
    lat = _float_or_none(loc.get("lat"))
    lng = _float_or_none(loc.get("lng"))
    return {
        "city": venue.get("city"),
        "lat": lat,
        "lng": lng,
        "timezone": tzs.get("venue") or tzs.get("home"),
    }


def _timezone_shift_hours(origin_tz: Any, current_tz: Any, ts: datetime) -> Optional[int]:
    if not origin_tz or not current_tz:
        return None
    try:
        origin_offset = ZoneInfo(str(origin_tz)).utcoffset(ts)
        current_offset = ZoneInfo(str(current_tz)).utcoffset(ts)
    except Exception:
        return None
    if origin_offset is None or current_offset is None:
        return None
    return int(round((current_offset - origin_offset).total_seconds() / 3600.0))


def _pick_latest_game_summary_files(repo_root: Path, game_ids: set[str]) -> dict[str, Path]:
    best: dict[str, tuple[str, Path]] = {}
    for path in (repo_root / "data" / "bronze").glob("game_summary__*__*.json"):
        parts = path.name.split("__")
        if len(parts) < 3:
            continue
        gid = parts[1]
        if gid not in game_ids:
            continue
        ts = parts[2].replace(".json", "")
        if gid not in best or ts > best[gid][0]:
            best[gid] = (ts, path)
    return {gid: item[1] for gid, item in best.items()}


def _extract_style_inputs(team_block: Mapping[str, Any]) -> dict[str, float]:
    stats = (team_block or {}).get("statistics") or {}
    if stats.get("total_turnovers") is not None:
        turnovers = _to_float(stats.get("total_turnovers"))
    else:
        player_tov = stats.get("player_turnovers")
        team_tov = stats.get("team_turnovers")
        turnovers = (
            _to_float(player_tov) + _to_float(team_tov)
            if player_tov is not None and team_tov is not None
            else _to_float(stats.get("turnovers"))
        )
    return {
        "FGA": _to_float(stats.get("field_goals_att")),
        "3PA": _to_float(stats.get("three_points_att")),
        "FTA": _to_float(stats.get("free_throws_att")),
        "TO": turnovers,
    }


def _style_from_totals(totals: Mapping[str, float]) -> dict[str, float]:
    return {
        "off_3pa_rate": _safe_div(totals["3PA"], totals["FGA"]),
        "def_3pa_allowed": _safe_div(totals["Opp3PA"], totals["OppFGA"]),
        "off_2pa_rate": _safe_div(totals["FGA"] - totals["3PA"], totals["FGA"]),
        "def_2pa_allowed": _safe_div(totals["OppFGA"] - totals["Opp3PA"], totals["OppFGA"]),
        "off_tov_pct": _safe_div(totals["TO"], totals["FGA"] + 0.44 * totals["FTA"] + totals["TO"]),
        "def_forced_tov": _safe_div(
            totals["OppTO"],
            totals["OppFGA"] + 0.44 * totals["OppFTA"] + totals["OppTO"],
        ),
    }


def _zero_style_totals() -> dict[str, float]:
    return {
        "FGA": 0.0,
        "3PA": 0.0,
        "FTA": 0.0,
        "TO": 0.0,
        "OppFGA": 0.0,
        "Opp3PA": 0.0,
        "OppFTA": 0.0,
        "OppTO": 0.0,
    }


def _haversine_miles(lat1: Any, lng1: Any, lat2: Any, lng2: Any) -> Optional[float]:
    lat1 = _float_or_none(lat1)
    lng1 = _float_or_none(lng1)
    lat2 = _float_or_none(lat2)
    lng2 = _float_or_none(lng2)
    if None in (lat1, lng1, lat2, lng2):
        return None
    radius = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _safe_div(numer: float, denom: float) -> float:
    try:
        denom = float(denom)
        if denom <= 0 or not math.isfinite(denom):
            return 0.0
        out = float(numer) / denom
        return out if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def _logit(p: float) -> float:
    p = max(P_CLIP, min(1.0 - P_CLIP, float(p)))
    return math.log(p / (1.0 - p))


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _to_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _float_or_none(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    clean = _jsonable(payload)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return str(value)
