"""
Kalshi WNBA market mapping.

This module maps one Sportradar game to the two Kalshi winner markets for
that game, then builds the two equivalent selected-team-wins routes:

    - BUY YES on the selected team's market
    - BUY NO on the opponent team's market

The production side-mapping source of truth is the audited
`data/config/kalshi_team_name_map.csv` when available. Kalshi
`custom_strike.basketball_team` is retained as a diagnostic cross-check
because observed Kalshi values do not match this pipeline's Sportradar
team UUIDs.
"""
from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

OPEN_MARKET_STATUSES = {"active", "open"}
WNBA_SERIES_TICKERS = ("KXWNBAGAME", "KXWNBAH")
EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SportRadarGameRef:
    game_id: str
    scheduled: datetime
    home_team_id: str
    away_team_id: str
    home_team_name: str = ""
    away_team_name: str = ""

    @property
    def team_ids(self) -> frozenset[str]:
        return frozenset([self.home_team_id, self.away_team_id])

    @property
    def scheduled_dates_for_matching(self) -> Tuple[date, ...]:
        dates = [self.scheduled.date()]
        if self.scheduled.tzinfo is not None:
            dates.append(self.scheduled.astimezone(EASTERN).date())
        return tuple(dict.fromkeys(dates))


@dataclass(frozen=True)
class KalshiMarketSide:
    ticker: str
    event_ticker: str
    yes_team_id: str
    yes_team_id_source: str = ""
    custom_strike_team_id: str = ""
    yes_team_name: str = ""
    title: str = ""
    status: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def is_open(self) -> bool:
        return self.status.strip().lower() in OPEN_MARKET_STATUSES


@dataclass(frozen=True)
class KalshiGameMapping:
    game: SportRadarGameRef
    event_ticker: str
    home_market: KalshiMarketSide
    away_market: KalshiMarketSide
    side_mapping_confirmed: bool
    complement_market_confirmed: bool
    settlement_mapping_confirmed: bool
    candidate_count: int
    diagnostics: Tuple[str, ...] = ()

    @property
    def confirmed(self) -> bool:
        return (
            self.side_mapping_confirmed
            and self.complement_market_confirmed
            and self.settlement_mapping_confirmed
        )

    def market_for_team(self, team_id: str) -> KalshiMarketSide:
        if team_id == self.home_market.yes_team_id:
            return self.home_market
        if team_id == self.away_market.yes_team_id:
            return self.away_market
        raise KeyError(f"team_id {team_id} is not in mapped event {self.event_ticker}")


@dataclass(frozen=True)
class RouteCandidate:
    route_id: str
    canonical_exposure: str
    selected_team_id: str
    opponent_team_id: str
    selected_team_name: str
    opponent_team_name: str
    market_ticker: str
    event_ticker: str
    route_type: str
    action: str
    side: str
    market_yes_team_id: str
    market_yes_team_name: str
    side_mapping_confirmed: bool
    complement_market_confirmed: bool
    settlement_mapping_confirmed: bool

    @property
    def confirmed(self) -> bool:
        return (
            self.side_mapping_confirmed
            and self.complement_market_confirmed
            and self.settlement_mapping_confirmed
        )


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def parse_event_date(event_ticker: str) -> Optional[date]:
    match = re.match(r"^[A-Z0-9]+-(\d{2})([A-Z]{3})(\d{2})", event_ticker or "")
    if not match:
        return None
    yy, mon, dd = match.groups()
    month = MONTH_MAP.get(mon)
    if not month:
        return None
    return date(2000 + int(yy), month, int(dd))


def extract_custom_strike(raw: Mapping[str, Any]) -> Dict[str, Any]:
    value = raw.get("custom_strike")
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def extract_yes_team_id(raw: Mapping[str, Any]) -> str:
    strike = extract_custom_strike(raw)
    value = strike.get("basketball_team") or strike.get("team") or strike.get("team_id")
    return str(value or "").strip()


def normalize_team_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def market_series_ticker(raw: Mapping[str, Any]) -> str:
    explicit = str(raw.get("series_ticker") or "").strip()
    if explicit:
        return explicit
    event_ticker = str(raw.get("event_ticker") or "").strip()
    return event_ticker.split("-", 1)[0] if event_ticker else ""


def is_wnba_moneyline_market(raw: Mapping[str, Any]) -> bool:
    """Return True only for WNBA team-wins game markets.

    Kalshi can add multiple markets under nearby WNBA series over time. The
    live system must ignore props, totals, spreads, season futures, and other
    non-moneyline contracts even if they appear in a queried series.
    """
    series = market_series_ticker(raw)
    if series not in WNBA_SERIES_TICKERS:
        return False

    title = str(raw.get("title") or "").strip().lower()
    rules_primary = str(raw.get("rules_primary") or "").strip().lower()
    market_type = str(raw.get("market_type") or "").strip().lower()
    strike = extract_custom_strike(raw)

    if market_type and market_type != "binary":
        return False
    if "winner" not in title:
        return False
    if "wins" not in rules_primary:
        return False
    if "basketball game" not in rules_primary:
        return False
    if "resolves to yes" not in rules_primary:
        return False
    if "basketball_team" not in strike:
        return False
    return True


def filter_wnba_moneyline_markets(
    markets: Iterable[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    return [market for market in markets if is_wnba_moneyline_market(market)]


def is_open_wnba_moneyline_market(raw: Mapping[str, Any]) -> bool:
    if not is_wnba_moneyline_market(raw):
        return False
    return str(raw.get("status") or "").strip().lower() in OPEN_MARKET_STATUSES


def filter_open_wnba_moneyline_markets(
    markets: Iterable[Mapping[str, Any]],
) -> List[Mapping[str, Any]]:
    return [market for market in markets if is_open_wnba_moneyline_market(market)]


def load_team_name_map(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        for row in rows:
            team_id = str(row.get("sportradar_team_id") or "").strip()
            if not team_id:
                continue
            names = [str(row.get("kalshi_team_name") or "").strip()]
            aliases = str(row.get("kalshi_aliases") or row.get("aliases") or "").strip()
            if aliases:
                names.extend(part.strip() for part in re.split(r"[|;]", aliases) if part.strip())

            for name in names:
                if not name:
                    continue
                key = normalize_team_name(name)
                existing = out.get(key)
                if existing and existing != team_id:
                    raise ValueError(
                        f"ambiguous Kalshi team-name mapping for {name!r}: "
                        f"{existing} vs {team_id}"
                    )
                out[key] = team_id
    return out


def normalize_market(
    raw: Mapping[str, Any],
    *,
    team_name_to_id: Optional[Mapping[str, str]] = None,
) -> Optional[KalshiMarketSide]:
    if not is_wnba_moneyline_market(raw):
        return None
    ticker = str(raw.get("ticker") or raw.get("market_ticker") or "").strip()
    event_ticker = str(raw.get("event_ticker") or "").strip()
    yes_team_name = str(raw.get("yes_sub_title") or raw.get("yes_title") or "").strip()
    custom_team_id = extract_yes_team_id(raw)
    mapped_team_id = ""
    if team_name_to_id and yes_team_name:
        mapped_team_id = str(team_name_to_id.get(normalize_team_name(yes_team_name)) or "").strip()
    yes_team_id = mapped_team_id or custom_team_id
    source = "team_name_map" if mapped_team_id else ("custom_strike" if custom_team_id else "")
    if not ticker or not event_ticker or not yes_team_id:
        return None
    return KalshiMarketSide(
        ticker=ticker,
        event_ticker=event_ticker,
        yes_team_id=yes_team_id,
        yes_team_id_source=source,
        custom_strike_team_id=custom_team_id,
        yes_team_name=yes_team_name,
        title=str(raw.get("title") or "").strip(),
        status=str(raw.get("status") or "").strip(),
        raw=raw,
    )


def group_markets_by_event(
    markets: Iterable[Mapping[str, Any]],
    *,
    team_name_to_id: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[KalshiMarketSide]]:
    grouped: Dict[str, List[KalshiMarketSide]] = {}
    for raw in markets:
        market = normalize_market(raw, team_name_to_id=team_name_to_id)
        if market is None:
            continue
        grouped.setdefault(market.event_ticker, []).append(market)
    return grouped


def _date_distance(event_dt: Optional[date], allowed_dates: Sequence[date]) -> int:
    if event_dt is None:
        return 999
    return min(abs((event_dt - allowed).days) for allowed in allowed_dates)


def map_game_to_kalshi_markets(
    game: SportRadarGameRef,
    markets: Iterable[Mapping[str, Any]],
    *,
    max_date_slop_days: int = 1,
    require_open: bool = False,
    team_name_to_id: Optional[Mapping[str, str]] = None,
) -> KalshiGameMapping:
    grouped = group_markets_by_event(markets, team_name_to_id=team_name_to_id)
    diagnostics: List[str] = []
    candidates: List[Tuple[Tuple[int, int, str], str, Dict[str, KalshiMarketSide], List[KalshiMarketSide]]] = []

    for event_ticker, event_markets in grouped.items():
        by_team: Dict[str, KalshiMarketSide] = {}
        for market in event_markets:
            by_team.setdefault(market.yes_team_id, market)
            if (
                market.yes_team_id_source == "team_name_map"
                and market.custom_strike_team_id
                and market.custom_strike_team_id != market.yes_team_id
            ):
                diagnostics.append(
                    f"{event_ticker} {market.ticker}: custom_strike_team_id "
                    f"{market.custom_strike_team_id} overridden_by_team_name_map {market.yes_team_id}"
                )

        event_team_ids = frozenset(by_team)
        if game.team_ids != event_team_ids:
            diagnostics.append(
                f"skip {event_ticker}: team_ids={sorted(event_team_ids)} expected={sorted(game.team_ids)}"
            )
            continue

        event_dt = parse_event_date(event_ticker)
        distance = _date_distance(event_dt, game.scheduled_dates_for_matching)
        if distance > max_date_slop_days:
            diagnostics.append(f"skip {event_ticker}: date_distance={distance}")
            continue

        if require_open and not all(m.is_open for m in by_team.values()):
            statuses = {m.status for m in by_team.values()}
            diagnostics.append(f"skip {event_ticker}: not_open statuses={sorted(statuses)}")
            continue

        open_penalty = 0 if all(m.is_open for m in by_team.values()) else 1
        candidates.append(((distance, open_penalty, event_ticker), event_ticker, by_team, event_markets))

    if not candidates:
        empty = KalshiMarketSide("", "", "")
        return KalshiGameMapping(
            game=game,
            event_ticker="",
            home_market=empty,
            away_market=empty,
            side_mapping_confirmed=False,
            complement_market_confirmed=False,
            settlement_mapping_confirmed=False,
            candidate_count=0,
            diagnostics=tuple(diagnostics),
        )

    candidates.sort(key=lambda item: item[0])
    best_score = candidates[0][0]
    tied = [item for item in candidates if item[0][:2] == best_score[:2]]
    if len(tied) > 1:
        empty = KalshiMarketSide("", "", "")
        tied_events = ", ".join(item[1] for item in tied)
        return KalshiGameMapping(
            game=game,
            event_ticker="",
            home_market=empty,
            away_market=empty,
            side_mapping_confirmed=False,
            complement_market_confirmed=False,
            settlement_mapping_confirmed=False,
            candidate_count=len(candidates),
            diagnostics=tuple([*diagnostics, f"ambiguous candidates: {tied_events}"]),
        )

    _, event_ticker, by_team, _ = candidates[0]
    home_market = by_team[game.home_team_id]
    away_market = by_team[game.away_team_id]
    side_ok = (
        home_market.yes_team_id == game.home_team_id
        and away_market.yes_team_id == game.away_team_id
        and home_market.ticker != away_market.ticker
    )
    complement_ok = (
        home_market.event_ticker == away_market.event_ticker
        and frozenset([home_market.yes_team_id, away_market.yes_team_id]) == game.team_ids
    )
    settlement_ok = side_ok and complement_ok and bool(home_market.yes_team_id and away_market.yes_team_id)
    return KalshiGameMapping(
        game=game,
        event_ticker=event_ticker,
        home_market=home_market,
        away_market=away_market,
        side_mapping_confirmed=side_ok,
        complement_market_confirmed=complement_ok,
        settlement_mapping_confirmed=settlement_ok,
        candidate_count=len(candidates),
        diagnostics=tuple(diagnostics),
    )


def build_equivalent_routes(
    mapping: KalshiGameMapping,
    selected_team_id: str,
) -> Tuple[RouteCandidate, RouteCandidate]:
    if not mapping.confirmed:
        raise ValueError(f"cannot build routes from unconfirmed mapping: {mapping.diagnostics}")
    if selected_team_id not in mapping.game.team_ids:
        raise ValueError(f"selected_team_id {selected_team_id} is not in game {mapping.game.game_id}")

    opponent_team_id = (
        mapping.game.away_team_id
        if selected_team_id == mapping.game.home_team_id
        else mapping.game.home_team_id
    )
    selected_market = mapping.market_for_team(selected_team_id)
    opponent_market = mapping.market_for_team(opponent_team_id)
    selected_name = selected_market.yes_team_name
    opponent_name = opponent_market.yes_team_name

    base = {
        "canonical_exposure": "selected_team_wins",
        "selected_team_id": selected_team_id,
        "opponent_team_id": opponent_team_id,
        "selected_team_name": selected_name,
        "opponent_team_name": opponent_name,
        "event_ticker": mapping.event_ticker,
        "side_mapping_confirmed": mapping.side_mapping_confirmed,
        "complement_market_confirmed": mapping.complement_market_confirmed,
        "settlement_mapping_confirmed": mapping.settlement_mapping_confirmed,
    }
    buy_yes = RouteCandidate(
        **base,
        route_id=f"{mapping.event_ticker}:BUY_YES_SELECTED:{selected_market.ticker}",
        market_ticker=selected_market.ticker,
        route_type="BUY_YES_SELECTED",
        action="buy",
        side="yes",
        market_yes_team_id=selected_market.yes_team_id,
        market_yes_team_name=selected_market.yes_team_name,
    )
    buy_no = RouteCandidate(
        **base,
        route_id=f"{mapping.event_ticker}:BUY_NO_OPPONENT:{opponent_market.ticker}",
        market_ticker=opponent_market.ticker,
        route_type="BUY_NO_OPPONENT",
        action="buy",
        side="no",
        market_yes_team_id=opponent_market.yes_team_id,
        market_yes_team_name=opponent_market.yes_team_name,
    )
    return buy_yes, buy_no
