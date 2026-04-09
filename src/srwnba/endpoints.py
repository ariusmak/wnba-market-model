from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointConfig:
    # Most Sportradar feeds follow:
    # https://api.sportradar.com/wnba/{access_level}/v8/{language_code}/...
    access_level: str = "trial"     # change to "production" if your key is prod
    version: str = "v8"
    language_code: str = "en"
    base_url: str = "https://api.sportradar.com"


def wnba_base(cfg: EndpointConfig) -> str:
    return f"{cfg.base_url}/wnba/{cfg.access_level}/{cfg.version}/{cfg.language_code}"


def league_hierarchy(cfg: EndpointConfig) -> str:
    return f"{wnba_base(cfg)}/league/hierarchy.json"

def season_schedule(cfg: EndpointConfig, season_year: int, season_type: str = "REG") -> str:
    # season_type usually one of: "REG" (regular), "PST" (playoffs), "PRE" (preseason)
    return f"{wnba_base(cfg)}/games/{season_year}/{season_type}/schedule.json"

def game_summary(cfg: EndpointConfig, game_id: str) -> str:
    return f"{wnba_base(cfg)}/games/{game_id}/summary.json"

def injuries(cfg: EndpointConfig) -> str:
    return f"{wnba_base(cfg)}/league/injuries.json"

def daily_injuries(cfg: EndpointConfig, year: int, month: int, day: int) -> str:
    return f"{wnba_base(cfg)}/league/{year:04d}/{month:02d}/{day:02d}/daily_injuries.json"

def game_boxscore(cfg: EndpointConfig, game_id: str) -> str:
    return f"{wnba_base(cfg)}/games/{game_id}/boxscore.json"