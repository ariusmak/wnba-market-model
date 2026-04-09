"""
Franchise ID mapper: stable identity above raw Sportradar team_id.

Usage:
    from srwnba.util.franchise import load_franchise_map, add_franchise_cols

    map_df = load_franchise_map()
    df = add_franchise_cols(df, season_year_col="season_year")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Module-level singleton cache
_MAP_DF: Optional[pd.DataFrame] = None
_MAP_PATH_USED: Optional[str] = None


def load_franchise_map(path: str = "data/config/franchise_map.csv") -> pd.DataFrame:
    """Load and cache the franchise mapping CSV.

    Columns: team_id (str), start_year (int), end_year (int),
             franchise_id (str), franchise_name (str).
    """
    global _MAP_DF, _MAP_PATH_USED
    resolved = str(Path(path).resolve())
    if _MAP_DF is not None and _MAP_PATH_USED == resolved:
        return _MAP_DF

    df = pd.read_csv(path, dtype={"team_id": str, "franchise_id": str, "franchise_name": str})
    df["start_year"] = df["start_year"].astype(int)
    df["end_year"] = df["end_year"].astype(int)

    # Validate: no team_id + overlapping year ranges
    for team_id, grp in df.groupby("team_id"):
        rows = grp.sort_values("start_year")
        prev_end: Optional[int] = None
        for _, r in rows.iterrows():
            if prev_end is not None and r["start_year"] <= prev_end:
                raise ValueError(
                    f"Overlapping year ranges for team_id={team_id} in {path}"
                )
            prev_end = r["end_year"]

    _MAP_DF = df
    _MAP_PATH_USED = resolved
    return _MAP_DF


def map_team_to_franchise(
    team_id: str,
    season_year: int,
    map_df: pd.DataFrame,
) -> str:
    """Return franchise_id for (team_id, season_year).

    Falls back to team_id itself (identity mapping) if no match.
    Raises if multiple rows match (data integrity error).
    """
    mask = (
        (map_df["team_id"] == team_id)
        & (map_df["start_year"] <= season_year)
        & (map_df["end_year"] >= season_year)
    )
    matches = map_df[mask]
    if len(matches) == 0:
        return team_id
    if len(matches) > 1:
        raise ValueError(
            f"Multiple franchise mappings for team_id={team_id}, season_year={season_year}: "
            f"{matches[['franchise_id', 'start_year', 'end_year']].to_dict('records')}"
        )
    return str(matches.iloc[0]["franchise_id"])


def add_franchise_cols(
    df: pd.DataFrame,
    season_year_col: str,
    team_id_col: str = "team_id",
    opp_id_col: str = "opponent_team_id",
    map_path: str = "data/config/franchise_map.csv",
) -> pd.DataFrame:
    """Add franchise_id and opponent_franchise_id columns to df.

    Modifies a copy; does not mutate input.
    """
    map_df = load_franchise_map(map_path)
    df = df.copy()

    def _map(row: pd.Series, col: str) -> str:
        return map_team_to_franchise(
            str(row[col]), int(row[season_year_col]), map_df
        )

    df["franchise_id"] = df.apply(lambda r: _map(r, team_id_col), axis=1)
    if opp_id_col in df.columns:
        df["opponent_franchise_id"] = df.apply(lambda r: _map(r, opp_id_col), axis=1)

    return df
