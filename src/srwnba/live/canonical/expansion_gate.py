"""
First-season expansion-team trading gate.

Expansion teams are forecasted and kept in all model-state updates from
game 1, but the live trading layer is blocked until every true first-season
expansion team in the game has completed at least 14 prior games.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


EXPANSION_TEAM_MIN_COMPLETED_GAMES = 14

TRUE_EXPANSION_TEAMS_2026: Dict[str, str] = {
    "4e4f726e-a015-4306-91a7-28e8576c7868": "Toronto Tempo",
    "d54283cc-c5ec-4dbd-bb61-166f217e3864": "Portland Fire",
}


@dataclass(frozen=True)
class ExpansionGateResult:
    allowed: bool
    reason: str
    applies: bool
    expansion_team_ids: Tuple[str, ...]
    completed_games_by_team: Dict[str, int]
    min_completed_games_required: int = EXPANSION_TEAM_MIN_COMPLETED_GAMES

    def to_log_payload(self) -> Dict[str, object]:
        return {
            "expansion_team_gate_passed": self.allowed,
            "expansion_team_gate_reason": self.reason,
            "expansion_team_gate_applies": self.applies,
            "expansion_team_ids": list(self.expansion_team_ids),
            "expansion_team_completed_games": dict(self.completed_games_by_team),
            "expansion_team_min_completed_games_required": self.min_completed_games_required,
        }


def evaluate_expansion_team_gate(
    *,
    home_team_id: str,
    away_team_id: str,
    completed_games_by_team: Mapping[str, int],
    expansion_team_ids: Optional[Mapping[str, str]] = None,
    min_completed_games: int = EXPANSION_TEAM_MIN_COMPLETED_GAMES,
) -> ExpansionGateResult:
    """Return whether live trading is allowed for a game.

    This function is deliberately pure. Callers must pass completed prior-game
    counts that were calculated chronologically before the target game.
    """
    expansion_lookup = expansion_team_ids or TRUE_EXPANSION_TEAMS_2026
    game_team_ids = (str(home_team_id), str(away_team_id))
    expansion_in_game = tuple(team_id for team_id in game_team_ids if team_id in expansion_lookup)
    counts = {
        team_id: int(completed_games_by_team.get(team_id, 0))
        for team_id in expansion_in_game
    }

    if not expansion_in_game:
        return ExpansionGateResult(
            allowed=True,
            reason="no_expansion_teams",
            applies=False,
            expansion_team_ids=(),
            completed_games_by_team={},
            min_completed_games_required=min_completed_games,
        )

    blockers = [
        (
            team_id,
            str(expansion_lookup.get(team_id) or team_id),
            counts[team_id],
        )
        for team_id in expansion_in_game
        if counts[team_id] < min_completed_games
    ]
    if blockers:
        details = "; ".join(
            f"{name} completed_games={count} < {min_completed_games}"
            for _, name, count in blockers
        )
        return ExpansionGateResult(
            allowed=False,
            reason=f"blocked_expansion_team_under_14_completed_games: {details}",
            applies=True,
            expansion_team_ids=expansion_in_game,
            completed_games_by_team=counts,
            min_completed_games_required=min_completed_games,
        )

    return ExpansionGateResult(
        allowed=True,
        reason="expansion_team_gate_passed",
        applies=True,
        expansion_team_ids=expansion_in_game,
        completed_games_by_team=counts,
        min_completed_games_required=min_completed_games,
    )
