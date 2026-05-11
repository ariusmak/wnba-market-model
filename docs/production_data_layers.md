# Production Data Layer Contract

This repository is production-only. The data pipeline has four layers plus run audit artifacts.

## Bronze

Purpose: immutable accepted API payloads.

Grain: one file per API response.

Rules:
- Write every new pull to `data/bronze_runs/<run_id>/` first.
- Promote only validated payloads to `data/bronze/`.
- Never edit promoted bronze files.
- Do not pre-create future daily-injury files.

Bronze answers: what exactly did an API return, and when did we know it?

## Silver

Purpose: canonical chronological facts at their natural grain.

Allowed examples:
- schedules and played-game manifests
- game outcomes
- player-game box scores
- game availability facts
- injury events, updates, DNP evidence, and episodes
- daily/as-of `player_state_history`

Rules:
- Keep source chronology and knowledge timing explicit.
- Deduplicate by table primary key.
- Do not build model slots here.
- Do not write game-as-of feature-family tables here.

Silver answers: what is the clean historical record?

## Silver Plus

Purpose: game-as-of feature-family tables.

Allowed examples:
- `game_team_player_{year}_REGPST.csv`
- `elo_franchise_team_game_{year}_REGPST.csv`
- `game_franchise_recent_form_{year}_REGPST.csv`
- `game_franchise_style_profile_{year}_REGPST.csv`
- `game_team_schedule_context_{year}_REGPST.csv`
- final state snapshots used for next-season carryover

Rules:
- Every row must represent feature state available before a specific game, or a carryover snapshot used to create that state.
- `game_team_player` includes every listed player; it is not the top-7 projection.
- Do not store daily `player_state_history` here.

Silver Plus answers: what did we know for this game before tipoff?

## Gold

Purpose: strict model-input projection.

Rules:
- One row per game.
- Metadata plus exactly 160 ordinary model features.
- `base_margin` is present but passed separately to XGBoost.
- Player block is top 7 strength-ranked slots per side from `silver_plus/game_team_player`.
- Do not include p8-p12, player IDs/names, `strength_pre`, origin/current city debug fields, or research/debug columns.

Gold answers: what exact matrix does `FinalModel` train/score on?

## Runs

Purpose: audit each daily refresh or T-20 probability run.

Recommended path:

```text
data/runs/<run_id>/
  manifest.json
  validation_report.json
  bronze_inputs.json
  promoted_outputs.json
  prediction_packet.json
  logs.jsonl
```

Runs answer: what happened in this production execution?
