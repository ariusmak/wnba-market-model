# Data Directory - Production Layer Contract

This repo uses four production data layers plus per-run audit artifacts.

## `bronze/`

Immutable canonical raw payloads promoted from staged API pulls.

Grain:
- one file per accepted API response
- no manual merging
- no edits after promotion

Examples:
- `schedule_{year}_{season_type}__{timestamp}.json`
- `game_summary__{game_id}__{timestamp}.json`
- `daily_injuries__YYYY-MM-DD__{timestamp}.json`

## `bronze_runs/`

Isolated run cache for every new pull before canonical promotion.

Each run stores the raw response plus a manifest containing run id, pull timestamp, sanitized endpoint/request metadata, response hash, validation result, payload size, and promoted canonical path.

## `silver/`

Canonical chronological facts at their natural fact grain. This layer cleans, dedupes, normalizes, and merges source facts, but does not create final model slots.

Examples:
- `played_games_{year}_REGPST.csv`
- `game_outcomes_{year}_REGPST.csv`
- `player_game_box_{year}_REGPST.csv`
- `game_availability_{year}_REGPST.csv`
- `injury_events_{year}.csv`
- `injury_updates_clean_{year}.csv`
- `injury_episodes_{year}.csv`
- `player_state_history_{year}.csv`

`player_state_history` is daily/as-of state and belongs in silver.

## `silver_plus/`

Game-as-of feature-family tables. These tables answer: what pregame feature state was available for this specific game?

Examples:
- `game_team_player_{year}_REGPST.csv` - full all-listed-player game-as-of player feature store
- `elo_franchise_team_game_{year}_REGPST.csv`
- `game_franchise_recent_form_{year}_REGPST.csv`
- `game_franchise_style_profile_{year}_REGPST.csv`
- `game_team_schedule_context_{year}_REGPST.csv`
- final state snapshots used for next-season carryover, such as `franchise_style_profile_final_{year}.csv`

`silver_plus` must not contain daily `player_state_history` files.

## `gold/`

Strict model-input projections.

Examples:
- `game_xgboost_input_{year}_REGPST.csv`
- `game_xgboost_input_2015_2024_REGPST.csv` - 2025 holdout regression baseline
- `game_xgboost_input_2015_20YY_REGPST.csv` - current all-settled production training file

Gold is one row per game and contains metadata plus the locked 160 ordinary model features. It must not include full-player debug state, p8-p12 slots, player IDs/names, `strength_pre`, or research/debug columns.

## `runs/`

Recommended audit location for production refresh and T-20 probability runs.

Expected contents:
- `manifest.json`
- `validation_report.json`
- `bronze_inputs.json`
- `promoted_outputs.json`
- `prediction_packet.json`
- `logs.jsonl`

`runs/live_refresh/` is the live scheduler's audit area. Every scheduler wakeup writes a run directory with due-job decisions, command logs, bronze deltas, validation output, optional Kalshi market snapshots, and a manifest. Kalshi snapshots in the live path are filtered to active/open WNBA moneyline/team-wins markets only before they are used for mapping or scheduling. `runs/live_refresh/scheduler_state.json` is the duplicate-run guard for fixed daily jobs and T-20 market/game-time refreshes.

`runs/live_daemon/` is the 24/7 daemon audit area. It stores daemon sessions, heartbeats, health verdicts, every Kalshi market-list snapshot, detected market changes, and the daemon's persistent market/ticker state. The daemon calls the live refresh scheduler rather than rebuilding data itself. `health_latest.json` is the primary machine-readable current status file.

`runs/live_games/<game_id>/` is the canonical per-game execution ledger. It is append-only across route-loop restarts and stores:
- `manifest.json`
- `events.jsonl`
- `prediction_packet.json`
- `market_mapping.json`
- `market_snapshots.jsonl`
- `route_quotes.jsonl`
- `portfolio_sizing.jsonl`
- `execution_plans.jsonl`
- `orders.jsonl`
- `fills.jsonl`
- `positions.jsonl`
- `errors.jsonl`
- `summary.json`
- `sessions/<run_id>/...` copies for each process run

Use `summary.json` for a quick current view of prediction, mapping, latest quotes, plan, and position state. Use the JSONL files for chronology and audit.

## Other Production Directories

- `config/` - static mappings such as franchise and Kalshi team maps
- `kalshi/` - Kalshi market data used by live/reconciliation code
- `live_logs/` - flat JSONL execution and dry-run logs; useful as raw streams, but not the primary per-game audit packet
- `spec_sheets/` - table and feature specifications

Research/tuning artifacts do not belong in this production repo.
