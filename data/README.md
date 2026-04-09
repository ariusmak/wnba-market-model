# Data Directory

This directory contains all data artifacts produced by the pipeline.
Data files are **not** checked into version control — only the folder structure is tracked via `.gitkeep` files.

To populate, run the pipeline scripts in order from `pipelines/01_ingestion/` through `pipelines/05_modeling/`.

## Folder Layout

### `bronze/`
Raw JSON responses from the Sportradar API.
- `schedule_{year}_{season_type}__{timestamp}.json`
- `game_summary__{game_id}__{timestamp}.json`
- `daily_injuries__YYYY-MM-DD__{timestamp}.json`

### `silver/`
First-pass normalized CSVs parsed from bronze JSON.
- Game metadata: `games_{year}_{st}.csv`, `game_teams_{year}_{st}.csv`
- Game outcomes: `game_outcomes_{year}_REGPST.csv`
- Injury pipeline: `injury_events_{year}.csv`, `injury_updates_clean_{year}.csv`, `injury_dnp_evidence_{year}.csv`, `injury_episodes_{year}.csv`
- Player data: `player_game_box_{year}_{st}.csv`, `player_state_history_{year}_{st}.csv`
- Game rosters: `game_team_player_{year}_{st}.csv`, `game_availability_{year}_{st}.csv`
- Manifests: `played_games_{year}_REGPST.csv`, `played_franchise_games_{year}_REGPST.csv`
- Team form/style (team-keyed): `game_team_recent_form_{year}_{st}.csv`

### `silver_plus/`
Intermediate feature tables keyed by franchise (handles Stars → Aces continuity).
- `elo_franchise_team_game_{year}_REGPST.csv` — pregame Elo ratings, p_elo, base_margin
- `game_franchise_recent_form_{year}_REGPST.csv` — EWMA form metrics
- `game_franchise_style_profile_{year}_REGPST.csv` — season-to-date style tendencies
- `franchise_style_profile_final_{year}.csv` — end-of-season snapshot for next-year initialization
- `game_team_schedule_context_{year}_REGPST.csv` — rest, travel, timezone
- `player_state_history_{year}.csv` — player m_ewma, q, injury flags

### `gold/`
Final ML-ready datasets (one row per game, 160 ordinary features + base_margin).
- `game_xgboost_input_{year}_REGPST.csv` — per-year
- `game_xgboost_input_2015_2024_REGPST.csv` — combined training set
- `game_outcomes_2025_REG.csv` — 2025 outcomes for final evaluation

#### `gold/stage1/`, `gold/stage2/`, `gold/stage3/`
Tuning stage outputs (grid search results, variant gold tables).

#### `gold/features_only/`, `gold/final/`
Feature-only variants and final frozen model outputs.

### `config/`
Static configuration files.
- `franchise_map.csv` — team_id → franchise_id mapping

### `calibration/`
Platt scaling calibration outputs.
- `platt_calibrator_2020_2024.csv`, `oof_raw_preds_*.csv`, `oof_calibrated_preds_*.csv`
- `reliability_diagram_*.png`

### `kalshi/`
Kalshi prediction market data.
- `wnba_2025_game_markets.csv`, `wnba_2025_game_markets_matched.csv`
- `kalshi_markets.csv`, `kalshi_candles_1m.csv`, `kalshi_trades.csv`, `kalshi_settlements.csv`
- Backtest outputs: `backtest_trades_*.csv`, `backtest_*_summary.csv`

### `polymarket/`
Polymarket prediction market data.
- `polymarket_events.csv`, `polymarket_markets.csv`, `polymarket_tokens.csv`
- `polymarket_prices_history.csv`, `polymarket_trades.csv`, `polymarket_settlements.csv`
- Backtest outputs: `backtest_*_summary.csv`

### `final_comparisons/`
Cross-model comparison outputs (Elo vs XGBoost vs ensemble vs markets).

### `logit_benchmark/`
Logistic regression benchmark outputs.

### `Tables/`
Publication-ready tables and plots.

### `spec_sheets/`
Data specification documents for each table in the pipeline.
