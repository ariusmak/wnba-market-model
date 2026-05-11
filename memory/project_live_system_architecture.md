---
name: live-system-architecture
description: current production live-trading architecture and data-layer contract
type: project
---

This repo is production-only. Research/tuning workspaces live elsewhere.

Production execution lives in:
- `src/srwnba/live/canonical/`
- `pipelines/07_live/canonical/`

Legacy explicit YES-only execution lives in:
- `src/srwnba/live/legacy/`
- `pipelines/07_live/legacy/`

Legacy is retained for comparison/tests only. Do not add canonical production behavior there.

Prediction:
- `src/srwnba/util/final_model.py` is the production probability source.
- `src/srwnba/util/model_schema.py` is the canonical 160-feature schema and gold column order.
- Production probabilities use `p_raw` / `p_home`; no Platt calibrator.
- Production training uses the current all-settled combined gold CSV at the T-20 probability run.
- `data/gold/game_xgboost_input_2015_2024_REGPST.csv` is only the 2025 holdout regression baseline.

Data-layer contract:
- `bronze`: immutable accepted per-API-call payloads promoted from staged pulls.
- `bronze_runs`: isolated run cache and manifests before promotion.
- `silver`: canonical chronological facts at natural grain plus daily/as-of state. Examples: outcomes, player-game boxes, game availability, injury events/episodes, played-game manifests, `player_state_history`.
- `silver_plus`: game-as-of feature-family tables. Examples: `game_team_player`, franchise Elo, recent form, style, schedule/travel.
- `gold`: strict model-input projection, one row per game, metadata plus exactly 160 ordinary features.
- `runs`: recommended per-refresh/T-20 audit packets.

Important separation:
- `data/silver/player_state_history_{year}.csv` is daily/as-of player state.
- `data/silver_plus/game_team_player_{year}_REGPST.csv` is the full all-listed-player game-as-of player feature store.
- `data/gold/game_xgboost_input_{year}_REGPST.csv` is only the top-7-per-side/160-feature model matrix.

Production append sequence:
1. `python pipelines/07_live/08_append_year.py --year 2026`
2. `python pipelines/07_live/11_extend_elo_to_year.py --year 2026 --force` when skipping the full multiyear Elo rebuild
3. `python pipelines/07_live/08_append_year.py --year 2026 --from-phase gold`
4. `python pipelines/07_live/09_combine_gold.py --start-year 2015 --end-year 2026 --force`
5. `python pipelines/07_live/13_validate_production_artifacts.py --year 2026`

The validator enforces the layer split, gold schema, non-cold live-year player priors, full player-store presence, and future-injury hygiene.
