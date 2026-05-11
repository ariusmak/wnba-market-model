# WNBA Prediction Market Model - Production Spec

This repository is now a production live-trading system. Research and tuning are complete. The forecasting stack, feature set, training procedure, data-pull timing, and v1.2 execution architecture are locked. Do not tune, ablate, substitute, reinterpret, or revive research-era alternatives unless the user explicitly unlocks the spec and requests a fresh validation pass.

System purpose: price WNBA moneyline markets on Kalshi with the locked Elo + XGBoost model, convert qualifying model-market disagreement into controlled pre-tipoff canonical selected-team-wins exposure through the v1.2 execution engine, and hold filled positions to settlement.

## 1. Locked Forecasting Truth

All forecasting constants live in `config/final_hyperparams.py`. Do not shadow them elsewhere.

### 1.1 Elo
- `H = 25`
- `K = 20`
- `a = 0.45`
- `b = 1.0`
- `mu = 1505`
- Single zero-sum Elo track.
- San Antonio Stars -> Las Vegas Aces is franchise continuity.
- Elo enters XGBoost as `base_margin = logit(p_elo)`, not as a feature column.

### 1.2 Features
- `N_players = 7`
- `h_M = 7`
- `L_inj = 14`
- `tau = 150`
- `h_team = 7`
- Recent form resets each season.
- Player priors and style profiles carry over.
- 6 raw style stats per team.
- Travel origin is previous-game city if `days_rest < 4`, else home city.
- First 9 games of 2015 are excluded because there is no 2014 prior.

### 1.3 XGBoost
- `max_depth = 6`
- `min_child_weight = 3`
- `gamma = 0.1`
- `colsample_bytree = 0.6`
- `subsample = 0.8`
- `reg_lambda = 1.0`
- `reg_alpha = 0.0`
- `learning_rate = 0.02`
- `num_boost_round = 3000`
- `early_stopping_rounds = 150`
- `XGB_PRODUCTION_NUM_BOOST_ROUND = 88`
- `seed = 42`
- 2015-2024 early-stop sanity value: best round 88.
- Production `FinalModel` trains on all cold-start-filtered rows in the supplied training CSV for exactly 88 trees. It must not reselect `best_round` from the final season of the training file, especially during a partial live season.

### 1.4 Full Feature Store vs Model Input

These are two different artifacts and must stay separate.

`data/silver/` contains canonical chronological facts at their natural grain: schedules, outcomes, player-game boxes, injury events/episodes, availability facts, played-game manifests, and daily/as-of `player_state_history`.

`data/silver_plus/game_team_player_{year}_REGPST.csv` is the full game-as-of player feature-building store. It tracks every listed player for each game/team, including player identifiers, names, `strength_pre`, minutes state, quality state, injury state, and audit fields. Other `silver_plus` tables are also game-as-of feature families: Elo, recent form, style, and schedule/travel context.

`data/gold/game_xgboost_input_{year}_REGPST.csv` is the production model-input projection. It contains metadata plus exactly the top 7 strength-ranked player slots per side and only the 9 locked player model features per slot. It must not contain p8-p12 columns, player IDs/names, `strength_pre`, origin/current city debug fields, or any research/debug columns.

The canonical schema is `src/srwnba/util/model_schema.py`. Builders and `FinalModel` must use that schema instead of rebuilding feature lists independently.

Locked ordinary feature count: 160 columns:
- 126 player features: 7 slots x 9 features x 2 teams
- 10 recent-form features
- 12 style-profile features
- 12 schedule/travel features

`base_margin` is passed separately.

### 1.5 Probability
- `p_raw = sigmoid(base_margin + xgb_output)` is production probability.
- No Platt scaling.
- `FinalModel.predict(df)["p_home"]` is the canonical accessor.

## 2. Locked Execution Truth

There are no open strategy or execution-design items in this repo. Entry timing, route selection, cadence, passive probes, child slicing, liquidity caps, brakes, and expansion-team gates are locked by v1.2.

Summary:
- Canonical exposure is selected team wins.
- The bot may buy YES selected-team market or buy NO opponent market only after side, complement, and settlement mapping are confirmed.
- Bankroll reference is `$5,000`.
- Live sizing defaults to Kalshi `/portfolio/balance.portfolio_value`; if unavailable, fall back to Kalshi `/portfolio/balance.balance`.
- `--sizing-bankroll-override <dollars>` intentionally replaces the Kalshi sizing bankroll with an imaginary bankroll, even if smaller or larger than real Kalshi wealth.
- Actual order feasibility uses Kalshi `/portfolio/balance.balance` after the 2% cash buffer unless `--available-cash-override <dollars>` is explicitly passed.
- Sizing is fee-adjusted half-Kelly.
- Per canonical market exposure is capped at 15% of bankroll plus liquidity/cash caps.
- If available cash is binding across live tickets, allocate scarce child-order cash by exact marginal expected log growth per dollar consumed, including existing fills/reserves as diminishing marginal utility. Tie scores within 5% by higher absolute edge, higher normalized edge, earlier first qualification, larger executable liquidity, then lower slippage. The only canonical implementation is `src/srwnba/live/canonical/cash_priority.py`.
- Edge gates: absolute edge at least 0.05 and normalized edge at least 0.25.
- All edge/Kelly math uses expected fee-adjusted executable average price.
- Core orders are limit-IOC sweeps.
- Passive post-only limits are allowed only as small early price-improvement probes under v1.2.
- No sells/reductions before settlement.
- Filled canonical exposure is monotone non-decreasing.
- True expansion teams are forecasted and update state from game 1, but trading is blocked until every first-season expansion team in the game has completed at least 14 prior games.

## 3. Data Timing and Bronze Discipline

Production timing is locked:
- Main settled-history refresh: 02:30 ET daily during season.
- Backup/correction refresh: 09:00 ET.
- Official pregame probability packet: T-20h.
- Manual review deadline and earliest automated trading eligibility: T-18h.
- From T-20h through the T-8h no-new-entry boundary, the daemon must keep
  date-indexed injury data fresh at least hourly. Any refreshed injury/player
  state that changes an open mapped game's live feature row must regenerate that
  game's live prediction packet; route loops treat the refreshed probability as
  the new truth for downstream planning. After exposure exists, a model-side
  flip must not create offsetting/reducing exposure; block new entry instead and
  hold existing fills to settlement.

Production training after launch uses the current all-settled combined gold CSV. For 2026 production, the T-20 prediction packet builder and canonical live route loop are hardcoded to train from `data/gold/game_xgboost_input_2015_2026_REGPST.csv`. Do not pass, re-enable, or depend on live `--train-csv` overrides. `FinalModel` uses the locked `XGB_PRODUCTION_NUM_BOOST_ROUND = 88` tree count on that full file; it does not early-stop against the partial 2026 slice. The `2015_2024` file is only the 2025 holdout regression baseline.

Date-indexed `daily_injuries` is the historical injury source of truth. It must be backfilled from the last accepted injury date through today. Recent dates are mutable, so the ingest must force-refresh the recent lookback window, defaulting to today plus yesterday, and preserve each response as a timestamped bronze payload. Do not create or reuse future daily-injury placeholders. `/league/injuries.json` may be pulled once at T-20 as a current-state audit cross-check, but it is not the primary historical feature feed.

Every Sportradar pull must be staged under `data/bronze_runs/<run_id>/` with a manifest before canonical promotion into `data/bronze`. The manifest must include run id, pull timestamp, request/endpoint metadata with secrets removed, response hash, validation result, and promoted canonical path. Downstream builders consume only accepted canonical sources with knowledge timestamps at or before the target as-of time.

`pipelines/07_live/14_live_data_refresh.py` is the production scheduler runner. Fixed daily refreshes and market/game-time T-20 refresh checks must go through this entrypoint so every wakeup gets a run packet, command logs, Kalshi market snapshot when applicable, bronze deltas, validation output, and duplicate-run state.

`pipelines/07_live/15_live_daemon.py` is the production 24/7 heartbeat. It monitors and caches Kalshi market creation/update state, hard-filters live working snapshots to active/open WNBA moneyline/team-wins markets only, writes heartbeats, invokes `14_live_data_refresh.py` for due refresh work, and invokes `16_execution_supervisor.py` to launch/supervise per-game route loops. It must not place orders itself or duplicate data-refresh logic.

All live Kalshi market snapshots must pass `srwnba.live.canonical.kalshi_mapping.is_open_wnba_moneyline_market` before mapping, scheduling, or execution. Finalized/closed markets, spreads, totals, player props, season futures, and any other non-moneyline markets are forbidden in the live path.

Due `market_t20` jobs record Kalshi side/complement/settlement mapping as a preflight audit check, but snapshot/mapping issues must not block the data refresh itself. Prefer the daemon's cached snapshot; if mapping is unavailable, write the issue into the run packet. Execution remains blocked until mapping is confirmed.

The live daemon health surface is `data/runs/live_daemon/health_latest.json`. It must expose `ok`/`degraded`/`failed`, current checks, failure counters, market freshness, refresh-worker freshness, execution-supervisor freshness, and active/open market counts. Do not let diagnostic `--ignore-lock` runs overwrite global health or heartbeat files.

## 4. Validation Reference Numbers

These are regression-test truth. Moving them is a regression unless explicitly authorized.

2025 holdout:
- Elo-only log loss: 0.6151
- Elo-only accuracy: 66.8%
- XGBoost + Elo log loss: 0.6121
- XGBoost + Elo accuracy: 67.4%
- Games after cold-start filter: 310
- Best round on 2015-2024 split: 88

Trading 2025 locked config:
- Trades: 134
- Hit rate: 40.3%
- Mean edge: 13.4%
- Ideal return: +1062%
- Sweep execution at $5K bankroll: +765%

Forecasting reproducer:

```bash
python -c "
import sys; sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from sklearn.metrics import log_loss
from srwnba.util.final_model import FinalModel, _cold_start_mask
m = FinalModel('data/gold/game_xgboost_input_2015_2024_REGPST.csv')
df = pd.read_csv('data/gold/game_xgboost_input_2025_REGPST.csv')
df = df[_cold_start_mask(df)].reset_index(drop=True)
res = m.predict(df); y = df['home_win'].values.astype(float)
print('log loss:', round(log_loss(y, np.clip(res['p_raw'], 1e-7, 1-1e-7)), 4))
print('accuracy:', round(((np.array(res['p_raw'])>0.5)==y).mean(), 4))
"
```

Expected:
- log loss: 0.6121
- accuracy: 0.6742

## 5. Production Pipeline

Run from repo root with `PYTHONPATH=src` in the `kalshi-wnba` environment.

Daily refresh:

```bash
python pipelines/07_live/08_append_year.py --year 2026
python pipelines/07_live/11_extend_elo_to_year.py --year 2026 --force
python pipelines/07_live/08_append_year.py --year 2026 --from-phase gold
python pipelines/07_live/09_combine_gold.py --start-year 2015 --end-year 2026 --force
python pipelines/07_live/13_validate_production_artifacts.py --year 2026
```

Canonical live route loop:

```bash
python pipelines/07_live/canonical/05_run_route_entry_loop.py \
    --game-id <sportradar_game_id> \
    --scheduled <ISO_scheduled_timestamp> \
    --home-team-id <sportradar_home_team_id> \
    --away-team-id <sportradar_away_team_id> \
    --home-team-name "<home name>" \
    --away-team-name "<away name>" \
    --tipoff-ts <unix_seconds> \
    --feature-csv data/live_features/<game>.csv \
    --completed-games-csv data/silver/played_games_2026_REGPST.csv \
    --ledger-dir data/runs/live_games/<sportradar_game_id> \
    --dry-run
```

Canonical execution must resolve portfolio sizing from Kalshi before planning and refresh it during the run, defaulting to a 300-second refresh interval. Canonical execution must reconcile Kalshi fills/positions on startup so restarts seed existing route exposure rather than double-sizing it. It must also discover open route orders: bot-owned open orders are cancelled before planning; unknown open orders block execution.

Local operator controls live under `data/runs/live_control/`. Missing files mean all eligible games trade normally. Global auto-trade off, risk mode `kill`, or a per-game `abort` file blocks new execution. Remote Supabase controls are available through explicit `--control-plane-mode` settings: `local-only`, `supabase-shadow`, and `supabase-live`. In Supabase modes the worker reads remote controls before planning and before live order submission; local JSON controls remain the final emergency brake. The webapp must display the execution default/risk mode plus worker control acknowledgment and expose global default plus per-game abort/clear controls.

Canonical execution must write the structured per-game ledger at `data/runs/live_games/<game_id>/` unless a narrow diagnostic explicitly passes `--no-ledger`. Required review files are `prediction_packet.json` with model best round/source, `market_mapping.json`, raw `market_snapshots.jsonl`, evaluated `route_quotes.jsonl`, `portfolio_sizing.jsonl`, `execution_plans.jsonl`, `orders.jsonl`, `fills.jsonl`, `positions.jsonl`, `errors.jsonl`, and `summary.json`. `data/live_logs/<game_id>.route.jsonl` is the flat raw stream; the per-game ledger is the production audit source for prediction, market, and trade history.

For live-run games that later settle, the historical gold row should prefer the captured live prediction packet/feature row whose source as-of time is no later than T-8. This keeps the settled training database aligned to the latest conditions available before the locked no-new-entry boundary, with the older strict historical silver_plus rebuild only as fallback when no valid live packet exists.

## 6. Key Files

- `src/srwnba/util/model_schema.py`: canonical 160-feature schema and gold column order.
- `src/srwnba/util/final_model.py`: production probability source.
- `pipelines/03_features/21_build_player_state_history_year.py`: player state and prior carryover.
- `pipelines/03_features/22_build_game_team_player_year.py`: full player/team/game feature store.
- `pipelines/04_gold/30_build_game_xgboost_input.py`: strict 7-slot model-input projection.
- `pipelines/07_live/09_combine_gold.py`: all-settled combined training CSV builder.
- `pipelines/07_live/13_validate_production_artifacts.py`: hard production artifact checks.
- `pipelines/07_live/16_execution_supervisor.py`: one-process-per-game launcher and route-loop health registry.
- `src/srwnba/storage/bronze.py`: staged bronze writer.
- `src/srwnba/live/canonical/portfolio.py`: Kalshi wealth/cash resolver plus imaginary sizing/cash overrides.
- `src/srwnba/live/canonical/operator_control.py`: trade-default, risk-mode, and per-game abort resolver.
- `src/srwnba/live/canonical/reconciliation.py`: Kalshi fill/position/open-order reconciliation.
- `src/srwnba/live/canonical/v1_2.py`: locked v1.2 planner.
- `src/srwnba/live/canonical/game_ledger.py`: per-game prediction/market/order/fill audit packet writer.
- `src/srwnba/live/canonical/route_entry_loop.py`: production route runtime.
- `src/srwnba/live/legacy/`: legacy YES-only path retained only for comparison/testing.

## 7. Runtime Invariants

1. Instantiate `FinalModel` once per game process.
2. Use `p_raw`/`p_home`; do not add calibration.
3. Never sell/reduce before settlement.
4. Do not exceed fee-adjusted half-Kelly and v1.2 liquidity/cash caps.
5. Use unique `client_order_id` values.
6. Train production probabilities on the current all-settled combined gold file.
7. Keep chronological facts and daily player state in silver, full game-as-of feature-family tables in silver_plus, and strict 7-slot/160-feature model input in gold.
8. Stage and validate bronze before canonical promotion.
9. Do not prefetch or reuse future injury date files.
10. Hyperparameters come only from `config/final_hyperparams.py`.
11. Preserve the canonical per-game ledger under `data/runs/live_games/<game_id>/`; do not replace it with only flat `data/live_logs` output.
12. Do not default live production sizing to hardcoded `$5,000`; use Kalshi portfolio value/balance unless an explicit sizing override is provided and logged.
13. Missing operator-control files mean trade-by-default. Only global auto-trade off, risk mode `kill`, or per-game `abort` may block operator permission.
14. Startup reconciliation is mandatory before live planning; do not allow restarts to duplicate exposure.
15. `16_execution_supervisor.py` must avoid duplicate active route-loop processes for the same game.

## 8. Research Boundary

The codebase is now a runtime. Research notebooks, tuning scripts, Polymarket code/data, and rejected approaches live outside this production repo. Do not reintroduce them to override production behavior. Bootstrap ensembles, Platt scaling, early exits, in-play modeling, Polymarket execution, and alternate entry mechanics are out of scope unless explicitly unlocked.
