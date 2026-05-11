# Production Build Log

Purpose: durable audit trail for production-readiness work. This log records material changes, safety posture, verification commands, and known gaps. Do not store secrets here.

## 2026-05-10 - Kalshi Account Link And Safety Latch

- Confirmed `.env` had Sportradar keys but no Kalshi credentials.
- Updated `utils/kalshi_authed_client.py` for current Kalshi prod/demo hosts and `/trade-api/v2` signing path behavior.
- Added `.env.example` Kalshi placeholders.
- Added `pipelines/07_live/00_check_kalshi_auth.py` as a read-only auth/balance smoke test.
- Resolved Windows hidden extension issue: private key file was `kalshi-demo-private-key.pem.txt`; renamed to `.pem`.
- Confirmed prod read-only Kalshi auth works:
  - base URL: `https://external-api.kalshi.com/trade-api/v2`
  - key path: `C:\Users\arius\.kalshi\kalshi-read-private-key.pem`
  - balance check succeeded
  - no private key material printed or logged
- Added `KALSHI_TRADING_ENABLED` safety latch:
  - default false
  - every authenticated `POST`, `PUT`, `PATCH`, and `DELETE` is blocked unless env value is exactly `true`
  - live entry loop also fails fast when non-dry-run is requested while latch is false
- Added `pipelines/07_live/01_smoke_kalshi_market_data.py` as a read-only WNBA market/orderbook smoke test.
- Verification:
  - `python pipelines\07_live\03_trader_unit_test.py` passed, 10 tests.
  - `python pipelines\07_live\04_entry_loop_dryrun.py` passed.
  - `python pipelines\07_live\00_check_kalshi_auth.py` passed with `trading_enabled: False`.
  - Dummy `create_order` was blocked locally before a network call.
  - `python pipelines\07_live\01_smoke_kalshi_market_data.py` discovered 8 active WNBA markets under `KXWNBAGAME` and fetched one orderbook read-only.

Known gaps after this checkpoint:
- Live entry loop was still a simple buy-YES-per-team loop and did not yet implement v1.2 selected-team canonical route mapping.
- No production market mapper yet connected Sportradar game/team IDs to Kalshi market tickers through confirmed side/complement checks.
- No v1.2 execution router yet accounted for BUY YES selected vs BUY NO opponent route alternatives.

## 2026-05-10 - Market Mapping And Execution Router Foundation

- Started production mapping layer work.
- Finding: stored 2025 Kalshi `custom_strike.basketball_team` values do not match the Sportradar team UUIDs used by this pipeline. UUID-only mapping would be unsafe.
- Design decision: use explicit audited `data/config/kalshi_team_name_map.csv` as the production side-mapping key when available, keyed by Kalshi display team name to Sportradar team UUID. Retain `custom_strike.basketball_team` as a diagnostic field and future cross-check, not as the primary key unless no trusted name map entry exists.
- Safety constraint: implementation must remain read-only unless `KALSHI_TRADING_ENABLED=true`; all new route/execution work initially plans orders only and does not submit them.
- Added `data/config/kalshi_team_name_map.csv` with audited 2025-2026 Kalshi display names to Sportradar team UUIDs for current known WNBA teams.
- Added `src/srwnba/live/kalshi_mapping.py`:
  - maps one Sportradar game to a two-market Kalshi event
  - requires exact expected team set
  - supports date slop of one day for Kalshi local-date/UTC-date differences
  - returns diagnostics instead of silently guessing
  - builds equivalent routes: `BUY_YES_SELECTED` and `BUY_NO_OPPONENT`
- Added `src/srwnba/live/execution.py`:
  - converts Kalshi bid-only books into executable route ask books for BUY YES and BUY NO
  - computes fee-adjusted all-in executable average price
  - enforces locked absolute and normalized edge checks at all-in price
  - computes initial half-Kelly / 15% market-cap / cash-buffer target
  - applies visible-depth participation and normal IOC child cap
  - returns planned child orders only; no network writes
  - converts planned child orders into exact Kalshi `create_order` kwargs, including BUY NO via `no_price_cents`
  - includes a submission bridge that still relies on the real client's `KALSHI_TRADING_ENABLED` latch before any write can leave the process
- Added `pipelines/07_live/02_mapping_execution_unit_test.py`.
- Added `pipelines/07_live/02_smoke_kalshi_routes.py` as a read-only live route-planning smoke CLI for one Sportradar game.
- Updated README and CLAUDE operational pipeline lists with the new mapping/execution checks.
- Verification:
  - `python -m py_compile src\srwnba\live\kalshi_mapping.py src\srwnba\live\execution.py src\srwnba\live\__init__.py pipelines\07_live\02_mapping_execution_unit_test.py` passed.
  - `python pipelines\07_live\02_mapping_execution_unit_test.py` passed, including BUY NO order-kwargs bridge checks.
  - `python -m py_compile pipelines\07_live\02_smoke_kalshi_routes.py` passed.
  - `python pipelines\07_live\03_trader_unit_test.py` passed, 10 tests.
  - `python pipelines\07_live\04_entry_loop_dryrun.py` passed.
  - `python pipelines\07_live\00_check_kalshi_auth.py` passed with `trading_enabled: False`.
  - `python pipelines\07_live\01_smoke_kalshi_market_data.py` passed read-only, discovered 8 active WNBA markets, and fetched one orderbook.
  - Dummy `create_order` guard check passed: blocked locally with `POST /portfolio/orders blocked`.
  - `git diff --check` passed for changed production files.

Known gaps after this checkpoint:
- Current execution router slice does not yet implement recent-trade volume caps, global cumulative volume cap, passive probes, burst mode, signal-memory/timing gates, expansion-team gate, or full v1.2 log schema.
- Current live entry loop has not yet been rewired to consume the new canonical route planner; the new planner is available through pure functions and the read-only route smoke CLI.

## 2026-05-10 - Canonical Route Entry Loop Integration

- Added `src/srwnba/live/route_entry_loop.py`, a production-shaped entry loop for one canonical exposure:
  - scores `p_home` once through `FinalModel`
  - selects the higher-probability side as `selected_team_wins`
  - confirms Kalshi game mapping through `kalshi_mapping`
  - builds `BUY_YES_SELECTED` and `BUY_NO_OPPONENT` routes
  - polls both route orderbooks
  - computes route quotes through `execution.evaluate_route_quote`
  - plans normal IOC child orders through `execution.plan_normal_ioc_orders`
  - tracks canonical filled cost across both routes
  - writes JSONL audit events: loop start, mapping, route candidates, route quotes, execution plans, dry orders, order submissions, fills, errors, and tipoff stop
- Added `pipelines/07_live/05_run_route_entry_loop.py`:
  - production-shaped CLI for the route loop
  - requires `--dry-run` unless `KALSHI_TRADING_ENABLED=true`
  - uses active Kalshi credentials for market discovery/orderbooks in dry-run mode
- Added `pipelines/07_live/04_route_entry_loop_dryrun.py`:
  - no-network fake-client end-to-end dryrun for the canonical route loop
  - exercises both `BUY_YES_SELECTED` and `BUY_NO_OPPONENT`
  - verifies parseable JSONL audit log and canonical exposure accounting
- Exported route-loop classes through `src/srwnba/live/__init__.py`.
- Updated README and CLAUDE operational pipeline references.
- Verification:
  - `python -m py_compile src\srwnba\live\route_entry_loop.py pipelines\07_live\05_run_route_entry_loop.py pipelines\07_live\04_route_entry_loop_dryrun.py` passed.
  - `python pipelines\07_live\04_route_entry_loop_dryrun.py` passed:
    - selected Atlanta from a 2025 holdout row at `p_selected=0.6203`
    - confirmed fake Kalshi event `KXWNBAGAME-25MAY16ROUTE`
    - filled $60 total across both equivalent routes
    - submitted one fake BUY YES child and one fake BUY NO child
    - audit log contained mapping, route candidates, route quotes, execution plans, submissions, fills, and tipoff stop
  - `python pipelines\07_live\02_smoke_kalshi_routes.py ...PHX/GS...` passed read-only against active Kalshi markets:
    - confirmed event `KXWNBAGAME-26MAY10PHXGS`
    - built both Phoenix selected-team routes
    - both routes rejected because fee-adjusted all-in executable price was above `q_max`
    - no orders submitted
  - Dummy `create_order` guard check passed again with `KALSHI_TRADING_ENABLED=false`.
- Finding: Kalshi list-market objects report status `active`, but the `/markets` list endpoint rejects `status=active` as a query filter. Route discovery now omits the status filter and filters returned market objects locally to `active/open`.

Known gaps after this checkpoint:
- The new route loop implements normal IOC only. It still does not implement v1.2 passive probes, burst mode, recent-trade volume caps, global cumulative qualifying-volume cap, signal-memory timing gates, expansion-team gate, operational brakes, or the full required v1.2 logging schema.
- The old `05_run_entry_loop.py` remains available for legacy/simple dryruns, but production work should move toward `05_run_route_entry_loop.py`.

## 2026-05-10 - Expansion Team Mapping And Gate Patch

- User caught that the audited Kalshi side map omitted the two 2026 expansion teams.
- Added 2026 expansion-team mappings to `data/config/kalshi_team_name_map.csv`.
- Correction after user review: the CSV must stay one row per canonical Sportradar team ID. Alias rows double-counted Toronto and Portland and could silently affect inverse `team_id -> name` consumers. The fixed CSV uses a `kalshi_aliases` column instead:
  - canonical `Toronto` with alias `Toronto Tempo` -> `4e4f726e-a015-4306-91a7-28e8576c7868`
  - canonical `Portland` with alias `Portland Fire` -> `d54283cc-c5ec-4dbd-bb61-166f217e3864`
- Evidence checked:
  - local `data/silver/played_games_2026_REGPST.csv` contains Toronto Tempo with the same Sportradar team ID
  - local 2026 injury/schedule data contains Toronto Tempo and Portland Fire with the IDs above
  - official WNBA expansion-draft materials identify Toronto Tempo and Portland Fire as the 2026 expansion teams: https://www.wnba.com/webview/news/wnba-expansion-draft-2026-results
- Added `src/srwnba/live/expansion_gate.py`:
  - pure trading-layer gate
  - forecasts/state updates are not blocked
  - live trading is blocked for any game involving a true first-season expansion team until every such team in the game has at least 14 completed prior games
  - missing completed-game counts default to zero and therefore block trading
- Wired the expansion gate into `src/srwnba/live/route_entry_loop.py`:
  - `RouteEntryContext` now carries `completed_games_by_team`
  - loop-start and dedicated `expansion_gate` events log the gate outcome
  - route quotes are still logged for diagnostics
  - order planning is blocked with an `execution_plan` no-trade event when the gate fails
- Updated `pipelines/07_live/05_run_route_entry_loop.py`:
  - optional `--completed-games-csv`
  - defaults to `data/silver/played_games_<scheduled_year>_REGPST.csv` when present
  - computes completed prior games chronologically before the target game's scheduled timestamp
- Updated README and CLAUDE with the expansion-team IDs, aliases, mandatory gate, and component references.
- Updated `load_team_name_map()` so aliases expand in memory while duplicate/ambiguous names are rejected.
- Added a regression check that `kalshi_team_name_map.csv` has exactly one row per 2026 WNBA team and no duplicate Sportradar team IDs.

Verification:
- `python -m py_compile src\srwnba\live\kalshi_mapping.py src\srwnba\live\expansion_gate.py src\srwnba\live\route_entry_loop.py src\srwnba\live\__init__.py pipelines\07_live\02_mapping_execution_unit_test.py pipelines\07_live\04_route_entry_loop_dryrun.py pipelines\07_live\05_run_route_entry_loop.py` passed.
- `python pipelines\07_live\02_mapping_execution_unit_test.py` passed, including:
  - no duplicate Sportradar team IDs in `kalshi_team_name_map.csv`
  - 15 canonical 2026 WNBA team rows
  - Toronto/Portland city and full-name alias expansion through `load_team_name_map()`
  - expansion gate pass/block cases
  - route-loop no-trade block for an under-gate Toronto game
- `python pipelines\07_live\04_route_entry_loop_dryrun.py` passed; non-expansion route execution path still submits/fills fake BUY YES and BUY NO child orders.
- `python pipelines\07_live\03_trader_unit_test.py` passed, 10 tests.
- `git diff --check -- ...` passed for changed files.

## 2026-05-10 - Live Loop Folder Separation

- Physically separated the two live execution paths to prevent legacy/canonical confusion.
- New canonical locations:
  - `src/srwnba/live/canonical/`
  - `pipelines/07_live/canonical/`
- New legacy locations:
  - `src/srwnba/live/legacy/`
  - `pipelines/07_live/legacy/`
- Added `src/srwnba/live/common.py` for neutral shared Kalshi orderbook/fee primitives so canonical execution no longer imports helpers from the legacy planner.
- Added folder-level READMEs:
  - `src/srwnba/live/README.md`
  - `pipelines/07_live/README.md`
- Updated live imports, script `REPO_ROOT` resolution, README, CLAUDE, AGENTS, and memory notes to point at canonical/legacy folders.

Verification:
- `python -m py_compile src\srwnba\live\common.py src\srwnba\live\__init__.py src\srwnba\live\canonical\__init__.py src\srwnba\live\canonical\kalshi_mapping.py src\srwnba\live\canonical\execution.py src\srwnba\live\canonical\expansion_gate.py src\srwnba\live\canonical\route_entry_loop.py src\srwnba\live\legacy\__init__.py src\srwnba\live\legacy\trader.py src\srwnba\live\legacy\entry_loop.py pipelines\07_live\canonical\02_mapping_execution_unit_test.py pipelines\07_live\canonical\02_smoke_kalshi_routes.py pipelines\07_live\canonical\04_route_entry_loop_dryrun.py pipelines\07_live\canonical\05_run_route_entry_loop.py pipelines\07_live\legacy\03_trader_unit_test.py pipelines\07_live\legacy\04_entry_loop_dryrun.py pipelines\07_live\legacy\05_run_entry_loop.py pipelines\07_live\01_smoke_kalshi_market_data.py pipelines\07_live\06_passive_book_recorder.py` passed.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python pipelines\07_live\legacy\03_trader_unit_test.py` passed, 10 tests.
- `python pipelines\07_live\canonical\04_route_entry_loop_dryrun.py` passed.
- `python pipelines\07_live\legacy\04_entry_loop_dryrun.py` passed.

## 2026-05-10 - Remaining v1.2 Execution Logic

- Added the locked v1.2 side-effect-free planner in `src/srwnba/live/canonical/v1_2.py`:
  - T-24/T-17/T-12/T-8/T-4 timing windows and locked poll cadence
  - signal memory, early-stable vs late-only classification, and late-only rejection
  - route-level visible-depth, recent/cumulative qualifying-volume, cold-start, and global cumulative caps
  - normal IOC, completion IOC, burst IOC, and passive-probe order modes
  - burst depth-refresh trigger, first-poll burst suppression, burst debounce, and 5-minute burst notional cap
  - operational brakes for repeated order rejects, API errors, and position mismatch
- Extended canonical execution dataclasses in `src/srwnba/live/canonical/execution.py`:
  - route quotes now carry bid/ask, spread, route liquidity cap diagnostics, and cold-start/cumulative fields
  - child orders now carry `order_mode`, `time_in_force`, `post_only`, and optional `expiration_ts`
  - execution plans now log timing window, lead hours, signal class, binding cap, route capacity sum, and global cumulative remaining
- Wired the canonical route loop to the v1.2 planner in `src/srwnba/live/canonical/route_entry_loop.py`:
  - updates signal memory every poll
  - writes `signal_state` events
  - plans through `plan_v1_2_orders()` instead of the old normal-IOC-only helper
  - pulls Kalshi `/markets/trades` per route when the client supports it, computes recent and cumulative qualifying-price volume, and feeds those into `VolumeSnapshot`
  - writes `trade_volume_snapshot` and enriched `route_capacity` events so final cap binders can be audited separately from raw top-of-book quotes
  - reserves passive order exposure, cancels passives at T-8, cancels/reprices passive probes when qmax/spread changes or timeout fires, and reconciles passive fills from order snapshots/cancel responses
  - gives IOC priority over live passives by canceling the passive first and preserving reserved exposure if cancellation fails
  - tracks IOC/burst timestamps, burst 5-minute notional usage, order rejects, API errors, filled cost by route, and reserved exposure
- Updated `utils/kalshi_authed_client.py` order submission support:
  - added `get_trades()` for paginated `/markets/trades` reads used by v1.2 recent/cumulative volume caps
  - `create_order()` accepts and forwards `time_in_force`, `expiration_ts`, `post_only`, `cancel_order_on_pause`, and `self_trade_prevention_type`
  - field names were checked against Kalshi's Create Order docs on 2026-05-10
- Updated `pipelines/07_live/canonical/05_run_route_entry_loop.py`:
  - default `--poll-interval-s 0` now means use the locked v1.2 timing-window cadence
  - positive values cap cadence for diagnostics
  - negative values disable sleeping for local tests only
- Updated `pipelines/07_live/canonical/04_route_entry_loop_dryrun.py`:
  - fake clock now exercises the T-17 to T-8 execution window, not the post-T-4 hard-stop window
  - fake orderbook depth is large enough to pass v1.2 cold-start/visible-depth caps
- Updated README/CLAUDE component references so `v1_2.py` is explicitly documented as the locked planner and `route_entry_loop.py` is documented as the wiring/runtime layer.
- Extended `pipelines/07_live/canonical/02_mapping_execution_unit_test.py`:
  - v1.2 timing and signal-memory order planning
  - late-only rejection
  - first-poll normal IOC vs later refreshed-depth burst IOC
  - order-reject brake enforcement

Verification:
- `python -m py_compile src\srwnba\live\canonical\execution.py src\srwnba\live\canonical\v1_2.py src\srwnba\live\canonical\route_entry_loop.py pipelines\07_live\canonical\02_mapping_execution_unit_test.py pipelines\07_live\canonical\04_route_entry_loop_dryrun.py pipelines\07_live\canonical\05_run_route_entry_loop.py` passed.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python pipelines\07_live\canonical\04_route_entry_loop_dryrun.py` passed:
  - selected Atlanta from a 2025 holdout row at `p_selected=0.6203`
  - submitted one fake canonical route order
  - filled 125 contracts at 40c for `$50.00`
  - audit log contained route loop start, mapping, route candidates, expansion gate, route quotes, signal states, route capacity snapshots, execution plans, order submission, fill, and tipoff stop
- `python pipelines\07_live\legacy\03_trader_unit_test.py` passed, 10 tests.
- `python pipelines\07_live\legacy\04_entry_loop_dryrun.py` passed.

## 2026-05-10 - Production Data Layer Contract

- Locked the production data-layer architecture:
  - `bronze` = immutable accepted per-API-call payloads
  - `bronze_runs` = staged pull cache and manifests before promotion
  - `silver` = canonical chronological facts at natural grain plus daily/as-of state
  - `silver_plus` = game-as-of feature-family tables
  - `gold` = strict model-input projection
  - `runs` = per-refresh / T-20 audit packets
- Moved daily `player_state_history_2015.csv` through `player_state_history_2025.csv` from `data/silver_plus/` to `data/silver/`.
- Moved `game_team_player_2026_REGPST.csv` from `data/silver/` to `data/silver_plus/`.
- Updated `pipelines/03_features/22_build_game_team_player_year.py` so the full all-listed-player game/team/player feature store writes to `data/silver_plus/`, and archives any legacy silver copy.
- Updated `pipelines/04_gold/30_build_game_xgboost_input.py` so gold consumes `game_team_player` from `data/silver_plus/`.
- Removed fallback reads from `data/silver_plus/player_state_history_*`; daily player state now belongs in `data/silver/`.
- Extended `pipelines/07_live/13_validate_production_artifacts.py` to enforce the layer split:
  - no `game_team_player` in silver
  - no `player_state_history` in silver_plus
  - full game-wise player store exists in silver_plus
  - daily player state exists in silver
  - gold remains exact 174-column metadata + 160-feature schema
- Added `docs/production_data_layers.md`.
- Updated README, CLAUDE, AGENTS, data README, and spec sheets for the layer contract.
- Removed production-repo research clutter:
  - notebooks
  - `pipelines/05_modeling`
  - `pipelines/06_markets`
  - `pipelines/04_gold/31_build_gold_variant.py`
  - Polymarket utils/spec/data
  - stale saved live artifacts and Platt artifact
  - model-comparison/trading-result outputs
  - tuning methodology doc and old changelog
  - quarantined future injury placeholders
  - generated `__pycache__` directories
- Rebuilt 2026 game-wise player store, 2026 gold, and combined 2015-2026 gold training file.

Verification:
- `python pipelines\03_features\22_build_game_team_player_year.py --year 2026` passed and wrote `data\silver_plus\game_team_player_2026_REGPST.csv`.
- `python pipelines\07_live\12_build_gold_year.py --year 2026` passed and wrote 5 rows x 174 columns.
- `python pipelines\07_live\09_combine_gold.py --start-year 2015 --end-year 2026 --force` passed and wrote 2537 rows x 174 columns.
- `python pipelines\07_live\13_validate_production_artifacts.py --year 2026 --today 2026-05-10` passed.
- Forecasting regression passed: best round 88, log loss 0.6121, accuracy 0.6742.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python pipelines\07_live\canonical\04_route_entry_loop_dryrun.py` passed.
- `python pipelines\07_live\legacy\03_trader_unit_test.py` passed, 10 tests.
- `python pipelines\07_live\legacy\04_entry_loop_dryrun.py` passed.
- `python -m py_compile ...` passed for changed production modules.
- `git diff --check` passed.

## 2026-05-10 - Live Pull Verification and Freshness Fixes

- Ran a live 2026 ingest against Sportradar.
- Pulled fresh schedules:
  - REG latest: 331 scheduled games, 7 closed games.
  - PST latest: 0 games.
- Pulled the two newly closed game summaries:
  - Atlanta Dream at Minnesota Lynx.
  - Chicago Sky at Portland Fire.
- Found and fixed season-start schedule context nulls:
  - Away teams with no prior home summary had null `origin_city_pre`, `travel_miles_pre`, and timezone shift.
  - `pipelines/03_features/29_build_game_team_schedule_context.py` now uses team-market city fallback when no home venue has been observed yet, while still preferring real home venue data when available.
  - Added market aliases for Golden State, Indiana, and Minnesota.
- Found and fixed false opener back-to-backs:
  - Season openers keep `days_rest_pre = 0`, but now set `is_b2b_pre = 0` when no previous game exists.
  - Updated `data/spec_sheets/game_team_schedule_context_spec.md` to make the previous-game condition explicit.
- Removed stale fixed-franchise-count warnings from franchise style/Elo builders; 2026 expansion-era counts should reflect teams that have actually played.
- Found and fixed gold metadata nulls:
  - `is_playoff` was null because gold tried to source it from Elo rows.
  - `pipelines/04_gold/30_build_game_xgboost_input.py` now derives `is_playoff` from `game_outcomes_{year}_REGPST.csv`.
  - `pipelines/07_live/13_validate_production_artifacts.py` now fails on gold metadata nulls, feature nulls, and invalid `p_elo`.
- Found and fixed injury freshness behavior:
  - Daily injury ingest was date-idempotent and skipped May 9/May 10 even though recent injury reports can change intraday.
  - `pipelines/01_ingestion/10_backfill_daily_injuries_year.py` now force-refreshes a recent lookback window, defaulting to today plus yesterday, while keeping old-date backfill idempotent.
  - Updated README, CLAUDE, and AGENTS to document the recent injury force-refresh rule.

Final artifact state after rebuild:
- Bronze game summaries: 7.
- Bronze daily injury files: 12, with no future injury placeholders.
- Silver outcomes: 7 games.
- Silver injury events: 22 rows across 7 non-empty report days.
- Silver availability/player box: 187 rows, no games missing players.
- Silver player state: 748 rows, 187 players, 4 as-of timestamps.
- Silver_plus game player store: 187 rows.
- Silver_plus Elo/recent/style/schedule: 14 team-game rows each.
- Gold 2026: 7 rows x 174 columns, 160 model features, zero null cells.
- Combined all-settled gold 2015-2026: 2539 rows x 174 columns.
- Kalshi mapping: all 14 played teams mapped; no duplicate Sportradar team IDs.
- Schedule sanity: no travel/timezone nulls, opener B2B count 0, travel miles range 0.0 to 1753.9.

Verification:
- `python pipelines\07_live\08_append_year.py --year 2026 --to-phase ingest` passed after force-refresh change; daily injuries re-pulled May 9 and May 10.
- `python pipelines\07_live\08_append_year.py --year 2026 --from-phase parse --to-phase feature` passed.
- `python pipelines\07_live\11_extend_elo_to_year.py --year 2026 --force` passed.
- `python pipelines\07_live\12_build_gold_year.py --year 2026` passed.
- `python pipelines\07_live\13_validate_production_artifacts.py --year 2026 --today 2026-05-10` passed.
- `python pipelines\07_live\09_combine_gold.py --start-year 2015 --end-year 2026 --force` passed.
- 2026 rows are consumable by `FinalModel` on `game_xgboost_input_2015_2026_REGPST.csv`; predicted home probabilities ranged from 0.4696 to 0.6725.
- Forecasting regression still matches: best round 88, 2025 log loss 0.6121, accuracy 0.6742.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python -m compileall ...` passed for changed production modules.

## 2026-05-10 - Live Refresh Scheduler and Audit Layer

- Added richer per-pull bronze manifests in `src/srwnba/storage/bronze.py`:
  - source
  - endpoint
  - HTTP method
  - sanitized request URL
  - sanitized request params
  - payload byte size
  - SHA-256 hash
  - staged payload path
  - promoted canonical path
  - validation result
- Updated production Sportradar ingestion scripts to populate the new manifest fields:
  - `pipelines/01_ingestion/00_backfill_schedule_year.py`
  - `pipelines/01_ingestion/12_backfill_game_summaries_year.py`
  - `pipelines/01_ingestion/10_backfill_daily_injuries_year.py`
- Added `pipelines/07_live/14_live_data_refresh.py` as the production live refresh scheduler/runner:
  - fixed daily settled-history jobs at 02:30 ET and 09:00 ET
  - market/game-time T-20 due-job detection from the latest Sportradar schedule
  - optional Kalshi active-market snapshot cache under each run packet
  - duplicate-run state in `data/runs/live_refresh/scheduler_state.json`
  - command logs for every subprocess
  - bronze file and bronze-run deltas
  - validation report
  - promoted-output manifest
  - JSONL run log
  - Kalshi snapshot failures are logged into the run packet but do not block a due Sportradar settled-history refresh.
- Updated README, CLAUDE, AGENTS, data README, and the live pipeline README so scheduled production refreshes go through `14_live_data_refresh.py`.
- Created active app automations:
  - `wnba-02-30-settled-refresh`
  - `wnba-09-00-safety-refresh`
  - `wnba-hourly-t-20-scheduler`

Live verification:
- Ran `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due`.
- Run packet: `data/runs/live_refresh/20260510T193913Z_due_09dc5373`.
- The scheduler pulled and cached a Kalshi market snapshot:
  - `KXWNBAGAME`: 44 markets
  - `KXWNBAH`: 0 markets
  - trading writes disabled
- Sportradar bronze deltas from that live run:
  - fresh REG schedule
  - fresh PST schedule
  - newly closed Seattle Storm at Connecticut Sun game summary
  - refreshed May 9 daily injuries
  - refreshed May 10 daily injuries
- The run rebuilt silver, silver_plus, gold, validated production artifacts, and recombined all-settled training.
- Scheduler state now marks the 2026-05-10 02:30 and 09:00 settled-history jobs complete, preventing duplicate reruns.

Final artifact state:
- 2026 played/outcomes/gold rows: 8 games.
- 2026 injury events: 24 rows.
- 2026 game player store: 213 rows.
- 2026 schedule context: 16 team-game rows.
- 2026 gold: 8 rows x 174 columns, zero null cells.
- Current all-settled model can consume 2026 gold; p_home range on the 8 2026 rows was 0.3946 to 0.6764.

Verification:
- `python -m compileall src\srwnba\storage\bronze.py pipelines\01_ingestion\00_backfill_schedule_year.py pipelines\01_ingestion\10_backfill_daily_injuries_year.py pipelines\01_ingestion\12_backfill_game_summaries_year.py pipelines\07_live\14_live_data_refresh.py` passed.
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due --dry-run --skip-market-api --now 2026-05-10T10:05:00-04:00` planned the 02:30 and 09:00 jobs.
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode market --dry-run --skip-market-api --now 2026-05-12T04:15:00+00:00` planned the Atlanta at Dallas T-20 refresh.
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due --dry-run --skip-market-api --now 2026-05-10T15:45:00-04:00` found zero due jobs after state was marked by the live run.
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due --dry-run --skip-market-api --now 2026-05-11T02:35:00-04:00` planned the next day's 02:30 settled refresh only.
- `python pipelines\07_live\13_validate_production_artifacts.py --year 2026 --today 2026-05-10` passed.
- Forecasting regression still matches: best round 88, 2025 log loss 0.6121, accuracy 0.6742.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `git diff --check` passed.

## 2026-05-10 - Always-On Live Daemon

- Added `pipelines/07_live/15_live_daemon.py` as the production 24/7 coordinator:
  - polls Kalshi WNBA market series on a fixed cadence
  - caches every market-list API response under `data/runs/live_daemon/market_pulls/YYYYMMDD/`
  - tracks new, updated, and missing market tickers in `data/runs/live_daemon/daemon_state.json`
  - writes market-change packets for audit
  - writes heartbeat files to `data/runs/live_daemon/heartbeat_latest.json` and JSONL history
  - holds a lock file at `data/runs/live_daemon/live_daemon.lock` so a second daemon cannot silently double-run
  - invokes `pipelines/07_live/14_live_data_refresh.py --mode due` as the worker
  - passes the already-cached Kalshi market snapshot into the worker, avoiding duplicate market-list calls inside the same loop
  - wakes the worker immediately when Kalshi market state changes, even if the normal worker interval has not elapsed
  - never places orders; execution still belongs to the entry-loop layer
- Updated `pipelines/07_live/14_live_data_refresh.py` with `--market-snapshot-json` so the daemon can reuse its audited Kalshi snapshot.
- Added `scripts/install_live_daemon_task.ps1` to register the daemon as a Windows Task Scheduler job at user logon.
- Added `scripts/stop_live_daemon.ps1` to stop the daemon by lock-file PID and clean stale locks.
- Updated README, CLAUDE, AGENTS, data README, and the live pipeline README with the daemon contract and run paths.

Live verification:
- Dry no-network one-shot daemon passed with `--skip-market-api --worker-dry-run`; it wrote heartbeat/session logs and released its lock.
- Live one-shot daemon pulled Kalshi markets successfully:
  - `KXWNBAGAME`: 44 markets
  - `KXWNBAH`: 0 markets
  - detected 44 new markets on first state build
  - worker ran with the cached market snapshot and returned code 0
- Live market-change one-shot detected 6 updated markets and immediately woke the worker despite `--worker-check-s 300`.
- Started the persistent daemon as hidden background PID `27696`.
- First persistent heartbeat:
  - run id `20260510T195700Z_daemon_9bdcd281`
  - latest market count 44
  - tracked markets 44
  - detected 4 updated markets
  - worker return code 0
  - worker run `20260510T195702Z_due_823cae4b` had `jobs=0`, `refresh=False`, `success=True`

Operational note:
- Task Scheduler registration was blocked by Windows `Access is denied` in this session for both `Register-ScheduledTask` and `schtasks.exe`.
- The installer script is ready, but it must be run from an elevated PowerShell window:
  `powershell -ExecutionPolicy Bypass -File scripts\install_live_daemon_task.ps1`

Verification:
- `python -m compileall pipelines\07_live\14_live_data_refresh.py pipelines\07_live\15_live_daemon.py` passed.
- `python pipelines\07_live\15_live_daemon.py --year 2026 --once --skip-market-api --worker-dry-run --market-poll-s 1 --worker-check-s 1 --heartbeat-s 1` passed.
- `python pipelines\07_live\15_live_daemon.py --year 2026 --once --market-poll-s 1 --worker-check-s 1 --heartbeat-s 1` passed.
- `python pipelines\07_live\15_live_daemon.py --year 2026 --once --market-poll-s 1 --worker-check-s 300 --heartbeat-s 1` passed and confirmed market-change-triggered worker wakeup.

## 2026-05-10 - Moneyline-Only Market Guard

- Added `is_wnba_moneyline_market()` and `filter_wnba_moneyline_markets()` in `src/srwnba/live/canonical/kalshi_mapping.py`.
- The live path now requires all Kalshi markets to be WNBA moneyline/team-wins contracts before mapping, scheduling, state tracking, or execution:
  - WNBA series ticker (`KXWNBAGAME` or `KXWNBAH`)
  - binary market
  - title contains `winner`
  - rules say a team wins a basketball game and the market resolves to Yes
  - structured strike contains `basketball_team`
- Non-moneyline markets are filtered out of daemon snapshots and refresh snapshots before downstream use.
- Daemon and refresh snapshots now log `raw_market_count`, `moneyline_count`, `filtered_out_count`, and `moneyline_only`.
- When the refresh worker receives a daemon snapshot, it writes a filtered copy into its own run packet so `manifest.json` points at the exact moneyline-only snapshot used for scheduling.
- `RouteEntryLoop` now filters discovered markets through the same shared predicate.
- Added a second live-path status gate: snapshots used by daemon/refresh/entry only retain active/open WNBA moneyline markets. Finalized moneylines remain visible only as raw filtered-out counts.
- Fixed `15_live_daemon.py --ignore-lock` so one-shot smoke checks cannot overwrite/remove the live daemon's lock.
- Fixed daemon heartbeat semantics so `tracked_markets` means the latest active/open market set, while `tracked_market_history` preserves historical ticker state.
- Updated `scripts/stop_live_daemon.ps1` with a no-lock fallback for stopping daemon processes by command line.
- Updated README, CLAUDE, AGENTS, data README, and live pipeline README to make the moneyline-only rule explicit.

Live verification:
- Live one-shot daemon poll:
  - `KXWNBAGAME`: raw 44, moneyline 44, active/open moneyline 6, finalized moneyline filtered out 38
  - `KXWNBAH`: raw 0, moneyline 0, filtered out 0
  - snapshot `moneyline_only=true`, `open_markets_only=true`
- Restarted persistent daemon as hidden background PID `15824`.
- First restarted daemon heartbeat:
  - run id `20260510T200337Z_daemon_2e192de9`
  - latest market count 44
  - tracked markets 44
  - worker return code 0

Verification:
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed, including moneyline-filter acceptance/rejection.
- `python -m compileall src\srwnba\live\canonical\kalshi_mapping.py src\srwnba\live\canonical\route_entry_loop.py pipelines\07_live\14_live_data_refresh.py pipelines\07_live\15_live_daemon.py pipelines\07_live\canonical\02_mapping_execution_unit_test.py pipelines\07_live\canonical\04_route_entry_loop_dryrun.py` passed.
- `python pipelines\07_live\15_live_daemon.py --year 2026 --once --skip-market-api --worker-dry-run --market-poll-s 1 --worker-check-s 1 --heartbeat-s 1 --ignore-lock` passed.
- `python pipelines\07_live\15_live_daemon.py --year 2026 --once --market-poll-s 1 --worker-check-s 300 --heartbeat-s 1 --ignore-lock --worker-dry-run` passed.
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due --dry-run --market-snapshot-json <latest_daemon_snapshot>` passed and wrote a moneyline-only snapshot into the refresh run packet.

## 2026-05-10 - Live Refresh Uses Daemon Kalshi Snapshot First

- Corrected the scheduler behavior: Kalshi market preflight is an audit check for `market_t20` jobs, not a blocker for the data refresh itself.
- `pipelines/07_live/14_live_data_refresh.py` now tries to load the daemon's latest cached Kalshi snapshot before making a direct Kalshi API call.
- If the daemon snapshot is fresh, manual `14_live_data_refresh.py --mode due` runs avoid opening a second outbound Kalshi socket.
- If direct Kalshi API fails and a daemon snapshot exists, the scheduler falls back to the cached daemon snapshot and records a warning.
- Added `preflight_report.json` to every live refresh run packet:
  - `issues` records missing/failed/unconfirmed Kalshi mapping
  - `warnings` records dry-run checks that did not have a snapshot
  - `market_preflight_ok` is recorded in `manifest.json`
- Refresh `success` is now based on refresh/build/validation command success. Kalshi preflight issues do not make the data refresh fail.
- Execution remains separately blocked until active/open side/complement/settlement mapping is confirmed.

Verification:
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due` passed without a Kalshi socket error by loading the daemon snapshot:
  - run `20260511T004211Z_due_721f0208`
  - `kalshi_market_snapshot_loaded_from_daemon`
  - 6 active/open markets
  - `jobs=0`, `refresh=False`, `success=True`
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due --dry-run --now 2026-05-10T00:30:00-04:00` passed and confirmed Phoenix/Golden State mapping from the daemon snapshot.
- `python pipelines\07_live\14_live_data_refresh.py --year 2026 --mode due --dry-run --skip-market-api --daemon-snapshot-max-age-minutes 0 --now 2026-05-10T00:30:00-04:00` passed with a preflight warning, showing data-refresh planning is not blocked by missing Kalshi mapping.

## 2026-05-10 - Daemon Health Metrics

- Added `data/runs/live_daemon/health_latest.json` as the primary current health verdict.
- Added global `data/runs/live_daemon/health.jsonl` and per-session `sessions/<run_id>/health.jsonl` health histories.
- Heartbeats now include `health_status`, `health_summary`, and `health_path`.
- Health status can be:
  - `ok`
  - `degraded`
  - `failed`
- Health checks include:
  - process executing
  - lock file matches the daemon process
  - Kalshi market polling success/failure
  - latest market snapshot freshness
  - active/open WNBA moneyline market count
  - refresh worker success/failure
  - refresh worker freshness
  - current daemon `last_error`
- Health metrics include:
  - uptime
  - heartbeat interval
  - market poll interval
  - worker check interval
  - latest active/open market count
  - latest market snapshot age/path
  - tracked market history count
  - consecutive market poll failures
  - consecutive worker failures
- Diagnostic `--ignore-lock` runs no longer overwrite global `health_latest.json` or `heartbeat_latest.json`; they only write session-local health/heartbeat data.

Live verification:
- Restarted persistent daemon as hidden background PID `15304`.
- Current `health_latest.json`:
  - status `ok`
  - failed checks `0`
  - warning checks `0`
  - ok checks `8`
  - active/open moneyline markets `6`
  - market poll success `true`
  - worker return code `0`

Verification:
- `python -m compileall pipelines\07_live\15_live_daemon.py` passed.
- One-shot `--ignore-lock` smoke no longer changed global `health_latest.json`.

## 2026-05-10 - Per-Game Execution Ledger

- Added `src/srwnba/live/canonical/game_ledger.py`.
- Canonical route-loop execution can now write an append-only per-game ledger under `data/runs/live_games/<game_id>/`.
- Default production CLI behavior writes the ledger unless `--no-ledger` is explicitly passed.
- Ledger files:
  - `manifest.json`
  - `events.jsonl`
  - `prediction_packet.json`
  - `market_mapping.json`
  - `market_snapshots.jsonl`
  - `route_quotes.jsonl`
  - `execution_plans.jsonl`
  - `orders.jsonl`
  - `fills.jsonl`
  - `positions.jsonl`
  - `errors.jsonl`
  - `summary.json`
  - `sessions/<run_id>/...` per-process copies
- `route_entry_loop.py` now logs raw orderbook payloads as `market_snapshot` events before deriving route quotes.
- `route_loop_start` now includes the exact scored feature row, `p_raw`, `p_elo` when available, selected side, and model best round so the prediction packet can be audited without reconstructing from external files.
- Updated README/CLAUDE/AGENTS/data docs to make the per-game ledger the production audit source for prediction, market, and trade history.
- Added `data/live_logs/` and `data/runs/` to `.gitignore` because they are local runtime/audit artifacts that can contain order/fill details.

Verification:
- `python -m compileall src\srwnba\live\canonical\game_ledger.py src\srwnba\live\canonical\route_entry_loop.py pipelines\07_live\canonical\05_run_route_entry_loop.py pipelines\07_live\canonical\04_route_entry_loop_dryrun.py` passed.
- `python pipelines\07_live\canonical\04_route_entry_loop_dryrun.py --polls 2 --ledger-dir data\runs\live_games\__route_dryrun_ledger_smoke` passed and produced all required ledger files.
- Parsed `events.jsonl`, `market_snapshots.jsonl`, `route_quotes.jsonl`, `execution_plans.jsonl`, `orders.jsonl`, `fills.jsonl`, and `positions.jsonl` successfully from the smoke ledger.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python -m compileall src\srwnba\live pipelines\07_live\canonical` passed.

## 2026-05-10 - Kalshi Wealth-Based Sizing

- Added `src/srwnba/live/canonical/portfolio.py`.
- Production route-loop sizing now defaults to Kalshi `/portfolio/balance`:
  - `portfolio_value` is the default sizing bankroll.
  - `balance` is the fallback sizing bankroll if `portfolio_value` is missing or non-positive.
  - `balance` is the available-cash cap for order feasibility.
- Added explicit override controls:
  - `--sizing-bankroll-override <dollars>` replaces the Kalshi sizing bankroll for Kelly/cap math.
  - deprecated alias `--bankroll` maps to `--sizing-bankroll-override`.
  - `--available-cash-override <dollars>` replaces the Kalshi available-cash cap.
  - deprecated alias `--available-cash` maps to `--available-cash-override`.
- Route loop refreshes portfolio sizing during execution, defaulting to `--portfolio-refresh-interval-s 300`.
- Per-game ledgers now include `portfolio_sizing.jsonl`; `summary.json` carries the latest portfolio sizing snapshot.
- Updated route smoke test to use the same Kalshi wealth/override resolution as the production route loop.
- Updated README/CLAUDE/AGENTS/data docs to state that live production must not default to hardcoded `$5,000`.

Verification:
- `python -m compileall src\srwnba\live\canonical\portfolio.py src\srwnba\live\canonical\game_ledger.py src\srwnba\live\canonical\route_entry_loop.py pipelines\07_live\canonical\05_run_route_entry_loop.py pipelines\07_live\canonical\02_smoke_kalshi_routes.py pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed, including portfolio sizing default/fallback/override checks and route-loop Kalshi wealth application.
- `python pipelines\07_live\canonical\04_route_entry_loop_dryrun.py --polls 1 --ledger-dir data\runs\live_games\__route_dryrun_ledger_smoke` passed.

## 2026-05-10 - Launch Safety Layer 1-5

- Added `src/srwnba/live/canonical/operator_control.py`.
  - Global file: `data/runs/live_control/operator_control.json`.
  - Per-game files: `data/runs/live_control/game_overrides/<game_id>.json`.
  - Missing files mean normal trade-by-default execution.
  - Global auto-trade off, risk mode `kill`, or per-game `abort` blocks new execution.
- Added `src/srwnba/live/canonical/reconciliation.py`.
  - Startup route reconciliation fetches Kalshi fills/positions for both route tickers and seeds filled route cost/contracts from Kalshi.
  - Startup open-order recovery cancels bot-owned route orders and blocks on unknown open route orders.
  - Runtime route reconciliation computes dollar mismatch and feeds the v1.2 brake.
- Updated canonical `route_entry_loop.py` and `canonical/05_run_route_entry_loop.py`.
  - New CLI controls: `--operator-control-path`, `--operator-override-path`, `--position-reconcile-interval-s`, and `--skip-startup-reconciliation`.
  - `route_loop_start` and every execution plan log operator mode, risk mode, and position mismatch.
- Replaced `pipelines/07_live/07_game_telemetry.py` with canonical ledger telemetry.
  - Reads `data/runs/live_games/<game_id>/events.jsonl`.
  - Summarizes prediction, mapping, plans, route fills, operator state, settlement PnL, and optional Kalshi reconciliation.
- Added `pipelines/07_live/16_execution_supervisor.py`.
  - Finds upcoming gold feature rows.
  - Confirms active/open WNBA moneyline mapping.
  - Enforces local operator controls.
  - Avoids duplicate live route-loop PIDs.
  - Launches one `canonical/05_run_route_entry_loop.py` process per eligible game.
- Updated `pipelines/07_live/15_live_daemon.py`.
  - Adds `--execution-check-s`, `--execution-dry-run`, and `--disable-execution-supervisor`.
  - Invokes the execution supervisor on cadence and includes execution-supervisor health/freshness in heartbeat health.
- Updated `app.py`.
  - Displays local execution default mode and risk mode.
  - Adds global trade-by-default toggle and risk-mode selector.
  - Adds per-game Abort Game and Clear Abort controls.
  - Dashboard still does not call Kalshi directly.
- Updated README, CLAUDE, AGENTS, live pipeline docs, live package docs, and control-plane docs with the launch safety layer.

Verification:
- `python -m compileall src\srwnba\live\canonical pipelines\07_live\canonical pipelines\07_live\07_game_telemetry.py pipelines\07_live\15_live_daemon.py pipelines\07_live\16_execution_supervisor.py app.py` passed.
- `python pipelines\07_live\canonical\02_mapping_execution_unit_test.py` passed.
- `python pipelines\07_live\canonical\04_route_entry_loop_dryrun.py --polls 1 --ledger-dir data\runs\live_games\__route_dryrun_operator_smoke2` passed.
- `python pipelines\07_live\07_game_telemetry.py --ledger-dir data\runs\live_games\__route_dryrun_operator_smoke2 --json-out data\runs\live_games\__route_dryrun_operator_smoke2\telemetry_summary.json` passed.
- `python pipelines\07_live\16_execution_supervisor.py --year 2026 --plan-only --route-dry-run` passed; current gold had no future games inside the 24h launch window, so it launched none.
- `python pipelines\07_live\15_live_daemon.py --year 2026 --once --skip-market-api --worker-dry-run --execution-dry-run --ignore-lock` passed; session health included execution-supervisor checks with status `ok`.
- Operator-control smoke confirmed default allowed, per-game abort blocked, and clear-abort restored allowed.

## 2026-05-10 - Cash-Limited Ticket Priority

- Added `src/srwnba/live/canonical/cash_priority.py`.
- Locked the scarce-cash allocation rule to exact marginal expected log growth per dollar of cash consumed.
- The scorer uses the exact YES-equivalent binary expected log-growth formula:
  - `p * log(1 + f * ((1 - q) / q)) + (1 - p) * log(1 - f)`
  - `f = cost / bankroll_for_sizing`
- Existing fills plus reserved open orders are treated as current position cost, so priority is incremental and captures Kelly diminishing marginal utility.
- Added deterministic tie-breakers for scores within 5%:
  - higher absolute edge
  - higher normalized edge
  - earlier first qualification time
  - larger executable liquidity
  - lower route slippage
- Added plan/ledger fields for cash-limited mode, priority score, expected log growth of the next child, q-current, q-after-child, candidate child dollars, and skipped-due-to-cash state.
- Updated Streamlit dashboard surfaces and Supabase schema/patch SQL for cash-priority rank and diagnostics.
- Updated README/CLAUDE/AGENTS/live docs with the locked cash-scarcity behavior.
