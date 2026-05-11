# WNBA Prediction Market Model

A live trading system for WNBA moneyline contracts on Kalshi. It prices each game with the locked Elo + XGBoost model, then uses the locked v1.2 execution engine to build controlled pre-tipoff selected-team-wins exposure through confirmed equivalent routes, liquidity caps, fee-adjusted half-Kelly sizing, and hold-to-settlement exits.

> Research and tuning are complete. The forecasting stack, feature set, training procedure, data-pull timing, and execution architecture are **frozen** as production truth. The v1.2 execution spec locks entry timing, routing, liquidity caps, order cadence, passive probes, risk checks, audit logging, and expansion-team trading gates. See [`CLAUDE.md`](CLAUDE.md) for the strict AI implementation contract.

---

## How it works

```text
live game features (one row in gold-table schema)
        |
        v
FinalModel: Elo prior + XGBoost correction
trained on all accepted settled games
        |
        v
p_selected for the model-selected team
        |
        v
Kalshi market + route state
BUY YES selected-team market and/or BUY NO opponent market
        |
        v
Execution router v1.2
side mapping checks, fee-adjusted executable price,
edge gates, timing gates, liquidity caps, child slicing
        |
        v
Core limit-IOC sweeps + small early passive probes
        |
        v
Canonical selected-team-wins exposure
held to settlement
```

### Forecasting layers
1. **Elo backbone** — margin-of-victory Elo with home advantage, season carryover, and franchise continuity. Fixed knobs: `H=25, K=20, a=0.45, b=1.0, μ=1505`. Used as the structural prior.
2. **XGBoost correction** — 160 pregame features across player availability (126), recent form (10), style profile (12), and rest/travel (12). Elo passed as `base_margin = logit(p_elo)`, **not** as a feature column. Hyperparameters frozen.
3. **No Platt calibrator.** Tested against `p_raw` on 2025 holdout and was slightly worse (0.6130 vs 0.6121 log loss). Production uses `p_raw` directly.

### Production feature contract
- `data/silver/` contains canonical chronological facts at their natural grain: schedules, outcomes, player-game boxes, injury events/episodes, availability facts, played-game manifests, and daily/as-of `player_state_history`.
- `data/silver_plus/game_team_player_{year}_REGPST.csv` is the full game-as-of player feature-building store. It tracks every listed player for each team/game, including IDs, names, strength, minutes state, quality state, injury state, and audit fields.
- Other `silver_plus` tables are also game-as-of feature families: Elo, recent form, style, and schedule/travel context.
- `data/gold/game_xgboost_input_{year}_REGPST.csv` is the strict model-input projection. It contains metadata plus exactly the top 7 strength-ranked player slots per side and only the 9 locked player model features per slot.
- Gold must not contain p8-p12 columns, player IDs/names, `strength_pre`, origin/current city debug fields, or any research/debug columns.
- `src/srwnba/util/model_schema.py` is the single schema source for the 160 ordinary features and full gold column order.

### Trading layer
- **Canonical exposure:** selected team wins. The router may buy YES on the selected-team market or buy NO on the opponent market, but only after side, complement, and settlement mapping are confirmed.
- **Bankroll:** `$5,000` reference bankroll. Live sizing defaults to Kalshi `portfolio_value`; if unavailable it falls back to Kalshi `balance`. Actual order feasibility uses Kalshi `balance` after a 2% cash buffer.
- **Sizing override:** pass `--sizing-bankroll-override <dollars>` to size from an imaginary bankroll above or below Kalshi wealth. This changes Kelly/cap math only; use `--available-cash-override <dollars>` separately for dry-run/simulation cash caps.
- **Entry filters:** absolute edge `p_selected - q_exec_all_in >= 0.05` and normalized edge `(p_selected - q_exec_all_in) / q_exec_all_in >= 0.25`.
- **Pricing:** all edge and Kelly math uses expected fee-adjusted executable average price for the proposed child order, not just top-of-book.
- **Sizing:** half-Kelly, capped by 15% max portfolio exposure per canonical market, visible depth, recent qualifying-price volume, cold-start allowance, cumulative qualifying-volume share, and available cash.
- **Cash-limited priority:** when available cash is the binding cap across live tickets, allocate child-order cash by marginal expected log growth per dollar consumed. The exact score is `(expected_log_wealth(new_cost, p, q_avg_after_child) - expected_log_wealth(current_cost, p, q_current_position)) / proposed_child_cost`, so existing fills/reserves get diminishing marginal utility. Ties within 5% prefer higher absolute edge, higher normalized edge, earlier first qualification, larger executable liquidity, then lower slippage.
- **Timing:** monitor from T-24h, trade from T-17h, require first qualification before T-8h, cancel passive probes at T-8h, allow only prequalified tickets until T-4h, and stop all orders after T-4h.
- **Orders:** core exposure uses limit-IOC sweeps. Passive post-only limits are allowed only as small early price-improvement probes.
- **Exit:** hold to settlement. No early exits and no sell/reduce orders.
- **Expansion teams:** true first-season expansion teams are forecasted and update model state from game 1, but trading is blocked until every expansion team in the game has completed at least 14 prior games. The 2026 expansion IDs are Toronto Tempo (`4e4f726e-a015-4306-91a7-28e8576c7868`) and Portland Fire (`d54283cc-c5ec-4dbd-bb61-166f217e3864`). `data/config/kalshi_team_name_map.csv` must stay one row per team; alternate display names belong in `kalshi_aliases`, not duplicate rows.

### Implementation status
The strategy and execution design are no longer open. Remaining work is implementation and verification against the locked specs; do not introduce new entry mechanics, cadence rules, sizing caps, exits, or venue changes without an explicit unlock decision.

### Remote control plane
The Streamlit/Supabase control plane lives in [`app.py`](app.py) and [`supabase_io.py`](supabase_io.py). It is a remote dashboard only: it reads/writes Supabase control, snapshot, heartbeat, account, and audit tables, but it must never call Kalshi directly. When run on the same machine/workspace as the worker, it also writes local operator-control JSON: default mode is **trade all eligible games unless explicitly aborted**, with a webapp toggle for that default and a visible risk mode. See [`docs/control_plane_webapp.md`](docs/control_plane_webapp.md) before changing dashboard code or wiring the local trading worker into Supabase.

---

## Locked data-pull timing

Production data timing is locked:

- Run the settled-history refresh at **02:30 ET** every day during the season.
- Run a backup/correction refresh at **09:00 ET**.
- Generate the official pregame probability packet **20 hours before tipoff**.
- Treat **T-18h** as the manual-review deadline and earliest automated trading eligibility checkpoint, not the first probability calculation.
- Train live production probabilities on the current all-settled combined gold CSV. The 2025 holdout workflow is a regression reference, not the live training rule.

The historical injury backbone is the date-indexed `daily_injuries` feed. Each refresh must verify coverage for every calendar date from the last accepted injury date through today, inclusive, so missed runs are backfilled. Because recent injury reports can change after an earlier pull, the daily injury ingest must also force-refresh a recent lookback window, defaulting to today plus yesterday, and store each response as a new timestamped bronze payload. `/league/injuries.json` may be pulled once at T-20 as an audit/current-state cross-check, but it is not the primary feature feed and must not replace the daily injury event stream without an explicit unlock decision.

[`pipelines/07_live/14_live_data_refresh.py`](pipelines/07_live/14_live_data_refresh.py) is the production scheduler runner. It connects fixed 02:30/09:00 ET settled-history refreshes with market/game-time T-20 refresh triggers, writes a run packet under `data/runs/live_refresh/`, caches Kalshi market snapshots when market scheduling is checked, records bronze deltas, runs validation, and updates a local duplicate-run state file only after success.

[`pipelines/07_live/15_live_daemon.py`](pipelines/07_live/15_live_daemon.py) is the always-on local heartbeat. It polls and caches Kalshi WNBA markets, hard-filters every live working snapshot to active/open WNBA moneyline/team-wins markets only, detects ticker creation/update/disappearance, writes heartbeats under `data/runs/live_daemon/`, invokes the scheduler runner when due, and invokes the execution supervisor to launch one canonical route loop per eligible game. Install it as a Windows Task Scheduler job with [`scripts/install_live_daemon_task.ps1`](scripts/install_live_daemon_task.ps1).

The scheduler prefers the daemon's latest cached Kalshi snapshot before attempting a direct Kalshi API pull, so manual refresh runs should not depend on opening a second outbound Kalshi socket. For due T-20 market jobs, Kalshi mapping is recorded as a market preflight audit check. Snapshot/mapping issues do not fail the data refresh, but execution still requires confirmed active/open two-contract moneyline mapping before any order path.

Daemon health is written to `data/runs/live_daemon/health_latest.json` with an explicit `ok`/`degraded`/`failed` verdict, per-check status, market/worker/execution-supervisor freshness, consecutive failure counts, and the latest active moneyline count.

Every Sportradar pull must be staged before merge: write the raw response into an isolated run cache, log endpoint/request metadata with secrets removed, response hash, HTTP status, validation result, and promoted canonical path. Only validated pulls can be promoted into canonical `data/bronze`; failed or quarantined pulls stay isolated. Downstream feature builders must use only accepted sources whose knowledge timestamp is less than or equal to the probability as-of time.

Layer contract: `bronze` is immutable per-call payloads, `silver` is canonical chronological facts and daily/as-of state, `silver_plus` is game-as-of feature-family tables, and `gold` is the strict model-input projection. See [`docs/production_data_layers.md`](docs/production_data_layers.md).

---

## Validation reference numbers

These are regression-test ground truth. Production code that moves these numbers is a regression unless explicitly authorized.

### Forecasting (2025 holdout, 310 games)
| Model | Log Loss | Brier | Accuracy |
|-------|----------|-------|----------|
| Elo only | 0.6151 | 0.2132 | 66.8% |
| **XGBoost + Elo (production)** | **0.6121** | **0.2112** | **67.4%** |

Reproducer (run from repo root, `kalshi-wnba` env, `PYTHONPATH=src`):

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
# → log loss: 0.6121
# → accuracy: 0.6742
"
```

### Trading (2025 backtest at frozen config)
- 134 trades, 40.3% hit rate, 13.4% mean edge
- Ideal-fill return at $100 base: **+1062%**
- Sweep-execution return at $5K base: **+765%** (28% liquidity haircut)

Detailed research notebooks live in the frozen research workspace outside this production repo.

### Model vs. Kalshi on disagreement
When the model and Kalshi pre-tipoff implied probability disagree on the game direction (59 games in 2025), the model is correct **61%** of the time. This is the edge the production system extracts.

---

## Locked hyperparameters

All values live in [`config/final_hyperparams.py`](config/final_hyperparams.py) — single source of truth. Do not shadow elsewhere.

| Component | Parameters |
|-----------|-----------|
| Elo | `H=25, K=20, a=0.45, b=1.0, μ=1505` |
| Features | `N_players=7, h_M=7, L_inj=14, τ=150, h_team=7` |
| XGBoost | `max_depth=6, mcw=3, γ=0.1, cbt=0.6, sub=0.8, λ=1.0, α=0.0, lr=0.02, num_boost_round=3000, early_stopping_rounds=150, XGB_PRODUCTION_NUM_BOOST_ROUND=88, seed=42` |
| Trading | `bankroll=$5000, edge_min=0.05, norm_edge_min=0.25, kelly_fraction=0.5, fee_rate=0.07` |

XGBoost's final-fit best round on the 2015-2024 ES split is **88 trees**. Production training uses that fixed 88-tree count on the full current all-settled training CSV; it does not reselect a new `best_round` from a partial live season.

---

## Operations

### Daily data refresh
Pulls new bronze, builds silver/silver_plus/gold for the current season:

Run this on the locked **02:30 ET** main schedule and **09:00 ET** backup/correction schedule. Pulls should be staged, validated, and promoted before the canonical pipeline consumes them.

```bash
# Pull new bronze + parse + build features for current season
python pipelines/07_live/08_append_year.py --year 2026

# Extend Elo from prior season's saved end-state
python pipelines/07_live/11_extend_elo_to_year.py --year 2026 --force

# Build the gold table for the current season
python pipelines/07_live/08_append_year.py --year 2026 --from-phase gold

# Optional: combine 2015-2026 into one master training CSV
python pipelines/07_live/09_combine_gold.py --start-year 2015 --end-year 2026 --force

# Validate production artifacts before publishing them
python pipelines/07_live/13_validate_production_artifacts.py --year 2026
```

The orchestrator phases are `ingest → parse → feature → multiyear → gold`. Each stage script is idempotent. Use `--from-phase` / `--to-phase` to resume mid-pipeline.

### Live entry loop (per game, started ahead of tipoff)
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

The production T-20 prediction packet builder and live route loop are hardcoded to train from `data/gold/game_xgboost_input_2015_2026_REGPST.csv`. Passing `--train-csv` is deprecated and ignored. `FinalModel` trains on that full file with the locked 88-tree final count, not an early-stop round selected from partial 2026. Keep `2015_2024` for reproducing the 2025 holdout reference numbers, not for live production trading.

By default the route loop refreshes Kalshi wealth every 300 seconds. It sizes from Kalshi `portfolio_value` and uses Kalshi `balance` as the available-cash cap. `--sizing-bankroll-override` replaces only the sizing bankroll, even if it is smaller or larger than Kalshi wealth. `--available-cash-override` replaces the cash cap.

At startup and during execution, the canonical route loop reconciles Kalshi positions/fills against the two route tickers. Startup reconciliation seeds filled route cost/contract state from Kalshi so restarts cannot double-size existing exposure. It also discovers open route orders: bot-created open orders are cancelled before planning; unknown open orders on either route ticker block execution until manually resolved. Runtime reconciliation feeds the v1.2 position-mismatch brake.

Operator controls live in `data/runs/live_control/operator_control.json` and `data/runs/live_control/game_overrides/<game_id>.json`. Missing files mean normal trading is allowed. `auto_trade_enabled=false`, risk mode `kill`, or per-game decision `abort` blocks new execution. The webapp exposes the global trade-default toggle, risk mode, and per-game abort/clear-abort buttons.

Every canonical route-loop run writes a structured per-game audit packet under `data/runs/live_games/<game_id>/` by default. The packet is append-only across restarts and contains `prediction_packet.json` with model best round/source, `market_mapping.json`, raw `market_snapshots.jsonl`, evaluated `route_quotes.jsonl`, `portfolio_sizing.jsonl`, `execution_plans.jsonl`, `orders.jsonl`, `fills.jsonl`, `positions.jsonl`, `errors.jsonl`, and `summary.json`. The older `data/live_logs/<game_id>.route.jsonl` file remains the raw event stream; the per-game ledger is the primary local place to inspect prediction, market, and trade history for one game.

### Other operational pipelines
- [`pipelines/07_live/00_check_kalshi_auth.py`](pipelines/07_live/00_check_kalshi_auth.py) - read-only auth and balance smoke test
- [`pipelines/07_live/01_smoke_kalshi_market_data.py`](pipelines/07_live/01_smoke_kalshi_market_data.py) - read-only WNBA market/orderbook smoke test
- [`pipelines/07_live/canonical/02_mapping_execution_unit_test.py`](pipelines/07_live/canonical/02_mapping_execution_unit_test.py) - no-network checks for Kalshi mapping, expansion-team aliases/gate, equivalent routes, and initial IOC child planning
- [`pipelines/07_live/canonical/02_smoke_kalshi_routes.py`](pipelines/07_live/canonical/02_smoke_kalshi_routes.py) - read-only live route mapping and execution-plan smoke test for one Sportradar game
- [`pipelines/07_live/canonical/04_route_entry_loop_dryrun.py`](pipelines/07_live/canonical/04_route_entry_loop_dryrun.py) - end-to-end fake-client dryrun for canonical route execution, including BUY YES and BUY NO fills
- [`pipelines/07_live/canonical/05_run_route_entry_loop.py`](pipelines/07_live/canonical/05_run_route_entry_loop.py) - production-shaped canonical-route entry loop CLI; keep `--dry-run` until explicitly enabling write mode
- [`pipelines/07_live/16_execution_supervisor.py`](pipelines/07_live/16_execution_supervisor.py) - process supervisor that finds upcoming feature rows, confirms active/open moneyline mapping, enforces operator controls, avoids duplicate game loops, and launches one canonical route loop per eligible game
- [`pipelines/07_live/legacy/03_trader_unit_test.py`](pipelines/07_live/legacy/03_trader_unit_test.py) - legacy pure-planner unit tests (10 cases, must pass)
- [`pipelines/07_live/legacy/04_entry_loop_dryrun.py`](pipelines/07_live/legacy/04_entry_loop_dryrun.py) - legacy YES-only end-to-end dryrun against a real 2025 row with a fake Kalshi client
- [`pipelines/07_live/legacy/05_run_entry_loop.py`](pipelines/07_live/legacy/05_run_entry_loop.py) - legacy explicit-ticker YES-only live loop; retained for reference/testing
- [`pipelines/07_live/13_validate_production_artifacts.py`](pipelines/07_live/13_validate_production_artifacts.py) - hard production checks for schema, player priors, full-player store separation, and future-injury hygiene
- [`pipelines/07_live/06_passive_book_recorder.py`](pipelines/07_live/06_passive_book_recorder.py) — orderbook telemetry recorder (no orders)
- [`pipelines/07_live/07_game_telemetry.py`](pipelines/07_live/07_game_telemetry.py) — canonical per-game ledger plan/fill/PnL telemetry and optional Kalshi fill/position reconciliation

---

## Repository layout

```
config/
  final_hyperparams.py             # Single source of truth for all hyperparameters
src/srwnba/
  util/
    final_model.py                 # FinalModel — production probability source
    model_schema.py                 # Canonical 160-feature schema and gold column order
    elo.py                         # Elo update math (H=25, K=20, a=0.45, b=1.0)
  live/
    common.py                      # Shared Kalshi orderbook/fee helpers
    canonical/                     # Production-direction selected-team-wins route path
      kalshi_mapping.py            # Sportradar game/team IDs -> confirmed Kalshi routes
      execution.py                 # Fee-adjusted route quotes and order payload primitives
      v1_2.py                      # Locked timing, liquidity, slicing, passive, burst, and brake planner
      expansion_gate.py            # 14-completed-prior-games trading gate for true expansion teams
      portfolio.py                 # Kalshi wealth/cash resolution plus imaginary sizing overrides
      game_ledger.py               # Per-game prediction/market/order/fill audit packet writer
      route_entry_loop.py          # Canonical selected-team route loop wired to the v1.2 planner
    legacy/                        # Older explicit home/away YES-only path
      trader.py                    # Legacy pure sweep planner
      entry_loop.py                # Legacy FinalModel + Kalshi client + planner runtime
  client.py                        # Sportradar API client
  endpoints.py                     # Sportradar URL builders
  config.py                        # .env loader
  storage/                         # Bronze staging/promotion helpers
utils/
  kalshi_authed_client.py          # RSA-PSS signed Kalshi trade-API client
  kalshi_client.py                 # Unauthed read-only Kalshi client
pipelines/
  01_ingestion/                    # Sportradar API → bronze JSON
  02_parsing/                      # Bronze JSON → silver CSVs
  03_features/                     # Silver → feature tables
  04_gold/                         # Feature assembly → 160-feature XGBoost input
  07_live/                         # Live system pipelines (shared plus canonical/legacy subfolders)
data/
  config/
    kalshi_team_name_map.csv                      # One row per team; audited Kalshi display name + aliases -> Sportradar team UUID side map
  gold/
    game_xgboost_input_2015_2024_REGPST.csv     # 2025 holdout regression baseline
    game_xgboost_input_2015_20YY_REGPST.csv     # Live all-settled production training set
    game_xgboost_input_{year}_REGPST.csv         # Per-year tables
  bronze_runs/                                  # Isolated staged Sportradar pulls with manifests
  silver/                                        # Chronological canonical facts and daily/as-of state
  silver_plus/                                   # Game-as-of feature-family tables
  runs/                                          # Per-refresh, daemon, and per-game audit packets
    live_games/<game_id>/                        # Prediction, market, plan, order, fill, and summary ledger
  spec_sheets/                                   # Table & feature specifications
  config/                                        # franchise_map.csv, etc.
  live_logs/                                     # JSONL event logs from entry loops
CLAUDE.md                                        # Production spec for assistant context
.env                                             # SPORTRADAR_API_KEY (gitignored)
.env.example                                     # Template
```

---

## Environment

```bash
conda create -n kalshi-wnba python=3.11
conda activate kalshi-wnba
pip install -r requirements.txt
```

Required environment variables in `.env` (gitignored):
- `SPORTRADAR_API_KEY` — Sportradar WNBA API key
- `SPORTRADAR_KEY_ROTATED_AT`, `SPORTRADAR_KEY_EXPIRES_AT` — rotation bookkeeping
- `KALSHI_ACCESS_KEY`, `KALSHI_PRIVATE_KEY_PATH` — for the authed Kalshi trade-API client (RSA-PSS signing)
- `KALSHI_TRADING_ENABLED` - must be exactly `true` before any authenticated Kalshi write request is allowed. Keep `false` for read-only testing and dry runs.

`PYTHONPATH=src` is required when invoking pipeline scripts directly. The orchestrator [`08_append_year.py`](pipelines/07_live/08_append_year.py) sets this automatically for child processes.

---

## Frozen design decisions

1. **Elo as `base_margin`, not a feature.** Confirmed by the XGBoost-without-Elo benchmark in development (substantially worse than Elo + XGBoost).
2. **`p_raw`, not `p_platt`.** Platt calibration on 2020-2024 OOF predictions was tested and slightly degrades 2025 holdout log loss (0.6130 vs 0.6121).
3. **Walk-forward CV, not k-fold.** Sports data is temporal; future-data leakage is not acceptable.
4. **Pre-tipoff entry only.** No in-play trading. In-game price moves reflect live information not captured by the feature set.
5. **Hold to settlement.** Pre-tipoff convergence exits and partial-edge exits were tested and destroy realized return.
6. **Kalshi only.** Polymarket WNBA liquidity was rejected during research; Polymarket code/data do not belong in this production repo.
7. **Franchise continuity.** San Antonio Stars → Las Vegas Aces (2018) is treated as one franchise; Elo and player priors carry over.
8. **First 9 games of 2015 excluded.** No 2014 priors → uninformative EWMA + quality features. Enforced by `final_model._cold_start_mask`.

---

## What this codebase is *not*

- A research framework. Tuning is complete; do not run grid searches against it.
- A multi-sport library. The Elo + XGBoost architecture may transfer to NBA / others, but pursue that in a separate project.
- An in-play system. All entries are pre-tipoff.
- A multi-venue engine. Kalshi only.

For the assistant-facing operational spec — invariants, edge cases, and the bootstrap notes for refreshing the data pipeline — see [`CLAUDE.md`](CLAUDE.md).
---

## Execution System v1.2 - Human Operator Guide

This section is the complete locked WNBA Kalshi execution spec, imported from `wnba_kalshi_execution_spec_v1_2_expansion_gate.md`. It is written for human operators first: the main idea is that the bot is not allowed to "trade whenever edge exists." It must only build controlled pre-tipoff selected-team-wins exposure when the edge appears early enough, survives real all-in executable pricing, passes side-mapping checks, and fits inside strict liquidity and risk limits.

Quick plain-English summary:

- The model decides which team is mispriced; the execution layer decides whether that edge can be safely expressed in the real Kalshi market.
- The bot targets one canonical exposure: the selected team wins. It can express that by buying YES on the selected-team market or buying NO on the opponent market if the markets are confirmed complements.
- The bot starts monitoring 24 hours before tipoff, starts trading 17 hours before tipoff, refuses late-only signals after T-8h, and stops all orders after T-4h.
- Core orders are limit-IOC sweeps. Passive orders are small early probes for price improvement, not the main way exposure is built.
- The bot sizes from fee-adjusted half-Kelly, then cuts that target down by a 15% max-market cap, available cash, visible depth, recent traded volume, cumulative volume share, and child-order slicing rules.
- Open orders reserve exposure before new orders are placed. Filled exposure only increases; the bot never sells down before settlement.
- Behavioral/performance brakes are mostly log-only by default, while operational/data/market integrity brakes are enforced.
- Expansion-team games are forecasted from game 1, but trading is blocked until every true first-season expansion team in the game has 14 completed prior games.

# WNBA Kalshi Moneyline Execution System — Final Locked Spec v1.2 Expansion Gate

**Status:** Final locked execution architecture after QA patch pass plus expansion-team trading gate
**Scope:** WNBA Kalshi pre-tipoff moneyline execution only
**Primary objective:** Convert model-market price disagreement into controlled pre-tipoff exposure while avoiding late adverse-selection flow, overconsumption of thin liquidity, side-mapping errors, and operational failure modes.

---

## 1. Executive Summary

This system is an **early-window, liquidity-capped, adverse-selection-aware execution engine** for WNBA Kalshi moneyline markets.

The final locked design is:

```text
Pre-tipoff only.
Hold to settlement.
Forecast stack is fixed.
Canonical exposure = selected team wins.
Execution route may be BUY YES selected-team market or BUY NO opposing-team market.
Route selection/splitting is handled by a smart order router.
Half-Kelly desired sizing using fee-adjusted executable price.
15% max portfolio exposure per canonical market exposure.
No slate exposure cap.
No new signals after T-8h.
No additional orders after T-4h.
Reject late-only signals.
Core execution via limit-IOC sweeps.
Passive orders are small early price-improvement probes only.
Liquidity capped by visible depth, recent qualifying-price volume, cold-start allowance, and cumulative qualifying-volume share.
Open orders reserve exposure before new orders are sent.
Behavioral brakes and stop-losses are toggleable and default to log-only.
Operational integrity brakes default to enforce.
Hard market/data integrity checks remain mandatory.
True first-season expansion-team games are forecasted and updated from game 1, but trading is blocked until every expansion team in the game has completed at least 14 prior games.
```

The system should **not** be described as “trade whenever edge exists.” It should be described as:

> A pre-tipoff execution engine that enters only when the edge appears early enough, remains executable at the real all-in fill price, and can be expressed without becoming too large relative to the available WNBA Kalshi tape.

---

## 2. Fixed Strategy Assumptions

These are locked and are not part of the execution-parameter search.

```python
forecast_stack = "Elo prior + XGBoost correction"
p_raw = sigmoid(logit(p_elo) + xgb_output)

market_type = "WNBA moneyline"
venue = "Kalshi"
entry_type = "pre_tipoff_only"
exit_policy = "hold_to_settlement"
canonical_exposure = "selected_team_wins"
sizing_base = "half_kelly"
expansion_team_trading_gate = "require_14_completed_games_before_trade"
```

### 2.1 Forecast Probability

For each game and selected team:

```python
p_selected = sigmoid(logit(p_elo_selected) + xgb_output_selected)
```

where:

- `p_elo_selected` is the locked Elo-derived prior probability for the selected team.
- `xgb_output_selected` is the locked XGBoost correction for the selected team.
- `p_selected` is the production probability used by the execution system.

The execution system does **not** alter, recalibrate, or override `p_selected`.

### 2.2 No Exit / No Reduction Rule

Positions are held to settlement.

```text
Filled canonical exposure is monotone non-decreasing.
The bot never sells to reduce exposure.
If target_position_now falls below current filled exposure, the bot simply stops adding.
```

---

## 3. Side Normalization and Smart Order Routing

The execution system does **not** hardcode “buy YES only.” It hardcodes **canonical exposure**.

### 3.1 Canonical Exposure

For a selected team:

```text
canonical_exposure = selected_team_wins
```

Example:

```text
Selected team = New York Liberty
Canonical exposure = New York Liberty win the game
```

### 3.2 Allowed Equivalent Routes

For selected team `A` and opponent `B`, the allowed routes are:

```python
allowed_routes = [
    "BUY_YES_A_wins_market",
    "BUY_NO_B_wins_market",
]
```

Interpretation:

```text
BUY YES on A wins = A wins exposure
BUY NO on B wins = B does not win = A wins exposure, assuming the complement mapping is confirmed
```

### 3.3 Required Side-Mapping Checks

A route is eligible only if all side-mapping checks pass:

```python
require_side_mapping_confirmed = True
require_complement_market_confirmed = True
require_contract_side_confirmed = True
require_settlement_mapping_confirmed = True
```

The engine must verify:

```text
1. Selected-team market resolves YES if selected team wins.
2. Opponent market resolves YES if opponent wins.
3. The two markets are complementary for the same game and settlement event.
4. BUY NO on the opponent market is economically equivalent to selected team wins.
5. The order side sent to Kalshi exactly matches the intended canonical exposure.
```

If any mapping is uncertain:

```python
reject_reason = "side_mapping_not_confirmed"
```

### 3.4 Route Candidate Object

For every candidate route:

```python
route_candidate = {
    "route_id": str,
    "canonical_exposure": "selected_team_wins",
    "selected_team": str,
    "opponent_team": str,
    "market_ticker": str,
    "order_action": "buy",
    "outcome_side": "yes" | "no",
    "route_type": "BUY_YES_SELECTED" | "BUY_NO_OPPONENT",
    "side_mapping_confirmed": bool,
    "complement_market_confirmed": bool,

    "best_bid": float,
    "best_ask": float,
    "visible_depth_to_qmax": list,
    "visible_cost_dollars_at_or_below_qmax": float,
    "q_exec_raw": float,
    "q_exec_all_in": float,

    "rolling_liquidity_cap_dollars": float,
    "cumulative_liquidity_cap_remaining_dollars": float,
    "route_capacity_dollars": float,
}
```

### 3.5 Shared Canonical Target

There is **one target position** per selected-team canonical exposure.

If target exposure is `$600`, that means:

```text
Own up to $600 of selected-team-wins exposure in total.
```

It does **not** mean:

```text
$600 through BUY YES selected team plus $600 through BUY NO opponent.
```

Filled exposure from both routes rolls up to the same canonical position:

```python
canonical_filled_cost_dollars = (
    filled_cost_buy_yes_selected_team
    + filled_cost_buy_no_opponent
)
```

Open orders on both routes reserve against the same canonical target:

```python
canonical_reserved_open_order_cost_dollars = (
    reserved_cost_buy_yes_selected_team
    + reserved_cost_buy_no_opponent
)
```

### 3.6 Smart Order Router: Price-First, Capacity-Aware

After child order size is determined at the canonical level, the smart order router allocates it across eligible routes.

Eligible route condition:

```python
route_eligible = (
    route.side_mapping_confirmed
    and route.complement_market_confirmed
    and route.q_exec_all_in <= q_max_tick
    and route.route_capacity_dollars > 0
)
```

Default routing rule:

```text
1. Use the cheapest all-in executable route first.
2. If the cheapest route cannot absorb the full child size under route liquidity caps, spill into the next route.
3. If routes are within one tick of each other, split proportional to route capacity.
```

Price-tie threshold:

```python
route_price_tie_threshold_ticks = 1
```

If routes are tied:

```python
route_weight_i = route_capacity_i / sum(route_capacity_all_tied_routes)
route_order_size_i = child_order_size * route_weight_i
```

Otherwise:

```python
remaining_child = child_order_size

for route in sorted(eligible_routes, key=lambda r: r.q_exec_all_in):
    route_order_size = min(
        remaining_child,
        route.route_capacity_dollars,
        route.visible_cost_dollars_at_or_below_qmax,
    )
    send_child_order(route, route_order_size)
    remaining_child -= route_order_size
    if remaining_child <= 0:
        break
```

### 3.7 Route Logging

Every order attempt must log:

```python
side_route_log = {
    "model_team": str,
    "model_probability": float,
    "selected_team": str,
    "opponent_team": str,
    "canonical_exposure": "selected_team_wins",
    "market_ticker": str,
    "order_action": "buy",
    "outcome_side": "yes" | "no",
    "route_type": str,
    "side_mapping_rule": str,
    "q_exec_raw": float,
    "q_exec_all_in": float,
    "route_capacity_dollars": float,
    "route_order_size_dollars": float,
}
```

---

## 4. Core Price, Fee, and Edge Definitions

### 4.1 Price Units

All contract prices are represented as decimal probabilities in `[0, 1]`.

Example:

```text
Kalshi price 31 cents = q = 0.31
```

Dollar cost of a long contract position:

```python
position_cost_dollars = contracts * fill_price
```

Settlement payoff for winning long contracts:

```python
settlement_value = contracts * 1.00
```

### 4.2 Fee-Adjusted Executable Price

All edge checks and Kelly sizing must use **all-in executable price**.

```python
q_exec_all_in = q_exec_raw + expected_fee_per_contract_as_price_equivalent
```

where:

```python
expected_fee_per_contract_as_price_equivalent = expected_fee_dollars_per_contract / 1.00
```

If the fee model is unavailable or stale:

```python
reject_reason = "fee_model_unavailable"
```

unless explicitly configured to use a conservative fee fallback.

### 4.3 Edge Filters

Locked filters:

```python
edge_min = 0.05
norm_edge_min = 0.25
```

Definitions:

```python
absolute_edge = p_selected - q_exec_all_in
normalized_edge = (p_selected - q_exec_all_in) / q_exec_all_in
```

A route is eligible only if both hold at all-in executable price:

```python
absolute_edge >= 0.05
normalized_edge >= 0.25
```

### 4.4 Maximum Acceptable Price

For canonical selected-team exposure:

```python
q_max_raw = min(
    p_selected - 0.05,
    p_selected / 1.25,
)
```

Before order placement, round down to valid tick:

```python
q_max_tick = floor_to_valid_tick(q_max_raw)
```

The bot must never pay above `q_max_tick` all-in:

```python
q_exec_all_in <= q_max_tick
```

### 4.5 Executable Average Price

The edge check must be performed at the expected executable average price for the proposed route/order size, not merely the top ask.

```python
q_exec_raw = volume_weighted_average_price_for_proposed_child_order(route_book)
q_exec_all_in = q_exec_raw + fee_price_equivalent
```

If a proposed child order would sweep multiple price levels, `q_exec_raw` is the weighted average across those levels.

---

## 5. Binary Half-Kelly Formula

### 5.1 Full-Kelly Cost Fraction

For a binary contract that costs `q`, pays `$1` if it wins, and pays `$0` if it loses, with win probability `p`, the full-Kelly **cost fraction of bankroll** is:

```python
full_kelly_cost_fraction = (p - q) / (1 - q)
```

For this system:

```python
q = q_exec_all_in
p = p_selected
```

### 5.2 Half-Kelly Target

Locked Kelly fraction:

```python
kelly_fraction = 0.50
```

Formula:

```python
full_kelly_cost_fraction = (p_selected - q_exec_all_in) / (1 - q_exec_all_in)
full_kelly_cost_fraction = max(0.0, full_kelly_cost_fraction)

half_kelly_cost_fraction = 0.50 * full_kelly_cost_fraction
half_kelly_target_dollars = bankroll_for_sizing * half_kelly_cost_fraction
```

If `q_exec_all_in >= 1`, reject.

If `p_selected <= q_exec_all_in`, target is zero.

---

## 6. Bankroll, Cash, and Position Accounting

### 6.1 Bankroll for Sizing

Sizing bankroll is account equity, not just idle cash.

```python
bankroll_for_sizing = settled_cash + conservative_mtm_value_of_open_positions
```

For long selected-team exposure, conservative MTM should use liquidation value, not optimistic midpoint.

Example:

```python
conservative_mtm_value = contracts * current_best_exit_bid
```

If MTM is unreliable or unavailable, use a conservative fallback.

### 6.2 Available Cash for Orders

Order feasibility must use available cash.

```python
cash_buffer_pct = 0.02
cash_buffer_dollars = 0.02 * bankroll_for_sizing
available_cash_after_buffer = max(0.0, available_cash - cash_buffer_dollars)
```

A child order must satisfy:

```python
child_order_cost_dollars <= available_cash_after_buffer
```

This is **not** a slate exposure cap. It is account solvency and order-feasibility control.

### 6.3 Filled, Reserved, Remaining

For each canonical exposure ticket:

```python
filled_position_cost_dollars = cumulative_filled_cost_across_all_equivalent_routes
reserved_open_order_cost_dollars = max_cost_of_live_unfilled_orders_across_all_routes
```

Remaining desired exposure:

```python
remaining_position_now = max(
    0.0,
    target_position_now
    - filled_position_cost_dollars
    - reserved_open_order_cost_dollars
)
```

Open orders must reserve exposure until they are confirmed filled, cancelled, expired, or rejected.

### 6.4 Cancel Confirmation Rule

If a passive order is live and the router wants to send IOC exposure:

```python
send_cancel_passive()
wait_for_cancel_confirmation_or_fill_update()
recompute_filled_and_reserved_exposure()
then_evaluate_ioc()
```

Alternative implementation is allowed only if open passive cost remains fully reserved before the IOC is sized.

The bot must never assume cancellation is instantaneous.

---

## 7. Dynamic Target Position

### 7.1 Locked Naming

Use:

```python
target_position_now
```

Do **not** use `max_desired_position` as a permanent stored target.

### 7.2 Definition

`target_position_now` is the maximum total dollar exposure the bot wants to hold in this canonical market exposure **at the current evaluation moment**, before liquidity and child-order slicing.

```python
portfolio_cap_now = 0.15 * bankroll_for_sizing

half_kelly_target_now = compute_half_kelly_target(
    p_selected=p_selected,
    q_exec_all_in=best_route_or_planned_route_q_exec_all_in,
    bankroll_for_sizing=bankroll_for_sizing,
)

target_position_now = min(
    half_kelly_target_now,
    portfolio_cap_now,
    available_cash_after_buffer,
)

target_position_now = max(0.0, target_position_now)
```

### 7.3 Target Can Increase or Decrease

`target_position_now` is dynamic.

It can increase if:

```text
bankroll rises,
q_exec_all_in improves,
edge improves,
fees fall,
or available cash rises.
```

It can decrease if:

```text
price worsens,
fees rise,
p_selected falls,
bankroll falls,
or cash becomes constrained.
```

The filled position does not decrease because exits are disabled.

```python
if filled_position_cost_dollars >= target_position_now:
    remaining_position_now = 0
    no_additional_orders_allowed = True
```

### 7.4 Ticket Storage

Trade tickets store state and history, not a frozen target.

```python
trade_ticket = {
    "game_id": str,
    "canonical_exposure": "selected_team_wins",
    "selected_team": str,
    "opponent_team": str,

    "first_qualified_time": datetime | None,
    "first_qualified_lead_hours": float | None,
    "activated_before_T8": bool,
    "activation_time": datetime | None,

    "filled_position_cost_dollars": float,
    "filled_contracts_by_route": dict,
    "reserved_open_order_cost_dollars": float,
    "open_orders": list,

    "last_target_position_now": float,
    "last_remaining_position_now": float,
    "last_allowed_to_try_now": float,
    "last_child_order_size_dollars": float,

    "status": "monitoring" | "active" | "closed" | "expired" | "rejected",
}
```

---

## 8. High-Level Architecture

```text
Forecast p_selected
   ↓
Market and route state ingestion
   ↓
Side-normalization / complement-market confirmation
   ↓
Route-level executable price and all-in fee adjustment
   ↓
Edge computation
   ↓
Signal memory / first-qualification tracker
   ↓
Timing gate
   ↓
Signal eligibility gate
   ↓
Dynamic target_position_now
   ↓
Canonical remaining exposure after filled + reserved orders
   ↓
Route-level rolling liquidity caps
   ↓
Global combined liquidity caps
   ↓
Child-order slicer
   ↓
Smart order router across equivalent routes
   ↓
Passive manager or IOC manager
   ↓
Risk / integrity checks
   ↓
Order placement
   ↓
Fill/cancel processing
   ↓
Audit logging
```

---

## 9. Market and Route State Module

For every game/canonical exposure, the engine must maintain game state, route state, and source timestamps.

### 9.1 Required Game-Level Fields

```python
game_state = {
    "game_id": str,
    "selected_team": str,
    "opponent_team": str,
    "tipoff_time": datetime,

    "p_selected": float,
    "q_max_raw": float,
    "q_max_tick": float,

    "market_status_all_routes": dict,
    "settlement_mapping_confirmed": bool,
    "game_start_time_confirmed": bool,
    "complement_market_confirmed": bool,

    "market_data_timestamp": datetime,
    "orderbook_timestamp": datetime,
    "trades_feed_timestamp": datetime,
    "injury_feed_timestamp": datetime,
    "model_run_timestamp": datetime,
    "schedule_mapping_timestamp": datetime,
}
```

### 9.2 Required Route-Level Fields

```python
route_state = {
    "route_id": str,
    "market_ticker": str,
    "route_type": "BUY_YES_SELECTED" | "BUY_NO_OPPONENT",
    "outcome_side": "yes" | "no",

    "best_bid": float,
    "best_ask": float,
    "mid": float,
    "spread_ticks": int,

    "visible_depth_levels_at_or_below_qmax": list,
    "visible_contracts_at_or_below_qmax": int,
    "visible_cost_dollars_at_or_below_qmax": float,
    "q_exec_raw": float,
    "q_exec_all_in": float,

    "traded_cost_dollars_at_or_below_current_qmax_last_3h_ex_self": float,
    "traded_cost_dollars_at_or_below_current_qmax_since_first_qualification_ex_self": float,

    "last_trade_price": float,
    "last_trade_time": datetime,
    "visible_cost_after_last_order": float,
}
```

### 9.3 Visible Depth Definition

For each route:

```python
visible_cost_dollars_at_or_below_qmax = sum(
    price_i * quantity_i
    for level_i in route_executable_book
    if price_i <= q_max_tick
)
```

This is dollar cost, not number of contracts.

### 9.4 Qualifying-Price Volume Definition

Qualifying-price traded volume is historical volume that printed at prices the bot could theoretically have bought without violating the **current** `q_max_tick`.

```python
qualifying_trade_at_eval_time = trade_price <= current_q_max_tick
```

For each rolling or cumulative window:

```python
traded_cost_dollars_at_or_below_current_qmax = sum(
    trade_price_i * trade_quantity_i
    for trade_i in route_trades_in_window
    if trade_price_i <= current_q_max_tick
)
```

Self-trades/fills must be excluded where possible:

```python
traded_cost_dollars_at_or_below_current_qmax_ex_self
```

This avoids feedback loops where the bot creates volume and then interprets its own activity as external liquidity.

---

## 10. Signal Memory Module

The bot must be stateful. It must remember when each canonical exposure first became eligible.

### 10.1 Required Signal State

```python
signal_state = {
    "game_id": str,
    "canonical_exposure": "selected_team_wins",
    "selected_team": str,
    "p_selected": float,
    "q_max_tick": float,

    "first_qualified_time": datetime | None,
    "first_qualified_lead_hours": float | None,
    "first_qualified_route_id": str | None,
    "first_qualified_price_all_in": float | None,
    "first_qualified_edge": float | None,

    "currently_qualified": bool,
    "last_qualified_time": datetime | None,
    "num_qualifying_snapshots": int,
    "qualification_gaps": list,

    "signal_class": str,
    "drift_class": str,
}
```

### 10.2 Signal Classes

Diagnostic classes:

```text
early_stable
early_disappeared_then_late
late_only
```

Only `late_only` is a hard reject in v1.

### 10.3 First Qualification Rule

At each evaluation, qualification is checked across eligible equivalent routes.

```python
currently_qualified = any(
    route.q_exec_all_in <= q_max_tick
    for route in eligible_equivalent_routes
)
```

When the canonical exposure first passes both edge filters at all-in executable price:

```python
if signal.first_qualified_time is None and currently_qualified:
    signal.first_qualified_time = now
    signal.first_qualified_lead_hours = lead_hours
    signal.first_qualified_route_id = best_eligible_route.route_id
    signal.first_qualified_price_all_in = best_eligible_route.q_exec_all_in
```

A canonical exposure must first qualify at least 8 hours before tipoff:

```python
first_qualified_lead_hours >= 8
```

If first qualification occurs inside T-8h:

```python
reject_reason = "late_only_signal"
```

---

## 11. Timing Gates

### 11.1 Locked Timing Parameters

```python
monitor_start = "T-24h"
trade_start = "T-17h"

main_execution_window = "T-17h_to_T-8h"
no_new_entry_after = "T-8h"
cancel_passive_after = "T-8h"

prequalified_execution_window = "T-8h_to_T-4h"
no_orders_after = "T-4h"
```

### 11.2 Expected Behavior by Window

| Window | Behavior |
|---|---|
| T-24h to T-17h | Monitor only; record market state and first qualification; do not send orders. |
| T-17h to T-8h | Main execution window; new prequalified tickets and adds allowed. |
| T-8h to T-4h | Prequalified-ticket execution only; no late-only signals; no passive orders. |
| T-4h to tipoff | No orders. Monitor/log only. |

### 11.3 Important T-8h Interpretation

A canonical exposure that first qualifies before T-8h may remain eligible for execution until T-4h, even if it had zero fills before T-8h.

This window is better described as:

```text
prequalified-ticket execution only
```

not literal “completion only.”

A canonical exposure that first qualifies after T-8h is rejected, even if the current price appears attractive.

```python
if first_qualified_lead_hours < 8:
    reject_reason = "late_only_signal"
```

### 11.4 Ticket Activation

If a signal first qualifies before T-17h, the ticket may be created in monitoring state but cannot send orders yet.

```python
if first_qualified_lead_hours >= 8:
    ticket.status = "monitoring" if lead_hours > 17 else "active"
    ticket.activated_before_T8 = True
```

After T-8h, the bot may only work tickets with:

```python
ticket.activated_before_T8 == True
```

It must not create a new ticket for a canonical exposure that did not qualify before T-8h.

---

## 12. Signal Eligibility Gate

A canonical exposure is eligible for order routing only if all conditions pass.

```python
def signal_eligible(signal, routes, now, tipoff):
    lead_hours = (tipoff - now).total_seconds() / 3600

    if lead_hours <= 4:
        return False, "after_hard_no_add_cutoff"

    if signal.first_qualified_lead_hours is None:
        return False, "never_qualified"

    if signal.first_qualified_lead_hours < 8:
        return False, "late_only_signal"

    if not any(route.q_exec_all_in <= signal.q_max_tick for route in routes):
        return False, "edge_failed_at_executable_price"

    if not all_required_side_mappings_confirmed(routes):
        return False, "side_mapping_not_confirmed"

    if not market_status_open_for_any_eligible_route(routes):
        return False, "market_not_open"

    return True, "eligible"
```

### 12.1 Drift Logging

Drift is diagnostic only in v1.

```python
drift_filter = "log_only"
```

No hard reject is made solely because of drift class.

---

## 12A. Expansion-Team Trading Gate

This section handles true first-season WNBA expansion franchises. It is a **trading-layer gate**, not a forecasting or feature-state exclusion.

### 12A.1 Locked Policy

For true expansion franchises in their first WNBA season:

```python
forecast_games_with_expansion_teams = True
update_elo_for_expansion_teams = True
update_player_features_for_expansion_teams = True
update_recent_form_for_expansion_teams = True
update_style_features_for_expansion_teams = True
update_rest_travel_features_for_expansion_teams = True

expansion_team_min_completed_games_before_trading = 14
```

Interpretation:

```text
Forecast all expansion-team games from game 1.
Update all states after every expansion-team game from game 1.
Do not place live trades on games involving first-season expansion teams until every expansion team in the game has completed at least 14 prior games.
The first tradable game for a new expansion franchise is therefore its 15th game.
```

### 12A.2 Why This Is a Trading Gate Only

The model needs expansion-team games to enter the state pipeline immediately. Blocking state updates would prevent Elo, team recent form, team style, player roles, and rest/travel context from stabilizing.

Therefore, the locked behavior is:

```text
Forecasting: allowed from game 1.
Feature creation: allowed from game 1.
Elo updates: allowed from game 1.
Trading: blocked until the 14-completed-games threshold is met.
```

### 12A.3 Expansion-Team Initialization

For true expansion teams in their first season:

```python
elo_start = mu  # 1505
recent_form_start = 0
style_game_1_init = previous_season_league_average
```

Player-level priors carry by `player_id` when available:

```python
if player_has_prior_wnba_history:
    carry_player_q_and_m_ewma_by_player_id()
else:
    use_existing_no_history_player_prior_or_missing_handling()
```

This preserves the existing feature design: Elo starts at the locked league mean, recent form hard-resets, style uses a league-average fallback when no franchise prior exists, and player priors remain player-specific rather than team-specific.

### 12A.4 Franchise Continuity Rule

Do not confuse true expansion teams with relocations or franchise continuity cases.

```python
if franchise_continuity_confirmed:
    use_existing_franchise_carryover_rules
elif true_expansion_team_in_first_season:
    apply_expansion_team_gate
```

The existing continuity rule still applies to relocation cases such as San Antonio Stars → Las Vegas Aces. True first-season expansion franchises receive no franchise-level carryover.

### 12A.5 Trade Gate Function

```python
def expansion_team_trade_gate(game):
    expansion_teams = [
        team for team in [game.home_team, game.away_team]
        if team.is_true_expansion_team
        and game.season == team.expansion_season
    ]

    if not expansion_teams:
        return True, "normal_team_game"

    min_completed_games = min(
        team.games_played_before_game for team in expansion_teams
    )

    if min_completed_games < 14:
        return False, "blocked_expansion_team_under_14_completed_games"

    return True, "expansion_team_gate_passed"
```

If a game contains two first-season expansion teams, the gate uses the minimum completed games across those teams. Every expansion team involved in the game must have completed at least 14 prior games.

### 12A.6 Order-Routing Effect

If the expansion gate blocks a game:

```python
trade_allowed = False
orders_allowed = False
passive_allowed = False
ioc_allowed = False
```

The system should still log the forecast, edge, route prices, and hypothetical eligibility for diagnostics:

```python
log_market_eval(
    route_decision="no_trade",
    reject_reason="blocked_expansion_team_under_14_completed_games",
    expansion_team_gate_passed=False,
    expansion_team_min_completed_games=min_completed_games,
)
```

### 12A.7 Configuration

```python
expansion_team_gate = {
    "enabled": True,
    "applies_to_true_expansion_teams_only": True,
    "min_completed_games_before_trading": 14,
    "first_tradable_game_number": 15,
    "block_passive_orders": True,
    "block_ioc_orders": True,
    "forecast_and_update_states_while_blocked": True,
    "use_min_completed_games_if_multiple_expansion_teams": True,
}
```

---

## 13. Portfolio Sizing

### 13.1 Locked Sizing Parameters

```python
kelly_fraction = 0.50
max_market_exposure_pct = 0.15
min_target_size_dollars = 25
slate_exposure_cap = None
cash_buffer_pct = 0.02
```

### 13.2 Target Position Now

```python
half_kelly_target_now = compute_half_kelly_target(
    p_selected=p_selected,
    q_exec_all_in=best_available_route_q_exec_all_in,
    bankroll_for_sizing=bankroll_for_sizing,
)

portfolio_cap_now = 0.15 * bankroll_for_sizing
available_cash_cap_now = available_cash_after_buffer

target_position_now = min(
    half_kelly_target_now,
    portfolio_cap_now,
    available_cash_cap_now,
)

target_position_now = max(0.0, target_position_now)
```

If:

```python
target_position_now < 25
```

then:

```python
reject_reason = "target_too_small"
```

### 13.3 Remaining Position

```python
remaining_position_now = max(
    0.0,
    target_position_now
    - filled_position_cost_dollars
    - reserved_open_order_cost_dollars,
)
```

### 13.4 Interpretation

The system does **not** simply bet 15% per market.

Correct interpretation:

```text
Half-Kelly defines desired exposure.
15% of bankroll is a hard per-market tail-risk guardrail.
Available cash is an order-feasibility cap.
Actual exposure is usually smaller because rolling liquidity caps bind.
```

### 13.5 No Slate Exposure Cap

There is no hard same-day WNBA slate cap.

```python
slate_exposure_cap = None
```

Operational brakes still exist separately, but not as a normal slate-level exposure cap.

---

## 14. Liquidity Cap v1

This is the locked launch/default liquidity cap.

### 14.1 Locked Parameters

```python
max_visible_depth_participation = 0.25

recent_volume_window_hours = 3
max_recent_qualifying_volume_participation = 0.15

cold_start_bankroll_cap = 0.01
cold_start_visible_depth_participation = 0.15

max_cumulative_qualifying_volume_share = 0.30
exclude_self_volume = True
```

### 14.2 Route-Level Visible Depth Cap

For each route:

```python
route_visible_depth_cap = (
    0.25 * route.visible_cost_dollars_at_or_below_qmax
)
```

### 14.3 Route-Level Recent Qualifying-Volume Cap

```python
route_qualifying_volume_cap = (
    0.15 * route.traded_cost_dollars_at_or_below_current_qmax_last_3h_ex_self
)
```

### 14.4 Route-Level Cold-Start Allowance

```python
route_cold_start_cap = min(
    0.01 * bankroll_for_sizing,
    0.15 * route.visible_cost_dollars_at_or_below_qmax,
)
```

### 14.5 Route-Level Rolling Liquidity Cap

```python
route_effective_volume_cap = max(
    route_qualifying_volume_cap,
    route_cold_start_cap,
)

route_rolling_liquidity_cap = min(
    route_visible_depth_cap,
    route_effective_volume_cap,
)
```

### 14.6 Route-Level Cumulative Cap

```python
route_raw_cumulative_cap = (
    0.30 * route.traded_cost_dollars_at_or_below_current_qmax_since_first_qualification_ex_self
)

route_effective_cumulative_cap = max(
    route_raw_cumulative_cap,
    route_cold_start_cap,
)

route_cumulative_remaining = max(
    0.0,
    route_effective_cumulative_cap - route_filled_cost_dollars,
)
```

### 14.7 Route Capacity

```python
route_capacity_dollars = max(
    0.0,
    min(
        route_rolling_liquidity_cap,
        route_cumulative_remaining,
    )
)
```

### 14.8 Global Combined Cumulative Cap

Because equivalent routes represent the same canonical exposure, the bot must not take 30% of each route independently and accidentally become too much of the combined executable tape.

```python
combined_qualifying_volume_since_first_qualification_ex_self = sum(
    route.traded_cost_dollars_at_or_below_current_qmax_since_first_qualification_ex_self
    for route in equivalent_routes
)

global_raw_cumulative_cap = (
    0.30 * combined_qualifying_volume_since_first_qualification_ex_self
)

global_cold_start_cap = max(route.route_cold_start_cap for route in equivalent_routes)

global_effective_cumulative_cap = max(
    global_raw_cumulative_cap,
    global_cold_start_cap,
)

global_cumulative_remaining = max(
    0.0,
    global_effective_cumulative_cap - canonical_filled_cost_dollars,
)
```

### 14.9 Allowed to Try Now

```python
route_capacity_sum = sum(route.route_capacity_dollars for route in eligible_routes)

allowed_to_try_now = max(
    0.0,
    min(
        remaining_position_now,
        route_capacity_sum,
        global_cumulative_remaining,
        available_cash_after_buffer,
    )
)
```

If:

```python
allowed_to_try_now < min_child_order_dollars
```

then no order is sent, except for final cleanup of a small remaining position on an already partially filled trade ticket.

---

## 15. Cadence

### 15.1 Locked Cadence Parameters

```python
poll_T24_to_T17 = "15min"
poll_T17_to_T12 = "5min"
poll_T12_to_T8 = "2min"
poll_T8_to_T4 = "5min"
poll_T4_to_tip = "15min"

event_driven_recompute = True
event_driven_ordering = True
```

### 15.2 Debounce Parameters

```python
normal_min_time_between_ioc_sweeps = "60s"
burst_min_time_between_ioc_sweeps = "15s"
min_time_between_passive_updates = "5min"
```

Burst debounce overrides normal same-market debounce:

```python
if burst_mode:
    applicable_ioc_debounce = burst_min_time_between_ioc_sweeps
else:
    applicable_ioc_debounce = normal_min_time_between_ioc_sweeps
```

### 15.3 Behavior by Window

| Window | Mode | Cadence | Orders? |
|---|---|---:|---|
| T-24h to T-17h | monitor only | 15 min | No |
| T-17h to T-12h | entry detection | 5 min | Yes |
| T-12h to T-8h | main execution | 2 min | Yes |
| T-8h to T-4h | prequalified-ticket execution only | 5 min | Only tickets activated before T-8h |
| T-4h to tipoff | monitor only | 15 min | No |

### 15.4 Event-Driven Wakeups

Scheduled polling is the reliability backbone. The bot may wake up earlier on meaningful events.

Event triggers:

```python
event_triggers = {
    "best_ask_changes": True,
    "best_bid_changes": True,
    "visible_depth_to_qmax_changes_by_pct": 0.25,
    "new_trade_print": True,
    "spread_changes_by_ticks": 2,
    "q_exec_crosses_qmax": True,
}
```

The bot may recompute on every event, but order placement is still constrained by debounce, eligibility, liquidity cap, reserved exposure, and child-size rules.

### 15.5 Effective Remaining Opportunities

For child-order sizing:

```python
expected_remaining_opportunities = max(
    1,
    floor(time_until_execution_cutoff / current_poll_interval),
)

effective_remaining_opportunities = min(
    expected_remaining_opportunities,
    12,
)
```

---

## 16. Child-Order Slicing

### 16.1 Locked Parameters

```python
normal_max_ioc_child_order_pct = 0.025
completion_max_ioc_child_order_pct = 0.030
burst_max_ioc_child_order_pct = 0.050

passive_child_fraction_of_allowed = 0.25
max_passive_child_order_pct = 0.010

min_child_order_dollars = 25

urgency_multiplier_T17_to_T12 = 1.00
urgency_multiplier_T12_to_T8 = 1.50
urgency_multiplier_T8_to_T4 = 2.00

max_effective_remaining_opportunities = 12
```

### 16.2 Normal IOC Child Size

```python
base_slice = remaining_position_now / max(1, effective_remaining_opportunities)
urgency_adjusted_slice = base_slice * urgency_multiplier

normal_ioc_child_size = min(
    remaining_position_now,
    allowed_to_try_now,
    urgency_adjusted_slice,
    0.025 * bankroll_for_sizing,
    available_cash_after_buffer,
)
```

### 16.3 Prequalified Execution Window IOC Child Size

From T-8h to T-4h, only pre-T-8h tickets are eligible.

```python
completion_ioc_child_size = min(
    remaining_position_now,
    allowed_to_try_now,
    urgency_adjusted_slice,
    0.030 * bankroll_for_sizing,
    available_cash_after_buffer,
)
```

### 16.4 Burst IOC Child Size

```python
burst_ioc_child_size = min(
    remaining_position_now,
    allowed_to_try_now,
    0.050 * bankroll_for_sizing,
    2.5 * normal_ioc_child_size,
    available_cash_after_buffer,
)
```

### 16.5 Passive Child Size

```python
passive_child_size = min(
    0.25 * allowed_to_try_now,
    0.01 * bankroll_for_sizing,
    remaining_position_now,
    available_cash_after_buffer,
)
```

For a $5,000 bankroll:

```text
max passive child = $50
```

### 16.6 Minimum Effective Child Size

```python
min_child_order_dollars = 25
```

If calculated child size is below `$25`, the bot skips that cadence unless it is cleaning up a small remainder on a partially filled position.

Cleanup exception:

```python
if 0 < remaining_position_now < 25 and filled_position_cost_dollars > 0:
    allow_final_cleanup = True
```

### 16.7 Contract Rounding

Dollar child sizes must be converted into valid contract quantities.

For safety, use the worst allowed route limit price when converting dollars to contracts:

```python
child_contracts = floor(child_order_size_dollars / route_limit_price_tick)
```

Then:

```python
max_order_cost_dollars = child_contracts * route_limit_price_tick
```

If:

```python
child_contracts < 1
```

then skip the order.

The rounded order must still satisfy:

```python
max_order_cost_dollars <= child_order_size_dollars
max_order_cost_dollars <= available_cash_after_buffer
```

### 16.8 Tick Rounding

All limit prices must be valid Kalshi ticks.

```python
q_max_tick = floor_to_valid_tick(q_max_raw)
ioc_limit_price = q_max_tick
```

Passive prices must also be rounded down for buy orders:

```python
passive_price_tick = floor_to_valid_tick(passive_price_raw)
```

Never round up above `q_max_raw`.

---

## 17. Passive Limit Order Manager

Passive orders are small early probes for price improvement. They are not the primary execution mechanism.

### 17.1 Locked Parameters

```python
passive_enabled = True
passive_order_type = "post_only_limit"

passive_allowed_start = "T-17h"
passive_allowed_end = "T-8h"

min_spread_for_passive_ticks = 2

passive_child_fraction_of_allowed = 0.25
max_passive_child_order_pct = 0.010
min_passive_order_dollars = 25

passive_timeout_T17_to_T12 = "15min"
passive_timeout_T12_to_T8 = "10min"

max_upward_reprices_per_passive_episode = 2
min_time_between_passive_reprices = "5min"
passive_episode_cooldown_after_chase_limit = "10min"

cancel_all_passives_at_T8 = True
```

### 17.2 Route-Specific Passive Price

For each route, use the route-specific buy book.

```python
passive_price_raw = min(
    route.best_bid + 1_tick,
    route.midpoint_rounded_down,
    q_max_tick - 1_tick,
)

passive_price_tick = floor_to_valid_tick(passive_price_raw)
```

Only place if:

```python
passive_price_tick > route.best_bid
passive_price_tick < route.best_ask
passive_price_tick <= q_max_tick - 1_tick
route.spread_ticks >= 2
```

If spread is one tick, skip passive and either route to IOC if executable or wait.

### 17.3 Post-Only Requirement

Passive orders must be post-only.

If a post-only passive order would cross and execute immediately:

```python
do_not_convert_to_taker
cancel_or_skip_passive
```

If the bot wants to take liquidity, it must use the IOC manager.

### 17.4 Passive Repricing Rule

Repricing limit is episode-based, not market-lifetime-based.

```python
max_upward_reprices_per_passive_episode = 2
```

An upward reprice for a buy order means:

```text
Bid 29 → 30 = chase, counts
Bid 30 → 31 = chase, counts
Bid 31 → 29 = defensive lower reprice, does not count
```

After two upward reprices within one passive episode:

```python
cancel_passive_order()
start_passive_episode_cooldown("10min")
```

After cooldown, if the canonical exposure still qualifies and time is before T-8h, a new passive episode may begin.

### 17.5 Passive Timeout Reset

If a passive order partially fills:

```text
Timeout resets only if the remaining passive order size is still meaningful.
If remaining passive size is below min_passive_order_dollars, cancel/recompute.
```

### 17.6 Passive Cancel Conditions

Cancel passive immediately if any of the following occur:

```python
lead_hours <= 8
current_q_max_tick < passive_order_price
edge_filters_fail
route.best_ask <= q_max_tick
route.spread_ticks <= 1
market_data_stale
orderbook_disconnect
filled_position_cost_dollars >= target_position_now
```

If:

```python
route.best_ask <= q_max_tick
```

then:

```python
send_cancel_passive()
wait_for_cancel_confirmation_or_fill_update()
recompute_remaining_position()
evaluate_ioc()
```

Passive orders must never block IOC execution, but they must remain reserved until cancellation is confirmed.

---

## 18. Execution Router

The execution router decides whether to use passive, normal IOC, burst IOC, or no order.

### 18.1 Routing Hierarchy

```python
if not signal_eligible:
    route = "no_trade"

elif lead_hours <= 4:
    route = "no_trade"

elif burst_mode_triggered:
    route = "burst_ioc"

elif any(route.q_exec_all_in <= q_max_tick for route in eligible_routes):
    route = "normal_ioc"

elif passive_allowed and any(route.spread_ticks >= 2 for route in eligible_routes):
    route = "passive_probe"

else:
    route = "wait"
```

### 18.2 Passive vs IOC Priority

IOC has priority when executable liquidity is available inside `q_max_tick`.

```python
if passive_order_live and any(route.q_exec_all_in <= q_max_tick for route in eligible_routes):
    send_cancel_passive()
    wait_for_cancel_confirmation_or_fill_update()
    recompute_remaining_position()
    evaluate_ioc()
```

---

## 19. IOC Sweep Manager

IOC sweeps are the core exposure mechanism.

### 19.1 Locked Parameters

```python
ioc_order_type = "limit_ioc"
ioc_limit_price = "q_max_tick"

ioc_main_window = "T-17h_to_T-8h"
ioc_prequalified_window = "T-8h_to_T-4h"
ioc_hard_stop = "T-4h"

normal_child_cap_pct = 0.025
completion_child_cap_pct = 0.030
normal_debounce = "60s"

burst_enabled = True
burst_child_cap_pct = 0.050
burst_depth_multiplier_trigger = 2.0
burst_min_visible_depth_pct_bankroll = 0.03
burst_debounce = "15s"
max_burst_orders_per_5min = 3
max_burst_total_per_5min_pct = 0.07
```

### 19.2 Normal IOC Trigger

```python
normal_ioc_allowed = (
    signal_eligible
    and any(route.q_exec_all_in <= q_max_tick for route in eligible_routes)
    and remaining_position_now > 0
    and allowed_to_try_now >= min_child_order_dollars
    and normal_ioc_debounce_expired
)
```

### 19.3 Burst Mode Trigger

Burst mode handles short-lived liquidity pockets.

```python
burst_mode = (
    signal_eligible
    and any(route.q_exec_all_in <= q_max_tick for route in eligible_routes)
    and combined_visible_cost_at_or_below_qmax_now >= 2.0 * combined_visible_cost_at_or_below_qmax_prev
    and combined_visible_cost_at_or_below_qmax_now >= 0.03 * bankroll_for_sizing
    and depth_refresh_confirmed
)
```

### 19.4 Depth Refresh Confirmation

Burst mode must only trigger on new or refreshed executable depth.

```python
refresh_threshold = max(
    25.0,
    0.005 * bankroll_for_sizing,
    0.10 * combined_visible_cost_at_or_below_qmax_after_last_order,
)

depth_refresh_confirmed = (
    combined_visible_cost_at_or_below_qmax_now
    > combined_visible_cost_at_or_below_qmax_after_last_order + refresh_threshold
)
```

For a `$5,000` bankroll, the minimum meaningful refresh is at least `$25`.

### 19.5 Burst Debounce and Burst Limits

```python
burst_min_time_between_ioc_sweeps = "15s"
max_burst_orders_per_5min = 3
max_burst_total_per_5min = 0.07 * bankroll_for_sizing
```

For a `$5,000` bankroll:

```text
max_burst_total_per_5min = $350
```

Burst debounce overrides normal same-market debounce.

### 19.6 Sweep-Short Handling

If IOC partially fills:

```python
filled_position_cost_dollars += fill_cost_dollars
remaining_position_now = recompute_remaining_position()
log_sweep_short()
```

Do not immediately resend unless one of the following is true:

```python
new_depth_refresh_confirmed
burst_debounce_expired
next_scheduled_or_event_cadence_arrives
```

---

## 20. Risk Controls and Kill Switches

Risk controls are divided into:

1. **Mandatory hard integrity checks**
2. **Toggleable operational brakes**
3. **Toggleable behavioral/performance brakes**

Hard integrity checks cannot be disabled. Brakes are configurable.

### 20.1 Brake Modes

Every brake must support:

```python
enabled_flag
threshold
action
mode  # enforce | log_only | disabled
```

Mode behavior:

```text
enforce  = trigger action and log
log_only = log would-trigger but do not alter trading
disabled = do not evaluate or act
```

---

## 21. Mandatory Hard Integrity Checks

These are always enforced.

### 21.1 Data Freshness

```python
max_market_data_staleness_seconds = 30
max_orderbook_disconnect_seconds = 30
max_injury_data_age_minutes = 60
```

Model freshness is not only wall-clock age. A model snapshot is stale if either:

```text
1. no successful model run occurred within the configured freshness interval, or
2. a required upstream input feed updated after the last successful model run and the model has not rerun.
```

Config:

```python
max_model_snapshot_age_minutes = 60
require_model_newer_than_required_inputs = True
```

Mandatory behavior:

```python
if market_data_stale:
    block_new_orders

if model_snapshot_stale:
    block_new_orders

if injury_data_stale:
    block_new_orders

if orderbook_disconnected:
    cancel_passives
    block_new_orders
```

### 21.2 Source-Level Timestamp Checks

The system must log and check:

```python
source_timestamps = {
    "odds_feed_last_success": datetime,
    "orderbook_last_success": datetime,
    "trades_feed_last_success": datetime,
    "injury_feed_last_success": datetime,
    "model_run_last_success": datetime,
    "schedule_mapping_last_success": datetime,
}
```

### 21.3 Market Integrity Checks

Before any order:

```python
require_market_open = True
require_settlement_mapping_confirmed = True
require_game_start_time_confirmed = True
require_contract_side_confirmed = True
require_complement_market_confirmed = True
require_side_mapping_confirmed = True
```

If any fail:

```python
reject_market
block_new_orders
```

### 21.4 Order Sanity Checks

Before every order:

```python
assert route.q_exec_all_in <= q_max_tick
assert lead_hours > 4
assert remaining_position_now > 0
assert child_order_size_dollars >= min_child_order_dollars
assert child_order_size_dollars <= allowed_to_try_now
assert child_order_size_dollars <= available_cash_after_buffer
assert total_reserved_plus_filled_after_order <= target_position_now
assert total_reserved_plus_filled_after_order <= 0.15 * bankroll_for_sizing
assert side_mapping_confirmed
assert child_contracts >= 1
```

If any fail:

```python
block_order
log_reject_reason
```

---

## 22. Toggleable Operational Brakes

Operational brakes default to `enforce`.

### 22.1 Order Reject Brake

```python
enable_order_reject_brake = True
max_order_rejects_per_hour = 5
order_reject_brake_mode = "enforce"
```

Behavior:

```python
if enable_order_reject_brake and order_rejects_last_hour >= 5:
    block_new_orders
```

### 22.2 API Error Brake

```python
enable_api_error_brake = True
max_api_errors_per_10min = 10
api_error_brake_mode = "enforce"
```

Behavior:

```python
if enable_api_error_brake and api_errors_last_10min >= 10:
    cancel_passives
    block_new_orders
```

### 22.3 Position Mismatch Brake

```python
enable_position_mismatch_brake = True
max_position_mismatch_dollars = 10
position_mismatch_brake_mode = "enforce"
```

Behavior:

```python
if enable_position_mismatch_brake and abs(local_position - exchange_position) > 10:
    cancel_passives
    block_new_orders
```

---

## 23. Toggleable Behavioral / Performance Brakes

Behavioral and performance brakes default to `log_only`, not `enforce`.

Reason:

```text
The strategy has a low hit rate but positive expectancy.
Normal losing streaks can occur.
Performance brakes should be observed live before they are allowed to suppress trading.
```

### 23.1 Daily Realized Loss Brake

```python
enable_daily_realized_loss_stop = True
daily_realized_loss_stop_pct = 0.10
daily_realized_loss_stop_mode = "log_only"
```

If enforced:

```python
if daily_realized_loss <= -0.10 * start_of_day_bankroll:
    block_new_orders_for_day
```

### 23.2 Daily Mark-to-Market Loss Brake

```python
enable_daily_mtm_loss_stop = True
daily_mtm_loss_stop_pct = 0.15
daily_mtm_loss_stop_mode = "log_only"
```

MTM must use conservative liquidation value:

```python
mtm_value = contracts * current_best_exit_bid
```

If enforced:

```python
if daily_mtm_loss <= -0.15 * start_of_day_bankroll:
    block_new_orders_for_day
```

### 23.3 Consecutive-Loss Soft Brake

```python
enable_consecutive_loss_soft_brake = True
consecutive_loss_soft_brake = 4
consecutive_loss_soft_brake_mode = "log_only"
```

If enforced:

```python
if consecutive_losses >= 4:
    switch_to_conservative_mode
```

### 23.4 Consecutive-Loss Hard Brake

```python
enable_consecutive_loss_hard_brake = True
consecutive_loss_hard_brake = 6
consecutive_loss_hard_brake_mode = "log_only"
```

If enforced:

```python
if consecutive_losses >= 6:
    block_new_orders_until_manual_review
```

### 23.5 Conservative Mode

```python
enable_conservative_mode = True
conservative_max_market_exposure_pct = 0.12
disable_burst_in_conservative_mode = True
```

When active:

```python
max_market_exposure_pct = 0.12
burst_mode_enabled = False
```

### 23.6 Calibration Brake

```python
enable_calibration_brake = True
calibration_window_trades = 30
min_expected_edge_realization_ratio = 0.50
calibration_brake_mode = "log_only"
```

Use rolling sums, not average of individual ratios:

```python
if settled_trades < calibration_window_trades:
    calibration_status = "insufficient_sample"
    do_not_enforce

if sum_expected_log_return <= small_positive_threshold:
    calibration_status = "invalid_denominator"
    do_not_enforce

realized_edge_ratio = sum_actual_log_returns / sum_expected_log_returns
```

If enforced:

```python
if realized_edge_ratio < 0.50:
    switch_to_conservative_mode
```

---

## 24. Binding-Cap Logging

For every market decision and every order attempt, log which constraint binds.

```python
binding_cap = argmin({
    "half_kelly": half_kelly_target_now,
    "portfolio_cap": 0.15 * bankroll_for_sizing,
    "available_cash": available_cash_after_buffer,
    "remaining_position": remaining_position_now,
    "route_capacity_sum": route_capacity_sum,
    "global_cumulative_liquidity": global_cumulative_remaining,
    "child_order_cap": child_order_cap,
    "urgency_adjusted_slice": urgency_adjusted_slice,
})
```

This is essential because the system changes regime across the season:

```text
Early season: portfolio cap may bind more often.
Mid/late season: liquidity caps may bind more often as bankroll grows.
If 2026 liquidity is much deeper: portfolio cap becomes more important again.
```

---

## 25. Required Audit Logging

Every market evaluation should produce structured logs with standardized units.

Use suffixes:

```text
*_contracts
*_cost_dollars
*_price
*_pct
*_timestamp
```

### 25.1 Market Evaluation Log

```python
market_eval_log = {
    "timestamp": datetime,
    "game_id": str,
    "canonical_exposure": "selected_team_wins",
    "selected_team": str,
    "opponent_team": str,
    "lead_hours": float,

    "p_selected": float,
    "q_max_raw": float,
    "q_max_tick": float,

    "best_route_id": str | None,
    "best_route_q_exec_raw": float | None,
    "best_route_q_exec_all_in": float | None,
    "absolute_edge_at_q_exec_all_in": float | None,
    "normalized_edge_at_q_exec_all_in": float | None,

    "first_qualified_time": datetime | None,
    "first_qualified_lead_hours": float | None,
    "signal_class": str,
    "drift_class": str,

    "bankroll_for_sizing": float,
    "available_cash_after_buffer": float,
    "target_position_now": float,
    "filled_position_cost_dollars": float,
    "reserved_open_order_cost_dollars": float,
    "remaining_position_now": float,

    "route_capacity_sum": float,
    "global_cumulative_remaining": float,
    "allowed_to_try_now": float,

    "binding_cap": str,
    "route_decision": str,
    "reject_reason": str | None,
}
```

### 25.2 Route Evaluation Log

```python
route_eval_log = {
    "timestamp": datetime,
    "game_id": str,
    "route_id": str,
    "market_ticker": str,
    "route_type": "BUY_YES_SELECTED" | "BUY_NO_OPPONENT",
    "outcome_side": "yes" | "no",

    "side_mapping_confirmed": bool,
    "complement_market_confirmed": bool,
    "best_bid_price": float,
    "best_ask_price": float,
    "q_exec_raw": float,
    "q_exec_all_in": float,

    "visible_cost_dollars_at_or_below_qmax": float,
    "route_rolling_liquidity_cap": float,
    "route_cumulative_remaining": float,
    "route_capacity_dollars": float,
}
```

### 25.3 Order Attempt Log

```python
order_attempt_log = {
    "timestamp": datetime,
    "game_id": str,
    "route_id": str,
    "market_ticker": str,
    "order_type": str,
    "order_mode": "passive" | "normal_ioc" | "burst_ioc",
    "order_action": "buy",
    "outcome_side": "yes" | "no",
    "limit_price": float,
    "child_order_size_dollars": float,
    "child_order_contracts": int,
    "expected_q_exec_raw": float,
    "expected_q_exec_all_in": float,
    "q_max_tick": float,
    "allowed_to_try_now": float,
    "binding_cap": str,
    "debounce_state": str,
    "risk_state": str,
}
```

### 25.4 Fill Log

```python
fill_log = {
    "timestamp": datetime,
    "game_id": str,
    "route_id": str,
    "market_ticker": str,
    "order_id": str,
    "outcome_side": "yes" | "no",
    "fill_price": float,
    "fill_quantity_contracts": int,
    "fill_cost_dollars": float,
    "fee_dollars": float,
    "fill_cost_all_in_dollars": float,
    "partial_fill": bool,
    "sweep_short": bool,
    "canonical_filled_position_after_cost_dollars": float,
    "canonical_remaining_position_after_cost_dollars": float,
}
```

### 25.5 Settlement Log

```python
settlement_log = {
    "game_id": str,
    "canonical_exposure": "selected_team_wins",
    "selected_team": str,
    "result": "win" | "loss",
    "total_contracts_by_route": dict,
    "total_cost_dollars": float,
    "total_fees_dollars": float,
    "settlement_value_dollars": float,
    "pnl_dollars": float,
    "log_return": float,
    "bankroll_before": float,
    "bankroll_after": float,
}
```

---

## 26. Full Decision Flow Pseudocode

```python
def evaluate_canonical_exposure(game, signal, bankroll_for_sizing, available_cash, now):
    lead_hours = hours_until(game.tipoff_time, now)

    # Mandatory integrity checks
    if not hard_integrity_checks_pass(game, signal, now):
        cancel_passives_if_needed(game)
        return "blocked_integrity"

    # Expansion-team trading gate
    expansion_ok, expansion_reason = expansion_team_trade_gate(game)
    if not expansion_ok:
        log_market_eval(
            route_decision="no_trade",
            reject_reason=expansion_reason,
            expansion_team_gate_passed=False,
        )
        return "no_trade"

    # Build/refresh equivalent route candidates
    routes = build_equivalent_routes(game, signal)
    routes = [r for r in routes if side_mapping_checks_pass(r)]

    if not routes:
        log_market_eval(route_decision="no_trade", reject_reason="no_confirmed_equivalent_route")
        return "no_trade"

    # Fee-adjusted route prices
    for route in routes:
        route.q_exec_raw = compute_route_q_exec(route)
        route.q_exec_all_in = add_expected_fees(route.q_exec_raw, route)

    # Update signal memory
    update_signal_qualification_state(game, signal, routes, now)

    # Timing gate
    timing_state = get_timing_state(lead_hours)
    if timing_state == "monitor_only":
        log_market_eval(route_decision="monitor_only")
        return "monitor_only"

    if timing_state == "no_orders":
        cancel_passives_if_needed(game)
        log_market_eval(route_decision="no_orders")
        return "no_orders"

    # Signal eligibility
    eligible, reason = signal_eligible(signal, routes, now, game.tipoff_time)
    if not eligible:
        cancel_passives_if_needed(game)
        log_market_eval(route_decision="no_trade", reject_reason=reason)
        return "no_trade"

    # Ticket gating
    ticket = get_or_create_prequalified_ticket_if_allowed(game, signal, lead_hours)
    if ticket is None:
        log_market_eval(route_decision="no_ticket", reject_reason="late_or_invalid_ticket")
        return "no_trade"

    # Bankroll/cash accounting
    available_cash_after_buffer = compute_available_cash_after_buffer(
        available_cash=available_cash,
        bankroll_for_sizing=bankroll_for_sizing,
    )

    # Dynamic target
    best_route = min(routes, key=lambda r: r.q_exec_all_in)
    half_kelly_target_now = compute_half_kelly_target(
        p_selected=signal.p_selected,
        q_exec_all_in=best_route.q_exec_all_in,
        bankroll_for_sizing=bankroll_for_sizing,
    )

    target_position_now = min(
        half_kelly_target_now,
        0.15 * bankroll_for_sizing,
        available_cash_after_buffer,
    )
    target_position_now = max(0.0, target_position_now)

    if target_position_now < 25:
        log_market_eval(route_decision="no_trade", reject_reason="target_too_small")
        return "no_trade"

    # Reserved exposure accounting
    filled_position = ticket.filled_position_cost_dollars
    reserved_position = ticket.reserved_open_order_cost_dollars
    remaining_position_now = max(0.0, target_position_now - filled_position - reserved_position)

    if remaining_position_now <= 0:
        log_market_eval(route_decision="wait", reject_reason="target_already_filled_or_reserved")
        return "wait"

    # Route-level and global liquidity caps
    compute_route_capacities(routes, signal, bankroll_for_sizing)
    global_cumulative_remaining = compute_global_cumulative_remaining(routes, ticket)

    allowed_to_try_now = max(
        0.0,
        min(
            remaining_position_now,
            sum(r.route_capacity_dollars for r in routes),
            global_cumulative_remaining,
            available_cash_after_buffer,
        ),
    )

    if allowed_to_try_now <= 0:
        log_market_eval(route_decision="wait", reject_reason="liquidity_cap_zero")
        return "wait"

    # Toggleable brakes
    if operational_or_behavioral_brakes_block_orders():
        log_market_eval(route_decision="blocked_brake")
        return "blocked_brake"

    # Choose execution mode
    if burst_mode_triggered(game, routes, signal, bankroll_for_sizing):
        child_size = compute_burst_child_size(...)
        return smart_route_ioc_child(game, ticket, routes, child_size, mode="burst_ioc")

    if any(r.q_exec_all_in <= signal.q_max_tick for r in routes):
        child_size = compute_normal_or_completion_child_size(...)
        return smart_route_ioc_child(game, ticket, routes, child_size, mode="normal_ioc")

    if passive_allowed(game, signal, routes, lead_hours):
        child_size = compute_passive_child_size(...)
        return route_passive_probe(game, ticket, routes, child_size)

    log_market_eval(route_decision="wait")
    return "wait"
```

---

## 27. Full Locked Config Object

```python
EXECUTION_CONFIG = {
    "strategy": {
        "market_type": "WNBA_moneyline",
        "venue": "Kalshi",
        "canonical_exposure": "selected_team_wins",
        "allowed_order_expressions": [
            "BUY_YES_selected_team_market",
            "BUY_NO_opponent_team_market",
        ],
        "entry_type": "pre_tipoff_only",
        "exit_policy": "hold_to_settlement",
        "forecast_stack": "Elo_prior_plus_XGBoost_correction",
    },

    "expansion_team_gate": {
        "enabled": True,
        "applies_to_true_expansion_teams_only": True,
        "min_completed_games_before_trading": 14,
        "first_tradable_game_number": 15,
        "block_passive_orders": True,
        "block_ioc_orders": True,
        "forecast_and_update_states_while_blocked": True,
        "use_min_completed_games_if_multiple_expansion_teams": True,
    },

    "side_normalization": {
        "require_side_mapping_confirmed": True,
        "require_complement_market_confirmed": True,
        "choose_route_by": "lowest_q_exec_all_in_subject_to_liquidity",
        "allow_route_splitting": True,
        "route_price_tie_threshold_ticks": 1,
        "log_model_side": True,
        "log_contract_side": True,
        "log_normalized_side": True,
    },

    "edge": {
        "edge_min": 0.05,
        "norm_edge_min": 0.25,
        "q_max_rule": "min(p_selected - 0.05, p_selected / 1.25)",
        "edge_check_price": "q_exec_all_in",
        "require_fee_adjusted_price": True,
    },

    "timing": {
        "monitor_start_hours_before_tip": 24,
        "trade_start_hours_before_tip": 17,
        "no_new_entry_after_hours_before_tip": 8,
        "cancel_passive_after_hours_before_tip": 8,
        "no_orders_after_hours_before_tip": 4,
        "reject_late_only_signals": True,
        "allow_prequalified_ticket_execution_until_T4": True,
    },

    "portfolio_sizing": {
        "kelly_fraction": 0.50,
        "kelly_formula": "0.5 * (p_selected - q_exec_all_in) / (1 - q_exec_all_in)",
        "max_market_exposure_pct": 0.15,
        "min_target_size_dollars": 25,
        "slate_exposure_cap": None,
        "cash_buffer_pct": 0.02,
        "target_position_dynamic": True,
    },

    "liquidity": {
        "max_visible_depth_participation": 0.25,
        "recent_volume_window_hours": 3,
        "max_recent_qualifying_volume_participation": 0.15,
        "cold_start_bankroll_cap": 0.01,
        "cold_start_visible_depth_participation": 0.15,
        "max_cumulative_qualifying_volume_share": 0.30,
        "exclude_self_volume": True,
        "use_current_qmax_for_historical_qualifying_volume": True,
        "use_global_combined_cumulative_cap_across_routes": True,
    },

    "cadence": {
        "poll_T24_to_T17": "15min",
        "poll_T17_to_T12": "5min",
        "poll_T12_to_T8": "2min",
        "poll_T8_to_T4": "5min",
        "poll_T4_to_tip": "15min",
        "event_driven_recompute": True,
        "event_driven_ordering": True,
        "normal_min_time_between_ioc_sweeps": "60s",
        "burst_min_time_between_ioc_sweeps": "15s",
        "burst_debounce_overrides_normal_debounce": True,
        "min_time_between_passive_updates": "5min",
    },

    "child_slicing": {
        "normal_max_ioc_child_order_pct": 0.025,
        "completion_max_ioc_child_order_pct": 0.030,
        "burst_max_ioc_child_order_pct": 0.050,
        "passive_child_fraction_of_allowed": 0.25,
        "max_passive_child_order_pct": 0.010,
        "min_child_order_dollars": 25,
        "urgency_multiplier_T17_to_T12": 1.00,
        "urgency_multiplier_T12_to_T8": 1.50,
        "urgency_multiplier_T8_to_T4": 2.00,
        "max_effective_remaining_opportunities": 12,
        "round_contracts_down": True,
        "round_prices_down_to_valid_tick": True,
    },

    "passive": {
        "enabled": True,
        "order_type": "post_only_limit",
        "allowed_start_hours_before_tip": 17,
        "allowed_end_hours_before_tip": 8,
        "min_spread_for_passive_ticks": 2,
        "price_rule": "min(route.best_bid + 1_tick, route.midpoint_rounded_down, q_max_tick - 1_tick)",
        "timeout_T17_to_T12": "15min",
        "timeout_T12_to_T8": "10min",
        "max_upward_reprices_per_passive_episode": 2,
        "min_time_between_passive_reprices": "5min",
        "passive_episode_cooldown_after_chase_limit": "10min",
        "require_cancel_confirmation_before_ioc": True,
        "reserve_open_passive_exposure": True,
    },

    "ioc": {
        "order_type": "limit_ioc",
        "limit_price": "q_max_tick",
        "main_window": "T-17h_to_T-8h",
        "prequalified_window": "T-8h_to_T-4h",
        "hard_stop": "T-4h",
        "normal_debounce": "60s",
        "burst_enabled": True,
        "burst_child_cap_pct": 0.050,
        "burst_depth_multiplier_trigger": 2.0,
        "burst_min_visible_depth_pct_bankroll": 0.03,
        "burst_refresh_threshold_rule": "max(25, 0.005 * bankroll, 0.10 * visible_after_last_order)",
        "burst_debounce": "15s",
        "max_burst_orders_per_5min": 3,
        "max_burst_total_per_5min_pct": 0.07,
    },

    "risk": {
        # Mandatory hard checks
        "max_market_data_staleness_seconds": 30,
        "max_model_snapshot_age_minutes": 60,
        "require_model_newer_than_required_inputs": True,
        "max_injury_data_age_minutes": 60,
        "max_orderbook_disconnect_seconds": 30,
        "require_market_open": True,
        "require_settlement_mapping_confirmed": True,
        "require_game_start_time_confirmed": True,
        "require_contract_side_confirmed": True,
        "require_complement_market_confirmed": True,
        "require_side_mapping_confirmed": True,

        # Toggleable daily/performance brakes: default log_only
        "enable_daily_realized_loss_stop": True,
        "daily_realized_loss_stop_pct": 0.10,
        "daily_realized_loss_stop_mode": "log_only",

        "enable_daily_mtm_loss_stop": True,
        "daily_mtm_loss_stop_pct": 0.15,
        "daily_mtm_loss_stop_mode": "log_only",

        "enable_consecutive_loss_soft_brake": True,
        "consecutive_loss_soft_brake": 4,
        "consecutive_loss_soft_brake_mode": "log_only",

        "enable_consecutive_loss_hard_brake": True,
        "consecutive_loss_hard_brake": 6,
        "consecutive_loss_hard_brake_mode": "log_only",

        "enable_conservative_mode": True,
        "conservative_max_market_exposure_pct": 0.12,
        "disable_burst_in_conservative_mode": True,

        "enable_calibration_brake": True,
        "calibration_window_trades": 30,
        "min_expected_edge_realization_ratio": 0.50,
        "calibration_brake_mode": "log_only",

        # Toggleable ops brakes: default enforce
        "enable_order_reject_brake": True,
        "max_order_rejects_per_hour": 5,
        "order_reject_brake_mode": "enforce",

        "enable_api_error_brake": True,
        "max_api_errors_per_10min": 10,
        "api_error_brake_mode": "enforce",

        "enable_position_mismatch_brake": True,
        "max_position_mismatch_dollars": 10,
        "position_mismatch_brake_mode": "enforce",
    },
}
```

---

## 28. Final Locked Behavioral Summary

```text
1. Monitor markets from T-24h.
2. Start trading at T-17h.
3. A canonical exposure must first qualify before T-8h.
4. Reject all late-only signals.
5. Stop creating new prequalified tickets after T-8h.
6. Cancel all passive orders at T-8h.
7. Allow only pre-T-8h qualified tickets to execute until T-4h.
8. Stop all orders after T-4h.
9. For true first-season expansion teams, forecast and update states from game 1 but block live trades until every expansion team in the game has completed at least 14 prior games.
10. First tradable game for a true expansion franchise is its 15th game.
11. Canonical exposure is selected team wins.
12. Use BUY YES selected-team market and/or BUY NO opponent market when side mapping is confirmed.
13. Split child volume across equivalent routes based on all-in price and liquidity capacity.
14. Use limit-IOC orders for core exposure.
15. Use passive post-only limits only as small early probes.
16. Never pay above q_max_tick all-in.
17. Check edge at fee-adjusted expected executable average price.
18. Use explicit binary half-Kelly formula.
19. Cap target market exposure at 15% of bankroll.
20. Use available-cash constraint with 2% cash buffer.
21. Do not use a slate exposure cap.
22. Recompute target_position_now every cadence.
23. Filled exposure is monotone non-decreasing; target may rise or fall.
24. Reserve open order exposure before placing new orders.
25. Apply rolling v1 liquidity cap every cadence and per route.
26. Apply global combined cumulative liquidity cap across equivalent routes.
27. Use normal IOC slicing by cadence and urgency.
28. Use burst IOC mode only when executable depth refreshes materially.
29. Burst debounce overrides normal debounce.
30. Round prices down to valid ticks and contract quantities down.
31. Performance brakes are toggleable and default to log-only.
32. Operational brakes are toggleable and default to enforce.
33. Hard market/data/side integrity checks are mandatory.
34. Log every decision, route, cap binder, order attempt, fill, rejection, expansion-team gate outcome, and settlement.
```

---

## 29. Design Rationale in One Paragraph

The engine is designed around the empirical finding that the strategy’s edge is strongest when the signal appears early, while late-only signals are structurally weaker and more exposed to adverse selection. Therefore, the system rejects late-only trades, builds exposure primarily from T-17h to T-8h, and allows only controlled execution of prequalified tickets until T-4h. True first-season expansion-team games are still forecasted and used to update Elo and feature state from game 1, but the trading layer is blocked until every expansion team in the game has completed at least 14 prior games, making the first tradable game the team’s 15th. Sizing begins with fee-adjusted half-Kelly but is capped at 15% of bankroll per canonical market exposure, while actual realized exposure is usually governed by rolling liquidity caps based on visible executable depth, recent qualifying-price volume, a small cold-start allowance, and cumulative qualifying-volume participation. Execution is not hardcoded to buy YES only: the bot targets canonical selected-team-wins exposure and may express it through BUY YES on the selected-team market or BUY NO on the opponent market when the complement mapping is confirmed. Passive orders are small early probes; core exposure is obtained through limit-IOC sweeps, with burst mode for short-lived executable liquidity. Performance brakes are toggleable and default to log-only, while operational integrity checks remain mandatory.
