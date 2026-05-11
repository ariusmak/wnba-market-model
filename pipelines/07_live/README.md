# 07 Live Pipeline Layout

Top-level scripts are shared infrastructure:

- `00_check_kalshi_auth.py` - Kalshi auth smoke test.
- `01_smoke_kalshi_market_data.py` - read-only market/orderbook smoke test.
- `06_passive_book_recorder.py` - book telemetry recorder.
- `07_game_telemetry.py` - log/fill/settlement summary.
- `08_append_year.py` through `12_build_gold_year.py` - live-season data refresh and gold/training CSV builders.
- `13_validate_production_artifacts.py` - hard contract checks for layer layout, gold schema, live-year player priors, and future-injury hygiene.
- `14_live_data_refresh.py` - production scheduler/runner that connects fixed daily refreshes, T-20 market/game-time refresh triggers, run-level audit packets, validation, and duplicate-run state.
- `15_live_daemon.py` - always-on process that polls/caches Kalshi markets, keeps daily-injury data fresh at least hourly, filters live working snapshots to active/open WNBA moneyline/team-wins markets only, detects market creations/updates, writes heartbeats, invokes `14_live_data_refresh.py` when refresh jobs are due, and invokes `16_execution_supervisor.py` to launch/supervise per-game route loops.
- `16_execution_supervisor.py` - one-process-per-game launcher. It finds upcoming gold feature rows, confirms active/open moneyline mapping, enforces local operator controls, avoids duplicate route-loop PIDs, and starts `canonical/05_run_route_entry_loop.py`.

`market_t20` jobs include a Kalshi preflight audit check. The scheduler prefers the daemon's cached snapshot, writes snapshot/mapping issues to `preflight_report.json`, and still lets the data refresh proceed. Execution remains blocked until active/open two-contract moneyline mapping is confirmed.

Hourly `injury` refresh jobs force-refresh the recent daily-injury window, rebuild injury/player-state silver, and regenerate live prediction packets for mapped open games from T-20 through T-8. Route loops watch their `data/live_features/<game_id>.csv` file and treat a refreshed probability as the new planning truth. If the model-selected side flips after exposure exists, the loop blocks new entry rather than creating offsetting exposure.

When a live-run game settles, the gold builder prefers a captured live prediction packet/feature row whose source as-of time is no later than T-8. This keeps the settled training row aligned to the latest conditions available before the no-new-entry boundary.

Canonical route-loop execution resolves sizing from Kalshi portfolio state before planning. By default, sizing bankroll follows `/portfolio/balance.portfolio_value` with fallback to `/portfolio/balance.balance`, while available-cash feasibility uses `/portfolio/balance.balance`. `--sizing-bankroll-override` supplies an imaginary sizing bankroll above or below Kalshi wealth; `--available-cash-override` separately supplies an imaginary cash cap.

Canonical route-loop execution reconciles Kalshi fills/positions on startup and during runtime. Startup seeds existing route exposure from Kalshi and handles open route orders: bot-owned orders are cancelled before planning, while unknown open orders block execution. Runtime mismatch feeds the planner brake.

Local operator controls live under `data/runs/live_control/`. Missing files mean all eligible games trade normally. Global auto-trade off, risk mode `kill`, or a per-game `abort` override blocks execution. The Streamlit webapp exposes this trade-default toggle, risk mode display, and per-game abort/clear controls.

Remote dashboard wiring is controlled explicitly with `--control-plane-mode`:

- `local-only` keeps execution on local JSON controls only.
- `supabase-shadow` reads Supabase controls and publishes heartbeat/route/order/equity tables, but intended orders are logged as skipped shadow events.
- `supabase-live` fails closed if Supabase controls cannot be read and rechecks the merged local+Supabase decision before every Kalshi order attempt.

Local JSON controls remain the final emergency brake in every mode; Supabase can only add restrictions or lower caps.

When available cash is binding across multiple otherwise eligible live tickets, cash allocation must use `src/srwnba/live/canonical/cash_priority.py`: exact marginal expected log growth per dollar, with current fills plus reserved open orders included as existing position cost.

Canonical route-loop execution writes a per-game ledger under `data/runs/live_games/<game_id>/` by default. It contains the prediction packet, confirmed market mapping, raw route orderbook snapshots, evaluated route quotes, portfolio sizing snapshots, execution plans, orders, fills, errors, summaries, and per-run session copies. `data/live_logs/<game_id>.route.jsonl` remains the flat event stream.

Daemon health files:
- `data/runs/live_daemon/health_latest.json` - current health verdict (`ok`, `degraded`, or `failed`) with checks and metrics.
- `data/runs/live_daemon/health.jsonl` - global health history.
- `data/runs/live_daemon/heartbeat_latest.json` - current heartbeat plus health summary.
- `data/runs/live_daemon/sessions/<run_id>/health.jsonl` - per-session health history.

Daemon health includes market polling, hourly injury freshness, market snapshot freshness, refresh-worker freshness, execution-supervisor freshness, active/open moneyline counts, and consecutive failure counters.

Loop-specific code is separated:

- `canonical/` - production-direction selected-team-wins route loop.
- `legacy/` - older explicit home/away YES-only loop, retained for tests/reference.
