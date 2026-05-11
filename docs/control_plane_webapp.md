# Control Plane Webapp

This document is the handoff for agents working on the Streamlit/Supabase control plane.

## Purpose

The webapp is a remote control room for the WNBA Kalshi trading worker. It is intentionally boring and deterministic:

1. show live worker/account/market state,
2. write audited kill/pause/risk commands,
3. never place or cancel Kalshi orders directly,
4. let the local Python trading worker enforce every command before touching Kalshi.

The dashboard also has a local desktop-control path for the current production worker. When `app.py` runs on the same machine/workspace as the worker, it writes `data/runs/live_control/operator_control.json` for global execution default/risk mode and `data/runs/live_control/game_overrides/<game_id>.json` for explicit per-game aborts. Missing local files mean all eligible games trade normally.

The dashboard is deployed from the private GitHub repo `ariusmak/wnba_kalshi_prod` with `app.py` as the Streamlit entrypoint.

## Files

- `app.py` - Streamlit dashboard UI.
- `supabase_io.py` - all Supabase reads/writes used by the dashboard and seed script.
- `sql/schema.sql` - canonical full schema for a fresh Supabase project.
- `sql/patch_worker_ack_status.sql` - adds `bot_heartbeat.last_control_seen_at` for worker command acknowledgment.
- `scripts/seed_fake_data.py` - fake data seeder for UI QA only.
- `.streamlit/secrets.example.toml` - secret names only. Do not put real secrets in git.

## Non-negotiable safety rules

- Do not import or call Kalshi clients from `app.py`, `supabase_io.py`, or SMS/dashboard code.
- Do not submit, cancel, or modify Kalshi orders from the dashboard.
- Do not treat dashboard controls as advisory. The worker must read Supabase controls before every order attempt.
- Do not invert the local operator-control default. Default is trade all eligible games unless global auto-trade is turned off, risk mode is `kill`, or the game is explicitly aborted.
- Do not commit `.env`, `.streamlit/secrets.toml`, private keys, Supabase secret keys, Kalshi keys, or Twilio tokens.
- Keep database prices/probabilities as decimals in `[0, 1]`, not cents.
- If adding a dangerous command, add an explicit mobile-safe confirmation gate.

## Runtime architecture

```text
phone / Streamlit dashboard
        |
        v
Supabase control tables
        |
        v
local Python trading worker
        |
        v
Kalshi
```

The online dashboard cannot read local files or local memory. Anything live must be published by the local worker to Supabase.

## Secrets

For local development, use ignored `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "YOUR_ROTATED_SERVICE_ROLE_KEY"
DASHBOARD_PASSWORD = "strong-dashboard-password"

TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_FROM_NUMBER = ""
ALLOWED_PHONE_NUMBER = ""
SMS_PIN = ""
```

For Streamlit Cloud, paste the same values into the app's Secrets UI. Do not commit them.

Use a rotated Supabase service-role/secret key. The publishable key is not enough because the server-side dashboard writes control commands and control rows.

## Tables The Dashboard Reads

- `control_state` - global mode, kill switch, trading flags, max exposure, shadow mode.
- `market_controls` - per-game manual override state.
- `control_commands` - immutable command audit log.
- `live_market_snapshots` - one current row per active game/canonical exposure.
  It includes the current model probability plus `model_prob_t20`,
  `model_prob_latest_pre_t8`, and `model_prob_change_t20_to_t8` so the
  per-game cards can show whether injury refreshes moved the price between
  the official T-20 packet and the T-8 no-new-entry boundary.
- `route_snapshots` - route-level smart-router comparison data.
- `order_events` - submit/fill/cancel/reject/skip audit stream.
- `closed_market_summaries` - settled-market summary rows.
- `equity_curve` - account/equity snapshots. The latest row powers Total NAV, Cash, Deployed, P&L, and Drawdown.
- `bot_heartbeat` - worker status, connection health, open order/position counts, and control acknowledgment.
- `system_alerts` - optional operational alerts.

## Commands

Dashboard commands are implemented in `supabase_io.GLOBAL_COMMAND_UPDATES` and `supabase_io.MARKET_COMMAND_UPDATES`.

Global commands:

- `KILL_BOT` - sets kill switch, disables trading, entries, IOC, passives, burst, and mode `killed`.
- `LAUNCH_BOT` - clears kill, enables trading/entries/orders/burst, mode `normal`, 15% max market exposure.
- `RESUME_ALL` - clears kill, enables trading/entries/orders/burst, mode `normal`.
- `PAUSE_ALL_NEW_ENTRIES` - disables new entries and sets mode `paused`.
- `CONSERVATIVE_MODE` - mode `conservative`, 12% max market exposure, disables burst.
- `NORMAL_RISK_MODE` - mode `normal`, 15% max market exposure, enables burst. It intentionally does not clear kill switch, unpause entries, or relaunch trading.
- `CANCEL_ALL_PASSIVES` - disables passive orders.
- `ENABLE_PASSIVES` - enables passive orders.

Market commands:

- `PAUSE_MARKET`
- `UNPAUSE_MARKET`
- `CANCEL_ENTRY`
- `CANCEL_MARKET_PASSIVES`
- `BLOCK_GAME`
- `FORCE_CONSERVATIVE_MARKET`
- `CLEAR_MARKET_CONTROLS`

`CANCEL_MARKET_PASSIVES` only disables/cancels passive resting orders for that
game. It does not block IOC sweeps if all other gates still pass.
`CLEAR_MARKET_CONTROLS` resets the remote per-game pause/cancel/block/passive
cancel/conservative fields back to normal. Local JSON abort files remain a
separate emergency brake and are not cleared by remote commands.

Every command should insert into `control_commands` and then update the target control table. If the target update fails, mark the command failed.

Local execution controls:

- Global file: `data/runs/live_control/operator_control.json`
- Per-game file: `data/runs/live_control/game_overrides/<game_id>.json`
- `auto_trade_enabled=true` and `risk_mode=normal` is the normal/default state.
- `auto_trade_enabled=false` blocks new execution globally.
- `risk_mode=kill` blocks new execution globally.
- Per-game `decision=abort` blocks that game until cleared back to `default`.
- The UI must display the current execution default mode and risk mode, and expose the global trade-default toggle plus per-game abort/clear buttons.

## Worker Integration Contract

The local worker should use Supabase as the bridge. Before every order attempt:

1. read fresh `control_state`,
2. read `market_controls` for the game,
3. read local operator control and per-game override files,
4. reconcile Kalshi positions/fills/open orders on startup and periodically during runtime,
5. enforce hard data integrity gates,
6. enforce timing gates,
7. enforce expansion-team gate,
8. enforce edge/size/liquidity gates,
9. only then submit/cancel Kalshi orders if allowed.

Cancellation is special: kill should block new orders, but must still allow cancelling passives.

Execution supports three explicit control-plane modes:

- `local-only` - use only local JSON controls. This is the compatibility/default mode for local diagnostics.
- `supabase-shadow` - read Supabase controls and publish worker/table state, but log intended orders as skipped shadow events instead of sending them to Kalshi.
- `supabase-live` - read Supabase controls before planning and again before every live order attempt. Missing/unreadable Supabase controls fail closed for new orders.

Local JSON controls remain an emergency brake in every mode. The effective worker decision is the most restrictive combination of local JSON plus Supabase global and per-game controls. Supabase can lower max market exposure, disable IOC/passive/burst orders, force conservative mode, or block entries, but it cannot loosen the locked v1.2 caps or override a local kill/abort.

The worker should continuously publish:

- `bot_heartbeat`
  - `last_seen_at`
  - `last_control_seen_at` after it reads the current `control_state.updated_at`
  - `current_mode`
  - `kalshi_connected`
  - `market_data_connected`
  - `database_connected`
  - `open_orders_count`
  - `open_positions_count`
  - `last_error`
- `equity_curve`
  - `equity_dollars` = total Kalshi NAV/equity
  - `cash_dollars` = available cash
  - `open_position_value_dollars` = deployed/marked open position value
  - `realized_pnl_dollars`
  - `drawdown_dollars`
- `live_market_snapshots`
- `route_snapshots`
- `order_events`
- `system_alerts` for important failures or stale data.

## Dashboard UX Features

The Control Room includes:

- manual Refresh button,
- freshness/worker-obedience banner,
- current mode banner,
- Kalshi account strip,
- confirmation-gated global controls,
- local execution-default/risk-mode controls,
- recent command strip,
- market search/filter,
- confirmation-gated market override buttons,
- confirmation-gated local abort/clear-abort buttons,
- mobile-friendly cards and stat grids.

The freshness banner marks stale/missing data based on current table timestamps. It will show worker obedience as unconfirmed until the worker writes `bot_heartbeat.last_control_seen_at`.

## Setup / Deployment

Fresh Supabase:

1. run `sql/schema.sql`,
2. run any patch SQL files if this is an older project,
3. optionally run `python scripts/seed_fake_data.py` for fake dashboard cards.

Existing Supabase used in this project:

1. run `sql/patch_worker_ack_status.sql` if `bot_heartbeat.last_control_seen_at` is missing,
2. run `sql/patch_market_controls_cancel_passives.sql` if `market_controls.cancel_passive_orders` is missing,
3. set Streamlit secrets in Streamlit Cloud,
4. deploy from private GitHub repo with entrypoint `app.py`.

## QA Checklist

Run before pushing dashboard changes:

```powershell
python -m compileall -q app.py supabase_io.py scripts\seed_fake_data.py
```

Run the Streamlit render test pattern from local dev if secrets are configured:

```powershell
python -c "import tomllib
from pathlib import Path
from streamlit.testing.v1 import AppTest
secrets=tomllib.loads(Path('.streamlit/secrets.toml').read_text(encoding='utf-8-sig'))
at=AppTest.from_file('app.py', default_timeout=30)
at.run()
at.text_input[0].set_value(secrets['DASHBOARD_PASSWORD']).run()
buttons=[b.label for b in at.button]
assert 'KILL BOT' in buttons and 'Normal Risk' in buttons
print('ok')"
```

Also confirm no secrets are staged or committed.
