# Live Package Layout

- `canonical/` contains the production-direction execution path:
  market mapping, route-level quote evaluation, expansion gate, portfolio
  sizing, operator controls, Kalshi reconciliation, and the canonical
  selected-team-wins route loop.
- `legacy/` contains the older explicit home/away YES-only loop and its
  pure sweep planner. Keep it for comparison and regression tests.
- `common.py` contains shared Kalshi orderbook and fee helpers used by both
  paths.

New execution work should go in `canonical/` unless it is explicitly
maintaining the legacy loop.

Canonical live execution is trade-by-default when local operator-control
files are missing. Use `operator_control.py` for the global trade-default /
risk-mode file and per-game abort overrides, and `reconciliation.py` for
startup/runtime Kalshi fills, positions, and open-order recovery.

Remote webapp control wiring lives in `control_plane.py`. Route loops default
to `local-only`, can be rehearsed with `supabase-shadow`, and should only use
`supabase-live` once Supabase secrets, dashboard controls, and worker
acknowledgment are verified. Local JSON controls remain the emergency brake in
all modes.
Remote per-game passive cancellation is represented by
`market_controls.cancel_passive_orders`; it disables/cancels passive resting
orders for that game while leaving IOC eligibility to the other locked gates.
The live daemon publishes a Supabase `bot_heartbeat` acknowledgment in remote
control-plane modes, so the dashboard can confirm global commands during quiet
periods before route workers are launched.

Live feature rows are mutable until the T-8 no-new-entry boundary. The daemon's
hourly injury refresh rewrites `data/live_features/<game_id>.csv` for mapped
open games between T-20 and T-8, and the route loop reloads that file so the
latest probability becomes the planning truth. If the model-selected side flips
after exposure exists, the loop blocks new entry rather than creating offsetting
exposure.
