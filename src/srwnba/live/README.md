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
