---
name: live-deployment-constraints
description: frozen production trading and data constraints for 2026 live deployment
type: project
---

Live WNBA Kalshi trading config is frozen for the 2026 season.

Execution:
- Canonical production execution lives in `src/srwnba/live/canonical/`.
- The v1.2 planner in `src/srwnba/live/canonical/v1_2.py` owns timing, cadence, signal memory, liquidity caps, passive probes, child slicing, burst logic, and brakes.
- Runtime wiring lives in `src/srwnba/live/canonical/route_entry_loop.py`.
- Legacy YES-only execution in `src/srwnba/live/legacy/` is for comparison/testing only.

Forecasting:
- `FinalModel.predict(df)["p_home"]` is the production probability accessor.
- Use `p_raw`; no Platt calibrator.
- Hyperparameters come only from `config/final_hyperparams.py`.
- `src/srwnba/util/model_schema.py` defines the strict 160 ordinary model features.

Data:
- Bronze: immutable per-call payloads promoted only after staged validation.
- Silver: chronological facts and daily/as-of state.
- Silver plus: game-as-of feature families.
- Gold: final 160-feature model matrix.
- `player_state_history` belongs in `data/silver`.
- `game_team_player` belongs in `data/silver_plus`.

Core trading rules:
- Bankroll reference: `$5,000`.
- Edge gates: absolute edge at least `0.05`; normalized edge at least `0.25`.
- Sizing: fee-adjusted half-Kelly, capped by v1.2 exposure, liquidity, and cash constraints.
- Entry: pre-tipoff only under v1.2 windows.
- Exit: hold to settlement; no sell/reduce orders.
- Venue: Kalshi only.
- Expansion teams: Toronto Tempo and Portland Fire are forecasted from game 1 but trading is blocked until every first-season expansion team in the game has at least 14 completed prior games.

Production data timing:
- Settled-history refresh: 02:30 ET.
- Backup/correction refresh: 09:00 ET.
- Official probability packet: T-20h.
- Manual-review deadline / earliest automated trading eligibility: T-18h.
