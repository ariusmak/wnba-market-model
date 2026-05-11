# `game_franchise_recent_form` Specification

`game_franchise_recent_form_{year}_REGPST.csv` is the canonical production
recent-form feature family consumed by the gold builder. It is keyed by current
team identity plus `franchise_id`, so rebrands/relocations preserve continuity
where the franchise map says they should.

The production builder is:

```bash
python pipelines/03_features/28_build_franchise_recent_form.py --year <year>
```

The production gold projection reads this file directly. Do not point the
model-input builder back at legacy `game_team_recent_form` artifacts.

## Grain

One row per game/team before the game is played.

Required identity columns:

- `season`
- `game_id`
- `game_ts`
- `game_date`
- `team_id`
- `franchise_id`
- `opponent_team_id`
- `opponent_franchise_id`
- `is_home`
- `is_playoff`

## Canonical Model Features

These columns are projected into gold with `home_` / `away_` prefixes:

- `net_rtg_ewma_pre`
- `efg_ewma_pre`
- `tov_pct_ewma_pre`
- `orb_pct_ewma_pre`
- `ftr_ewma_pre`

Use `net_rtg_ewma_pre`; `net_rating_ewma_pre` is a stale research-era name and
must not appear in production gold.

## State Rules

- EWMA half-life is locked at 7 games.
- Recent form resets each season.
- The state key is `franchise_id`, not raw `team_id`.
- San Antonio Stars and Las Vegas Aces continuity is represented by
  `FR_LVA_ACES`.

## Production Contract

- Output lives in `data/silver_plus/`.
- Downstream gold output must use the canonical schema in
  `src/srwnba/util/model_schema.py`.
- Production validation should fail if a current team with prior-season history
  falls back to a raw team id because the franchise map is stale.
