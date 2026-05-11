# `game_franchise_style_profile` Specification

`game_franchise_style_profile_{year}_REGPST.csv` is the canonical production
style-profile feature family consumed by the gold builder. It is keyed by
current team identity plus `franchise_id`, so rebrands/relocations preserve
style priors where the franchise map says they should.

The production builder is:

```bash
python pipelines/03_features/26_build_franchise_style_profile.py --year <year>
```

The production gold projection reads this file directly. Do not point the
model-input builder back at legacy `game_team_style_profile` artifacts.

## Grain

One row per game/team before the game is played.

Required identity columns:

- `game_id`
- `game_ts`
- `game_date`
- `season`
- `team_id`
- `franchise_id`
- `opponent_team_id`
- `opponent_franchise_id`
- `is_home`
- `is_playoff`

## Canonical Model Features

These columns are projected into gold with `home_` / `away_` prefixes:

- `off_3pa_rate_pre`
- `def_3pa_allowed_pre`
- `off_2pa_rate_pre`
- `def_2pa_allowed_pre`
- `off_tov_pct_pre`
- `def_forced_tov_pre`

## Prior Rules

- Style priors carry over from the previous season when a mapped franchise
  existed in that previous season.
- New true expansion franchises start from league-level priors.
- The state key is `franchise_id`, not raw `team_id`.
- San Antonio Stars and Las Vegas Aces continuity is represented by
  `FR_LVA_ACES`.

## Production Contract

- Output lives in `data/silver_plus/`.
- `prior_source=league_init` for a team that existed in the prior season is a
  production QA warning.
- Downstream gold output must use the canonical schema in
  `src/srwnba/util/model_schema.py`.
