# `game_xgboost_input` Production Specification

This document specifies the final one-row-per-game XGBoost input table used by `FinalModel`.

The model computes:

```text
logit(p_raw) = logit(p_elo) + g(x)
```

where `base_margin = logit(p_elo)` is passed separately and `x` is the locked 160-column ordinary feature matrix.

## 1. Production Separation

The production pipeline has two distinct player-related artifacts:

1. `data/silver_plus/game_team_player_{year}_REGPST.csv`
   - Full feature-building/audit store.
   - One row per `(game_id, team_id, player_id)`.
   - Tracks every listed player for a team/game.
   - May contain `player_id`, `player_name`, `strength_pre`, injury text, minutes-in-game, and other audit columns.
   - Used to preserve chronology, debug roster availability, and rank players for the model projection.

2. `data/gold/game_xgboost_input_{year}_REGPST.csv`
   - Narrow model-input projection.
   - One row per `game_id`.
   - Contains metadata plus exactly the top 7 strength-ranked slots per side and only the 9 locked model features per slot.
   - Must not contain p8-p12 columns, player IDs/names, `strength_pre`, origin/current city debug fields, injury text, minutes-in-game, or any research/debug columns.

The canonical production schema is implemented in `src/srwnba/util/model_schema.py`. Gold builders and `FinalModel` must import that schema rather than rebuilding column lists independently.

## 2. Row Grain

Table name:

```text
game_xgboost_input
```

Primary key:

```text
game_id
```

There is exactly one row per game.

## 3. Source Tables

The gold model-input table is built from:

- `elo_franchise_team_game_{year}_REGPST.csv`
- `game_franchise_recent_form_{year}_REGPST.csv`
- `game_franchise_style_profile_{year}_REGPST.csv`
- `game_team_schedule_context_{year}_REGPST.csv`
- `game_team_player_{year}_REGPST.csv`

## 4. Metadata Columns

Gold must begin with these metadata/target columns in this order:

```text
game_id
game_ts
game_date
season
is_playoff
home_team_id
away_team_id
home_franchise_id
away_franchise_id
home_elo_pre
away_elo_pre
p_elo
base_margin
home_win
```

`home_win` is the training target and is not an ordinary model feature. `base_margin` is passed separately to XGBoost and is not counted in the 160 ordinary features.

## 5. Player Slot Construction

For each `(game_id, team_id)` group in `game_team_player`, sort players by:

1. `strength_pre` descending
2. `m_ewma_pre` descending
3. `q_pre` descending
4. `player_id` ascending

Assign only the top 7 sorted players to slots `p1` through `p7`.

This is done separately for the home team and away team.

`strength_pre` is used for ranking only. It is deterministic from role and quality state and must not be emitted into the gold model-input table.

## 6. Locked Player Features Per Slot

For each side and each slot `p1` through `p7`, emit exactly these 9 features:

```text
m_ewma_pre
q_pre
days_since_first_report_pre
days_since_last_dnp_pre
consec_dnps_pre
played_last_game_pre
minutes_last_game_pre
days_since_last_played_pre
injury_present_flag_pre
```

Column names are:

```text
home_p{k}_{feature}
away_p{k}_{feature}
```

for `k = 1..7`.

If a team has fewer than 7 listed players, missing slot feature values remain null. The production validator should flag this for review because normal WNBA game summaries should provide enough listed players.

## 7. Team Blocks

Recent-form features, emitted as home/away pairs:

```text
net_rtg_ewma_pre
efg_ewma_pre
tov_pct_ewma_pre
orb_pct_ewma_pre
ftr_ewma_pre
```

Style-profile features, emitted as home/away pairs:

```text
off_3pa_rate_pre
def_3pa_allowed_pre
off_2pa_rate_pre
def_2pa_allowed_pre
off_tov_pct_pre
def_forced_tov_pre
```

Schedule/travel features, emitted as home/away pairs:

```text
days_rest_pre
is_b2b_pre
games_last_4_days_pre
games_last_7_days_pre
travel_miles_pre
timezone_shift_hours_pre
```

Do not emit `origin_city_pre` or `current_city_pre` into gold. Those are audit inputs inside the schedule-context table only.

## 8. Ordinary Feature Count

Player features:

```text
7 slots * 9 features * 2 teams = 126
```

Non-player features:

```text
recent form: 5 features * 2 teams = 10
style:       6 features * 2 teams = 12
schedule:   6 features * 2 teams = 12
```

Total ordinary XGBoost features:

```text
126 + 10 + 12 + 12 = 160
```

## 9. Column Order

The complete gold column order is:

1. metadata columns listed in Section 4
2. all home player-slot features for `p1..p7`
3. all away player-slot features for `p1..p7`
4. recent-form home/away pairs
5. style-profile home/away pairs
6. schedule/travel home/away pairs

The authoritative implementation is `GOLD_MODEL_INPUT_COLS` in `src/srwnba/util/model_schema.py`.

## 10. Columns That Must Not Enter Gold

Gold must not include:

- `home_p8_*` through `home_p12_*`
- `away_p8_*` through `away_p12_*`
- `*_player_id`
- `*_player_name`
- `*_strength_pre`
- `*_origin_city_pre`
- `*_current_city_pre`
- injury descriptions or status text
- actual in-game minutes/results fields other than the `home_win` target
- any notebook/debug/research columns

## 11. Validation

After building gold, run:

```bash
python pipelines/07_live/13_validate_production_artifacts.py --year <year>
```

The validator must confirm:

- exactly 160 ordinary feature columns
- exact `GOLD_MODEL_INPUT_COLS` order
- no stale p8-p12/debug columns
- non-cold top player state for the live year
- full all-player silver store exists separately
- no future daily-injury placeholders in canonical bronze

This is the locked production specification for the XGBoost input layer.
