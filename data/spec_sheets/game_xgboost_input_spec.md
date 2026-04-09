# `game_xgboost_input` Specification

This document specifies the **final one-row-per-game XGBoost input table**.

It is the table used to train and score the XGBoost correction model in:

\[
\operatorname{logit}(p_{raw}) = \operatorname{logit}(p_{elo}) + g(x)
\]

where:

- `base_margin = logit(p_elo)`
- `x` = all non-Elo feature columns defined below

This spec uses the actual source tables / column names from the current pipeline:

1. `elo_franchise_team_game_{season}_REGPST.csv`
2. `game_franchise_recent_form_{season}_REGPST.csv`
3. `game_franchise_style_profile_{season}_REGPST.csv`
4. `game_team_schedule_context_{season}_REGPST.csv`
5. `game_team_player_{season}_REGPST.csv`

This document includes:

- row grain
- metadata columns
- exact source of each feature block
- exact join logic
- exact player-slot ordering logic
- exact final column order
- which columns are model inputs vs debug metadata
- null-handling rules

---

# 1. Purpose of this table

`game_xgboost_input` is the **final supervised-learning table** used for the XGBoost correction layer.

Each row represents a single game and contains:

- metadata / join identifiers
- target (for training)
- Elo base margin
- home player-slot block
- away player-slot block
- home/away recent-form block
- home/away style-profile block
- home/away schedule / travel block

This table is **one row per game**.

---

# 2. Row grain / primary key

## Table name
`game_xgboost_input`

## Grain
One row per:

\[
\boxed{game\_id}
\]

## Primary key
- `game_id`

There is exactly one row per game.

---

# 3. Source tables

This final table is built from the following upstream/downstream source tables.

## 3.1 Elo source
`elo_franchise_team_game_{season}_REGPST.csv`

Sample columns:
- `season_year`
- `scheduled`
- `game_id`
- `team_id`
- `franchise_id`
- `opponent_team_id`
- `opponent_franchise_id`
- `is_home`
- `elo_pre`
- `elo_post`
- `p_win_pre`

Use this source for:
- home / away team identifiers
- home / away franchise identifiers
- home / away Elo values (metadata/debug)
- `p_elo`
- `base_margin`

## 3.2 Recent form source
`game_franchise_recent_form_{season}_REGPST.csv`

Sample columns:
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
- `net_rtg_ewma_pre`
- `efg_ewma_pre`
- `tov_pct_ewma_pre`
- `orb_pct_ewma_pre`
- `ftr_ewma_pre`

Use this source for the recent-form block.

## 3.3 Style source
`game_franchise_style_profile_{season}_REGPST.csv`

Sample columns:
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
- `off_3pa_rate_pre`
- `def_3pa_allowed_pre`
- `off_2pa_rate_pre`
- `def_2pa_allowed_pre`
- `off_tov_pct_pre`
- `def_forced_tov_pre`

Use this source for the style-profile block.

## 3.4 Schedule / travel source
`game_team_schedule_context_{season}_REGPST.csv`

Sample columns:
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
- `days_rest_pre`
- `is_b2b_pre`
- `games_last_4_days_pre`
- `games_last_7_days_pre`
- `origin_city_pre`
- `current_city_pre`
- `travel_miles_pre`
- `timezone_shift_hours_pre`

Use this source for the schedule / travel block.

## 3.5 Player slot source
`game_team_player_{season}_REGPST.csv`

Sample columns:
- `game_id`
- `game_ts`
- `game_date`
- `season`
- `team_id`
- `opponent_team_id`
- `is_home`
- `is_playoff`
- `player_id`
- `player_name`
- `listed_on_game_summary_flag`
- `state_asof_ts`
- `m_ewma_pre`
- `q_pre`
- `strength_pre`
- `days_since_first_report_pre`
- `days_since_last_dnp_pre`
- `consec_dnps_pre`
- `played_last_game_pre`
- `minutes_last_game_pre`
- `days_since_last_played_pre`
- `injury_present_flag_pre`
- `played_in_game`
- `minutes_in_game`
- `not_playing_reason`
- `not_playing_description`

Use this source for the player-slot block.

---

# 4. Build order / assembly logic

The final table should be built in the following order.

---

## Step 1. Start from the Elo source and create one row per game

The Elo table has **2 rows per game**:
- one home team row
- one away team row

For each `game_id`, split the Elo rows into:

### Home Elo row
where:
- `is_home = 1`

### Away Elo row
where:
- `is_home = 0`

Then create one final game row with:

- `home_team_id = team_id from home Elo row`
- `away_team_id = team_id from away Elo row`
- `home_franchise_id = franchise_id from home Elo row`
- `away_franchise_id = franchise_id from away Elo row`
- `game_ts = scheduled` (or normalized timestamp)
- `game_date`
- `season`

## Step 2. Create Elo base-margin fields
From the home Elo row:

- `home_elo_pre = elo_pre`
- `away_elo_pre = elo_pre from away row`
- `p_elo = p_win_pre` from the home Elo row

Then compute:

\[
base\_margin = \log\left(\frac{p_{elo}}{1-p_{elo}}\right)
\]

This `base_margin` is **not** an ordinary XGBoost feature column; it is passed separately into XGBoost.

### Recommended safety clip
Before logit transform, clip `p_elo` to something like:

\[
p_{elo}^{clipped} \in [10^{-6}, 1-10^{-6}]
\]

to avoid infinite logits.

---

## Step 3. Join recent-form block
Join `game_franchise_recent_form` twice:

- once for the home team row
- once for the away team row

Recommended join keys:

- `game_id`
- `team_id`

(or equivalently `game_id` + `franchise_id` if you standardize on franchise keying; but use the actual available keys consistently)

From the home row, bring in:
- `home_net_rtg_ewma_pre`
- `home_efg_ewma_pre`
- `home_tov_pct_ewma_pre`
- `home_orb_pct_ewma_pre`
- `home_ftr_ewma_pre`

From the away row, bring in:
- `away_net_rtg_ewma_pre`
- `away_efg_ewma_pre`
- `away_tov_pct_ewma_pre`
- `away_orb_pct_ewma_pre`
- `away_ftr_ewma_pre`

---

## Step 4. Join style-profile block
Join `game_franchise_style_profile` twice:

- once for the home team row
- once for the away team row

Bring in:

### Home
- `home_off_3pa_rate_pre`
- `home_def_3pa_allowed_pre`
- `home_off_2pa_rate_pre`
- `home_def_2pa_allowed_pre`
- `home_off_tov_pct_pre`
- `home_def_forced_tov_pre`

### Away
- `away_off_3pa_rate_pre`
- `away_def_3pa_allowed_pre`
- `away_off_2pa_rate_pre`
- `away_def_2pa_allowed_pre`
- `away_off_tov_pct_pre`
- `away_def_forced_tov_pre`

---

## Step 5. Join schedule / travel block
Join `game_team_schedule_context` twice:

- once for the home team row
- once for the away team row

Bring in:

### Home
- `home_days_rest_pre`
- `home_is_b2b_pre`
- `home_games_last_4_days_pre`
- `home_games_last_7_days_pre`
- `home_travel_miles_pre`
- `home_timezone_shift_hours_pre`

### Away
- `away_days_rest_pre`
- `away_is_b2b_pre`
- `away_games_last_4_days_pre`
- `away_games_last_7_days_pre`
- `away_travel_miles_pre`
- `away_timezone_shift_hours_pre`

### Optional schedule/travel metadata
If you want it in the final table for debugging only, you may also carry:
- `home_origin_city_pre`
- `away_origin_city_pre`
- `home_current_city_pre`
- `away_current_city_pre`

These are **not** recommended model features.

---

## Step 6. Build the player-slot block from `game_team_player`

### 6.1 Ranking rule
For each `(game_id, team_id)` group in `game_team_player`, sort players by:

1. `strength_pre` descending
2. `m_ewma_pre` descending
3. `q_pre` descending
4. `player_id` ascending

This ranking rule is locked and deterministic.

### 6.2 Slot assignment
After sorting, assign slots:

- strongest player → `p1`
- second strongest → `p2`
- ...
- twelfth strongest → `p12`

Do this separately for:
- home team
- away team

### 6.3 Important note
`strength_pre` is used only for:
- ranking
- debugging / audit

It is **not** one of the locked model input features.

---

# 5. Locked player-level model inputs per slot

For each player slot, the model gets exactly these **9** features:

1. `m_ewma_pre`
2. `q_pre`
3. `days_since_first_report_pre`
4. `days_since_last_dnp_pre`
5. `consec_dnps_pre`
6. `played_last_game_pre`
7. `minutes_last_game_pre`
8. `days_since_last_played_pre`
9. `injury_present_flag_pre`

These are the **only** locked player-slot model features.

---

# 6. Player-slot metadata/debug columns

In addition to the 9 model features, include these **debug / audit only** columns for each slot:

- `player_id`
- `player_name`
- `strength_pre`

These are useful for:
- confirming slot ordering
- debugging roster logic
- inspecting which player each slot actually corresponds to

They should **not** be part of the XGBoost feature matrix.

---

# 7. Missing-slot handling

You decided that if a team has fewer than 12 players in the game summary, the unused player slots should be **NULL**, not zero.

This applies to:
- all 9 player-level input features
- all player-slot debug fields (`player_id`, `player_name`, `strength_pre`)

## Why NULL, not 0
Because “no player exists in this slot” is not the same thing as:
- a real player with `m_ewma = 0`
- a real player with `injury_present_flag = 0`
- a real player with `minutes_last_game = 0`

So for missing slots, use NULL consistently.

XGBoost can handle missing values.

---

# 8. Final column groups

This section defines the exact final column ordering.

## 8.1 Metadata columns (not model inputs)

These should be present in the final table for joining, auditing, and backtesting:

- `game_id`
- `game_ts`
- `game_date`
- `season`
- `is_playoff`
- `home_team_id`
- `away_team_id`
- `home_franchise_id`
- `away_franchise_id`
- `home_elo_pre`
- `away_elo_pre`
- `p_elo`
- `base_margin`
- `home_win` (training target only; not a model feature)

### Optional extra metadata
If desired:
- `elo_source_scheduled`
- source filenames / bronze file references
- schedule/travel origin-city metadata
- player-slot debug metadata

---

## 8.2 Player-slot block

For each home slot \(k = 1, \dots, 12\):

### Debug / metadata only
- `home_p{k}_player_id`
- `home_p{k}_player_name`
- `home_p{k}_strength_pre`

### Locked model inputs
- `home_p{k}_m_ewma_pre`
- `home_p{k}_q_pre`
- `home_p{k}_days_since_first_report_pre`
- `home_p{k}_days_since_last_dnp_pre`
- `home_p{k}_consec_dnps_pre`
- `home_p{k}_played_last_game_pre`
- `home_p{k}_minutes_last_game_pre`
- `home_p{k}_days_since_last_played_pre`
- `home_p{k}_injury_present_flag_pre`

Then do the same for each away slot \(k = 1, \dots, 12\):

### Debug / metadata only
- `away_p{k}_player_id`
- `away_p{k}_player_name`
- `away_p{k}_strength_pre`

### Locked model inputs
- `away_p{k}_m_ewma_pre`
- `away_p{k}_q_pre`
- `away_p{k}_days_since_first_report_pre`
- `away_p{k}_days_since_last_dnp_pre`
- `away_p{k}_consec_dnps_pre`
- `away_p{k}_played_last_game_pre`
- `away_p{k}_minutes_last_game_pre`
- `away_p{k}_days_since_last_played_pre`
- `away_p{k}_injury_present_flag_pre`

---

## 8.3 Recent-form block (home first, then away)

### Home
- `home_net_rtg_ewma_pre`
- `home_efg_ewma_pre`
- `home_tov_pct_ewma_pre`
- `home_orb_pct_ewma_pre`
- `home_ftr_ewma_pre`

### Away
- `away_net_rtg_ewma_pre`
- `away_efg_ewma_pre`
- `away_tov_pct_ewma_pre`
- `away_orb_pct_ewma_pre`
- `away_ftr_ewma_pre`

---

## 8.4 Style-profile block (home first, then away)

### Home
- `home_off_3pa_rate_pre`
- `home_def_3pa_allowed_pre`
- `home_off_2pa_rate_pre`
- `home_def_2pa_allowed_pre`
- `home_off_tov_pct_pre`
- `home_def_forced_tov_pre`

### Away
- `away_off_3pa_rate_pre`
- `away_def_3pa_allowed_pre`
- `away_off_2pa_rate_pre`
- `away_def_2pa_allowed_pre`
- `away_off_tov_pct_pre`
- `away_def_forced_tov_pre`

---

## 8.5 Schedule / travel block (home first, then away)

### Home
- `home_days_rest_pre`
- `home_is_b2b_pre`
- `home_games_last_4_days_pre`
- `home_games_last_7_days_pre`
- `home_travel_miles_pre`
- `home_timezone_shift_hours_pre`

### Away
- `away_days_rest_pre`
- `away_is_b2b_pre`
- `away_games_last_4_days_pre`
- `away_games_last_7_days_pre`
- `away_travel_miles_pre`
- `away_timezone_shift_hours_pre`

---

# 9. Final model-input order

The final column order should be:

1. **metadata block**
2. **home player slots (1–12)**
3. **away player slots (1–12)**
4. **recent form block**
5. **style block**
6. **rest / travel block**

Within the non-player blocks, the order is:
- home block first
- away block second

This matches the organization choice you locked earlier.

---

# 10. Which columns go into XGBoost

## Pass separately as `base_margin`
- `base_margin`

## Use as XGBoost ordinary features
Everything below **except** metadata / debug columns:

### Include
- the 9 player-level features for each home and away slot
- recent form features
- style-profile features
- schedule / travel features

### Do not include
- `game_id`
- `game_ts`
- `game_date`
- `season`
- `is_playoff`
- `home_team_id`
- `away_team_id`
- `home_franchise_id`
- `away_franchise_id`
- `home_elo_pre`
- `away_elo_pre`
- `p_elo`
- `home_win`
- all player-slot debug metadata:
  - `player_id`
  - `player_name`
  - `strength_pre`

## Important
Do **not** include `strength_pre` as a model feature, because it is deterministic from `m_ewma_pre * q_pre`.

---

# 11. Target

For training, the target is:

- `home_win`

Definition:

\[
home\_win =
\begin{cases}
1 & \text{if home team wins the game} \\
0 & \text{otherwise}
\end{cases}
\]

This target should be sourced from the official game result / summary outcome, not inferred from any model-derived columns.

---

# 12. Feature counts

## Player-slot block
Each team has:
- 12 slots
- 9 model features per slot

So:
\[
12 \times 9 = 108
\]
per team, and:
\[
216
\]
total player-level model features.

## Team blocks
- recent form: 10
- style: 12
- rest / travel: 12

Total non-player features:
\[
10 + 12 + 12 = 34
\]

## Total ordinary XGBoost features
\[
216 + 34 = 250
\]

### Plus
- `base_margin` (passed separately, not counted inside `X`)

---

# 13. Final summary

The final XGBoost input table is:

- **one row per game**
- `base_margin = logit(p_elo)`
- ordinary feature matrix = 250 columns:
  - 216 player-slot features
  - 10 recent-form features
  - 12 style-profile features
  - 12 schedule / travel features

with:
- metadata columns preserved for auditability
- player-slot ordering determined by `strength_pre`
- `strength_pre` included only for debugging / ordering validation
- NULLs for missing player slots

This is the final locked specification for the XGBoost input layer.
