# `game_team_style_profile` Specification

This document specifies the **team-level stylistic tendency / matchup-profile table** that will be built at the **game-team** level.

It is written as a handoff document for feature engineering / coding.

It is explicit about:

- row grain
- time semantics
- primary key
- exact feature definitions
- exact formulas
- update / build logic
- initialization / prior rules
- why the design is structured this way

---

# 1. Purpose of this table

`game_team_style_profile` stores the **pregame team style / tactical tendency state** for one team in one game.

Each row answers the question:

> “What were this team’s **current-season style tendencies** before this game?”

This table is not meant to represent short-run form.  
It is meant to represent **mean current-season tendencies** such as:

- how much the team likes threes
- how much it allows threes
- how 2-point heavy it is
- how 2-point heavy its opponents are against it
- how turnover-prone it is
- how much turnover pressure it creates

These features are later joined into the final game-level model input as **raw team-level inputs**, not pre-collapsed deltas.

---

# 2. Locked design choices

The following design choices are already locked.

## 2.1 Use raw per-team style stats, not pre-collapsed mismatch deltas
We explicitly decided to feed the model **6 raw team profile stats per team**, rather than only 3 hand-engineered mismatch deltas.

This gives the model more information and lets XGBoost learn the relevant interactions itself.

## 2.2 Use current-season season-to-date values
We explicitly decided **not** to use EWMA for these style features.

Reason: these variables are intended to represent a team’s **current-season mean tendencies**, not short-run form.

## 2.3 Hard seasonal reset, except for game 1 initialization
We explicitly decided **not** to roll these style features straight through seasons.

However, for the **first game of a season**, we use the team’s **final value from the previous season** as the pregame initialization.

After game 1, all subsequent values are computed using **pure current-season season-to-date data only**.

So the design is:

- game 1 pregame value = previous-season final value
- games 2+ pregame value = current-season season-to-date value through the previous game

## 2.4 Use per-team inputs later, not only deltas
The downstream model will later receive:

- home team style profile
- away team style profile

separately.

We explicitly chose **not** to reduce them only to home-away deltas, because absolute team levels may matter in addition to differences.

---

# 3. Table name and grain

## Table name
`game_team_style_profile`

## Grain
One row per:

\[
(\text{game\_id},\ \text{team\_id})
\]

meaning:
- one row for the home team in the game
- one row for the away team in the game

So every game contributes exactly **2 rows**.

## Primary key
\[
\boxed{(\text{game\_id},\ \text{team\_id})}
\]

---

# 4. Required columns

## A. Game context
1. `game_id`
2. `game_ts`
3. `game_date`
4. `season`
5. `team_id`
6. `opponent_team_id`
7. `is_home`
8. `is_playoff`

## B. Pregame style / tendency features
9. `off_3pa_rate_pre`
10. `def_3pa_allowed_pre`
11. `off_2pa_rate_pre`
12. `def_2pa_allowed_pre`
13. `off_tov_pct_pre`
14. `def_forced_tov_pre`

## C. Optional audit / debug columns
These are not required by the final model, but they are useful:

15. `games_played_before_game`
16. `prior_source`
17. `last_completed_game_id`
18. `last_completed_game_ts`

---

# 5. The 6 locked stylistic features

These are the exact 6 team-level stylistic inputs.

## 5.1 `off_3pa_rate_pre`

### Purpose
Measures how three-point oriented the team’s offense has been this season.

### Definition
For a set of current-season games up to but not including the current game:

\[
off\_3pa\_rate
=
\frac{\sum 3PA}{\sum FGA}
\]

where:
- `3PA` = team 3-point attempts
- `FGA` = team field goal attempts

### Interpretation
Higher means the team takes a larger share of its shots from three.

---

## 5.2 `def_3pa_allowed_pre`

### Purpose
Measures how much the team’s defense allows opponent three-point volume.

### Definition
For current-season games up to but not including the current game:

\[
def\_3pa\_allowed
=
\frac{\sum Opp3PA}{\sum OppFGA}
\]

where:
- `Opp3PA` = opponent 3-point attempts
- `OppFGA` = opponent field goal attempts

### Interpretation
Higher means the defense allows a larger share of opponent shots from three.

---

## 5.3 `off_2pa_rate_pre`

### Purpose
This is our current proxy for interior / non-three shot tendency.

### Definition
For current-season games up to but not including the current game:

\[
off\_2pa\_rate
=
\frac{\sum 2PA}{\sum FGA}
=
\frac{\sum (FGA - 3PA)}{\sum FGA}
\]

where:
- `2PA = FGA - 3PA`

### Interpretation
Higher means the offense takes a larger share of its shots from two.

### Important note
This is a **2PA mix proxy**, not a true at-rim rate.  
We explicitly accepted this proxy instead of a more granular shot-location variable for v1.

---

## 5.4 `def_2pa_allowed_pre`

### Purpose
Measures how much the defense allows opponents to take two-point shots.

### Definition
For current-season games up to but not including the current game:

\[
def\_2pa\_allowed
=
\frac{\sum Opp2PA}{\sum OppFGA}
=
\frac{\sum (OppFGA - Opp3PA)}{\sum OppFGA}
\]

### Interpretation
Higher means the defense allows a larger share of opponent shots from two.

### Important note
This is the defensive analog of the 2PA mix proxy, not a true at-rim-allowed rate.

---

## 5.5 `off_tov_pct_pre`

### Purpose
Measures how turnover-prone the team’s offense has been this season.

### Definition
For current-season games up to but not including the current game:

\[
off\_tov\_pct
=
\frac{\sum TO}{\sum (FGA + 0.44 \cdot FTA + TO)}
\]

where:
- `TO` = team turnovers
- `FGA` = team field goal attempts
- `FTA` = team free throw attempts

### Interpretation
Lower is better offensively:
- higher values mean the team loses a larger share of possessions to turnovers

---

## 5.6 `def_forced_tov_pre`

### Purpose
Measures how much turnover pressure the team’s defense creates.

### Definition
For current-season games up to but not including the current game:

\[
def\_forced\_tov
=
\frac{\sum OppTO}{\sum (OppFGA + 0.44 \cdot OppFTA + OppTO)}
\]

where:
- `OppTO` = opponent turnovers
- `OppFGA` = opponent field goal attempts
- `OppFTA` = opponent free throw attempts

### Interpretation
Higher is better defensively:
- higher values mean the defense forces more opponent possessions to end in turnovers

---

# 6. Why these 6 features

These 6 stats form a compact representation of:

- offensive shot mix
- defensive shot mix allowed
- offensive turnover tendency
- defensive turnover pressure

They directly support the matchup concepts we previously identified:

1. **3PA matchup**
2. **2PA / rim-proxy matchup**
3. **turnover matchup**

But instead of hard-coding only 3 mismatch deltas, we preserve the full underlying raw team tendencies and let the model learn the interactions.

This is more information-preserving and better aligned with a flexible tree-based model.

---

# 7. Time semantics

These features are all **pregame**, based only on games completed before the current game.

For a game with tipoff timestamp `game_ts`, the feature values must use only data from games with:

\[
\text{completed game ts} < \text{game\_ts}
\]

So each row in `game_team_style_profile` is a **strictly pregame** snapshot.

---

# 8. Season logic: reset and prior

This section is important because we explicitly discussed how to handle season boundaries.

## 8.1 No rolling carryover across seasons
We do **not** roll these style stats straight through from the previous season.

Reason:
- team tactics can change materially across seasons
- rolling carryover would contaminate current-season style with stale information

## 8.2 No ongoing prior weighting after the season begins
We also do **not** use previous-season priors continuously throughout the season.

Reason:
- because these are percentages / rates intended to represent current-season team tendencies
- ongoing prior weighting could make the feature slow to reflect real tactical changes

## 8.3 Game 1 initialization only
We do, however, use a previous-season prior **only for the first game of the season**.

For a team’s first game in season \(s\):

\[
x^{pre}_{team,1}
=
x^{final}_{team,s-1}
\]

for each stylistic stat \(x\) in the 6-feature set.

So opening-game pregame values are seeded from the prior season.

## 8.4 Games 2+ use pure current-season season-to-date values
After the first game:

For game \(g \ge 2\),

\[
x^{pre}_{team,g}
=
\text{current-season season-to-date value through game } g-1
\]

So the prior is used **only once**, as the opening-game seed.

---

# 9. Edge-case fallback for first modeled season

If the previous season is unavailable in the historical dataset (e.g. the first modeled season in the full history), then the game-1 prior needs a fallback.

Recommended fallback:

\[
x^{pre}_{team,1} = \bar{x}_{league, init}
\]

where:
- \(\bar{x}_{league, init}\) is a fixed league-average initialization constant for that stylistic stat

This should be computed from the earliest available season context or explicitly chosen and documented.

This situation should only affect the very first modeled season.

---

# 10. Build logic

This is the exact build procedure another agent should follow.

## Step 1. Build team-game realized stat rows
For every completed game and every team in that game, compute a realized row containing:

- `FGA`
- `3PA`
- `FTA`
- `TO`
- and the opponent analogs:
  - `OppFGA`
  - `Opp3PA`
  - `OppFTA`
  - `OppTO`

These are the raw inputs needed to compute the 6 style metrics.

## Step 2. Sort games chronologically within team and season
For each team and season, sort team games by:

\[
game\_ts
\]

ascending.

All season-to-date calculations must use this chronological order.

## Step 3. Build game-team pregame style values

For each team-game row \(g\):

### Case A: game 1 of the season
Set each `*_pre` stylistic feature to the team’s **final value from the previous season**.

Example:
- `off_3pa_rate_pre = previous season final off_3pa_rate`
- `def_3pa_allowed_pre = previous season final def_3pa_allowed`
- etc.

If no previous-season team value exists, use the league-average initialization constant for that stat.

### Case B: game 2 or later
Set each `*_pre` stylistic feature to the team’s **current-season season-to-date value through the previous game**.

This means:

\[
off\_3pa\_rate\_pre
=
\frac{\sum_{games < g} 3PA}{\sum_{games < g} FGA}
\]

and analogously for the other 5 features.

## Step 4. Store one row per `(game_id, team_id)`
After computing the 6 pregame style values, write the row to `game_team_style_profile`.

---

# 11. Exact formulas for season-to-date updates

For a team entering game \(g\), let all sums run over that team’s completed games in the current season strictly before \(g\).

Then:

## Offensive 3PA rate
\[
off\_3pa\_rate\_pre
=
\frac{\sum 3PA}{\sum FGA}
\]

## Defensive 3PA allowed
\[
def\_3pa\_allowed\_pre
=
\frac{\sum Opp3PA}{\sum OppFGA}
\]

## Offensive 2PA rate
\[
off\_2pa\_rate\_pre
=
\frac{\sum (FGA - 3PA)}{\sum FGA}
\]

## Defensive 2PA allowed
\[
def\_2pa\_allowed\_pre
=
\frac{\sum (OppFGA - Opp3PA)}{\sum OppFGA}
\]

## Offensive turnover percentage
\[
off\_tov\_pct\_pre
=
\frac{\sum TO}{\sum (FGA + 0.44 \cdot FTA + TO)}
\]

## Defensive forced turnover percentage
\[
def\_forced\_tov\_pre
=
\frac{\sum OppTO}{\sum (OppFGA + 0.44 \cdot OppFTA + OppTO)}
\]

---

# 12. Why game-team grain is correct

We explicitly changed this from a day-team table to a game-team table.

This is correct because:
- these features update on completed games, not arbitrary dates
- downstream predictions are game-level
- the target variable is game-level
- game-team snapshots are more compact and natural than copying values to off-days

So the natural row is:

\[
(\text{game\_id},\ \text{team\_id})
\]

not a day-team panel.

---

# 13. Why the model should receive raw per-team style stats later

We explicitly decided that the final model should receive raw team style stats, not only pre-collapsed mismatch deltas.

So later the game-level model input will contain:

### Home team
- `home_off_3pa_rate_pre`
- `home_def_3pa_allowed_pre`
- `home_off_2pa_rate_pre`
- `home_def_2pa_allowed_pre`
- `home_off_tov_pct_pre`
- `home_def_forced_tov_pre`

### Away team
- `away_off_3pa_rate_pre`
- `away_def_3pa_allowed_pre`
- `away_off_2pa_rate_pre`
- `away_def_2pa_allowed_pre`
- `away_off_tov_pct_pre`
- `away_def_forced_tov_pre`

This preserves more information than compressing everything to 3 matchup deltas.

---

# 14. Final schema summary

## Table name
`game_team_style_profile`

## Primary key
- `game_id`
- `team_id`

## Required columns

### Game context
- `game_id`
- `game_ts`
- `game_date`
- `season`
- `team_id`
- `opponent_team_id`
- `is_home`
- `is_playoff`

### Pregame stylistic tendency features
- `off_3pa_rate_pre`
- `def_3pa_allowed_pre`
- `off_2pa_rate_pre`
- `def_2pa_allowed_pre`
- `off_tov_pct_pre`
- `def_forced_tov_pre`

## Optional audit / debug columns
- `games_played_before_game`
- `prior_source`
- `last_completed_game_id`
- `last_completed_game_ts`

---

# 15. Final locked specification summary

The locked stylistic tendency features are exactly:

\[
\boxed{
\{off\_3pa\_rate\_pre,\ def\_3pa\_allowed\_pre,\ off\_2pa\_rate\_pre,\ def\_2pa\_allowed\_pre,\ off\_tov\_pct\_pre,\ def\_forced\_tov\_pre\}
}
\]

with these season rules:

- game 1 pregame values = previous-season final values
- games 2+ pregame values = pure current-season season-to-date values through the previous game
- no full rolling cross-season carryover
- no ongoing prior weighting after game 1

This is the exact spec for the stylistic/team-tendency table.
