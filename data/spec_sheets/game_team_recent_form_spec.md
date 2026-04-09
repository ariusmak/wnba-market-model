# `game_team_recent_form` Specification

This document specifies the **team-level recent form table** that is built at the **game-team** level.

It reflects two locked design changes:

1. this table is **not** a generic “team state history” table; it is specifically a **recent form** table
2. this table is **game-team level**, not day-team level, because all recent-form inputs update on **game increments**, not calendar-day increments

This is written as a handoff document for feature engineering / coding.

It is explicit about:

- row grain
- time semantics
- primary key
- exact feature definitions
- exact formulas
- update logic
- chosen hyperparameters
- why the design is structured this way

---

# 1. Purpose of this table

`game_team_recent_form` stores the **pregame recent-form state** for one team in one game.

Each row answers the question:

> “What was this team’s recent form **before this game**?”

The table contains exactly the 5 locked recent-form features:

1. `net_rating_ewma_pre`
2. `efg_ewma_pre`
3. `tov_ewma_pre`
4. `orb_ewma_pre`
5. `ftr_ewma_pre`

These are later joined into the final game-level model input as:

- `home_net_rating_ewma_pre`
- `away_net_rating_ewma_pre`
- `home_efg_ewma_pre`
- `away_efg_ewma_pre`
- `home_tov_ewma_pre`
- `away_tov_ewma_pre`
- `home_orb_ewma_pre`
- `away_orb_ewma_pre`
- `home_ftr_ewma_pre`
- `away_ftr_ewma_pre`

This table is **not** the final model table. It is the canonical team-level game snapshot for recent form.

---

# 2. Why this is game-team level, not day-team level

We explicitly chose **game-team** instead of **day-team** because these features update on the sequence of completed games, not on arbitrary calendar days.

Recent-form inputs such as:

- net rating
- shooting efficiency
- turnover rate
- offensive rebound rate
- free throw rate

change when a team completes a game. They are not inherently daily variables.

So the natural row unit is:

\[
(\text{game\_id},\ \text{team\_id})
\]

not:

\[
(\text{team\_id},\ \text{asof\_date})
\]

This is cleaner because:

- each row directly corresponds to one prediction event
- there is no need to carry forward duplicate daily rows on off-days
- downstream joins become simpler
- the table is more compact and more naturally aligned with the target variable

---

# 3. Table name and grain

## Table name
`game_team_recent_form`

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

## B. Pregame recent-form features
9. `net_rating_ewma_pre`
10. `efg_ewma_pre`
11. `tov_ewma_pre`
12. `orb_ewma_pre`
13. `ftr_ewma_pre`

## C. Optional audit / debug columns
These are not required for the final model, but they are useful:

14. `last_completed_game_id`
15. `last_completed_game_ts`
16. `net_rating_last_game`
17. `efg_last_game`
18. `tov_last_game`
19. `orb_last_game`
20. `ftr_last_game`

---

# 5. Locked recent-form features

The recent-form block contains exactly these 5 team-level features.

---

## 5.1 `net_rating_ewma_pre`

### Purpose
Captures the team’s recent overall strength in possession-adjusted terms.

### Game-level definition
For a completed team-game:

\[
NetRtg = ORtg - DRtg
\]

where:
- `ORtg` = points scored per 100 possessions
- `DRtg` = points allowed per 100 possessions

### Interpretation
Higher is better:
- positive values mean the team has recently outscored opponents on a possession-adjusted basis
- negative values mean the team has recently been outscored

The table stores the **pregame EWMA state**, not the current game’s realized `NetRtg`.

---

## 5.2 `efg_ewma_pre`

### Purpose
Captures recent shooting efficiency.

### Game-level definition
\[
eFG\% = \frac{FGM + 0.5 \cdot 3PM}{FGA}
\]

where:
- `FGM` = field goals made
- `3PM` = 3-pointers made
- `FGA` = field goal attempts

### Interpretation
Higher is better:
- better shot-making efficiency
- extra value for made 3-pointers

The table stores the **pregame EWMA state**, not the current game’s realized `eFG%`.

---

## 5.3 `tov_ewma_pre`

### Purpose
Captures recent offensive turnover tendency.

### Game-level definition
\[
TOV\% = \frac{TO}{FGA + 0.44 \cdot FTA + TO}
\]

where:
- `TO` = turnovers
- `FGA` = field goal attempts
- `FTA` = free throw attempts

### Interpretation
Lower is better offensively:
- high `TOV%` means more wasted possessions
- low `TOV%` means stronger ball security

The table stores the **pregame EWMA state**, not the current game’s realized `TOV%`.

---

## 5.4 `orb_ewma_pre`

### Purpose
Captures recent offensive rebounding / second-chance ability.

### Game-level definition
\[
ORB\% = \frac{ORB}{ORB + OppDRB}
\]

where:
- `ORB` = offensive rebounds by the team
- `OppDRB` = opponent defensive rebounds

### Interpretation
Higher is better:
- more second chances
- stronger offensive rebounding profile

The table stores the **pregame EWMA state**, not the current game’s realized `ORB%`.

---

## 5.5 `ftr_ewma_pre`

### Purpose
Captures recent free-throw generation / foul pressure.

### Game-level definition
\[
FTr = \frac{FTA}{FGA}
\]

where:
- `FTA` = free throw attempts
- `FGA` = field goal attempts

### Interpretation
Higher is better:
- more trips to the line
- more rim pressure / contact generation

The table stores the **pregame EWMA state**, not the current game’s realized `FTr`.

---

# 6. Supporting game-level quantities

These are required to compute the recent-form features.

They do not have to be stored permanently in the final table, but they must be available during construction.

---

## 6.1 Team possessions

For a completed team-game:

\[
Poss_{team} = FGA + 0.44 \cdot FTA - ORB + TO
\]

where:
- `FGA` = field goal attempts
- `FTA` = free throw attempts
- `ORB` = offensive rebounds
- `TO` = turnovers

---

## 6.2 Opponent possessions
For the opponent in the same game:

\[
Poss_{opp} = OppFGA + 0.44 \cdot OppFTA - OppORB + OppTO
\]

---

## 6.3 Game possessions
Use the average of team and opponent possessions:

\[
Poss_{game} = \frac{Poss_{team} + Poss_{opp}}{2}
\]

This is the denominator used for possession-adjusted ratings.

---

## 6.4 Offensive rating
\[
ORtg = 100 \cdot \frac{PTS}{Poss_{game}}
\]

where:
- `PTS` = team points scored

---

## 6.5 Defensive rating
\[
DRtg = 100 \cdot \frac{OppPTS}{Poss_{game}}
\]

where:
- `OppPTS` = opponent points scored

---

## 6.6 Net rating
\[
NetRtg = ORtg - DRtg
\]

---

# 7. EWMA definition

For any team-level game statistic \(x\), define the EWMA recursively as:

\[
x^{EWMA}_{team,g}
=
\lambda_{team} \cdot x_{team,g-1}
+
(1-\lambda_{team}) \cdot x^{EWMA}_{team,g-1}
\]

where:
- \(x_{team,g-1}\) is the team’s realized value in its most recent completed game
- \(x^{EWMA}_{team,g-1}\) is the previous EWMA state after the prior game

This applies separately to:
- `net_rating_ewma_pre`
- `efg_ewma_pre`
- `tov_ewma_pre`
- `orb_ewma_pre`
- `ftr_ewma_pre`

---

# 8. Locked hyperparameter

## Team recent-form EWMA half-life
\[
h_{team} = 7
\]

This implies:

\[
\lambda_{team} = 1 - 2^{-1/7}
\]

Numerically:

\[
\lambda_{team} \approx 0.094276
\]

### Justification
A 7-game half-life:
- is recent enough to capture meaningful form changes
- is smoother than a very short 3–5 game window
- fits the idea that team form should move more slowly than individual player minutes

---

# 9. Initialization / opening-game behavior

We explicitly chose **not** to use a complex cross-season prior for team recent-form EWMA.

So the recent-form block is treated as a same-season rolling state.

## First game of the season
Before a team has completed any game in the current season, there is no same-season recent-form history.

For the team’s first game of the season, use:

- `net_rating_ewma_pre = 0`
- `efg_ewma_pre = 0`
- `tov_ewma_pre = 0`
- `orb_ewma_pre = 0`
- `ftr_ewma_pre = 0`

### Why
This is acceptable because:
- Elo already provides the structural pregame baseline
- recent form is explicitly a short-memory adjustment layer
- opening-game recent form is weakly identified by construction

So this block starts from neutral and then updates once games are played.

---

# 10. Build logic

This is the exact build procedure another agent should follow.

---

## Step 1. Build team-game realized stats
For every completed game, compute one realized team-game record for each team containing:

- `PTS`
- `OppPTS`
- `FGA`
- `FGM`
- `3PM`
- `FTA`
- `TO`
- `ORB`
- `OppDRB`

Then derive:
- `Poss_team`
- `Poss_opp`
- `Poss_game`
- `ORtg`
- `DRtg`
- `NetRtg`
- `eFG%`
- `TOV%`
- `ORB%`
- `FTr`

This is the raw game-level input.

---

## Step 2. Sort team games chronologically within season
For each team and each season, sort that team’s games by:

\[
game\_ts
\]

ascending.

All EWMA updates must follow this chronological order.

---

## Step 3. Initialize team EWMA state at season start
Before the team’s first game of the season, initialize the running state to zeros:

- `net_rating_ewma = 0`
- `efg_ewma = 0`
- `tov_ewma = 0`
- `orb_ewma = 0`
- `ftr_ewma = 0`

These values become the `*_pre` values for the team’s first game.

---

## Step 4. Write one row per team-game using the pregame state
For each team-game row \(g\):

Store the current EWMA state **before** updating it with that game.

So for game \(g\), write:

- `net_rating_ewma_pre = current running net rating EWMA`
- `efg_ewma_pre = current running eFG EWMA`
- `tov_ewma_pre = current running TOV EWMA`
- `orb_ewma_pre = current running ORB EWMA`
- `ftr_ewma_pre = current running FTr EWMA`

These are the pregame features.

---

## Step 5. Update the running EWMA state after the game
After writing the team-game row for game \(g\), update each running EWMA using the realized game stats:

For each stat \(x\):

\[
x^{EWMA}_{new}
=
\lambda_{team} \cdot x_{game}
+
(1-\lambda_{team}) \cdot x^{EWMA}_{old}
\]

Where \(x_{game}\) is the realized value from the current completed game.

---

# 11. Why this table uses separate home and away team values later

We explicitly chose **not** to reduce recent-form features only to home-away deltas.

The downstream final model will receive:

- home team recent-form values
- away team recent-form values

separately.

### Why
For tree-based models like XGBoost, the absolute levels can matter in addition to the difference.

A `+5` net-rating gap between:
- two strong teams
and
- two weak teams

may not mean the same thing.

So this table should preserve team-specific values, not just deltas.

---

# 12. Recommended final schema

## Table name
`game_team_recent_form`

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

### Recent-form features
- `net_rating_ewma_pre`
- `efg_ewma_pre`
- `tov_ewma_pre`
- `orb_ewma_pre`
- `ftr_ewma_pre`

## Optional audit / debug columns
- `last_completed_game_id`
- `last_completed_game_ts`
- `net_rating_last_game`
- `efg_last_game`
- `tov_last_game`
- `orb_last_game`
- `ftr_last_game`

---

# 13. Final locked specification summary

The locked recent-form team-level features are exactly:

\[
\boxed{
\{net\_rating\_ewma\_pre,\ efg\_ewma\_pre,\ tov\_ewma\_pre,\ orb\_ewma\_pre,\ ftr\_ewma\_pre\}
}
\]

with:

- game-team grain:
\[
\boxed{(\text{game\_id},\ \text{team\_id})}
\]

- EWMA half-life:
\[
\boxed{h_{team} = 7}
\]

- smoothing parameter:
\[
\boxed{\lambda_{team} = 1 - 2^{-1/7}}
\]

This is the exact spec for the recent-form team-level table.
