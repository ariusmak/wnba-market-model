# `game_team_schedule_context` Specification

This document specifies the **team-level rest / schedule / travel context table** that is built at the **game-team** level.

It is written as a handoff document for feature engineering / coding.

It is explicit about:

- row grain
- time semantics
- primary key
- exact feature definitions
- exact formulas
- build logic
- locked hyperparameters / thresholds
- why the design is structured this way

---

# 1. Purpose of this table

`game_team_schedule_context` stores the **pregame schedule burden and travel burden** for one team in one game.

Each row answers the question:

> “What was this team’s rest / travel context **before this game**?”

This table is not meant to represent team quality or style.  
It represents **game-specific context** derived from the sequence of scheduled games and travel locations.

These features are later joined into the final game-level model input as **raw team-level inputs**, not pre-collapsed home-away deltas.

---

# 2. Locked design choices

The following design choices are already locked.

## 2.1 Game-team grain
This table is **game-team** level, not day-team level.

The natural row is:

\[
(\text{game\_id},\ \text{team\_id})
\]

because rest and travel context are defined relative to a specific upcoming game.

## 2.2 Per-team inputs later, not only deltas
The downstream model will later receive:

- home team context features
- away team context features

separately.

We explicitly chose **not** to reduce them only to home-away deltas, because absolute burden can matter in addition to relative burden.

## 2.3 Travel origin rule
We explicitly chose this practical observable rule:

\[
\text{origin city} =
\begin{cases}
\text{previous game city}, & \text{if rest gap} < 4 \\
\text{home city}, & \text{if rest gap} \ge 4
\end{cases}
\]

This is the key locked travel rule.

### Justification
- teams often stay on the road between closely spaced games
- after a longer break, assuming a return toward home base is a more realistic heuristic
- this is more defensible than assuming either “always previous city” or “always home city”

## 2.4 Long-break threshold
The long-break threshold is locked at:

\[
R_{home\_reset} = 4 \text{ days}
\]

So:
- if rest gap is 0, 1, 2, or 3 days: origin = previous game city
- if rest gap is 4 or more days: origin = home city

---

# 3. Table name and grain

## Table name
`game_team_schedule_context`

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

## B. Pregame schedule / rest features
9. `days_rest_pre`
10. `is_b2b_pre`
11. `games_last_4_days_pre`
12. `games_last_7_days_pre`

## C. Pregame travel / timezone features
13. `origin_city_pre`
14. `current_city_pre`
15. `travel_miles_pre`
16. `timezone_shift_hours_pre`

## D. Optional audit / debug columns
17. `previous_game_id`
18. `previous_game_ts`
19. `previous_game_city`
20. `home_city`
21. `origin_rule_used`

---

# 5. Exact feature definitions

These are the exact locked context features.

## 5.1 `days_rest_pre`

### Purpose
Measures the number of full off-days since the team’s most recent completed game.

### Definition
If the team has a previous completed game before the current game:

\[
days\_rest\_pre =
(\text{current game date} - \text{previous game date}) - 1
\]

measured in whole days.

Examples:
- previous game on July 1, current game on July 2:
  \[
  days\_rest\_pre = 0
  \]
- previous game on July 1, current game on July 4:
  \[
  days\_rest\_pre = 2
  \]

### Season opener
If the team has no previous game in the season:
- set `days_rest_pre = 0`
- and let the downstream model learn that opening games are a special context through the combination of other features

### Interpretation
Higher means more rest.

---

## 5.2 `is_b2b_pre`

### Purpose
Captures whether the team is on the second leg of a back-to-back.

### Definition
\[
is\_b2b\_pre =
\begin{cases}
1 & \text{if } days\_rest\_pre = 0 \\
0 & \text{otherwise}
\end{cases}
\]

### Interpretation
A simple binary fatigue indicator.

---

## 5.3 `games_last_4_days_pre`

### Purpose
Measures recent schedule compression.

### Definition
Count the number of completed games by the team in the inclusive window:

\[
(\text{game\_date} - 4,\ \text{game\_date})
\]

strictly before the current game.

Equivalent plain-English definition:
- number of team games in the **4 calendar days before the current game date**

### Example
If current game is on July 10, count games on:
- July 6
- July 7
- July 8
- July 9

Do not include the current game.

### Interpretation
Higher means more compressed recent schedule.

---

## 5.4 `games_last_7_days_pre`

### Purpose
Measures broader short-run schedule density.

### Definition
Count the number of completed team games in the **7 calendar days before the current game date**, excluding the current game.

### Interpretation
Higher means heavier recent game load.

---

## 5.5 `origin_city_pre`

### Purpose
The city assumed to be the team’s travel origin for the current game.

### Locked rule
Let `rest_gap_days` be the number of calendar days between the previous game date and current game date minus 1, i.e. `days_rest_pre`.

Then:

\[
origin\_city\_pre =
\begin{cases}
\text{previous game city}, & \text{if } days\_rest\_pre < 4 \\
\text{team home city}, & \text{if } days\_rest\_pre \ge 4
\end{cases}
\]

### Season opener
If there is no previous game:
- `origin_city_pre = team home city`

### Interpretation
This is the practical observable approximation to where the team is traveling from.

---

## 5.6 `current_city_pre`

### Purpose
The city of the current game venue.

### Definition
This is the host city of the game venue for the team’s current game.

### Notes
- for home teams, this will usually equal home city
- for away teams, this is the opponent’s city (or neutral site city if relevant)

---

## 5.7 `travel_miles_pre`

### Purpose
Measures travel burden in distance terms.

### Definition
Compute the great-circle distance (Haversine distance) between:

- `origin_city_pre`
- `current_city_pre`

\[
travel\_miles\_pre = \text{distance in miles}(origin\_city\_pre, current\_city\_pre)
\]

### Season opener
If `origin_city_pre = current_city_pre`, then:
\[
travel\_miles\_pre = 0
\]

### Interpretation
Higher means more travel distance before the game.

---

## 5.8 `timezone_shift_hours_pre`

### Purpose
Measures travel burden in circadian / timezone terms.

### Definition
Let:

- `tz(origin_city_pre)` = timezone offset or timezone identity for the origin city
- `tz(current_city_pre)` = timezone offset or timezone identity for the current city

Then define:

\[
timezone\_shift\_hours\_pre =
\text{UTC offset}(current\_city\_pre)
-
\text{UTC offset}(origin\_city\_pre)
\]

Expressed in hours.

### Interpretation
Examples:
- origin Eastern, current Central:
  \[
  timezone\_shift\_hours\_pre = -1
  \]
- origin Pacific, current Eastern:
  \[
  timezone\_shift\_hours\_pre = +3
  \]

The sign is preserved.  
Positive means traveling eastward into later local time.  
Negative means traveling westward into earlier local time.

### Season opener
If origin and current city are in the same timezone:
\[
timezone\_shift\_hours\_pre = 0
\]

---

# 6. Build logic

This is the exact build procedure another agent should follow.

## Step 1. Build team schedule chronology
For each team and season, sort all team games by:

\[
game\_ts
\]

ascending.

This gives the chronological game sequence required for schedule context.

## Step 2. For each team-game, identify the previous completed game
For each row \((game\_id, team\_id)\):

- find the team’s immediately preceding completed game in the same season
- extract:
  - `previous_game_id`
  - `previous_game_ts`
  - `previous_game_date`
  - `previous_game_city`

If none exists, this is the season opener.

## Step 3. Compute rest features

### `days_rest_pre`
If previous game exists:

\[
days\_rest\_pre =
(\text{current game date} - \text{previous game date}) - 1
\]

Else:
- `days_rest_pre = 0`

### `is_b2b_pre`
\[
is\_b2b\_pre = 1 \iff days\_rest\_pre = 0
\]

### `games_last_4_days_pre`
Count the team’s completed games in the 4-day pregame window.

### `games_last_7_days_pre`
Count the team’s completed games in the 7-day pregame window.

## Step 4. Determine travel origin
Use the locked origin rule:

\[
origin\_city\_pre =
\begin{cases}
\text{previous game city}, & \text{if } days\_rest\_pre < 4 \\
\text{team home city}, & \text{if } days\_rest\_pre \ge 4
\end{cases}
\]

If no previous game exists:
- `origin_city_pre = team home city`

Also store:
- `origin_rule_used = previous_game_city` or `home_city`

for debugging if desired.

## Step 5. Compute travel features

### `current_city_pre`
Read from the current game venue.

### `travel_miles_pre`
Compute Haversine distance from `origin_city_pre` to `current_city_pre`.

### `timezone_shift_hours_pre`
Compute timezone offset difference from origin city to current city.

## Step 6. Persist one row per `(game_id, team_id)`
Store all game context, rest, and travel fields.

---

# 7. Edge cases

## 7.1 Season opener
No previous game exists.

Use:
- `days_rest_pre = 0`
- `is_b2b_pre = 0`
- `games_last_4_days_pre = 0`
- `games_last_7_days_pre = 0`
- `origin_city_pre = team home city`

Then compute:
- `travel_miles_pre` from home city to current city
- `timezone_shift_hours_pre` from home city to current city

For a home opener, these will both usually be 0.

## 7.2 Neutral-site games
If the game is at a neutral site:
- `current_city_pre` should be the neutral-site city
- `is_home` may still follow the game summary’s official designation
- travel should still be computed to the actual neutral-site city

## 7.3 Long breaks
The long-break rule is already encoded via:

\[
days\_rest\_pre \ge 4
\Rightarrow origin\_city\_pre = home\_city
\]

No additional override is needed.

---

# 8. Why this design is appropriate

## Why not use previous game city always?
Because after a long break, teams may reasonably return home or reset logistically, so previous-game-city-only can overstate continuous road-trip burden.

## Why not use home city always?
Because teams often stay on the road across short gaps and road trips. Home-city-only would ignore that and frequently be wrong.

## Why this hybrid rule?
The hybrid rule is the best practical observable compromise:
- previous game city for short gaps
- home city after longer breaks

It is simple, defensible, and matches realistic scheduling behavior better than either extreme.

---

# 9. Final schema summary

## Table name
`game_team_schedule_context`

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

### Rest / schedule
- `days_rest_pre`
- `is_b2b_pre`
- `games_last_4_days_pre`
- `games_last_7_days_pre`

### Travel / timezone
- `origin_city_pre`
- `current_city_pre`
- `travel_miles_pre`
- `timezone_shift_hours_pre`

## Optional audit / debug columns
- `previous_game_id`
- `previous_game_ts`
- `previous_game_city`
- `home_city`
- `origin_rule_used`

---

# 10. Final locked specification summary

The locked schedule/travel context features are exactly:

\[
\boxed{
\{days\_rest\_pre,\ is\_b2b\_pre,\ games\_last\_4\_days\_pre,\ games\_last\_7\_days\_pre,\ travel\_miles\_pre,\ timezone\_shift\_hours\_pre\}
}
\]

with the locked origin rule:

\[
\boxed{
origin\_city\_pre =
\begin{cases}
\text{previous game city}, & \text{if } days\_rest\_pre < 4 \\
\text{home city}, & \text{if } days\_rest\_pre \ge 4
\end{cases}
}
\]

This is the exact spec for the rest/travel table.
