# `player_state_history` Specification

Below is the exact specification for the upstream historical player-state table we have designed so far.

This is written as a handoff document for feature engineering / coding.

It is explicit about:

- row grain
- time semantics
- primary key
- column definitions
- exact formulas
- how to build each column
- edge cases / defaults

---

# 1. Purpose of this table

`player_state_history` is the **upstream historical state layer** for backtesting and later live inference.

It stores, for each player and each daily snapshot timestamp, the player’s most recent **pregame-usable state**:

- role proxy (`m_ewma`)
- quality proxy (`q`)
- combined strength (`strength = m_ewma * q`)
- injury-window timing features
- recent participation features

This table is **not** the final model table.

It is used to build the downstream:

- `game_team_player` table: one row per `(game_id, team_id, player_id)`
- then the final game-level model input table

---

# 2. Grain / primary key

## Table name
`player_state_history`

## Grain
One row per:

\[
(\text{player\_id},\ \text{asof\_ts})
\]

where `asof_ts` is a **daily snapshot timestamp**.

## Primary key
\[
\boxed{(\text{player\_id},\ \text{asof\_ts})}
\]

---

# 3. Time semantics

## Locked design choice
This upstream historical layer is built at **daily frequency**, not intraday.

So `asof_ts` should represent the state of the player **at the end of that day**, after all known updates for that day have been processed.

## Recommended implementation
Use:

- `asof_date` as the conceptual day
- `asof_ts` as the timestamp representation of that daily snapshot

A simple convention is:

\[
\text{asof\_ts} = \text{asof\_date at 23:59:59 in chosen canonical timezone}
\]

If the pipeline uses UTC internally, store UTC.  
If the league is modeled in a local timezone, store that consistently.

## Important downstream join rule
When building `game_team_player` for a game with tipoff timestamp `game_ts`, join to the **latest** row satisfying:

\[
\text{asof\_ts} < \text{game\_ts}
\]

This ensures the player state used for the game is strictly pregame.

---

# 4. Daily snapshot scope

For every date in a season, create one row for each player who is relevant to the season universe.

At minimum, that means:
- every player who appears in the season data
- every player who appears on a roster / in a game / in injury data

This gives you a complete historical state panel.

---

# 5. Columns and exact definitions

The columns we decided on are:

1. `player_id`
2. `asof_ts`
3. `current_team_id`
4. `m_ewma`
5. `q`
6. `strength`
7. `days_since_first_report`
8. `days_since_last_dnp`
9. `consec_dnps`
10. `played_last_game`
11. `minutes_last_game`
12. `days_since_last_played`
13. `injury_present_flag`

I’ll define each exactly.

---

## 5.1 `player_id`
Unique player identifier.

### Type
String or integer, depending on source system.

### Source
Your canonical player ID from your sports data source.

---

## 5.2 `asof_ts`
Daily snapshot timestamp representing the player’s state as of the end of that date.

### Type
Timestamp.

### Example meaning
If `asof_ts = 2024-07-10 23:59:59`, all fields in that row reflect the player’s state **after** all events on July 10, 2024, and can be used for games on July 11 (or later) until a new snapshot supersedes it.

---

## 5.3 `current_team_id`
Team the player belongs to as of `asof_ts`.

### Type
String or integer.

### Role
Metadata only; not part of the primary key.

### Build rule
At each `asof_ts`, set this to the latest known team assignment for the player as of that date.

If the player changed teams mid-season, this value changes accordingly.

---

## 5.4 `m_ewma`
The player’s exponentially weighted moving average of minutes.

This is the **role / recent usage** component of player strength.

### Locked design
EWMA half-life in games:
\[
h_M = 5
\]

### Equivalent EWMA smoothing parameter
\[
\lambda_M = 1 - 2^{-1/5}
\]

Numerically:

\[
\lambda_M \approx 0.129449
\]

### Exact recursive definition
Let \(m_{i,g}\) be actual minutes played by player \(i\) in game \(g\).

For the player’s most recent completed game before the current state update:

\[
m^{EWMA}_{i,t}
=
\lambda_M \cdot m_{i,\text{last game}}
+
(1-\lambda_M)\cdot m^{EWMA}_{i,t^-}
\]

where:
- \(m^{EWMA}_{i,t^-}\) is the previous EWMA state before incorporating the most recent game
- \(m_{i,\text{last game}}\) is actual minutes in the last completed game
- if player did not play in the last completed game, minutes = 0 **if the player was on the team and recorded as DNP/inactive**

### Practical event-update interpretation
Whenever a player’s team completes a game, update `m_ewma` once using that game’s actual minutes.

Then carry that value forward daily until the next game update.

### Season start initialization
At the start of a new season, initialize from prior-season minutes information.

A reasonable implementation consistent with prior decisions is:

\[
m^{EWMA}_{i,\text{season start}} = \text{previous season average minutes}
\]

If previous-season minutes are unavailable:
- use 0 for true rookies / new players, or
- use league-average bench minutes if you later decide to soften rookies

### Important interpretation
`m_ewma` is **role-sensitive**, not quality-sensitive.  
It should rise when a player’s recent minutes rise and fall when they stop playing.

---

## 5.5 `q`
The player’s quality-per-minute proxy.

This is the **quality** component of player strength.

### Locked design choice
We are using **season-to-date EFF per minute with a decaying prior** from the previous season.

### Step 1: define EFF
\[
EFF = PTS + REB + AST + STL + BLK - (FGA-FGM) - (FTA-FTM) - TO
\]

Where:
- `PTS` = points
- `REB` = total rebounds
- `AST` = assists
- `STL` = steals
- `BLK` = blocks
- `FGA` = field goal attempts
- `FGM` = field goals made
- `FTA` = free throw attempts
- `FTM` = free throws made
- `TO` = turnovers

### Step 2: current-season cumulative quality pieces
As of `asof_ts`, define:

- \(EFF^{curr}_{i,\le t}\): cumulative current-season EFF **through** `asof_ts`
- \(MIN^{curr}_{i,\le t}\): cumulative current-season minutes **through** `asof_ts`

These should include only completed games strictly before any downstream game being predicted.

### Step 3: prior-season quality
Define:

\[
q^{prev}_i = \frac{EFF^{prevseason}_i}{MIN^{prevseason}_i}
\]

if previous-season minutes exist.

### Prior fallback
If the player has no usable previous-season minutes, use a fallback prior:

\[
q^{prev}_i = q_{\text{league avg prev season}}
\]

This is the cleanest default and is consistent with our earlier discussion that prior carryover should stabilize early season.

### Locked prior-strength hyperparameter
\[
\tau = 150
\]

Interpretation: previous-season quality counts like 150 pseudo-minutes at season start.

### Exact formula
\[
q_{i,t}
=
\frac{\tau \cdot q^{prev}_i + EFF^{curr}_{i,\le t}}
{\tau + MIN^{curr}_{i,\le t}}
\]

### Interpretation
- early season: \(q\) stays close to prior-season quality
- later season: current-season EFF/min takes over

### Daily update behavior
`q` only changes when cumulative season totals change, i.e. after games.

On non-game days, it is carried forward.

---

## 5.6 `strength`
Player strength proxy.

### Locked formula
\[
strength_{i,t} = m^{EWMA}_{i,t} \cdot q_{i,t}
\]

### Interpretation
- `m_ewma` captures current role / expected usage
- `q` captures quality per minute
- `strength` is the combined role-weighted quality proxy

This is the quantity used to rank players when later creating player slots in the downstream game snapshot layer.

---

## 5.7 `days_since_first_report`
Days since the **first injury report date** in the player’s most recent relevant injury window.

### Injury-window concept
An injury window is the most recent contiguous injury episode we are tracking for the player.

### Exact definition
If the player has a relevant injury window as of `asof_ts`, then:

\[
days\_since\_first\_report
=
(\text{asof\_date} - \text{first\_report\_date of most recent injury window})
\]

measured in whole days.

If there is no relevant injury window:
- set to 0
- and `injury_present_flag = 0`

### Interpretation
Captures how old the current/recent injury episode is.

---

## 5.8 `days_since_last_dnp`
Days since the most recent **DNP due to that injury window**.

### Exact definition
If the player has at least one DNP in the most recent relevant injury window:

\[
days\_since\_last\_dnp
=
(\text{asof\_date} - \text{most recent DNP date in injury window})
\]

If there has been no DNP in the relevant injury window:
- set to 0
- and let the model interpret this jointly with `consec_dnps`, `played_last_game`, and `injury_present_flag`

### Interpretation
Captures how recently the player actually missed a game due to the injury.

---

## 5.9 `consec_dnps`
Number of consecutive DNPs in the most recent injury window.

### Exact definition
Count the number of consecutive games immediately preceding the player’s return or current state in which the player recorded a DNP attributable to the injury window.

If no such DNP streak exists:
\[
consec\_dnps = 0
\]

### Interpretation
A severity / persistence proxy:
- 1 = missed one straight game
- 5 = missed five straight games
- 0 = no DNP streak in the current relevant window

---

## 5.10 `played_last_game`
Indicator for whether the player played in their most recent team game before `asof_ts`.

### Definition
\[
played\_last\_game =
\begin{cases}
1 & \text{if player logged minutes in most recent team game} \\
0 & \text{otherwise}
\end{cases}
\]

### Important note
If the player existed but did not appear (DNP/inactive), this should be 0.

### Interpretation
Signals whether the player is currently participating.

---

## 5.11 `minutes_last_game`
Minutes played in the player’s most recent team game before `asof_ts`.

### Definition
\[
minutes\_last\_game = \text{actual minutes in most recent team game}
\]

If the player did not play:
\[
minutes\_last\_game = 0
\]

### Interpretation
Helps identify return-from-injury / limited-minutes situations.

---

## 5.12 `days_since_last_played`
Days since the player last appeared in a game.

### Definition
Let `last_played_date` be the date of the player’s most recent game with positive minutes.

\[
days\_since\_last\_played
=
(\text{asof\_date} - \text{last\_played\_date})
\]

If the player has not yet played in the current season, use the most recent played game from prior-season history if available; otherwise initialize consistently with your historical data start.

If truly no prior play record exists, set to a large sentinel only if that is explicitly supported later.  
For now, the safer v1 choice is:
- use the most recent historical played date if available
- otherwise set to 0 and let rookie/no-history be handled through prior defaults in `q`

### Interpretation
Different from `days_since_last_dnp`; it measures how long since the player actually participated.

---

## 5.13 `injury_present_flag`
Binary flag indicating whether the player is inside the injury inclusion window.

### Locked injury inclusion rule
A player is included if they are within **14 days of the most recent of**:

- last injury report date in the most recent injury window
- last DNP date in that injury window

Formally, let:

\[
recent\_injury\_activity\_date
=
\max(\text{last injury report date},\ \text{last DNP date})
\]

Then:

\[
injury\_present\_flag
=
\begin{cases}
1 & \text{if } (\text{asof\_date} - recent\_injury\_activity\_date) \le 14 \\
0 & \text{otherwise}
\end{cases}
\]

### Important consequence
If `injury_present_flag = 0`, the injury-window timing features should default to 0:
- `days_since_first_report = 0`
- `days_since_last_dnp = 0`
- `consec_dnps = 0`

This keeps the state table clean and downstream modeling simple.

---

# 6. Build order / update order

This is the exact build logic another agent should follow.

## Step A. Build base historical game logs and injury-window histories
Need player-level history of:
- games played
- minutes
- DNPs
- injury report dates / windows

These are the raw sources.

## Step B. For each season, initialize player state
For each player at season start:

### `m_ewma`
Initialize from previous-season average minutes.

### `q`
Compute previous-season quality prior:
\[
q^{prev}_i = \frac{EFF^{prevseason}_i}{MIN^{prevseason}_i}
\]
or league-average fallback if unavailable.

Then current-season initial quality is effectively:
\[
q_{i,\text{start}} = q^{prev}_i
\]
because current-season minutes are 0.

### `strength`
\[
strength = m\_ewma \cdot q
\]

### Injury fields
Initialize from any active/recent injury history crossing the season boundary if such cases exist in the source data. Otherwise zeros.

## Step C. Iterate day by day through the season
For each `asof_date`:

1. process any completed games from that date  
   - update cumulative EFF and minutes  
   - update `m_ewma`  
   - update `played_last_game`, `minutes_last_game`, `days_since_last_played`

2. process injury-window updates for that date  
   - first report date  
   - last report date  
   - DNP dates  
   - consecutive DNP count

3. compute / refresh:  
   - `q`  
   - `strength`  
   - injury timing fields  
   - `injury_present_flag`

4. write one row per player for that `asof_ts`

---

# 7. Exact formulas grouped together

For convenience, here are the locked formulas in one place.

## EWMA minutes
Half-life:
\[
h_M = 5
\]

Smoothing:
\[
\lambda_M = 1 - 2^{-1/5}
\]

Update:
\[
m^{EWMA}_{i,t}
=
\lambda_M m_{i,\text{last game}}
+
(1-\lambda_M)m^{EWMA}_{i,t^-}
\]

## Quality with prior
Prior weight:
\[
\tau = 150
\]

Quality:
\[
q_{i,t}
=
\frac{\tau q^{prev}_i + EFF^{curr}_{i,\le t}}
{\tau + MIN^{curr}_{i,\le t}}
\]

## Strength
\[
strength_{i,t} = m^{EWMA}_{i,t} \cdot q_{i,t}
\]

## Injury inclusion flag
\[
recent\_injury\_activity\_date
=
\max(\text{last injury report date},\ \text{last DNP date})
\]

\[
injury\_present\_flag
=
1 \iff (\text{asof\_date} - recent\_injury\_activity\_date)\le 14
\]

---

# 8. Default value rules

These should be implemented consistently.

## If `injury_present_flag = 0`
Set:
- `days_since_first_report = 0`
- `days_since_last_dnp = 0`
- `consec_dnps = 0`

Keep:
- `played_last_game`
- `minutes_last_game`
- `days_since_last_played`
- `m_ewma`
- `q`
- `strength`

because those are always defined as player state, not only injury state.

## If player did not play last game
Set:
- `played_last_game = 0`
- `minutes_last_game = 0`

## If player has no previous-season quality prior
Use:
\[
q^{prev}_i = q_{\text{league avg prev season}}
\]

---

# 9. Recommended final schema

Here is the exact schema to implement.

## Table: `player_state_history`

### Primary key
- `player_id`
- `asof_ts`

### Columns
- `player_id`
- `asof_ts`
- `current_team_id`
- `m_ewma`
- `q`
- `strength`
- `days_since_first_report`
- `days_since_last_dnp`
- `consec_dnps`
- `played_last_game`
- `minutes_last_game`
- `days_since_last_played`
- `injury_present_flag`

### Suggested optional metadata columns
These are not required by the feature list but can help debugging:
- `season`
- `asof_date`
- `last_game_date`
- `last_report_date`
- `last_dnp_date`
- `first_report_date_current_window`
