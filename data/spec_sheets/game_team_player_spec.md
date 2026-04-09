# `game_team_player` Specification

This document specifies the **downstream layer** that is built from:

1. the upstream historical state layer  
   `player_state_history`
2. the official game summary / box score roster for each game

This table is the canonical **pregame player snapshot by game**.

It is the bridge between:
- upstream player-state history
- and the later model-ready flattened / slotted game-level table

---

# 1. Purpose of this table

`game_team_player` exists to answer the question:

> For a specific upcoming or historical game, what did we know **before tipoff** about every player listed for each team in the game summary?

This table must include **every player listed for each team in the official game summary**, even if that player:

- did not play
- logged 0 minutes
- was inactive
- was DNP
- was out due to injury

This is crucial because the injury model needs the absent injured players to remain visible downstream.

This table is **not yet the final ML table**.  
It is the canonical player-level game snapshot from which the later slotted / flattened game-level features will be built.

---

# 2. Grain / primary key

## Table name
`game_team_player`

## Grain
One row per:

\[
(\text{game\_id},\ \text{team\_id},\ \text{player\_id})
\]

for **every player listed under each team in the game summary**.

## Primary key
\[
\boxed{(\text{game\_id},\ \text{team\_id},\ \text{player\_id})}
\]

## Number of rows per game
If a game summary lists \(N_h\) players for the home team and \(N_a\) players for the away team, then the game contributes:

\[
N_h + N_a
\]

rows.

This is intentionally **not restricted to players who played**.

---

# 3. Why this structure is correct

We considered whether this table should contain:

- only players who actually played, or
- every player listed in the game summary

We explicitly chose the second option.

## Reason
If a player is:

- inactive
- DNP due to injury
- listed but unavailable

then their pregame injury state is exactly the thing the model needs to see.  
Dropping them because they did not play would erase important signal.

So this table must preserve:

- healthy players
- active but limited players
- inactive / injured players
- DNP players

in one unified game-level snapshot.

---

# 4. Build inputs

This table is built from two sources:

## Source A: `player_state_history`
Upstream historical player-state table with key:

\[
(\text{player\_id},\ \text{asof\_ts})
\]

containing:
- `m_ewma`
- `q`
- `strength`
- injury-window timing features
- participation state

## Source B: official game summary
The game summary provides:
- `game_id`
- `game_ts` / tipoff timestamp
- `game_date`
- home team ID
- away team ID
- roster/player list for each team
- player metadata
- actual played / DNP / inactive info for that game

---

# 5. Exact join logic

For each game \(g\) with tipoff timestamp `game_ts`:

1. read the full player list for the home team from the game summary
2. read the full player list for the away team from the game summary
3. for each listed player \(i\), join to the **latest** row in `player_state_history` satisfying:

\[
(\text{player\_id} = i)
\quad \text{and} \quad
\text{asof\_ts} < \text{game\_ts}
\]

This row is the player’s pregame state snapshot.

## Important rule
The join must use the **latest available upstream state strictly before tipoff**.

That ensures the downstream table is truly pregame and backtest-safe.

---

# 6. What if no upstream state row exists?

This should be rare if `player_state_history` is complete, but the rule should still be explicit.

If a player is listed in the game summary and no prior state row exists:

- keep the row
- fill missing upstream-derived fields using their default initialization logic
- preserve all game-summary metadata

Recommended default behavior:

- `m_ewma = 0` if no prior history exists
- `q = league-average prior quality`
- `strength = m_ewma * q = 0`
- injury fields = 0
- `injury_present_flag = 0`

This preserves the row and avoids silent row loss.

---

# 7. Column list and exact definitions

The downstream table should contain:

## A. Game identifiers and team/game context
1. `game_id`
2. `game_ts`
3. `game_date`
4. `season`
5. `team_id`
6. `opponent_team_id`
7. `is_home`
8. `is_playoff`

## B. Player identifiers / roster metadata from game summary
9. `player_id`
10. `player_name`
11. `jersey_number` (optional)
12. `position` (optional if provided)
13. `listed_on_game_summary_flag

## C. Pregame state features pulled from `player_state_history`
14. `state_asof_ts`
15. `m_ewma_pre`
16. `q_pre`
17. `strength_pre`
18. `days_since_first_report_pre`
19. `days_since_last_dnp_pre`
20. `consec_dnps_pre`
21. `played_last_game_pre`
22. `minutes_last_game_pre`
23. `days_since_last_played_pre`
24. `injury_present_flag_pre`

## D. Same-game participation / realized metadata from the game summary
25. `played_in_game`
26. `minutes_in_game`
27. `not_playing_reason`
28. `not_playing_description`

The D-columns are **not** pregame model inputs.  
They are included for auditing, debugging, and future validation of the injury-state logic.

---

# 8. Exact definitions of every column

---

## 8.1 `game_id`
Unique game identifier from the game summary source.

---

## 8.2 `game_ts`
Pregame tipoff timestamp for the game.

### Use
This is the reference timestamp used for the upstream join:
\[
\text{state\_asof\_ts} < \text{game\_ts}
\]

---

## 8.3 `game_date`
Calendar date of the game.

Can be derived from `game_ts`, but store explicitly for convenience.

---

## 8.4 `season`
Season label / year for the game.

---

## 8.5 `team_id`
The team associated with this row.

This row is always from the perspective of one team in one game.

---

## 8.6 `opponent_team_id`
The opposing team in the game.

---

## 8.7 `is_home`
Binary indicator:

\[
is\_home =
\begin{cases}
1 & \text{if team_id is the home team} \\
0 & \text{if team_id is the away team}
\end{cases}
\]

---

## 8.8 `is_playoff`
Binary or categorical indicator for postseason / regular season.

Use the game summary / schedule metadata if available.

---

## 8.9 `player_id`
Canonical player identifier from the game summary.

This must match the ID system used in `player_state_history`.

---

## 8.10 `player_name`
Human-readable player name from the game summary.

Metadata only.

---

## 8.11 `jersey_number` (optional)
If available from the game summary.

Metadata only.

---

## 8.12 `position` (optional)
If available from the game summary.

Metadata only.

---

## 8.13 `listed_on_game_summary_flag`
Binary indicator:

\[
listed\_on\_game\_summary\_flag = 1
\]

for every row in this table.

This column is technically redundant, but can be useful for debugging and downstream merges.

---

## 8.14 `state_asof_ts`
Timestamp of the upstream `player_state_history` row that was joined into this downstream row.

### Definition
The latest upstream timestamp satisfying:

\[
state\_asof\_ts < game\_ts
\]

This is critical for auditability.

---

## 8.15 `m_ewma_pre`
Copied from upstream `player_state_history.m_ewma`.

This is the player’s pregame EWMA minutes.

---

## 8.16 `q_pre`
Copied from upstream `player_state_history.q`.

This is the player’s pregame quality proxy.

---

## 8.17 `strength_pre`
Copied from upstream `player_state_history.strength`.

This should equal:

\[
strength\_pre = m\_ewma\_pre \cdot q\_pre
\]

---

## 8.18 `days_since_first_report_pre`
Copied from upstream `player_state_history.days_since_first_report`.

Pregame injury timing feature.

---

## 8.19 `days_since_last_dnp_pre`
Copied from upstream `player_state_history.days_since_last_dnp`.

Pregame injury timing feature.

---

## 8.20 `consec_dnps_pre`
Copied from upstream `player_state_history.consec_dnps`.

Pregame injury persistence feature.

---

## 8.21 `played_last_game_pre`
Copied from upstream `player_state_history.played_last_game`.

Pregame recent participation feature.

---

## 8.22 `minutes_last_game_pre`
Copied from upstream `player_state_history.minutes_last_game`.

Pregame recent participation feature.

---

## 8.23 `days_since_last_played_pre`
Copied from upstream `player_state_history.days_since_last_played`.

Pregame recent participation feature.

---

## 8.24 `injury_present_flag_pre`
Copied from upstream `player_state_history.injury_present_flag`.

Binary indicator that this player is inside the injury inclusion window.

---

## 8.25 `played_in_game`
Binary realized indicator from the game summary:

\[
played\_in\_game =
\begin{cases}
1 & \text{if minutes\_in\_game > 0} \\
0 & \text{otherwise}
\end{cases}
\]

This is not a pregame feature; it is postgame realized metadata.

---

## 8.26 `minutes_in_game`
Realized game minutes from the game summary.

This is not a pregame feature; it is postgame realized metadata.

---

## 8.27 `not_playing_reason`
Raw game-summary non-playing reason, if available.

Examples:
- `"DNP - Coach's Decision"`
- `"Inactive - Injury/Illness"`

This is included only for audit / debugging, not as a pregame model feature.

---

## 8.28 `not_playing_description`
Raw game-summary non-playing description, if available.

Example:
- `"Right MCL"`

Again, audit / debugging only.

---

# 9. Build procedure

Below is the exact downstream build sequence.

---

## Step 1. Read one game summary
For each game summary file / record:

- extract `game_id`
- extract `game_ts`
- extract `game_date`
- extract `season`
- identify home and away teams
- extract each team’s `players` list

---

## Step 2. Expand both team rosters into rows
Create one row per player listed under:

- home team players
- away team players

Each row should immediately get:
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
- optional roster metadata

This creates the raw skeleton of `game_team_player`.

---

## Step 3. Join pregame player state
For each row, lookup the latest upstream state:

\[
(\text{player\_id},\ \text{latest asof\_ts such that asof\_ts < game\_ts})
\]

Join in:
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

---

## Step 4. Attach realized same-game metadata
From the game summary player entry, attach:
- `played_in_game`
- `minutes_in_game`
- `not_playing_reason`
- `not_playing_description`

Again, these are for audit and downstream checks, not for pregame prediction inputs.

---

## Step 5. Persist the downstream table
Store one row per:

\[
(\text{game\_id},\ \text{team\_id},\ \text{player\_id})
\]

This becomes the canonical player-level game snapshot table.

---

# 10. Why we are doing it this way

This section explains the reasoning so another agent understands why the design looks like this.

## Why not only include players who played?
Because for injury modeling, some of the most important players are precisely those who:
- were listed
- but did not play

Dropping them would remove the signal we are trying to capture.

## Why keep pregame and postgame columns together?
Because it makes the table auditable.

You can later validate questions like:
- Did a player with a strong injury signal actually fail to play?
- Did `minutes_last_game_pre` predict reduced `minutes_in_game`?
- Are our injury windows behaving sensibly?

## Why copy upstream fields with `_pre` suffix?
Because these values are explicitly **pregame** state values.

The suffix helps keep them distinct from:
- realized same-game outcomes
- later transformed slotted features

## Why include `state_asof_ts`?
So the join is fully auditable and reproducible.

If anything looks wrong downstream, you can trace exactly which upstream player-state row was used.

---

# 11. Default value rules

These apply only if the upstream join fails or the player truly has no history.

## If no upstream state row exists
Use initialized fallback values:

- `m_ewma_pre = 0`
- `q_pre = league-average prior quality`
- `strength_pre = 0`
- injury timing fields = 0
- `injury_present_flag_pre = 0`

The row must still be kept.

## If a player is listed in the game summary but did not play
Then:
- `played_in_game = 0`
- `minutes_in_game = 0`
- preserve any `not_playing_reason` fields if available

---

# 12. What this table is used for next

This table will later be transformed into the final model input by:

1. sorting each team’s players by `strength_pre`
2. selecting the desired top \(N\) player slots
3. flattening those slots into one row per game

For now, however, this table should be built for **all players listed in each team’s game summary** so later experiments can choose:
- top 5
- top 8
- top 12
without rebuilding the upstream/downstream pipeline.

---

# 13. Final schema summary

## Table name
`game_team_player`

## Primary key
- `game_id`
- `team_id`
- `player_id`

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

### Player identity / roster metadata
- `player_id`
- `player_name`
- `listed_on_game_summary_flag`

### Pregame joined state
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

### Same-game realized metadata
- `played_in_game`
- `minutes_in_game`
- `not_playing_reason`
- `not_playing_description`

## Optional metadata
- `jersey_number`
- `position`
