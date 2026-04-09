# WNBA Prediction Market Model

A calibrated sports forecasting and trading system for WNBA prediction markets, designed for both:
1. **research / thesis-quality methodology**, and
2. **practical trading evaluation** on market platforms such as Kalshi / Polymarket.

This README is the current project context document for development, debugging, and future iteration.

---

## 1. Project objectives

### Primary objective
Build a **defensible, portfolio-quality forecasting + trading system** that is:
- methodologically clean
- modular
- auditable
- strong enough for a thesis and GitHub portfolio

### Secondary objective
Produce a **useful pregame win-probability model** for WNBA single-game outcomes.

### Tertiary objective
Use the model in a **prediction-market trading framework** that:
- identifies mispriced YES/NO contracts
- evaluates entry / exit logic
- measures out-of-sample trading performance

---

## 2. Core modeling philosophy

The model is intentionally split into layers:

### Elo baseline
A structural prior for team strength.

### ML correction layer
Learns contextual adjustments to Elo using:
- player availability / injury context
- recent team form
- stylistic tendencies
- rest / travel burden

### Calibration layer
Converts raw corrected probabilities into final tradable probabilities.

The current core equation is:

\[
\operatorname{logit}(p_{raw}) = \operatorname{logit}(p_{elo}) + g(x)
\]

where:
- `p_elo` = Elo baseline probability
- `g(x)` = XGBoost correction function
- `p_raw` = raw corrected probability

Then:

\[
\operatorname{logit}(p_{final}) = a + b \cdot \operatorname{logit}(p_{raw})
\]

using **Platt scaling**.

---

## 3. Final evaluation philosophy

### Development period
- 2015–2024

### Final untouched backtest period
- 2025

### Important exclusion
- exclude the **first 9 games of 2015** from development/evaluation because:
  - no 2014 prior data exists
  - `m_ewma` and `q` are effectively zero/uninformative there

### Walk-forward CV
Use walk-forward validation on 2015–2024 for model / feature selection.

### Calibration
Fit the final calibrator on **OOF predictions from 2020–2024**, not on 2025.

### 2025
2025 is reserved for the final once-only evaluation of the chosen pipeline.

---

## 4. Elo backbone

### Final locked Elo hyperparameters
- `H = 25`
- `K = 20`
- `alpha = 0.45`
- `beta = 1.0`
- `mu = 1505`

### Elo equations

Pregame gap:
\[
d = (R_{home} + H) - R_{away}
\]

Pregame home win probability:
\[
p_{elo} = \frac{1}{1 + 10^{-d/400}}
\]

Margin-of-victory multiplier:
\[
M = \frac{(MOV + 3)^{1.0}}{7.5 + 0.006 d_{win}}
\]

Update:
\[
\Delta = K \cdot M \cdot (y - p_{elo})
\]

Season carryover:
\[
R_{start} = 0.45 R_{end} + 0.55 \mu
\]

### Elo design principles
- zero-sum updates
- single Elo track
- no direct injury adjustment inside Elo
- Elo is the base margin, not just another feature

### In XGBoost
Use:
- `base_margin = logit(p_elo)`

Do **not** include ordinary Elo features inside `X` if using the base-margin setup.

---

## 5. Data sources

### Sports data backbone
- **Sportradar WNBA API**
  - schedule
  - results
  - game summaries / box scores
  - player / team stats
  - injury reports / windows

### Prediction market data
- Kalshi API
- Polymarket API / CLOB

### Current project stance
Model is built against sports data first, then later merged with market data for trading backtests.

---

## 6. Feature families

The final ordinary XGBoost features are grouped into four blocks:

1. **Player injury / availability block**
2. **Recent form block**
3. **Style / matchup-profile block**
4. **Rest / travel block**

Elo sits outside these as `base_margin`.

---

## 7. Player injury / availability block

### Core idea
For each team in each game, players are ranked by:

\[
strength = m_{ewma} \cdot q
\]

Then the top `N_players` slots are fed into the final model.

### Current best Stage 1 result
- `N_players = 7`
- `h_M = 7`
- `L_inj = 14` (chosen as the more stable winner region)

### Locked player-slot input features (9 per player)
1. `m_ewma`
2. `q`
3. `days_since_first_report`
4. `days_since_last_dnp`
5. `consec_dnps`
6. `played_last_game`
7. `minutes_last_game`
8. `days_since_last_played`
9. `injury_present_flag`

### Player role feature: `m_ewma`
Player EWMA minutes.

Current tuned winner:
- `h_M = 7` games (from Stage 1)

Earlier design default was 5, but Stage 1 selected 7.

### Player quality feature: `q`
Season-to-date EFF per minute with prior carryover:

\[
q_{i,t} = \frac{\tau q_i^{prev} + EFF^{curr}_{i,\le t}}{\tau + MIN^{curr}_{i,\le t}}
\]

Current default / Stage 2 candidate:
- `tau = 150`

### Injury inclusion window
A player is injury-present if inside the window:

- within `L_inj` days of the most recent of:
  - last injury report date
  - last DNP date

Current tuned winner:
- `L_inj = 14`

### Slot ordering
Within each `(game_id, team_id)`:
1. sort by `strength_pre` descending
2. tie-break by `m_ewma_pre` descending
3. then `q_pre` descending
4. then `player_id` ascending

### Missing slots
If a team has fewer than `N_players` available slots:
- use `NULL`, not zero

### Important
`strength_pre` is included for:
- ordering
- debugging

but **not** as a model input, because it is deterministic from `m_ewma * q`.

---

## 8. Recent form block

### Philosophy
Recent form should capture how a team has been playing **lately this season**, not long-run structural strength.

So:
- recent form **resets to 0 each season**
- unlike player priors, it does **not** carry over

This asymmetry is intentional:
- player role / quality are partly persistent across seasons
- recent form is a same-season short-memory concept

### Team recent-form features (5 per team)
1. `net_rating_ewma`
2. `efg_ewma`
3. `tov_pct_ewma`
4. `orb_pct_ewma`
5. `ftr_ewma`

### Input structure
Feed home and away separately:
- `home_net_rtg_ewma_pre`, `away_net_rtg_ewma_pre`, etc.

Do **not** collapse to deltas only.

### Team recent-form EWMA half-life
Current fixed default / Stage 2 search set:
- `h_team ∈ {5, 7, 10}`
- provisional current default: `h_team = 7`

---

## 9. Style / matchup-profile block

### Philosophy
This block captures **current-season team tendencies**, not recent hot/cold form.

So:
- use **season-to-date current-season values**
- **hard reset each season**
- but for **game 1 only**, initialize from the previous season’s final values

### Raw per-team style inputs (6 per team)
1. `off_3pa_rate_pre`
2. `def_3pa_allowed_pre`
3. `off_2pa_rate_pre`
4. `def_2pa_allowed_pre`
5. `off_tov_pct_pre`
6. `def_forced_tov_pre`

### Why raw stats instead of 3 mismatch deltas
We explicitly chose:
- 6 raw stats per team
- rather than 3 hand-compressed mismatch features

Reason:
- preserves more information
- lets XGBoost learn interactions itself

### Initialization rule
For each season:
- game 1 pregame values = previous season final values
- games 2+ = pure current-season season-to-date through previous game

For the first modeled season (2015), where 2014 is unavailable:
- use league-average initialization constants computed from 2015 full-season totals

---

## 10. Rest / travel block

### Features (6 per team)
1. `days_rest_pre`
2. `is_b2b_pre`
3. `games_last_4_days_pre`
4. `games_last_7_days_pre`
5. `travel_miles_pre`
6. `timezone_shift_hours_pre`

### Travel origin rule
For each team-game:

\[
origin\_city =
\begin{cases}
\text{previous game city}, & \text{if } days\_rest < 4 \\
\text{home city}, & \text{if } days\_rest \ge 4
\end{cases}
\]

This is the locked travel rule.

### Timezone encoding
Use:
- signed `timezone_shift_hours_pre`

We decided **not** to tune absolute timezone encoding in the main path.

---

## 11. Table / pipeline structure

### Upstream historical state
#### `player_state_history`
Key:
- `(player_id, asof_ts)`

Contains:
- `m_ewma`
- `q`
- `strength`
- injury-window timing features
- participation state

### Downstream player game snapshot
#### `game_team_player`
Key:
- `(game_id, team_id, player_id)`

Contains:
- every listed player in the official game summary, even if they did not play
- joined pregame player-state features
- same-game realized metadata for debugging

### Team recent form
#### `game_franchise_recent_form`
Key:
- `(game_id, franchise_id)` or equivalent implementation
- conceptually one row per team per game

### Team style profile
#### `game_franchise_style_profile`
Key:
- `(game_id, franchise_id)` or equivalent implementation
- conceptually one row per team per game

### Team schedule context
#### `game_team_schedule_context`
Key:
- `(game_id, team_id)`

### Final gold table
#### `game_xgboost_input`
Key:
- `game_id`
- one row per game

---

## 12. Final XGBoost input layout

### Metadata columns (not ordinary model inputs)
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
- `home_win` (target)

### Ordinary model-input blocks
1. home player slots
2. away player slots
3. recent form block
4. style block
5. rest / travel block

### Current feature count with `N_players = 7`
- player block: \(7 \times 9 \times 2 = 126\)
- recent form: 10
- style: 12
- rest/travel: 12

Total ordinary XGBoost input columns:

\[
126 + 10 + 12 + 12 = 160
\]

`base_margin` is passed separately, not inside `X`.

---

## 13. Calibration layer

### Current design
Use **Platt scaling**:

\[
\operatorname{logit}(p_{final}) = a + b \cdot \operatorname{logit}(p_{raw})
\]

### Calibration workflow
- generate OOF raw predictions on 2020–2024
- fit Platt scaling on those OOF predictions
- do **not** touch 2025 yet until the full chosen pipeline is frozen

### Important
At the current stage:
- no ensemble yet
- calibration notebook is for the **single-model** pipeline only

---

## 14. Ensemble layer

### Current status
Not implemented yet.

### Planned design
Once the single-model pipeline is frozen and validated:

- train multiple bootstrap models
- each uses the same:
  - feature design
  - Elo config
  - XGBoost hyperparameters
- diversity comes from bootstrap resampling
- combine member predictions into:
  - ensemble median probability
  - prediction dispersion / confidence

### Current recommendation
Given the final architecture, **row bootstrap** is a reasonable default.  
We no longer think time-block bootstrap is strictly necessary because all features are already explicitly “as-of.”

---

## 15. Tuning strategy

### Elo (already finalized)
Final locked Elo:
- `H = 25`
- `K = 20`
- `alpha = 0.45`
- `beta = 1.0`
- `mu = 1505`

### Stage 1 feature/model joint tuning
Searched over:
- `N_players ∈ {5, 7, 10}`
- `h_M ∈ {3, 5, 7, 10}`
- `L_inj ∈ {7, 14, 21}`

with a conservative XGBoost grid.

Best region found:
- `N_players = 7`
- `h_M = 7`
- `L_inj ≈ 14`

### Stage 2 feature tuning

- `h_team = 7`
- `tau = 150`

### Stage 3 XGBoost tuning
Current best Stage 3 config used in evaluation:
- `max_depth = 6`
- `min_child_weight = 3`
- `colsample_bytree = 0.6`
- `subsample = 0.9`
- `reg_lambda = 1`
- `reg_alpha = 0`
- `learning_rate = 0.02`

---

## 16. Evaluation philosophy

### Primary metric
- log loss

### Secondary metrics
- Brier score
- calibration diagnostics
- accuracy (reported, but not primary)

### Clean benchmark sequence
1. Elo-only
2. Elo + XGBoost raw
3. Elo + XGBoost calibrated
4. later: ensemble

### Final untouched evaluation
- 2025 only

---

## 17. Trading layer (current design direction)

The trading layer should be **event-driven**, not just a single check after market creation.

Core logic:

1. compute:
   \[
   edge(t) = p_{final}(t) - q(t)
   \]
2. trade only if:
   - edge exceeds a cost buffer
   - model confidence is high enough
3. later, ensemble dispersion will feed the confidence / certainty score

### Important
The trading layer is **not fully locked yet**.  
Market microstructure choices such as:
- exact price source \(q(t)\)
- entry/exit thresholds
- cost buffer
- convergence logic
remain to be finalized.

---

## 18. Special edge cases and important notes

### San Antonio Stars → Las Vegas Aces
Treat the 2018 San Antonio-to-Las Vegas move as **franchise continuity**, not a brand-new team.

Cross-season priors / carryover should follow franchise continuity, not raw team ID alone.

### First 9 games of 2015
Exclude them from development/evaluation because:
- no 2014 prior data exists
- `m_ewma` and `q` are uninformative there

### Recent form asymmetry
Recent form resets each season, while player priors do not.  
This is intentional.


---

## 19. Project summary

This project is a **layered, interpretable forecasting and trading system**:

- Elo provides the structural baseline
- XGBoost learns contextual corrections
- calibration converts raw corrections into tradable probabilities
- later, an ensemble will quantify uncertainty
- the trading layer will exploit probability-vs-market edge when confidence is sufficient

The overall goal is not just raw predictive performance, but a clean, defensible, modular system suitable for:
- thesis presentation
- portfolio use
- and realistic market evaluation
