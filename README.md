# WNBA Prediction Market Model

A sports outcome forecasting and trading model for WNBA prediction markets on Kalshi, implementing Elo structural priors, Gradient-Boosted Trees with in-depth feature engineering, no look-ahead bias and practical market application.

---

## Key Results

### Forecasting Performance

Walk-forward cross-validation on 2020–2024 (development), with final holdout evaluation on the untouched 2025 season.

| Model | Dev Log Loss | Dev Brier | Dev Accuracy | Holdout Log Loss | Holdout Brier | Holdout Accuracy |
|-------|-------------|-----------|--------------|------------------|---------------|------------------|
| Elo-only | 0.6022 | 0.2072 | 67.8% | 0.6151 | 0.2132 | 66.8% |
| XGBoost + Elo | **0.5994** | **0.2055** | **69.5%** | **0.6121** | **0.2112** | **67.4%** |
| XGBoost (no Elo) | 0.6228 | 0.2165 | 65.1% | 0.6327 | 0.2188 | 66.8% |
| Logistic Reg + Elo | 0.7322 | 0.2285 | 65.9% | 0.6684 | 0.2332 | 64.8% |

The XGBoost + Elo model improves over the Elo baseline in both development (−0.28 log loss points) and on the untouched 2025 holdout (−0.30 points), with a consistent accuracy advantage. XGBoost without Elo is substantially worse, confirming the Elo-as-base-margin architecture. Logistic regression has the worst performance of all models, confirming the importance of models learning non-linear relationships.

Full model comparison, per-fold breakdowns, feature importance, and calibration diagnostics: [`notebooks/analysis/forecasting_results.ipynb`](notebooks/analysis/forecasting_results.ipynb).

### Trading Performance (2025 Kalshi Backtest)

Best configurations from a grid search over entry rules and position sizing, using half-life entry timing (~17h pre-tipoff) on Kalshi settlement markets. Drawdowns are assuming $100 starting balance.

| Model | Sizing | Trades | Hit Rate | Mean Edge | ROI / Return | Max Drawdown |
|-------|--------|--------|----------|-----------|-------------|--------------|
| Elo | Fixed $1 | 68 | 39.7% | 18.7% | 27.8% ROI | — |
| Full model | Fixed $1 | 59 | 52.5% | 19.2% | **36.1% ROI** | — |
| Elo | Half-Kelly | 155 | 34.2% | 12.7% | 417% return | $1,096 |
| Full model | Half-Kelly | 134 | 40.3% | 13.4% | **1,062% return** | $2,706 |

### Trading Strategy

The trading system is a **pre-game entry, hold-to-settlement** strategy on Kalshi moneyline markets. For each WNBA game with an active Kalshi market:

1. **Entry window.** Begin scanning at the market's half-life (~17 hours pre-tipoff), when spreads have settled to 2–3 cents but prices still reflect meaningful model edge. Snapshots are evaluated every 15 minutes.

2. **Side selection.** At each snapshot, compute edge on both YES and NO sides:
   - `edge_yes = p_model − ask_yes`
   - `edge_no = (1 − p_model) − (1 − bid_yes)`
   - Take the side with the larger edge

3. **Entry filters.** A trade is placed at the *first* qualifying snapshot where:
   - **Absolute edge** ≥ threshold (grid-searched over 5, 10, 15 cents)
   - **Normalized edge** (`edge / entry_price`) ≥ threshold (grid-searched over 0, 10, 20, 25%)

4. **Position sizing.**
   - *Fixed $1*: risk exactly $1 per trade, isolating model edge from compounding effects
   - *Half-Kelly*: `f* = (p_model − entry_price) / (1 − entry_price)`, wager `= (f*/2) × bankroll`. Sizes proportionally to perceived edge while halving the theoretically optimal fraction to reduce variance
   - *Full-Kelly*: same formula with `f*` instead of `f*/2` (tested but rejected — too aggressive)

5. **Exit.** All positions are held to settlement. Pre-tipoff convergence exits were tested but trigger on <2% of positions; real edge is captured at settlement.

6. **Fees.** Kalshi taker fee is applied at entry: `ceil(0.07 × n × p × (1−p) × 100) / 100`. No exit fees.

The full grid search evaluates 144 configurations (3 edge thresholds × 4 normalized-edge thresholds × 2 entry windows × 3 sizing methods × 2 models). Details: [`notebooks/analysis/trading_results2.ipynb`](notebooks/analysis/trading_results2.ipynb).

### Statistical Significance

A bootstrap comparison of per-trade log-returns (10K resamples) tests whether the full model's growth-rate advantage over Elo is robust:

| Metric | Value |
|--------|-------|
| Full model mean log-return | +0.0183 per trade |
| Elo mean log-return | +0.0106 per trade |
| P(Full Model > Elo) | 0.647 |
| Growth-rate difference 95% CI | [−0.033, +0.048] |

With ~130–155 trades in a single season, the difference is directionally consistent but not statistically significant at conventional levels. Roughly 2–3 seasons of similar performance would be needed for significance.

Full trading analysis: [`notebooks/analysis/trading_results2.ipynb`](notebooks/analysis/trading_results2.ipynb).

### Model vs Market Comparison

On the 2025 holdout, the model's predictions are compared head-to-head against Kalshi and Polymarket pre-tipoff implied probabilities:

| Source | n | Log Loss | Brier | Accuracy |
|--------|---|----------|-------|----------|
| XGB + Elo (model) | 366 | 0.620 | 0.214 | 66.1% |
| Elo only | 366 | 0.619 | 0.214 | 67.2% |
| Kalshi pre-tipoff | 349 | 0.612 | 0.213 | 63.0% |
| Polymarket pre-tipoff | 277 | 0.674 | 0.237 | 62.5% |

On the common subset (263 games with all four sources), the model and Kalshi are closely matched on calibration while the model maintains higher accuracy.

When the model and Kalshi disagree on the game direction (59 games), the model is correct **61%** of the time.

---

## Analysis & Interpretation

### Elo captures most of the signal

The most striking result in the forecasting table is that XGBoost *without* Elo — using only player, form, style, and schedule features — achieves a dev log loss of 0.623, within 0.021 of the Elo-only baseline (0.602). These two models use completely different data sources and methodologies: Elo sees only game outcomes and margin of victory, while the features-only XGBoost sees player availability, box-score tendencies, rest patterns, and team style. The fact that they converge to similar performance suggests that Elo already encodes much of what matters, team strength is the dominant signal, and contextual features provide only a marginal correction.

This is further confirmed by feature importance. When XGBoost has no Elo base margin, it learns sensible structure: net rating EWMA and top-player quality (`p1_q`, `p2_q`) dominate importance, essentially reconstructing a team-strength signal from available data. When XGBoost *does* have Elo as a base margin, the remaining feature importance is scattered across low-level player slots (e.g., `home_p2_played_last_game`, `away_p5_days_since_last_played`) with no clear interpretable pattern — it is fitting noise around an already-strong prior. The logistic regression tells the same story: `base_margin` has a coefficient of 0.92 (nearly 1.0, meaning Elo is passed through almost unchanged), and the largest feature coefficients are schedule and player availability variables with modest magnitude.

### Small log loss improvements, large trading returns

The most counterintuitive result is the gap between forecasting and trading performance. On the 2025 holdout, the full model's log loss improvement over Elo is modest (0.6121 vs 0.6151 — just 0.003 points), yet it produces **1,062% half-Kelly return** vs Elo's **417%** — a 2.5x difference in terminal wealth from a nearly negligible calibration improvement.

Several mechanisms may explain this:

1. **Tail accuracy matters more than average calibration.** Log loss averages over all games equally. But trading only occurs on games where the model sees sufficient edge — typically 30–40% of the season. If the model is even slightly more accurate in these high-edge games (where predictions diverge most from market prices), the trading returns compound dramatically even though the average log loss barely changes.

2. **Trade selection is itself a prediction.** The full model doesn't just predict differently — it *selects different games to trade*. With its best entry rule, it takes 134 trades at a 40.3% hit rate, while Elo takes 155 trades at 34.2%. Under its tightest filter (fixed $1), the model achieves a 52.5% hit rate on 59 trades vs Elo's 39.7% on 68. Fewer, higher-conviction bets with a substantially higher win rate is exactly what Kelly sizing rewards exponentially.

3. **The model-vs-market disagreement edge.** When the model and Kalshi disagree on the favored team (59 games), the model is correct 61% of the time. These disagreement games are precisely the games with the largest perceived edge, and the ones that drive the bulk of trading returns. Even a small accuracy advantage in this specific subset can translate to large compounding gains.

4. **Half-Kelly amplifies small edges.** Kelly sizing is exponential in the number of positive-edge bets. A model that identifies even slightly better opportunities — higher mean edge, better side selection, or fewer losing trades — compounds that advantage multiplicatively across 130+ bets, turning a statistically insignificant per-trade improvement into an economically meaningful difference in terminal wealth.

### The honest uncertainty

Despite the compelling return numbers, the bootstrap significance test gives P(Full Model > Elo) = 0.647 — suggestive but far from conclusive. A single season of ~130–155 trades is simply insufficient to statistically distinguish two models that both have positive edge. This is a structural limitation of WNBA market size, not a modeling failure.

### Future research directions

- **Multi-season validation.** The most direct path to significance: 2–3 additional seasons of Kalshi WNBA data under the same pipeline would dramatically tighten the confidence interval.
- **Cross-sport transfer.** Testing the same Elo + XGBoost architecture on NBA or other leagues with deeper markets could validate whether the approach generalizes. An initial NBA scaffold exists in this repository.
- **In-play model.** The current system is pre-tipoff only. A live model that updates with in-game information could capture additional edge, particularly for second-half or live markets.
- **Ensemble with market prices.** Rather than treating market prices as the adversary, incorporating pre-tipoff Kalshi/Polymarket implied probabilities as features could improve calibration — the market captures information (injury rumors, sharp money, lineup leaks) that the model's feature set may miss.
- **Disentangling the trading advantage.** A controlled study isolating trade selection vs. probability accuracy vs. Kelly sizing would clarify which mechanism drives the return gap between the full model and Elo. This could inform whether to invest in better features or better entry rules.

---

## Modeling Architecture

The model is a **two-layer system**:

```
logit(p_raw) = logit(p_elo) + g(x)          # XGBoost correction on Elo base margin
```

### Layer 1: Elo Baseline
Margin-of-victory Elo with home-court advantage, season carryover, and franchise continuity. Provides a structural prior for team strength. See [`CLAUDE.md` §4](CLAUDE.md) for full Elo equations.

### Layer 2: XGBoost Correction
Learns contextual adjustments using 160 pregame features across four blocks:

| Block | Features | Description | Spec sheet |
|-------|----------|-------------|------------|
| Player availability | 126 (7 slots × 9 features × 2 teams) | EWMA minutes, quality rating, injury status, participation history | [`player_state_history_spec.md`](data/spec_sheets/player_state_history_spec.md), [`game_team_player_spec.md`](data/spec_sheets/game_team_player_spec.md) |
| Recent form | 10 (5 × 2 teams) | EWMA net rating, eFG%, TOV%, ORB%, FTr | [`game_team_recent_form_spec.md`](data/spec_sheets/game_team_recent_form_spec.md) |
| Style profile | 12 (6 × 2 teams) | Season-to-date shooting tendencies, turnover rates | [`game_team_style_profile_spec.md`](data/spec_sheets/game_team_style_profile_spec.md) |
| Rest / travel | 12 (6 × 2 teams) | Days rest, back-to-back, travel miles, timezone shift | [`game_team_schedule_context_spec.md`](data/spec_sheets/game_team_schedule_context_spec.md) |

Elo probability is passed as `base_margin`, not as an ordinary feature. Full gold table layout: [`game_xgboost_input_spec.md`](data/spec_sheets/game_xgboost_input_spec.md).

## Hyperparameter Tuning

Three-stage tuning strategy with walk-forward CV. See [`docs/tuning_methodology.md`](docs/tuning_methodology.md) for full details including search grids and the Stage 3 top-10 configuration table.

**Final locked parameters:**

| Component | Parameters |
|-----------|-----------|
| Elo | H=25, K=20, α=0.45, β=1.0, μ=1505 |
| Features | N_players=7, h_M=7, L_inj=14, τ=150, h_team=7 |
| XGBoost | max_depth=6, mcw=3, γ=0.1, cbt=0.6, sub=0.8, λ=1.0, α=0.0, lr=0.02 |

All hyperparameters are also defined in [`config/final_hyperparams.py`](config/final_hyperparams.py).

The XGBoost configuration was chosen as rank 2 out of 1,296 candidates in the refined grid search. The rank-1 config (lr=0.03) was rejected due to unstable early stopping (min_best_round=2 in one fold), while rank 2 (lr=0.02) showed consistent convergence across all folds (min_best_round=39) with only 0.00038 higher mean log loss.

---

## Exploration Summary

Several alternative approaches were investigated and excluded from the final pipeline. These are documented in [`notebooks/scratchwork/`](notebooks/scratchwork/) for completeness — see [`notebooks/scratchwork/README.md`](notebooks/scratchwork/README.md) for details.

| Approach | Finding | Notebook |
|----------|---------|----------|
| **Polymarket trading** | Thin WNBA liquidity, wide spreads (10–20+ cents) | `scratchwork/poly_trading.ipynb` |
| **Pre-tipoff convergence exits** | Prices rarely move enough pre-game (0–2% trigger rate) | `scratchwork/trading_results.ipynb` |
| **Bootstrap ensemble** | Did not meaningfully improve over the single model | `scratchwork/ensemble_comparison.ipynb` |
| **Neural network (MLP)** | Did not outperform XGBoost; higher variance across folds | `scratchwork/NN_test.ipynb` |
| **XGBoost without Elo** | Worse than Elo + XGBoost, confirming base-margin design | `scratchwork/XGBpure.ipynb` |
| **Full-Kelly sizing** | Too aggressive at 35–44% hit rates; ruin risk | `analysis/trading_results2.ipynb` §7 |
| **Two-thirds-life entry** | Half-life (~17h) consistently outperformed (~12h) | `analysis/trading_results2.ipynb` §8 |

### Key market microstructure findings

- Kalshi WNBA markets open ~35 hours pre-tipoff on average
- Volume ramps dramatically toward tipoff: <5 contracts/hour at open, 50+ near game time
- Spreads compress from 20+ cents at open to 1 cent near tipoff
- Optimal entry: half-life (~17h pre-tipoff), where spreads are 2–3 cents but prices still reflect model edge
- Kalshi taker fee: `ceil(0.07 * n * p * (1-p) * 100) / 100`

---

## Repository Structure

```
organized/
├── config/
│   └── final_hyperparams.py        # Single source of truth for all hyperparameters
├── src/srwnba/                     # Core library (API client, Elo engine, franchise mapping)
├── utils/                          # Market API clients (Kalshi, Polymarket)
├── pipelines/                      # Numbered data pipeline scripts
│   ├── 01_ingestion/               # Sportradar API → bronze JSON
│   ├── 02_parsing/                 # Bronze JSON → silver CSVs
│   ├── 03_features/                # Silver → feature tables
│   ├── 04_gold/                    # Feature assembly → XGBoost input (160 features)
│   ├── 05_modeling/                # XGBoost CV, calibration, Elo tuning
│   └── 06_markets/                 # Kalshi & Polymarket data ingestion
├── notebooks/
│   ├── analysis/                   # Final result notebooks
│   │   ├── forecasting_results.ipynb   # Model comparison & holdout evaluation
│   │   ├── trading_results2.ipynb      # Kalshi trading backtest & significance testing
│   │   └── prelim.ipynb                # Preliminary data exploration
│   ├── xgb_tuning/                 # XGBoost tuning (Stage 3)
│   │   ├── XGB_tuning3.ipynb           # Final Stage 3 grid search
│   │   └── complexity_curve.ipynb
│   └── scratchwork/                # Exploration notebooks (see scratchwork/README.md)
├── data/
│   ├── spec_sheets/                # Table and feature specifications
│   │   ├── player_state_history_spec.md
│   │   ├── game_team_player_spec.md
│   │   ├── game_team_recent_form_spec.md
│   │   ├── game_team_style_profile_spec.md
│   │   ├── game_team_schedule_context_spec.md
│   │   ├── game_xgboost_input_spec.md
│   │   ├── kalshi_api_schema.md
│   │   └── polymarket_ingest_spec.md
│   ├── config/                     # Static config (franchise_map.csv)
│   └── xgb_stage3_top10.csv        # Top 10 XGB configs from Stage 3
├── docs/
│   └── tuning_methodology.md       # Full tuning strategy with search grids
├── CLAUDE.md                       # Detailed methodology specification
├── CHANGELOG.md                    # Inclusion/exclusion decisions log
└── requirements.txt
```

---

## Data Pipeline

All pipeline scripts are CLI tools. Run from the `organized/` directory.

### 1. Ingestion (`pipelines/01_ingestion/`)
Fetches raw data from Sportradar WNBA API (schedules, game summaries, daily injuries) for each year 2015–2025.

### 2. Parsing (`pipelines/02_parsing/`)
Normalizes bronze JSON into silver CSVs: game outcomes, player box scores, injury events, availability records.

### 3. Feature Engineering (`pipelines/03_features/`)
Builds feature tables: Elo ratings, player state history (EWMA minutes, quality scores), recent form, style profiles, schedule context. See the [spec sheets](data/spec_sheets/) for column-level documentation of each feature table.

### 4. Gold Assembly (`pipelines/04_gold/`)
Assembles the final 160-feature XGBoost input table with `base_margin = logit(p_elo)`. Layout documented in [`game_xgboost_input_spec.md`](data/spec_sheets/game_xgboost_input_spec.md).

### 5. Modeling (`pipelines/05_modeling/`)
Walk-forward XGBoost CV, Platt scaling calibration, Elo grid search.

### 6. Market Data (`pipelines/06_markets/`)
Kalshi and Polymarket market ingestion, matching to Sportradar game IDs.

### Sportradar API Requirements

Requires a **Sportradar WNBA API** key. The pipeline uses three endpoints:

| Endpoint | Purpose | Bronze files per year |
|----------|---------|----------------------|
| Season Schedule | Game IDs, dates, teams | 2 (REG + PST) |
| Game Summary | Scores, box stats, rosters | ~200–250 |
| Daily Injuries | Player injury reports | ~150–180 |

Total: ~4,300 bronze files across 2015–2025.

---

## Environment Setup

```bash
conda create -n kalshi-wnba python=3.11
conda activate kalshi-wnba
pip install -r requirements.txt
```

Requires a `.env` file with `SPORTRADAR_API_KEY` (see `.env.example`).

---

## Key Design Decisions

1. **Elo as base_margin, not a feature.** Elo provides the structural prior; XGBoost learns corrections on top of it. This is more principled than including Elo as just another feature — confirmed by the XGBoost-without-Elo benchmark performing substantially worse.

2. **Walk-forward CV, not k-fold.** Sports data is temporal. Using future data to predict past games would be leakage.

3. **Separate calibration period.** Platt scaling is fit on OOF predictions from 2020–2024, not on the 2025 holdout.

4. **Pre-tipoff only trading.** All entry decisions happen before tipoff. In-game price movements reflect live information, not pregame model edge.

5. **Franchise continuity.** The San Antonio Stars → Las Vegas Aces (2018) relocation is treated as franchise continuity, preserving Elo and player priors across the move.

6. **First 9 games of 2015 excluded.** No 2014 prior data exists, so EWMA and quality features are uninformative for these games.
