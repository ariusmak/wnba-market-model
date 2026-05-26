# WNBA Prediction Market Model

A WNBA win-probability and market-evaluation project combining a margin-of-victory Elo prior, as-of feature engineering, XGBoost corrections, walk-forward validation, and Kalshi settlement-market backtests.

---

## Key Results

### Forecasting Performance

Development results use walk-forward cross-validation on 2020–2024; 2025 is held out for final evaluation.

| Model | Dev Log Loss | Dev Brier | Dev Accuracy | 2025 Log Loss | 2025 Brier | 2025 Accuracy |
|-------|-------------|-----------|--------------|---------------|------------|---------------|
| Elo-only | 0.6022 | 0.2072 | 67.8% | 0.6151 | 0.2132 | 66.8% |
| XGBoost + Elo | **0.5994** | **0.2055** | **69.5%** | **0.6121** | **0.2112** | **67.4%** |
| XGBoost (no Elo) | 0.6228 | 0.2165 | 65.1% | 0.6327 | 0.2188 | 66.8% |
| Logistic Reg + Elo | 0.7322 | 0.2285 | 65.9% | 0.6684 | 0.2332 | 64.8% |

The full model improves over Elo by −0.0028 log loss in development and −0.0030 on the 2025 holdout. XGBoost without Elo performs substantially worse, supporting the base-margin architecture.

### Model vs Market

Pre-tipoff Kalshi and Polymarket implied probabilities are evaluation targets, not model inputs.

| Source | n | Log Loss | Brier | Accuracy |
|--------|---|----------|-------|----------|
| XGB + Elo (model) | 310 | 0.612 | 0.211 | 67.4% |
| Elo only | 310 | 0.615 | 0.213 | 66.8% |
| Kalshi pre-tipoff | 296 | 0.612 | 0.213 | 62.8% |
| Polymarket pre-tipoff | 221 | 0.626 | 0.219 | 64.3% |

On the common 210-game subset with all sources, Kalshi has lower log loss (0.619 vs 0.625), the model has higher accuracy (67.6% vs 62.9%), and Brier scores are effectively tied (0.2163 vs 0.2162). On 51 model-vs-Kalshi directional disagreements, the model is correct 65% of the time, a useful but small trade-selection slice.

### Trading Backtest

These are retrospective 2025 policy-selection results from a grid over entry rules and sizing, not a separately validated live policy. Fixed-$ rows report ROI on dollars staked; Kelly rows report bankroll return and max drawdown from a $100 starting balance.

| Model | Sizing | Trades | Hit Rate | Mean Edge | Result | Max Drawdown |
|-------|--------|--------|----------|-----------|--------|--------------|
| Elo | Fixed $1 | 68 | 39.7% | 18.7% | 27.8% ROI | — |
| Full model | Fixed $1 | 59 | 52.5% | 19.2% | **36.1% ROI** | — |
| Elo | Half-Kelly | 155 | 34.2% | 12.7% | 417% return | $1,096 |
| Full model | Half-Kelly | 134 | 40.3% | 13.4% | **1,062% return** | $2,706 |

The trading edge comes primarily from game selection. On games traded by both models, the full model does not outperform Elo; the profit gap comes from suppressing low-quality Elo trades and adding a smaller set of high-performing model-only trades.

---

## Result Audit Trail

Main claims are backed by regenerated tables in [`outputs/`](outputs/) and source notebooks in [`notebooks/analysis/`](notebooks/analysis/) or [`notebooks/output generation/`](notebooks/output%20generation/).

| Claim area | Primary table(s) | Primary notebook(s) |
|------------|------------------|---------------------|
| Forecasting benchmark | [`forecast_model_performance_summary.csv`](outputs/forecast_model_performance_summary.csv), [`forecast_per_fold_performance_table.csv`](outputs/forecast_per_fold_performance_table.csv) | [`forecasting_results.ipynb`](notebooks/analysis/forecasting_results.ipynb), [`forecasting.ipynb`](notebooks/output%20generation/forecasting.ipynb) |
| Feature ablations / robustness | [`feature_block_ablation_summary.csv`](outputs/feature_block_ablation_summary.csv), [`training_windows_logloss_summary.csv`](outputs/training_windows_logloss_summary.csv) | [`ablations.ipynb`](notebooks/analysis/ablations.ipynb), [`training_windows.ipynb`](notebooks/analysis/training_windows.ipynb) |
| Market comparison | [`market_model_performance_summary.csv`](outputs/market_model_performance_summary.csv), [`market_directional_disagreement_table.csv`](outputs/market_directional_disagreement_table.csv) | [`market_comparison.ipynb`](notebooks/output%20generation/market_comparison.ipynb) |
| Trading grid / decomposition | [`trade_half_kelly_best_config_table.csv`](outputs/trade_half_kelly_best_config_table.csv), [`trade_return_decomposition_table.csv`](outputs/trade_return_decomposition_table.csv) | [`trading_results2.ipynb`](notebooks/analysis/trading_results2.ipynb), [`return_decomposition.ipynb`](notebooks/output%20generation/return_decomposition.ipynb) |
| Liquidity / execution sensitivity | [`liq_execution_summary_table.csv`](outputs/liq_execution_summary_table.csv), [`trade_volume_share_summary_table.csv`](outputs/trade_volume_share_summary_table.csv), [`trade_cutoff_sweep_summary.csv`](outputs/trade_cutoff_sweep_summary.csv) | [`return_investigation.ipynb`](notebooks/analysis/return_investigation.ipynb), [`liquidity.ipynb`](notebooks/output%20generation/liquidity.ipynb) |
| Statistical significance | [`sig_bootstrap_summary_table.csv`](outputs/sig_bootstrap_summary_table.csv) | [`significance.ipynb`](notebooks/output%20generation/significance.ipynb) |

---

## Model Design

The model is a two-layer system:

```text
logit(p_raw) = logit(p_elo) + g(x)
```

`p_elo` is a structural team-strength prior; `g(x)` is an XGBoost correction learned from player availability, recent form, team style, and rest/travel features.

### Elo Baseline

Final tuned parameters: `H = 25`, `K = 20`, `alpha = 0.45`, `beta = 1.0`, `mu = 1505`.

```text
d = (R_home + H) - R_away
p_elo = 1 / (1 + 10^(-d / 400))

d_abs = |(R_home + H) - R_away|
M = ((MOV + 3)^beta) / (7.5 + 0.006 * d_abs)
delta = K * M * (home_win - p_elo)

R_home_post = R_home + delta
R_away_post = R_away - delta
R_next_season_start = alpha * R_season_end + (1 - alpha) * mu
```

`home_win` is 1 for a home win and 0 otherwise; `MOV` is absolute score margin. Updates are zero-sum within each game, with season carryover and franchise continuity.

### XGBoost Correction

Elo probability is passed as XGBoost `base_margin`, not as an ordinary feature. The ordinary feature matrix has 160 pregame, as-of columns:

| Block | Features | Description | Spec sheet |
|-------|----------|-------------|------------|
| Player availability | 126 (7 slots × 9 features × 2 teams) | EWMA minutes, quality rating, injury status, participation history | [`player_state_history_spec.md`](data/spec_sheets/player_state_history_spec.md), [`game_team_player_spec.md`](data/spec_sheets/game_team_player_spec.md) |
| Recent form | 10 (5 × 2 teams) | EWMA net rating, eFG%, TOV%, ORB%, FTr | [`game_team_recent_form_spec.md`](data/spec_sheets/game_team_recent_form_spec.md) |
| Style profile | 12 (6 × 2 teams) | Season-to-date shooting tendencies, turnover rates | [`game_team_style_profile_spec.md`](data/spec_sheets/game_team_style_profile_spec.md) |
| Rest / travel | 12 (6 × 2 teams) | Days rest, back-to-back, travel miles, timezone shift | [`game_team_schedule_context_spec.md`](data/spec_sheets/game_team_schedule_context_spec.md) |

Full model-input layout: [`game_xgboost_input_spec.md`](data/spec_sheets/game_xgboost_input_spec.md).

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Elo as `base_margin` | XGBoost learns corrections on top of a calibrated structural prior; the no-Elo benchmark performs worse. |
| Walk-forward CV | Sports outcomes are temporal; k-fold CV would leak future seasons into past predictions. |
| Pregame-only trading | In-game prices reflect live information outside the pregame feature set. |
| Franchise continuity | San Antonio Stars → Las Vegas Aces (2018) preserves Elo and player priors. |
| Cold-start exclusion | Earliest 2015 rows with zero top-player EWMA minutes are dropped because no 2014 player priors exist. |

---

## Validation and Robustness

### Feature Ablations

Feature-block ablations keep the Elo base margin and final XGBoost configuration fixed. Every block removal worsens 2025 log loss relative to the full model.

| Model variant | 2025 Log Loss | Δ vs full | Interpretation |
|---------------|---------------|-----------|----------------|
| Full model | **0.6121** | — | Elo + all contextual feature blocks |
| No player block | 0.6144 | +0.0023 | Player availability contributes, but modestly |
| No style block | 0.6171 | +0.0050 | Style features add a small correction |
| No recent-form block | 0.6251 | +0.0130 | Recent form is a larger holdout contributor |
| No rest/travel block | 0.6259 | +0.0137 | Schedule context is a larger holdout contributor |

The same direction holds in 2020–2024 OOF results, though with smaller deltas. This supports the feature architecture without implying every individual feature is stable or causal.

### Training Windows and Calibration

| Training window | OOF Log Loss | 2025 Log Loss |
|-----------------|--------------|---------------|
| Expanding from 2015 | **0.5994** | **0.6121** |
| Expanding from 2018 | 0.6041 | 0.6145 |
| Rolling 2-year | 0.6017 | 0.6168 |
| Rolling 3-year | 0.6040 | 0.6171 |

Older WNBA seasons still add signal despite league drift. Platt scaling was evaluated as a calibration diagnostic; it slightly improved pooled OOF calibration but worsened 2025 holdout log loss from 0.61215 to 0.61298, so final trading results use raw XGBoost + Elo probabilities.

### Statistical Uncertainty

Bootstrap comparison of per-trade log-returns (10K resamples):

| Metric | Value |
|--------|-------|
| Full model mean log-return | +0.0183 per trade |
| Elo mean log-return | +0.0106 per trade |
| P(Full Model > Elo) | 0.652 |
| Growth-rate difference 95% CI | [−0.033, +0.048] |

Under a one-sided null-centered bootstrap, the full-model strategy clears the null at `p = 0.111`, Elo at `p = 0.222`, and the full-model − Elo difference at `p = 0.355`. The direction is favorable, but one season of ~130–155 trades is not enough to make a conventional significance claim.

### Scope and Limitations

- **Single-season market test.** Forecasting uses 2020–2024 walk-forward OOF plus a 2025 holdout, but trading evidence comes from one Kalshi season.
- **Execution simulation is approximate.** Historical trade sweeps approximate available liquidity but are not a full order-book replay and do not model market impact from the strategy's own orders.
- **Pregame-only information set.** The model excludes in-game updates and late unstructured news. Late-window diagnostics suggest lineup/news timing matters.
- **Market prices are not model inputs.** Kalshi and Polymarket prices are used for comparison and trading entry, not as forecasting features.

---

## Interpretation

### Elo Carries Most of the Forecasting Signal

XGBoost without Elo reaches 0.623 dev log loss, only 0.021 worse than the Elo-only baseline despite using a completely different information set. Feature importance supports the same pattern: without Elo, XGBoost reconstructs team strength from net rating EWMA and top-player quality; with Elo as `base_margin`, remaining importance is spread across smaller contextual corrections. Logistic regression also passes the Elo signal through almost unchanged (`base_margin` coefficient ≈ 0.92).

### Trading Return Decomposition

The 2025 holdout log-loss improvement over Elo is small (0.6121 vs 0.6151), but the trading backtest return gap is large. Under identical entry rules (`edge ≥ 0.05`, `norm_edge ≥ 0.25`, half-life entry), the gap decomposes as follows:

| Trade partition | Games | Full hit rate | Full P&L | Elo hit rate | Elo P&L |
|-----------------|-------|---------------|----------|--------------|---------|
| Traded by both models (always same side) | 111 | 35.1% | −$1 | 35.1% | +$210 |
| Only the full model traded | 23 | **65.2%** | **+$1,063** | — | — |
| Only Elo traded | 44 | 31.8% | — | 31.8% | +$207 |
| **Total** | — | **40.3%** | **+$1,062** | **34.2%** | **+$417** |

On shared games, the full model does not outperform Elo. The advantage is selection: contextual features suppress 44 low-hit-rate Elo-only trades and surface 23 model-only trades that drive most of the net profit. Because those winners occur later in the season, half-Kelly compounding amplifies their dollar impact; under fixed $1 sizing, the same 23 trades produce only +$15.

---

## Trading Strategy and Execution

The trading system is a pregame, hold-to-settlement strategy on Kalshi WNBA moneyline markets. The full grid evaluates 144 configurations: 3 edge thresholds × 4 normalized-edge thresholds × 2 entry windows × 3 sizing methods × 2 models.

| Component | Rule |
|-----------|------|
| Entry window | Scan every 15 minutes beginning at market half-life (~17h pre-tipoff). |
| Side selection | `edge_yes = p_model − ask_yes`; `edge_no = (1 − p_model) − (1 − bid_yes)`; take the larger edge. |
| Entry filters | First snapshot with absolute edge ≥ 5/10/15 cents and normalized edge (`edge / entry_price`) ≥ 0/10/20/25%. |
| Sizing | Fixed $1, half-Kelly, and full-Kelly tested. Full-Kelly was rejected as too aggressive. |
| Exit | Hold to settlement. Pre-tipoff convergence exits trigger on <2% of positions. |
| Fees | Kalshi taker fee: `ceil(0.07 × n × p × (1-p) × 100) / 100`; no exit fees. |

### Liquidity and Capacity

The headline 1,062% half-Kelly return assumes infinite liquidity at the best offer. A sweep-execution simulation walks historical trades in the entry window and fills only contracts available at or below the strategy's max entry price.

| Starting bankroll | Ideal return | Sweep return | Mean fill rate |
|-------------------|--------------|--------------|----------------|
| $100 | 1,062% | 2,547% | 98% |
| $500 | 1,064% | 1,888% | 95% |
| $1,000 | 1,065% | 1,439% | 92% |
| $2,500 | 1,065% | 1,261% | 84% |
| $5,000 | 1,065% | 806% | 79% |
| $7,500 | 1,065% | 662% | 76% |
| $10,000 | 1,065% | 577% | 71% |

Small-bankroll sweep returns are optimistic because historical VWAP below the entry threshold can be cheaper than the snapshot price. At larger bankrolls, the bias likely reverses because the strategy would compete with the same historical liquidity. The $5k path is a capacity case, not a target: mean fill rate drops to 79%, the median trade would consume 50% of qualifying-price tape volume, and 41% of trades would fully consume qualifying-price volume.

Exposure caps reduce drawdowns and improve fills but lower terminal return in this sample: a 6% cap improves mean fill rate from 79% to 89% while reducing return from +806% to +273%; a 15% cap lands at 81% fill and +714% return. Plain half-Kelly at a $5k bankroll prescribes average wagers of $5,492 and a max of $46,741, beyond typical WNBA market depth.

Stopping new entries 3 hours before tipoff improves the $5k liquidity-constrained simulation from +806% to +963%, hit rate from 40.3% to 42.1%, and per-trade Sharpe from +0.121 to +0.141. Trades first qualifying inside the final 8 hours have a 27.3% hit rate and negative mean log-return, consistent with adverse-selection risk from late lineup and news information.

### Market Microstructure

- Kalshi WNBA markets open ~35 hours pre-tipoff on average.
- Volume ramps from <5 contracts/hour near open to 50+ near game time.
- Spreads compress from 20+ cents at open to 1 cent near tipoff.
- Half-life entry (~17h pre-tipoff) balances tighter spreads with remaining model edge.

---

## Hyperparameter Tuning

Three-stage tuning uses walk-forward CV. Full grids and the Stage 3 top-10 table are in [`docs/tuning_methodology.md`](docs/tuning_methodology.md).

| Component | Final parameters |
|-----------|------------------|
| Elo | `H=25`, `K=20`, `alpha=0.45`, `beta=1.0`, `mu=1505` |
| Features | `N_players=7`, `h_M=7`, `L_inj=14`, `tau=150`, `h_team=7` |
| XGBoost | `max_depth=6`, `mcw=3`, `gamma=0.1`, `colsample_bytree=0.6`, `subsample=0.8`, `lambda=1.0`, `alpha=0.0`, `lr=0.02` |

All hyperparameters are also defined in [`config/final_hyperparams.py`](config/final_hyperparams.py). The final XGBoost configuration uses the rank-2 row from the executed Stage 3 main grid (2,592 candidates): the rank-1 row had slightly lower mean log loss but unstable early stopping (`min_best_round=2` in one fold), while the selected row traded +0.00038 mean log loss for consistent convergence (`min_best_round=39`). A later aggressive-grid diagnostic was run in the same notebook but did not replace the main-grid selection.

---

## Data Pipeline and Reproducibility

Analysis outputs are committed under [`outputs/`](outputs/) for inspection. Rebuilding raw data requires Sportradar credentials, and rebuilding market ingestion requires Kalshi credentials.

```bash
conda create -n kalshi-wnba python=3.11
conda activate kalshi-wnba
pip install -r requirements.txt
```

Requires `.env` values for `SPORTRADAR_API_KEY` and Kalshi credentials; see [`.env.example`](.env.example).

| Stage | Directory | Purpose |
|-------|-----------|---------|
| 1. Ingestion | `pipelines/01_ingestion/` | Sportradar schedules, game summaries, daily injuries → bronze JSON |
| 2. Parsing | `pipelines/02_parsing/` | Bronze JSON → silver CSVs for outcomes, box scores, injuries, availability |
| 3. Feature Engineering | `pipelines/03_features/` | Elo, player state history, recent form, style profiles, schedule context |
| 4. Gold Assembly | `pipelines/04_gold/` | Final 160-feature XGBoost table with `base_margin = logit(p_elo)` |
| 5. Modeling | `pipelines/05_modeling/` | Walk-forward XGBoost CV, calibration diagnostics, Elo tuning |
| 6. Market Data | `pipelines/06_markets/` | Kalshi and Polymarket ingestion, matching to Sportradar game IDs |

Sportradar ingestion uses three WNBA endpoints: season schedule, game summary, and daily injuries. Across 2015–2025 this produces ~4,300 bronze files.

---

## Repository Structure

```text
organized/
├── config/final_hyperparams.py      # Single source of truth for final parameters
├── src/srwnba/                      # Core library: API client, Elo, franchise mapping, analysis helpers
├── utils/                           # Market API clients
├── pipelines/                       # Numbered ingestion, parsing, feature, modeling, and market scripts
├── notebooks/
│   ├── analysis/                    # Final analysis notebooks
│   ├── output generation/           # Rebuilds publication-ready tables and figures
│   ├── xgb_tuning/                  # Stage 3 XGBoost tuning
│   └── scratchwork/                 # Excluded experiments and exploratory notebooks
├── data/
│   ├── gold/                        # Final ML-ready inputs
│   ├── kalshi/                      # Kalshi market data and matched markets
│   ├── polymarket/                  # Polymarket market data
│   ├── model_comparison/            # Model comparison outputs
│   ├── trading_results/             # Backtest trade/result tables
│   ├── spec_sheets/                 # Feature/table specifications
│   ├── config/                      # Static franchise map
│   └── xgb_stage3_top10.csv         # Stage 3 top-10 XGB configs
├── outputs/                         # Canonical figures and summary tables
├── docs/tuning_methodology.md       # Full tuning strategy and grids
├── CHANGELOG.md                     # Inclusion/exclusion decisions log
└── requirements.txt
```

---

## Excluded Alternatives

Several approaches were tested and left out of the final pipeline. Details are in [`notebooks/scratchwork/`](notebooks/scratchwork/) and [`notebooks/scratchwork/README.md`](notebooks/scratchwork/README.md).

| Approach | Finding | Notebook |
|----------|---------|----------|
| Polymarket trading | Thin WNBA liquidity, wide spreads (10–20+ cents) | `scratchwork/poly_trading.ipynb` |
| Pre-tipoff convergence exits | Prices rarely move enough pregame (0–2% trigger rate) | `scratchwork/trading_results.ipynb` |
| Bootstrap ensemble | Did not meaningfully improve over the single model | `scratchwork/ensemble_comparison.ipynb` |
| Neural network (MLP) | Did not outperform XGBoost; higher variance across folds | `scratchwork/NN_test.ipynb` |
| XGBoost without Elo | Worse than Elo + XGBoost, supporting base-margin design | `scratchwork/XGBpure.ipynb` |
| Full-Kelly sizing | Too aggressive at 35–44% hit rates | `analysis/trading_results2.ipynb` §7 |
| Two-thirds-life entry | Half-life (~17h) outperformed two-thirds-life (~12h) | `analysis/trading_results2.ipynb` §8 |

---

## Future Work

- **Multi-season validation.** Additional Kalshi WNBA seasons would test whether the trade-selection edge persists and narrow confidence intervals.
- **Cross-sport transfer.** Applying the same Elo + XGBoost architecture to NBA or other leagues would test generalization in deeper markets.
- **In-play model.** Live updates could capture information that the pregame feature snapshot intentionally excludes.
- **Market-price features.** Kalshi/Polymarket implied probabilities may improve calibration if treated as pregame features rather than only evaluation targets.
- **Trade-selection study.** Isolating probability accuracy, trade selection, and Kelly sizing would clarify which mechanism deserves further investment.
