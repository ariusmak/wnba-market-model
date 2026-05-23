# WNBA Prediction Market Model

A sports outcome forecasting and trading model for WNBA prediction markets on Kalshi, implementing Elo structural priors, gradient-boosted trees with as-of feature construction, walk-forward evaluation, and practical market application.

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

The XGBoost + Elo model improves over the Elo baseline in both development (−0.0028 log loss) and on the untouched 2025 holdout (−0.0030), with a consistent accuracy advantage. XGBoost without Elo is substantially worse, supporting the Elo-as-base-margin architecture. Logistic regression has the weakest performance of the tested models, suggesting that the useful corrections are non-linear rather than a simple linear adjustment to Elo.

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
| P(Full Model > Elo) | 0.652 |
| Growth-rate difference 95% CI | [−0.033, +0.048] |

Under a one-sided null-centered bootstrap (H₀: mean log-return = 0), the full-model strategy clears the null at **p = 0.111**, Elo at **p = 0.222**, and the full-model − Elo difference at **p = 0.355**. With ~130–155 trades in a single season, the difference is directionally consistent but not statistically significant at conventional levels; additional seasons would be needed to tighten the interval enough for a formal model-vs-Elo claim.

Full trading analysis: [`notebooks/analysis/trading_results2.ipynb`](notebooks/analysis/trading_results2.ipynb).

### Model vs Market Comparison

On the 2025 holdout, the model's predictions are compared head-to-head against Kalshi and Polymarket pre-tipoff implied probabilities (one row per game, after deduplicating Polymarket's repeat `condition_id` listings):

| Source | n | Log Loss | Brier | Accuracy |
|--------|---|----------|-------|----------|
| XGB + Elo (model) | 310 | 0.612 | 0.211 | 67.4% |
| Elo only | 310 | 0.615 | 0.213 | 66.8% |
| Kalshi pre-tipoff | 296 | 0.612 | 0.213 | 62.8% |
| Polymarket pre-tipoff | 221 | 0.626 | 0.219 | 64.3% |

On the common subset (210 games with all four sources), Kalshi has the lower log loss (0.619 vs 0.625), the model has much higher accuracy (67.6% vs 62.9%), and Brier scores are effectively tied (0.2163 vs 0.2162). This points to different strengths rather than a clean market-beating calibration result.

When the model and Kalshi disagree on the game direction (51 games), the model is correct **65%** of the time. This is a high-signal slice for trade selection, but it is still a small 2025-only sample.

### Result Audit Trail

The main claims in this README are backed by regenerated tables and figures in [`outputs/`](outputs/), with source notebooks in [`notebooks/analysis/`](notebooks/analysis/) and [`notebooks/output generation/`](notebooks/output%20generation/):

| Claim area | Primary table(s) | Primary notebook(s) |
|------------|------------------|---------------------|
| Forecasting benchmark | [`forecast_model_performance_summary.csv`](outputs/forecast_model_performance_summary.csv), [`forecast_per_fold_performance_table.csv`](outputs/forecast_per_fold_performance_table.csv) | [`forecasting_results.ipynb`](notebooks/analysis/forecasting_results.ipynb), [`forecasting.ipynb`](notebooks/output%20generation/forecasting.ipynb) |
| Feature ablations / robustness | [`feature_block_ablation_summary.csv`](outputs/feature_block_ablation_summary.csv), [`training_windows_logloss_summary.csv`](outputs/training_windows_logloss_summary.csv) | [`ablations.ipynb`](notebooks/analysis/ablations.ipynb), [`training_windows.ipynb`](notebooks/analysis/training_windows.ipynb) |
| Market comparison | [`market_model_performance_summary.csv`](outputs/market_model_performance_summary.csv), [`market_directional_disagreement_table.csv`](outputs/market_directional_disagreement_table.csv) | [`market_comparison.ipynb`](notebooks/output%20generation/market_comparison.ipynb) |
| Trading grid / return decomposition | [`trade_half_kelly_best_config_table.csv`](outputs/trade_half_kelly_best_config_table.csv), [`trade_return_decomposition_table.csv`](outputs/trade_return_decomposition_table.csv) | [`trading_results2.ipynb`](notebooks/analysis/trading_results2.ipynb), [`return_decomposition.ipynb`](notebooks/output%20generation/return_decomposition.ipynb) |
| Liquidity and execution sensitivity | [`liq_execution_summary_table.csv`](outputs/liq_execution_summary_table.csv), [`trade_volume_share_summary_table.csv`](outputs/trade_volume_share_summary_table.csv), [`trade_cutoff_sweep_summary.csv`](outputs/trade_cutoff_sweep_summary.csv) | [`return_investigation.ipynb`](notebooks/analysis/return_investigation.ipynb), [`liquidity.ipynb`](notebooks/output%20generation/liquidity.ipynb) |
| Statistical significance | [`sig_bootstrap_summary_table.csv`](outputs/sig_bootstrap_summary_table.csv) | [`significance.ipynb`](notebooks/output%20generation/significance.ipynb) |

---

## Analysis & Interpretation

### Elo captures most of the signal

The most striking result in the forecasting table is that XGBoost *without* Elo — using only player, form, style, and schedule features — achieves a dev log loss of 0.623, within 0.021 of the Elo-only baseline (0.602). These two models use completely different data sources and methodologies: Elo sees only game outcomes and margin of victory, while the features-only XGBoost sees player availability, box-score tendencies, rest patterns, and team style. The fact that they converge to similar performance suggests that Elo already encodes much of what matters, team strength is the dominant signal, and contextual features provide only a marginal correction.

This is further supported by feature importance. When XGBoost has no Elo base margin, it learns sensible structure: net rating EWMA and top-player quality (`p1_q`, `p2_q`) dominate importance, essentially reconstructing a team-strength signal from available data. When XGBoost *does* have Elo as a base margin, the remaining feature importance is scattered across low-level player slots (e.g., `home_p2_played_last_game`, `away_p5_days_since_last_played`) with no single dominant correction signal. The logistic regression tells the same story: `base_margin` has a coefficient of 0.92 (nearly 1.0, meaning Elo is passed through almost unchanged), and the largest feature coefficients are schedule and player availability variables with modest magnitude.

### Small log loss improvements, large trading returns

The most counterintuitive result is the gap between forecasting and trading performance. On the 2025 holdout, the full model's log loss improvement over Elo is modest (0.6121 vs 0.6151 — just 0.003 points), yet it produces **1,062% half-Kelly return** vs Elo's **417%** — a 2.5x difference in terminal wealth from a nearly negligible calibration improvement.

**The entire return gap is driven by differential game selection, not by better sizing or higher accuracy on the same games.** A direct head-to-head analysis ([`trading_results2.ipynb`](notebooks/analysis/trading_results2.ipynb) §14) under identical entry rules (edge ≥ 0.05, norm_edge ≥ 0.25, half-life entry) gives:

| Trade partition | Games | FM hit rate | FM P&L | Elo hit rate | Elo P&L |
|----|----|----|----|----|----|
| Traded by both models (always same side) | 111 | 35.1% | −$1 | 35.1% | +$210 |
| Only the full model traded | 23 | **65.2%** | **+$1,063** | — | — |
| Only Elo traded | 44 | 31.8% | — | 31.8% | +$207 |
| **Total** | — | **40.3%** | **+$1,062** | **34.2%** | **+$417** |

The interpretation is sharp:

1. **On shared games, the full model is actually slightly worse.** Both models pick the same side on all 111 shared games, with nearly identical Kelly fractions (mean 0.211 vs 0.216) and identical hit rates (35.1%). The full model's mean edge on these games is *lower* in normalized terms (0.494 vs 0.555). Compounding noise leaves it −$1 while Elo books +$210 on the same positions. So the FM does not win by "betting bigger on winners" or "tail accuracy in high-edge games."

2. **The full model's edge is knowing which additional games to trade.** Elo uniquely triggers on 44 games that hit only 31.8% — it overtrades games where its flat team-strength prior sees edge that isn't there. The full model's player-availability, recent-form, and style features *suppress* these false-edge trades while *surfacing* 23 new games that Elo misses. Those 23 games hit **65.2%** and produce +$1,063 — essentially all of the full model's profit.

3. **Compounding amplifies the selection advantage.** Half-Kelly sizes proportionally to current bankroll. Because the 23 FM-exclusive winners come concentrated in mid-to-late season (when the bankroll is already inflated from earlier trades), their dollar contribution is much larger than a fixed-$1 simulation would show. The same 23 games under fixed $1 sizing would produce only +$15 of profit.

**In short:** the full model's feature set does not improve probability accuracy on games both models want to trade. It improves *trade selection* — suppressing overconfident Elo bets on contextually unfavorable matchups and surfacing high-conviction games Elo's team-strength-only view cannot distinguish. The thin 0.003-log-loss gap reflects the fact that this selection advantage is localized to ~20% of the season; average calibration across all 310 games barely moves.

### Realistic execution: liquidity and bankroll sensitivity

The 1,062% figure above assumes infinite liquidity at the best offer. Historical Kalshi trade data tells a different story at realistic capital levels ([`return_investigation.ipynb`](notebooks/analysis/return_investigation.ipynb) §4, §6).

**Sweep-execution simulation** — for each trade, walk actual historical trades in the entry window chronologically, take any contracts at or below our max entry price up to our required size, and leave unfilled quantity unexecuted:

| Starting bankroll | Ideal return | Sweep return | Mean fill rate |
|---|---|---|---|
| $100 | 1,062% | 2,547% | 98% |
| $500 | 1,064% | 1,888% | 95% |
| $1,000 | 1,065% | 1,439% | 92% |
| $2,500 | 1,065% | 1,261% | 84% |
| $5,000 | 1,065% | 806% | 79% |
| $7,500 | 1,065% | 662% | 76% |
| $10,000 | 1,065% | 577% | 71% |

Two important caveats on the sweep-return column:

- At **$100–$1,000**, sweep *exceeds* ideal. This is an upward bias: historical trades represent all market activity, and the VWAP below our threshold is sometimes cheaper than the entry snapshot price. A real trader placing limit orders would not systematically get those improved fills. Treat small-bankroll sweep returns as an optimistic envelope.
- At **$5,000+**, the opposite bias dominates: we are competing with the same historical participants for that liquidity, not observing resting orders. Realistic execution likely lies *below* the sweep return.

The meaningful signal across the table is the trajectory: **Kelly % returns are flat under infinite liquidity but degrade monotonically above $1k** once order size exceeds typical in-window volume. At the $5k bankroll used as a realistic case, the sweep path delivers **+806% ($40,276 P&L)** — a 24% haircut from the ideal path.

**Capacity breakpoint.** Trade-by-trade fill rates drop from 98% (Q1) to 67% (Q4) at $5k because Kelly wagers grow with the bankroll and late-season contract sizes routinely exceed pre-tipoff window liquidity (mean 14,161 contracts needed vs median 8,364 available). Above ~$2.5k, the strategy is liquidity-constrained rather than edge-constrained. Per-trade volume-share analysis supports this: the median trade would have been **50% of qualifying-price tape volume in its entry window**, and **41% of trades would have fully consumed the qualifying-price book**. The $5k path is a capacity ceiling, not a target — scaling beyond requires multi-venue execution, in-game entries (currently disallowed), or deliberate under-sizing below Kelly.

**Exposure-cap stress test.** Static exposure caps reduce drawdowns and improve fill rates, but they also materially reduce terminal return in this sample: a 6% cap improves mean fill rate from 79% to 89% while reducing return from +806% to +273%; a 15% cap lands at 81% fill and +714% return. These caps are sensitivity checks rather than a recommended live policy. Plain half-Kelly on a $5k bankroll prescribes wagers averaging $5,492 and max $46,741 — well outside what the order book can absorb. See [`return_investigation.ipynb`](notebooks/analysis/return_investigation.ipynb) §4 for the full analysis.

**Late-window execution discipline.** Stopping new entries 3 hours before tipoff (T-3h) improves total return from **+806% → +963%**, hit rate from 40.3% → 42.1%, and per-trade Sharpe from **+0.121 → +0.141** on the same liquidity-constrained $5k simulation. The 13 dropped trades had measurably worse realized outcomes per dollar staked, consistent with adverse-selection risk from late-breaking information (lineup announcements, scratches, beat-reporter rumors) that the pregame feature snapshot cannot see. The cutoff sweep is non-monotonic, but T-3h and T-8h both improve on the no-cutoff baseline before the T-12h sample becomes too small. The mechanism is supported by a first-qualification-time bucket analysis showing trades that first qualified inside the final 8 hours have a 27.3% hit rate and negative mean log-return, versus 40.7% / +0.019 for early-qualifying trades.

### The honest uncertainty

Despite the compelling return numbers, the bootstrap significance test gives P(Full Model > Elo) = 0.652 — suggestive but far from conclusive. A single season of ~130–155 trades is simply insufficient to statistically distinguish two models that both have positive edge. This is a structural limitation of WNBA market size, not a modeling failure.

### Scope and limitations

- **Single-season market test.** Forecasting is evaluated with 2020–2024 walk-forward OOF and a 2025 holdout, but trading evidence comes from one Kalshi season. The return profile is promising, not a long-run proof.
- **Execution simulation is conservative in some places and optimistic in others.** Historical trade sweeps approximate available liquidity but are not a full order-book replay, and they do not model market impact from the strategy's own orders.
- **Pregame-only information set.** The model intentionally excludes in-game updates and late unstructured news. The late-window cutoff results suggest that lineup/news timing matters.
- **Market prices are evaluation targets, not model inputs.** Kalshi and Polymarket prices are used for comparison and trading entry, but not as forecasting features in the final model.

### Future research directions

- **Multi-season validation.** The most direct path to significance is additional Kalshi WNBA seasons under the same pipeline, which would tighten the confidence interval and test whether the trade-selection edge persists.
- **Cross-sport transfer.** Testing the same Elo + XGBoost architecture on NBA or other leagues with deeper markets could validate whether the approach generalizes.
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

### Robustness Checks

Feature-block ablations keep the Elo base margin and final XGBoost configuration fixed, then remove one contextual block at a time. On the 2025 holdout, every block removal worsens log loss relative to the full model:

| Model variant | 2025 Log Loss | Δ vs full | Interpretation |
|---------------|---------------|-----------|----------------|
| Full model | **0.6121** | — | Elo + all contextual feature blocks |
| No player block | 0.6144 | +0.0023 | Player availability contributes, but modestly |
| No style block | 0.6171 | +0.0050 | Style features add a small correction |
| No recent-form block | 0.6251 | +0.0130 | Recent form is a larger holdout contributor |
| No rest/travel block | 0.6259 | +0.0137 | Schedule context is a larger holdout contributor |

The same direction holds in 2020–2024 OOF results, though the deltas are smaller. This supports the feature architecture without implying that every individual feature is stable or causal.

Training-window sensitivity also favors the final expanding-history setup:

| Training window | OOF Log Loss | 2025 Log Loss |
|-----------------|--------------|---------------|
| Expanding from 2015 | **0.5994** | **0.6121** |
| Expanding from 2018 | 0.6041 | 0.6145 |
| Rolling 2-year | 0.6017 | 0.6168 |
| Rolling 3-year | 0.6040 | 0.6171 |

This suggests that older WNBA seasons still add useful signal despite league drift. Platt scaling was evaluated as a calibration diagnostic, but it is not used in the final forecast: it slightly improved pooled OOF calibration, while worsening the 2025 holdout log loss from 0.61215 to 0.61298. Final trading results therefore use raw XGBoost + Elo probabilities.

## Hyperparameter Tuning

Three-stage tuning strategy with walk-forward CV. See [`docs/tuning_methodology.md`](docs/tuning_methodology.md) for full details including search grids and the Stage 3 top-10 configuration table.

**Final locked parameters:**

| Component | Parameters |
|-----------|-----------|
| Elo | H=25, K=20, α=0.45, β=1.0, μ=1505 |
| Features | N_players=7, h_M=7, L_inj=14, τ=150, h_team=7 |
| XGBoost | max_depth=6, mcw=3, γ=0.1, cbt=0.6, sub=0.8, λ=1.0, α=0.0, lr=0.02 |

All hyperparameters are also defined in [`config/final_hyperparams.py`](config/final_hyperparams.py).

The XGBoost configuration was chosen as rank 2 out of 2,592 candidates in the executed Stage 3 main grid. The rank-1 config (lr=0.03) was rejected due to unstable early stopping (min_best_round=2 in one fold), while rank 2 (lr=0.02) showed consistent convergence across all folds (min_best_round=39) with only 0.00038 higher mean log loss. A later aggressive-grid diagnostic was run in the same notebook but did not replace the main-grid selection.

---

## Exploration Summary

Several alternative approaches were investigated and excluded from the final pipeline. These are documented in [`notebooks/scratchwork/`](notebooks/scratchwork/) for completeness — see [`notebooks/scratchwork/README.md`](notebooks/scratchwork/README.md) for details.

| Approach | Finding | Notebook |
|----------|---------|----------|
| **Polymarket trading** | Thin WNBA liquidity, wide spreads (10–20+ cents) | `scratchwork/poly_trading.ipynb` |
| **Pre-tipoff convergence exits** | Prices rarely move enough pre-game (0–2% trigger rate) | `scratchwork/trading_results.ipynb` |
| **Bootstrap ensemble** | Did not meaningfully improve over the single model | `scratchwork/ensemble_comparison.ipynb` |
| **Neural network (MLP)** | Did not outperform XGBoost; higher variance across folds | `scratchwork/NN_test.ipynb` |
| **XGBoost without Elo** | Worse than Elo + XGBoost, supporting base-margin design | `scratchwork/XGBpure.ipynb` |
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
│   ├── 05_modeling/                # XGBoost CV, calibration diagnostics, Elo tuning
│   └── 06_markets/                 # Kalshi & Polymarket data ingestion
├── notebooks/
│   ├── analysis/                   # Final result notebooks
│   │   ├── ablations.ipynb             # Feature-block ablation checks
│   │   ├── forecasting_results.ipynb   # Model comparison & holdout evaluation
│   │   ├── platt_check.ipynb           # Calibration diagnostic
│   │   ├── return_investigation.ipynb  # Liquidity and execution sensitivity
│   │   ├── trading_results2.ipynb      # Kalshi trading backtest & significance testing
│   │   ├── training_windows.ipynb      # Expanding vs rolling training-window sensitivity
│   │   └── prelim.ipynb                # Preliminary data exploration
│   ├── output generation/          # Rebuilds publication-ready output tables/figures
│   │   ├── forecasting.ipynb
│   │   ├── feature_importance.ipynb
│   │   ├── market_comparison.ipynb
│   │   ├── trading_strategy.ipynb
│   │   ├── return_decomposition.ipynb
│   │   ├── liquidity.ipynb
│   │   └── significance.ipynb
│   ├── xgb_tuning/                 # XGBoost tuning (Stage 3)
│   │   ├── XGB_tuning3.ipynb           # Final Stage 3 grid search
│   │   └── complexity_curve.ipynb
│   └── scratchwork/                # Exploration notebooks (see scratchwork/README.md)
├── data/
│   ├── gold/                       # Final ML-ready XGBoost inputs
│   ├── kalshi/                     # Kalshi market data and matched game markets
│   ├── polymarket/                 # Polymarket market data
│   ├── model_comparison/           # Model comparison outputs used by notebooks
│   ├── trading_results/            # Backtest trade/result tables
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
├── outputs/                        # Canonical figures and summary tables
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
Walk-forward XGBoost CV, calibration diagnostics, Elo grid search.

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

Requires a `.env` file with `SPORTRADAR_API_KEY` for Sportradar ingestion and Kalshi credentials for live market ingestion (see `.env.example`).

---

## Key Design Decisions

1. **Elo as base_margin, not a feature.** Elo provides the structural prior; XGBoost learns corrections on top of it. This is more principled than including Elo as just another feature — supported by the XGBoost-without-Elo benchmark performing substantially worse.

2. **Walk-forward CV, not k-fold.** Sports data is temporal. Using future data to predict past games would be leakage.

3. **Engineered and tuned features.** Features are carefully crafted and feature engineering hyperparameters are tuned where appropriate.

4. **Pre-tipoff only trading.** All entry decisions happen before tipoff. In-game price movements reflect live information, not pregame model edge.

5. **Franchise continuity.** The San Antonio Stars → Las Vegas Aces (2018) relocation is treated as franchise continuity, preserving Elo and player priors across the move.

6. **Cold-start 2015 rows excluded.** Rows are dropped when either team's top-player EWMA minutes are zero. This affects the earliest 2015 games, where no 2014 player-prior history exists and player features are not yet informative.
