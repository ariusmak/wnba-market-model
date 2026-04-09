# Hyperparameter Tuning Methodology

This document describes the multi-stage tuning strategy used to select final model hyperparameters.

All tuning was conducted via **walk-forward cross-validation** on the 2015–2024 development set, with test folds on 2020–2024 (5 folds). The 2025 season was held out entirely and never used during tuning.

---

## Stage 0: Elo Tuning

Elo hyperparameters were tuned via iterative grid refinement. Each round expanded the search in directions where the best value was at a grid edge, and narrowed the search where values had stabilized.

**Round 1 — Tune H (home advantage) in isolation:**

| Parameter | Values |
|-----------|--------|
| H | 35, 40, 43, 45, 50, 55 |
| K | 20 (fixed) |
| alpha | 0.7 (fixed) |
| beta | 0.8 (fixed) |

Initial H was anchored at 43 (derived from empirical home win rate of 0.562).

**Round 2 — Full simultaneous grid:**

| Parameter | Values |
|-----------|--------|
| H | 35, 40, 43, 45, 50, 55 |
| K | 10, 15, 20, 25, 30 |
| alpha (season carryover) | 0.50, 0.60, 0.70, 0.75, 0.85 |
| beta (MOV exponent) | 0 (no MOV), 0.6, 0.8, 1.0 |

**Round 3 — Expand edges (H, alpha, beta were at boundaries):**

| Parameter | Values |
|-----------|--------|
| H | 45, 50, 55, 60, 65 |
| K | 15, 20, 25 |
| alpha | 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60 |
| beta | 0.9, 1.0, 1.1, 1.2 |

**Round 4 — Narrow K (now at edge), keep others local:**

| Parameter | Values |
|-----------|--------|
| H | 50, 55, 60 |
| K | 5, 7.5, 10, 12.5, 15, 17.5, 20 |
| alpha | 0.40, 0.45, 0.50 |
| beta | 1.0, 1.1, 1.2 |

**Round 5 — Final refinement of beta:**

| Parameter | Values |
|-----------|--------|
| H | 50, 55, 60 |
| K | 7.5, 10, 12.5 |
| alpha | 0.40, 0.45, 0.50 |
| beta | 1.15, 1.2, 1.25, 1.3, 1.35, 1.4 |

**Metric:** Walk-forward log loss on 2020–2024 folds.

**Final locked Elo parameters:**
- H = 25, K = 20, alpha = 0.45, beta = 1.0, mu = 1505

**Rationale:** Selected from a flat region of the loss surface, prioritizing stability over marginal log-loss gains. The final values were chosen to be conservative and generalizable rather than chasing the minimum of a noisy objective.

See: `notebooks/elo_tuning/elo_tuning.ipynb`

---

## Stage 1: Joint Feature + Model Tuning

Feature-engineering hyperparameters were searched jointly with a conservative XGBoost grid. For each combination of feature parameters, a separate gold dataset was built (12 feature variants x 3 N_players values = 36 feature configs), and each was evaluated across a compact XGBoost grid.

**Feature search space:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| N_players | 5, 7, 10 | Player slots per team |
| h_M | 3, 5, 7, 10 | EWMA half-life for player minutes (games) |
| L_inj | 7, 14, 21 | Injury inclusion window (days) |

**Companion XGBoost grid (held conservative to isolate feature effects):**

| Parameter | Values |
|-----------|--------|
| max_depth | 3, 4, 5, 6 |
| min_child_weight | 1, 3, 6 |
| colsample_bytree | 0.6, 0.8, 1.0 |
| reg_lambda | 1.0, 2.0, 4.0 |

Fixed: subsample=0.9, learning_rate=0.02, NUM_BOOST_ROUND=3000, early_stopping_rounds=30.

**Total configurations evaluated:** 3,888 (36 feature combos x 108 XGB combos).

**Result:** N_players=7, h_M=7, L_inj=14

**Rationale:** 7 player slots captured the top contributors without overfitting to bench noise. h_M=7 and L_inj=14 were chosen from the most stable winning region across the XGB grid — i.e., feature configs that performed well regardless of which XGB settings were paired with them.

---

## Stage 2: Feature Tuning

Fine-tuned remaining feature-engineering parameters with Stage 1 winners locked (N_players=7, h_M=7, L_inj=14). A single XGBoost config (the best from Stage 1: d=3, mcw=3, cbt=0.6, rl=2) was used to evaluate each feature variant.

| Parameter | Values | Selected |
|-----------|--------|----------|
| tau (player quality prior weight) | 100, 150, 200, 250 | 150 |
| h_team (team EWMA half-life) | 5, 7, 10 | 7 |

**Total configurations evaluated:** 12 (4 tau x 3 h_team).

---

## Stage 3: XGBoost Hyperparameter Tuning

With all feature parameters locked, XGBoost hyperparameters were tuned. Preliminary grid searches explored a wide parameter space and identified a promising region (depth 4–8, min_child_weight 2–3, colsample_bytree ~0.6, moderate regularization). These preliminary results informed the design of the final grid search, which was conducted with corrected walk-forward CV methodology.

**Final grid (1,296 configs):**

| Parameter | Values |
|-----------|--------|
| max_depth | 4, 5, 6, 7, 8 |
| min_child_weight | 2, 3 |
| gamma | 0.0, 0.1, 0.5, 1.0 |
| colsample_bytree | 0.6 |
| reg_lambda | 0.5, 1.0, 2.0 |
| learning_rate | 0.01, 0.02, 0.03 |

All configs used subsample=0.8, reg_alpha=0.0, NUM_BOOST_ROUND=3000, early_stopping_rounds=150.

### Top 10 Configurations (Final Grid)

| Rank | depth | mcw | gamma | cbt | rl  | lr   | mean_ll  | min_best_round |
|------|-------|-----|-------|-----|-----|------|----------|----------------|
| 1    | 6     | 3   | 1.0   | 0.6 | 1.0 | 0.03 | 0.59749  | 2              |
| 2    | 6     | 3   | 0.1   | 0.6 | 1.0 | 0.02 | 0.59787  | 39             |
| 3    | 6     | 3   | 0.1   | 0.6 | 0.5 | 0.01 | 0.59813  | 39             |
| 4    | 8     | 3   | 0.1   | 0.6 | 1.0 | 0.02 | 0.59815  | 29             |
| 5    | 7     | 3   | 0.5   | 0.6 | 1.0 | 0.02 | 0.59820  | 1              |
| 6    | 6     | 3   | 1.0   | 0.6 | 1.0 | 0.02 | 0.59823  | 29             |
| 7    | 8     | 3   | 0.0   | 0.6 | 2.0 | 0.02 | 0.59839  | 21             |
| 8    | 6     | 3   | 0.0   | 0.6 | 1.0 | 0.02 | 0.59841  | 39             |
| 9    | 6     | 2   | 0.1   | 0.6 | 0.5 | 0.03 | 0.59844  | 4              |
| 10   | 6     | 2   | 1.0   | 0.6 | 1.0 | 0.03 | 0.59860  | 2              |

### Final Chosen Configuration (Rank 2)

```
max_depth        = 6
min_child_weight = 3
gamma            = 0.1
colsample_bytree = 0.6
subsample        = 0.8
reg_lambda       = 1.0
reg_alpha        = 0.0
learning_rate    = 0.02
```

**Why rank 2 over rank 1?**

The rank-1 configuration (lr=0.03) had `min_best_round=2` across the 5 CV folds, meaning early stopping triggered after just 2 boosting rounds in at least one fold. This indicates **unstable convergence** — the model is learning rate-dependent and may not generalize reliably. The rank-2 configuration (lr=0.02, gamma=0.1) had `min_best_round=39`, indicating consistent training across all folds.

The log-loss difference between rank 1 and rank 2 is only 0.00038 (0.59749 vs 0.59787), well within noise. Stability was prioritized over marginal mean performance.

See: `notebooks/xgb_tuning/XGB_tuning3.ipynb`, `notebooks/xgb_tuning/complexity_curve.ipynb`

---

## Summary of Final Locked Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Elo | H | 25 |
| Elo | K | 20 |
| Elo | alpha | 0.45 |
| Elo | beta | 1.0 |
| Elo | mu | 1505 |
| Features | N_players | 7 |
| Features | h_M | 7 |
| Features | L_inj | 14 |
| Features | tau | 150 |
| Features | h_team | 7 |
| XGBoost | max_depth | 6 |
| XGBoost | min_child_weight | 3 |
| XGBoost | gamma | 1.0 |
| XGBoost | colsample_bytree | 0.6 |
| XGBoost | subsample | 0.8 |
| XGBoost | reg_lambda | 1.0 |
| XGBoost | reg_alpha | 0.0 |
| XGBoost | learning_rate | 0.02 |
