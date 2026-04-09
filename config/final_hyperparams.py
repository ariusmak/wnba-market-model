"""
Final locked hyperparameters for the WNBA prediction model.
This file is the single source of truth. All notebooks and scripts should reference these values.

Tuning history:
  - Stage 0: Elo tuned via iterative walk-forward grid search (5 rounds)
  - Stage 1: Feature params (N_players, h_M, L_inj) tuned jointly with conservative XGB grid (3,888 configs)
  - Stage 2: Remaining feature params (tau, h_team) tuned with Stage 1 winners locked (12 configs)
  - Stage 3: XGBoost params tuned via walk-forward CV (1,296 configs in final grid)
  - Final XGB config chosen as rank 2 for stability (rank 1 had min_best_round=2, indicating unstable convergence)
"""

# ── Elo (Stage 0) ──────────────────────────────────────────────────
ELO_H     = 25      # home-court advantage (Elo points)
ELO_K     = 20      # update rate
ELO_ALPHA = 0.45    # season carryover fraction (R_new = alpha*R_end + (1-alpha)*mu)
ELO_BETA  = 1.0     # margin-of-victory exponent
ELO_MU    = 1505    # prior rating

# ── Feature engineering (Stages 1 & 2) ─────────────────────────────
N_PLAYERS = 7       # player slots per team
H_M       = 7       # EWMA half-life for player minutes (games)
L_INJ     = 14      # injury inclusion window (days)
TAU       = 150     # player quality prior strength (minutes)
H_TEAM    = 7       # team recent-form EWMA half-life (games)

# ── XGBoost (Stage 3, rank 2 — stability-chosen) ───────────────────
XGB_PARAMS = dict(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    max_depth        = 6,
    min_child_weight = 3,
    gamma = 0.1,
    colsample_bytree = 0.6,
    subsample        = 0.8,
    reg_lambda       = 1.0,
    reg_alpha        = 0.0,
    learning_rate    = 0.02,
    seed             = 42,
    nthread          = -1,
)
NUM_BOOST_ROUND       = 3000
EARLY_STOPPING_ROUNDS = 150
