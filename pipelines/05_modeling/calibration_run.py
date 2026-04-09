import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path('../..').resolve()
GOLD_DIR = PROJECT_ROOT / 'data' / 'gold' / 'stage1' / 'hM7_Linj14_tau150_hT7'
OUT_DIR  = PROJECT_ROOT / 'data' / 'calibration'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Winner hyperparams from tuning3 gridsearch (honest nested CV)
XGB_PARAMS = {
    'objective':        'binary:logistic',
    'eval_metric':      'logloss',
    'max_depth':        6,
    'min_child_weight': 2,
    'subsample':        0.9,
    'colsample_bytree': 0.8,
    'reg_lambda':       0.5,
    'reg_alpha':        0.0,
    'gamma':            0.5,
    'learning_rate':    0.03,
    'seed':             42,
    'nthread':         -1,
}

N_PLAYERS      = 7
NUM_BOOST_ROUND = 3000
EARLY_STOPPING  = 150
OOF_YEARS       = list(range(2020, 2025))   # 2020-2024
LABEL_COL       = 'home_win'

print('Gold dir:', GOLD_DIR)
print('Output dir:', OUT_DIR)
print('OOF years:', OOF_YEARS)

# ---

CLIP_EPS = 1e-6
FEATURE_COLS = None  # populated after loading first file
ELO_PROB_COL = 'p_elo'   # column holding pre-game Elo win probability

PLAYER_MODEL_FEATURES = [
    'm_ewma_pre', 'q_pre', 'days_since_first_report_pre', 'days_since_last_dnp_pre',
    'consec_dnps_pre', 'played_last_game_pre', 'minutes_last_game_pre',
    'days_since_last_played_pre', 'injury_present_flag_pre',
]
RECENT_FORM_FEATURES = [
    'net_rtg_ewma_pre', 'efg_ewma_pre', 'tov_pct_ewma_pre', 'orb_pct_ewma_pre', 'ftr_ewma_pre',
]
STYLE_FEATURES = [
    'off_3pa_rate_pre', 'def_3pa_allowed_pre', 'off_2pa_rate_pre', 'def_2pa_allowed_pre',
    'off_tov_pct_pre', 'def_forced_tov_pre',
]
SCHEDULE_FEATURES = [
    'days_rest_pre', 'is_b2b_pre', 'games_last_4_days_pre', 'games_last_7_days_pre',
    'travel_miles_pre', 'timezone_shift_hours_pre',
]


def build_feature_cols(n_players):
    """Exact same logic as 32_run_xgboost_cv.py — 160 cols for n_players=7."""
    cols = []
    for side in ('home', 'away'):
        for slot in range(1, n_players + 1):
            for feat in PLAYER_MODEL_FEATURES:
                cols.append(f'{side}_p{slot}_{feat}')
    for feat in RECENT_FORM_FEATURES:
        cols.append(f'home_{feat}')
        cols.append(f'away_{feat}')
    for feat in STYLE_FEATURES:
        cols.append(f'home_{feat}')
        cols.append(f'away_{feat}')
    for feat in SCHEDULE_FEATURES:
        cols.append(f'home_{feat}')
        cols.append(f'away_{feat}')
    return cols


def clip_probs(p):
    return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def logit(p):
    p = clip_probs(np.asarray(p, dtype=float))
    return np.log(p / (1 - p))


def load_year(year):
    """Load gold-level XGBoost input for a given year."""
    path = GOLD_DIR / f'game_xgboost_input_{year}_REGPST.csv'
    df = pd.read_csv(path)
    return df


def cold_start_mask(df):
    """True for rows to KEEP — drop rows where home p1 m_ewma_pre == 0 (cold start)."""
    col = 'home_p1_m_ewma_pre'
    if col not in df.columns:
        return pd.Series(True, index=df.index)
    return df[col] != 0


def make_dmatrix(df, feature_cols):
    avail = [c for c in feature_cols if c in df.columns]
    X = df[avail].values.astype(float)
    y = df[LABEL_COL].values.astype(float) if LABEL_COL in df.columns else None
    dm = xgb.DMatrix(X, label=y, feature_names=avail, missing=np.nan)
    if ELO_PROB_COL in df.columns:
        p_elo = clip_probs(df[ELO_PROB_COL].values)
        dm.set_base_margin(logit(p_elo))
    return dm


def log_loss_score(y_true, p_pred):
    p = clip_probs(np.asarray(p_pred))
    y = np.asarray(y_true)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def xgb_fit_predict(train_df, val_df, feature_cols, num_rounds, early_stop=None):
    """Fit XGBoost on train_df, predict on val_df. Returns (model, p_val, best_round)."""
    dtrain = make_dmatrix(train_df, feature_cols)
    dval   = make_dmatrix(val_df,   feature_cols)
    params = XGB_PARAMS.copy()
    model  = xgb.train(
        params, dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, 'val')],
        early_stopping_rounds=early_stop,
        verbose_eval=False,
    )
    p_val = model.predict(dval)
    best_round = model.best_iteration + 1 if early_stop else num_rounds
    return model, p_val, best_round


def fit_platt(p_raw, y_true):
    """Fit Platt scaler: logit(p_cal) = a + b*logit(p_raw). Returns (a, b)."""
    lr = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    lr.fit(logit(p_raw).reshape(-1, 1), y_true)
    return float(lr.intercept_[0]), float(lr.coef_[0][0])


def apply_platt(p_raw, a, b):
    return clip_probs(1 / (1 + np.exp(-(a + b * logit(p_raw)))))


print('Helpers defined.')

# ---

# Load all years 2015-2024 into a dict
all_data = {}
for yr in range(2015, 2025):
    df = load_year(yr)
    all_data[yr] = df

# Establish feature columns from first loaded file
sample_df = all_data[2015]
FEATURE_COLS = build_feature_cols(N_PLAYERS)
print(f'Feature cols ({len(FEATURE_COLS)}): {FEATURE_COLS[:5]} ...')

# Describe folds
print('\nFold layout:')
for Y in OOF_YEARS:
    train_yrs = list(range(2015, Y - 1))
    es_yr     = Y - 1
    print(f'  Fold {Y}: train={train_yrs}, ES-val={es_yr}, test={Y}')

# ---

oof_records = []
fold_meta   = {}   # {Y: {'best_round': int, 'es_ll': float, 'oof_ll': float}}

for Y in OOF_YEARS:
    es_yr     = Y - 1
    train_yrs = list(range(2015, es_yr))   # 2015 .. Y-2

    # --- build ES-train set (cold-start filtered) ---
    es_train_parts = [all_data[yr][cold_start_mask(all_data[yr])] for yr in train_yrs]
    es_train_df = pd.concat(es_train_parts, ignore_index=True) if es_train_parts else pd.DataFrame()
    es_val_df   = all_data[es_yr]

    if es_train_df.empty:
        print(f'  WARNING: fold {Y} ES-train is empty, skipping.')
        continue

    # --- Step 1: early-stop run to find best_round ---
    _, p_es_val, best_round = xgb_fit_predict(
        es_train_df, es_val_df, FEATURE_COLS,
        num_rounds=NUM_BOOST_ROUND, early_stop=EARLY_STOPPING
    )
    es_ll = log_loss_score(es_val_df[LABEL_COL].values, p_es_val)

    # --- Step 2: retrain on 2015..Y-1 for exactly best_round trees ---
    full_train_parts = []
    for yr in range(2015, Y):
        df = all_data[yr]
        full_train_parts.append(df[cold_start_mask(df)] if yr == 2015 else df)
    full_train_df = pd.concat(full_train_parts, ignore_index=True)

    _, p_oof, _ = xgb_fit_predict(
        full_train_df, all_data[Y], FEATURE_COLS,
        num_rounds=best_round, early_stop=None
    )
    oof_ll = log_loss_score(all_data[Y][LABEL_COL].values, p_oof)

    fold_meta[Y] = {'best_round': best_round, 'es_ll': es_ll, 'oof_ll': oof_ll}
    print(f'Fold {Y}: best_round={best_round:4d} | ES-val LL={es_ll:.5f} | OOF LL={oof_ll:.5f}')

    # --- collect OOF records ---
    test_df = all_data[Y][['game_id', 'game_date', 'home_team_id', 'away_team_id',
                            'season', LABEL_COL, ELO_PROB_COL]].copy()
    test_df['p_raw']     = p_oof
    test_df['fold_year'] = Y
    oof_records.append(test_df)

oof_df = pd.concat(oof_records, ignore_index=True)
print(f'\nTotal OOF rows: {len(oof_df)}')

# ---

# Save OOF raw predictions
out_path = OUT_DIR / 'oof_raw_preds_2020_2024.csv'
oof_df.to_csv(out_path, index=False)
print(f'Saved: {out_path}')
oof_df.head()

# ---

y_oof = oof_df[LABEL_COL].values.astype(float)
p_raw = oof_df['p_raw'].values

# Fit Platt on pooled OOF
a, b = fit_platt(p_raw, y_oof)
print(f'Platt params: a (intercept) = {a:.6f},  b (slope) = {b:.6f}')
print(f'  logit(p_cal) = {a:.4f} + {b:.4f} * logit(p_raw)')

p_cal = apply_platt(p_raw, a, b)
oof_df['p_cal'] = p_cal

# Save calibrator params
cal_params = pd.DataFrame({'param': ['intercept_a', 'slope_b'], 'value': [a, b]})
cal_params.to_csv(OUT_DIR / 'platt_calibrator_2020_2024.csv', index=False)
print(f"Saved: {OUT_DIR / 'platt_calibrator_2020_2024.csv'}")

oof_df.to_csv(OUT_DIR / 'oof_calibrated_preds_2020_2024.csv', index=False)
print(f"Saved: {OUT_DIR / 'oof_calibrated_preds_2020_2024.csv'}")

# ---

def accuracy(y, p, threshold=0.5):
    return np.mean((p >= threshold) == y.astype(bool))

# Overall summary
rows = []
for model_name, p_col in [('elo', ELO_PROB_COL), ('xgb_raw', 'p_raw'), ('xgb_cal', 'p_cal')]:
    p = oof_df[p_col].values
    y = oof_df[LABEL_COL].values
    rows.append({
        'model':    model_name,
        'log_loss': round(log_loss_score(y, p), 5),
        'accuracy': round(accuracy(y, p), 4),
        'n_games':  len(y),
    })
summary_df = pd.DataFrame(rows)
print('=== Overall (2020-2024 pooled) ===')
print(summary_df.to_string(index=False))

# Per-fold breakdown
fold_rows = []
for Y in OOF_YEARS:
    sub = oof_df[oof_df['fold_year'] == Y]
    y   = sub[LABEL_COL].values
    for model_name, p_col in [('elo', ELO_PROB_COL), ('xgb_raw', 'p_raw'), ('xgb_cal', 'p_cal')]:
        p = sub[p_col].values
        fold_rows.append({
            'fold_year':  Y,
            'model':      model_name,
            'log_loss':   round(log_loss_score(y, p), 5),
            'accuracy':   round(accuracy(y, p), 4),
            'n_games':    len(y),
            'best_round': fold_meta.get(Y, {}).get('best_round') if model_name == 'xgb_raw' else None,
        })
fold_df = pd.DataFrame(fold_rows)
print('\n=== Per-fold ===')
print(fold_df.to_string(index=False))

summary_df.to_csv(OUT_DIR / 'calibration_eval_summary_2020_2024.csv', index=False)
fold_df.to_csv(OUT_DIR / 'calibration_eval_by_fold_2020_2024.csv', index=False)
print('\nSaved eval CSVs.')

# ---

def reliability_data(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    mean_pred, mean_true, counts = [], [], []
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() == 0:
            continue
        mean_pred.append(p[mask].mean())
        mean_true.append(y[mask].mean())
        counts.append(mask.sum())
    return np.array(mean_pred), np.array(mean_true), np.array(counts)


y   = oof_df[LABEL_COL].values.astype(float)
fig, ax = plt.subplots(figsize=(7, 6))

for label, p_col, color in [('Elo', ELO_PROB_COL, 'steelblue'),
                              ('XGB raw', 'p_raw', 'tomato'),
                              ('XGB cal', 'p_cal', 'seagreen')]:
    mp, mt, _ = reliability_data(y, oof_df[p_col].values)
    ax.plot(mp, mt, marker='o', label=label, color=color, linewidth=1.8)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Empirical win rate')
ax.set_title('Reliability diagram — OOF 2020-2024')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

fig_path = OUT_DIR / 'reliability_diagram_2020_2024.png'
fig.savefig(fig_path, dpi=150)
print(f'Saved: {fig_path}')
plt.show()