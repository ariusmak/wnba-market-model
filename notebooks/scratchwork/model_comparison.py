"""
model_comparison.py

Final results notebook: compares 5 forecasting models on development OOF (2020-2024)
and the untouched 2025 holdout.

Models:
  1. elo                – p = p_elo (no fitting)
  2. logreg_no_elo      – logistic regression on X only
  3. xgb_no_elo         – XGBoost on X only (no base_margin)
  4. logreg_with_elo    – logistic regression on X with logit(p_elo) as offset
  5. xgb_with_elo       – XGBoost on X with base_margin = logit(p_elo)

All hyperparameters and features are frozen.  This is a reporting
notebook, not a tuning notebook.

Usage:
  cd <project_root>
  python notebooks/model_comparison.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================================
# Section 1 — Setup
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR     = PROJECT_ROOT / "data" / "gold"
OUT_DIR      = PROJECT_ROOT / "data" / "model_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Frozen XGBoost hyperparameters (from tuning3 + follow-up grid) ---------
XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "logloss",
    "max_depth":        6,
    "min_child_weight": 2,
    "subsample":        0.9,
    "colsample_bytree": 0.8,
    "reg_lambda":       0.5,
    "reg_alpha":        0.0,
    "gamma":            0.5,
    "learning_rate":    0.03,
    "seed":             42,
    "nthread":         -1,
}
NUM_BOOST_ROUND       = 3000
EARLY_STOPPING_ROUNDS = 150
N_PLAYERS             = 7

CLIP_EPS = 1e-6
OOF_YEARS   = list(range(2020, 2025))
TRAIN_START = 2015

# --- Feature column definitions (same as 32_run_xgboost_cv.py) -------------
PLAYER_MODEL_FEATURES = [
    "m_ewma_pre", "q_pre", "days_since_first_report_pre",
    "days_since_last_dnp_pre", "consec_dnps_pre", "played_last_game_pre",
    "minutes_last_game_pre", "days_since_last_played_pre",
    "injury_present_flag_pre",
]

RECENT_FORM_FEATURES = [
    "net_rtg_ewma_pre", "efg_ewma_pre", "tov_pct_ewma_pre",
    "orb_pct_ewma_pre", "ftr_ewma_pre",
]

STYLE_FEATURES = [
    "off_3pa_rate_pre", "def_3pa_allowed_pre", "off_2pa_rate_pre",
    "def_2pa_allowed_pre", "off_tov_pct_pre", "def_forced_tov_pre",
]

SCHEDULE_FEATURES = [
    "days_rest_pre", "is_b2b_pre", "games_last_4_days_pre",
    "games_last_7_days_pre", "travel_miles_pre", "timezone_shift_hours_pre",
]

METADATA_COLS = {
    "game_id", "game_ts", "game_date", "season", "is_playoff",
    "home_team_id", "away_team_id", "home_franchise_id", "away_franchise_id",
    "home_elo_pre", "away_elo_pre", "p_elo", "base_margin", "home_win",
    "home_origin_city_pre", "away_origin_city_pre",
    "home_current_city_pre", "away_current_city_pre",
}


def build_feature_cols(n_players: int) -> list[str]:
    cols = []
    for side in ("home", "away"):
        for slot in range(1, n_players + 1):
            for feat in PLAYER_MODEL_FEATURES:
                cols.append(f"{side}_p{slot}_{feat}")
    for feat in RECENT_FORM_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")
    for feat in STYLE_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")
    for feat in SCHEDULE_FEATURES:
        cols.append(f"home_{feat}")
        cols.append(f"away_{feat}")
    return cols


FEATURE_COLS = build_feature_cols(N_PLAYERS)
print(f"Feature columns: {len(FEATURE_COLS)}")


# --- Data loader ------------------------------------------------------------
def load_gold(season: int) -> pd.DataFrame:
    p = GOLD_DIR / f"game_xgboost_input_{season}_REGPST.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    df = pd.read_csv(p)
    df["season"] = df["season"].astype(int)
    # Exclude cold-start games (first ~9 of 2015)
    cold = (df["home_p1_m_ewma_pre"] == 0) | (df["away_p1_m_ewma_pre"] == 0)
    n_drop = cold.sum()
    if n_drop:
        df = df[~cold].reset_index(drop=True)
        print(f"  [load {season}] dropped {n_drop} cold-start rows")
    return df


def load_range(start: int, end: int) -> pd.DataFrame:
    """Load and concatenate gold tables for seasons [start, end] inclusive."""
    dfs = []
    for y in range(start, end + 1):
        try:
            dfs.append(load_gold(y))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    return pd.concat(dfs, ignore_index=True).dropna(subset=["home_win", "base_margin"])


# ============================================================================
# Section 2 — Helper functions
# ============================================================================

def clip(p: np.ndarray) -> np.ndarray:
    return np.clip(p, CLIP_EPS, 1 - CLIP_EPS)


def logit(p: np.ndarray) -> np.ndarray:
    p = clip(p)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = clip(p)
    return {
        "log_loss":  log_loss(y, p),
        "brier":     brier_score_loss(y, p),
        "accuracy":  accuracy_score(y, (p >= 0.5).astype(int)),
        "n_games":   len(y),
    }


# ============================================================================
# Section 2b — Model fit/predict wrappers
# ============================================================================

def fit_predict_logreg_no_elo(X_train, y_train, X_test):
    """Logistic regression on features only (no Elo)."""
    lr = LogisticRegression(
        max_iter=2000, solver="lbfgs", penalty="l2", C=1.0, random_state=42,
    )
    lr.fit(X_train, y_train)
    return lr.predict_proba(X_test)[:, 1]


def fit_predict_logreg_with_elo(X_train, y_train, bm_train, X_test, bm_test):
    """Logistic regression with logit(p_elo) as offset.

    Implementation: include logit(p_elo) as a feature with its coefficient
    unconstrained.  This is equivalent to learning  logit(p) = α + β_0·logit(p_elo) + β'x.
    If β_0 ≈ 1 the model behaves like a correction on top of Elo.
    """
    X_tr = np.column_stack([bm_train, X_train])
    X_te = np.column_stack([bm_test, X_test])
    lr = LogisticRegression(
        max_iter=2000, solver="lbfgs", penalty="l2", C=1.0, random_state=42,
    )
    lr.fit(X_tr, y_train)
    return lr.predict_proba(X_te)[:, 1]


def fit_predict_xgb(X_train, y_train, X_es_val, y_es_val,
                     X_test, bm_train=None, bm_es_val=None, bm_test=None,
                     feature_names=None):
    """XGBoost with optional base_margin (Elo prior).

    Uses nested early stopping: find best_round on es_val, then retrain on
    train+es_val for exactly best_round trees, predict on test.
    """
    def make_dm(X, y, bm):
        kw = dict(
            data=X.astype(float), label=y.astype(float),
            feature_names=feature_names, missing=np.nan,
        )
        if bm is not None:
            kw["base_margin"] = bm.astype(float)
        return xgb.DMatrix(**kw)

    # We split train into es_train / es_val for early stopping
    # But the caller already provides these splits
    dm_es_train = make_dm(X_train, y_train, bm_train)
    dm_es_val   = make_dm(X_es_val, y_es_val, bm_es_val)

    es_model = xgb.train(
        XGB_PARAMS, dm_es_train,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dm_es_val, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_round = es_model.best_iteration + 1

    # Retrain on train + es_val
    X_full  = np.vstack([X_train, X_es_val])
    y_full  = np.concatenate([y_train, y_es_val])
    bm_full = np.concatenate([bm_train, bm_es_val]) if bm_train is not None else None
    dm_full = make_dm(X_full, y_full, bm_full)
    dm_test = make_dm(X_test, np.zeros(len(X_test)), bm_test)

    model = xgb.train(XGB_PARAMS, dm_full, num_boost_round=best_round, verbose_eval=False)
    preds = model.predict(dm_test)
    return clip(preds), best_round, model


# ============================================================================
# Section 3 — Walk-forward comparison on 2020–2024
# ============================================================================

print("\n" + "=" * 70)
print("WALK-FORWARD OOF COMPARISON  (2020-2024)")
print("=" * 70)

all_oof = []

for test_year in OOF_YEARS:
    print(f"\n--- Fold: test_year = {test_year} ---")
    es_val_year = test_year - 1
    es_train_years = list(range(TRAIN_START, es_val_year))

    # Load splits
    es_train_df = load_range(TRAIN_START, es_val_year - 1)
    es_val_df   = load_gold(es_val_year).dropna(subset=["home_win", "base_margin"])
    test_df     = load_gold(test_year).dropna(subset=["home_win", "base_margin"])

    avail = [c for c in FEATURE_COLS if c in es_train_df.columns]

    X_es_train = es_train_df[avail].values
    X_es_val   = es_val_df[avail].values
    X_test     = test_df[avail].values
    y_es_train = es_train_df["home_win"].values.astype(float)
    y_es_val   = es_val_df["home_win"].values.astype(float)
    y_test     = test_df["home_win"].values.astype(float)
    bm_es_train = es_train_df["base_margin"].values.astype(float)
    bm_es_val   = es_val_df["base_margin"].values.astype(float)
    bm_test     = test_df["base_margin"].values.astype(float)
    p_elo_test  = clip(test_df["p_elo"].values.astype(float))

    fold_preds = {}

    # 1. Elo-only
    fold_preds["elo"] = p_elo_test

    # 2. Logistic regression without Elo
    # For logreg, we train on full train+es_val (no early stopping needed)
    X_lr_train = np.vstack([X_es_train, X_es_val])
    y_lr_train = np.concatenate([y_es_train, y_es_val])
    bm_lr_train = np.concatenate([bm_es_train, bm_es_val])

    fold_preds["logreg_no_elo"] = fit_predict_logreg_no_elo(
        X_lr_train, y_lr_train, X_test,
    )

    # 3. XGBoost without Elo
    preds_xgb_no_elo, br_no_elo, _ = fit_predict_xgb(
        X_es_train, y_es_train, X_es_val, y_es_val, X_test,
        feature_names=avail,
    )
    fold_preds["xgb_no_elo"] = preds_xgb_no_elo
    print(f"  xgb_no_elo best_round={br_no_elo}")

    # 4. Logistic regression with Elo
    fold_preds["logreg_with_elo"] = fit_predict_logreg_with_elo(
        X_lr_train, y_lr_train, bm_lr_train, X_test, bm_test,
    )

    # 5. XGBoost with Elo
    preds_xgb_elo, br_elo, _ = fit_predict_xgb(
        X_es_train, y_es_train, X_es_val, y_es_val, X_test,
        bm_train=bm_es_train, bm_es_val=bm_es_val, bm_test=bm_test,
        feature_names=avail,
    )
    fold_preds["xgb_with_elo"] = preds_xgb_elo
    print(f"  xgb_with_elo best_round={br_elo}")

    # Collect per-game predictions
    for model_name, preds in fold_preds.items():
        for i in range(len(test_df)):
            all_oof.append({
                "game_id":   test_df.iloc[i]["game_id"],
                "season":    int(test_df.iloc[i]["season"]),
                "home_win":  int(y_test[i]),
                "model":     model_name,
                "pred_prob": float(preds[i]),
            })

    # Print fold metrics
    for name, preds in fold_preds.items():
        m = metrics(y_test, preds)
        print(f"  {name:25s}  LL={m['log_loss']:.5f}  Brier={m['brier']:.5f}  Acc={m['accuracy']:.3f}")


oof_df = pd.DataFrame(all_oof)
oof_df.to_csv(OUT_DIR / "oof_model_comparison_2020_2024.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'oof_model_comparison_2020_2024.csv'}")


# ============================================================================
# Table 1 — Pooled OOF summary
# ============================================================================

MODEL_ORDER = ["elo", "logreg_no_elo", "xgb_no_elo",
               "logreg_with_elo", "xgb_with_elo"]

print("\n" + "=" * 70)
print("TABLE 1 — Development-period model comparison (OOF 2020-2024 pooled)")
print("=" * 70)

rows = []
for model_name in MODEL_ORDER:
    sub = oof_df[oof_df["model"] == model_name]
    m = metrics(sub["home_win"].values, sub["pred_prob"].values)
    m["model"] = model_name
    rows.append(m)

dev_summary = pd.DataFrame(rows)[["model", "log_loss", "brier", "accuracy", "n_games"]]
print(dev_summary.to_string(index=False))
dev_summary.to_csv(OUT_DIR / "development_summary_table.csv", index=False)


# ============================================================================
# Table 2 — Per-fold breakdown
# ============================================================================

print("\n" + "=" * 70)
print("TABLE 2 — Per-fold development-period comparison")
print("=" * 70)

fold_rows = []
for test_year in OOF_YEARS:
    for model_name in MODEL_ORDER:
        sub = oof_df[(oof_df["model"] == model_name) & (oof_df["season"] == test_year)]
        if len(sub) == 0:
            continue
        m = metrics(sub["home_win"].values, sub["pred_prob"].values)
        m["fold_year"] = test_year
        m["model"] = model_name
        fold_rows.append(m)

fold_table = pd.DataFrame(fold_rows)[["fold_year", "model", "log_loss", "brier", "accuracy", "n_games"]]
print(fold_table.to_string(index=False))
fold_table.to_csv(OUT_DIR / "development_by_fold_table.csv", index=False)


# ============================================================================
# Section 4 — Final 2025 holdout
# ============================================================================

print("\n" + "=" * 70)
print("FINAL 2025 HOLDOUT COMPARISON")
print("=" * 70)

# Load all training data (2015-2024) and 2025 holdout
full_train_df = load_range(TRAIN_START, 2024)
holdout_df    = load_gold(2025).dropna(subset=["home_win", "base_margin"])

avail = [c for c in FEATURE_COLS if c in full_train_df.columns]

X_full_train = full_train_df[avail].values
y_full_train = full_train_df["home_win"].values.astype(float)
bm_full_train = full_train_df["base_margin"].values.astype(float)

X_holdout   = holdout_df[avail].values
y_holdout   = holdout_df["home_win"].values.astype(float)
bm_holdout  = holdout_df["base_margin"].values.astype(float)
p_elo_holdout = clip(holdout_df["p_elo"].values.astype(float))

# For XGBoost holdout: use 2015-2023 as ES-train, 2024 as ES-val, then
# retrain on full 2015-2024 for best_round trees.
es_train_holdout_df = load_range(TRAIN_START, 2023)
es_val_holdout_df   = load_gold(2024).dropna(subset=["home_win", "base_margin"])

X_es_tr_h = es_train_holdout_df[avail].values
y_es_tr_h = es_train_holdout_df["home_win"].values.astype(float)
bm_es_tr_h = es_train_holdout_df["base_margin"].values.astype(float)

X_es_val_h = es_val_holdout_df[avail].values
y_es_val_h = es_val_holdout_df["home_win"].values.astype(float)
bm_es_val_h = es_val_holdout_df["base_margin"].values.astype(float)

holdout_preds = {}

# 1. Elo-only
holdout_preds["elo"] = p_elo_holdout

# 2. Logistic regression without Elo
holdout_preds["logreg_no_elo"] = fit_predict_logreg_no_elo(
    X_full_train, y_full_train, X_holdout,
)

# 3. XGBoost without Elo
preds_xgb_no_elo_h, br, _ = fit_predict_xgb(
    X_es_tr_h, y_es_tr_h, X_es_val_h, y_es_val_h, X_holdout,
    feature_names=avail,
)
holdout_preds["xgb_no_elo"] = preds_xgb_no_elo_h
print(f"  xgb_no_elo holdout best_round={br}")

# 4. Logistic regression with Elo
holdout_preds["logreg_with_elo"] = fit_predict_logreg_with_elo(
    X_full_train, y_full_train, bm_full_train, X_holdout, bm_holdout,
)

# 5. XGBoost with Elo
preds_xgb_elo_h, br, _ = fit_predict_xgb(
    X_es_tr_h, y_es_tr_h, X_es_val_h, y_es_val_h, X_holdout,
    bm_train=bm_es_tr_h, bm_es_val=bm_es_val_h, bm_test=bm_holdout,
    feature_names=avail,
)
holdout_preds["xgb_with_elo"] = preds_xgb_elo_h
print(f"  xgb_with_elo holdout best_round={br}")

# Save per-game predictions
holdout_rows = []
for model_name, preds in holdout_preds.items():
    for i in range(len(holdout_df)):
        holdout_rows.append({
            "game_id":   holdout_df.iloc[i]["game_id"],
            "season":    2025,
            "home_win":  int(y_holdout[i]),
            "model":     model_name,
            "pred_prob": float(preds[i]),
        })

holdout_pred_df = pd.DataFrame(holdout_rows)
holdout_pred_df.to_csv(OUT_DIR / "holdout_model_comparison_2025.csv", index=False)
print(f"\nSaved: {OUT_DIR / 'holdout_model_comparison_2025.csv'}")


# ============================================================================
# Table 3 — 2025 holdout summary
# ============================================================================

print("\n" + "=" * 70)
print("TABLE 3 — Final 2025 holdout comparison")
print("=" * 70)

holdout_rows_summary = []
for model_name in MODEL_ORDER:
    sub = holdout_pred_df[holdout_pred_df["model"] == model_name]
    m = metrics(sub["home_win"].values, sub["pred_prob"].values)
    m["model"] = model_name
    holdout_rows_summary.append(m)

holdout_summary = pd.DataFrame(holdout_rows_summary)[["model", "log_loss", "brier", "accuracy", "n_games"]]
print(holdout_summary.to_string(index=False))
holdout_summary.to_csv(OUT_DIR / "holdout_summary_table.csv", index=False)


# ============================================================================
# Table 4 — Improvement relative to Elo
# ============================================================================

print("\n" + "=" * 70)
print("TABLE 4 — Improvement relative to Elo")
print("=" * 70)

elo_dev  = dev_summary[dev_summary["model"] == "elo"].iloc[0]
elo_hold = holdout_summary[holdout_summary["model"] == "elo"].iloc[0]

delta_rows = []
for _, row in dev_summary.iterrows():
    if row["model"] == "elo":
        continue
    delta_rows.append({
        "model":          row["model"],
        "oof_d__logloss":  row["log_loss"] - elo_dev["log_loss"],
        "oof_d__brier":    row["brier"] - elo_dev["brier"],
        "oof_d__accuracy": row["accuracy"] - elo_dev["accuracy"],
    })

for _, row in holdout_summary.iterrows():
    if row["model"] == "elo":
        continue
    for d in delta_rows:
        if d["model"] == row["model"]:
            d["holdout_d__logloss"]  = row["log_loss"] - elo_hold["log_loss"]
            d["holdout_d__brier"]    = row["brier"] - elo_hold["brier"]
            d["holdout_d__accuracy"] = row["accuracy"] - elo_hold["accuracy"]

delta_df = pd.DataFrame(delta_rows)
print(delta_df.to_string(index=False))


# ============================================================================
# Section 5 — Graphs
# ============================================================================

MODEL_LABELS = {
    "elo":               "Elo Only",
    "logreg_no_elo":     "LogReg\n(no Elo)",
    "xgb_no_elo":        "XGBoost\n(no Elo)",
    "logreg_with_elo":   "LogReg\n(+ Elo)",
    "xgb_with_elo":      "XGBoost\n(+ Elo)",
}

COLORS = {
    "elo":               "#7f8c8d",
    "logreg_no_elo":     "#e67e22",
    "xgb_no_elo":        "#e74c3c",
    "logreg_with_elo":   "#2ecc71",
    "xgb_with_elo":      "#3498db",
}


def bar_chart(summary_df, metric, title, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(MODEL_ORDER))
    vals = [summary_df[summary_df["model"] == m][metric].iloc[0] for m in MODEL_ORDER]
    colors = [COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(x, vals, color=colors, edgecolor="white", width=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Value labels on bars
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# Graph 1 — OOF log loss
bar_chart(dev_summary, "log_loss",
          "Development OOF Log Loss (2020-2024 pooled)",
          OUT_DIR / "oof_logloss_by_model.png")

# Graph 2 — Holdout log loss
bar_chart(holdout_summary, "log_loss",
          "2025 Holdout Log Loss",
          OUT_DIR / "holdout_logloss_by_model.png")

# Graph 3 — Per-fold line plot
fig, ax = plt.subplots(figsize=(10, 5))
for model_name in MODEL_ORDER:
    sub = fold_table[fold_table["model"] == model_name]
    ax.plot(sub["fold_year"], sub["log_loss"], marker="o", linewidth=1.5,
            label=MODEL_LABELS[model_name].replace("\n", " "),
            color=COLORS[model_name])

ax.set_xlabel("Validation Year", fontsize=11)
ax.set_ylabel("Log Loss", fontsize=11)
ax.set_title("Per-Fold Log Loss (2020-2024)", fontsize=13, fontweight="bold")
ax.legend(fontsize=8, loc="best")
ax.set_xticks(OOF_YEARS)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(OUT_DIR / "per_fold_logloss.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'per_fold_logloss.png'}")


# Graph 4 — Calibration / reliability curves (OOF)
def reliability_curve(y_true, y_prob, n_bins=10):
    """Return (mean_pred, frac_positive, count) per bin."""
    bins = np.linspace(0, 1, n_bins + 1)
    mean_pred, frac_pos, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
        counts.append(mask.sum())
    return np.array(mean_pred), np.array(frac_pos), np.array(counts)


cal_models = ["elo", "logreg_with_elo", "xgb_with_elo"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (title, source_df) in zip(axes, [
    ("OOF 2020-2024", oof_df),
    ("2025 Holdout", holdout_pred_df),
]):
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")
    for model_name in cal_models:
        sub = source_df[source_df["model"] == model_name]
        mp, fp, _ = reliability_curve(sub["home_win"].values, sub["pred_prob"].values)
        ax.plot(mp, fp, marker="o", linewidth=1.5, color=COLORS[model_name],
                label=MODEL_LABELS[model_name].replace("\n", " "))
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction Positive")
    ax.set_title(f"Calibration — {title}", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "calibration_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'calibration_curves.png'}")

# Graph 5 — Brier score comparison
bar_chart(dev_summary, "brier",
          "Development OOF Brier Score (2020-2024 pooled)",
          OUT_DIR / "oof_brier_by_model.png")

bar_chart(holdout_summary, "brier",
          "2025 Holdout Brier Score",
          OUT_DIR / "holdout_brier_by_model.png")


# ============================================================================
# Section 6 — Summary of findings
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)

best_dev     = dev_summary.loc[dev_summary["log_loss"].idxmin()]
best_holdout = holdout_summary.loc[holdout_summary["log_loss"].idxmin()]

print(f"\nBest development model (OOF log loss):  {best_dev['model']}  ({best_dev['log_loss']:.5f})")
print(f"Best holdout model (2025 log loss):      {best_holdout['model']}  ({best_holdout['log_loss']:.5f})")

elo_ll_dev  = dev_summary[dev_summary["model"] == "elo"]["log_loss"].iloc[0]
elo_ll_hold = holdout_summary[holdout_summary["model"] == "elo"]["log_loss"].iloc[0]

print(f"\nElo-only baseline:  dev={elo_ll_dev:.5f}  holdout={elo_ll_hold:.5f}")

for model_name in ["logreg_no_elo", "xgb_no_elo", "logreg_with_elo",
                    "xgb_with_elo"]:
    dev_ll  = dev_summary[dev_summary["model"] == model_name]["log_loss"].iloc[0]
    hold_ll = holdout_summary[holdout_summary["model"] == model_name]["log_loss"].iloc[0]
    dev_d   = dev_ll - elo_ll_dev
    hold_d  = hold_ll - elo_ll_hold
    print(f"  {model_name:25s}  dev d_={dev_d:+.5f}  holdout d_={hold_d:+.5f}"
          f"  {'+ beats Elo' if hold_d < 0 else '- worse than Elo'}")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Done.")
