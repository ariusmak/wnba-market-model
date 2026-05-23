"""Walk-forward CV + holdout helpers for forecasting and ablation notebooks.

Generic enough to support:
  - Multi-model OOF (Elo, LogReg with/without Elo, XGB with/without Elo).
  - Single-model OOF with feature-set variation (block ablation).
  - Final 2025 holdout fit using identical nested early-stopping protocol.

Every entry point uses the locked `XGB_PARAMS`, `MAX_ROUNDS`, `EARLY_STOP` from
`final_model`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# Make `organized/pipelines/05_modeling` importable from notebooks
_PIPE = Path(__file__).resolve().parents[3] / "pipelines" / "05_modeling"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from final_model import (  # noqa: E402
    CLIP_EPS, EARLY_STOP, FEAT_COLS, LABEL_COL, MAX_ROUNDS, XGB_PARAMS,
    clip, load_year,
)

LABEL = LABEL_COL


def logit(p: np.ndarray) -> np.ndarray:
    p = clip(p)
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = clip(p)
    return {
        "log_loss":  float(log_loss(y, p)),
        "brier":     float(brier_score_loss(y, p)),
        "accuracy":  float(accuracy_score(y, (p >= 0.5).astype(int))),
        "n_games":   int(len(y)),
    }


# --------------------------------------------------------------------------- #
# Single-fit wrappers                                                         #
# --------------------------------------------------------------------------- #

def _make_dm(X, y, bm, feat_names):
    kw = dict(data=X.astype(float), label=y.astype(float),
              feature_names=feat_names, missing=np.nan)
    if bm is not None:
        kw["base_margin"] = bm.astype(float)
    return xgb.DMatrix(**kw)


def fit_predict_xgb(
    X_tr, y_tr, X_val, y_val, X_te,
    bm_tr=None, bm_val=None, bm_te=None, feat_names=None,
):
    """Nested early stopping. Find best_round on (X_tr, X_val), retrain on the
    union for best_round trees, predict on X_te. Returns (preds, best_round, model)."""
    es_model = xgb.train(
        XGB_PARAMS, _make_dm(X_tr, y_tr, bm_tr, feat_names), MAX_ROUNDS,
        evals=[(_make_dm(X_val, y_val, bm_val, feat_names), "val")],
        early_stopping_rounds=EARLY_STOP, verbose_eval=False,
    )
    best_round = es_model.best_iteration + 1

    X_full = np.vstack([X_tr, X_val])
    y_full = np.concatenate([y_tr, y_val])
    bm_full = np.concatenate([bm_tr, bm_val]) if bm_tr is not None else None
    model = xgb.train(
        XGB_PARAMS, _make_dm(X_full, y_full, bm_full, feat_names),
        best_round, verbose_eval=False,
    )
    preds = model.predict(_make_dm(X_te, np.zeros(len(X_te)), bm_te, feat_names))
    return clip(preds), int(best_round), model


def fit_predict_logreg(X_tr, y_tr, X_te, *, with_elo: bool = False,
                       bm_tr=None, bm_te=None):
    """Logistic regression. If with_elo, prepend logit(p_elo) (i.e. base_margin)."""
    if with_elo:
        X_tr = np.column_stack([bm_tr, X_tr])
        X_te = np.column_stack([bm_te, X_te])
    lr = LogisticRegression(max_iter=2000, solver="lbfgs", penalty="l2",
                            C=1.0, random_state=42)
    lr.fit(X_tr, y_tr)
    return clip(lr.predict_proba(X_te)[:, 1]), lr


# --------------------------------------------------------------------------- #
# OOF runners                                                                 #
# --------------------------------------------------------------------------- #

def _slice_arrays(df: pd.DataFrame, feat_cols: list[str]):
    avail = [c for c in feat_cols if c in df.columns]
    return (
        df[avail].values,
        df[LABEL].values.astype(float),
        df["base_margin"].values.astype(float),
        clip(df["p_elo"].values.astype(float)),
        avail,
    )


def walk_forward_models(
    model_specs: dict[str, dict],
    *,
    oof_years: Iterable[int] = range(2020, 2025),
    train_start: int = 2015,
    feat_cols: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run walk-forward OOF for one or more model specs.

    `model_specs` is a dict like:
        {
          "elo":             {"type": "elo"},
          "logreg_no_elo":   {"type": "logreg", "with_elo": False},
          "logreg_with_elo": {"type": "logreg", "with_elo": True},
          "xgb_no_elo":      {"type": "xgb",    "use_bm": False},
          "xgb_with_elo":    {"type": "xgb",    "use_bm": True},
        }

    Returns a long DataFrame with columns:
        game_id, season, home_win, model, pred_prob.
    """
    feat_cols = feat_cols if feat_cols is not None else FEAT_COLS
    rows: list[dict] = []

    for test_year in oof_years:
        if verbose:
            print(f"--- Fold: test_year = {test_year} ---")
        es_val_year = test_year - 1
        es_tr_df = pd.concat(
            [load_year(y) for y in range(train_start, es_val_year)],
            ignore_index=True,
        ).dropna(subset=[LABEL, "base_margin"])
        es_val_df = load_year(es_val_year).dropna(subset=[LABEL, "base_margin"])
        test_df = load_year(test_year).dropna(subset=[LABEL, "base_margin"])

        X_tr, y_tr, bm_tr, _, _ = _slice_arrays(es_tr_df, feat_cols)
        X_val, y_val, bm_val, _, _ = _slice_arrays(es_val_df, feat_cols)
        X_te, y_te, bm_te, p_elo_te, avail = _slice_arrays(test_df, feat_cols)

        # Full train (train+val) for non-XGB models
        X_full = np.vstack([X_tr, X_val])
        y_full = np.concatenate([y_tr, y_val])
        bm_full = np.concatenate([bm_tr, bm_val])

        for name, spec in model_specs.items():
            mtype = spec["type"]
            if mtype == "elo":
                preds = p_elo_te
            elif mtype == "logreg":
                preds, _ = fit_predict_logreg(
                    X_full, y_full, X_te,
                    with_elo=spec.get("with_elo", False),
                    bm_tr=bm_full, bm_te=bm_te,
                )
            elif mtype == "xgb":
                use_bm = spec.get("use_bm", True)
                preds, best_round, _ = fit_predict_xgb(
                    X_tr, y_tr, X_val, y_val, X_te,
                    bm_tr=bm_tr if use_bm else None,
                    bm_val=bm_val if use_bm else None,
                    bm_te=bm_te if use_bm else None,
                    feat_names=avail,
                )
                if verbose:
                    print(f"  {name:24s} best_round={best_round}")
            else:
                raise ValueError(f"unknown model type: {mtype}")

            for i in range(len(test_df)):
                rows.append({
                    "game_id":   test_df.iloc[i]["game_id"],
                    "season":    int(test_df.iloc[i]["season"]),
                    "home_win":  int(y_te[i]),
                    "model":     name,
                    "pred_prob": float(preds[i]),
                })

    return pd.DataFrame(rows)


def fit_holdout_models(
    model_specs: dict[str, dict],
    *,
    holdout_year: int = 2025,
    train_start: int = 2015,
    feat_cols: list[str] | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Train each model on 2015..(H-1) using nested ES on H-1 and predict on H.

    Returns (per-game predictions DataFrame, dict of artefacts {name: model_obj}).
    """
    feat_cols = feat_cols if feat_cols is not None else FEAT_COLS
    es_tr_df = pd.concat(
        [load_year(y) for y in range(train_start, holdout_year - 1)],
        ignore_index=True,
    ).dropna(subset=[LABEL, "base_margin"])
    es_val_df = load_year(holdout_year - 1).dropna(subset=[LABEL, "base_margin"])
    test_df = load_year(holdout_year).dropna(subset=[LABEL, "base_margin"])

    X_tr, y_tr, bm_tr, _, _ = _slice_arrays(es_tr_df, feat_cols)
    X_val, y_val, bm_val, _, _ = _slice_arrays(es_val_df, feat_cols)
    X_te, y_te, bm_te, p_elo_te, avail = _slice_arrays(test_df, feat_cols)

    X_full = np.vstack([X_tr, X_val])
    y_full = np.concatenate([y_tr, y_val])
    bm_full = np.concatenate([bm_tr, bm_val])

    rows: list[dict] = []
    artefacts: dict = {}
    for name, spec in model_specs.items():
        mtype = spec["type"]
        if mtype == "elo":
            preds = p_elo_te
            artefacts[name] = None
        elif mtype == "logreg":
            preds, lr = fit_predict_logreg(
                X_full, y_full, X_te,
                with_elo=spec.get("with_elo", False),
                bm_tr=bm_full, bm_te=bm_te,
            )
            artefacts[name] = lr
        elif mtype == "xgb":
            use_bm = spec.get("use_bm", True)
            preds, best_round, model = fit_predict_xgb(
                X_tr, y_tr, X_val, y_val, X_te,
                bm_tr=bm_tr if use_bm else None,
                bm_val=bm_val if use_bm else None,
                bm_te=bm_te if use_bm else None,
                feat_names=avail,
            )
            if verbose:
                print(f"  {name:24s} best_round={best_round}")
            artefacts[name] = model
        else:
            raise ValueError(f"unknown model type: {mtype}")

        for i in range(len(test_df)):
            rows.append({
                "game_id":   test_df.iloc[i]["game_id"],
                "season":    int(test_df.iloc[i]["season"]),
                "home_win":  int(y_te[i]),
                "model":     name,
                "pred_prob": float(preds[i]),
            })

    preds_df = pd.DataFrame(rows)
    return preds_df, artefacts


def summarize_predictions(preds_df: pd.DataFrame, *, by_fold: bool = False) -> pd.DataFrame:
    """Pool log-loss / Brier / accuracy by model (and optionally by fold-year)."""
    keys = ["model"] + (["season"] if by_fold else [])
    out = []
    for vals, sub in preds_df.groupby(keys):
        m = metrics(sub["home_win"].values, sub["pred_prob"].values)
        if isinstance(vals, tuple):
            row = dict(zip(keys, vals))
        else:
            row = {keys[0]: vals}
        row.update(m)
        out.append(row)
    cols = keys + ["log_loss", "brier", "accuracy", "n_games"]
    return pd.DataFrame(out)[cols]


# --------------------------------------------------------------------------- #
# Standard model spec dictionaries used by the paper notebooks                #
# --------------------------------------------------------------------------- #

PAPER_MODEL_SPECS: dict[str, dict] = {
    "elo":              {"type": "elo"},
    "logreg_no_elo":    {"type": "logreg", "with_elo": False},
    "xgb_no_elo":       {"type": "xgb",    "use_bm": False},
    "logreg_with_elo":  {"type": "logreg", "with_elo": True},
    "xgb_with_elo":     {"type": "xgb",    "use_bm": True},
}

PAPER_MODEL_ORDER = list(PAPER_MODEL_SPECS.keys())

PAPER_MODEL_LABELS = {
    "elo":             "Elo Only",
    "logreg_no_elo":   "LogReg (no Elo)",
    "xgb_no_elo":      "XGBoost (no Elo)",
    "logreg_with_elo": "LogReg (+ Elo)",
    "xgb_with_elo":    "XGBoost (+ Elo)",
}

PAPER_MODEL_COLORS = {
    "elo":             "#7f8c8d",
    "logreg_no_elo":   "#e67e22",
    "xgb_no_elo":      "#e74c3c",
    "logreg_with_elo": "#2ecc71",
    "xgb_with_elo":    "#3498db",
}
