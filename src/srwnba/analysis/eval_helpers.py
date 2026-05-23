"""Evaluation helpers: reliability curves, calibration tables, Platt scaling, bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_PIPE = Path(__file__).resolve().parents[3] / "pipelines" / "05_modeling"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from final_model import clip  # noqa: E402


def logit(p: np.ndarray) -> np.ndarray:
    p = clip(np.asarray(p))
    return np.log(p / (1 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


# --------------------------------------------------------------------------- #
# Reliability                                                                 #
# --------------------------------------------------------------------------- #

def reliability_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width binning. Returns DataFrame with columns:
    bin_lo, bin_hi, mean_pred, observed_rate, n.
    """
    y = np.asarray(y); p = np.asarray(p)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "mean_pred": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Platt scaling                                                               #
# --------------------------------------------------------------------------- #

def fit_platt(y: np.ndarray, p_raw: np.ndarray) -> tuple[float, float]:
    """Fit Platt scaling: logit(p_cal) = a + b * logit(p_raw). Returns (a, b)."""
    z = logit(p_raw).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs",
                            max_iter=5000, fit_intercept=True)
    lr.fit(z, np.asarray(y).astype(int))
    return float(lr.intercept_[0]), float(lr.coef_[0, 0])


def apply_platt(p_raw: np.ndarray, a: float, b: float) -> np.ndarray:
    return clip(sigmoid(a + b * logit(p_raw)))


# --------------------------------------------------------------------------- #
# Bootstrap                                                                   #
# --------------------------------------------------------------------------- #

def bootstrap_diff(values_a: np.ndarray, values_b: np.ndarray,
                   *, n_boot: int = 10_000, statistic=np.mean,
                   ci: tuple[float, float] = (2.5, 97.5),
                   seed: int = 42) -> dict:
    """Paired bootstrap of (statistic(values_a) - statistic(values_b)) over
    matched indices. `values_a` and `values_b` need not be the same length —
    we resample independently if lengths differ.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(values_a); b = np.asarray(values_b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = statistic(sa) - statistic(sb)
    lo, hi = np.percentile(diffs, ci)
    return {
        "mean_a": float(statistic(a)),
        "mean_b": float(statistic(b)),
        "diff": float(statistic(a) - statistic(b)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_a_gt_b": float((diffs > 0).mean()),
        "diffs": diffs,
    }


def bootstrap_distribution(values: np.ndarray, *, n_boot: int = 10_000,
                           statistic=np.mean, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = np.asarray(values)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = statistic(rng.choice(a, size=len(a), replace=True))
    return out
