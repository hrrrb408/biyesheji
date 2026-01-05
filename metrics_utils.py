from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def predict_proba_pos(model: Any, X, *, pos_label: int = 1) -> np.ndarray:
    proba = model.predict_proba(X)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba[:, 0]

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        try:
            clf = model.named_steps.get("clf")
            classes = getattr(clf, "classes_", None)
        except Exception:
            classes = None

    if classes is None:
        return proba[:, 1]

    classes_arr = np.asarray(classes)
    idx = np.where(classes_arr == pos_label)[0]
    if idx.size == 0:
        return proba[:, 1]
    return proba[:, int(idx[0])]


def decision_function_pos(model: Any, X, *, pos_label: int = 1) -> np.ndarray:
    score = model.decision_function(X)
    score = np.asarray(score, dtype=float)

    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        try:
            clf = model.named_steps.get("clf")
            classes = getattr(clf, "classes_", None)
        except Exception:
            classes = None

    if classes is None:
        if score.ndim == 2 and score.shape[1] == 1:
            return score[:, 0]
        return score

    classes_arr = np.asarray(classes)
    if classes_arr.size == 2 and score.ndim == 1:
        if classes_arr[1] == pos_label:
            return score
        if classes_arr[0] == pos_label:
            return -score
        return score

    if score.ndim == 2:
        if score.shape[1] == 1:
            return score[:, 0]
        idx = np.where(classes_arr == pos_label)[0]
        if idx.size == 0:
            return score[:, 1]
        return score[:, int(idx[0])]

    return score


def bootstrap_auc_ci(
    y_true,
    y_prob,
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(int(seed))
    n = int(y_true.size)
    aucs: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n, endpoint=False)
        yt = y_true[idx]
        if np.unique(yt).size < 2:
            continue
        yp = y_prob[idx]
        aucs.append(float(roc_auc_score(yt, yp)))
    if not aucs:
        return (np.nan, np.nan)
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return (lo, hi)


def compute_metrics_with_ci(
    compute_metrics: Callable[..., dict[str, Any]],
    y_true,
    y_prob,
    *,
    thr: float = 0.5,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    metrics = compute_metrics(y_true, y_prob, thr=float(thr))
    lo, hi = bootstrap_auc_ci(y_true, y_prob, n_boot=int(n_boot), seed=int(seed))
    metrics["AUC_CI95_L"] = float(lo) if np.isfinite(lo) else np.nan
    metrics["AUC_CI95_U"] = float(hi) if np.isfinite(hi) else np.nan
    return metrics
