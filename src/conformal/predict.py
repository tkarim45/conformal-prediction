"""Two ways to turn softmax probabilities into a prediction SET.

split_conformal — LAC (Least Ambiguous set-valued Classifier). On the calibration set the
nonconformity score of the true label is s = 1 - p[true]. Take the finite-sample-corrected
(1-alpha) quantile q̂ of those scores; the prediction set for a new x is every class with
1 - p_y ≤ q̂, i.e. p_y ≥ 1 - q̂. This carries a *distribution-free, finite-sample* guarantee:
P(y ∈ set) ≥ 1 - alpha, no matter how miscalibrated the model is.

naive_softmax — what practitioners actually do: predict the top class and add any other class
whose softmax clears the nominal confidence 1 - alpha. Because it never grows the set much, its
coverage plateaus near the model's *accuracy* — so it simply cannot deliver a 90% or 95% target
on an 81%-accurate model, no matter how you set alpha. It trusts the softmax to mean what it says.
"""
from __future__ import annotations

import numpy as np


def conformal_qhat(p_calib: np.ndarray, y_calib: np.ndarray, alpha: float) -> float:
    n = len(y_calib)
    scores = 1.0 - p_calib[np.arange(n), y_calib]          # nonconformity of the true label
    # finite-sample correction: the ceil((n+1)(1-alpha))/n quantile
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def split_conformal_sets(p: np.ndarray, qhat: float) -> list[set]:
    keep = p >= (1.0 - qhat)
    return [set(np.where(row)[0]) for row in keep]


def naive_softmax_sets(p: np.ndarray, alpha: float) -> list[set]:
    # predict the argmax, then add any class above the nominal confidence 1 - alpha.
    thresh = 1.0 - alpha
    sets = []
    for row in p:
        s = {int(row.argmax())}
        s.update(int(j) for j in np.where(row >= thresh)[0])
        sets.append(s)
    return sets


def coverage_and_size(sets: list[set], y_true: np.ndarray) -> dict:
    covered = sum(int(y in s) for s, y in zip(sets, y_true))
    sizes = np.array([len(s) for s in sets])
    return {
        "coverage": round(covered / len(y_true), 4),
        "avg_set_size": round(float(sizes.mean()), 3),
        "empty_set_rate": round(float((sizes == 0).mean()), 4),
    }
