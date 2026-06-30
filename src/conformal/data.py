"""A synthetic multiclass problem + a deliberately over-confident classifier.

We use a Random Forest on a noisy 6-class dataset: RF softmax (vote fractions) is well known to
be over-confident on hard data, which is exactly the setting where naive "trust the softmax"
uncertainty breaks and conformal's distribution-free guarantee earns its keep.

The data is split three ways: train (fit the model), calibration (compute conformal thresholds —
the model never trains on it), and test (evaluate coverage).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Split:
    model: RandomForestClassifier
    p_calib: np.ndarray   # softmax probs on calibration set
    y_calib: np.ndarray
    p_test: np.ndarray    # softmax probs on test set
    y_test: np.ndarray
    n_classes: int
    test_accuracy: float


def build(n_samples: int = 6000, n_classes: int = 6, seed: int = 7) -> Split:
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=8, n_redundant=4,
        n_classes=n_classes, n_clusters_per_class=1, flip_y=0.10, class_sep=1.0,
        random_state=seed,
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_samples)
    a, b = n_samples // 2, int(n_samples * 0.75)
    tr, cal, te = idx[:a], idx[a:b], idx[b:]

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
    model.fit(X[tr], y[tr])

    p_cal, p_te = model.predict_proba(X[cal]), model.predict_proba(X[te])
    acc = float((model.predict(X[te]) == y[te]).mean())
    return Split(model=model, p_calib=p_cal, y_calib=y[cal], p_test=p_te, y_test=y[te],
                 n_classes=n_classes, test_accuracy=acc)
