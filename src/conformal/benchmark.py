"""Sweep the target miscoverage alpha and compare, on held-out test data, the realized coverage
and set size of split-conformal vs naive softmax thresholding. The headline: conformal tracks the
target coverage line; naive softmax falls below it because the model is over-confident."""
from __future__ import annotations

from .data import build
from .predict import (conformal_qhat, coverage_and_size, naive_softmax_sets, split_conformal_sets)


def run(alphas=(0.20, 0.10, 0.05), seed: int = 7) -> dict:
    split = build(seed=seed)
    rows = []
    for a in alphas:
        target = round(1 - a, 4)
        qhat = conformal_qhat(split.p_calib, split.y_calib, a)
        conf = coverage_and_size(split_conformal_sets(split.p_test, qhat), split.y_test)
        naive = coverage_and_size(naive_softmax_sets(split.p_test, a), split.y_test)
        rows.append({
            "alpha": a, "target_coverage": target,
            "conformal": conf, "naive": naive,
            "conformal_meets_target": conf["coverage"] >= target - 0.02,
            "naive_shortfall": round(target - naive["coverage"], 4),
        })
    return {"grid": rows,
            "_meta": {"n_test": len(split.y_test), "n_classes": split.n_classes,
                      "model": "RandomForest", "test_accuracy": round(split.test_accuracy, 4)}}
