import numpy as np

from conformal.benchmark import run
from conformal.data import build
from conformal.predict import (conformal_qhat, coverage_and_size, naive_softmax_sets,
                               split_conformal_sets)


def test_build_shapes():
    s = build(n_samples=2000, n_classes=5, seed=1)
    assert s.p_calib.shape[1] == 5 and s.p_test.shape[1] == 5
    assert 0.4 < s.test_accuracy < 1.0


def test_conformal_meets_marginal_coverage():
    s = build(seed=3)
    for alpha in (0.2, 0.1, 0.05):
        qhat = conformal_qhat(s.p_calib, s.y_calib, alpha)
        cov = coverage_and_size(split_conformal_sets(s.p_test, qhat), s.y_test)["coverage"]
        # distribution-free guarantee: coverage >= 1 - alpha (small sampling slack)
        assert cov >= (1 - alpha) - 0.03, (alpha, cov)


def test_lower_alpha_gives_larger_sets():
    s = build(seed=3)
    q_loose = conformal_qhat(s.p_calib, s.y_calib, 0.20)
    q_tight = conformal_qhat(s.p_calib, s.y_calib, 0.05)
    size_loose = coverage_and_size(split_conformal_sets(s.p_test, q_loose), s.y_test)["avg_set_size"]
    size_tight = coverage_and_size(split_conformal_sets(s.p_test, q_tight), s.y_test)["avg_set_size"]
    assert size_tight >= size_loose          # higher coverage demands bigger sets


def test_naive_softmax_undercovers():
    res = run()
    # the headline: at least one target where naive softmax misses the guarantee
    assert any(r["naive_shortfall"] > 0.02 for r in res["grid"])
    # and conformal meets every target
    assert all(r["conformal_meets_target"] for r in res["grid"])


def test_conformal_covers_better_than_naive_at_same_target():
    res = run()
    for r in res["grid"]:
        assert r["conformal"]["coverage"] >= r["naive"]["coverage"] - 1e-9
