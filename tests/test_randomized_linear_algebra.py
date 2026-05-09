import numpy as np
import pytest

import crabbymetrics as cm


def make_low_rank_matrix(n=120, p=70, rank=8, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(n, rank))
    right = rng.normal(size=(rank, p))
    return left @ right + noise * rng.normal(size=(n, p))


def test_randomized_range_is_orthonormal():
    a = make_low_rank_matrix(seed=1)
    q = cm.randomized_range(a, rank=8, oversamples=6, power_iter=1, seed=123)
    assert q.shape == (a.shape[0], 14)
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)


def test_randomized_svd_approximates_low_rank_matrix():
    a = make_low_rank_matrix(seed=2)
    result = cm.randomized_svd(a, rank=8, oversamples=8, power_iter=2, seed=123)
    u = result["u"]
    s = result["singular_values"]
    vt = result["vt"]
    assert u.shape == (a.shape[0], 8)
    assert s.shape == (8,)
    assert vt.shape == (8, a.shape[1])

    approx = u @ np.diag(s) @ vt
    randomized_error = np.linalg.norm(a - approx, ord="fro") / np.linalg.norm(a, ord="fro")

    u_ref, s_ref, vt_ref = np.linalg.svd(a, full_matrices=False)
    optimal = u_ref[:, :8] @ np.diag(s_ref[:8]) @ vt_ref[:8]
    optimal_error = np.linalg.norm(a - optimal, ord="fro") / np.linalg.norm(a, ord="fro")
    assert randomized_error < max(1e-3, 3.0 * optimal_error)


def test_sketch_ols_function_tracks_full_ols_on_well_conditioned_design():
    rng = np.random.default_rng(42)
    n = 2500
    p = 12
    x = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = 0.7 + x @ beta + 0.1 * rng.normal(size=n)

    full = cm.OLS()
    full.fit(x, y)
    full_summary = full.summary(vcov="vanilla")
    sketched = cm.sketch_ols(x, y, sketch_size=8 * (p + 1), seed=99)

    assert abs(sketched["intercept"] - full_summary["intercept"]) < 0.05
    assert np.linalg.norm(sketched["coef"] - full_summary["coef"]) / np.linalg.norm(full_summary["coef"]) < 0.05


def test_ols_fit_sketch_predicts_like_full_ols():
    rng = np.random.default_rng(7)
    n = 1800
    p = 8
    x = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = -0.25 + x @ beta + 0.15 * rng.normal(size=n)

    full = cm.OLS()
    full.fit(x, y)
    sketch = cm.OLS()
    sketch.fit_sketch(x, y, sketch_size=10 * (p + 1), seed=88)

    x_test = rng.normal(size=(300, p))
    full_pred = full.predict(x_test)
    sketch_pred = sketch.predict(x_test)
    rmse = np.sqrt(np.mean((full_pred - sketch_pred) ** 2))
    assert rmse < 0.08


def test_sketch_ols_rejects_undersized_sketch():
    x = np.ones((10, 3))
    y = np.ones(10)
    with pytest.raises(ValueError, match="sketch_size"):
        cm.sketch_ols(x, y, sketch_size=2)
