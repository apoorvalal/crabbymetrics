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


def test_randomized_qr_approximates_low_rank_matrix():
    a = make_low_rank_matrix(seed=11)
    result = cm.randomized_qr(a, rank=8, oversamples=8, power_iter=2, seed=123)
    q = result["q"]
    r = result["r"]
    assert q.shape == (a.shape[0], 8)
    assert r.shape == (8, a.shape[1])
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)
    approx = q @ r
    randomized_error = np.linalg.norm(a - approx, ord="fro") / np.linalg.norm(a, ord="fro")
    assert randomized_error < 1e-3


def test_qr_solve_tracks_full_least_squares_on_low_rank_design():
    rng = np.random.default_rng(12)
    n = 900
    p = 12
    latent = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, p))
    x = latent @ loadings + 0.01 * rng.normal(size=(n, p))
    beta = rng.normal(size=(p, 2))
    y = x @ beta + 0.01 * rng.normal(size=(n, 2))
    coef = cm.qr_solve(x, y, rank=8, oversamples=4, power_iter=2, seed=45)
    ref, *_ = np.linalg.lstsq(x, y, rcond=None)
    assert np.linalg.norm(coef - ref) / np.linalg.norm(ref) < 0.03


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


def test_twosls_fit_sketch_tracks_full_twosls():
    rng = np.random.default_rng(123)
    n = 4000
    z = rng.normal(size=(n, 3))
    x_exog = rng.normal(size=(n, 2))
    v = rng.normal(size=n)
    x_endog = (0.8 * z[:, [0]] + 0.4 * z[:, [1]] + 0.3 * x_exog[:, [0]] + v[:, None])
    eps = 0.6 * v + rng.normal(scale=0.2, size=n)
    y = 1.0 + 2.0 * x_endog[:, 0] - 0.5 * x_exog[:, 0] + 0.25 * x_exog[:, 1] + eps

    full = cm.TwoSLS()
    full.fit(x_endog, x_exog, z, y)
    sketch = cm.TwoSLS()
    sketch.fit_sketch(x_endog, x_exog, z, y, sketch_size=250, seed=321)

    x_pred = np.column_stack([x_endog, x_exog])[:500]
    rmse = np.sqrt(np.mean((full.predict(x_pred) - sketch.predict(x_pred)) ** 2))
    assert rmse < 0.2


def test_sketch_ols_rejects_undersized_sketch():
    x = np.ones((10, 3))
    y = np.ones(10)
    with pytest.raises(ValueError, match="sketch_size"):
        cm.sketch_ols(x, y, sketch_size=2)
