import numpy as np

import crabbymetrics as cm


def hc1_se(x, y):
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    xtx_inv = np.linalg.inv(design.T @ design)
    meat = design.T @ ((resid[:, None] ** 2) * design)
    scale = x.shape[0] / (x.shape[0] - design.shape[1])
    cov = scale * xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def test_ols_matches_closed_form_hc1_and_predict_round_trip():
    rng = np.random.default_rng(2026)
    n = 800
    intercept_true = 1.75
    beta_true = np.array([0.8, -1.2, 0.35])

    x = rng.normal(size=(n, beta_true.size))
    y = intercept_true + x @ beta_true + rng.normal(scale=0.6, size=n)

    model = cm.OLS()
    model.fit(x, y)
    summary = model.summary()

    beta_hat, se_hat = hc1_se(x, y)
    pred_all = model.predict(x)
    pred_subset = model.predict(x[:57])

    np.testing.assert_allclose(summary["intercept"], beta_hat[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], beta_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["intercept_se"], se_hat[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef_se"], se_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pred_all, beta_hat[0] + x @ beta_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        pred_subset,
        beta_hat[0] + x[:57] @ beta_hat[1:],
        atol=1e-8,
        rtol=1e-8,
    )
    assert pred_all.shape == (n,)
    assert pred_subset.shape == (57,)
    assert model.bootstrap(5, seed=11).shape == (5, x.shape[1] + 1)


def test_fixed_effects_ols_is_invariant_to_fixed_effect_relabeling():
    rng = np.random.default_rng(31415)
    n = 600
    beta_true = np.array([1.1, -0.45])

    worker = rng.integers(0, 30, size=n, dtype=np.uint32)
    firm = rng.integers(0, 20, size=n, dtype=np.uint32)
    fe = np.column_stack([worker, firm]).astype(np.uint32)
    fe_relabelled = np.column_stack([worker + 100, firm + 500]).astype(np.uint32)

    x = rng.normal(size=(n, beta_true.size))
    y = (
        x @ beta_true
        + rng.normal(scale=0.8, size=30)[worker]
        + rng.normal(scale=0.5, size=20)[firm]
        + rng.normal(scale=0.1, size=n)
    )

    model_a = cm.FixedEffectsOLS()
    model_a.fit(x, fe, y)

    model_b = cm.FixedEffectsOLS()
    model_b.fit(x, fe_relabelled, y)

    np.testing.assert_allclose(
        model_a.summary()["coef"],
        model_b.summary()["coef"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_poisson_predict_round_trip_and_satisfies_score_conditions():
    rng = np.random.default_rng(9090)
    n = 900
    intercept_true = 0.2
    beta_true = np.array([0.3, -0.25, 0.15])

    x = rng.normal(size=(n, beta_true.size))
    mu_true = np.exp(intercept_true + x @ beta_true)
    y = rng.poisson(mu_true).astype(float)

    model = cm.Poisson(max_iterations=200, tolerance=1e-8)
    model.fit(x, y)
    summary = model.summary()

    pred_all = model.predict(x)
    pred_subset = model.predict(x[:41])
    residual = pred_all - y

    assert summary["coef"].shape == (x.shape[1],)
    assert summary["coef_se"].shape == (x.shape[1],)
    assert np.isfinite(summary["intercept"])
    assert np.isfinite(summary["intercept_se"])
    assert np.all(pred_all > 0.0)
    np.testing.assert_allclose(pred_subset, pred_all[:41], atol=1e-10, rtol=1e-10)
    assert model.bootstrap(4, seed=19).shape == (4, x.shape[1] + 1)

    assert abs(residual.sum()) < 1e-3
    np.testing.assert_allclose(
        x.T @ residual,
        np.zeros(x.shape[1]),
        atol=1e-3,
        rtol=0.0,
    )
