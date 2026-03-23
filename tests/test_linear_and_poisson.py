import numpy as np
import pytest

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


def vanilla_ols_se(x, y):
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    xtx_inv = np.linalg.inv(design.T @ design)
    sigma2 = (resid @ resid) / (design.shape[0] - design.shape[1])
    cov = sigma2 * xtx_inv
    return beta, np.sqrt(np.diag(cov))


def poisson_covariances(x, y, intercept, coef):
    design = np.column_stack([np.ones(x.shape[0]), x])
    mu = np.exp(intercept + x @ coef)
    fisher = np.linalg.inv(design.T @ (mu[:, None] * design))
    scores = design * (y - mu)[:, None]
    qmle = fisher @ (scores.T @ scores) @ fisher
    return fisher, qmle


def twosls_closed_form(x_endog, x_exog, z, y):
    if x_exog.shape[1] > 0:
        x_rhs = np.column_stack([x_endog, x_exog])
        z_rhs = np.column_stack([x_exog, z])
    else:
        x_rhs = x_endog
        z_rhs = z

    x_design = np.column_stack([np.ones(x_rhs.shape[0]), x_rhs])
    z_design = np.column_stack([np.ones(z_rhs.shape[0]), z_rhs])
    ztz_inv = np.linalg.inv(z_design.T @ z_design)
    beta = np.linalg.solve(
        x_design.T @ z_design @ ztz_inv @ z_design.T @ x_design,
        x_design.T @ z_design @ ztz_inv @ z_design.T @ y,
    )
    return beta[0], beta[1:]


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


def test_ols_summary_supports_vanilla_and_hc1_vcov():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(700, 2))
    y = 0.3 + x @ np.array([1.1, -0.4]) + (0.5 + x[:, 0] ** 2) * rng.normal(size=700)

    model = cm.OLS()
    model.fit(x, y)

    vanilla = model.summary(vcov="vanilla")
    hc1 = model.summary(vcov="hc1")
    beta_vanilla, se_vanilla = vanilla_ols_se(x, y)
    beta_hc1, se_hc1 = hc1_se(x, y)

    np.testing.assert_allclose(vanilla["intercept"], beta_vanilla[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["coef"], beta_vanilla[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["intercept_se"], se_vanilla[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["coef_se"], se_vanilla[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["intercept"], beta_hc1[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["coef"], beta_hc1[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["intercept_se"], se_hc1[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["coef_se"], se_hc1[1:], atol=1e-8, rtol=1e-8)
    assert vanilla["vcov_type"] == "vanilla"
    assert hc1["vcov_type"] == "hc1"


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


def test_poisson_summary_supports_vanilla_and_qmle_vcov():
    rng = np.random.default_rng(9091)
    n = 800
    intercept_true = -0.1
    beta_true = np.array([0.45, -0.2])

    x = rng.normal(size=(n, beta_true.size))
    mu = np.exp(intercept_true + x @ beta_true)
    mixing = rng.gamma(shape=2.0, scale=0.5, size=n)
    y = rng.poisson(mu * mixing).astype(float)

    model = cm.Poisson(alpha=0.0, max_iterations=200, tolerance=1e-8)
    model.fit(x, y)

    vanilla = model.summary(vcov="vanilla")
    qmle = model.summary(vcov="sandwich")
    fisher, sandwich = poisson_covariances(x, y, vanilla["intercept"], vanilla["coef"])

    np.testing.assert_allclose(vanilla["intercept_se"], np.sqrt(fisher[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["coef_se"], np.sqrt(np.diag(fisher))[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(qmle["intercept_se"], np.sqrt(sandwich[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(qmle["coef_se"], np.sqrt(np.diag(sandwich))[1:], atol=1e-8, rtol=1e-8)
    assert vanilla["vcov_type"] == "vanilla"
    assert qmle["vcov_type"] == "sandwich"


def test_twosls_matches_closed_form_with_multiple_endogenous_regressors():
    rng = np.random.default_rng(4242)
    n = 1200
    intercept_true = 0.4
    beta_endog = np.array([1.15, -0.9])
    beta_exog = np.array([0.55])

    z = rng.normal(size=(n, 2))
    x_exog = rng.normal(size=(n, 1))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    pi = np.array([[0.9, 0.2], [-0.35, 0.8]])

    x_endog = z @ pi + 0.3 * x_exog + v
    u = 0.6 * v[:, 0] - 0.45 * v[:, 1] + 0.25 * eps
    y = intercept_true + x_endog @ beta_endog + x_exog[:, 0] * beta_exog[0] + u

    model = cm.TwoSLS()
    model.fit(x_endog, x_exog, z, y)
    summary = model.summary()
    intercept_hat, coef_hat = twosls_closed_form(x_endog, x_exog, z, y)

    pred = model.predict(np.column_stack([x_endog, x_exog]))

    np.testing.assert_allclose(summary["intercept"], intercept_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], coef_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pred, intercept_hat + np.column_stack([x_endog, x_exog]) @ coef_hat)
    assert summary["coef"].shape == (x_endog.shape[1] + x_exog.shape[1],)
    assert summary["coef_se"].shape == (x_endog.shape[1] + x_exog.shape[1],)
    assert np.isfinite(summary["intercept_se"])
    assert model.bootstrap(4, seed=7).shape == (4, x_endog.shape[1] + x_exog.shape[1] + 1)


def test_twosls_overidentified_matches_closed_form_formula():
    rng = np.random.default_rng(1717)
    n = 1400

    z = rng.normal(size=(n, 4))
    x_exog = rng.normal(size=(n, 2))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    pi = np.array(
        [
            [1.0, 0.2],
            [0.4, -0.3],
            [-0.25, 0.9],
            [0.3, 0.15],
        ]
    )
    beta = np.array([1.0, -0.7, 0.45, -0.2])

    x_endog = z @ pi + x_exog @ np.array([[0.25, -0.1], [0.15, 0.2]]) + v
    u = 0.5 * v[:, 0] - 0.35 * v[:, 1] + 0.2 * eps
    y = -0.3 + np.column_stack([x_endog, x_exog]) @ beta + u

    model = cm.TwoSLS()
    model.fit(x_endog, x_exog, z, y)
    summary = model.summary()
    intercept_hat, coef_hat = twosls_closed_form(x_endog, x_exog, z, y)

    np.testing.assert_allclose(summary["intercept"], intercept_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], coef_hat, atol=1e-8, rtol=1e-8)
    assert np.all(np.isfinite(summary["coef_se"]))
    assert summary["coef"].shape == (beta.size,)


def test_twosls_rejects_underidentified_design():
    rng = np.random.default_rng(33)
    n = 100
    x_endog = rng.normal(size=(n, 2))
    x_exog = np.empty((n, 0))
    z = rng.normal(size=(n, 1))
    y = rng.normal(size=n)

    model = cm.TwoSLS()
    with pytest.raises(ValueError, match="need at least as many excluded instruments"):
        model.fit(x_endog, x_exog, z, y)


def test_synthetic_control_recovers_convex_weights_and_post_path():
    rng = np.random.default_rng(777)
    true_weights = np.array([0.55, 0.3, 0.15])

    donors_pre = rng.normal(size=(60, true_weights.size))
    treated_pre = donors_pre @ true_weights
    donors_post = rng.normal(size=(25, true_weights.size))
    treated_post = donors_post @ true_weights

    model = cm.SyntheticControl(max_iterations=400)
    model.fit(donors_pre, treated_pre)
    summary = model.summary()
    weights = np.asarray(summary["weights"])
    pred_post = model.predict(donors_post)

    np.testing.assert_allclose(weights, true_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pred_post, treated_post, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8, rtol=0.0)
    assert np.all(weights >= 0.0)
    assert summary["pre_rmse"] < 1e-6
    assert model.bootstrap(4, seed=5).shape == (4, donors_pre.shape[1])
