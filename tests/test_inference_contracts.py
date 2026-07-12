import numpy as np
import pytest
import statsmodels.api as sm

import crabbymetrics as cm


def test_logit_unpenalized_inference_matches_statsmodels():
    rng = np.random.default_rng(1201)
    x = rng.normal(size=(900, 2))
    design = sm.add_constant(x)
    probability = 1.0 / (1.0 + np.exp(-(design @ np.array([-0.2, 0.6, -0.35]))))
    y = rng.binomial(1, probability).astype(np.int32)

    model = cm.Logit(max_iterations=500, gradient_tolerance=1e-10)
    model.fit(x, y)
    summary = model.summary()
    reference = sm.Logit(y, design).fit(disp=False)

    assert summary["inference_available"] is True
    np.testing.assert_allclose(
        np.r_[summary["intercept"], summary["coef"]],
        reference.params,
        atol=2e-5,
        rtol=2e-5,
    )
    np.testing.assert_allclose(summary["vcov"], reference.cov_params(), atol=2e-6, rtol=2e-5)
    assert summary["converged"] is True
    assert summary["iterations"] > 0
    assert summary["termination_reason"] == "Solver converged"
    assert np.isfinite(summary["objective"])


@pytest.mark.parametrize("estimator", [cm.Logit, cm.MultinomialLogit])
def test_logistic_estimators_reject_iteration_budget_exhaustion(estimator):
    rng = np.random.default_rng(1210)
    x = rng.normal(size=(300, 3))
    if estimator is cm.Logit:
        probability = 1.0 / (1.0 + np.exp(-(x @ np.array([1.0, -0.7, 0.4]))))
        y = rng.binomial(1, probability).astype(np.int32)
    else:
        logits = np.column_stack([x[:, 0], x[:, 1], -x[:, 0] - x[:, 1]])
        y = (logits + rng.normal(size=logits.shape)).argmax(axis=1).astype(np.int32)

    model = estimator(max_iterations=1, gradient_tolerance=1e-14)
    with pytest.raises(ValueError, match="Maximum number of iterations reached"):
        model.fit(x, y)
    with pytest.raises(ValueError, match="not fitted"):
        model.summary()


def test_penalized_likelihood_models_do_not_report_unpenalized_inference():
    rng = np.random.default_rng(1202)
    x = rng.normal(size=(500, 2))

    binary_probability = 1.0 / (1.0 + np.exp(-(0.1 + x @ np.array([0.4, -0.2]))))
    binary_y = rng.binomial(1, binary_probability).astype(np.int32)
    logit = cm.Logit(alpha=0.5, max_iterations=500)
    logit.fit(x, binary_y)
    logit_summary = logit.summary()
    assert logit_summary["inference_available"] is False
    assert logit_summary["vcov"] is None
    assert logit_summary["coef_se"] is None
    with pytest.raises(ValueError, match="only available for unpenalized"):
        logit.wald_test(np.eye(3))

    count_y = rng.poisson(np.exp(0.2 + x @ np.array([0.25, -0.15]))).astype(float)
    poisson = cm.Poisson(alpha=0.5, max_iterations=300, tolerance=1e-8)
    poisson.fit(x, count_y)
    poisson_summary = poisson.summary()
    assert poisson_summary["inference_available"] is False
    assert poisson_summary["vcov"] is None
    assert poisson_summary["coef_se"] is None
    assert poisson_summary["converged"] is True
    assert poisson_summary["iterations"] > 0
    assert np.isfinite(poisson_summary["objective"])
    with pytest.raises(ValueError, match="only available for unpenalized"):
        poisson.wald_test(np.eye(3))


def test_multinomial_reference_contrasts_and_covariance_match_statsmodels():
    rng = np.random.default_rng(1203)
    x = rng.normal(size=(1400, 2))
    design = sm.add_constant(x)
    logits = design @ np.array(
        [
            [0.25, -0.15, 0.0],
            [0.55, -0.35, 0.0],
            [-0.20, 0.45, 0.0],
        ]
    )
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in probabilities], dtype=np.int32)

    model = cm.MultinomialLogit(max_iterations=800, gradient_tolerance=1e-10)
    model.fit(x, y)
    summary = model.summary()
    reference = sm.MNLogit(y, design).fit(disp=False)

    identity = np.eye(design.shape[1])
    transform = np.block(
        [
            [np.zeros_like(identity), -identity],
            [identity, -identity],
        ]
    )
    expected_coef = np.vstack(
        [
            -reference.params[:, 1],
            reference.params[:, 0] - reference.params[:, 1],
        ]
    )
    expected_vcov = transform @ np.asarray(reference.cov_params()) @ transform.T

    assert summary["reference_class"] == 2
    np.testing.assert_array_equal(summary["class_labels"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_allclose(summary["coef"], expected_coef, atol=3e-5, rtol=3e-5)
    np.testing.assert_allclose(summary["vcov"], expected_vcov, atol=3e-6, rtol=3e-5)
    np.testing.assert_allclose(summary["se"], np.sqrt(np.diag(expected_vcov)).reshape(2, 3))
    assert summary["converged"] is True
    assert summary["iterations"] > 0
    assert summary["termination_reason"] == "Solver converged"
    assert np.isfinite(summary["objective"])


def test_mestimator_sandwich_covariance_matches_statsmodels_hc0():
    rng = np.random.default_rng(1204)
    design = sm.add_constant(rng.normal(size=(700, 2)))
    scale = np.exp(0.2 * design[:, 1])
    y = design @ np.array([0.3, 0.55, -0.25]) + rng.normal(scale=scale)
    data = {"x": design, "y": y, "n": y.size}

    def objective(theta, values):
        indices = values.get("indices", np.arange(values["n"]))
        x_sample = values["x"][indices]
        residual = values["y"][indices] - x_sample @ theta
        return 0.5 * np.mean(residual**2), -(x_sample.T @ residual) / indices.size

    def scores(theta, values):
        indices = values.get("indices", np.arange(values["n"]))
        x_sample = values["x"][indices]
        residual = values["y"][indices] - x_sample @ theta
        return x_sample * residual[:, None]

    model = cm.MEstimator(
        objective,
        scores,
        max_iterations=500,
        tolerance=1e-10,
        derivative_step=1e-6,
    )
    model.fit(data, np.zeros(design.shape[1]))
    summary = model.summary()
    reference = sm.OLS(y, design).fit(cov_type="HC0")

    assert summary["converged"] is True
    assert summary["iterations"] < 500
    assert summary["termination_reason"] == "Solver converged"
    assert np.isfinite(summary["objective"])
    np.testing.assert_allclose(summary["coef"], reference.params, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(summary["vcov"], reference.cov_params(), atol=2e-8, rtol=2e-6)


def test_predictive_regularized_models_mark_analytic_inference_unavailable():
    rng = np.random.default_rng(1205)
    x = rng.normal(size=(350, 3))
    y = 0.2 + x @ np.array([0.5, 0.0, -0.25]) + rng.normal(scale=0.4, size=x.shape[0])

    elastic_net = cm.ElasticNet(penalty=0.1, l1_ratio=0.6)
    elastic_net.fit(x, y)
    elastic_summary = elastic_net.summary()
    assert elastic_summary["inference_available"] is False
    assert elastic_summary["intercept_se"] is None
    assert elastic_summary["coef_se"] is None
    assert elastic_summary["converged"] is True
    assert elastic_summary["iterations"] > 0
    assert elastic_summary["duality_gap"] <= elastic_summary["duality_gap_tolerance"]
    assert np.isfinite(elastic_summary["objective"])

    binary_y = (y > np.median(y)).astype(np.int32)
    ftrl = cm.FTRL()
    ftrl.fit(x, binary_y)
    ftrl_summary = ftrl.summary()
    assert ftrl_summary["inference_available"] is False
    assert ftrl_summary["coef_se"] is None


def test_elastic_net_rejects_unsatisfied_duality_gap():
    rng = np.random.default_rng(1206)
    x = rng.normal(size=(400, 8))
    y = x @ np.array([1.0, -0.7, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0])
    y += rng.normal(scale=0.3, size=x.shape[0])

    model = cm.ElasticNet(
        penalty=0.05,
        l1_ratio=0.6,
        tolerance=1e-6,
        max_iterations=1,
    )
    with pytest.raises(ValueError, match="duality gap"):
        model.fit(x, y)
    with pytest.raises(ValueError, match="not fitted"):
        model.summary()
