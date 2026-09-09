"""Parity tests for the Chronos marginal-policy-effect CBPS estimator.

The reference functions below are a minimal transcription of the MIT-licensed
implementations released with Qiu et al. (2026), commit 06c29f4:

* ab_validation/run_simulation.py::_cbps_obj/_cbps_weights/_policy_gradient
* switchback_validation/run_simulation.py::_cbps_obj/_cbps_weights/_policy_gradient

https://github.com/chenyuqiu/ltv_of_reliability
"""

import numpy as np
import pytest

import crabbymetrics as cm

scipy_optimize = pytest.importorskip("scipy.optimize")


def _canonical_standardize(x):
    z_raw = np.column_stack([np.ones(len(x)), x])
    mean = z_raw.mean(axis=0)
    mean[0] = 0.0
    scale = z_raw.std(axis=0)
    scale[0] = 1.0
    scale[scale == 0.0] = 1.0
    return (z_raw - mean) / scale


def _canonical_cbps_weights(z, arm, policy_derivative):
    n = len(arm)

    def objective(theta):
        linear = z @ theta
        exp_neg = np.exp(np.clip(-linear, -50.0, 50.0))
        value = np.mean(
            policy_derivative * (arm * exp_neg + (1.0 - arm) * linear)
        )
        gradient = (
            z.T
            @ (policy_derivative * ((1.0 - arm) - arm * exp_neg))
            / n
        )
        return value, gradient

    result = scipy_optimize.minimize(
        objective,
        np.zeros(z.shape[1]),
        method="BFGS",
        jac=True,
        options={"maxiter": 500, "gtol": 1e-9},
    )
    assert result.success, result.message
    return result.x, 1.0 + np.exp(np.clip(-(z @ result.x), -50.0, 50.0))


def _sample(seed=260611526, n=1_400, p=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    x[:, 2] = 0.5 * x[:, 0] + rng.normal(scale=0.8, size=n)
    x[:, 5] = np.sin(x[:, 1])
    index = -0.25 + x @ np.array([0.42, -0.33, 0.15, 0.21, -0.12, 0.19])
    propensity = 1.0 / (1.0 + np.exp(-index))
    treatment = rng.binomial(1, propensity).astype(np.int32)
    outcome = 1.2 + 0.5 * x[:, 0] - 0.3 * x[:, 1] - 0.4 * treatment
    outcome += rng.normal(scale=0.35, size=n)
    baseline_spend = np.exp(0.1 + 0.15 * x[:, 0])
    return x, treatment, outcome, baseline_spend


def test_mpe_cbps_matches_released_ab_and_switchback_cbps_weights():
    x, treatment, _, _ = _sample()
    policy_derivative = np.full(len(x), 0.01)
    z = _canonical_standardize(x)
    beta_zero, weights_zero = _canonical_cbps_weights(
        z, (treatment == 0).astype(float), policy_derivative
    )
    beta_one, weights_one = _canonical_cbps_weights(
        z, (treatment == 1).astype(float), policy_derivative
    )

    model = cm.MPE_CBPS(max_iterations=500, tolerance=1e-9)
    model.fit(
        x,
        treatment,
        policy_derivative=policy_derivative.tolist(),
    )
    summary = model.summary()

    assert model.success
    np.testing.assert_allclose(summary["beta_zero"], beta_zero, atol=2e-7, rtol=2e-7)
    np.testing.assert_allclose(summary["beta_one"], beta_one, atol=2e-7, rtol=2e-7)
    # The two optimizers stop on different criteria (analytic Newton score versus
    # SciPy's inverse-Hessian BFGS score). Coefficients agree more tightly; the
    # exponential link magnifies their last few ulps in the largest weights.
    np.testing.assert_allclose(summary["weights_zero"], weights_zero, atol=5e-6, rtol=1e-6)
    np.testing.assert_allclose(summary["weights_one"], weights_one, atol=5e-6, rtol=1e-6)


def test_mpe_cbps_policy_gradient_matches_released_formula():
    x, treatment, outcome, baseline_spend = _sample(seed=91)
    policy_derivative = np.full(len(x), 0.01)
    z = _canonical_standardize(x)
    _, weights_zero = _canonical_cbps_weights(
        z, (treatment == 0).astype(float), policy_derivative
    )
    _, weights_one = _canonical_cbps_weights(
        z, (treatment == 1).astype(float), policy_derivative
    )
    expected = (
        (
            (treatment == 1) * weights_one * outcome
            - (treatment == 0) * weights_zero * outcome
        ).sum()
        * 0.01
        / baseline_spend.sum()
    )

    model = cm.MPE_CBPS(tolerance=1e-9)
    model.fit(x, treatment, policy_derivative=policy_derivative.tolist())
    actual = model.estimate(outcome, denominator=float(baseline_spend.sum()))

    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-7)


def test_mpe_cbps_hits_both_full_sample_balance_targets():
    x, treatment, outcome, _ = _sample(seed=13, n=900, p=6)
    model = cm.MPE_CBPS(tolerance=1e-9)
    model.fit(x, treatment)
    summary = model.summary()

    assert summary["success"] is True
    np.testing.assert_allclose(summary["weight_sum_zero"], len(x), atol=1e-6, rtol=1e-8)
    np.testing.assert_allclose(summary["weight_sum_one"], len(x), atol=1e-6, rtol=1e-8)
    np.testing.assert_allclose(
        summary["weighted_mean_zero"], summary["target_mean"], atol=1e-8, rtol=1e-8
    )
    np.testing.assert_allclose(
        summary["weighted_mean_one"], summary["target_mean"], atol=1e-8, rtol=1e-8
    )
    assert np.isfinite(model.estimate(outcome))


def test_mpe_cbps_supports_heterogeneous_positive_policy_derivatives():
    x, treatment, _, _ = _sample(seed=131, n=750, p=6)
    policy_derivative = np.exp(0.15 * x[:, 0] - 0.08 * x[:, 1])
    model = cm.MPE_CBPS(tolerance=1e-9)
    model.fit(x, treatment, policy_derivative=policy_derivative.tolist())
    summary = model.summary()

    expected_target = np.average(x, axis=0, weights=policy_derivative)
    np.testing.assert_allclose(summary["target_mean"], expected_target, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(summary["weighted_mean_zero"], expected_target, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["weighted_mean_one"], expected_target, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        summary["policy_weighted_mass_zero"],
        summary["target_policy_mass"],
        atol=1e-7,
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        summary["policy_weighted_mass_one"],
        summary["target_policy_mass"],
        atol=1e-7,
        rtol=1e-9,
    )


def test_mpe_cbps_validates_fit_and_estimate_inputs():
    x, treatment, outcome, _ = _sample(n=100)

    with pytest.raises(ValueError, match="both arms"):
        cm.MPE_CBPS().fit(x, np.ones(len(x), dtype=np.int32))

    with pytest.raises(ValueError, match="positive finite"):
        cm.MPE_CBPS().fit(
            x,
            treatment,
            policy_derivative=np.where(treatment == 1, 1.0, 0.0).tolist(),
        )

    model = cm.MPE_CBPS()
    model.fit(x, treatment)
    with pytest.raises(ValueError, match="length"):
        model.estimate(outcome[:-1])
    with pytest.raises(ValueError, match="denominator"):
        model.estimate(outcome, denominator=0.0)
    with pytest.raises(ValueError, match="arm"):
        model.get_weights(2)
