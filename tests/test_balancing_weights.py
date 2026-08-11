import numpy as np

import crabbymetrics as cm


def test_entropy_balancing_matches_target_mean_and_keeps_positive_weights():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(300, 3))
    target = x[:60].copy()

    model = cm.BalancingWeights(objective="entropy")
    model.fit(x, target)
    summary = model.summary()

    assert model.success
    np.testing.assert_allclose(summary["weight_sum"], 1.0, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(summary["weighted_mean"], summary["target_mean"], atol=1e-6, rtol=1e-6)
    assert np.all(np.asarray(summary["weights"]) > 0.0)


def test_entropy_balancing_supports_baseline_and_target_weights():
    rng = np.random.default_rng(456)
    x = rng.normal(size=(250, 4))
    target = x[20:80].copy()
    baseline_weights = 0.2 + rng.random(x.shape[0])
    target_weights = 0.5 + rng.random(target.shape[0])

    model = cm.BalancingWeights(objective="entropy")
    model.fit(
        x,
        target,
        baseline_weights=baseline_weights.tolist(),
        target_weights=target_weights.tolist(),
    )
    summary = model.summary()

    expected_target_mean = np.average(target, axis=0, weights=target_weights)
    np.testing.assert_allclose(summary["target_mean"], expected_target_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["weighted_mean"], expected_target_mean, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(summary["weight_sum"], 1.0, atol=1e-8, rtol=0.0)
    assert summary["effective_sample_size"] > 0.0


def test_quadratic_balancing_detects_infeasible_exact_balance_and_relaxes_with_l2():
    x = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]], dtype=float)
    target = np.ones((20, 1), dtype=float)

    exact = cm.BalancingWeights(objective="quadratic", max_weight=0.25, l2_norm=0.0)
    exact.fit(x, target)
    exact_summary = exact.summary()
    assert not exact.success
    assert exact_summary["l2_diff"] > 1e-3

    relaxed = cm.BalancingWeights(objective="quadratic", max_weight=0.25, l2_norm=0.30)
    relaxed.fit(x, target)
    relaxed_summary = relaxed.summary()
    assert relaxed.success
    assert relaxed_summary["l2_diff"] <= 0.30 + 1e-6
    np.testing.assert_allclose(relaxed_summary["weight_sum"], 1.0, atol=1e-8, rtol=0.0)
    assert np.all(np.asarray(relaxed_summary["weights"]) >= -1e-10)
    assert np.all(np.asarray(relaxed_summary["weights"]) <= 0.25 + 1e-8)


def test_cressie_read_entropy_limit_balances_with_lbfgs():
    rng = np.random.default_rng(789)
    x = rng.normal(size=(240, 3))
    target = x[:70] + np.array([0.05, -0.03, 0.02])
    baseline = np.full(x.shape[0], 1.0 / x.shape[0])

    model = cm.BalancingWeights(
        objective="cressie_read",
        solver="lbfgs",
        divergence_power=0.0,
        dual_ridge=1e-10,
        max_iterations=500,
        tolerance=1e-8,
    )
    model.fit(x, target, baseline_weights=baseline.tolist())
    summary = model.summary()

    assert model.success
    assert summary["solver"] == "lbfgs"
    assert summary["objective"] == "cressie_read"
    np.testing.assert_allclose(summary["divergence_power"], 0.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(summary["weight_sum"], 1.0, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(summary["weighted_mean"], summary["target_mean"], atol=1e-6, rtol=1e-6)
    assert np.all(np.asarray(summary["weights"]) > 0.0)


def test_cressie_read_power_one_matches_first_moments_and_reports_ridge():
    rng = np.random.default_rng(321)
    x = rng.normal(size=(220, 2))
    target = x[30:100] + np.array([0.04, -0.02])

    model = cm.BalancingWeights(
        objective="cressie_read",
        solver="auto",
        divergence_power=1.0,
        dual_ridge=1e-8,
        max_iterations=500,
        tolerance=1e-8,
    )
    model.fit(x, target)
    summary = model.summary()

    assert model.success
    assert summary["effective_sample_size"] > 1.0
    np.testing.assert_allclose(summary["weight_sum"], 1.0, atol=1e-8, rtol=0.0)
    np.testing.assert_allclose(summary["weighted_mean"], summary["target_mean"], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(summary["divergence_power"], 1.0, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(summary["dual_ridge"], 1e-8, rtol=0.0, atol=0.0)
def test_autoscaled_solver_convergence_is_separate_from_original_unit_balance():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(200, 2)) * np.array([1.0, 1e12])
    target = rng.normal(loc=[0.2, -0.15], size=(80, 2)) * np.array([1.0, 1e12])

    model = cm.BalancingWeights(
        objective="quadratic",
        autoscale=True,
        max_weight=0.05,
        tolerance=1e-8,
    )
    model.fit(source, target)
    summary = model.summary()

    assert model.success
    assert model.solver_converged
    assert summary["converged"] is True
    assert summary["scaled_residual_norm"] < 1e-8
    assert summary["original_balance_l2"] > 1e-6
    assert summary["original_balance_l2"] == summary["l2_diff"]
