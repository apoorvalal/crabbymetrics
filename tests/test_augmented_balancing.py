import numpy as np
import pytest

import crabbymetrics as cm


def make_convex_panel(seed=811):
    rng = np.random.default_rng(seed)
    n_control = 6
    n_treated = 2
    n_pre = 8
    n_post = 4
    n_periods = n_pre + n_post
    factors = np.vstack(
        [
            np.linspace(-1.0, 1.0, n_periods),
            np.sin(np.linspace(0.0, 2.0 * np.pi, n_periods)),
        ]
    )
    controls = rng.normal(size=(n_control, 2)) @ factors
    true_weights = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25])
    untreated_mean = true_weights @ controls
    perturbation = 0.08 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_periods))
    treated = np.vstack(
        [
            untreated_mean - 0.15 + perturbation,
            untreated_mean + 0.15 - perturbation,
        ]
    )
    effect = np.r_[np.zeros(n_pre), np.full(n_post, 1.2)]
    y = np.vstack([controls, treated + effect])
    w = np.zeros_like(y)
    w[n_control:, n_pre:] = 1.0
    return y, w, controls, n_pre


def test_outcome_only_uses_supplied_counterfactual_matrix():
    y, w, _, n_pre = make_convex_panel()
    outcome_model = y.copy()
    outcome_model[-2:, n_pre:] -= 1.2

    model = cm.AugmentedBalancing(balance="none")
    model.fit(y, w, outcome_model)
    summary = model.summary()

    np.testing.assert_allclose(summary["att"], 1.2, atol=1e-12)
    np.testing.assert_allclose(model.predict()[-2:], outcome_model[-2:], atol=1e-12)
    assert np.asarray(summary["unit_weights"]).shape == (0, y.shape[0])


def test_unit_balancing_recovers_convex_counterfactual():
    y, w, _, _ = make_convex_panel()
    model = cm.AugmentedBalancing(
        balance="unit",
        zeta_omega=1e-10,
        max_iterations=3000,
    )
    model.fit(y, w)
    summary = model.summary()

    np.testing.assert_allclose(summary["att"], 1.2, atol=1e-5)
    control_units = np.asarray(summary["control_units"])
    weights = np.asarray(summary["unit_weights"])[0, control_units]
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)
    assert np.all(weights >= 0.0)


def test_double_balancing_matches_stored_weight_formula():
    y, w, _, n_pre = make_convex_panel(seed=812)
    outcome_model = 0.15 * np.add.outer(np.arange(y.shape[0]), np.arange(y.shape[1]))
    model = cm.AugmentedBalancing(
        balance="double",
        unit_target="cohort",
        time_target="all",
        balance_on="raw",
        zeta_omega=0.01,
        zeta_lambda=0.01,
        max_iterations=3000,
    )
    model.fit(y, w, outcome_model)
    summary = model.summary()

    controls = np.asarray(summary["control_units"])
    treated = np.asarray(summary["treated_units"])
    omega = np.asarray(summary["unit_weights"])[0, controls]
    lam = np.asarray(summary["time_weights"])[0, :n_pre]
    residual = y - outcome_model
    correction = omega @ residual[np.ix_(controls, np.arange(n_pre))] @ lam
    manual = np.empty((treated.size, y.shape[1]))
    for row, unit in enumerate(treated):
        manual[row] = (
            outcome_model[unit]
            + omega @ residual[controls]
            + residual[unit, :n_pre] @ lam
            - correction
        )

    np.testing.assert_allclose(
        model.predict()[treated, n_pre:], manual[:, n_pre:], atol=1e-10
    )
    np.testing.assert_allclose(lam.sum(), 1.0, atol=1e-10)


def test_oracle_outcome_model_makes_augmentation_exact():
    y, w, _, n_pre = make_convex_panel(seed=813)
    outcome_model = y.copy()
    outcome_model[-2:, n_pre:] -= 1.2
    model = cm.AugmentedBalancing(
        balance="double",
        balance_on="residual",
        zeta_omega=0.1,
        zeta_lambda=0.1,
    )
    model.fit(y, w, outcome_model)

    np.testing.assert_allclose(model.summary()["att"], 1.2, atol=1e-12)
    np.testing.assert_allclose(model.summary()["pre_rmse"], 0.0, atol=1e-12)


def test_individual_target_stores_one_weight_vector_per_treated_unit():
    y, w, _, _ = make_convex_panel(seed=814)
    model = cm.AugmentedBalancing(
        balance="unit",
        unit_target="individual",
        balance_on="residual",
        zeta_omega=0.01,
    )
    model.fit(y, w, np.zeros_like(y))
    summary = model.summary()

    weights = np.asarray(summary["unit_weights"])
    target_units = np.asarray(summary["target_units"])
    assert weights.shape == (2, y.shape[0])
    np.testing.assert_array_equal(target_units, summary["treated_units"])
    assert not np.allclose(weights[0], weights[1])


@pytest.mark.parametrize("balance", ["unit", "time"])
def test_one_dimension_balancing_uses_uniform_weights_for_other_dimension(balance):
    y, w, _, n_pre = make_convex_panel(seed=816)
    model = cm.AugmentedBalancing(
        balance=balance,
        zeta_omega=0.02,
        zeta_lambda=0.02,
        max_iterations=3000,
    )
    model.fit(y, w)
    summary = model.summary()

    controls = np.asarray(summary["control_units"])
    treated = np.asarray(summary["treated_units"])
    omega = np.asarray(summary["unit_weights"])[0, controls]
    lam = np.asarray(summary["time_weights"])[0, :n_pre]
    if balance == "unit":
        np.testing.assert_allclose(lam, np.full(n_pre, 1.0 / n_pre))
    else:
        np.testing.assert_allclose(omega, np.full(controls.size, 1.0 / controls.size))

    correction = omega @ y[np.ix_(controls, np.arange(n_pre))] @ lam
    manual = np.empty((treated.size, y.shape[1] - n_pre))
    for row, unit in enumerate(treated):
        manual[row] = (
            omega @ y[controls, n_pre:]
            + y[unit, :n_pre] @ lam
            - correction
        )
    np.testing.assert_allclose(model.predict()[treated, n_pre:], manual, atol=1e-10)


def test_period_time_target_stores_one_weight_vector_per_post_period():
    y, w, _, n_pre = make_convex_panel(seed=817)
    model = cm.AugmentedBalancing(
        balance="double",
        unit_target="individual",
        time_target="period",
        zeta_omega=0.01,
        zeta_lambda=0.01,
        max_iterations=3000,
    )
    model.fit(y, w)
    summary = model.summary()

    time_weights = np.asarray(summary["time_weights"])
    np.testing.assert_array_equal(
        summary["time_target_periods"], np.arange(n_pre, y.shape[1])
    )
    assert time_weights.shape == (y.shape[1] - n_pre, y.shape[1])
    np.testing.assert_allclose(time_weights[:, :n_pre].sum(axis=1), 1.0, atol=1e-10)


def test_augmented_balancing_rejects_invalid_options_and_inputs():
    y, w, _, _ = make_convex_panel(seed=815)
    with pytest.raises(ValueError, match="balance must be"):
        cm.AugmentedBalancing(balance="triple")
    with pytest.raises(ValueError, match="unit_target must be"):
        cm.AugmentedBalancing(unit_target="unit")
    with pytest.raises(ValueError, match="time_target must be"):
        cm.AugmentedBalancing(time_target="cohort")
    with pytest.raises(ValueError, match="balance_on must be"):
        cm.AugmentedBalancing(balance_on="fitted")
    with pytest.raises(ValueError, match="same shape"):
        cm.AugmentedBalancing().fit(y, w, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        cm.AugmentedBalancing(max_iterations=0)
