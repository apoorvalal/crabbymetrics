import numpy as np
import pytest

import crabbymetrics as cm


def test_regression_blip_recovers_contemporaneous_and_lagged_effects():
    rng = np.random.default_rng(1234)
    n_units, n_periods = 1_500, 7
    treatment = np.zeros((n_units, n_periods))
    outcome = np.zeros((n_units, n_periods))
    covariate = np.zeros((n_units, n_periods, 1))

    for time in range(n_periods):
        previous_treatment = treatment[:, time - 1] if time else 0.0
        covariate[:, time, 0] = 0.8 * previous_treatment + rng.normal(size=n_units)
        propensity = 1.0 / (
            1.0
            + np.exp(
                -(
                    -0.2
                    + 0.8 * covariate[:, time, 0]
                    + 0.2 * previous_treatment
                )
            )
        )
        treatment[:, time] = rng.binomial(1, propensity)
        outcome[:, time] = (
            2.0 * treatment[:, time]
            + 0.5 * previous_treatment
            + 1.5 * covariate[:, time, 0]
            + rng.normal(scale=0.7, size=n_units)
        )

    model = cm.RegressionBlip(max_lag=1)
    model.fit(outcome, treatment, covariate)
    result = model.summary()

    np.testing.assert_allclose(result["coef"], np.array([2.0, 1.7]), atol=0.10)
    assert result["stage_se_scope"] == "conditional_on_earlier_blip_estimates"
    assert np.all(result["stage_se"] > 0)
    blipped = model.blip_down(outcome, treatment)
    assert blipped.shape == outcome.shape


def test_parallel_trends_snmm_recovers_initiation_response_with_level_confounding():
    rng = np.random.default_rng(91)
    n_units, n_treatment_periods = 4_000, 6
    horizon_effect = np.array([1.0, 1.6, 2.1])
    history = rng.normal(size=(n_units, n_treatment_periods, 1))
    level_confounder = rng.normal(size=n_units)

    treatment = np.zeros((n_units, n_treatment_periods))
    adoption_time = np.full(n_units, n_treatment_periods + 1, dtype=int)
    untreated = np.zeros((n_units, n_treatment_periods + 1))
    untreated[:, 0] = 2.0 * level_confounder + rng.normal(scale=0.4, size=n_units)

    for time in range(n_treatment_periods):
        untreated[:, time + 1] = (
            untreated[:, time]
            + 0.35 * history[:, time, 0]
            + rng.normal(scale=0.5, size=n_units)
        )
        at_risk = adoption_time > time
        propensity = 1.0 / (
            1.0
            + np.exp(
                -(
                    -1.2
                    + 0.55 * history[:, time, 0]
                    + 0.65 * level_confounder
                )
            )
        )
        newly_treated = at_risk & (rng.uniform(size=n_units) < propensity)
        adoption_time[newly_treated] = time
        treatment[adoption_time <= time, time] = 1.0

    outcome = untreated.copy()
    for unit in range(n_units):
        first = adoption_time[unit]
        if first <= n_treatment_periods - 1:
            for horizon, effect in enumerate(horizon_effect, start=1):
                outcome_time = first + horizon
                if outcome_time <= n_treatment_periods:
                    outcome[unit, outcome_time] += effect

    model = cm.ParallelTrendsSNMM(
        max_horizon=3,
        treatment_mode="initiation",
        n_folds=3,
        nuisance_penalty=1e-6,
        seed=7,
    )
    model.fit(outcome, treatment, history)
    result = model.summary()

    np.testing.assert_allclose(result["coef"], horizon_effect, atol=0.18)
    assert result["treatment_mode"] == "initiation"
    assert result["n_moment_rows"] > n_units
    assert result["max_abs_moment"] < 1e-8
    assert np.all(result["se"] > 0)


def test_dynamic_covariate_balance_recovers_treatment_path_contrast():
    rng = np.random.default_rng(45)
    n_units = 5_000
    baseline = rng.normal(size=n_units)
    first_propensity = 1.0 / (1.0 + np.exp(-(-0.1 + 0.7 * baseline)))
    first_treatment = rng.binomial(1, first_propensity)
    second_covariate = (
        0.5 * baseline + 0.4 * first_treatment + rng.normal(size=n_units)
    )
    intermediate_outcome = (
        0.3 * baseline + 0.8 * first_treatment + rng.normal(size=n_units)
    )
    second_propensity = 1.0 / (
        1.0
        + np.exp(
            -(
                -0.2
                + 0.4 * baseline
                + 0.4 * second_covariate
                + 0.3 * intermediate_outcome
                + 0.2 * first_treatment
            )
        )
    )
    second_treatment = rng.binomial(1, second_propensity)
    final_outcome = (
        1.0
        + 0.5 * baseline
        + 0.6 * second_covariate
        + 0.5 * intermediate_outcome
        + first_treatment
        + 1.5 * second_treatment
        + rng.normal(scale=0.7, size=n_units)
    )

    treatment = np.column_stack([first_treatment, second_treatment]).astype(float)
    history = np.zeros((n_units, 2, 4))
    history[:, 0, 0] = baseline
    history[:, 1, :] = np.column_stack(
        [baseline, second_covariate, intermediate_outcome, first_treatment]
    )

    estimates = []
    for path in ([0.0, 0.0], [1.0, 1.0]):
        model = cm.DynamicCovariateBalance()
        model.fit(final_outcome, treatment, history, path)
        result = model.summary()
        weights = model.get_weights()
        estimates.append(model.potential_outcome)

        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)
        assert np.all(result["effective_sample_size"] > 0)
        assert np.max(result["max_abs_balance"]) < 1e-5
        for time in range(treatment.shape[1]):
            matches = np.all(
                treatment[:, : time + 1] == np.asarray(path[: time + 1]), axis=1
            )
            assert np.all(weights[~matches, time] == 0.0)

    # The path contrast includes direct effects 1.0 + 1.5 and the two mediated
    # first-period paths 0.6 * 0.4 + 0.5 * 0.8.
    np.testing.assert_allclose(estimates[1] - estimates[0], 3.14, atol=0.15)


def test_dynamic_blip_shape_and_treatment_validation():
    outcome = np.zeros((4, 4))
    treatment = np.zeros((4, 3))

    with pytest.raises(ValueError, match="treatment.shape"):
        cm.ParallelTrendsSNMM().fit(outcome[:, :3], treatment)

    nonabsorbing = treatment.copy()
    nonabsorbing[0] = [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="absorbing"):
        cm.ParallelTrendsSNMM(treatment_mode="initiation").fit(
            outcome, nonabsorbing
        )

    with pytest.raises(ValueError, match="same shape"):
        cm.RegressionBlip().fit(outcome, treatment)

    with pytest.raises(ValueError, match="target_path length"):
        cm.DynamicCovariateBalance().fit(
            np.zeros(4), treatment, np.zeros((4, 3, 1)), [0.0, 0.0]
        )
