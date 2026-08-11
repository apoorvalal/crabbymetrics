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
