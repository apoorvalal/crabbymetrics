"""Missing outcomes are excluded, not treated as observed zeros or outcomes."""

import crabbymetrics as cm
import numpy as np
import pytest
from scipy.optimize import minimize


def panel():
    rng = np.random.default_rng(941)
    t = np.arange(14)
    controls = rng.normal(size=(6, 1)) + rng.normal(size=(6, 2)) @ np.vstack(
        [np.sin(t / 3), t / 8]
    )
    base = 0.3 * controls[0] + 0.7 * controls[3] + 0.4
    y = np.vstack([controls, base - 0.15, base + 0.15])
    w = np.zeros_like(y)
    w[6:, 8:] = 1
    y += 1.25 * w
    mask = np.ones(y.shape, dtype=bool)
    for i, t in [(0, 1), (1, 3), (2, 10), (3, 12), (6, 2), (7, 9)]:
        mask[i, t] = False
    return y, w, mask


def estimators():
    return [
        cm.MatrixCompletion(lambda_fraction=0.03, max_iterations=3000, tolerance=1e-9),
        cm.HorizontalPanelRidge(penalty=0.2),
        cm.SyntheticDID(zeta_omega=0.3, zeta_lambda=0.3, max_iterations=20000),
    ]


@pytest.mark.parametrize("index", range(3))
def test_mask_invariance_no_post_outcome_leakage_and_observed_target(index):
    y, w, mask = panel()
    model = estimators()[index]
    model.fit(y, w, observed=mask)
    prediction = model.predict().copy()
    summary = model.summary()
    assert np.isnan(summary["treatment_effect"][6:][~mask[6:]]).all()
    assert summary["n_observed"] == mask.sum()
    assert summary["n_treated_observed"] == np.sum(mask & (w == 1))
    group = summary["group_means"]
    np.testing.assert_allclose(
        group["treated_mean"] - group["counterfactual_mean"], group["effect"]
    )

    hidden = y.copy()
    hidden[~mask] = 1e12
    hidden[w == 1] += 50
    other = estimators()[index]
    other.fit(hidden, w, observed=mask)
    np.testing.assert_allclose(other.predict(), prediction, equal_nan=True, atol=1e-10)
    assert other.summary()["att"] == pytest.approx(summary["att"] + 50, abs=1e-9)

    y[~mask] = np.nan
    other.fit(y, w)
    np.testing.assert_allclose(other.predict(), prediction, equal_nan=True, atol=1e-10)
    assert other.summary()["att"] == pytest.approx(summary["att"], abs=1e-10)
    if index != 2:
        assert summary["att"] == pytest.approx(
            np.mean((y - prediction)[mask & (w == 1)])
        )


@pytest.mark.parametrize("index", range(3))
def test_explicit_complete_mask_preserves_complete_panel_result(index):
    y, w, _ = panel()
    model = estimators()[index]
    model.fit(y, w)
    prediction, att = model.predict().copy(), model.summary()["att"]
    model.fit(y, w, observed=np.ones_like(y, dtype=bool))
    np.testing.assert_array_equal(model.predict(), prediction)
    assert model.summary()["att"] == att


def test_masked_mc_matches_observed_only_two_way_fe_least_squares():
    y, w, observed = panel()
    train = observed & (w == 0)
    n, t = y.shape
    unit, time = np.indices(y.shape)
    design = np.c_[np.eye(n)[unit.ravel()], np.eye(t)[time.ravel(), 1:]]
    coef = np.linalg.lstsq(design[train.ravel()], y[train], rcond=None)[0]
    expected = (design @ coef).reshape(y.shape)
    model = cm.MatrixCompletion(
        lambda_l=1e6, max_iterations=1000, effect_iterations=30, tolerance=1e-12
    )
    model.fit(y, w, observed=observed)
    np.testing.assert_allclose(model.predict(), expected, atol=1e-7)
    assert np.linalg.norm(model.summary()["low_rank"]) < 1e-8


def test_mc_low_rank_completion_recovers_randomly_missing_cells():
    rng = np.random.default_rng(12)
    y0 = rng.normal(size=(35, 2)) @ rng.normal(size=(2, 27))
    w = np.zeros_like(y0)
    w[:5, 18:] = 1
    observed = rng.random(y0.shape) > 0.15
    y = y0 + 2 * w
    model = cm.MatrixCompletion(
        lambda_fraction=0.015, max_iterations=3000, tolerance=1e-9
    )
    model.fit(y, w, observed=observed)
    prediction = model.predict()
    assert np.mean((prediction[~observed] - y0[~observed]) ** 2) < 0.05
    assert model.summary()["att"] == pytest.approx(2, abs=0.1)


def test_ridge_matches_independent_overlap_regressions():
    y, w, observed = panel()
    y[~observed] = np.nan
    model = cm.HorizontalPanelRidge(penalty=0.2)
    model.fit(y, w)
    pred = model.predict()
    target = np.nanmean(y[6:, :8], axis=0)
    for t in range(y.shape[1]):
        donors = np.flatnonzero(np.isfinite(y[:6, t]))
        train = np.isfinite(y[donors, :8]).all(axis=0) & np.isfinite(target)
        x = np.c_[np.ones(train.sum()), y[donors, :8][:, train].T]
        penalty = np.diag(np.r_[0, np.full(len(donors), 0.2)])
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ target[train])
        expected = np.r_[1, y[donors, t]] @ coef
        np.testing.assert_allclose(pred[6:, t], expected, atol=1e-10)
    assert len(model.summary()["overlap_models"]) > 1
    assert np.isnan(model.summary()["cohort_coef"]).all()


def normalized_objective(weights, x, target, zeta):
    available = np.isfinite(x)
    mass = available @ weights
    if np.any(mass <= 1e-12):
        return 1e100
    prediction = np.where(available, x, 0) @ weights / mass
    residual = target - prediction
    residual -= residual.mean()
    return np.mean(residual**2) + zeta**2 * (weights @ weights)


def test_sdid_masked_weights_match_scipy_and_att_matches_available_contrast():
    y, w, observed = panel()
    y[~observed] = np.nan
    model = estimators()[2]
    model.fit(y, w)
    s = model.summary()
    omega = s["unit_weights"][0, :6]
    lam = s["time_weights"][0, :8]
    target = np.nanmean(y[6:, :8], axis=0)
    counts = np.isfinite(y[6:, 8:]).sum(axis=0)
    control_post = np.nansum(y[:6, 8:] * counts, axis=1) / (
        np.isfinite(y[:6, 8:]) @ counts
    )
    for x, response, weight in [
        (y[:6, :8].T, target, omega),
        (y[:6, :8], control_post, lam),
    ]:
        ref = minimize(
            normalized_objective,
            np.full(x.shape[1], 1 / x.shape[1]),
            args=(x, response, 0.3),
            method="SLSQP",
            bounds=[(0, 1)] * x.shape[1],
            constraints=[
                {"type": "eq", "fun": lambda b: b.sum() - 1},
                {"type": "ineq", "fun": lambda b, x=x: np.isfinite(x) @ b - 1e-9},
            ],
            options={"ftol": 1e-12, "maxiter": 5000},
        )
        assert ref.success
        assert normalized_objective(weight, x, response, 0.3) == pytest.approx(
            ref.fun, abs=1e-8
        )
        assert np.all(weight >= 0)
        assert weight.sum() == pytest.approx(1)
    synth = np.nansum(y[:6] * omega[:, None], axis=0) / (np.isfinite(y[:6]).T @ omega)
    treated = np.nanmean(y[6:], axis=0)
    expected = np.average(treated[8:] - synth[8:], weights=counts) - lam @ (
        treated[:8] - synth[:8]
    )
    assert s["att"] == pytest.approx(expected, abs=1e-10)
    assert all(
        d["unit_projected_gradient"] < 1e-8 and d["time_projected_gradient"] < 1e-8
        for d in s["missing_diagnostics"]
    )


@pytest.mark.parametrize("index", range(3))
def test_mask_validation_and_failed_refit_clear_state(index):
    y, w, observed = panel()
    model = estimators()[index]
    model.fit(y, w)
    with pytest.raises(ValueError, match="same shape"):
        model.fit(y, w, observed=observed[:-1])
    with pytest.raises(ValueError, match="not fitted"):
        model.summary()
    unknown_w = w.copy()
    unknown_w[0, 0] = np.nan
    with pytest.raises(ValueError, match="W must be known"):
        model.fit(y, unknown_w, observed=observed)
    y[0, 0] = np.nan
    with pytest.raises(ValueError, match="observed Y"):
        model.fit(y, w, observed=observed)


def test_mc_rejects_unidentified_rows_and_disconnected_fe_graph():
    y, w, observed = panel()
    observed[0] = False
    with pytest.raises(ValueError, match="unit 0.*no observed untreated"):
        cm.MatrixCompletion().fit(y, w, observed=observed)
    y = np.arange(24, dtype=float).reshape(4, 6)
    w = np.zeros_like(y)
    w[3, 4:] = 1
    mask = np.zeros_like(y, dtype=bool)
    mask[:2, :2] = True
    mask[2:, 2:] = True
    with pytest.raises(ValueError, match="disconnected"):
        cm.MatrixCompletion().fit(y, w, observed=mask)


def test_sdid_masked_solver_rejects_exhausted_budget():
    y, w, observed = panel()
    with pytest.raises(ValueError, match="did not converge"):
        cm.SyntheticDID(zeta_omega=0.3, zeta_lambda=0.3, max_iterations=1).fit(
            y, w, observed=observed
        )


def test_sdid_bootstrap_resamples_missingness_with_unit_histories():
    y, w, observed = panel()
    model = estimators()[2]
    model.fit(y, w, observed=observed)
    se = model.se("bootstrap", replications=20, seed=123)
    assert np.isfinite(se) and se > 0


@pytest.mark.parametrize("index", [1, 2])
def test_no_donors_at_observed_post_period_is_not_silently_dropped(index):
    y, w, observed = panel()
    observed[:6, 10] = False
    with pytest.raises(ValueError, match="period 10.*no (donor|observed positive)"):
        estimators()[index].fit(y, w, observed=observed)


def test_staggered_sdid_aggregation_matches_separate_cohort_fits():
    y, w, observed = panel()
    y -= 1.25 * w
    w[7, 8:10] = 0
    y += 1.25 * w
    model = estimators()[2]
    model.fit(y, w, observed=observed)
    estimates, counts = [], []
    for i in [6, 7]:
        rows = [*range(6), i]
        separate = estimators()[2]
        separate.fit(y[rows], w[rows], observed=observed[rows])
        estimates.append(separate.summary()["att"])
        counts.append(np.sum(observed[i] & (w[i] == 1)))
        np.testing.assert_allclose(
            model.predict()[i], separate.predict()[-1], equal_nan=True
        )
    assert model.summary()["att"] == pytest.approx(
        np.average(estimates, weights=counts), abs=1e-10
    )
