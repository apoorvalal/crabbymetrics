"""Small deterministic parity checks against public reference implementations.

These tests compare fitted solutions rather than statistical recovery from a
known DGP.  Keeping the samples small makes the optional-reference suite useful
as an ordinary unit-test gate on every supported Python version.
"""

import crabbymetrics as cm
import numpy as np
import pandas as pd
import pyfixest as pf
import pytest
from lifelines import CoxTimeVaryingFitter
from sklearn.linear_model import (
    ElasticNet as SklearnElasticNet,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    PoissonRegressor,
)
from sklearn.linear_model import (
    Ridge as SklearnRidge,
)


def linear_sample(seed=20260825, n=480, k=5):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, k))
    beta = np.linspace(0.9, -0.3, k)
    y = 0.35 + x @ beta + rng.normal(scale=0.4, size=n)
    return rng, x, y


def test_ols_solution_matches_sklearn_linear_regression():
    _rng, x, y = linear_sample()

    crab = cm.OLS()
    crab.fit(x, y)
    reference = LinearRegression().fit(x, y)

    summary = crab.summary(vcov="vanilla")
    np.testing.assert_allclose(summary["intercept"], reference.intercept_, atol=1e-12)
    np.testing.assert_allclose(summary["coef"], reference.coef_, atol=1e-12)
    np.testing.assert_allclose(
        crab.predict(x[:37]), reference.predict(x[:37]), atol=1e-12
    )


def test_ridge_solution_matches_sklearn_ridge():
    _rng, x, y = linear_sample(seed=20260826)
    penalty = 0.7

    crab = cm.Ridge(penalty=penalty)
    crab.fit(x, y)
    reference = SklearnRidge(alpha=penalty).fit(x, y)

    summary = crab.summary(vcov="vanilla")
    np.testing.assert_allclose(summary["intercept"], reference.intercept_, atol=1e-12)
    np.testing.assert_allclose(summary["coef"], reference.coef_, atol=1e-12)
    np.testing.assert_allclose(
        crab.predict(x[:37]), reference.predict(x[:37]), atol=1e-12
    )


def test_elastic_net_solution_matches_sklearn_on_centered_design():
    rng, x, _y = linear_sample(seed=20260827, n=650)
    x -= x.mean(axis=0)
    y = x @ np.array([0.8, -0.55, 0.3, 0.0, -0.1])
    y += rng.normal(scale=0.35, size=x.shape[0])
    y -= y.mean()
    penalty = 0.03
    l1_ratio = 0.35

    crab = cm.ElasticNet(
        penalty=penalty,
        l1_ratio=l1_ratio,
        tolerance=1e-9,
        max_iterations=10_000,
    )
    crab.fit(x, y)
    reference = SklearnElasticNet(
        alpha=penalty,
        l1_ratio=l1_ratio,
        tol=1e-9,
        max_iter=10_000,
    ).fit(x, y)

    summary = crab.summary()
    np.testing.assert_allclose(summary["intercept"], reference.intercept_, atol=1e-11)
    np.testing.assert_allclose(summary["coef"], reference.coef_, atol=1e-10)
    np.testing.assert_allclose(
        crab.predict(x[:41]), reference.predict(x[:41]), atol=1e-10
    )


def test_unpenalized_logit_matches_sklearn_probabilities():
    rng, x, _y = linear_sample(seed=20260828, n=900, k=4)
    beta = np.array([0.7, -0.5, 0.25, 0.1])
    probability = 1.0 / (1.0 + np.exp(-(-0.2 + x @ beta)))
    y = rng.binomial(1, probability).astype(np.int32)

    crab = cm.Logit(max_iterations=1_000, gradient_tolerance=1e-10)
    crab.fit(x, y)
    reference = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        tol=1e-10,
        max_iter=1_000,
    ).fit(x, y)

    summary = crab.summary()
    np.testing.assert_allclose(summary["intercept"], reference.intercept_[0], atol=1e-7)
    np.testing.assert_allclose(summary["coef"], reference.coef_[0], atol=1e-7)
    np.testing.assert_allclose(
        crab.predict(x[:53]), reference.predict_proba(x[:53])[:, 1], atol=1e-7
    )


def test_unpenalized_multinomial_logit_matches_sklearn_probabilities():
    rng, x, _y = linear_sample(seed=20260829, n=1_100, k=4)
    coef = np.array(
        [
            [0.7, -0.2, 0.1, 0.3],
            [-0.4, 0.6, 0.2, -0.1],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    intercept = np.array([0.2, -0.1, 0.0])
    logits = x @ coef.T + intercept
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in probabilities], dtype=np.int32)

    crab = cm.MultinomialLogit(max_iterations=1_000, gradient_tolerance=1e-9)
    crab.fit(x, y)
    reference = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        tol=1e-10,
        max_iter=1_000,
    ).fit(x, y)

    np.testing.assert_allclose(
        crab.predict(x[:61]), reference.predict_proba(x[:61]), atol=2e-7
    )
    np.testing.assert_array_equal(crab.predict_label(x[:61]), reference.predict(x[:61]))


def test_unpenalized_poisson_matches_sklearn_mean_predictions():
    rng, x, _y = linear_sample(seed=20260830, n=850, k=4)
    beta = np.array([0.3, -0.2, 0.15, 0.05])
    y = rng.poisson(np.exp(-0.1 + x @ beta)).astype(float)

    crab = cm.Poisson(max_iterations=1_000, tolerance=1e-10)
    crab.fit(x, y)
    reference = PoissonRegressor(
        alpha=0.0,
        tol=1e-10,
        max_iter=1_000,
    ).fit(x, y)

    summary = crab.summary(vcov="vanilla")
    np.testing.assert_allclose(summary["intercept"], reference.intercept_, atol=2e-6)
    np.testing.assert_allclose(summary["coef"], reference.coef_, atol=2e-6)
    np.testing.assert_allclose(
        crab.predict(x[:47]), reference.predict(x[:47]), atol=3e-6
    )


def test_horizontal_panel_ridge_matches_sklearn_donor_regression():
    rng = np.random.default_rng(20260831)
    n_control, n_treated, t_pre, t_post = 12, 3, 15, 6
    controls = rng.normal(size=(n_control, t_pre + t_post))
    treated = rng.dirichlet(np.ones(n_control), size=n_treated) @ controls
    treated += rng.normal(scale=0.02, size=treated.shape)
    panel = np.vstack([controls, treated])
    treatment = np.zeros_like(panel)
    treatment[n_control:, t_pre:] = 1.0
    penalty = 0.4

    crab = cm.HorizontalPanelRidge(penalty=penalty)
    crab.fit(panel, treatment)
    reference = SklearnRidge(alpha=penalty).fit(
        controls[:, :t_pre].T,
        treated[:, :t_pre].mean(axis=0),
    )

    summary = crab.summary()
    np.testing.assert_allclose(summary["intercept"], reference.intercept_, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(summary["coef"])[:n_control], reference.coef_, atol=1e-12
    )
    expected = reference.predict(controls.T)
    for unit in range(n_control, n_control + n_treated):
        np.testing.assert_allclose(crab.predict()[unit], expected, atol=1e-12)


@pytest.mark.parametrize(
    ("crab_vcov", "pyfixest_vcov"),
    [("vanilla", "iid"), ("hc1", "hetero")],
)
def test_one_way_fixed_effects_matches_pyfixest(crab_vcov, pyfixest_vcov):
    rng = np.random.default_rng(20260901)
    n = 720
    group = rng.integers(0, 30, size=n)
    x = rng.normal(size=(n, 3))
    y = x @ np.array([0.7, -0.2, 0.4]) + rng.normal(size=30)[group]
    y += rng.normal(size=n) * (0.5 + 0.2 * np.abs(x[:, 0]))

    crab = cm.FixedEffectsOLS()
    crab.fit(x, group.astype(np.uint32)[:, None], y)
    frame = pd.DataFrame(
        {"y": y, "group": group, **{f"x{j}": x[:, j] for j in range(3)}}
    )
    reference = pf.feols(
        "y ~ x0 + x1 + x2 | group",
        data=frame,
        vcov=pyfixest_vcov,
    )

    summary = crab.summary(vcov=crab_vcov)
    names = ["x0", "x1", "x2"]
    np.testing.assert_allclose(summary["coef"], reference.coef().loc[names], atol=1e-10)
    np.testing.assert_allclose(summary["coef_se"], reference.se().loc[names], atol=1e-9)


@pytest.mark.parametrize(
    ("crab_vcov", "pyfixest_vcov"),
    [("vanilla", "iid"), ("hc1", "hetero")],
)
def test_overidentified_two_sls_matches_pyfixest(crab_vcov, pyfixest_vcov):
    rng = np.random.default_rng(20260902)
    n = 900
    instruments = rng.normal(size=(n, 3))
    exogenous = rng.normal(size=(n, 2))
    confounder = rng.normal(size=n)
    endogenous = instruments @ np.array([0.8, -0.4, 0.2])
    endogenous += 0.4 * confounder + rng.normal(size=n)
    y = 1.3 * endogenous + exogenous @ np.array([0.5, -0.2])
    y += confounder * (0.7 + 0.2 * np.abs(exogenous[:, 0]))

    crab = cm.TwoSLS()
    crab.fit(endogenous[:, None], exogenous, instruments, y)
    frame = pd.DataFrame(
        {
            "y": y,
            "d": endogenous,
            "x0": exogenous[:, 0],
            "x1": exogenous[:, 1],
            **{f"z{j}": instruments[:, j] for j in range(3)},
        }
    )
    reference = pf.feols(
        "y ~ x0 + x1 | d ~ z0 + z1 + z2",
        data=frame,
        vcov=pyfixest_vcov,
    )

    summary = crab.summary(vcov=crab_vcov)
    names = ["d", "x0", "x1"]
    np.testing.assert_allclose(
        summary["intercept"], reference.coef()["Intercept"], atol=1e-10
    )
    np.testing.assert_allclose(summary["coef"], reference.coef().loc[names], atol=1e-10)
    np.testing.assert_allclose(
        summary["intercept_se"], reference.se()["Intercept"], atol=1e-6
    )
    np.testing.assert_allclose(summary["coef_se"], reference.se().loc[names], atol=1e-6)


def test_andersen_gill_solution_matches_lifelines_time_varying_cox():
    rng = np.random.default_rng(20260903)
    n = 420
    x = rng.normal(size=(n, 2))
    start = rng.uniform(0.0, 0.8, size=n)
    stop = start + rng.exponential(np.exp(-(x @ np.array([0.55, -0.3])))) + 0.01
    event = (rng.random(n) < 0.7).astype(float)

    crab = cm.AndersenGill()
    crab.fit(x, start, stop, event)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "start": start,
            "stop": stop,
            "event": event.astype(int),
            "x0": x[:, 0],
            "x1": x[:, 1],
        }
    )
    reference = CoxTimeVaryingFitter().fit(
        frame,
        id_col="id",
        start_col="start",
        stop_col="stop",
        event_col="event",
        show_progress=False,
    )

    np.testing.assert_allclose(
        crab.summary()["coef"], reference.params_.loc[["x0", "x1"]], atol=1e-9
    )
