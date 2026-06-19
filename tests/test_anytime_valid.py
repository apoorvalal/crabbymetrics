import numpy as np
from scipy import stats

import crabbymetrics as cm


def readme_style_data(seed=1, n=100):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    x = x - x.mean(axis=0)
    trt = rng.choice([0.0, 1.0], size=n)
    y = (
        1.0
        + 1.4 * x[:, 2]
        + 2.3 * trt
        + 2.0 * x[:, 0] * trt
        + 3.0 * x[:, 1] * trt
        + rng.normal(size=n)
    )
    design = np.column_stack([x, trt, x * trt[:, None]])
    return design, y


def vanilla_reference(design, y):
    x = np.column_stack([np.ones(design.shape[0]), design])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    df_resid = x.shape[0] - x.shape[1]
    cov = (resid @ resid / df_resid) * np.linalg.inv(x.T @ x)
    se = np.sqrt(np.diag(cov))
    return beta, se, df_resid


def test_optimal_g_locally_minimizes_anytime_confint_radius():
    n = 100
    k = 8
    alpha = 0.05
    g_star = cm.optimal_g(n, k, alpha)

    def radius(g):
        nu = n - k
        t = g / (g + n)
        powered = (t * alpha**2) ** (1 / (nu + 1))
        return np.sqrt(nu * (1 - powered) / (powered - t))

    delta = max(0.01 * g_star, 1e-4)
    assert radius(g_star) <= radius(max(1.0, g_star - delta))
    assert radius(g_star) <= radius(g_star + delta)


def test_av_readme_style_example_returns_anytime_summary_and_intervals():
    x, y = readme_style_data()
    model = cm.OLS()
    model.fit(x, y)

    g_star = cm.optimal_g(x.shape[0], x.shape[1] + 1, 0.05)
    summary = model.summary(vcov="vanilla", anytime_valid=True, g=g_star)
    ci = summary["confint"]
    av_summary = cm.av(model, g=g_star)

    assert summary["anytime_valid"] is True
    assert summary["g"] == g_star
    np.testing.assert_allclose(av_summary["p_value"], summary["p_value"])
    assert summary["estimate"].shape == (x.shape[1] + 1,)
    assert ci.shape == (x.shape[1] + 1, 2)
    assert np.all(ci[:, 0] <= summary["estimate"])
    assert np.all(ci[:, 1] >= summary["estimate"])


def test_av_p_values_and_confints_are_more_conservative_than_classical_ols():
    x, y = readme_style_data()
    model = cm.OLS()
    model.fit(x, y)
    g_star = cm.optimal_g(x.shape[0], x.shape[1] + 1, 0.05)

    beta, se, df_resid = vanilla_reference(x, y)
    classical_p = 2.0 * stats.t.sf(np.abs(beta / se), df_resid)
    classical_radius = stats.t.ppf(0.975, df_resid) * se
    classical_ci = np.column_stack([beta - classical_radius, beta + classical_radius])

    av_summary = model.summary(vcov="vanilla", anytime_valid=True, g=g_star)
    av_ci = av_summary["confint"]

    assert np.all(av_summary["p_value"] >= classical_p)
    assert np.all(av_ci[:, 0] <= classical_ci[:, 0])
    assert np.all(av_ci[:, 1] >= classical_ci[:, 1])


def test_av_hc0_changes_standard_errors_and_validates_vcov():
    x, y = readme_style_data()
    model = cm.OLS()
    model.fit(x, y)

    classic = model.summary(vcov="vanilla", anytime_valid=True, g=2.0)
    robust = model.summary(vcov="hc0", anytime_valid=True, g=2.0)

    assert robust["vcov_type"] == "hc0"
    assert not np.allclose(classic["std_error"], robust["std_error"])

    try:
        model.summary(vcov="bad", anytime_valid=True, g=2.0)
    except ValueError as exc:
        assert "vcov" in str(exc)
    else:
        raise AssertionError("invalid vcov should raise")
