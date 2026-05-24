import math

import numpy as np
import pytest
import statsmodels.api as sm

import crabbymetrics as cm


def scalar_stat(result):
    return float(np.asarray(result.statistic).squeeze())


def scalar_pvalue(result):
    return float(np.asarray(result.pvalue).squeeze())


def test_array_wald_test_matches_statsmodels_ols_chi_square():
    rng = np.random.default_rng(123)
    n = 240
    x = rng.normal(size=(n, 3))
    y = 1.0 + x @ np.array([0.25, 0.0, 0.8]) + rng.normal(scale=0.7, size=n)
    x_sm = sm.add_constant(x)

    sm_res = sm.OLS(y, x_sm).fit()
    r = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    out = cm.wald_test(sm_res.params, sm_res.cov_params(), r)
    ref = sm_res.wald_test(r, use_f=False, scalar=True)

    assert out["test"] == "wald"
    assert out["df"] == 2
    np.testing.assert_allclose(out["statistic"], scalar_stat(ref), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(out["p_value"], scalar_pvalue(ref), atol=1e-10, rtol=1e-10)


def test_ols_wald_method_matches_statsmodels_vanilla_and_hc1():
    rng = np.random.default_rng(124)
    n = 300
    x = rng.normal(size=(n, 3))
    hetero_scale = 0.4 + 0.4 * np.abs(x[:, 0])
    y = 0.8 + x @ np.array([0.35, -0.15, 0.5]) + rng.normal(scale=hetero_scale)
    r = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    q = np.array([0.3, -0.2])

    model = cm.OLS()
    model.fit(x, y)
    sm_res = sm.OLS(y, sm.add_constant(x)).fit()

    for vcov, sm_res_vcov in [
        ("vanilla", sm_res),
        ("hc1", sm_res.get_robustcov_results(cov_type="HC1")),
    ]:
        out = model.wald_test(r, q, vcov=vcov)
        ref = sm_res_vcov.wald_test((r, q), use_f=False, scalar=True)
        np.testing.assert_allclose(out["statistic"], scalar_stat(ref), atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(out["p_value"], scalar_pvalue(ref), atol=1e-8, rtol=1e-8)
        assert out["df"] == 2


def test_array_wald_test_matches_statsmodels_logit_chi_square():
    rng = np.random.default_rng(456)
    n = 600
    x = rng.normal(size=(n, 2))
    logits = -0.2 + x @ np.array([0.55, -0.35])
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p).astype(np.int32)
    r = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sm_res = sm.Logit(y, sm.add_constant(x)).fit(disp=False)
    out = cm.wald_test(sm_res.params, sm_res.cov_params(), r)
    ref = sm_res.wald_test(r, use_f=False, scalar=True)

    np.testing.assert_allclose(out["statistic"], scalar_stat(ref), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(out["p_value"], scalar_pvalue(ref), atol=1e-10, rtol=1e-10)
    assert out["df"] == 2


def test_logit_wald_method_matches_its_summary_covariance():
    rng = np.random.default_rng(457)
    n = 400
    x = rng.normal(size=(n, 2))
    p = 1.0 / (1.0 + np.exp(-(0.1 + x @ np.array([0.45, -0.25]))))
    y = rng.binomial(1, p).astype(np.int32)
    r = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    model = cm.Logit()
    model.fit(x, y)
    summary = model.summary()
    coef = np.r_[summary["intercept"], summary["coef"]]
    expected = cm.wald_test(coef, summary["vcov"], r)
    out = model.wald_test(r)

    np.testing.assert_allclose(out["statistic"], expected["statistic"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(out["p_value"], expected["p_value"], atol=1e-12, rtol=1e-12)
    assert out["df"] == 2


def test_poisson_wald_method_matches_statsmodels_vanilla_and_sandwich():
    rng = np.random.default_rng(789)
    n = 500
    x = rng.normal(size=(n, 2))
    mu = np.exp(0.15 + x @ np.array([0.25, -0.2]))
    y = rng.poisson(mu).astype(float)
    r = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    model = cm.Poisson()
    model.fit(x, y)
    sm_res = sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit()

    for vcov, sm_res_vcov in [
        ("vanilla", sm_res),
        ("sandwich", sm.GLM(y, sm.add_constant(x), family=sm.families.Poisson()).fit(cov_type="HC0")),
    ]:
        out = model.wald_test(r, vcov=vcov)
        ref = sm_res_vcov.wald_test(r, use_f=False, scalar=True)
        # The Poisson estimator uses crabbymetrics' Newton path, while the
        # reference uses statsmodels IRLS; the fitted optima agree to optimizer
        # tolerance rather than bit-for-bit.
        np.testing.assert_allclose(out["statistic"], scalar_stat(ref), atol=1e-2, rtol=1e-4)
        np.testing.assert_allclose(out["p_value"], scalar_pvalue(ref), atol=1e-4, rtol=1e-4)
        assert out["df"] == 2


def test_likelihood_ratio_test_matches_statsmodels_compare_lr_test_for_nested_ols():
    rng = np.random.default_rng(321)
    n = 180
    x = rng.normal(size=(n, 3))
    y = 0.5 + x @ np.array([0.4, 0.0, 0.2]) + rng.normal(scale=0.8, size=n)
    unrestricted = sm.OLS(y, sm.add_constant(x)).fit()
    restricted = sm.OLS(y, sm.add_constant(x[:, [0]])).fit()

    stat, pvalue, df = unrestricted.compare_lr_test(restricted)
    out = cm.lr_test(unrestricted.llf, restricted.llf, int(df))
    alias = cm.likelihood_ratio_test(unrestricted.llf, restricted.llf, int(df))

    assert out["test"] == "likelihood_ratio"
    assert out == alias
    assert out["df"] == int(df)
    np.testing.assert_allclose(out["statistic"], stat, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(out["p_value"], pvalue, atol=1e-10, rtol=1e-10)


def test_likelihood_ratio_test_uses_known_chi_square_two_tail_value():
    out = cm.likelihood_ratio_test(-100.0, -103.0, 2)
    assert out["df"] == 2
    np.testing.assert_allclose(out["statistic"], 6.0, atol=0.0, rtol=0.0)
    # Chi-square(2) survival is exp(-x / 2).
    np.testing.assert_allclose(out["p_value"], math.exp(-3.0), atol=1e-10, rtol=1e-10)


def test_hypothesis_tests_validate_inputs():
    coef = np.ones(3)
    vcov = np.eye(3)
    with pytest.raises(ValueError, match="r must have len"):
        cm.wald_test(coef, vcov, np.ones((1, 2)))
    with pytest.raises(ValueError, match="q must have"):
        cm.wald_test(coef, vcov, np.ones((2, 3)), np.zeros(1))
    with pytest.raises(ValueError, match="unrestricted_loglik"):
        cm.likelihood_ratio_test(-10.0, -9.0, 1)
    with pytest.raises(ValueError, match="degrees of freedom"):
        cm.likelihood_ratio_test(-9.0, -10.0, 0)
