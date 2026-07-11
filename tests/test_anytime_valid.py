import numpy as np
from scipy import stats

import crabbymetrics as cm


MTCARS_MPG = np.array(
    [
        21.0,
        21.0,
        22.8,
        21.4,
        18.7,
        18.1,
        14.3,
        24.4,
        22.8,
        19.2,
        17.8,
        16.4,
        17.3,
        15.2,
        10.4,
        10.4,
        14.7,
        32.4,
        30.4,
        33.9,
        21.5,
        15.5,
        15.2,
        13.3,
        19.2,
        27.3,
        26.0,
        30.4,
        15.8,
        19.7,
        15.0,
        21.4,
    ]
)

MTCARS_WT = np.array(
    [
        2.620,
        2.875,
        2.320,
        3.215,
        3.440,
        3.460,
        3.570,
        3.190,
        3.150,
        3.440,
        3.440,
        4.070,
        3.730,
        3.780,
        5.250,
        5.424,
        5.345,
        2.200,
        1.615,
        1.835,
        2.465,
        3.520,
        3.435,
        3.840,
        3.845,
        1.935,
        2.140,
        1.513,
        3.170,
        2.770,
        3.570,
        2.780,
    ]
)

MTCARS_HP = np.array(
    [
        110,
        110,
        93,
        110,
        175,
        105,
        245,
        62,
        95,
        123,
        123,
        180,
        180,
        180,
        205,
        215,
        230,
        66,
        52,
        65,
        97,
        150,
        150,
        245,
        175,
        66,
        91,
        113,
        264,
        175,
        335,
        109,
    ],
    dtype=float,
)

AVLM_MTCARS_REFERENCE = {
    1.0: {
        "p_value": np.array(
            [1.8526048177043199e-16, 3.9603205911245655e-05, 3.3643443230645109e-02]
        ),
        "confint_95": np.array(
            [
                [31.863387622353187, 42.591152610541215],
                [-6.0006319484765864, -1.7550295363327777],
                [-0.062067342269780626, -0.0014785516945413635],
            ]
        ),
        "confint_90": np.array(
            [
                [32.335194919210657, 42.119345313683745],
                [-5.8139102781369889, -1.9417512066723754],
                [-0.059402646172071333, -0.0041432477922506532],
            ]
        ),
        "f_p_value": 4.2262170684811834e-10,
    },
    2.0: {
        "p_value": np.array(
            [1.0768385067924622e-14, 4.7998872742069833e-05, 2.8887593941210040e-02]
        ),
        "confint_95": np.array(
            [
                [31.980951354011111, 42.473588878883291],
                [-5.9541051217557293, -1.8015563630536353],
                [-0.061403360133286358, -0.0021425338310356347],
            ]
        ),
        "confint_90": np.array(
            [
                [32.477613963140605, 41.976926269753797],
                [-5.757546754410475, -1.9981147303988889],
                [-0.058598284999348181, -0.0049476089649738084],
            ]
        ),
        "f_p_value": 1.2426229776970821e-09,
    },
    5.0: {
        "p_value": np.array(
            [1.6947241835194956e-11, 1.1830908637715881e-04, 3.0473685890084916e-02]
        ),
        "confint_95": np.array(
            [
                [31.985337066982957, 42.469203165911445],
                [-5.9523694392913242, -1.8032920455180395],
                [-0.061378590291192775, -0.002167303673129211],
            ]
        ),
        "confint_90": np.array(
            [
                [32.541098004414145, 41.913442228480257],
                [-5.7324224159242796, -2.0232390688850845],
                [-0.058239736756929769, -0.0053061572073922238],
            ]
        ),
        "f_p_value": 2.6179680743471358e-08,
    },
}

AVLM_MTCARS_ESTIMATE = np.array([37.227270116447201, -3.8778307424046821, -0.031772946982160995])
AVLM_MTCARS_STD_ERROR = np.array([1.5987875379993937, 0.63273349437739523, 0.009029709675855719])
AVLM_MTCARS_T_VALUE = np.array([23.284688697930868, -6.1286952198104148, -3.5187119102087814])
AVLM_MTCARS_F_STATISTIC = 69.21121339177769


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


def mtcars_design():
    return np.column_stack([MTCARS_WT, MTCARS_HP]), MTCARS_MPG


def assert_mtcars_avlm_parity(g):
    x, y = mtcars_design()
    model = cm.OLS()
    model.fit(x, y)

    summary = model.summary(vcov="vanilla", anytime_valid=True, g=g)
    ref = AVLM_MTCARS_REFERENCE[g]

    np.testing.assert_allclose(summary["estimate"], AVLM_MTCARS_ESTIMATE, rtol=0, atol=2e-13)
    np.testing.assert_allclose(summary["std_error"], AVLM_MTCARS_STD_ERROR, rtol=0, atol=2e-13)
    np.testing.assert_allclose(summary["t_value"], AVLM_MTCARS_T_VALUE, rtol=0, atol=2e-12)
    np.testing.assert_allclose(summary["p_value"], ref["p_value"], rtol=2e-12, atol=1e-18)
    np.testing.assert_allclose(summary["confint"], ref["confint_95"], rtol=0, atol=5e-13)
    np.testing.assert_allclose(summary["f_statistic"], AVLM_MTCARS_F_STATISTIC, rtol=0, atol=1e-11)
    np.testing.assert_allclose(summary["f_p_value"], ref["f_p_value"], rtol=2e-12, atol=1e-18)

    summary_90 = model.summary(vcov="vanilla", anytime_valid=True, g=g, level=0.90)
    np.testing.assert_allclose(summary_90["confint"], ref["confint_90"], rtol=0, atol=5e-13)


def test_optimal_g_matches_avlm_reference_values():
    cases = [
        (10_000, 5, 0.05, 1217.1738249350553),
        (100, 8, 0.05, 11.581536162114757),
        (32, 3, 0.05, 3.3211373089891274),
    ]
    for n, k, alpha, expected in cases:
        np.testing.assert_allclose(cm.optimal_g(n, k, alpha), expected, rtol=0, atol=5e-4)


def test_avlm_mtcars_lm_p_values_confints_and_f_test_match_for_lindon_g_grid():
    for g in (1.0, 2.0, 5.0):
        assert_mtcars_avlm_parity(g)


def test_avlm_mtcars_hc0_summary_matches_lindon_robust_path():
    x, y = mtcars_design()
    model = cm.OLS()
    model.fit(x, y)
    summary = model.summary(vcov="hc0", anytime_valid=True, g=2.0)

    np.testing.assert_allclose(summary["estimate"], AVLM_MTCARS_ESTIMATE, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        summary["std_error"],
        np.array([1.9389139564175393, 0.61992750528988649, 0.0066460579081830942]),
        rtol=0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        summary["t_value"],
        np.array([19.200063000851607, -6.2552971263815049, -4.7807207552374633]),
        rtol=0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        summary["p_value"],
        np.array([1.5703848651104506e-13, 3.5382282066978221e-05, 1.3320264559467426e-03]),
        rtol=2e-12,
        atol=1e-18,
    )
    np.testing.assert_allclose(summary["f_statistic"], 1954.6146621113496, rtol=0, atol=2e-9)


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
