import crabbymetrics as cm
import numpy as np
import pytest


def data(n=120, p=3):
    rng = np.random.default_rng(20260905)
    x = rng.normal(size=(n, p))
    y = 0.5 + x @ np.linspace(0.3, 0.9, p) + rng.normal(scale=0.5, size=n)
    return x, y


@pytest.mark.parametrize("cls", [cm.OLS, cm.Ridge, cm.ElasticNet, cm.TwoSLS])
def test_prediction_contract(cls):
    x, y = data()
    model = cls()
    if cls is cm.TwoSLS:
        model.fit(x, np.empty((len(x), 0)), x, y)
    else:
        model.fit(x, y)
    for width in (0, 2, 4):
        with pytest.raises(ValueError, match="columns"):
            model.predict(np.ones((10, width)))
    bad = x.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.predict(bad)
    assert model.predict(np.empty((0, 3))).shape == (0,)
    np.testing.assert_allclose(model.predict(x[::2]), model.predict(x)[::2])


@pytest.mark.parametrize("penalty", [1.0, [0.1, 1.0]])
@pytest.mark.parametrize("bad", [[], [1.0], [np.nan] * 120, [-1.0] * 120, [0.0] * 120])
def test_ridge_weight_preflight(penalty, bad):
    x, y = data()
    with pytest.raises(ValueError, match="sample_weight"):
        cm.Ridge(penalty=penalty).fit_weighted(x, y, bad)


@pytest.mark.parametrize("cls", [cm.OLS, cm.Ridge, cm.ElasticNet])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_finite_fit_and_failed_refit(cls, value):
    x, y = data()
    model = cls()
    model.fit(x, y)
    x[0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        model.fit(x, y)
    with pytest.raises(ValueError, match="fitted"):
        model.predict(np.zeros((1, 3)))


@pytest.mark.parametrize("ratio", [0.0, 0.35, 1.0])
def test_elasticnet_translation_and_reference(ratio):
    sklearn = pytest.importorskip("sklearn.linear_model")
    x, y = data(n=600)
    x -= x.mean(axis=0)
    shift = np.array([8.0, -5.0, 12.0])
    a = cm.ElasticNet(
        penalty=0.03, l1_ratio=ratio, tolerance=1e-9, max_iterations=20000
    )
    b = cm.ElasticNet(
        penalty=0.03, l1_ratio=ratio, tolerance=1e-9, max_iterations=20000
    )
    a.fit(x, y)
    b.fit(x + shift, y)
    reference = (
        sklearn.Ridge(alpha=len(x) * 0.03)
        if ratio == 0
        else sklearn.ElasticNet(alpha=0.03, l1_ratio=ratio, tol=1e-10, max_iter=20000)
    )
    reference.fit(x + shift, y)
    np.testing.assert_allclose(a.predict(x), b.predict(x + shift), atol=1e-8)
    np.testing.assert_allclose(
        b.predict(x + shift), reference.predict(x + shift), atol=2e-6
    )
    summary = b.summary()
    np.testing.assert_allclose(
        summary["intercept"],
        y.mean() - (x + shift).mean(axis=0) @ summary["coef"],
        atol=1e-10,
    )


@pytest.mark.parametrize("cls", [cm.OLS, cm.Ridge, cm.TwoSLS, cm.FixedEffectsOLS])
@pytest.mark.parametrize("vcov", ["vanilla", "hc1", "newey_west", "cluster"])
def test_zero_weight_inference_invariance(cls, vcov):
    x, y = data()
    w = np.ones(len(x))
    w[::4] = 0
    keep = w > 0
    clusters = np.arange(len(x), dtype=np.int64) // 5
    clusters[~keep] = 1000 + np.arange((~keep).sum())
    fe = (np.arange(len(x), dtype=np.uint32) % 7)[:, None]
    fe[~keep, 0] = 99
    a, b = cls(), cls()
    if cls is cm.TwoSLS:
        a.fit_weighted(x, np.empty((len(x), 0)), x, y, w.tolist())
        b.fit(x[keep], np.empty((keep.sum(), 0)), x[keep], y[keep])
    elif cls is cm.FixedEffectsOLS:
        a.fit_weighted(x, fe, y, w.tolist())
        b.fit(x[keep], fe[keep], y[keep])
    else:
        a.fit_weighted(x, y, w.tolist())
        b.fit(x[keep], y[keep])
    kwargs = {"vcov": vcov}
    ka = kwargs | ({"clusters": clusters} if vcov == "cluster" else {})
    kb = kwargs | ({"clusters": clusters[keep]} if vcov == "cluster" else {})
    np.testing.assert_allclose(
        a.summary(**ka)["coef_se"], b.summary(**kb)["coef_se"], rtol=1e-8
    )
    if hasattr(a, "wald_test"):
        r = np.eye(3) if cls is cm.FixedEffectsOLS else np.eye(4)[1:]
        wa, wb = a.wald_test(r, **ka), b.wald_test(r, **kb)
        np.testing.assert_allclose(wa["statistic"], wb["statistic"], rtol=1e-8)


def test_ols_ill_conditioned_covariance_uses_design_qr():
    rng = np.random.default_rng(40)
    z = rng.normal(size=300)
    x = np.column_stack([z, z + 1e-7 * rng.normal(size=300)])
    y = z + rng.normal(scale=0.1, size=300)
    model = cm.OLS()
    model.fit(x, y)
    design = np.column_stack([np.ones(300), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    _, r = np.linalg.qr(design, mode="reduced")
    ri = np.linalg.solve(r, np.eye(3))
    expected = np.sqrt(np.diag(ri @ ri.T) * np.sum((y - design @ beta) ** 2) / 297)[1:]
    np.testing.assert_allclose(
        model.summary(vcov="vanilla")["coef_se"], expected, rtol=1e-7
    )


@pytest.mark.parametrize("scale", [1e-10, 1e-6, 1.0, 1e6, 1e10])
@pytest.mark.parametrize("analytic", [False, True])
def test_gmm_moment_units(scale, analytic):
    values = np.linspace(2.0, 6.0, 100)
    jac = (lambda theta, y: np.array([[-scale]])) if analytic else None
    model = cm.GMM(lambda theta, y: scale * (y - theta[0])[:, None], jac)
    model.fit(values, np.zeros(1), weighting="identity")
    summary = model.summary()
    np.testing.assert_allclose(summary["coef"], [4.0], atol=1e-6)
    np.testing.assert_allclose(summary["se"], [values.std() / 10], rtol=1e-6)


def test_callback_inference_is_snapshotted():
    values = np.linspace(-2.0, 2.0, 100)
    for cls in (cm.MEstimator, cm.GMM):
        y = values.copy()
        if cls is cm.MEstimator:
            model = cls(
                lambda t, z: (
                    float(np.mean((z - t[0]) ** 2) / 2),
                    np.array([t[0] - z.mean()]),
                ),
                lambda t, z: (z - t[0])[:, None],
            )
        else:
            model = cls(
                lambda t, z: (z - t[0])[:, None], lambda t, z: np.array([[-1.0]])
            )
        model.fit(y, np.zeros(1))
        y *= 4
        first = model.summary()
        np.testing.assert_allclose(first["se"], [values.std() / 10], rtol=1e-7)
        y *= 4
        np.testing.assert_array_equal(first["se"], model.summary()["se"])


def test_gmm_inference_assumption_and_invalid_callbacks():
    y = np.linspace(2, 6, 100)
    model = cm.GMM(lambda t, z: (z - t[0])[:, None])
    model.fit(y, np.zeros(1))
    with pytest.raises(ValueError, match="optimal"):
        model.summary(vcov="vanilla")
    assert model.summary(vcov="vanilla", assume_optimal_weighting=True)[
        "assume_optimal_weighting"
    ]
    with pytest.raises(ValueError, match="finite"):
        model.fit(y * np.nan, np.zeros(1))
    with pytest.raises(ValueError, match="fitted"):
        model.summary()


def test_gmm_damping_does_not_regularize_statistical_weighting():
    rng = np.random.default_rng(52)
    y = rng.normal(size=(200, 2)) + 3
    model = cm.GMM(
        lambda t, z: z - t[0],
        lambda t, z: -np.ones((2, 1)),
        ridge=0.5,
        max_iterations=200,
    )
    model.fit(y, np.zeros(1), weighting="two_step")
    summary = model.summary(vcov="vanilla")
    first_moments = y - y.mean()
    expected_weight = np.linalg.inv(first_moments.T @ first_moments / len(y))
    np.testing.assert_allclose(summary["weight_matrix"], expected_weight, rtol=1e-6)
    assert summary["j_test_valid"]


def test_gmm_rejects_changing_observation_count_in_derivative():
    def moments(t, y):
        return (y[:50] if t[0] > 0 else y)[:, None] - t[0]

    model = cm.GMM(moments)
    with pytest.raises(ValueError, match="shape changed"):
        model.fit(np.linspace(2, 6, 100), np.zeros(1))


def test_logit_does_not_invent_identification():
    x = np.ones((100, 1))
    y = (np.arange(100) % 2).astype(np.int32)
    model = cm.Logit()
    model.fit(x, y)
    summary = model.summary()
    assert summary["converged"]
    assert not summary["inference_available"]
    assert summary["coef_se"] is None
    assert "rank deficient" in summary["inference_reason"]
    with pytest.raises(ValueError, match="rank deficient"):
        model.wald_test(np.eye(2))


@pytest.mark.parametrize("start_stop", [False, True])
def test_cox_translation_ties_and_row_order(start_stop):
    rng = np.random.default_rng(38)
    x = rng.normal(size=(180, 2))
    stop = np.ceil(rng.exponential(np.exp(-0.7 * x[:, 0])) * 5) + 1
    event = rng.binomial(1, 0.8, len(x)).astype(float)
    start = rng.uniform(0, 0.8, len(x)) if start_stop else None
    order = rng.permutation(len(x))

    def fit(design, idx):
        model = cm.AndersenGill() if start_stop else cm.CoxPH()
        args = (
            (design[idx], start[idx], stop[idx], event[idx])
            if start_stop
            else (design[idx], stop[idx], event[idx])
        )
        model.fit(*args)
        return model.summary()

    a = fit(x, np.arange(len(x)))
    b = fit(x + np.array([1000, -2000]), order)
    for key in ("coef", "vcov", "log_likelihood"):
        np.testing.assert_allclose(a[key], b[key], atol=1e-8, rtol=1e-8)


def test_matrix_completion_trace_matches_returned_fit():
    rng = np.random.default_rng(41)
    y = rng.normal(size=(10, 16))
    w = np.zeros_like(y)
    w[7:, 10:] = 1
    model = cm.MatrixCompletion(
        lambda_l=0.05, max_iterations=1, fit_unit_effects=False, fit_time_effects=False
    )
    model.fit(y, w)
    summary = model.summary()
    expected = np.sqrt(np.mean((y[w == 0] - model.predict()[w == 0]) ** 2))
    np.testing.assert_allclose(summary["history_rmse"][-1], expected)
    assert not summary["converged"]
    light = model.summary(include_matrices=False)
    assert light["att"] == summary["att"]
    for key in ("completed", "low_rank", "counterfactual", "treatment_effect"):
        assert key not in light
    assert not np.shares_memory(summary["completed"], summary["counterfactual"])


def test_finite_shortcuts_and_allocation_guards():
    with pytest.raises(ValueError, match="finite"):
        cm.SyntheticControl().fit(np.full((10, 1), np.nan), np.ones(10))
    with pytest.raises(ValueError, match="finite"):
        cm.BalancingWeights(autoscale=True).fit(
            np.full((10, 2), np.nan), np.ones((5, 2))
        )
    with pytest.raises(ValueError, match="finite"):
        cm.KernelBasis(bandwidth=np.nan).fit(np.ones((4, 2)))
    with pytest.raises(ValueError, match="512 MiB"):
        cm.KernelBasis().fit(np.ones((9000, 1)))
    with pytest.raises(ValueError, match="contiguous"):
        cm.ABCOLS().fit(
            np.arange(4.0),
            np.arange(4.0)[:, None],
            np.array([0, 1, 2, 2**32 - 1], dtype=np.uint32)[:, None],
        )


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n,p", [(120, 3), (30, 45)])
def test_ridge_grid_matches_independent_scalar_folds(weighted, n, p):
    x, y = data(n, p)
    x += np.linspace(-3, 5, p)
    weights = np.linspace(0.2, 1.5, n)
    penalties = [0.1, 1.0, 10.0]
    model = cm.Ridge(penalty=penalties, cv=3)
    if weighted:
        model.fit_weighted(x, y, weights.tolist())
    else:
        model.fit(x, y)
    summary = model.summary()
    expected_mse = []
    for j, penalty in enumerate(penalties):
        scalar = cm.Ridge(penalty=penalty)
        scalar.fit_weighted(x, y, weights.tolist()) if weighted else scalar.fit(x, y)
        np.testing.assert_allclose(
            summary["coef_path"][:, j], scalar.summary()["coef"], atol=1e-8
        )
        np.testing.assert_allclose(
            summary["intercept_path"][j], scalar.summary()["intercept"], atol=1e-8
        )
        fold_mse = []
        for fold in range(3):
            test = np.arange(n) % 3 == fold
            if weighted:
                scalar.fit_weighted(x[~test], y[~test], weights[~test].tolist())
            else:
                scalar.fit(x[~test], y[~test])
            fold_mse.append(
                np.average(
                    (y[test] - scalar.predict(x[test])) ** 2,
                    weights=weights[test] if weighted else None,
                )
            )
        expected_mse.append(np.mean(fold_mse))
    np.testing.assert_allclose(summary["cv_mse"], expected_mse, rtol=1e-8)
    assert summary["best_penalty_index"] == int(np.argmin(expected_mse))


@pytest.mark.parametrize(
    "cls", [cm.EPLM, cm.AverageDerivative, cm.PartiallyLinearDML, cm.AIPW]
)
def test_semiparametric_failed_refit_clears_state(cls):
    x, y = data(n=200)
    d = (np.arange(len(y)) % 2).astype(float)
    model = cls()
    model.fit(y, d, x)
    with pytest.raises(ValueError):
        model.fit(y[:-1], d, x)
    with pytest.raises(ValueError, match="fitted"):
        model.summary()


def test_cox_releases_gil_for_owned_computation():
    import threading
    import time

    rng = np.random.default_rng(49)
    x = rng.normal(size=(200000, 4))
    duration = rng.exponential(np.exp(-0.4 * x[:, 0])) + 0.01
    event = rng.binomial(1, 0.8, len(x)).astype(float)
    model = cm.CoxPH()
    observed = []
    started = time.perf_counter()
    timer = threading.Timer(
        0.01, lambda: observed.append(time.perf_counter() - started)
    )
    timer.start()
    try:
        model.fit(x, duration, event)
        elapsed = time.perf_counter() - started
    finally:
        timer.join()
    if elapsed < 0.04:
        pytest.skip("fit too short for the timer-based responsiveness check")
    assert observed[0] < elapsed * 0.8
