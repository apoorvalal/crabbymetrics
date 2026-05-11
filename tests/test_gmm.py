import numpy as np

import crabbymetrics as cm


def efficient_linear_gmm(x, y, z, ridge=1e-8):
    zx = z.T @ x
    zy = z.T @ y
    beta_first = np.linalg.solve(zx.T @ zx, zx.T @ zy)
    resid_first = y - x @ beta_first
    gi_first = z * resid_first[:, None]
    omega = gi_first.T @ gi_first / x.shape[0]
    weight = np.linalg.inv(omega + ridge * np.eye(omega.shape[0]))
    beta_second = np.linalg.solve(zx.T @ weight @ zx, zx.T @ weight @ zy)
    return beta_second, weight


def omega_cluster(gi, clusters):
    n, m = gi.shape
    omega = np.zeros((m, m))
    for cluster in np.unique(clusters):
        summed = gi[clusters == cluster].sum(axis=0)
        omega += np.outer(summed, summed)
    return omega / n


def omega_newey_west(gi, lags):
    n = gi.shape[0]
    omega = gi.T @ gi / n
    max_lag = min(lags, n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = gi[lag:].T @ gi[:-lag] / n
        omega = omega + weight * (gamma + gamma.T)
    return omega


def stacked_moments(theta, data):
    mu, alpha, beta = theta
    x = data["x"]
    y = data["y"]
    centered = x - mu
    resid = y - alpha - beta * centered
    return np.column_stack([x - mu, resid, centered * resid])


def stacked_jacobian(theta, data):
    mu, alpha, beta = theta
    x = data["x"]
    y = data["y"]
    centered = x - mu
    resid = y - alpha - beta * centered

    out = np.zeros((3, 3))
    out[0, 0] = -1.0
    out[1, 0] = beta
    out[1, 1] = -1.0
    out[1, 2] = -centered.mean()
    out[2, 0] = (-resid + beta * centered).mean()
    out[2, 1] = -centered.mean()
    out[2, 2] = -(centered**2).mean()
    return out


def test_gmm_just_identified_poisson_score_matches_builtin_poisson():
    rng = np.random.default_rng(6060)
    n = 900
    intercept_true = 0.1
    beta_true = np.array([0.25, -0.2])

    x_raw = rng.normal(size=(n, beta_true.size))
    x = np.column_stack([np.ones(n), x_raw])
    theta_true = np.concatenate([[intercept_true], beta_true])
    mu = np.exp(np.clip(x @ theta_true, -20.0, 20.0))
    y = rng.poisson(mu).astype(float)

    def poisson_score_moments(theta, data):
        design = data["x"]
        outcome = data["y"]
        mean = np.exp(np.clip(design @ theta, -20.0, 20.0))
        return design * (outcome - mean)[:, None]

    gmm = cm.GMM(poisson_score_moments, max_iterations=200, tolerance=1e-8, fd_eps=1e-5)
    gmm.fit({"x": x, "y": y}, np.zeros(x.shape[1]))
    gmm_summary = gmm.summary()

    poisson = cm.Poisson(alpha=0.0, max_iterations=200, tolerance=1e-8)
    poisson.fit(x_raw, y)
    poisson_summary = poisson.summary()
    theta_poisson = np.concatenate([[poisson_summary["intercept"]], poisson_summary["coef"]])

    np.testing.assert_allclose(gmm_summary["coef"], theta_poisson, atol=2e-4, rtol=0.0)
    assert gmm_summary["weighting"] == "identity"
    assert gmm_summary["j_stat"] is None
    assert gmm_summary["j_df"] is None


def test_gmm_two_step_linear_iv_matches_closed_form_efficient_gmm():
    rng = np.random.default_rng(20260322)
    n = 3000

    beta_true = np.array([1.25, -0.85])
    z = rng.normal(size=(n, 5))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    pi = np.array(
        [
            [0.9, 0.1],
            [0.5, -0.3],
            [-0.4, 0.8],
            [0.3, 0.2],
            [-0.2, 0.5],
        ]
    )

    x = z @ pi + v
    u = 0.7 * v[:, 0] - 0.55 * v[:, 1] + 0.35 * eps
    y = x @ beta_true + u

    def iv_moments(theta, data):
        design = data["x"]
        outcome = data["y"]
        instruments = data["z"]
        resid = outcome - design @ theta
        return instruments * resid[:, None]

    def iv_jacobian(theta, data):
        del theta
        design = data["x"]
        instruments = data["z"]
        return -(instruments.T @ design) / design.shape[0]

    gmm = cm.GMM(iv_moments, jacobian_fn=iv_jacobian, max_iterations=200, tolerance=1e-10)
    gmm.fit({"x": x, "y": y, "z": z}, np.zeros(x.shape[1]))
    summary = gmm.summary()

    beta_closed, weight = efficient_linear_gmm(x, y, z)

    np.testing.assert_allclose(summary["coef"], beta_closed, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(summary["weight_matrix"], weight, atol=1e-6, rtol=0.0)
    assert summary["weighting"] == "two_step"
    assert summary["j_df"] == z.shape[1] - x.shape[1]
    assert summary["j_stat"] >= 0.0


def test_gmm_accepts_stacked_moments_with_estimated_centering():
    rng = np.random.default_rng(5151)
    n = 1200
    x = rng.normal(loc=1.5, scale=1.1, size=n)
    centered_true = x - x.mean()
    alpha_true = 2.0
    beta_true = -0.65
    y = alpha_true + beta_true * centered_true + rng.normal(scale=0.4, size=n)

    model = cm.GMM(
        stacked_moments,
        jacobian_fn=stacked_jacobian,
        max_iterations=200,
        tolerance=1e-10,
    )
    model.fit({"x": x, "y": y}, np.zeros(3))
    summary = model.summary()

    centered = x - x.mean()
    beta_ols = np.linalg.lstsq(
        np.column_stack([np.ones(n), centered]),
        y,
        rcond=None,
    )[0]
    theta_target = np.array([x.mean(), beta_ols[0], beta_ols[1]])

    np.testing.assert_allclose(summary["coef"], theta_target, atol=1e-8, rtol=0.0)
    assert summary["weighting"] == "identity"
    assert summary["n_moments"] == 3


def test_gmm_summary_supports_vanilla_newey_west_and_cluster_covariances():
    rng = np.random.default_rng(9191)
    n = 1000
    x = rng.normal(loc=0.5, scale=1.2, size=n)
    centered = x - x.mean()
    y = 1.5 + 0.7 * centered + rng.normal(scale=0.6, size=n)
    clusters = np.repeat(np.arange(50, dtype=np.int64), n // 50)

    model = cm.GMM(
        stacked_moments,
        jacobian_fn=stacked_jacobian,
        max_iterations=200,
        tolerance=1e-10,
    )
    model.fit({"x": x, "y": y}, np.zeros(3), weighting="identity")

    vanilla = model.summary(vcov="vanilla")
    cluster = model.summary(vcov="sandwich", omega="cluster", clusters=clusters)
    hac = model.summary(vcov="sandwich", omega="newey_west", lags=4)

    theta_hat = np.asarray(vanilla["coef"])
    gi = stacked_moments(theta_hat, {"x": x, "y": y})
    jacobian = stacked_jacobian(theta_hat, {"x": x, "y": y})
    weight = np.asarray(vanilla["weight_matrix"])
    nobs = gi.shape[0]
    a_inv = np.linalg.inv(jacobian.T @ weight @ jacobian)

    vanilla_manual = a_inv / nobs
    cluster_omega = omega_cluster(gi, clusters)
    cluster_manual = (
        a_inv
        @ (jacobian.T @ weight @ cluster_omega @ weight @ jacobian)
        @ a_inv
        / nobs
    )
    hac_omega = omega_newey_west(gi, lags=4)
    hac_manual = (
        a_inv @ (jacobian.T @ weight @ hac_omega @ weight @ jacobian) @ a_inv / nobs
    )

    np.testing.assert_allclose(vanilla["vcov"], vanilla_manual, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(cluster["vcov"], cluster_manual, atol=1e-10, rtol=0.0)
    np.testing.assert_allclose(hac["vcov"], hac_manual, atol=1e-10, rtol=0.0)


def test_gmm_fit_sketch_tracks_many_moment_linear_iv():
    rng = np.random.default_rng(20260509)
    n = 2500
    p = 3
    m = 40
    beta_true = np.array([0.7, -0.4, 1.1])
    z_base = rng.normal(size=(n, p))
    z_extra = rng.normal(size=(n, m - p))
    z = np.column_stack([z_base, z_extra])
    x = z_base + 0.2 * rng.normal(size=(n, p))
    y = x @ beta_true + rng.normal(scale=0.4, size=n)

    def iv_moments(theta, data):
        resid = data["y"] - data["x"] @ theta
        return data["z"] * resid[:, None]

    def iv_jacobian(theta, data):
        del theta
        return -(data["z"].T @ data["x"]) / data["x"].shape[0]

    full = cm.GMM(iv_moments, jacobian_fn=iv_jacobian, max_iterations=200, tolerance=1e-10)
    full.fit({"x": x, "y": y, "z": z}, np.zeros(p), weighting="identity")

    sketch = cm.GMM(iv_moments, jacobian_fn=iv_jacobian, max_iterations=200, tolerance=1e-10)
    sketch.fit_sketch(
        {"x": x, "y": y, "z": z},
        np.zeros(p),
        sketch_size=16,
        weighting="identity",
        seed=33,
    )
    full_summary = full.summary(vcov="vanilla")
    sketch_summary = sketch.summary(vcov="vanilla")

    assert sketch_summary["n_moments"] == 16
    assert sketch_summary["original_n_moments"] == m
    assert sketch_summary["sketch_size"] == 16
    np.testing.assert_allclose(sketch_summary["coef"], full_summary["coef"], atol=0.08, rtol=0.0)


def test_gmm_fit_sketch_rejects_too_small_sketch():
    def moments(theta, data):
        del data
        return np.ones((5, 3)) * theta[0]

    model = cm.GMM(moments)
    with np.testing.assert_raises(ValueError):
        model.fit_sketch({}, np.zeros(2), sketch_size=1)
