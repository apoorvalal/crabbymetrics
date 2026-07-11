from pathlib import Path

from statsmodels.sandbox.regression.gmm import IV2SLS

import numpy as np
import pytest

import crabbymetrics as cm


def score_cov_newey_west(scores, lags):
    cov = scores.T @ scores
    if scores.shape[0] <= 1 or lags == 0:
        return cov

    max_lag = min(lags, scores.shape[0] - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = scores[lag:].T @ scores[:-lag]
        cov = cov + weight * (gamma + gamma.T)
    return cov


def score_cov_cluster(scores, clusters):
    cov = np.zeros((scores.shape[1], scores.shape[1]))
    for cluster in np.unique(clusters):
        summed = scores[clusters == cluster].sum(axis=0)
        cov = cov + np.outer(summed, summed)
    return cov


def sandwich_from_parameter_scores(scores, df_resid, kind, lags=None, clusters=None):
    n = scores.shape[0]
    if kind == "hc1":
        return score_cov_newey_west(scores, lags=0) * (n / df_resid)
    if kind == "newey_west":
        return score_cov_newey_west(scores, lags=lags) * (n / df_resid)
    if kind == "cluster":
        n_clusters = np.unique(clusters).size
        scale = (n_clusters / (n_clusters - 1.0)) * ((n - 1.0) / df_resid)
        return score_cov_cluster(scores, clusters) * scale
    raise ValueError(f"unsupported kind: {kind}")


def hc1_se(x, y):
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    xtx_inv = np.linalg.inv(design.T @ design)
    meat = design.T @ ((resid[:, None] ** 2) * design)
    scale = x.shape[0] / (x.shape[0] - design.shape[1])
    cov = scale * xtx_inv @ meat @ xtx_inv
    return beta, np.sqrt(np.diag(cov))


def vanilla_ols_se(x, y):
    design = np.column_stack([np.ones(x.shape[0]), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    xtx_inv = np.linalg.inv(design.T @ design)
    sigma2 = (resid @ resid) / (design.shape[0] - design.shape[1])
    cov = sigma2 * xtx_inv
    return beta, np.sqrt(np.diag(cov))


def weighted_ols_summary_reference(x, y, weights, kind="vanilla"):
    design = np.column_stack([np.ones(x.shape[0]), x])
    sqrt_w = np.sqrt(weights)
    design_w = design * sqrt_w[:, None]
    y_w = y * sqrt_w
    beta, *_ = np.linalg.lstsq(design_w, y_w, rcond=None)
    resid = y - design @ beta
    resid_w = sqrt_w * resid
    xtx_inv = np.linalg.inv(design_w.T @ design_w)

    if kind == "vanilla":
        sigma2 = (resid_w @ resid_w) / (design.shape[0] - design.shape[1])
        cov = sigma2 * xtx_inv
    elif kind == "hc1":
        param_scores = (design_w * resid_w[:, None]) @ xtx_inv
        cov = sandwich_from_parameter_scores(param_scores, design.shape[0] - design.shape[1], "hc1")
    else:
        raise ValueError(f"unsupported kind: {kind}")

    return beta, cov


def ridge_augmented_solution(x, y, penalty):
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty_block = np.sqrt(penalty) * np.eye(x.shape[1] + 1)
    penalty_block[0, 0] = 0.0
    aug_x = np.vstack([design, penalty_block])
    aug_y = np.concatenate([y, np.zeros(x.shape[1] + 1)])
    beta, *_ = np.linalg.lstsq(aug_x, aug_y, rcond=None)
    return beta[0], beta[1:]


def weighted_ridge_augmented_solution(x, y, penalty, weights):
    design = np.column_stack([np.ones(x.shape[0]), x])
    sqrt_w = np.sqrt(weights)
    penalty_block = np.sqrt(penalty) * np.eye(x.shape[1] + 1)
    penalty_block[0, 0] = 0.0
    aug_x = np.vstack([design * sqrt_w[:, None], penalty_block])
    aug_y = np.concatenate([y * sqrt_w, np.zeros(x.shape[1] + 1)])
    beta, *_ = np.linalg.lstsq(aug_x, aug_y, rcond=None)
    return beta[0], beta[1:]


def ridge_cv_curve(x, y, penalties, cv):
    n = x.shape[0]
    n_folds = min(cv, n)
    fold_id = np.arange(n) % n_folds
    curve = np.zeros(len(penalties))

    for j, penalty in enumerate(penalties):
        fold_mse = 0.0
        for fold in range(n_folds):
            train = fold_id != fold
            test = ~train
            intercept, coef = ridge_augmented_solution(x[train], y[train], penalty)
            residual = y[test] - (intercept + x[test] @ coef)
            fold_mse += np.mean(residual**2)
        curve[j] = fold_mse / n_folds

    return curve


def weighted_ridge_cv_curve(x, y, penalties, cv, weights):
    n = x.shape[0]
    n_folds = min(cv, n)
    fold_id = np.arange(n) % n_folds
    curve = np.zeros(len(penalties))

    for j, penalty in enumerate(penalties):
        fold_mse = 0.0
        for fold in range(n_folds):
            train = fold_id != fold
            test = ~train
            intercept, coef = weighted_ridge_augmented_solution(x[train], y[train], penalty, weights[train])
            residual = y[test] - (intercept + x[test] @ coef)
            fold_mse += np.sum(weights[test] * residual**2) / np.sum(weights[test])
        curve[j] = fold_mse / n_folds

    return curve


def ridge_summary_reference(x, y, penalty, kind, lags=None, clusters=None):
    design = np.column_stack([np.ones(x.shape[0]), x])
    intercept, coef = ridge_augmented_solution(x, y, penalty)
    beta = np.concatenate([[intercept], coef])
    fitted = design @ beta
    resid = y - fitted
    penalty_matrix = penalty * np.eye(design.shape[1])
    penalty_matrix[0, 0] = 0.0
    bread_inv = np.linalg.inv(design.T @ design + penalty_matrix)
    df_eff = np.trace(design.T @ design @ bread_inv)
    if kind == "vanilla":
        sigma2 = (resid @ resid) / (design.shape[0] - df_eff)
        cov = bread_inv @ (design.T @ design) @ bread_inv * sigma2
    else:
        raw_scores = design * resid[:, None]
        param_scores = raw_scores @ bread_inv
        cov = sandwich_from_parameter_scores(
            param_scores,
            df_resid=design.shape[0] - df_eff,
            kind=kind,
            lags=lags,
            clusters=clusters,
        )
    return beta, cov


def poisson_covariances(x, y, intercept, coef):
    design = np.column_stack([np.ones(x.shape[0]), x])
    mu = np.exp(intercept + x @ coef)
    fisher = np.linalg.inv(design.T @ (mu[:, None] * design))
    scores = design * (y - mu)[:, None]
    qmle = fisher @ (scores.T @ scores) @ fisher
    return fisher, qmle


def twosls_closed_form(x_endog, x_exog, z, y):
    if x_exog.shape[1] > 0:
        x_rhs = np.column_stack([x_endog, x_exog])
        z_rhs = np.column_stack([x_exog, z])
    else:
        x_rhs = x_endog
        z_rhs = z

    x_design = np.column_stack([np.ones(x_rhs.shape[0]), x_rhs])
    z_design = np.column_stack([np.ones(z_rhs.shape[0]), z_rhs])
    ztz_inv = np.linalg.inv(z_design.T @ z_design)
    beta = np.linalg.solve(
        x_design.T @ z_design @ ztz_inv @ z_design.T @ x_design,
        x_design.T @ z_design @ ztz_inv @ z_design.T @ y,
    )
    return beta[0], beta[1:]


def twosls_summary_reference(x_endog, x_exog, z, y, kind, lags=None, clusters=None):
    if x_exog.shape[1] > 0:
        x_rhs = np.column_stack([x_endog, x_exog])
        z_rhs = np.column_stack([x_exog, z])
    else:
        x_rhs = x_endog
        z_rhs = z

    x_design = np.column_stack([np.ones(x_rhs.shape[0]), x_rhs])
    z_design = np.column_stack([np.ones(z_rhs.shape[0]), z_rhs])
    pi_hat, *_ = np.linalg.lstsq(z_design, x_endog, rcond=None)
    x_endog_hat = z_design @ pi_hat
    if x_exog.shape[1] > 0:
        x_hat_rhs = np.column_stack([x_endog_hat, x_exog])
    else:
        x_hat_rhs = x_endog_hat
    x_hat_design = np.column_stack([np.ones(x_hat_rhs.shape[0]), x_hat_rhs])

    beta, *_ = np.linalg.lstsq(x_hat_design, y, rcond=None)
    residuals = y - x_design @ beta
    n, p = x_design.shape
    weight = np.linalg.inv(z_design.T @ z_design / n)
    jacobian = -(z_design.T @ x_design) / n
    a_inv = np.linalg.inv(jacobian.T @ weight @ jacobian)

    if kind == "vanilla":
        sigma2 = (residuals @ residuals) / (n - p)
        cov = a_inv * sigma2 / n
    else:
        moment_scores = z_design * residuals[:, None]
        transform = weight @ jacobian @ a_inv / n
        param_scores = moment_scores @ transform
        cov = sandwich_from_parameter_scores(
            param_scores,
            df_resid=n - p,
            kind=kind,
            lags=lags,
            clusters=clusters,
        )
    return beta, cov


def weighted_twosls_closed_form(x_endog, x_exog, z, y, weights):
    x_rhs = np.column_stack([x_endog, x_exog])
    z_rhs = np.column_stack([x_exog, z])
    x_design = np.column_stack([np.ones(x_rhs.shape[0]), x_rhs])
    z_design = np.column_stack([np.ones(z_rhs.shape[0]), z_rhs])
    sqrt_w = np.sqrt(weights)
    fitted = IV2SLS(
        y * sqrt_w,
        x_design * sqrt_w[:, None],
        z_design * sqrt_w[:, None],
    ).fit()
    return fitted.params[0], fitted.params[1:]


def weighted_twosls_summary_reference(x_endog, x_exog, z, y, weights, kind, lags=None, clusters=None):
    x_rhs = np.column_stack([x_endog, x_exog])
    z_rhs = np.column_stack([x_exog, z])
    x_design = np.column_stack([np.ones(x_rhs.shape[0]), x_rhs])
    z_design = np.column_stack([np.ones(z_rhs.shape[0]), z_rhs])
    sqrt_w = np.sqrt(weights)
    x_work = x_design * sqrt_w[:, None]
    z_work = z_design * sqrt_w[:, None]
    y_work = y * sqrt_w
    beta = IV2SLS(y_work, x_work, z_work).fit().params
    residuals = y_work - x_work @ beta
    n, p = x_work.shape
    weight = np.linalg.inv(z_work.T @ z_work / n)
    jacobian = -(z_work.T @ x_work) / n
    a_inv = np.linalg.inv(jacobian.T @ weight @ jacobian)
    if kind == "vanilla":
        sigma2 = residuals @ residuals / (n - p)
        cov = a_inv * sigma2 / n
    else:
        moment_scores = z_work * residuals[:, None]
        transform = weight @ jacobian @ a_inv / n
        param_scores = moment_scores @ transform
        cov = sandwich_from_parameter_scores(
            param_scores,
            df_resid=n - p,
            kind=kind,
            lags=lags,
            clusters=clusters,
        )
    return beta, cov


def demean_by_group(x, y, groups):
    x_resid = np.empty_like(x, dtype=float)
    y_resid = np.empty_like(y, dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        x_resid[mask] = x[mask] - x[mask].mean(axis=0, keepdims=True)
        y_resid[mask] = y[mask] - y[mask].mean()
    return x_resid, y_resid


def weighted_demean_by_group(x, y, groups, weights):
    x_resid = np.empty_like(x, dtype=float)
    y_resid = np.empty_like(y, dtype=float)
    for group in np.unique(groups):
        mask = groups == group
        group_weights = weights[mask]
        x_mean = np.average(x[mask], axis=0, weights=group_weights)
        y_mean = np.average(y[mask], weights=group_weights)
        x_resid[mask] = x[mask] - x_mean
        y_resid[mask] = y[mask] - y_mean
    return x_resid, y_resid


def test_ols_matches_closed_form_hc1_and_predict_round_trip():
    rng = np.random.default_rng(2026)
    n = 800
    intercept_true = 1.75
    beta_true = np.array([0.8, -1.2, 0.35])

    x = rng.normal(size=(n, beta_true.size))
    y = intercept_true + x @ beta_true + rng.normal(scale=0.6, size=n)

    model = cm.OLS()
    model.fit(x, y)
    summary = model.summary()

    beta_hat, se_hat = hc1_se(x, y)
    pred_all = model.predict(x)
    pred_subset = model.predict(x[:57])

    np.testing.assert_allclose(summary["intercept"], beta_hat[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], beta_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["intercept_se"], se_hat[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef_se"], se_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pred_all, beta_hat[0] + x @ beta_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        pred_subset,
        beta_hat[0] + x[:57] @ beta_hat[1:],
        atol=1e-8,
        rtol=1e-8,
    )
    assert pred_all.shape == (n,)
    assert pred_subset.shape == (57,)
    assert model.bootstrap(5, seed=11).shape == (5, x.shape[1] + 1)


def test_ols_summary_supports_vanilla_and_hc1_vcov():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(700, 2))
    y = 0.3 + x @ np.array([1.1, -0.4]) + (0.5 + x[:, 0] ** 2) * rng.normal(size=700)

    model = cm.OLS()
    model.fit(x, y)

    vanilla = model.summary(vcov="vanilla")
    hc1 = model.summary(vcov="hc1")
    beta_vanilla, se_vanilla = vanilla_ols_se(x, y)
    beta_hc1, se_hc1 = hc1_se(x, y)

    np.testing.assert_allclose(vanilla["intercept"], beta_vanilla[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["coef"], beta_vanilla[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["intercept_se"], se_vanilla[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["coef_se"], se_vanilla[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["intercept"], beta_hc1[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["coef"], beta_hc1[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["intercept_se"], se_hc1[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(hc1["coef_se"], se_hc1[1:], atol=1e-8, rtol=1e-8)
    assert vanilla["vcov_type"] == "vanilla"
    assert hc1["vcov_type"] == "hc1"


def test_ols_summary_supports_newey_west_and_cluster_vcov():
    rng = np.random.default_rng(2027_1)
    n = 720
    x = rng.normal(size=(n, 3))
    clusters = np.repeat(np.arange(60, dtype=np.int64), n // 60)
    y = 0.25 + x @ np.array([0.8, -0.45, 0.3]) + (0.7 + 0.4 * x[:, 0] ** 2) * rng.normal(size=n)

    model = cm.OLS()
    model.fit(x, y)

    nw = model.summary(vcov="newey_west", lags=5)
    cluster = model.summary(vcov="cluster", clusters=clusters)

    design = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    param_scores = (design * resid[:, None]) @ np.linalg.inv(design.T @ design)
    cov_nw = sandwich_from_parameter_scores(param_scores, n - design.shape[1], "newey_west", lags=5)
    cov_cluster = sandwich_from_parameter_scores(
        param_scores,
        n - design.shape[1],
        "cluster",
        clusters=clusters,
    )

    np.testing.assert_allclose(nw["intercept"], beta[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef"], beta[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["intercept_se"], np.sqrt(cov_nw[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef_se"], np.sqrt(np.diag(cov_nw))[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        cluster["intercept_se"],
        np.sqrt(cov_cluster[0, 0]),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cluster["coef_se"],
        np.sqrt(np.diag(cov_cluster))[1:],
        atol=1e-8,
        rtol=1e-8,
    )
    assert nw["vcov_type"] == "newey_west"
    assert cluster["vcov_type"] == "cluster"


def test_ols_unit_sample_weights_match_unweighted_fit_and_summary():
    rng = np.random.default_rng(2027_2)
    x = rng.normal(size=(640, 3))
    y = 0.15 + x @ np.array([0.9, -0.35, 0.2]) + rng.normal(scale=0.55, size=640)
    unit_weights = np.ones(x.shape[0])

    baseline = cm.OLS()
    baseline.fit(x, y)

    weighted = cm.OLS()
    weighted.fit_weighted(x, y, unit_weights)

    baseline_summary = baseline.summary(vcov="hc1")
    weighted_summary = weighted.summary(vcov="hc1")

    np.testing.assert_allclose(weighted_summary["intercept"], baseline_summary["intercept"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["coef"], baseline_summary["coef"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["intercept_se"], baseline_summary["intercept_se"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["coef_se"], baseline_summary["coef_se"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted.predict(x[:31]), baseline.predict(x[:31]), atol=1e-8, rtol=1e-8)


def test_ols_sample_weights_match_weighted_least_squares():
    rng = np.random.default_rng(2027_3)
    x = rng.normal(size=(720, 2))
    weights = 0.4 + rng.random(x.shape[0]) * 1.8
    y = -0.1 + x @ np.array([1.0, -0.55]) + (0.6 + 0.3 * x[:, 0] ** 2) * rng.normal(size=x.shape[0])

    model = cm.OLS()
    model.fit_weighted(x, y, weights)
    summary = model.summary(vcov="vanilla")

    beta, cov = weighted_ols_summary_reference(x, y, weights, kind="vanilla")

    np.testing.assert_allclose(summary["intercept"], beta[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], beta[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["intercept_se"], np.sqrt(cov[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef_se"], np.sqrt(np.diag(cov))[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(model.predict(x[:27]), beta[0] + x[:27] @ beta[1:], atol=1e-8, rtol=1e-8)


def test_ridge_scalar_penalty_matches_augmented_least_squares():
    rng = np.random.default_rng(2028)
    x = rng.normal(size=(500, 3))
    y = 0.4 + x @ np.array([1.2, -0.8, 0.25]) + rng.normal(scale=0.7, size=500)
    penalty = 1.75

    model = cm.Ridge(penalty=penalty)
    model.fit(x, y)
    summary = model.summary(vcov="vanilla")
    intercept_hat, coef_hat = ridge_augmented_solution(x, y, penalty)

    np.testing.assert_allclose(summary["intercept"], intercept_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], coef_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(model.predict(x[:25]), intercept_hat + x[:25] @ coef_hat, atol=1e-8, rtol=1e-8)
    assert summary["penalty"] == penalty
    assert summary["best_penalty_index"] is None
    assert summary["coef_path"].shape == (x.shape[1], 1)
    assert model.best_penalty == penalty
    assert model.best_penalty_index is None
    assert model.bootstrap(4, seed=13).shape == (4, x.shape[1] + 1)


def test_ridge_zero_penalty_matches_ols_point_estimates():
    rng = np.random.default_rng(2029)
    x = rng.normal(size=(600, 2))
    y = -0.2 + x @ np.array([0.9, -1.1]) + rng.normal(scale=0.4, size=600)

    ols = cm.OLS()
    ols.fit(x, y)
    ridge = cm.Ridge(penalty=0.0)
    ridge.fit(x, y)

    ols_summary = ols.summary(vcov="vanilla")
    ridge_summary = ridge.summary(vcov="vanilla")

    np.testing.assert_allclose(ridge_summary["intercept"], ols_summary["intercept"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(ridge_summary["coef"], ols_summary["coef"], atol=1e-8, rtol=1e-8)


def test_ridge_summary_supports_newey_west_and_cluster_vcov():
    rng = np.random.default_rng(2029_1)
    n = 680
    x = rng.normal(size=(n, 4))
    penalty = 0.75
    clusters = np.repeat(np.arange(40, dtype=np.int64), n // 40)
    y = -0.15 + x @ np.array([0.9, -0.6, 0.25, 0.0]) + (0.5 + 0.3 * x[:, 1] ** 2) * rng.normal(size=n)

    model = cm.Ridge(penalty=penalty)
    model.fit(x, y)

    nw = model.summary(vcov="newey_west", lags=4)
    cluster = model.summary(vcov="cluster", clusters=clusters)
    beta_nw, cov_nw = ridge_summary_reference(x, y, penalty, kind="newey_west", lags=4)
    beta_cluster, cov_cluster = ridge_summary_reference(x, y, penalty, kind="cluster", clusters=clusters)

    np.testing.assert_allclose(nw["intercept"], beta_nw[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef"], beta_nw[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["intercept_se"], np.sqrt(cov_nw[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef_se"], np.sqrt(np.diag(cov_nw))[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(cluster["intercept"], beta_cluster[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(cluster["coef"], beta_cluster[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        cluster["intercept_se"],
        np.sqrt(cov_cluster[0, 0]),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cluster["coef_se"],
        np.sqrt(np.diag(cov_cluster))[1:],
        atol=1e-8,
        rtol=1e-8,
    )
    assert nw["vcov_type"] == "newey_west"
    assert cluster["vcov_type"] == "cluster"


def test_ridge_penalty_grid_returns_path_and_cv_optimal_index():
    rng = np.random.default_rng(2030)
    x = rng.normal(size=(450, 4))
    y = 0.6 + x @ np.array([1.0, -0.7, 0.0, 0.35]) + rng.normal(scale=0.9, size=450)
    penalties = np.array([0.0, 0.05, 0.2, 1.0, 4.0])

    model = cm.Ridge(penalty=penalties, cv=6)
    model.fit(x, y)
    summary = model.summary()
    cv_curve = ridge_cv_curve(x, y, penalties, cv=6)
    best_idx = int(np.argmin(cv_curve))

    np.testing.assert_allclose(summary["penalties"], penalties, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(summary["cv_mse"], cv_curve, atol=1e-8, rtol=1e-8)
    assert summary["coef_path"].shape == (x.shape[1], penalties.size)
    assert summary["intercept_path"].shape == (penalties.size,)
    assert summary["best_penalty_index"] == best_idx
    assert model.best_penalty_index == best_idx
    assert model.best_penalty == penalties[best_idx]
    np.testing.assert_allclose(summary["coef"], summary["coef_path"][:, best_idx], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["intercept"], summary["intercept_path"][best_idx], atol=1e-8, rtol=1e-8)


def test_ridge_cv_predict_matches_refit_at_selected_penalty():
    rng = np.random.default_rng(2031)
    x = rng.normal(size=(520, 5))
    y = -0.1 + x @ np.array([0.9, -0.5, 0.2, 0.0, 0.35]) + rng.normal(scale=0.75, size=520)
    penalties = np.logspace(-3, 2, 20)

    cv_model = cm.Ridge(penalty=penalties, cv=5)
    cv_model.fit(x, y)
    selected_penalty = cv_model.best_penalty

    refit_model = cm.Ridge(penalty=float(selected_penalty))
    refit_model.fit(x, y)

    x_new = rng.normal(size=(37, x.shape[1]))
    np.testing.assert_allclose(
        cv_model.predict(x_new),
        refit_model.predict(x_new),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cv_model.summary()["coef"],
        refit_model.summary()["coef"],
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cv_model.summary()["intercept"],
        refit_model.summary()["intercept"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_ridge_unit_sample_weights_match_unweighted_cv_fit():
    rng = np.random.default_rng(2031_1)
    x = rng.normal(size=(540, 4))
    y = 0.25 + x @ np.array([0.8, -0.45, 0.2, 0.1]) + rng.normal(scale=0.7, size=540)
    penalties = np.logspace(-2, 1.5, 18)
    unit_weights = np.ones(x.shape[0])

    baseline = cm.Ridge(penalty=penalties, cv=5)
    baseline.fit(x, y)

    weighted = cm.Ridge(penalty=penalties, cv=5)
    weighted.fit_weighted(x, y, unit_weights)

    baseline_summary = baseline.summary(vcov="vanilla")
    weighted_summary = weighted.summary(vcov="vanilla")

    np.testing.assert_allclose(weighted_summary["coef"], baseline_summary["coef"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["intercept"], baseline_summary["intercept"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["cv_mse"], baseline_summary["cv_mse"], atol=1e-8, rtol=1e-8)
    assert weighted.best_penalty_index == baseline.best_penalty_index
    np.testing.assert_allclose(weighted.predict(x[:29]), baseline.predict(x[:29]), atol=1e-8, rtol=1e-8)


def test_ridge_sample_weights_match_weighted_cv_curve_and_selected_fit():
    rng = np.random.default_rng(2031_2)
    x = rng.normal(size=(560, 4))
    y = -0.2 + x @ np.array([1.1, -0.7, 0.15, 0.0]) + rng.normal(scale=0.8, size=560)
    weights = 0.3 + rng.random(x.shape[0]) * 2.2
    penalties = np.logspace(-3, 2, 15)

    model = cm.Ridge(penalty=penalties, cv=6)
    model.fit_weighted(x, y, weights)
    summary = model.summary(vcov="vanilla")

    cv_curve = weighted_ridge_cv_curve(x, y, penalties, cv=6, weights=weights)
    best_idx = int(np.argmin(cv_curve))
    intercept_hat, coef_hat = weighted_ridge_augmented_solution(x, y, penalties[best_idx], weights)

    np.testing.assert_allclose(summary["cv_mse"], cv_curve, atol=1e-8, rtol=1e-8)
    assert summary["best_penalty_index"] == best_idx
    assert model.best_penalty_index == best_idx
    np.testing.assert_allclose(summary["intercept"], intercept_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], coef_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], summary["coef_path"][:, best_idx], atol=1e-8, rtol=1e-8)


def test_fixed_effects_ols_is_invariant_to_fixed_effect_relabeling():
    rng = np.random.default_rng(31415)
    n = 600
    beta_true = np.array([1.1, -0.45])

    worker = rng.integers(0, 30, size=n, dtype=np.uint32)
    firm = rng.integers(0, 20, size=n, dtype=np.uint32)
    fe = np.column_stack([worker, firm]).astype(np.uint32)
    fe_relabelled = np.column_stack([worker + 100, firm + 500]).astype(np.uint32)

    x = rng.normal(size=(n, beta_true.size))
    y = (
        x @ beta_true
        + rng.normal(scale=0.8, size=30)[worker]
        + rng.normal(scale=0.5, size=20)[firm]
        + rng.normal(scale=0.1, size=n)
    )

    model_a = cm.FixedEffectsOLS()
    model_a.fit(x, fe, y)

    model_b = cm.FixedEffectsOLS()
    model_b.fit(x, fe_relabelled, y)

    np.testing.assert_allclose(
        model_a.summary()["coef"],
        model_b.summary()["coef"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_fixed_effects_ols_summary_supports_newey_west_and_cluster_vcov():
    rng = np.random.default_rng(31416)
    n = 900
    groups = np.repeat(np.arange(75, dtype=np.uint32), n // 75)
    x = rng.normal(size=(n, 2))
    beta_true = np.array([0.75, -0.35])
    group_effect = rng.normal(scale=0.8, size=groups.max() + 1)
    y = x @ beta_true + group_effect[groups] + (0.4 + 0.2 * x[:, 0] ** 2) * rng.normal(size=n)

    model = cm.FixedEffectsOLS()
    model.fit(x, groups[:, None], y)

    nw = model.summary(vcov="newey_west", lags=3)
    cluster = model.summary(vcov="cluster", clusters=groups.astype(np.int64))

    x_resid, y_resid = demean_by_group(x, y, groups)
    coef, *_ = np.linalg.lstsq(x_resid, y_resid, rcond=None)
    resid = y_resid - x_resid @ coef
    param_scores = (x_resid * resid[:, None]) @ np.linalg.inv(x_resid.T @ x_resid)
    residual_df = n - x.shape[1] - np.unique(groups).size
    cov_nw = sandwich_from_parameter_scores(param_scores, residual_df, "newey_west", lags=3)
    cov_cluster = sandwich_from_parameter_scores(
        param_scores,
        residual_df,
        "cluster",
        clusters=groups.astype(np.int64),
    )

    np.testing.assert_allclose(nw["coef"], coef, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef_se"], np.sqrt(np.diag(cov_nw)), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(cluster["coef"], coef, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        cluster["coef_se"],
        np.sqrt(np.diag(cov_cluster)),
        atol=1e-8,
        rtol=1e-8,
    )
    assert nw["vcov_type"] == "newey_west"
    assert cluster["vcov_type"] == "cluster"
    assert nw["residual_df"] == residual_df
    assert nw["absorbed_df"] == np.unique(groups).size


def test_fixed_effects_ols_unit_sample_weights_match_unweighted_fit():
    rng = np.random.default_rng(31417)
    n = 720
    groups = rng.integers(0, 45, size=n, dtype=np.uint32)
    x = rng.normal(size=(n, 2))
    y = x @ np.array([0.85, -0.25]) + rng.normal(scale=0.7, size=45)[groups] + rng.normal(scale=0.15, size=n)
    unit_weights = np.ones(n)

    baseline = cm.FixedEffectsOLS()
    baseline.fit(x, groups[:, None], y)

    weighted = cm.FixedEffectsOLS()
    weighted.fit_weighted(x, groups[:, None], y, unit_weights)

    np.testing.assert_allclose(weighted.summary()["coef"], baseline.summary()["coef"], atol=1e-8, rtol=1e-8)


def test_fixed_effects_ols_sample_weights_match_weighted_one_way_within():
    rng = np.random.default_rng(31418)
    n = 840
    groups = rng.integers(0, 35, size=n, dtype=np.uint32)
    x = rng.normal(size=(n, 2))
    weights = 0.5 + rng.random(n) * 1.5
    alpha = rng.normal(scale=1.2, size=35)
    y = x @ np.array([1.1, -0.6]) + alpha[groups] + rng.normal(scale=0.25, size=n)

    model = cm.FixedEffectsOLS()
    model.fit_weighted(x, groups[:, None], y, weights)
    summary = model.summary(vcov="hc1")

    x_resid, y_resid = weighted_demean_by_group(x, y, groups, weights)
    sqrt_w = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(x_resid * sqrt_w[:, None], y_resid * sqrt_w, rcond=None)

    np.testing.assert_allclose(summary["coef"], coef, atol=1e-6, rtol=1e-6)


def test_poisson_predict_round_trip_and_satisfies_score_conditions():
    rng = np.random.default_rng(9090)
    n = 900
    intercept_true = 0.2
    beta_true = np.array([0.3, -0.25, 0.15])

    x = rng.normal(size=(n, beta_true.size))
    mu_true = np.exp(intercept_true + x @ beta_true)
    y = rng.poisson(mu_true).astype(float)

    model = cm.Poisson(max_iterations=200, tolerance=1e-8)
    model.fit(x, y)
    summary = model.summary()

    pred_all = model.predict(x)
    pred_subset = model.predict(x[:41])
    residual = pred_all - y

    assert summary["coef"].shape == (x.shape[1],)
    assert summary["coef_se"].shape == (x.shape[1],)
    assert np.isfinite(summary["intercept"])
    assert np.isfinite(summary["intercept_se"])
    assert np.all(pred_all > 0.0)
    np.testing.assert_allclose(pred_subset, pred_all[:41], atol=1e-10, rtol=1e-10)
    assert model.bootstrap(4, seed=19).shape == (4, x.shape[1] + 1)

    assert abs(residual.sum()) < 1e-3
    np.testing.assert_allclose(
        x.T @ residual,
        np.zeros(x.shape[1]),
        atol=1e-3,
        rtol=0.0,
    )


def test_poisson_summary_supports_vanilla_and_qmle_vcov():
    rng = np.random.default_rng(9091)
    n = 800
    intercept_true = -0.1
    beta_true = np.array([0.45, -0.2])

    x = rng.normal(size=(n, beta_true.size))
    mu = np.exp(intercept_true + x @ beta_true)
    mixing = rng.gamma(shape=2.0, scale=0.5, size=n)
    y = rng.poisson(mu * mixing).astype(float)

    model = cm.Poisson(alpha=0.0, max_iterations=200, tolerance=1e-8)
    model.fit(x, y)

    vanilla = model.summary(vcov="vanilla")
    qmle = model.summary(vcov="sandwich")
    fisher, sandwich = poisson_covariances(x, y, vanilla["intercept"], vanilla["coef"])

    np.testing.assert_allclose(vanilla["intercept_se"], np.sqrt(fisher[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vanilla["coef_se"], np.sqrt(np.diag(fisher))[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(qmle["intercept_se"], np.sqrt(sandwich[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(qmle["coef_se"], np.sqrt(np.diag(sandwich))[1:], atol=1e-8, rtol=1e-8)
    assert vanilla["vcov_type"] == "vanilla"
    assert qmle["vcov_type"] == "sandwich"


def test_twosls_matches_closed_form_with_multiple_endogenous_regressors():
    rng = np.random.default_rng(4242)
    n = 1200
    intercept_true = 0.4
    beta_endog = np.array([1.15, -0.9])
    beta_exog = np.array([0.55])

    z = rng.normal(size=(n, 2))
    x_exog = rng.normal(size=(n, 1))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    pi = np.array([[0.9, 0.2], [-0.35, 0.8]])

    x_endog = z @ pi + 0.3 * x_exog + v
    u = 0.6 * v[:, 0] - 0.45 * v[:, 1] + 0.25 * eps
    y = intercept_true + x_endog @ beta_endog + x_exog[:, 0] * beta_exog[0] + u

    model = cm.TwoSLS()
    model.fit(x_endog, x_exog, z, y)
    summary = model.summary()
    intercept_hat, coef_hat = twosls_closed_form(x_endog, x_exog, z, y)

    pred = model.predict(np.column_stack([x_endog, x_exog]))

    np.testing.assert_allclose(summary["intercept"], intercept_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], coef_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(pred, intercept_hat + np.column_stack([x_endog, x_exog]) @ coef_hat)
    assert summary["coef"].shape == (x_endog.shape[1] + x_exog.shape[1],)
    assert summary["coef_se"].shape == (x_endog.shape[1] + x_exog.shape[1],)
    assert np.isfinite(summary["intercept_se"])
    assert model.bootstrap(4, seed=7).shape == (4, x_endog.shape[1] + x_exog.shape[1] + 1)


def test_twosls_overidentified_matches_closed_form_formula():
    rng = np.random.default_rng(1717)
    n = 1400

    z = rng.normal(size=(n, 4))
    x_exog = rng.normal(size=(n, 2))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    pi = np.array(
        [
            [1.0, 0.2],
            [0.4, -0.3],
            [-0.25, 0.9],
            [0.3, 0.15],
        ]
    )
    beta = np.array([1.0, -0.7, 0.45, -0.2])

    x_endog = z @ pi + x_exog @ np.array([[0.25, -0.1], [0.15, 0.2]]) + v
    u = 0.5 * v[:, 0] - 0.35 * v[:, 1] + 0.2 * eps
    y = -0.3 + np.column_stack([x_endog, x_exog]) @ beta + u

    model = cm.TwoSLS()
    model.fit(x_endog, x_exog, z, y)
    summary = model.summary()
    intercept_hat, coef_hat = twosls_closed_form(x_endog, x_exog, z, y)

    np.testing.assert_allclose(summary["intercept"], intercept_hat, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], coef_hat, atol=1e-8, rtol=1e-8)
    assert np.all(np.isfinite(summary["coef_se"]))
    assert summary["coef"].shape == (beta.size,)


def test_twosls_summary_supports_newey_west_and_cluster_vcov():
    rng = np.random.default_rng(1718)
    n = 1600

    z = rng.normal(size=(n, 4))
    x_exog = rng.normal(size=(n, 2))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    clusters = np.repeat(np.arange(80, dtype=np.int64), n // 80)
    pi = np.array(
        [
            [0.9, 0.25],
            [0.35, -0.2],
            [-0.15, 0.75],
            [0.2, 0.1],
        ]
    )

    x_endog = z @ pi + x_exog @ np.array([[0.2, -0.15], [0.1, 0.25]]) + v
    u = (0.6 + 0.2 * z[:, 0] ** 2) * (0.55 * v[:, 0] - 0.4 * v[:, 1] + 0.2 * eps)
    y = 0.3 + np.column_stack([x_endog, x_exog]) @ np.array([1.05, -0.8, 0.5, -0.25]) + u

    model = cm.TwoSLS()
    model.fit(x_endog, x_exog, z, y)

    nw = model.summary(vcov="newey_west", lags=4)
    cluster = model.summary(vcov="cluster", clusters=clusters)
    beta_nw, cov_nw = twosls_summary_reference(x_endog, x_exog, z, y, kind="newey_west", lags=4)
    beta_cluster, cov_cluster = twosls_summary_reference(x_endog, x_exog, z, y, kind="cluster", clusters=clusters)

    np.testing.assert_allclose(nw["intercept"], beta_nw[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef"], beta_nw[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["intercept_se"], np.sqrt(cov_nw[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(nw["coef_se"], np.sqrt(np.diag(cov_nw))[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        cluster["intercept_se"],
        np.sqrt(cov_cluster[0, 0]),
        atol=1e-8,
        rtol=1e-8,
    )
    np.testing.assert_allclose(
        cluster["coef_se"],
        np.sqrt(np.diag(cov_cluster))[1:],
        atol=1e-8,
        rtol=1e-8,
    )
    assert nw["vcov_type"] == "newey_west"
    assert cluster["vcov_type"] == "cluster"


def test_twosls_unit_sample_weights_match_unweighted_fit_and_summary():
    rng = np.random.default_rng(1718_1)
    n = 1300
    z = rng.normal(size=(n, 3))
    x_exog = rng.normal(size=(n, 1))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    x_endog = z @ np.array([[0.9, 0.1], [0.2, 0.7], [-0.15, 0.3]]) + 0.25 * x_exog + v
    y = 0.2 + np.column_stack([x_endog, x_exog]) @ np.array([1.0, -0.75, 0.35]) + 0.5 * v[:, 0] - 0.3 * v[:, 1] + 0.2 * eps
    unit_weights = np.ones(n)

    baseline = cm.TwoSLS()
    baseline.fit(x_endog, x_exog, z, y)

    weighted = cm.TwoSLS()
    weighted.fit_weighted(x_endog, x_exog, z, y, unit_weights)

    baseline_summary = baseline.summary(vcov="hc1")
    weighted_summary = weighted.summary(vcov="hc1")

    np.testing.assert_allclose(weighted_summary["intercept"], baseline_summary["intercept"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["coef"], baseline_summary["coef"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["intercept_se"], baseline_summary["intercept_se"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(weighted_summary["coef_se"], baseline_summary["coef_se"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(
        weighted.predict(np.column_stack([x_endog[:33], x_exog[:33]])),
        baseline.predict(np.column_stack([x_endog[:33], x_exog[:33]])),
        atol=1e-8,
        rtol=1e-8,
    )


def test_twosls_sample_weights_match_weighted_closed_form_and_vanilla_summary():
    rng = np.random.default_rng(1718_2)
    n = 1500
    z = rng.normal(size=(n, 4))
    x_exog = rng.normal(size=(n, 2))
    v = rng.normal(size=(n, 2))
    eps = rng.normal(size=n)
    weights = 0.25 + rng.random(n) * 2.0
    x_endog = z @ np.array([[0.95, 0.2], [0.35, -0.25], [-0.2, 0.8], [0.25, 0.1]]) + x_exog @ np.array([[0.2, -0.1], [0.1, 0.2]]) + v
    y = -0.1 + np.column_stack([x_endog, x_exog]) @ np.array([1.1, -0.7, 0.45, -0.2]) + 0.55 * v[:, 0] - 0.35 * v[:, 1] + 0.2 * eps

    model = cm.TwoSLS()
    model.fit_weighted(x_endog, x_exog, z, y, weights)
    summary = model.summary(vcov="vanilla")

    beta_hat, cov = weighted_twosls_summary_reference(
        x_endog,
        x_exog,
        z,
        y,
        weights,
        kind="vanilla",
    )

    np.testing.assert_allclose(summary["intercept"], beta_hat[0], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef"], beta_hat[1:], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["intercept_se"], np.sqrt(cov[0, 0]), atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["coef_se"], np.sqrt(np.diag(cov))[1:], atol=1e-8, rtol=1e-8)


def test_twosls_rejects_underidentified_design():
    rng = np.random.default_rng(33)
    n = 100
    x_endog = rng.normal(size=(n, 2))
    x_exog = np.empty((n, 0))
    z = rng.normal(size=(n, 1))
    y = rng.normal(size=n)

    model = cm.TwoSLS()
    with pytest.raises(ValueError, match="need at least as many excluded instruments"):
        model.fit(x_endog, x_exog, z, y)


def test_synthetic_control_recovers_convex_weights_and_post_path():
    rng = np.random.default_rng(777)
    true_weights = np.array([0.55, 0.3, 0.15])

    donors_pre = rng.normal(size=(60, true_weights.size))
    treated_pre = donors_pre @ true_weights
    donors_post = rng.normal(size=(25, true_weights.size))
    treated_post = donors_post @ true_weights

    model = cm.SyntheticControl(max_iterations=400)
    model.fit(donors_pre, treated_pre)
    summary = model.summary()
    weights = np.asarray(summary["weights"])
    pred_post = model.predict(donors_post)

    np.testing.assert_allclose(weights, true_weights, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(pred_post, treated_post, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-8, rtol=0.0)
    assert np.all(weights >= 0.0)
    assert summary["pre_rmse"] < 1e-6
    assert summary["converged"] is True
    assert model.bootstrap(4, seed=5).shape == (4, donors_pre.shape[1])


def test_synthetic_did_recovers_constant_effect_in_factor_panel():
    rng = np.random.default_rng(778)
    n_control = 8
    n_treated = 3
    t_pre = 14
    t_post = 6
    n_periods = t_pre + t_post
    time = np.arange(n_periods)

    factors = np.column_stack(
        [
            np.linspace(-1.0, 1.0, n_periods),
            np.sin(np.linspace(0.0, 2.5 * np.pi, n_periods)),
            np.cos(np.linspace(0.0, 1.5 * np.pi, n_periods)),
        ]
    )
    loadings = rng.normal(size=(n_control, factors.shape[1]))
    controls = (
        rng.normal(scale=0.4, size=(n_control, 1))
        + rng.normal(scale=0.02, size=(n_control, 1)) * time
        + loadings @ factors.T
    )

    true_weights = rng.dirichlet(np.ones(n_control))
    untreated_treated_mean = true_weights @ controls
    tau = 1.25
    effect = np.r_[np.zeros(t_pre), np.full(t_post, tau)]
    treated_offsets = np.array([-0.2, 0.05, 0.15])
    treated = untreated_treated_mean + treated_offsets[:, None] + effect

    ordered_panel = np.vstack([controls, treated])
    permutation = rng.permutation(n_control + n_treated)
    panel = ordered_panel[permutation]
    treated_units = [
        int(idx)
        for idx, original_idx in enumerate(permutation)
        if original_idx >= n_control
    ]

    model = cm.SyntheticDID(
        zeta_omega=1e-10,
        zeta_lambda=1e-10,
        max_iterations=2000,
    )
    w = np.zeros_like(panel)
    w[treated_units, t_pre:] = 1.0
    model.fit(panel, w)
    summary = model.summary()

    control_units = np.asarray(summary["control_units"])
    unit_weights = np.asarray(summary["unit_weights"])[0, control_units]
    time_weights = np.asarray(summary["time_weights"])[0, :t_pre]

    np.testing.assert_allclose(summary["att"], tau, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(unit_weights.sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(time_weights.sum(), 1.0, atol=1e-8)
    assert np.all(unit_weights >= 0.0)
    assert np.all(time_weights >= 0.0)
    assert summary["pre_rmse"] < 1e-4
    assert summary["converged"] is True
    assert "event_study" in summary
    assert "group_means" in summary
    assert "weighted" in summary["group_means"]
    assert "unweighted" in summary["group_means"]
    assert "std_error" not in summary["event_study"]["weighted"]
    assert "lower" not in summary["event_study"]["weighted"]
    assert "upper" not in summary["event_study"]["weighted"]

    treated_units_summary = np.asarray(summary["treated_units"])
    y_reordered = np.vstack([panel[control_units], panel[treated_units_summary]])
    unit_vec = np.r_[
        -unit_weights,
        np.ones(n_treated) / n_treated,
    ]
    time_vec = np.r_[
        -time_weights,
        np.ones(t_post) / t_post,
    ]
    np.testing.assert_allclose(summary["att"], unit_vec @ y_reordered @ time_vec, atol=1e-10)


def test_synthetic_did_default_zeta_matches_synthdid_first_difference_sd():
    y, w = make_synthetic_did_panel(n_control=5, n_treated=2, t_pre=6, t_post=3, seed=995)
    model = cm.SyntheticDID(max_iterations=800)
    model.fit(y, w)
    summary = model.summary()

    control_pre = y[:5, :6]
    sigma = np.diff(control_pre, axis=1).ravel().std(ddof=1)
    expected_zeta_omega = (2 * 3) ** 0.25 * sigma
    expected_zeta_lambda = 1e-6 * sigma

    np.testing.assert_allclose(summary["zeta_omega"], [expected_zeta_omega], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(summary["zeta_lambda"], [expected_zeta_lambda], rtol=1e-10, atol=1e-10)


def test_synthetic_did_att_uses_time_weights_not_post_gap_average():
    panel = np.array(
        [
            [0.12573022, -0.13210486, 0.64042265, 0.10490012, -0.53566937, 0.36159505, 1.30400005],
            [0.94708096, -0.70373524, -1.26542147, -0.62327446, 0.04132598, -2.32503077, -0.21879166],
            [-1.24591095, -0.73226735, -0.54425898, -0.31630016, 0.41163054, 1.04251337, -0.12853466],
            [1.36646347, -0.66519467, 0.35151007, 0.90347018, 0.09401230, -0.74349925, -0.92172538],
            [-0.45772583, 0.22019512, -1.00961818, -0.20917557, 1.84077499, 2.54084558, 2.21465912],
        ]
    )
    t_pre = 4
    w = np.zeros_like(panel)
    w[-1, t_pre:] = 1.0

    model = cm.SyntheticDID(zeta_omega=0.01, zeta_lambda=0.01, max_iterations=3000)
    model.fit(panel, w)
    summary = model.summary()

    control_units = np.asarray(summary["control_units"])
    treated_units = np.asarray(summary["treated_units"])
    unit_weights = np.asarray(summary["unit_weights"])[0, control_units]
    time_weights = np.asarray(summary["time_weights"])[0, :t_pre]
    y_reordered = np.vstack([panel[control_units], panel[treated_units]])
    unit_vec = np.r_[-unit_weights, np.ones(treated_units.size) / treated_units.size]
    time_vec = np.r_[-time_weights, np.ones(panel.shape[1] - t_pre) / (panel.shape[1] - t_pre)]

    sdid_att = unit_vec @ y_reordered @ time_vec
    post_gap_average = np.nanmean(np.asarray(summary["treatment_effect"])[w == 1])

    np.testing.assert_allclose(summary["att"], sdid_att, atol=1e-10)
    assert abs(summary["att"] - post_gap_average) > 0.2


def test_synthetic_did_rejects_bad_panel_inputs():
    panel = np.ones((4, 5))
    model = cm.SyntheticDID()

    w = np.zeros_like(panel)
    with pytest.raises(ValueError, match="ever-treated"):
        model.fit(panel, w)
    bad = np.zeros_like(panel)
    bad[1, 2:] = 1.0
    bad[1, 4] = 0.0
    with pytest.raises(ValueError, match="absorbing"):
        model.fit(panel, bad)
    bad = np.zeros((4, 4))
    with pytest.raises(ValueError, match="same shape"):
        model.fit(panel, bad)


def load_prop99_panel():
    data_path = Path(__file__).resolve().parents[1] / "docs" / "data" / "california_prop99.csv"
    data = np.genfromtxt(data_path, delimiter=";", names=True, dtype=None, encoding="utf-8")
    states = np.array(sorted(np.unique(data["State"])))
    years = np.array(sorted(np.unique(data["Year"])))
    state_index = {state: i for i, state in enumerate(states)}
    year_index = {year: i for i, year in enumerate(years)}
    y = np.zeros((len(states), len(years)))
    w = np.zeros_like(y)
    for row in data:
        i = state_index[row["State"]]
        t = year_index[row["Year"]]
        y[i, t] = row["PacksPerCapita"]
        w[i, t] = row["treated"]
    treated = w.sum(axis=1) > 0
    order = np.r_[np.where(~treated)[0], np.where(treated)[0]]
    return y[order], w[order]


def make_synthetic_did_panel(n_control=8, n_treated=3, t_pre=8, t_post=4, seed=991):
    rng = np.random.default_rng(seed)
    n_periods = t_pre + t_post
    time = np.arange(n_periods)
    factors = np.column_stack(
        [
            np.linspace(-1.0, 1.0, n_periods),
            np.sin(np.linspace(0.0, 2.0 * np.pi, n_periods)),
        ]
    )
    controls = (
        rng.normal(scale=0.3, size=(n_control, 1))
        + rng.normal(scale=0.02, size=(n_control, 1)) * time
        + rng.normal(size=(n_control, factors.shape[1])) @ factors.T
        + rng.normal(scale=0.03, size=(n_control, n_periods))
    )
    true_weights = rng.dirichlet(np.ones(n_control))
    treated_base = true_weights @ controls
    treated = (
        treated_base
        + np.linspace(-0.1, 0.1, n_treated)[:, None]
        + np.r_[np.zeros(t_pre), np.full(t_post, 0.8)]
        + rng.normal(scale=0.02, size=(n_treated, n_periods))
    )
    y = np.vstack([controls, treated])
    w = np.zeros_like(y)
    w[n_control:, t_pre:] = 1.0
    return y, w


def test_synthetic_did_prop99_matches_synthdid_readme_point_and_placebo_scale():
    y, w = load_prop99_panel()
    model = cm.SyntheticDID(max_iterations=5000)
    model.fit(y, w)

    # Reference: R synthdid README's california_prop99 panel_estimate table.
    # The placebo SE is Monte Carlo, so this pins the inference scale with a
    # small deterministic replication count rather than requiring exact R RNG parity.
    np.testing.assert_allclose(model.summary()["att"], -15.6038278727, atol=0.01, rtol=0.0)
    placebo_se = model.se("placebo", replications=40, seed=123)
    np.testing.assert_allclose(placebo_se, 9.647, atol=1.25, rtol=0.0)
    np.testing.assert_allclose(
        np.asarray(model.vcov("placebo", replications=40, seed=123))[0, 0],
        placebo_se**2,
        atol=1e-10,
        rtol=1e-10,
    )


def test_synthetic_did_vcov_bootstrap_and_jackknife_shape_seeded():
    y, w = make_synthetic_did_panel()
    model = cm.SyntheticDID(zeta_omega=0.01, zeta_lambda=0.01, max_iterations=800)
    model.fit(y, w)

    boot = np.asarray(model.vcov("bootstrap", replications=12, seed=123))
    boot_again = np.asarray(model.vcov("bootstrap", replications=12, seed=123))
    jack = np.asarray(model.vcov("jackknife"))

    assert boot.shape == (1, 1)
    assert jack.shape == (1, 1)
    np.testing.assert_allclose(boot, boot_again, atol=0.0, rtol=0.0)
    assert np.isfinite(boot[0, 0]) and boot[0, 0] >= 0.0
    assert np.isfinite(jack[0, 0]) and jack[0, 0] >= 0.0
    np.testing.assert_allclose(model.se("bootstrap", replications=12, seed=123) ** 2, boot[0, 0])


def test_synthetic_did_single_treated_bootstrap_jackknife_nan_placebo_works():
    y, w = make_synthetic_did_panel(n_control=8, n_treated=1, seed=992)
    model = cm.SyntheticDID(zeta_omega=0.01, zeta_lambda=0.01, max_iterations=800)
    model.fit(y, w)

    assert np.isnan(model.vcov("bootstrap", replications=10, seed=12)[0, 0])
    assert np.isnan(model.vcov("jackknife")[0, 0])
    placebo = np.asarray(model.vcov("placebo", replications=10, seed=12))
    assert placebo.shape == (1, 1)
    assert np.isfinite(placebo[0, 0]) and placebo[0, 0] >= 0.0


def test_synthetic_did_placebo_requires_more_controls_than_treated():
    y, w = make_synthetic_did_panel(n_control=1, n_treated=1, seed=993)
    model = cm.SyntheticDID(zeta_omega=0.01, zeta_lambda=0.01, max_iterations=800)
    model.fit(y, w)
    with pytest.raises(ValueError, match="more controls than treated"):
        model.vcov("placebo", replications=10, seed=1)


def test_synthetic_did_vcov_rejects_bad_method_and_too_few_replications():
    y, w = make_synthetic_did_panel(seed=994)
    model = cm.SyntheticDID(zeta_omega=0.01, zeta_lambda=0.01, max_iterations=800)
    model.fit(y, w)
    with pytest.raises(ValueError, match="method must be"):
        model.vcov("sandwich")
    with pytest.raises(ValueError, match="replications must be at least 2"):
        model.vcov("bootstrap", replications=1)


def test_logit_prediction_api_separates_index_probability_and_labels():
    rng = np.random.default_rng(20260525)
    n = 1200
    intercept = -0.15
    beta = np.array([0.7, -0.45, 0.25])

    x = rng.normal(size=(n, beta.size))
    eta_true = intercept + x @ beta
    p_true = 1.0 / (1.0 + np.exp(-eta_true))
    y = rng.binomial(1, p_true).astype(np.int32)

    model = cm.Logit(max_iterations=300, gradient_tolerance=1e-8)
    model.fit(x, y)

    eta_hat = model.predict_lin(x)
    p_hat = model.predict(x)
    labels_default = model.predict_label(x)
    labels_loose = model.predict_label(x, cutoff=0.35)

    np.testing.assert_allclose(p_hat, 1.0 / (1.0 + np.exp(-eta_hat)), atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(labels_default, (p_hat >= 0.5).astype(np.int32))
    np.testing.assert_array_equal(labels_loose, (p_hat >= 0.35).astype(np.int32))
    assert np.all((p_hat > 0.0) & (p_hat < 1.0))



def test_multinomial_logit_prediction_api_returns_probabilities_and_labels():
    rng = np.random.default_rng(20260526)
    n = 1500
    coef = np.array(
        [
            [0.8, -0.2, 0.15],
            [-0.1, 0.65, -0.35],
            [0.25, -0.4, 0.45],
        ]
    )
    intercept = np.array([0.25, -0.15, 0.05])

    x = rng.normal(size=(n, coef.shape[1]))
    logits = x @ coef.T + intercept
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(coef.shape[0], p=probs[i]) for i in range(n)], dtype=np.int32)

    model = cm.MultinomialLogit(max_iterations=300, gradient_tolerance=1e-8)
    model.fit(x, y)

    eta_hat = model.predict_lin(x[:73])
    p_hat = model.predict(x[:73])
    labels_hat = model.predict_label(x[:73])

    eta_centered = eta_hat - eta_hat.max(axis=1, keepdims=True)
    softmax = np.exp(eta_centered)
    softmax = softmax / softmax.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(p_hat, softmax, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(p_hat.sum(axis=1), np.ones(p_hat.shape[0]), atol=1e-10, rtol=1e-10)
    np.testing.assert_array_equal(labels_hat, p_hat.argmax(axis=1).astype(np.int32))



def test_poisson_prediction_api_exposes_linear_index_and_mean_scale():
    rng = np.random.default_rng(20260527)
    n = 1000
    intercept = 0.2
    beta = np.array([0.35, -0.3])

    x = rng.normal(size=(n, beta.size))
    mu = np.exp(intercept + x @ beta)
    y = rng.poisson(mu).astype(float)

    model = cm.Poisson(max_iterations=250, tolerance=1e-8)
    model.fit(x, y)

    eta_hat = model.predict_lin(x[:91])
    mu_hat = model.predict(x[:91])

    np.testing.assert_allclose(mu_hat, np.exp(eta_hat), atol=1e-10, rtol=1e-10)
    assert np.all(mu_hat > 0.0)



def test_synthetic_estimators_reject_zero_iteration_budget():
    donors = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ]
    )
    treated = np.array([0.8, 0.5, 0.2])
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        cm.SyntheticControl(max_iterations=0).fit(donors, treated)

    panel, treatment = make_synthetic_did_panel(seed=20260712)
    with pytest.raises(ValueError, match="max_iterations must be positive"):
        cm.SyntheticDID(max_iterations=0).fit(panel, treatment)
