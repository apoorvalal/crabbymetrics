import numpy as np
import pytest

import crabbymetrics as cm


def _ols_fit(x, y):
    design = np.column_stack([np.ones(x.shape[0]), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _ridge_fit(x, y, penalty):
    design = np.column_stack([np.ones(x.shape[0]), x])
    if penalty == 0.0:
        return np.linalg.lstsq(design, y, rcond=None)[0]
    p = design.shape[1]
    aug = np.zeros((design.shape[0] + p - 1, p))
    aug[: design.shape[0], :] = design
    for j in range(1, p):
        aug[design.shape[0] + j - 1, j] = np.sqrt(penalty)
    aug_y = np.concatenate([y, np.zeros(p - 1)])
    return np.linalg.lstsq(aug, aug_y, rcond=None)[0]


def _ridge_predict(x, params):
    design = np.column_stack([np.ones(x.shape[0]), x])
    return design @ params



def _sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def _weighted_ridge_fit(x, y, weights, penalty):
    design = np.column_stack([np.ones(x.shape[0]), x])
    sw = np.sqrt(np.maximum(weights, 1e-8))
    wx = design * sw[:, None]
    wy = y * sw
    if penalty > 0.0:
        p = design.shape[1]
        aug = np.zeros((design.shape[0] + p - 1, p))
        aug[: design.shape[0], :] = wx
        for j in range(1, p):
            aug[design.shape[0] + j - 1, j] = np.sqrt(penalty)
        wx = aug
        wy = np.concatenate([wy, np.zeros(p - 1)])
    return np.linalg.lstsq(wx, wy, rcond=None)[0]


def _logistic_ridge_fit(x, y, penalty, max_iter=50, tol=1e-8):
    design = np.column_stack([np.ones(x.shape[0]), x])
    y_mean = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    beta = np.zeros(design.shape[1])
    beta[0] = np.log(y_mean / (1.0 - y_mean))
    for _ in range(max_iter):
        eta = design @ beta
        p = np.clip(_sigmoid(eta), 1e-6, 1 - 1e-6)
        w = np.maximum(p * (1.0 - p), 1e-6)
        z = eta + (y - p) / w
        next_beta = _weighted_ridge_fit(x, z, w, penalty)
        step = np.abs(next_beta - beta).sum()
        beta = next_beta
        if step < tol:
            break
    return beta


def _logistic_predict(x, params):
    return _sigmoid(np.column_stack([np.ones(x.shape[0]), x]) @ params)

def _manual_iv(x, z, y):
    z_pi = np.linalg.lstsq(z, x, rcond=None)[0]
    x_hat = z @ z_pi
    return np.linalg.lstsq(x_hat, y, rcond=None)[0]


def _kfold_splits(n, n_folds, seed):
    k = min(n, n_folds)
    fold_id = np.empty(n, dtype=int)
    offset = seed % k
    for idx in range(n):
        fold_id[idx] = (idx + offset) % k
    out = []
    for fold in range(k):
        test_idx = np.flatnonzero(fold_id == fold)
        train_idx = np.flatnonzero(fold_id != fold)
        out.append((train_idx, test_idx))
    return out


def test_eplm_matches_manual_ratio():
    rng = np.random.default_rng(123)
    w = rng.normal(size=(600, 4))
    d = 0.5 + w @ np.array([0.7, -0.5, 0.3, 0.2]) + rng.normal(scale=0.7, size=600)
    g = 1.0 + w @ np.array([0.3, -0.2, 0.4, 0.1]) + 0.5 * w[:, 0] * w[:, 1]
    y = 2.0 * d + g + rng.normal(scale=0.6, size=600)

    nuisance = _ols_fit(w, d)
    ehat = np.column_stack([np.ones(w.shape[0]), w]) @ nuisance
    z = d - ehat
    expected = z @ y / (z @ d)

    model = cm.EPLM()
    model.fit(y, d, w)
    summary = model.summary()

    np.testing.assert_allclose(summary["coef"], expected, atol=1e-8, rtol=1e-8)
    assert summary["se"] > 0.0


def test_average_derivative_ob_matches_interacted_ols():
    rng = np.random.default_rng(456)
    w = rng.normal(size=(500, 3))
    d = 0.2 + w @ np.array([0.4, -0.3, 0.2]) + rng.normal(scale=0.5, size=500)
    wc = w - w.mean(axis=0)
    y = (
        1.5
        + wc @ np.array([0.5, -0.1, 0.25])
        + d * (1.2 + wc @ np.array([0.3, -0.2, 0.1]))
        + rng.normal(scale=0.5, size=500)
    )

    rx = np.column_stack([np.ones(w.shape[0]), wc, wc * d[:, None], d])
    expected = np.linalg.lstsq(rx, y, rcond=None)[0][-1]

    model = cm.AverageDerivative(method="ob")
    model.fit(y, d, w)
    summary = model.summary()

    np.testing.assert_allclose(summary["coef"], expected, atol=1e-8, rtol=1e-8)
    assert summary["se"] > 0.0


def test_average_derivative_ipw_matches_manual_normal_working_model():
    rng = np.random.default_rng(789)
    w = rng.normal(size=(700, 4))
    d = -0.3 + w @ np.array([0.6, -0.4, 0.5, 0.2]) + rng.normal(scale=0.8, size=700)
    y = 0.7 + 1.4 * d + w @ np.array([0.2, 0.3, -0.1, 0.4]) + rng.normal(scale=0.7, size=700)

    nuisance = _ols_fit(w, d)
    ehat = np.column_stack([np.ones(w.shape[0]), w]) @ nuisance
    resid = d - ehat
    sigma2 = np.mean(resid**2)
    gw = resid / sigma2
    expected = gw @ y / (gw @ d)

    model = cm.AverageDerivative(method="ipw")
    model.fit(y, d, w)
    summary = model.summary()

    np.testing.assert_allclose(summary["coef"], expected, atol=1e-8, rtol=1e-8)
    assert summary["se"] > 0.0


def test_average_derivative_dr_matches_manual_iv_normal_working_model():
    rng = np.random.default_rng(321)
    w = rng.normal(size=(650, 3))
    d = 0.1 + w @ np.array([0.7, -0.5, 0.3]) + rng.normal(scale=0.6, size=650)
    wc = w - w.mean(axis=0)
    y = (
        0.9
        + wc @ np.array([0.2, -0.3, 0.4])
        + d * (1.1 + wc @ np.array([0.25, 0.1, -0.2]))
        + rng.normal(scale=0.4, size=650)
    )

    nuisance = _ols_fit(w, d)
    ehat = np.column_stack([np.ones(w.shape[0]), w]) @ nuisance
    resid = d - ehat
    sigma2 = np.mean(resid**2)
    rx = np.column_stack([np.ones(w.shape[0]), wc, wc * d[:, None], d])
    z = np.column_stack([np.ones(w.shape[0]), wc, wc * d[:, None], resid / sigma2])
    expected = _manual_iv(rx, z, y)[-1]

    model = cm.AverageDerivative(method="dr")
    model.fit(y, d, w)
    summary = model.summary()

    np.testing.assert_allclose(summary["coef"], expected, atol=1e-8, rtol=1e-8)
    assert summary["se"] > 0.0


def test_partially_linear_dml_matches_manual_crossfit_residual_regression():
    rng = np.random.default_rng(654)
    x = rng.normal(size=(480, 5))
    d = 0.3 + x @ np.array([0.7, -0.4, 0.2, 0.1, -0.3]) + rng.normal(scale=0.8, size=480)
    l = 1.0 + x @ np.array([0.5, -0.2, 0.3, 0.1, 0.4]) + 0.4 * x[:, 0] * x[:, 1]
    y = 1.6 * d + l + rng.normal(scale=0.7, size=480)

    l_hat = np.zeros_like(y)
    m_hat = np.zeros_like(d)
    for train_idx, test_idx in _kfold_splits(len(y), 4, 17):
        y_params = _ridge_fit(x[train_idx], y[train_idx], 0.0)
        d_params = _ridge_fit(x[train_idx], d[train_idx], 0.0)
        l_hat[test_idx] = _ridge_predict(x[test_idx], y_params)
        m_hat[test_idx] = _ridge_predict(x[test_idx], d_params)
    d_resid = d - m_hat
    y_resid = y - l_hat
    expected = d_resid @ y_resid / (d_resid @ d_resid)

    model = cm.PartiallyLinearDML(penalty=0.0, n_folds=4, seed=17)
    model.fit(y, d, x)
    summary = model.summary()

    np.testing.assert_allclose(summary["coef"], expected, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["outcome_penalties"], np.zeros(4))
    np.testing.assert_allclose(summary["treatment_penalties"], np.zeros(4))
    assert summary["se"] > 0.0


def test_aipw_matches_manual_crossfit_formula():
    rng = np.random.default_rng(987)
    x = rng.normal(size=(520, 4))
    pi_true = 1.0 / (1.0 + np.exp(-(0.2 + x @ np.array([0.6, -0.5, 0.3, 0.2]))))
    d = rng.binomial(1, pi_true, size=520).astype(float)
    mu0 = 0.5 + x @ np.array([0.4, -0.1, 0.2, 0.3])
    tau = 1.3
    y = mu0 + tau * d + rng.normal(scale=0.8, size=520)

    mu0_hat = np.zeros_like(y)
    mu1_hat = np.zeros_like(y)
    pi_hat = np.zeros_like(y)
    for train_idx, test_idx in _kfold_splits(len(y), 5, 29):
        train_d = d[train_idx]
        treat_idx = train_idx[train_d == 1.0]
        control_idx = train_idx[train_d == 0.0]

        mu1_params = _ridge_fit(x[treat_idx], y[treat_idx], 0.0)
        mu0_params = _ridge_fit(x[control_idx], y[control_idx], 0.0)
        pi_params = _ridge_fit(x[train_idx], d[train_idx], 0.0)

        mu1_hat[test_idx] = _ridge_predict(x[test_idx], mu1_params)
        mu0_hat[test_idx] = _ridge_predict(x[test_idx], mu0_params)
        pi_hat[test_idx] = np.clip(_ridge_predict(x[test_idx], pi_params), 0.02, 0.98)

    pseudo = mu1_hat - mu0_hat + d * (y - mu1_hat) / pi_hat - (1.0 - d) * (y - mu0_hat) / (1.0 - pi_hat)
    expected = pseudo.mean()

    model = cm.AIPW(penalty=0.0, n_folds=5, seed=29)
    model.fit(y, d, x)
    summary = model.summary()

    np.testing.assert_allclose(summary["ate"], expected, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["propensity_penalties"], np.zeros(5))
    np.testing.assert_allclose(summary["outcome0_penalties"], np.zeros(5))
    np.testing.assert_allclose(summary["outcome1_penalties"], np.zeros(5))
    assert summary["se"] > 0.0



def test_att_aipw_hajek_matches_manual_crossfit_formula():
    rng = np.random.default_rng(2468)
    x = rng.normal(size=(540, 5))
    pi_true = 1.0 / (1.0 + np.exp(-(0.1 + x @ np.array([0.5, -0.4, 0.25, 0.15, -0.2]))))
    d = rng.binomial(1, pi_true, size=540).astype(float)
    mu0 = 0.7 + x @ np.array([0.3, -0.2, 0.15, 0.25, -0.1])
    tau = 1.0 + 0.4 * x[:, 0]
    y = mu0 + tau * d + rng.normal(scale=0.6, size=540)

    mu0_hat = np.zeros_like(y)
    pi_hat = np.zeros_like(y)
    for train_idx, test_idx in _kfold_splits(len(y), 5, 31):
        train_d = d[train_idx]
        control_idx = train_idx[train_d == 0.0]

        mu0_params = _ridge_fit(x[control_idx], y[control_idx], 0.0)
        pi_params = _ridge_fit(x[train_idx], d[train_idx], 0.0)

        mu0_hat[test_idx] = _ridge_predict(x[test_idx], mu0_params)
        pi_hat[test_idx] = np.clip(_ridge_predict(x[test_idx], pi_params), 0.02, 0.98)

    residual = y - mu0_hat
    odds = pi_hat / (1.0 - pi_hat)
    treated_component = residual[d == 1.0].mean()
    control_component = np.average(residual[d == 0.0], weights=odds[d == 0.0])
    expected = treated_component - control_component

    model = cm.ATTAIPW(penalty=0.0, n_folds=5, seed=31)
    model.fit(y, d, x)
    summary = model.summary()

    np.testing.assert_allclose(summary["att"], expected, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(summary["propensity_penalties"], np.zeros(5))
    np.testing.assert_allclose(summary["outcome0_penalties"], np.zeros(5))
    assert summary["se"] > 0.0


def test_att_aipw_rejects_nonbinary_treatment_and_bad_constructor_args():
    rng = np.random.default_rng(1357)
    y = rng.normal(size=40)
    d = rng.normal(size=40)
    x = rng.normal(size=(40, 4))

    with pytest.raises(ValueError, match="cv must be at least 2"):
        cm.ATTAIPW(penalty=np.array([0.1, 1.0]), cv=1, n_folds=3)

    with pytest.raises(ValueError, match="n_folds must be at least 2"):
        cm.ATTAIPW(penalty=0.1, n_folds=1)

    with pytest.raises(ValueError, match="0/1"):
        cm.ATTAIPW(penalty=0.1, n_folds=3, seed=7).fit(y, d, x)


def test_did_semiparametric_or_ipw_aipw_match_manual_hajek_scores():
    rng = np.random.default_rng(97531)
    x = rng.normal(size=(620, 6))
    logits = -0.1 + x @ np.array([0.35, -0.45, 0.25, 0.15, -0.2, 0.1])
    d = rng.binomial(1, _sigmoid(logits), size=x.shape[0]).astype(float)
    y0 = 0.8 + x @ np.array([0.2, -0.3, 0.1, 0.05, 0.15, -0.1]) + rng.normal(scale=0.5, size=x.shape[0])
    untreated_trend = 0.4 + x @ np.array([0.3, -0.1, 0.2, -0.25, 0.1, 0.05])
    tau = 1.25 + 0.2 * x[:, 0]
    y1 = y0 + untreated_trend + d * tau + rng.normal(scale=0.35, size=x.shape[0])
    delta = y1 - y0

    mu = np.zeros_like(delta)
    for train_idx, test_idx in _kfold_splits(len(delta), 5, 11):
        control = train_idx[d[train_idx] == 0.0]
        params = _ridge_fit(x[control], delta[control], 0.1)
        mu[test_idx] = _ridge_predict(x[test_idx], params)
    rho = d.mean()
    expected_or = np.mean(d * (delta - mu) / rho)

    pi = np.zeros_like(delta)
    for train_idx, test_idx in _kfold_splits(len(delta), 5, 28):
        params = _logistic_ridge_fit(x[train_idx], d[train_idx], 0.1)
        pi[test_idx] = np.clip(_logistic_predict(x[test_idx], params), 0.02, 0.98)
    odds = pi / (1.0 - pi)
    expected_ipw = delta[d == 1.0].mean() - np.average(delta[d == 0.0], weights=odds[d == 0.0])
    residual = delta - mu
    expected_aipw = residual[d == 1.0].mean() - np.average(residual[d == 0.0], weights=odds[d == 0.0])

    or_model = cm.DIDSemiparametric(method="or", penalty=0.1, n_folds=5, seed=11)
    or_model.fit(y0, y1, d, x)
    np.testing.assert_allclose(or_model.summary()["att"], expected_or, atol=1e-8, rtol=1e-8)

    ipw_model = cm.DIDSemiparametric(method="ipw", penalty=0.1, n_folds=5, seed=11)
    ipw_model.fit(y0, y1, d, x)
    np.testing.assert_allclose(ipw_model.summary()["att"], expected_ipw, atol=1e-8, rtol=1e-8)

    aipw_model = cm.DIDSemiparametric(method="aipw", penalty=0.1, n_folds=5, seed=11)
    aipw_model.fit(y0, y1, d, x)
    summary = aipw_model.summary()
    np.testing.assert_allclose(summary["att"], expected_aipw, atol=1e-8, rtol=1e-8)
    assert summary["se"] > 0.0
    assert summary["method"] == "aipw"


def test_did_semiparametric_rejects_bad_inputs():
    rng = np.random.default_rng(8642)
    y0 = rng.normal(size=50)
    y1 = rng.normal(size=50)
    d = rng.normal(size=50)
    x = rng.normal(size=(50, 3))
    with pytest.raises(ValueError, match="method must be"):
        cm.DIDSemiparametric(method="did")
    with pytest.raises(ValueError, match="basis must be"):
        cm.DIDSemiparametric(method="aipw", basis="sieve")
    with pytest.raises(ValueError, match="0/1"):
        cm.DIDSemiparametric(method="aipw", penalty=0.1).fit(y0, y1, d, x)

def test_semiparametric_estimators_reject_nonfinite_inputs():
    rng = np.random.default_rng(1234)
    y = rng.normal(size=30)
    d = rng.normal(size=30)
    x = rng.normal(size=(30, 3))
    y[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        cm.EPLM().fit(y, d, x)

    with pytest.raises(ValueError, match="finite"):
        cm.AverageDerivative(method="dr").fit(y, d, x)

    with pytest.raises(ValueError, match="finite"):
        cm.PartiallyLinearDML(penalty=0.1, n_folds=3, seed=5).fit(y, d, x)


def test_partially_linear_dml_rejects_invalid_constructor_arguments():
    with pytest.raises(ValueError, match="cv must be at least 2"):
        cm.PartiallyLinearDML(penalty=np.array([0.1, 1.0]), cv=1, n_folds=3)

    with pytest.raises(ValueError, match="n_folds must be at least 2"):
        cm.PartiallyLinearDML(penalty=0.1, n_folds=1)


def test_eplm_and_average_derivative_reject_invalid_fd_eps():
    with pytest.raises(ValueError, match="fd_eps must be a positive finite float"):
        cm.EPLM(fd_eps=0.0)

    with pytest.raises(ValueError, match="fd_eps must be a positive finite float"):
        cm.AverageDerivative(method="ipw", fd_eps=-1.0)


def test_aipw_rejects_nonbinary_treatment_and_bad_constructor_args():
    rng = np.random.default_rng(5678)
    y = rng.normal(size=40)
    d = rng.normal(size=40)
    x = rng.normal(size=(40, 4))

    with pytest.raises(ValueError, match="cv must be at least 2"):
        cm.AIPW(penalty=np.array([0.1, 1.0]), cv=1, n_folds=3)

    with pytest.raises(ValueError, match="n_folds must be at least 2"):
        cm.AIPW(penalty=0.1, n_folds=1)

    with pytest.raises(ValueError, match="0/1"):
        cm.AIPW(penalty=0.1, n_folds=3, seed=7).fit(y, d, x)
