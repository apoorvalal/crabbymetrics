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


def _manual_iv(x, z, y):
    z_pi = np.linalg.lstsq(z, x, rcond=None)[0]
    x_hat = z @ z_pi
    return np.linalg.lstsq(x_hat, y, rcond=None)[0]


def _splitmix64(value):
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _kfold_splits(n, n_folds, seed, strata=None):
    k = min(n, n_folds)
    fold_id = np.empty(n, dtype=int)
    if strata is None:
        groups = [np.arange(n)]
    else:
        groups = [np.flatnonzero(strata == 0.0), np.flatnonzero(strata == 1.0)]
    for group in groups:
        ordered = sorted(group, key=lambda idx: _splitmix64(seed ^ int(idx)))
        for position, idx in enumerate(ordered):
            fold_id[idx] = position % k
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
    for train_idx, test_idx in _kfold_splits(len(y), 5, 29, strata=d):
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
