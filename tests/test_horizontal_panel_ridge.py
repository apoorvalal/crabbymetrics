import numpy as np
import pytest

import crabbymetrics as cm


def make_panel(seed=0, n_control=18, n_treated=3, t_pre=16, t_post=7, tau=-1.25):
    rng = np.random.default_rng(seed)
    n = n_control + n_treated
    tt = t_pre + t_post
    time = np.arange(tt)
    factors = np.column_stack(
        [
            np.linspace(-1.2, 1.4, tt),
            np.sin(np.linspace(0.0, 2.5 * np.pi, tt)),
        ]
    )
    loadings = rng.normal(size=(n, factors.shape[1]))
    unit_fe = rng.normal(scale=0.35, size=n)
    untreated = unit_fe[:, None] + loadings @ factors.T + rng.normal(scale=0.03, size=(n, tt))
    treated_units = list(range(n_control, n))
    observed = untreated.copy()
    observed[treated_units, t_pre:] += tau
    return observed, untreated, treated_units, t_pre, tau


def reference_horizontal_ridge(panel, treated_units, t_pre, penalty):
    controls = [i for i in range(panel.shape[0]) if i not in set(treated_units)]
    treated = panel[treated_units].mean(axis=0)
    x_pre = np.column_stack([np.ones(t_pre), panel[controls, :t_pre].T])
    gram = x_pre.T @ x_pre
    gram[1:, 1:] += penalty * np.eye(len(controls))
    params = np.linalg.solve(gram, x_pre.T @ treated[:t_pre])
    x_all = np.column_stack([np.ones(panel.shape[1]), panel[controls].T])
    counterfactual = x_all @ params
    effect = treated - counterfactual
    return params, counterfactual, effect


def test_horizontal_panel_ridge_matches_reference_formula():
    panel, _untreated, treated_units, t_pre, _tau = make_panel(seed=1)
    penalty = 0.4
    model = cm.HorizontalPanelRidge(penalty=penalty)
    model.fit(panel, treated_units, t_pre)
    summary = model.summary()

    params, counterfactual, effect = reference_horizontal_ridge(panel, treated_units, t_pre, penalty)

    np.testing.assert_allclose(summary["intercept"], params[0], atol=1e-10)
    np.testing.assert_allclose(summary["coef"], params[1:], atol=1e-10)
    np.testing.assert_allclose(model.predict(), counterfactual, atol=1e-10)
    np.testing.assert_allclose(model.treatment_effect(), effect, atol=1e-10)
    assert summary["control_units"] == list(range(panel.shape[0] - len(treated_units)))
    assert summary["treated_units"] == treated_units
    assert summary["t_pre"] == t_pre


def test_horizontal_panel_ridge_recovers_known_average_effect():
    panel, untreated, treated_units, t_pre, tau = make_panel(seed=3, n_control=28, n_treated=2)
    model = cm.HorizontalPanelRidge(penalty=0.15)
    model.fit(panel, treated_units, t_pre)
    summary = model.summary()

    true_counterfactual = untreated[treated_units].mean(axis=0)
    oracle_rmse = np.sqrt(np.mean((model.predict()[t_pre:] - true_counterfactual[t_pre:]) ** 2))

    assert abs(summary["att"] - tau) < 0.12
    assert summary["pre_rmse"] < 0.05
    assert oracle_rmse < 0.12


def test_horizontal_panel_ridge_validates_inputs():
    panel, _untreated, treated_units, t_pre, _tau = make_panel(seed=4)

    with pytest.raises(ValueError, match="penalty"):
        cm.HorizontalPanelRidge(penalty=-1.0)

    model = cm.HorizontalPanelRidge()
    with pytest.raises(ValueError, match="t_pre"):
        model.fit(panel, treated_units, 0)
    with pytest.raises(ValueError, match="treated unit index"):
        model.fit(panel, [panel.shape[0]], t_pre)
    with pytest.raises(ValueError, match="duplicates"):
        model.fit(panel, [treated_units[0], treated_units[0]], t_pre)
