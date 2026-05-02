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
    w = np.zeros_like(observed)
    w[treated_units, t_pre:] = 1.0
    return observed, untreated, w, treated_units, t_pre, tau


def reference_horizontal_ridge(panel, treated_units, t_pre, penalty):
    controls = [i for i in range(panel.shape[0]) if i not in set(treated_units)]
    treated = panel[treated_units].mean(axis=0)
    x_pre = np.column_stack([np.ones(t_pre), panel[controls, :t_pre].T])
    gram = x_pre.T @ x_pre
    gram[1:, 1:] += penalty * np.eye(len(controls))
    params = np.linalg.solve(gram, x_pre.T @ treated[:t_pre])
    x_all = np.column_stack([np.ones(panel.shape[1]), panel[controls].T])
    counterfactual = x_all @ params
    effect = panel[treated_units] - counterfactual[None, :]
    return params, counterfactual, effect


def test_horizontal_panel_ridge_matches_reference_formula():
    panel, _untreated, w, treated_units, t_pre, _tau = make_panel(seed=1)
    penalty = 0.4
    model = cm.HorizontalPanelRidge(penalty=penalty)
    model.fit(panel, w)
    summary = model.summary()

    params, counterfactual, effect = reference_horizontal_ridge(panel, treated_units, t_pre, penalty)
    pred = model.predict()
    te = model.treatment_effect()

    np.testing.assert_allclose(summary["intercept"], params[0], atol=1e-10)
    np.testing.assert_allclose(summary["coef"][: len(params) - 1], params[1:], atol=1e-10)
    for unit in treated_units:
        np.testing.assert_allclose(pred[unit], counterfactual, atol=1e-10)
    np.testing.assert_allclose(te[treated_units], effect, atol=1e-10)
    assert summary["control_units"] == list(range(panel.shape[0] - len(treated_units)))
    assert summary["treated_units"] == treated_units
    assert summary["cohorts"] == [t_pre]
    assert "event_study" in summary
    assert "group_means" in summary


def test_horizontal_panel_ridge_recovers_known_average_effect():
    panel, untreated, w, treated_units, t_pre, tau = make_panel(seed=3, n_control=28, n_treated=2)
    model = cm.HorizontalPanelRidge(penalty=0.15)
    model.fit(panel, w)
    summary = model.summary()

    true_counterfactual = untreated[treated_units].mean(axis=0)
    oracle_rmse = np.sqrt(np.mean((model.predict()[treated_units[0], t_pre:] - true_counterfactual[t_pre:]) ** 2))

    assert abs(summary["att"] - tau) < 0.12
    assert summary["pre_rmse"] < 0.05
    assert oracle_rmse < 0.12


def test_horizontal_panel_ridge_handles_staggered_adoption_event_outputs():
    panel, untreated, w, treated_units, t_pre, tau = make_panel(seed=8, n_control=24, n_treated=4, t_pre=10, t_post=10)
    # Split treated units into two adoption cohorts.
    later = treated_units[2:]
    w[later, :] = 0.0
    w[later, t_pre + 3 :] = 1.0
    panel[later, t_pre:] = untreated[later, t_pre:]
    panel[later, t_pre + 3 :] += tau

    model = cm.HorizontalPanelRidge(penalty=0.2)
    model.fit(panel, w)
    summary = model.summary()

    assert summary["cohorts"] == [t_pre, t_pre + 3]
    weighted = summary["event_study"]["weighted"]
    assert 0.0 in set(np.asarray(weighted["event_time"]))
    assert abs(summary["att"] - tau) < 0.2
    assert summary["group_means"]["cohort"].shape[0] > 0
    assert 0.0 in set(np.asarray(summary["group_means"]["weighted"]["event_time"]))


def test_horizontal_panel_ridge_validates_inputs():
    panel, _untreated, w, _treated_units, t_pre, _tau = make_panel(seed=4)

    with pytest.raises(ValueError, match="penalty"):
        cm.HorizontalPanelRidge(penalty=-1.0)

    model = cm.HorizontalPanelRidge()
    bad = w.copy()
    bad[:, :] = 0.0
    with pytest.raises(ValueError, match="ever-treated"):
        model.fit(panel, bad)
    bad = w.copy()
    bad[0, t_pre] = 0.5
    with pytest.raises(ValueError, match="binary"):
        model.fit(panel, bad)
    bad = w.copy()
    bad[-1, t_pre + 2] = 0.0
    with pytest.raises(ValueError, match="absorbing"):
        model.fit(panel, bad)
