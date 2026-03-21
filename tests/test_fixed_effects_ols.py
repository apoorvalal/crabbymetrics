import numpy as np

import crabbymetrics as cm


def demean_one_way(x, y, groups):
    x_tilde = np.empty_like(x)
    y_tilde = np.empty_like(y)

    for group in np.unique(groups):
        mask = groups == group
        x_tilde[mask] = x[mask] - x[mask].mean(axis=0, keepdims=True)
        y_tilde[mask] = y[mask] - y[mask].mean()

    return x_tilde, y_tilde


def dummy_block(levels):
    n_levels = int(levels.max()) + 1
    block = np.zeros((levels.shape[0], max(n_levels - 1, 0)))
    for level in range(1, n_levels):
        block[:, level - 1] = (levels == level).astype(float)
    return block


def test_fixed_effects_ols_matches_manual_one_way_within():
    rng = np.random.default_rng(123)
    n = 500
    beta_true = np.array([1.25, -0.8])

    groups = rng.integers(0, 25, size=n, dtype=np.uint32)
    fe = groups.reshape(-1, 1)
    x = rng.normal(size=(n, 2))
    alpha = rng.normal(scale=1.5, size=25)
    y = x @ beta_true + alpha[groups] + rng.normal(scale=0.1, size=n)

    x_tilde, y_tilde = demean_one_way(x, y, groups)
    beta_manual, *_ = np.linalg.lstsq(x_tilde, y_tilde, rcond=None)

    model = cm.FixedEffectsOLS()
    model.fit(x, fe, y)
    summary = model.summary()

    np.testing.assert_allclose(summary["coef"], beta_manual, atol=1e-6, rtol=1e-6)
    assert model.bootstrap(4, seed=7).shape == (4, x.shape[1])


def test_fixed_effects_ols_matches_dummy_ols_for_two_way_effects():
    rng = np.random.default_rng(321)
    n = 700
    beta_true = np.array([0.9, -1.1])

    worker = rng.integers(0, 35, size=n, dtype=np.uint32)
    firm = rng.integers(0, 18, size=n, dtype=np.uint32)
    fe = np.column_stack([worker, firm]).astype(np.uint32)

    x = rng.normal(size=(n, 2))
    worker_effect = rng.normal(scale=0.7, size=35)
    firm_effect = rng.normal(scale=0.5, size=18)
    y = (
        x @ beta_true
        + worker_effect[worker]
        + firm_effect[firm]
        + rng.normal(scale=0.1, size=n)
    )

    model = cm.FixedEffectsOLS()
    model.fit(x, fe, y)
    summary = model.summary()

    dummy_design = np.column_stack([x, dummy_block(worker), dummy_block(firm)])
    baseline = cm.OLS()
    baseline.fit(dummy_design, y)
    baseline_summary = baseline.summary()

    np.testing.assert_allclose(
        summary["coef"],
        baseline_summary["coef"][: x.shape[1]],
        atol=1e-5,
        rtol=1e-5,
    )
