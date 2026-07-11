import numpy as np
import statsmodels.api as sm

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
    reference_design = sm.add_constant(np.column_stack([x, dummy_block(groups)]))
    reference = sm.OLS(y, reference_design).fit(cov_type="HC1")
    np.testing.assert_allclose(summary["coef_se"], reference.bse[1:3], atol=1e-8, rtol=1e-8)
    assert summary["absorbed_df"] == np.unique(groups).size
    assert summary["residual_df"] == reference.df_resid
    assert summary["absorbed_df_method"] == "exact_one_way"
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
    np.testing.assert_allclose(
        summary["coef_se"],
        baseline_summary["coef_se"][: x.shape[1]],
        atol=1e-8,
        rtol=1e-8,
    )
    assert summary["absorbed_df"] == 35 + 18 - 1
    assert summary["residual_df"] == n - x.shape[1] - summary["absorbed_df"]
    assert summary["absorbed_df_method"] == "exact_two_way"



def test_fixed_effects_ols_two_way_rank_handles_disconnected_components():
    rng = np.random.default_rng(20260710)
    pairs = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 2],
            [2, 3],
            [3, 2],
            [3, 3],
        ],
        dtype=np.uint32,
    )
    fe = np.repeat(pairs, 30, axis=0)
    x = rng.normal(size=(fe.shape[0], 1))
    worker_effect = np.array([0.2, -0.4, 0.7, -0.1])
    firm_effect = np.array([-0.3, 0.5, 0.4, -0.2])
    y = 0.8 * x[:, 0] + worker_effect[fe[:, 0]] + firm_effect[fe[:, 1]]
    y += rng.normal(scale=0.2, size=y.size)

    model = cm.FixedEffectsOLS()
    model.fit(x, fe, y)
    summary = model.summary(vcov="vanilla")

    assert summary["absorbed_df"] == 4 + 4 - 2
    assert summary["residual_df"] == y.size - x.shape[1] - 6
    assert summary["absorbed_df_method"] == "exact_two_way"


def test_fixed_effects_ols_reports_conservative_multiway_rank():
    rng = np.random.default_rng(20260711)
    n = 400
    fe = np.column_stack(
        [
            rng.integers(0, 7, size=n),
            rng.integers(0, 5, size=n),
            rng.integers(0, 4, size=n),
        ]
    ).astype(np.uint32)
    x = rng.normal(size=(n, 2))
    effects = [rng.normal(size=levels) for levels in (7, 5, 4)]
    y = x @ np.array([0.4, -0.2])
    for column, values in enumerate(effects):
        y += values[fe[:, column]]
    y += rng.normal(scale=0.3, size=n)

    model = cm.FixedEffectsOLS()
    model.fit(x, fe, y)
    summary = model.summary()

    assert summary["absorbed_df"] == 7 + 5 + 4 - 2
    assert summary["residual_df"] == n - x.shape[1] - summary["absorbed_df"]
    assert summary["absorbed_df_method"] == "conservative_multiway"
