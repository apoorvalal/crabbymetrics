import numpy as np
import crabbymetrics as cm


def _reference_design(x, c, include_cont_cat=True, include_cat_cat=True):
    n = len(x)
    cols = [np.ones(n), x]
    # Base category coding: c0 baseline 0, c1 baseline 0.
    cols.append((c[:, 0] == 1).astype(float))
    cols.append((c[:, 0] == 2).astype(float))
    cols.append((c[:, 1] == 1).astype(float))
    if include_cont_cat:
        cols.append(x * (c[:, 0] == 1))
        cols.append(x * (c[:, 0] == 2))
    if include_cat_cat:
        cols.append(((c[:, 0] == 1) & (c[:, 1] == 1)).astype(float))
        cols.append(((c[:, 0] == 2) & (c[:, 1] == 1)).astype(float))
    return np.column_stack(cols)


def test_abc_ols_constraints_and_reference_fitted_values():
    rng = np.random.default_rng(123)
    c0 = np.repeat(np.array([0, 1, 2], dtype=np.uint32), [30, 45, 25])
    c1 = np.tile(np.array([0, 1], dtype=np.uint32), 50)
    c = np.column_stack([c0, c1])
    n = c.shape[0]
    x = rng.normal(size=n)
    effects = np.array([-0.4, 0.1, 0.5])
    slopes = np.array([-0.2, 0.0, 0.35])
    y = 1.0 + 0.8 * x + effects[c0] + slopes[c0] * x + 0.25 * c1 + rng.normal(scale=0.05, size=n)

    abc = cm.ABCOLS()
    abc.fit(y, x[:, None], c, cont_cat_interactions=[(0, 0)], cat_cat_interactions=[(0, 1)])
    s = abc.summary()
    assert s["max_constraint_violation"] < 1e-8

    X_ref = _reference_design(x - x.mean(), c)
    beta_ref, *_ = np.linalg.lstsq(X_ref, y, rcond=None)
    yhat_ref = X_ref @ beta_ref
    np.testing.assert_allclose(np.asarray(abc.fitted_values()), yhat_ref, atol=1e-8)


def test_abc_ols_continuous_slope_is_weighted_average_group_slope():
    rng = np.random.default_rng(456)
    c0 = np.repeat(np.array([0, 1, 2], dtype=np.uint32), [20, 50, 30])
    c = c0[:, None]
    x = rng.normal(size=len(c0))
    y = 2.0 + 1.2 * x + np.array([-1.0, 0.25, 0.8])[c0] + np.array([0.5, -0.1, 0.2])[c0] * x

    abc = cm.ABCOLS()
    abc.fit(y, x[:, None], c, cont_cat_interactions=[(0, 0)])
    s = abc.summary()
    names = s["column_names"]
    coef = dict(zip(names, s["coef"]))
    weights = np.bincount(c0) / len(c0)
    group_slopes = np.array([coef["x0"] + coef[f"x0:c0[{level}]"] for level in range(3)])
    np.testing.assert_allclose(coef["x0"], weights @ group_slopes, atol=1e-8)
