"""Array layout and ownership checks for shared NumPy conversion helpers."""

import crabbymetrics as cm
import numpy as np
import pytest


@pytest.mark.parametrize("layout", ["c", "fortran", "strided", "reversed"])
def test_weighted_ols_accepts_numpy_layouts_without_mutating_inputs(layout):
    rng = np.random.default_rng(35)
    storage = rng.normal(size=(100, 6))
    x = storage[:, ::2] if layout == "strided" else storage[:, :3].copy()
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "reversed":
        x = x[::-1, ::-1]
    y = 0.4 + x @ np.array([0.5, -0.2, 0.8]) + rng.normal(scale=0.1, size=100)
    weights = rng.uniform(0.1, 2, size=100)
    original = x.copy()
    model = cm.OLS()
    model.fit_weighted(x, y, weights)
    design = np.column_stack([np.ones(100), x])
    expected = np.linalg.lstsq(
        design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None
    )[0]
    np.testing.assert_allclose(model.predict(x), design @ expected, atol=1e-12)
    np.testing.assert_array_equal(x, original)


def test_matrix_outputs_preserve_empty_shapes_and_do_not_alias_fit_state():
    rng = np.random.default_rng(36)
    x = rng.normal(size=(120, 3))
    y = (np.arange(120) % 3).astype(np.int32)
    model = cm.MultinomialLogit()
    model.fit(x, y)
    assert model.predict(np.empty((0, 3))).shape == (0, 3)
    predictions = model.predict(x)
    expected = predictions.copy()
    predictions[:] = -999
    np.testing.assert_array_equal(model.predict(x), expected)
    ols = cm.OLS()
    ols.fit(x, x[:, 0])
    assert ols.bootstrap(0, seed=36).shape == (0, 4)
