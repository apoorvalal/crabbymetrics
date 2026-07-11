import numpy as np
import pytest
from sklearn.linear_model import Ridge as SklearnRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import crabbymetrics as cm


def test_single_polynomial_learner_matches_sklearn_pipeline():
    rng = np.random.default_rng(20260418)
    x = rng.normal(size=(240, 4))
    y = 0.5 + 0.8 * x[:, 0] * x[:, 1] - 0.4 * x[:, 2] ** 2
    y += rng.normal(scale=0.08, size=x.shape[0])
    penalty = 0.3

    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    expanded = polynomial.fit_transform(x)
    scaler = StandardScaler()
    expanded_scaled = scaler.fit_transform(expanded)
    reference = SklearnRidge(alpha=penalty, fit_intercept=True)
    reference.fit(expanded_scaled, y)

    model = cm.BaggedPolynomialRegressor(
        n_estimators=1,
        degree=2,
        max_features=x.shape[1],
        max_samples=x.shape[0],
        bootstrap=False,
        penalty=penalty,
        seed=11,
    )
    model.fit(x, y)

    np.testing.assert_allclose(model.predict(x), reference.predict(expanded_scaled), atol=1e-9)
    summary = model.summary()
    assert summary["n_terms"] == expanded.shape[1]
    assert summary["max_features"] == x.shape[1]
    assert summary["max_samples"] == x.shape[0]
    assert summary["inference_available"] is False


def test_bagged_polynomial_regressor_is_seed_reproducible_and_uses_subspaces():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(260, 8))
    y = 1.0 + x[:, 0] * x[:, 1] - 0.5 * x[:, 2] ** 2
    y += rng.normal(scale=0.2, size=x.shape[0])

    kwargs = dict(
        n_estimators=30,
        degree=2,
        max_features=4,
        max_samples=180,
        penalty=0.1,
    )
    first = cm.BaggedPolynomialRegressor(seed=77, **kwargs)
    second = cm.BaggedPolynomialRegressor(seed=77, **kwargs)
    different = cm.BaggedPolynomialRegressor(seed=78, **kwargs)
    first.fit(x, y)
    second.fit(x, y)
    different.fit(x, y)

    np.testing.assert_allclose(first.predict(x[:30]), second.predict(x[:30]))
    assert first.summary()["feature_indices"] == second.summary()["feature_indices"]
    assert first.summary()["feature_indices"] != different.summary()["feature_indices"]
    assert not np.allclose(first.predict(x[:30]), different.predict(x[:30]))


def test_bagging_improves_over_raw_ridge_and_reports_oob_diagnostics():
    rng = np.random.default_rng(311)
    x_train = rng.normal(size=(800, 6))
    x_test = rng.normal(size=(350, 6))

    def signal(x):
        return 1.0 + 0.8 * x[:, 0] * x[:, 1] - 0.6 * x[:, 2] ** 2 + 0.4 * x[:, 3]

    y_train = signal(x_train) + rng.normal(scale=0.25, size=x_train.shape[0])
    y_test = signal(x_test) + rng.normal(scale=0.25, size=x_test.shape[0])

    ridge = cm.Ridge(penalty=1.0)
    ridge.fit(x_train, y_train)
    ridge_mse = np.mean((y_test - ridge.predict(x_test)) ** 2)

    model = cm.BaggedPolynomialRegressor(
        n_estimators=60,
        degree=2,
        max_features=4,
        max_samples=600,
        bootstrap=True,
        penalty=0.5,
        seed=13,
    )
    model.fit(x_train, y_train)
    model_mse = np.mean((y_test - model.predict(x_test)) ** 2)
    summary = model.summary()

    assert model_mse < ridge_mse
    assert summary["oob_mse"] is not None
    assert summary["oob_mse"] > 0.0
    assert summary["oob_coverage"] > 0.95
    assert len(set(tuple(indices) for indices in summary["feature_indices"])) > 1


def test_bagged_polynomial_regressor_validates_resources_and_inputs():
    x = np.ones((20, 5))
    y = np.ones(20)

    with pytest.raises(ValueError, match="n_estimators"):
        cm.BaggedPolynomialRegressor(n_estimators=0)
    with pytest.raises(ValueError, match="degree"):
        cm.BaggedPolynomialRegressor(degree=0)
    with pytest.raises(ValueError, match="max_features"):
        cm.BaggedPolynomialRegressor(max_features=6).fit(x, y)
    with pytest.raises(ValueError, match="max_samples"):
        cm.BaggedPolynomialRegressor(max_samples=21).fit(x, y)
    with pytest.raises(ValueError, match="maximum supported"):
        cm.BaggedPolynomialRegressor(
            degree=25,
            max_features=5,
            max_samples=20,
        ).fit(x, y)

    nonfinite = x.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cm.BaggedPolynomialRegressor().fit(nonfinite, y)

    model = cm.BaggedPolynomialRegressor()
    with pytest.raises(ValueError, match="not fitted"):
        model.predict(x)
