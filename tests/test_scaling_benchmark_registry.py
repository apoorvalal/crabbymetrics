from __future__ import annotations

import importlib.util
from pathlib import Path

import crabbymetrics as cm


def load_registry():
    path = Path(__file__).parents[1] / "benchmarks" / "scaling" / "registry.py"
    spec = importlib.util.spec_from_file_location("scaling_registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaling_registry_covers_every_exported_estimator():
    registry = load_registry()
    expected = {
        "ABCOLS",
        "OLS",
        "FixedEffectsOLS",
        "ElasticNet",
        "Ridge",
        "BaggedPolynomialRegressor",
        "Logit",
        "MultinomialLogit",
        "Poisson",
        "ExponentialPH",
        "WeibullPH",
        "CoxPH",
        "AndersenGill",
        "TwoSLS",
        "HorizontalPanelRidge",
        "SyntheticControl",
        "SyntheticDID",
        "AugmentedBalancing",
        "MatrixCompletion",
        "InteractiveFixedEffects",
        "BalancingWeights",
        "MEstimator",
        "GMM",
        "EPLM",
        "AverageDerivative",
        "PartiallyLinearDML",
        "AIPW",
        "DynamicCovariateBalance",
        "ParallelTrendsSNMM",
        "RegressionBlip",
    }
    assert set(registry.ESTIMATORS) == expected
    assert all(hasattr(cm, name) for name in expected)
    assert all(
        reference in registry.REFERENCE_URLS
        for spec in registry.ESTIMATORS.values()
        for reference in spec["references"]
    )
