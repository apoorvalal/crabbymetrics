#!/usr/bin/env python3
"""Run one isolated scaling cell and emit one JSON object on stdout."""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
import traceback
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from registry import ESTIMATORS

_LAST_FIT_SECONDS: float | None = None


def measured_fit(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Time only the estimator call, excluding deterministic data construction."""

    global _LAST_FIT_SECONDS
    started = time.perf_counter()
    result = function(*args, **kwargs)
    _LAST_FIT_SECONDS = time.perf_counter() - started
    return result


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def tabular_data(n: int, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    x = rng.normal(size=(n, k))
    beta = np.linspace(0.2, 1.0, k) / math.sqrt(k)
    signal = x @ beta
    return {"x": x, "signal": signal, "y": signal + rng.normal(size=n)}


def panel_data(n: int, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    # Here n means periods and k means units, matching the registry's explicit semantics.
    n_controls = max(3, k // 2)
    n_treated = k - n_controls
    control_loadings = rng.normal(size=(n_controls, 2))
    factors = rng.normal(size=(2, n))
    controls = control_loadings @ factors + rng.normal(scale=0.1, size=(n_controls, n))
    if n_treated:
        weights = rng.dirichlet(np.ones(n_controls), size=n_treated)
        treated = weights @ controls + rng.normal(scale=0.05, size=(n_treated, n))
        y = np.vstack([controls, treated])
    else:
        y = controls
    w = np.zeros((k, n), dtype=np.float64)
    w[n_controls:, max(2, n * 2 // 3) :] = 1.0
    return {"y": y, "w": w}


def dynamic_data(n: int, k: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    periods = 4
    units = max(50, n // periods)
    history = rng.normal(size=(units, periods, k))
    logits = 0.15 * history[..., 0]
    treatment = (rng.random((units, periods)) < 1 / (1 + np.exp(-logits))).astype(float)
    innovations = rng.normal(size=(units, periods + 1))
    y_plus = np.cumsum(innovations, axis=1)
    y_plus[:, 1:] += 0.5 * treatment
    return {
        "history": history,
        "treatment": treatment,
        "y_plus": y_plus,
        "y": y_plus[:, 1:],
    }


def result_checksum(model: Any) -> float:
    for name in (
        "coef_",
        "params_",
        "theta_",
        "effect_",
        "att_",
        "weights_",
        "unit_weights_",
        "fitted_values_",
    ):
        if hasattr(model, name):
            values = np.asarray(getattr(model, name), dtype=float).reshape(-1)
            return float(np.nansum(values[: min(values.size, 64)]))
    return 0.0


def run_crabbymetrics(
    estimator: str, n: int, k: int, rng: np.random.Generator
) -> float:
    import crabbymetrics as cm

    if ESTIMATORS[estimator]["family"] == "panel":
        data = panel_data(n, k, rng)
    elif ESTIMATORS[estimator]["family"] == "dynamic":
        data = dynamic_data(n, k, rng)
    else:
        data = tabular_data(n, k, rng)

    x = data.get("x")
    y = data.get("y")

    if estimator == "ABCOLS":
        categories = np.column_stack(
            [np.arange(n, dtype=np.uint32) % 8, np.arange(n, dtype=np.uint32) % 5]
        )
        model = cm.ABCOLS()
        measured_fit(model.fit, y, x, categories)
    elif estimator == "OLS":
        model = cm.OLS()
        measured_fit(model.fit, x, y)
    elif estimator == "FixedEffectsOLS":
        fe = (np.arange(n, dtype=np.uint32) % max(2, min(n // 20, 1000)))[:, None]
        model = cm.FixedEffectsOLS()
        measured_fit(model.fit, x, fe, y)
    elif estimator == "ElasticNet":
        model = cm.ElasticNet(penalty=0.01, l1_ratio=0.5, max_iterations=300)
        measured_fit(model.fit, x, y)
    elif estimator == "Ridge":
        model = cm.Ridge(penalty=1.0)
        measured_fit(model.fit, x, y)
    elif estimator == "BaggedPolynomialRegressor":
        model = cm.BaggedPolynomialRegressor(
            n_estimators=10,
            degree=2,
            max_features=min(k, 12),
            max_samples=min(n, 100_000),
            seed=1729,
        )
        measured_fit(model.fit, x, y)
    elif estimator in {"Logit", "MultinomialLogit"}:
        if estimator == "Logit":
            target = (data["signal"] + rng.logistic(size=n) > 0).astype(np.int32)
            model = cm.Logit(max_iterations=100)
        else:
            target = np.digitize(
                data["signal"] + rng.normal(size=n), [-0.5, 0.5]
            ).astype(np.int32)
            model = cm.MultinomialLogit(max_iterations=100)
        measured_fit(model.fit, x, target)
    elif estimator == "Poisson":
        target = rng.poisson(np.exp(np.clip(data["signal"], -1.5, 1.5))).astype(float)
        model = cm.Poisson(max_iterations=100)
        measured_fit(model.fit, x, target)
    elif estimator in {"ExponentialPH", "WeibullPH", "CoxPH"}:
        time_values = rng.exponential(np.exp(-np.clip(data["signal"], -1, 1))) + 0.01
        event = (rng.random(n) < 0.8).astype(float)
        model = getattr(cm, estimator)()
        measured_fit(model.fit, x, time_values, event)
    elif estimator == "AndersenGill":
        start = rng.uniform(0, 1, n)
        stop = start + rng.exponential(1, n) + 0.01
        event = (rng.random(n) < 0.7).astype(float)
        model = cm.AndersenGill()
        measured_fit(model.fit, x, start, stop, event)
    elif estimator == "TwoSLS":
        z = rng.normal(size=(n, k))
        endog = 0.7 * z[:, :1] + rng.normal(size=(n, 1))
        exog = x
        outcome = endog[:, 0] + data["signal"] + rng.normal(size=n)
        model = cm.TwoSLS()
        measured_fit(model.fit, endog, exog, z, outcome)
    elif estimator == "HorizontalPanelRidge":
        pd = panel_data(max(8, n), max(4, k), rng)
        model = cm.HorizontalPanelRidge(penalty=1.0)
        measured_fit(model.fit, pd["y"], pd["w"])
    elif estimator == "SyntheticControl":
        donors = rng.normal(size=(n, k))
        treated = donors @ (np.ones(k) / k) + rng.normal(scale=0.1, size=n)
        model = cm.SyntheticControl(max_iterations=300)
        measured_fit(model.fit, donors, treated)
    elif estimator == "SyntheticDID":
        model = cm.SyntheticDID(zeta_omega=0.01, zeta_lambda=0.01, max_iterations=3000)
        measured_fit(model.fit, data["y"], data["w"])
    elif estimator == "AugmentedBalancing":
        model = cm.AugmentedBalancing(
            zeta_omega=0.01, zeta_lambda=0.01, max_iterations=3000
        )
        measured_fit(model.fit, data["y"], data["w"])
    elif estimator == "MatrixCompletion":
        model = cm.MatrixCompletion(max_iterations=50, svd_rank=min(6, k - 1))
        measured_fit(model.fit, data["y"], data["w"])
    elif estimator == "InteractiveFixedEffects":
        model = cm.InteractiveFixedEffects(rank=min(2, k - 1))
        measured_fit(model.fit, data["y"])
    elif estimator == "BalancingWeights":
        split = max(2, n * 4 // 5)
        model = cm.BalancingWeights(max_iterations=100)
        measured_fit(model.fit, x[:split], x[split:])
    elif estimator == "MEstimator":
        design = np.column_stack([np.ones(n), x])

        def objective(
            theta: np.ndarray, payload: dict[str, np.ndarray]
        ) -> tuple[float, np.ndarray]:
            residual = payload["y"] - payload["x"] @ theta
            gradient = -(payload["x"].T @ residual) / payload["x"].shape[0]
            return float(np.mean(residual**2) / 2), gradient

        def score(theta: np.ndarray, payload: dict[str, np.ndarray]) -> np.ndarray:
            residual = payload["y"] - payload["x"] @ theta
            return payload["x"] * residual[:, None]

        model = cm.MEstimator(objective, score, max_iterations=30)
        measured_fit(model.fit, {"x": design, "y": y}, np.zeros(k + 1))
    elif estimator == "GMM":
        design = np.column_stack([np.ones(n), x])

        def moments(theta: np.ndarray, payload: dict[str, np.ndarray]) -> np.ndarray:
            residual = payload["y"] - payload["x"] @ theta
            return payload["x"] * residual[:, None]

        def jacobian(theta: np.ndarray, payload: dict[str, np.ndarray]) -> np.ndarray:
            del theta
            return -(payload["x"].T @ payload["x"]) / payload["x"].shape[0]

        model = cm.GMM(moments, jacobian, max_iterations=30)
        measured_fit(
            model.fit, {"x": design, "y": y}, np.zeros(k + 1), weighting="identity"
        )
    elif estimator in {"EPLM", "AverageDerivative", "PartiallyLinearDML", "AIPW"}:
        d_cont = 0.25 * data["signal"] + rng.normal(size=n)
        if estimator == "AIPW":
            treatment = (d_cont > np.median(d_cont)).astype(float)
            model = cm.AIPW(penalty=0.1, n_folds=2, seed=1729)
        elif estimator == "PartiallyLinearDML":
            treatment = d_cont
            model = cm.PartiallyLinearDML(penalty=0.1, n_folds=2, seed=1729)
        elif estimator == "AverageDerivative":
            treatment = d_cont
            model = cm.AverageDerivative(method="dr")
        else:
            treatment = d_cont
            model = cm.EPLM()
        outcome = 0.8 * treatment + data["signal"] + rng.normal(size=n)
        measured_fit(model.fit, outcome, treatment, x)
    elif estimator == "DynamicCovariateBalance":
        model = cm.DynamicCovariateBalance(max_iterations=100)
        measured_fit(
            model.fit,
            data["y_plus"][:, -1],
            data["treatment"],
            data["history"],
            [0.0] * data["treatment"].shape[1],
        )
    elif estimator == "ParallelTrendsSNMM":
        model = cm.ParallelTrendsSNMM(n_folds=2, seed=1729)
        measured_fit(model.fit, data["y_plus"], data["treatment"], data["history"])
    elif estimator == "RegressionBlip":
        model = cm.RegressionBlip()
        measured_fit(model.fit, data["y"], data["treatment"], data["history"])
    else:  # pragma: no cover - inventory test guards this
        raise KeyError(estimator)
    return result_checksum(model)


def run_sklearn(implementation: str, n: int, k: int, rng: np.random.Generator) -> float:
    from sklearn.ensemble import BaggingRegressor
    from sklearn.linear_model import (
        ElasticNet,
        LinearRegression,
        LogisticRegression,
        PoissonRegressor,
        Ridge,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    data = tabular_data(n, k, rng)
    x, y = data["x"], data["y"]
    if implementation == "sklearn-linear-regression":
        model = measured_fit(LinearRegression().fit, x, y)
    elif implementation == "sklearn-ridge":
        model = measured_fit(Ridge(alpha=1.0).fit, x, y)
    elif implementation == "sklearn-elastic-net":
        model = measured_fit(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=300).fit, x, y
        )
    elif implementation == "sklearn-poisson-regressor":
        target = rng.poisson(np.exp(np.clip(data["signal"], -1.5, 1.5)))
        model = measured_fit(PoissonRegressor(max_iter=100).fit, x, target)
    elif implementation in {"sklearn-logistic-regression", "sklearn-multinomial-logit"}:
        if implementation == "sklearn-logistic-regression":
            target = (data["signal"] + rng.logistic(size=n) > 0).astype(int)
        else:
            target = np.digitize(data["signal"] + rng.normal(size=n), [-0.5, 0.5])
        model = measured_fit(
            LogisticRegression(max_iter=100, solver="lbfgs").fit, x, target
        )
    elif implementation == "sklearn-bagged-polynomial":
        base = make_pipeline(
            PolynomialFeatures(2, include_bias=False), StandardScaler(), Ridge()
        )
        model = measured_fit(
            BaggingRegressor(
                estimator=base,
                n_estimators=10,
                max_samples=min(1.0, 100_000 / n),
                max_features=min(1.0, 12 / k),
                random_state=1729,
                n_jobs=1,
            ).fit,
            x,
            y,
        )
    else:
        raise KeyError(implementation)
    if hasattr(model, "coef_"):
        return float(np.sum(np.asarray(model.coef_)))
    return float(
        np.sum(np.asarray(getattr(model, "estimators_", []), dtype=object).size)
    )


def run_pyfixest(
    implementation: str, n: int, k: int, rng: np.random.Generator
) -> float:
    import pandas as pd
    import pyfixest as pf

    data = tabular_data(n, k, rng)
    frame = pd.DataFrame(data["x"], columns=[f"x{j}" for j in range(k)])
    frame["y"] = data["y"]
    x_terms = "+".join(f"x{j}" for j in range(k))
    if implementation == "pyfixest-feols":
        frame["fe"] = np.arange(n) % max(2, min(n // 20, 1000))
        model = measured_fit(pf.feols, f"y~{x_terms}|fe", data=frame, vcov="iid")
    elif implementation == "pyfixest-iv":
        z = rng.normal(size=(n, k))
        z_names = [f"z{j}" for j in range(k)]
        frame = pd.concat([frame, pd.DataFrame(z, columns=z_names)], axis=1)
        frame["d"] = 0.7 * z[:, 0] + rng.normal(size=n)
        frame["y"] += frame["d"]
        z_terms = "+".join(z_names)
        model = measured_fit(
            pf.feols, f"y~{x_terms}|d~{z_terms}", data=frame, vcov="iid"
        )
    else:
        raise KeyError(implementation)
    return float(np.sum(np.asarray(model.coef())))


def run_lifelines(n: int, k: int, rng: np.random.Generator) -> float:
    import pandas as pd
    from lifelines import CoxPHFitter

    data = tabular_data(n, k, rng)
    frame = pd.DataFrame(data["x"], columns=[f"x{j}" for j in range(k)])
    frame["time"] = rng.exponential(np.exp(-np.clip(data["signal"], -1, 1))) + 0.01
    frame["event"] = rng.random(n) < 0.8
    model = measured_fit(
        CoxPHFitter().fit, frame, duration_col="time", event_col="event"
    )
    return float(model.params_.sum())


def run_doubleml(
    implementation: str, n: int, k: int, rng: np.random.Generator
) -> float:
    import doubleml as dml
    from doubleml.utils import PSProcessorConfig
    from sklearn.linear_model import LogisticRegression, Ridge

    data = tabular_data(n, k, rng)
    d_cont = 0.25 * data["signal"] + rng.normal(size=n)
    if implementation == "doubleml-irm":
        treatment = (d_cont > np.median(d_cont)).astype(float)
        outcome = 0.8 * treatment + data["signal"] + rng.normal(size=n)
        dml_data = dml.DoubleMLData.from_arrays(data["x"], outcome, treatment)
        model = measured_fit(
            dml.DoubleMLIRM(
                dml_data,
                ml_g=Ridge(alpha=0.1),
                ml_m=LogisticRegression(max_iter=100),
                n_folds=2,
                ps_processor_config=PSProcessorConfig(clipping_threshold=0.02),
            ).fit
        )
    elif implementation == "doubleml-plr":
        outcome = 0.8 * d_cont + data["signal"] + rng.normal(size=n)
        dml_data = dml.DoubleMLData.from_arrays(data["x"], outcome, d_cont)
        model = measured_fit(
            dml.DoubleMLPLR(
                dml_data,
                ml_l=Ridge(alpha=0.1),
                ml_m=Ridge(alpha=0.1),
                n_folds=2,
            ).fit
        )
    else:
        raise KeyError(implementation)
    return float(np.sum(np.asarray(model.coef)))


PYTHON_RUNNERS: dict[str, Callable[[int, int, np.random.Generator], float]] = {
    name: (
        lambda n, k, rng, implementation=name: run_sklearn(implementation, n, k, rng)
    )
    for name in (
        "sklearn-linear-regression",
        "sklearn-ridge",
        "sklearn-elastic-net",
        "sklearn-bagged-polynomial",
        "sklearn-logistic-regression",
        "sklearn-multinomial-logit",
        "sklearn-poisson-regressor",
    )
}
PYTHON_RUNNERS.update(
    {
        "pyfixest-feols": lambda n, k, rng: run_pyfixest("pyfixest-feols", n, k, rng),
        "pyfixest-iv": lambda n, k, rng: run_pyfixest("pyfixest-iv", n, k, rng),
        "lifelines-cox-ph": run_lifelines,
        "doubleml-plr": lambda n, k, rng: run_doubleml("doubleml-plr", n, k, rng),
        "doubleml-irm": lambda n, k, rng: run_doubleml("doubleml-irm", n, k, rng),
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", required=True, choices=ESTIMATORS)
    parser.add_argument("--implementation", required=True)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result: dict[str, Any] = {
        "estimator": args.estimator,
        "implementation": args.implementation,
        "n": args.n,
        "k": args.k,
        "python": platform.python_version(),
        "status": "ok",
    }
    rng = np.random.default_rng(args.seed)
    try:
        cell_started = time.perf_counter()
        if args.implementation == "crabbymetrics":
            checksum = run_crabbymetrics(args.estimator, args.n, args.k, rng)
            result["library_version"] = package_version("crabbymetrics")
        elif args.implementation in PYTHON_RUNNERS:
            checksum = PYTHON_RUNNERS[args.implementation](args.n, args.k, rng)
            package = {
                "sklearn": "scikit-learn",
                "pyfixest": "pyfixest",
                "lifelines": "lifelines",
                "doubleml": "doubleml",
            }[args.implementation.split("-", 1)[0]]
            result["library_version"] = package_version(package)
        elif args.implementation.startswith("r-"):
            raise RuntimeError("R references are dispatched by reference_runner.R")
        else:
            raise RuntimeError(
                "reference is provenance-only or lacks an exact runnable comparator"
            )
        result["cell_seconds"] = time.perf_counter() - cell_started
        if _LAST_FIT_SECONDS is None:
            raise RuntimeError("benchmark adapter did not mark an estimator fit call")
        result["fit_seconds"] = _LAST_FIT_SECONDS
        result["checksum"] = checksum
    except (ImportError, PackageNotFoundError) as exc:
        result.update(status="missing_dependency", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - one-cell process must preserve failures in-grid
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc(limit=6)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
