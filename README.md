# crabbymetrics

[![Tests](https://github.com/apoorvalal/crabbymetrics/actions/workflows/tests.yml/badge.svg)](https://github.com/apoorvalal/crabbymetrics/actions/workflows/tests.yml)
[![Build wheels](https://github.com/apoorvalal/crabbymetrics/actions/workflows/wheels.yml/badge.svg)](https://github.com/apoorvalal/crabbymetrics/actions/workflows/wheels.yml)
[![PyPI version](https://img.shields.io/pypi/v/crabbymetrics.svg)](https://pypi.org/project/crabbymetrics/)

<p align="center">
  <img src="docs/logo.png" alt="crabbymetrics logo" width="720">
</p>

Rust-backed econometrics 🦀🔢 models with a scikit-adjacent Python API. Focus: extremely low runtime dependency footprint, simple NumPy-facing estimators, robust standard errors, and bootstrap support where it fits the estimator.

## Features
- Linear, IV, and panel causal estimators: OLS, Ridge, FixedEffectsOLS, TwoSLS, SyntheticControl, SyntheticDID, MatrixCompletion, InteractiveFixedEffects
- Regularized and likelihood estimators: ElasticNet, Logit, Multinomial Logit, Poisson, FTRL
- Moment and semiparametric estimators: GMM, BalancingWeights, EPLM, AverageDerivative, PartiallyLinearDML, AIPW
- Shared robust covariance options for the main linear estimators: vanilla, HC1, Newey-West, and cluster
- Weighted fits for OLS, Ridge, FixedEffectsOLS, and TwoSLS
- `ElasticNet` spans the ridge and lasso corners: use `l1_ratio=0.0` for ridge-style shrinkage and `l1_ratio=1.0` for lasso-style shrinkage
- PCA and KernelBasis for feature engineering before regression-style estimation
- `Optimizers` namespace exposing LBFGS, BFGS, NonlinearConjugateGradient, Gauss-Newton least squares, and SimulatedAnnealing
- `fit`, `predict`, `summary`, and `bootstrap` where meaningful for the estimator

## Install
This package is built with pyo3/maturin and ships as native wheels.

PyPI: <https://pypi.org/project/crabbymetrics/>

```bash
uv pip install crabbymetrics
```

## Example
```python
import numpy as np
from crabbymetrics import OLS

x = np.random.randn(200, 3)
beta = np.array([1.0, -2.0, 0.5])
y = 0.3 + x @ beta + np.random.randn(200) * 0.1

model = OLS()
model.fit(x, y)
print(model.summary())
```

The direct optimizer wrappers live under `Optimizers` and follow a lightweight scipy-style interface:

```python
import numpy as np
from crabbymetrics import Optimizers

def objective(theta):
    return float((theta[0] - 1.0) ** 2 + 2.0 * (theta[1] + 2.0) ** 2)

def gradient(theta):
    return np.array([2.0 * (theta[0] - 1.0), 4.0 * (theta[1] + 2.0)])

result = Optimizers.minimize_lbfgs(objective, np.array([4.0, 3.0]), gradient)
print(result["x"], result["fun"])
```

## Benchmarks

The latest cross-library runtime snapshot is checked in as [`benchmarks/runtime_comparison.png`](benchmarks/runtime_comparison.png).

![Runtime comparison across crabbymetrics, scikit-learn, and statsmodels](benchmarks/runtime_comparison.png)

This benchmark used synthetic problems with `p=5`, sample sizes from `10^3` to `10^6`, fit-only timing, and a 45-second per-fit timeout.

- `OLS` is competitive already and was faster than both scikit-learn and statsmodels at `n=10^6`.
- `Poisson` beats statsmodels comfortably but still trails scikit-learn at larger `n`.
- `Logit` and especially `MultinomialLogit` are the main performance gaps to close before adding more iterative GLM-style estimators.

## Development

Create and populate the project virtual environment, then build the extension into that venv.

```bash
uv sync
uv run maturin develop
```

`uv run maturin develop` is sufficient for rebuilding and reinstalling the package in `.venv` once the environment exists. If you change Python dependencies or the `pyproject.toml` metadata, run `uv sync` again first.

Package versioning is sourced from `Cargo.toml`. The Python package metadata is dynamic, and `commit_tag_release.sh` reads the crate version directly before creating the `vX.Y.Z` tag.

Rendered examples and API docs live under `docs/`. Rebuild the site with `uv run quarto render docs`.
For docs work, install the docs extra first: `uv sync --extra docs`.

## Wheels
Wheels are platform-specific and included in GitHub releases. See the releases tab.
