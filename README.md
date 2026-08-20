# crabbymetrics

[![Tests](https://github.com/apoorvalal/crabbymetrics/actions/workflows/tests.yml/badge.svg)](https://github.com/apoorvalal/crabbymetrics/actions/workflows/tests.yml)
[![Build wheels](https://github.com/apoorvalal/crabbymetrics/actions/workflows/wheels.yml/badge.svg)](https://github.com/apoorvalal/crabbymetrics/actions/workflows/wheels.yml)
[![PyPI version](https://img.shields.io/pypi/v/crabbymetrics.svg)](https://pypi.org/project/crabbymetrics/)

<p align="center">
  <img src="docs/logo.png" alt="crabbymetrics logo" width="720">
</p>

Rust-backed econometrics 🦀🔢 models with a scikit-adjacent Python API. Focus: extremely low runtime dependency footprint, simple NumPy-facing estimators, robust standard errors, and bootstrap support where it fits the estimator.

## Features
- Linear, IV, and panel causal estimators: OLS, Ridge, FixedEffectsOLS, TwoSLS, HorizontalPanelRidge, SyntheticControl, SyntheticDID, AugmentedBalancing, MatrixCompletion, InteractiveFixedEffects
- Common panel causal API: `HorizontalPanelRidge`, `SyntheticDID`, `AugmentedBalancing`, and `MatrixCompletion` use balanced outcome and absorbing treatment matrices, then expose ATT, counterfactuals, treatment effects, event-study summaries, and group means through `summary()`
- Regularized and likelihood estimators: ElasticNet, Logit, Multinomial Logit, Poisson
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

Anytime-valid OLS inference is available from `OLS.summary(...)`:

```python
import numpy as np
import crabbymetrics as cm

rng = np.random.default_rng(1)
n = 100
x = rng.normal(size=(n, 3))
x = x - x.mean(axis=0)
trt = rng.choice([0.0, 1.0], size=n)
y = (
    1.0
    + 1.4 * x[:, 2]
    + 2.3 * trt
    + 2.0 * x[:, 0] * trt
    + 3.0 * x[:, 1] * trt
    + rng.normal(size=n)
)
design = np.column_stack([x, trt, x * trt[:, None]])

model = cm.OLS()
model.fit(design, y)
g_star = cm.optimal_g(n, design.shape[1] + 1, alpha=0.05)
summary = model.summary(vcov="vanilla", anytime_valid=True, g=g_star)
print(summary["p_value"])
print(summary["confint"])
```

Panel causal estimators take matrices directly rather than long data frames:

```python
import numpy as np
import crabbymetrics as cm

Y = np.random.randn(20, 12)
W = np.zeros_like(Y)
W[15:, 8:] = 1.0  # absorbing treatment matrix

model = cm.SyntheticDID()
model.fit(Y, W)
out = model.summary()
print(out["att"], out["event_study"].keys(), out["group_means"].keys())
```

`BalancingWeights` remains available as the lower-level calibration/reweighting API, but the paved panel path is estimator-first.

`AugmentedBalancing.fit(Y, W, outcome_model=None)` composes supplied untreated-outcome predictions with unit or unit-and-time residual balancing. Constructor options select cohort versus individual unit-weight targets and raw versus residualized weight fitting.

The [Augmented Balancing vignette](https://apoorvalal.github.io/crabbymetrics/examples/augmented-balancing.html) shows a staggered-adoption workflow with a matrix-completion nuisance surface, estimator comparisons, event-time plots, and weight-target diagnostics. The [class reference](https://apoorvalal.github.io/crabbymetrics/reference/AugmentedBalancing.html) documents every constructor option and `summary()` field.

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

Package versioning is sourced from `Cargo.toml`; the Python package metadata is dynamic.

To release to PyPI, use the **Build wheels** GitHub Actions workflow manually from `master` and enter the next version without the leading `v` (for example `0.6.1`). The workflow bumps `Cargo.toml`, commits to `master`, creates the `vX.Y.Z` tag, runs tests, builds wheels and an sdist, publishes the GitHub Release, and publishes to PyPI. The older `commit_tag_release.sh` script remains as a local fallback for manually tagging the current `Cargo.toml` version.

Rendered examples and API docs live under `docs/`. Rebuild the site with `uv run quarto render docs`.
For docs work, install the docs extra first: `uv sync --extra docs`.

## Wheels
Wheels are platform-specific and included in GitHub releases. See the releases tab.
