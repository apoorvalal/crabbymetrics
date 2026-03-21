# crabbymetrics



Rust-backed econometrics 🦀🔢 models with a scikit-adjacent Python API. Focus: extremely low dependency (just numpy), simple, fast estimators with robust standard errors and bootstrap support.

## Features
- OLS, FixedEffectsOLS, SyntheticControl, ElasticNet, Logit, Multinomial Logit, Poisson, TwoSLS, FTRL
- PCA and KernelBasis for feature engineering before regression-style estimation
- `fit`, `predict`, `summary`, `bootstrap`
- HC1 standard errors where applicable

## Install
This package is built with pyo3/maturin and ships as native wheels.

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

## Development

Create and populate the project virtual environment, then build the extension into that venv.

```bash
uv sync
uv run maturin develop
```

`uv run maturin develop` is sufficient for rebuilding and reinstalling the package in `.venv` once the environment exists. If you change Python dependencies or the `pyproject.toml` metadata, run `uv sync` again first.

Package versioning is sourced from `Cargo.toml`. The Python package metadata is dynamic, and `commit_tag_release.sh` reads the crate version directly before creating the `vX.Y.Z` tag.

Rendered examples and API docs live under `docs/`. Rebuild the site with `quarto render docs`.
For the plotting examples, install the docs extra first: `uv sync --extra docs`.

## Wheels
Wheels are platform-specific and included in GitHub releases. See the releases tab.
