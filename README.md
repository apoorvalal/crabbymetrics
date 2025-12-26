# crabbymetrics

Rust-backed econometrics models with a scikit-adjacent Python API. Focus: extremely low dependency (just numpy), simple, fast estimators with robust standard errors and bootstrap support.

## Features
- OLS, ElasticNet, Logit, Multinomial Logit, Poisson, TwoSLS, FTRL
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

model = OLS(fit_intercept=True)
model.fit(x, y)
print(model.summary())
```

## Development
```bash
maturin develop
```

Example scripts live in `examples/`.

## Wheels
Wheels are platform-specific. Prefer publishing wheels via CI; avoid committing wheels to the repo.
