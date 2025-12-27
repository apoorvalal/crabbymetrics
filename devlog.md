# Devlog (crabbymetrics)

## Overview
`crabbymetrics` is a Rust-backed econometrics package exposed to Python via `pyo3`/`maturin`. The goal is a scikit-adjacent API with good numerical performance and clear ergonomics. The current focus is a base set of econometrics-style estimators with robust standard errors and bootstrap support.

This devlog is intended to capture the current design, implementation decisions, and known caveats so future work can continue without re-reading all source history.

---

## Repository layout

```
crabbymetrics/
  Cargo.toml
  README.md
  devlog.md
  commit_tag_release.sh
  examples/
  .github/workflows/wheels.yml
  src/
    lib.rs
    estimators.rs
    utils.rs
```

- `src/lib.rs` is the pyo3 entrypoint that registers all Python classes.
- `src/estimators.rs` contains all estimator classes and their `fit/predict/summary/bootstrap` methods.
- `src/utils.rs` contains shared linear algebra, SE, bootstrap, and numpy conversion utilities.
- `examples/` holds simple scripts for each estimator, using simulated data with known parameters.

---

## Packaging model

- `maturin` builds native wheels. Wheels are platform-specific; CI builds for Linux/macOS/Windows.
- GitHub Actions workflow `wheels.yml` builds wheels on tags and attaches them to GitHub Releases.
- Do not commit wheels into the repo. Release artifacts are attached via CI.

---

## Core API (Python)

Each estimator exposes a scikit-adjacent API:

- `__init__(...)` constructor with hyperparameters.
- `fit(X, y)` (or `fit(x_endog, x_exog, z, y)` for TwoSLS).
- `predict(X)` returns predictions or probabilities.
- `summary()` returns a dict with coefficients and standard errors.
- `bootstrap(n_bootstrap, seed=None)` returns a matrix of bootstrap coefficient draws.

The summary output uses consistent keys when applicable:
- `intercept`: float
- `coef`: numpy vector (or matrix for multinomial)
- `intercept_se`: float or None
- `coef_se`: numpy vector (or matrix for multinomial)

---

## Estimators implemented

### OLS
- Uses `linfa-linear::LinearRegression` for fitting.
- HC1 robust covariance for SEs.
- `summary` returns intercept, coefficients, and HC1 SEs.

### ElasticNet
- Uses `linfa-elasticnet`.
- HC1 robust covariance on residuals (not a full ridge/L1-specific covariance; intended as a reasonable default).

### Logit (binary)
- Uses `linfa-logistic`.
- SEs are derived from the Fisher information (inverse of X'W X) using predicted probabilities.

### MultinomialLogit
- Uses `linfa-logistic` multi-class.
- SEs are derived from an explicit Fisher information matrix for all class coefficients.
- `summary()` returns coefficient and SE matrices of shape `(classes, features_with_intercept)`.

### Poisson
- Custom Poisson MLE using `argmin` Newton-CG with analytic gradient and Hessian.
- Log-likelihood (up to constant) per observation:
  - `y_i * (x_i'β) - exp(x_i'β)`
- Gradient:
  - `X' (exp(η) - y)`
- Hessian:
  - `X' diag(exp(η)) X` (plus optional L2 ridge on diagonal)
- Intercept handled by augmenting parameter vector.
- SEs from Fisher information: inverse of `X' W X` where `W = exp(η)`.

### TwoSLS (single endogenous)
- Stage 1: regress endog on instruments (+ exog) using OLS.
- Stage 2: regress y on predicted endog (+ exog) using OLS.
- Stores the original arrays to recompute SEs and bootstrap.
- TODO note in code: extend to multi-endogenous and GMM.

### FTRL
- Uses `linfa-ftrl` for classification.
- Returns weights and SEs from Fisher information based on predicted probabilities.

### M-Estimator Implementation - Development Log


Implemented general M-estimation framework allowing users to define arbitrary objective functions in Python and optimize them using Rust's argmin L-BFGS solver.

#### Core Implementation
-  Python callbacks for objective function: `(theta, data) -> (obj, grad)`
-  Python callbacks for per-observation scores: `(theta, data) -> (n_obs, n_params) array`
-  L-BFGS optimization via argmin (7 history vectors, More-Thuente line search)
-  Sandwich variance: A^{-1} B A^{-1} using BFGS approximation
-  Bootstrap inference via re-optimization

#### Validation
Tested on Poisson regression:
- Coefficients match built-in Poisson to <0.01% relative error
- Standard errors match to ~3.4% relative error
- Bootstrap produces reasonable distributions

Initial implementation failed spectacularly - optimizer was taking enormous steps (e.g., theta jumping from [0.1, 0.1, 0.1] to [76, 304, -554]) causing:
- Overflow in `exp(eta)` for Poisson likelihood
- NaN gradients
- Optimizer giving up and returning initial values

**This was silently failing** - no errors, just stuck parameters.

User must add safeguards in their objective function:

```python
def poisson_objective(theta, data):
    eta = X @ theta
    # CRITICAL: Clip to prevent overflow
    eta = np.clip(eta, -20, 20)
    mu = np.exp(eta)
    # ... rest of calculation
```

- L-BFGS doesn't know about domain constraints of your likelihood
- Line search can propose arbitrarily large steps
- For exponential family models, this causes immediate numerical failure
- The failure mode is silent - optimization just returns initial values

1. Use trust-region methods instead of line search (requires Hessian)
2. Add explicit box constraints to solver (argmin supports this but adds complexity)
3. Transform parameters (e.g., log-transform positive params)
4. Better initialization and scaling

For now, clipping in the objective is simplest and works well.

#### API Design Decisions

1. Per-observation scores (not batch)

```python
def score_fn(theta, data):
    # Must return (n_obs, n_params)
    return scores  # Shape: (n, p)
```

This enables sandwich variance computation where we need the full score matrix to compute B = (1/n) sum_i score_i score_i'.

2. Data as dict

```python
data = {'X': X, 'y': y, 'n': n}
model.fit(data, theta0)
```

Flexible format. For bootstrap, automatically adds `'indices'` key:
```python
# In objective function:
indices = data.get('indices', np.arange(len(y)))
X_sample = X[indices]
```

3. No automatic differentiation
Users must provide gradients. Rationale:
- They can use JAX externally if they want
- Gives full control over numerical stability
- Avoids adding heavy dependencies
- Forces users to think about their gradient (often catches bugs)

#### Key Rust Components

**Callback handling:**
```rust
impl CostFunction for MEstimatorProblem {
    fn cost(&self, theta: &Array1<f64>) -> Result<f64, Error> {
        Python::with_gil(|py| {
            let result = self.objective_fn.call1(py, (theta_py, data))?;
            let tuple = result.downcast_bound::<PyTuple>(py)?;
            let obj: f64 = tuple.get_item(0)?.extract()?;
            Ok(obj)
        })
    }
}
```

**Sandwich variance:**
```rust
// A = B = (1/n) scores' * scores (BFGS approximation)
let b_matrix = scores.t().dot(&scores) / (n as f64);
let a_matrix = b_matrix.clone();
let a_inv = invert_matrix(&a_matrix)?;
let vcov = a_inv.dot(&b_matrix).dot(&a_inv) / (n as f64);
```

#### Known Limitations

1. **Tolerance parameter not used**: Stored but not passed to LBFGS config (uses solver defaults)

2. **BFGS approximation for A matrix**: Uses outer product of scores instead of actual Hessian. Could optionally accept Hessian callback.

3. **No convergence diagnostics**: Doesn't report final gradient norm, iteration count, or convergence status.

4. **Bootstrap is slow**: Re-optimizes on every sample. Could implement fast score bootstrap using influence functions.

5. **Limited solver options**: Only L-BFGS. Could expose Newton-CG (requires Hessian), trust region, etc.

#### Usage Example

```python
import numpy as np
from crabbymetrics import MEstimator

def poisson_objective(theta, data):
    X, y = data['X'], data['y']
    indices = data.get('indices', np.arange(len(y)))

    eta = (X[indices] @ theta)
    eta = np.clip(eta, -20, 20)  # CRITICAL for numerical stability
    mu = np.exp(eta)

    obj = np.sum(mu - y[indices] * eta)
    grad = X[indices].T @ (mu - y[indices])
    return obj, grad

def poisson_scores(theta, data):
    X, y = data['X'], data['y']
    eta = X @ theta
    eta = np.clip(eta, -20, 20)  # Match objective clipping
    mu = np.exp(eta)
    return X * (mu - y)[:, np.newaxis]  # Shape: (n, p)

# Fit
model = MEstimator(poisson_objective, poisson_scores, max_iterations=200)
model.fit({'X': X, 'y': y, 'n': len(y)}, theta0=np.zeros(X.shape[1]))

# Inference
summary = model.summary()
print(f"Coef: {summary['coef']}")
print(f"SE:   {summary['se']}")

# Bootstrap
boots = model.bootstrap(n_bootstrap=100, seed=42)
```


#### High Priority
- [ ] Wire tolerance parameter to LBFGS config
- [ ] Add convergence diagnostics to summary
- [ ] Document the clipping requirement prominently
- [ ] Add more robust default behavior (maybe auto-detect NaN and warn?)

#### Medium Priority
- [ ] Fast score bootstrap using influence functions
- [ ] Option to accept Hessian callback for more accurate variance
- [ ] Expose more solver options (trust region, constraints, etc.)
- [ ] Add example with constraints

#### Low Priority
- [ ] JAX integration for auto-differentiation
- [ ] Parallel bootstrap
- [ ] Progress callbacks during optimization
- [ ] Warm-start capability for bootstrap

---

## Utility design

### ndarray version management
- `linfa` currently depends on `ndarray 0.16.x`, while `numpy` crate prefers newer versions.
- To avoid a version mismatch, numpy arrays are **manually copied** into `ndarray 0.16` structures via helper functions:
  - `to_array1`, `to_array1_i32`, `to_array2`
- Outputs are returned using `PyArray::from_vec` or `PyArray::from_vec2` to avoid `IntoPyArray` from numpy’s ndarray version.

### Robust SEs
- HC1 covariance used where applicable:
  - `hc1_cov`: `V = (X'X)^{-1} X' diag(u^2) X (X'X)^{-1} * n/(n-k)`
- Fisher information for Logit/Poisson/MultinomialLogit:
  - `fisher_cov_binary`, `fisher_cov_poisson`, `fisher_cov_multinomial`

### Matrix inversion
- Implemented via `nalgebra` (`DMatrix::try_inverse()`).
- Avoids hand-coded Gaussian elimination.

### Bootstrap
- `bootstrap_indices` generates row index draws with replacement.
- Each estimator re-fits on bootstrap samples and collects coefficient draws.
- Output shape: `(n_bootstrap, n_params)` (or flattened for multinomial).

---

## Conventions and defaults

- Intercepts are optional via `fit_intercept` in each model.
- Poisson defaults `alpha=0.0` (no ridge); can be set for stability.
- MLE solvers use `max_iterations` and `tolerance` fields; Poisson uses Newton-CG with a line search.

---

## Example scripts

`examples/` includes quick end-to-end demos with synthetic data:
- `ols_example.py`
- `elastic_net_example.py`
- `logit_example.py`
- `multinomial_logit_example.py`
- `poisson_example.py`
- `twosls_example.py`
- `ftrl_example.py`

Each script:
- simulates data with known parameters
- fits the estimator
- prints `summary()`

---

## CI and releases

- `wheels.yml` builds wheels for Linux/macOS/Windows and Python 3.10–3.12.
- Tagging `vX.Y.Z` triggers release and attaches wheels to a GitHub Release.
- Workflow includes `permissions: contents: write` so the GITHUB_TOKEN can create releases.

---

## Known quirks / caveats

- Multinomial Logit SEs can be very large for some synthetic configurations; this likely reflects weak class separation or poorly conditioned Fisher matrices.
- Poisson previously hung when wrapping TweedieRegressor; now uses explicit MLE via argmin.
- FTRL coefficients are on a different scale; no intercept term used.
- For robust inference, bootstrap is available on all estimators.

---

## To revisit

- Add Ridge closed-form and Lasso-specific solvers.
- Extend TwoSLS to multi-endogenous and GMM.
- Improve MLE diagnostics (gradient norms, convergence status) for logit/poisson/multinomial.
- Consider `abi3` wheels to reduce per-Python builds.
- Add formal tests (pytest) from example scripts.

---

## Build / dev commands

```
# Build and install into current venv
maturin develop

# Run example scripts
.venv/bin/python crabbymetrics/examples/ols_example.py
```

---

## Release helper script

`commit_tag_release.sh`:
- `./commit_tag_release.sh 0.0.1` will commit, push, tag `v0.0.1`, and push the tag to trigger release.
