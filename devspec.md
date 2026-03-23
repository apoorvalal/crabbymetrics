# crabbymetrics Extension Dev Spec

This file is just bookkeeping for future extension work. It is not a release checklist and it is not meant to freeze the API. The goal is to keep a parallel queue of ideas that fit the current philosophy of the library:

- minimal runtime dependency footprint
- Rust owns the numerical work
- Python stays a thin NumPy-facing binding layer
- estimators should keep the current `fit` / `predict` / `summary` pattern when that makes sense
- avoid bringing in pandas, patsy, scipy, or a formula system just to mimic larger libraries

## Design Guardrails

Any new work here should usually satisfy most of the following:

1. The hot path should live in Rust, not inside a Python callback loop.
2. New estimators should prefer closed form or simple iterative methods already compatible with the current stack.
3. If a feature mostly requires covariance bookkeeping rather than a new point estimator, prefer extending `summary(...)` over adding a whole new class.
4. Public APIs should continue to take NumPy arrays and return plain dictionaries or NumPy arrays.
5. If docs examples are numerically heavy, they should use Quarto caching and freeze.

## Highest-Value Extensions

### 1. Unified robust inference across linear estimators

Status: not started

Why it matters:

- `GMM` already has the most flexible covariance surface.
- `OLS`, `Ridge`, `TwoSLS`, and `FixedEffectsOLS` still have narrower inference support.
- This is high value without adding any large external dependency.

Scope:

- extend `summary(vcov=...)` for linear estimators to cover:
  - `vanilla`
  - `hc1`
  - `newey_west`
  - `cluster`
- use shared Rust helpers for bread / meat / sandwich assembly
- support one-way clustering first

Success condition:

- same covariance interface across `OLS`, `Ridge`, `TwoSLS`, and `FixedEffectsOLS`
- tests against hand-coded NumPy references

### 2. Difference-in-differences and event-study estimators

Status: not started

Why it matters:

- this library already has a workable fixed-effects backbone
- DiD / event-study fits the econometrics focus better than adding generic ML breadth
- these estimators are useful and still compatible with the low-dependency design

Scope:

- two-period / two-group DiD as a simple starting estimator
- then staggered-adoption event-study design-matrix helpers
- estimation can ride on top of the existing linear / fixed-effects machinery

Success condition:

- a small Rust-backed `DifferenceInDifferences` or `EventStudy` class
- summary exposes coefficient tables and robust standard errors

### 3. Negative binomial regression

Status: not started

Why it matters:

- current count-model surface stops at Poisson
- users will want an overdispersed alternative without leaving the library
- this stays within the current Newton / score / sandwich design style

Scope:

- start with NB2
- expose `fit`, `predict`, and `summary(vcov="vanilla" | "sandwich")`
- keep dispersion estimation in Rust

Success condition:

- stable fit on moderate count data
- inference checked against score / Hessian formulas

### 4. Linear hypothesis and Wald testing utilities

Status: not started

Why it matters:

- several estimators now expose enough covariance information to support restrictions
- users will want tests like `R beta = q` without manually unpacking arrays

Scope:

- helper utility rather than a full estimator
- start with Wald tests for linear restrictions
- work with any estimator summary that exposes `coef` and `vcov`

Success condition:

- one simple public entry point
- examples for OLS, TwoSLS, and GMM

## Medium-Priority Extensions

### 5. Probit

Status: not started

Notes:

- useful and standard
- lower priority than negative binomial and DiD
- should only happen if the numerical story remains clean without bloating dependencies

### 6. Better transformer pipelines

Status: partial

Notes:

- `PCA` and `KernelBasis` already exist
- the next step is not a full sklearn clone
- the useful extension is lightweight composition helpers or richer examples, not a whole pipeline framework

### 7. More IV / GMM summary diagnostics

Status: partial

Notes:

- `GMM` has core covariance support and a J-stat
- likely next useful items are first-stage diagnostics, weak-instrument diagnostics, and overidentification reporting in `TwoSLS`

## Probably Not Worth It Right Now

- formula parsing
- pandas-first APIs
- a giant preprocessing module
- broad sklearn parity for classification/regression models
- continuously updated GMM weighting or a large optimizer zoo without a real application
- generic auto-diff infrastructure just to support a handful of score models

## Working Backlog

### Near-term candidate branch order

1. unify robust covariance estimators for linear models
2. add DiD / event-study support on top of the fixed-effects foundation
3. add negative binomial regression
4. add linear restriction / Wald testing helpers

### Notes for future branches

- prefer small, vertically complete slices
- each extension should usually land with:
  - Rust implementation
  - Python binding
  - direct tests
  - one focused vignette or API example
- if an idea starts demanding a large dependency just for ergonomics, it probably does not belong here
