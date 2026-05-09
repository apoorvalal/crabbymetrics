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
5. If docs examples are numerically heavy, they should use Quarto caching and `freeze: auto`.

## Current Extension Status

### Active PR: randomized linear algebra sketches

Branch `feature/randomized-linear-algebra` / PR #8 adds native randomized linear algebra primitives and a sketching OLS path:

- `src/rla.rs` implements randomized range finding, randomized SVD, randomized QR, QR-based approximate least squares, and CountSketch OLS using the existing `ndarray` + `nalgebra` stack.
- Python exports include `randomized_range`, `randomized_qr`, `randomized_svd`, `qr_solve`, and `sketch_ols`.
- `OLS.fit_sketch(...)` fits an approximate least-squares model by sketching rows before solving the smaller problem.
- `TwoSLS.fit_sketch(...)` applies one CountSketch embedding jointly to the IV regressor design, instrument design, and outcome before solving compressed 2SLS.
- The ablation lives as a proper cached Quarto docs page at `docs/ablations/randomized-sketching-ols.qmd`, with rendered/freeze outputs and a full-site draft preview copied to `https://lalten.org/drafts/crabbymetrics-pr8-docs/`.
- Local gates passed on the branch: `cargo check`, `uv run maturin develop`, `uv run pytest`, and targeted Quarto renders for the new ablation page and index.

### Landed on `master`

- unified robust covariance support for the main linear estimators
  - `OLS`, `Ridge`, `FixedEffectsOLS`, and `TwoSLS` now share `summary(vcov="vanilla" | "hc1" | "newey_west" | "cluster", ...)`
- weighted linear estimation
  - `OLS`, `Ridge`, `FixedEffectsOLS`, and `TwoSLS` have weighted fits through `fit_weighted(...)`
- balancing / calibration weights
  - `BalancingWeights` supports entropy and quadratic objectives, baseline weights, autoscaling, and approximate balance
- panel causal estimators
  - `HorizontalPanelRidge`, `SyntheticDID`, and `MatrixCompletion` now share the matrix panel contract `fit(Y, W)`, where `Y` is `(n_units, n_periods)` and `W` is a same-shaped binary absorbing treatment matrix
  - fitted panel estimators infer ever-treated units, first-treatment cohorts, never-treated donors, and pre/post/event-time structure internally
  - fitted panel estimators expose high-level causal outputs through `summary()`: `att`, `counterfactual`, `treatment_effect`, `event_study`, and `group_means`
  - `HorizontalPanelRidge` exposes the horizontal forecasting leaf from Shen-Ding-Sekhon-Yu style panel comparisons: train cohort-specific ridge forecasts on treated pre-period outcomes using donor paths as features, then forecast treated counterfactual paths
  - `SyntheticDID` ports the matrix-form synthetic difference-in-differences estimator from the local `synthlearners` CVXPY reference into a Rust-backed NumPy API, now fit cohort-by-cohort under the common panel contract
  - low-level weight APIs remain public: `SyntheticControl` for simplex donor weights and `BalancingWeights` for calibration weights, but they are not the modal panel causal path
- semiparametric bundle
  - `EPLM`
  - `AverageDerivative(method="ob" | "ipw" | "dr")`
  - `PartiallyLinearDML`
  - `AIPW`
- docs and ablations
  - the main docs nav now uses `Regression And GLMs`, `Causal Inference`, and `Transforms` rather than the older supervised / semiparametric / unsupervised grouping
  - causal examples include separate public-facing pages for `SyntheticControl`, `SyntheticDID`, `HorizontalPanelRidge`, `MatrixCompletion`, and `InteractiveFixedEffects`, with the matrix panel pages using small self-contained synthetic panels
  - the Hainmueller--Hangartner staggered-adoption vignette demonstrates the shared `fit(Y, W)` matrix panel API on a real panel and compares the matrix estimators to pyfixest vanilla TWFE and saturated/Sun-Abraham-style event studies
  - cached ablation notebooks cover variance estimators, semiparametric comparisons, panel-DGP comparisons, and the Same Root Basque/California panel case studies with simulation, first-class `HorizontalPanelRidge`, plus HAC/placebo inference
  - the `First Course Ding` docs track now covers Chapters 1 through 8, Chapter 9 via the bridging ablation, Chapters 11 through 13, Chapters 21 and 23, and a narrow Chapter 27 Baron-Kenny mediation page with explicit simulation DGPs
  - the latest R-script cleanup filled in post-stratification, matched-pair regression/FRT, IPW truncation and balance diagnostics, ATT doubly robust formulas, the JOBS IV example, and an Anderson-Rubin IV grid
  - the Ding section lives under `docs/ding/` with chapter pages plus a few grouping pages to keep the navbar manageable

### Partial but not finished

- Ding translation track
  - still open:
    - Chapter 10
    - Chapters 15 through 20
    - Chapters 22 and 24 through 26
    - Appendix A
  - expected blockers:
    - matching utilities
    - RD helpers
    - sensitivity-analysis helpers
    - principal-stratification support

- weighted estimation outside the linear family
  - not yet in `Logit`, `Poisson`, or `GMM`
- IV / GMM diagnostics
  - core fitting and covariance support are there, but richer reporting is still thin
- `MEstimator`
  - useful and working, but still more of a low-level escape hatch than a polished estimator family

## Highest-Value Next Extensions

### 1. Difference-in-differences and event-study estimators

Status: partially started through `SyntheticDID`

Why it matters:

- the library already has a workable fixed-effects backbone
- DiD / event-study fits the econometrics focus better than generic ML breadth
- these estimators are widely useful and still compatible with the low-dependency design

Scope:

- start with a simple two-period / two-group DiD estimator
- then add event-study design-matrix helpers on top of the fixed-effects path
- ride on top of `OLS` / `FixedEffectsOLS` rather than building a separate regression engine
- keep the new `SyntheticDID` matrix estimator as the panel-reweighting path rather than forcing it into a regression-style event-study API

Success condition:

- a small Rust-backed `DifferenceInDifferences` or `EventStudy` class
- robust standard errors through the same `summary(vcov=...)` surface
- one focused vignette with a clear coefficient table

### 2. Negative binomial regression

Status: not started

Why it matters:

- the count-model surface currently stops at Poisson
- users will want an overdispersed alternative without leaving the library
- it fits the current Newton / score / sandwich design style

Scope:

- start with NB2
- expose `fit`, `predict`, and `summary(vcov="vanilla" | "sandwich")`
- keep dispersion estimation and likelihood work in Rust

Success condition:

- stable fit on moderate count data
- direct tests against known formulas or a trusted reference implementation

### 3. Linear hypothesis and Wald testing utilities

Status: not started

Why it matters:

- several estimators now expose enough covariance information to support restriction tests
- users will want `R beta = q` style tests without unpacking arrays by hand

Scope:

- helper utility rather than a full estimator
- start with Wald tests for linear restrictions
- work with any estimator summary exposing `coef` and `vcov`

Success condition:

- one simple public entry point
- examples for `OLS`, `TwoSLS`, and `GMM`

### 4. Weighted nonlinear estimators and weighted GMM

Status: partial foundation only

Why it matters:

- the weighted linear path is already implemented
- several semiparametric estimators and two-step procedures want weighted nuisances
- this is the cleanest remaining gap from the `frisch` reference set

Scope:

- add weighted fits to:
  - `Logit`
  - `Poisson`
  - `GMM`
- keep the public surface NumPy-only and consistent with the current style
- align covariance handling with the existing `summary(vcov=...)` interfaces

Success condition:

- weighted fits and weighted sandwich inference for the nonlinear core estimators
- tests against hand-coded references or a trusted external implementation

### 5. More IV / GMM diagnostics

Status: partial

Why it matters:

- point estimation and covariance support are now in good shape
- reporting is still thin relative to what econometrics users expect

Scope:

- first-stage fit diagnostics
- weak-instrument diagnostics where they are easy to support cleanly
- better overidentification reporting in `TwoSLS`

Success condition:

- `TwoSLS.summary()` and `GMM.summary()` expose diagnostics without bloating the estimator API

## Medium-Priority Extensions

### 6. Probit

Status: not started

Notes:

- useful and standard
- lower priority than negative binomial and DiD
- only worth doing if the numerical story remains clean without inflating dependencies

### 7. Better transformer composition

Status: partial

Notes:

- `PcaTransformer` and `KernelBasis` already exist
- the next useful step is lightweight composition helpers or richer examples, not sklearn-style pipeline sprawl

### 8. MEstimator diagnostics cleanup

Status: partial

Notes:

- current `MEstimator` is useful for custom objectives, but its variance path is still score-outer-product based and its solver diagnostics are thin
- worth improving only if there is a concrete downstream use case

## Frisch Status

The targeted reference pass over the old `frisch` code is effectively complete for now.

What was extracted into `crabbymetrics`:

- balancing / calibration weights
- partially linear E-estimation
- average derivative estimators
- semiparametric comparison designs that motivated the current docs and ablations

What remains useful from `frisch`:

- formulas and tests for weighted nonlinear estimators
- occasional reference checks for semiparametric estimating equations

What should not be ported literally:

- pandas / patsy interfaces
- dataframe-centric summaries
- sklearn-orchestrated nuisance wrappers as a public dependency story

Operational note:

- the local `frisch` symlink has been removed
- if the code is referenced again later, do it ad hoc as a formulas-and-tests reference, not as a standing subtree dependency

## Recommended Branch Order

1. add DiD / event-study support on top of the fixed-effects foundation
2. add negative binomial regression
3. add linear restriction / Wald testing helpers
4. add weighted nonlinear estimators and weighted GMM
5. add richer IV / GMM diagnostics

## Notes for Future Branches

- prefer small, vertically complete slices
- each extension should usually land with:
  - Rust implementation
  - Python binding
  - direct tests
  - one focused vignette or API example
- if an idea starts demanding a large dependency just for ergonomics, it probably does not belong here
