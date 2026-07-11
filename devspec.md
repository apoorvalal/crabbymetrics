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

## Refactor Branch Status (2026-07-10)

The `refactor` branch starts from `origin/master` 0.7.0. The original dirty 0.5.1 audit tree is preserved on `pre-refactor-audit-snapshot`. The correctness pass has completed every P0 and P1 item from `docs/evaluation-review.qmd`:

- correct weighted TwoSLS design and covariance behavior
- identified and objective-matched likelihood and M-estimator inference
- explicit no-inference contracts for predictive regularized estimators
- seeded shuffled DML folds and treatment-stratified AIPW folds
- absorbed fixed-effect rank in residual degrees of freedom
- strict covariance-diagonal validation
- explicit convergence semantics across iterative estimators and optimizers
- guarded, prediction-only `BaggedPolynomialRegressor` with OOB diagnostics

New estimator work should preserve these contracts. In particular, a summary must not expose standard errors that do not correspond to the fitted objective, and iterative estimators must not equate budget exhaustion with convergence.

## Current Extension Status

### Recently landed: randomized linear algebra, hypothesis tests, and ABC OLS

The 2026-05 extension sequence landed several items that used to be active or pending in this file:

- Randomized linear algebra / PR #8 is no longer an active PR. It added native randomized range finding, SVD, QR, QR solve, CountSketch OLS, `OLS.fit_sketch(...)`, `TwoSLS.fit_sketch(...)`, `GMM.fit_sketch(...)`, randomized SVD paths in `MatrixCompletion` / `InteractiveFixedEffects`, and the reusable `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA` transformers.
- Hypothesis-test helpers are landed. Estimator-level `wald_test(...)` methods exist for the main covariance-bearing estimators, module-level `wald_test(...)`, `likelihood_ratio_test(...)`, and `lr_test(...)` exist, and `TwoSLS.anderson_rubin_test(...)` covers scalar weak-IV-robust tests.
- `ABCOLS` is landed as an OLS-only abundance-based constraints / weighted-effect-coding estimator for categorical main effects, continuous-by-categorical interactions, and categorical-by-categorical interactions, with a detailed worked example at `docs/examples/abc-ols.qmd` and a class reference page at `docs/reference/ABCOLS.qmd`.


### Sketching / randomized linear algebra follow-on plan

The branch now has enough primitives to make sketching useful inside estimators. The prioritized sequence from the 2026-05-09 pass has been knocked out as explicit opt-in estimator integrations and reusable transformers, while preserving exact defaults for existing APIs.

1. **MatrixCompletion acceleration**
   - Status: implemented.
   - `MatrixCompletion` keeps `svd_method="exact"` as the default and accepts `svd_method="randomized"` with `svd_rank`, `svd_oversamples`, `svd_power_iter`, and `svd_seed` for randomized singular-value-thresholding updates.
   - Tests compare exact and randomized paths on controlled low-rank panels.

2. **InteractiveFixedEffects / panel-factor acceleration**
   - Status: implemented.
   - `panel_factor(...)` and `InteractiveFixedEffects` keep `factor_method="exact"` as the default and accept `factor_method="randomized"` with oversampling, power-iteration, and seed knobs.
   - Tests check randomized factor extraction, fitted values, and treatment-effect summaries against exact low-rank behavior.

3. **Nyström kernel approximation**
   - Status: implemented.
   - Added reusable `NystromBasis`, which samples landmarks, computes `K_nm K_mm^{-1/2}` features, and is intended to pair with existing `Ridge` / `OLS` estimators downstream.
   - Tests cover output shape, reproducibility, summary metadata, and kernel-ridge-style approximation behavior.

4. **Random Fourier features for shift-invariant kernels**
   - Status: implemented.
   - Added reusable `RandomFourierFeatures` for RBF/Gaussian features with explicit `n_components`, `bandwidth`, and `seed` controls.
   - Tests cover output shape, reproducibility, transformed test designs, and nonlinear ridge approximation behavior.

5. **Many-moment GMM sketching**
   - Status: implemented conservatively.
   - Added `GMM.fit_sketch(...)`, which applies a fixed Rademacher projection to many moments and their Jacobian before fitting the compressed moment system.
   - Summaries report sketched moment metadata; inference remains tied to the fitted/sketched moment system rather than silently pretending to use the full original moments.

6. **Balancing / synthetic-control feature compression**
   - Status: implemented as reusable preprocessing rather than hidden estimator behavior.
   - Added `RandomizedPCA` as an explicit randomized low-rank feature-compression transformer for long donor histories or wide balance designs.
   - Tests cover reproducibility, reconstruction quality on low-rank designs, and a `BalancingWeights` workflow on compressed features.

Implementation guardrails for sketching work:

- Default exact algorithms should remain available and should usually remain the default for small problems.
- Expose sketching as explicit `method`/`approximation` knobs, not separate estimator classes unless a transformer is naturally reusable.
- Tests should compare approximate paths to exact paths on controlled low-rank or smooth-kernel designs.
- Docs should report accuracy/runtime tradeoffs; do not present sketching as a free lunch.
- Avoid adding BLAS/LAPACK-heavy Python dependencies; keep the hot path in Rust and reuse `ndarray`/`nalgebra`.

### Landed on `master`

- abundance-based constraints / weighted effect coding for OLS
  - `ABCOLS` supports categorical main effects, continuous-by-categorical interactions, and categorical-by-categorical interactions
  - the docs page `docs/examples/abc-ols.qmd` explains the one-hot baseline vs ABC full-sample weighted-mean baseline distinction and includes coefficient-table comparisons to vanilla reference-coded OLS
- linear hypothesis testing helpers
  - fitted `wald_test(...)` methods are available on the main covariance-bearing estimators
  - module-level `wald_test(...)`, `likelihood_ratio_test(...)`, and `lr_test(...)` support array-level/manual workflows
  - `TwoSLS.anderson_rubin_test(...)` covers scalar weak-IV-robust tests
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

### 3. Abundance-based constraints beyond OLS

Status: OLS landed; broader families not started

Why it matters:

- `ABCOLS` covers the core weighted-effect-coding reparameterization for OLS
- the same interpretation problem can arise in GLMs or other fitted models with categorical modifiers
- any extension should preserve the current explicit NumPy-first categorical-code API rather than adding a formula/patsy dependency

Scope:

- first decide whether the next step should be GLM support, richer covariance options, or convenience transforms for design construction
- avoid broadening ABC until there is a concrete modeling use case beyond the current OLS page

Success condition:

- one vertically complete extension with tests and a focused docs page, not a partial general formula system

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
3. add weighted nonlinear estimators and weighted GMM
4. add richer IV / GMM diagnostics
5. consider ABC extensions beyond OLS only if a concrete use case appears

## Notes for Future Branches

- prefer small, vertically complete slices
- each extension should usually land with:
  - Rust implementation
  - Python binding
  - direct tests
  - one focused vignette or API example
- if an idea starts demanding a large dependency just for ergonomics, it probably does not belong here
