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

## v0.8.1 API-Hardening Release Status (2026-07-12)

PR #17 squash-merged the `api-hardening` branch into `master` as `854a63b`, after the `v0.8.0` refactor and estimator-audit release. Release `v0.8.1` now packages the remaining P0--P1 hardening items identified by the stocktake:

- removed the incoherent one-update FTRL class, its public/docs/test surface, and `linfa-ftrl`
- removed the unused `linfa-linear` dependency
- replaced Linfa binary and multinomial logistic fitting with native stable objectives, analytic gradients, and convergence-checked L-BFGS
- added shared internal `FitDiagnostics` and centralized Argmin status interpretation
- made survival fits reject budget exhaustion and made MatrixCompletion retain it only as an explicitly nonconverged result
- exposed and enforced ElasticNet iteration and duality-gap convergence
- separated scaled BalancingWeights solver convergence from original-unit balance diagnostics and weight feasibility
- consolidated reusable validation helpers
- split the old `linear.rs` monolith into linear, IV, panel, and synthetic modules without changing public imports
- hardened source-distribution exclusions so the local untracked `ding_ci` symlink cannot be followed into release artifacts

New estimator work must preserve these contracts. In particular, a summary must not expose standard errors that do not correspond to the fitted objective, and iterative estimators must not equate budget exhaustion with convergence. Common iterative summary keys are `converged`, `iterations`, `termination_reason`, and `objective`.

The branch's public reference pages now also document estimator internals at source-code granularity. All 29 estimator, transform, and optimizer pages explain parameter layout, initialization, numerical steps, stopping and failure behavior, prediction reconstruction, and dominant allocations. They explicitly separate package-owned algorithms from narrow delegation boundaries in `FixedEffectsOLS`, `ElasticNet`, exact `PCA`, and exact `KernelBasis`. The class-page generator is scaffold-only and skips these audited pages unless explicitly forced.

The release site was rebuilt as 92 Quarto pages after re-executing the four solver-sensitive ablations. It is deployed from clean `gh-pages` commit `c7a3ce4` at `https://apoorvalal.github.io/crabbymetrics/`; rendered outputs remain excluded from `master` and the PyPI sdist.

The `BaggedPolynomialRegressor` reference is contract-first rather than a disguised vignette: it documents the exact constructor defaults and constraints, array contracts, return values, failure conditions, complete `summary()` schema, objective, OOB semantics, implementation, and computational limits. The separate worked comparison remains under `docs/examples/`.

### Estimator Math And API Audit (2026-07-11)

The follow-up audit traced every public estimator through its Rust implementation and documented its implemented criterion or estimating equation, inference method, prediction contract, and performance characteristics in the API references. It also:

- corrected Poisson validation so invalid outcomes, penalties, tolerances, iteration budgets, and nonfinite designs fail before optimization
- corrected `PCA` explained variance, full-variance ratios, and whitened inverse transformation, with NumPy SVD parity tests
- added grouped first-class reference coverage for all five public transform classes
- identified the former FTRL wrapper as one full-batch proximal update with fresh state rather than a persistent online algorithm; the API-hardening branch subsequently removed it
- documented that `AndersenGill` cannot produce subject-clustered recurrent-event covariance without a subject identifier

The FTRL question is resolved. Subject-clustered Andersen--Gill inference and Cox-family risk-set performance remain the leading public-API and performance follow-up.

The audit branch was squash-merged as PR #16 and released as `v0.8.0` on 2026-07-11. The release includes Linux and macOS wheels for Python 3.10 through 3.14, a small docs-excluded sdist, and the separately deployed full Quarto site.

## Current Extension Status

### In review: canonical Chronos marginal-policy-effect CBPS

The `docs/chronos-ltv-vignette` branch adds `MPE_CBPS`, a focused native implementation of the two-arm tailored-loss CBPS estimator released with Qiu, Kuang, Liskovich, Rauh, and Wager (2026). It keeps the paper's inverse-logit weight family rather than relabeling generic entropy calibration as exact parity. The class standardizes the supplied basis, adds an intercept, solves both convex arm losses with analytic damped Newton steps in Rust, and aggregates cumulative future rewards with a supplied policy derivative and denominator.

Canonical parity is tied to commit `06c29f4` of `chenyuqiu/ltv_of_reliability`. Tests transcribe the released SciPy/BFGS A/B and switchback helper functions and compare both coefficient vectors, every observation-level weight, and the final normalized policy-gradient estimate. A dedicated class reference and the expanded Chronos vignette document the dynamic identification argument, exact implementation, entropy-calibration sensitivity check, horizon path, and unit-clustered bootstrap. Analytic inference remains out of scope; the vignette re-fits the complete estimator inside unit bootstrap draws.

### Recently landed: faithful augmented panel balancing

`master` now includes the R experiment's estimator family without treating horizontal ridge as a universal panel estimator. `AugmentedBalancing` supports outcome-only, unit-only, time-only, and double balancing; cohort or individual unit targets; pooled or period-specific time targets; raw-outcome or residual balance inputs; and ridge or penalized-SCM donor losses. Unit and time weights enter the full augmented score in every balancing mode, including the uniform weights used for a dimension that is not optimized.

The simplex ridge solver uses an active-set quadratic-programming step, and the penalized-SCM loss follows the executable R reference. A committed same-panel fixture checks 48 configurations against R and has a maximum absolute ATT difference below $1.8\times10^{-10}$. Separate external checks cover FE, IFE, MCPanel, generalized synthetic-control, and panel-VAR outcome surfaces and the complete 398-cell slide grid; those larger reproduction artifacts are not committed to this repository. Of the 398 cells, 390 are within three combined Monte Carlo standard errors. All FE, IFE, generalized synthetic-control, and panel-VAR cells pass this rule. The eight exceptions are five correlated MCPanel cells in the poor-overlap classic-factor design and three ARIMA cells in the low-persistence time-series design. A same-panel MCPanel diagnostic limits the maximum treated-post surface difference to $0.0096$, so the larger Monte Carlo RMSE differences are not evidence of a balancing-formula mismatch. This evidence validates the implemented balancing family. It does not imply that horizontal ridge dominates across the panel DGPs.

### Recently landed: Cressie-Read balancing, API hardening, sparse rotations, randomized linear algebra, hypothesis tests, ABC OLS, anytime-valid OLS, MLE prediction, survival, and v0.8.0

The 2026-05 through 2026-06 extension sequence landed several items that used to be active or pending in this file:

- Randomized linear algebra / PR #8 is no longer an active PR. It added native randomized range finding, SVD, QR, QR solve, CountSketch OLS, `OLS.fit_sketch(...)`, `TwoSLS.fit_sketch(...)`, `GMM.fit_sketch(...)`, randomized SVD paths in `MatrixCompletion` / `InteractiveFixedEffects`, and the reusable `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA` transformers.
- Cressie-Read balancing is integrated into `BalancingWeights` through `objective="cressie_read"` / `"power_divergence"`, a finite `divergence_power`, optional dual ridge stabilization, and an explicit L-BFGS solver. It preserves the API-hardening distinction between scaled solver convergence, original-unit balance, and final weight feasibility. Rényi divergence is documented only as the $\lambda=\alpha-1$ diagnostic mapping, not as a separate optimizer.
- Sparse factor rotations are implemented as low-level functions rather than estimator wrappers. The public surface includes Varimax, a seeded multi-start L1 sparse rotation, small-loading/local-factor diagnostics, and inverse/cumulative participation summaries. The implementation preserves explicit non-convexity caveats, seed control, and function-level diagnostics; the worked and design pages live at `docs/examples/sparse-rotations.qmd` and `docs/specs/sparse-rotations.qmd`.
- Hypothesis-test helpers are landed. Estimator-level `wald_test(...)` methods exist for the main covariance-bearing estimators, module-level `wald_test(...)`, `likelihood_ratio_test(...)`, and `lr_test(...)` exist, and `TwoSLS.anderson_rubin_test(...)` covers scalar weak-IV-robust tests.
- `ABCOLS` is landed as an OLS-only abundance-based constraints / weighted-effect-coding estimator for categorical main effects, continuous-by-categorical interactions, and categorical-by-categorical interactions, with a detailed worked example at `docs/examples/abc-ols.qmd` and a class reference page at `docs/reference/ABCOLS.qmd`.
- Anytime-valid OLS is landed and released in `v0.7.1`. `OLS.summary(...)` now accepts `anytime_valid=True`, `g=...`, and `level=...`, while module-level `optimal_g(...)` and `av(...)` cover the convenience path. Tests compare against `avlm` reference values.
- The v0.7.0 MLE prediction overhaul is landed. `Logit`, `MultinomialLogit`, and `Poisson` now follow the layered `predict_lin(...)` / `predict(...)` / classifier-only `predict_label(...)` contract, documented in `docs/examples/mle-prediction-interface.qmd`.
- The first survival/event-time module is landed. `ExponentialPH`, `WeibullPH`, `CoxPH`, and `AndersenGill` are exported, tested against simulated DGPs and `lifelines`, and documented through reference pages plus survival vignettes.
- Release automation for the current shape is working: `v0.8.0` built Linux/macOS wheels for Python 3.10 through 3.14, published a 410,390-byte sdist, created the GitHub Release, and published to PyPI.


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
- anytime-valid OLS inference
  - `OLS.summary(..., anytime_valid=True, g=..., level=...)` adds anytime-valid coefficient p-values, confidence intervals, and omnibus F-test fields
  - `optimal_g(n, number_of_coefficients, alpha)` and `av(model, g=..., vcov="vanilla", level=...)` are exported module-level helpers
  - `docs/examples/anytime-valid-ols.qmd` is present in source, and tests match `avlm` reference calculations
- balancing / calibration weights
  - `BalancingWeights` supports entropy and quadratic objectives, baseline weights, autoscaling, and approximate balance
- panel causal estimators
  - `HorizontalPanelRidge`, `SyntheticDID`, `AugmentedBalancing`, and `MatrixCompletion` use the matrix panel contract, where `Y` is `(n_units, n_periods)` and `W` is a same-shaped binary absorbing treatment matrix
  - fitted panel estimators infer ever-treated units, first-treatment cohorts, never-treated donors, and pre/post/event-time structure internally
  - fitted panel estimators expose high-level causal outputs through `summary()`: `att`, `counterfactual`, `treatment_effect`, `event_study`, and `group_means`
  - `HorizontalPanelRidge` exposes the horizontal forecasting leaf from Shen-Ding-Sekhon-Yu style panel comparisons: train cohort-specific ridge forecasts on treated pre-period outcomes using donor paths as features, then forecast treated counterfactual paths
  - `SyntheticDID` ports the matrix-form synthetic difference-in-differences estimator from the local `synthlearners` CVXPY reference into a Rust-backed NumPy API, now fit cohort-by-cohort under the common panel contract
  - `AugmentedBalancing` implements outcome-only, unit-balanced, time-balanced, double-balanced, and outcome-model-augmented variants; unit weights can target cohort means or individuals, time weights can be pooled or period-specific, balance inputs can be raw outcomes or outcome-model residuals, and the donor loss can use ridge or penalized SCM
  - low-level weight APIs remain public: `SyntheticControl` for simplex donor weights and `BalancingWeights` for calibration weights, but they are not the modal panel causal path
- semiparametric bundle
  - `EPLM`
  - `AverageDerivative(method="ob" | "ipw" | "dr")`
  - `PartiallyLinearDML`
  - `AIPW`
- dynamic treatment effects (landed on `master` via PR #19)
  - `RegressionBlip(max_lag=1, time_effects=True)` estimates additive lag-specific blips by recursive outcome regression under sequential ignorability
  - `ParallelTrendsSNMM(max_horizon=1, treatment_mode="blip" | "initiation", n_folds=2, nuisance_penalty=1e-6, propensity_clip=0.01, seed=42)` estimates additive horizon-specific blips using cross-fitted doubly robust moments under time-varying conditional parallel trends
  - `DynamicCovariateBalance(nuisance_penalty=1e-6, autoscale=True, max_weight=1.0, max_iterations=300, tolerance=1e-6)` estimates a Viviano--Bradic final mean potential outcome under one binary path using recursive ridge potential projections and the shared exact quadratic-calibration engine
  - all accept wide unit-by-time NumPy panels; optional or required histories use shape `(n_units, n_periods, n_features)`, and cross-fitting is always by unit where used
  - the standalone mathematical and API review page is `docs/examples/snmm-blips.qmd`
- MLE prediction contract and survival models
  - `Logit`, `MultinomialLogit`, and `Poisson` expose layered prediction APIs: `predict_lin(...)`, `predict(...)`, and classifier-only `predict_label(...)`
  - `ExponentialPH` and `WeibullPH` expose absolute hazard, cumulative-hazard, and survival predictions
  - `CoxPH` and `AndersenGill` expose semiparametric relative-risk predictions
- docs and ablations
  - the main docs nav now uses `Regression And GLMs`, `Causal Inference`, and `Transforms` rather than the older supervised / semiparametric / unsupervised grouping
  - the API overview now includes grouped anchors for regression/GLMs, survival/event-time models, causal inference/panels, hypothesis testing, transforms, and estimation interfaces
  - causal examples include separate public-facing pages for `SyntheticControl`, `SyntheticDID`, `HorizontalPanelRidge`, `MatrixCompletion`, and `InteractiveFixedEffects`, with the matrix panel pages using small self-contained synthetic panels
  - likelihood examples now include the MLE prediction interface and survival/event-time pages
  - anytime-valid OLS has a worked source page and nav/API links; public `gh-pages` deployment is separate from PyPI release and should be run explicitly when the live docs need to update
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
- likelihood family breadth
  - `Logit`, `MultinomialLogit`, `Poisson`, and the first survival models are in place
  - no negative binomial, grouped/binomial count model, probit, complementary-log-log discrete-time hazard, or baseline survival estimator for Cox-style models yet
- docs surface depth
  - all public estimator references now document implemented mathematics, inference, and performance behavior
  - grouped transform references cover `PCA`, `RandomizedPCA`, `KernelBasis`, `NystromBasis`, and `RandomFourierFeatures`
  - all 94 pages render from the `v0.8.0` tag and are deployed separately through `gh-pages`
- IV / GMM diagnostics
  - core fitting and covariance support are there, but richer reporting is still thin
- `MEstimator`
  - useful and working, but still more of a low-level escape hatch than a polished estimator family

## Highest-Value Next Extensions

### 0. Andersen--Gill subject-clustered inference and risk-set performance

Status: point estimator landed; recurrent-event inference incomplete

Why it matters:

- recurrent event rows from the same subject are dependent
- the current API has no subject identifier and therefore cannot form a subject-clustered score sandwich
- `CoxPH` and `AndersenGill` currently scan all rows and rebuild dense second moments for every event, costing approximately $O(Enp^2)$ per Newton iteration

Scope:

- add an explicit subject-ID argument for `AndersenGill`
- accumulate subject-level score contributions and expose clustered robust covariance as the primary recurrent-event inference path
- retain inverse-information covariance only as an explicitly naive option
- replace repeated risk-set scans with cumulative sufficient statistics after sorting by stop time, while preserving Breslow tie handling

Success condition:

- robust covariance matches a trusted counting-process Cox reference on repeated-event data
- summaries distinguish naive and subject-clustered inference
- benchmark coverage demonstrates the expected reduction in risk-set work

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

### 2. More likelihood methods

Status: foundation landed; new families not started

Why it matters:

- the current likelihood surface is coherent but narrow: `Logit`, `MultinomialLogit`, `Poisson`, and first-pass survival models
- the v0.7.0 prediction contract gives new likelihood estimators a clear API target
- negative binomial, grouped binomial, and discrete-time hazard models are natural econometrics additions that do not require pandas, formulas, or a large dependency stack

Integration plan:

1. **Shared likelihood foundation**
   - Reuse the existing Newton / line-search / score / Hessian style from `mle.rs` and `survival.rs`.
   - Keep the public contract consistent: `fit(...)`, `predict_lin(...)`, `predict(...)`, `summary(...)`, and `predict_label(...)` only when the fitted object is a classifier.
   - Standardize summary keys across likelihood models where possible: `coef`, `intercept` when applicable, `vcov`, `se`, `z`, `p_value`, `log_likelihood`, `iterations`, and family-specific parameters.
   - Add weights, offsets, and exposure only when each family has a clean algebraic interpretation; do not bolt them on inconsistently.

2. **NB2 / overdispersed counts**
   - First vertical slice after DiD unless a weighted-GLM branch becomes urgent.
   - Estimate mean parameters and dispersion in Rust.
   - Use `predict_lin(...)` for log mean and `predict(...)` for mean counts.
   - Support `summary(vcov="vanilla" | "sandwich")` and tests against hand-coded likelihood/score checks or `statsmodels` under test extras.

3. **Grouped binomial / trials API**
   - Add a model for `successes` out of `trials`, rather than forcing users to expand rows.
   - Start with logit link; consider complementary-log-log as a separate discrete-time hazard variant.
   - This should share code with binary `Logit` where possible but avoid changing `Logit.fit(x, y)` semantics.

4. **Discrete-time hazard bridge**
   - Use grouped binomial or complementary-log-log likelihoods to connect the survival docs' person-period discussion to an actual estimator.
   - Keep it distinct from continuous-time `CoxPH` / `AndersenGill` so baseline-hazard interpretation stays clear.

5. **Probit**
   - Add only after NB2 and grouped binomial unless there is a concrete need.
   - Requires stable normal CDF/PDF approximations in Rust without adding a heavy dependency.

6. **Survival follow-ons**
   - Add Cox baseline cumulative hazard and survival curves before broadening into more exotic survival families.
   - Consider piecewise exponential PH if grouped-duration/discrete-time docs expose a real use case.

Guardrails:

- Do not add a generic formula or dataframe interface as part of likelihood expansion.
- Prefer one vertically complete family at a time, with tests and one focused docs page.
- Use external packages only as test/reference extras, not runtime dependencies.
- Avoid exposing a generic GLM superclass until at least two new families reveal real shared public behavior.

Success condition:

- at least one new likelihood family, ideally NB2, lands with native Rust fitting, layered prediction, covariance summaries, parity/reference tests, and a docs page that explains why it belongs beside the existing Poisson/logit/survival surface.

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

### 6. Likelihood polish after NB2

Status: not started

Notes:

- grouped binomial / trials, complementary-log-log discrete-time hazards, probit, and Cox baseline survival are all useful, but none should jump ahead of the first new count-family slice unless there is a concrete downstream need
- keep each addition vertically complete rather than starting a broad unfinished GLM framework

### 7. Transformer composition only when demanded by workflows

Status: documentation complete; composition not started

Notes:

- public Python transform classes are `PCA`, `KernelBasis`, `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA`
- grouped reference pages now document the objective, transform, numerical method, output shape, and scaling of all five classes
- add lightweight composition helpers only if real downstream workflows demonstrate repeated boilerplate
- avoid sklearn-style pipeline sprawl

### 8. MEstimator diagnostics cleanup

Status: partial

Notes:

- current `MEstimator` uses a numerical score Jacobian for the sandwich bread and empirical score outer products for the meat; common solver diagnostics are now exposed, but the callback-heavy path remains expensive and specialized
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

1. add subject IDs and clustered robust inference to `AndersenGill`, then optimize Cox-family risk-set accumulation
2. add DiD / event-study support on top of the fixed-effects foundation
3. add the first new likelihood family, preferably NB2, under the likelihood-methods plan
4. add weighted nonlinear estimators and weighted GMM
5. add richer IV / GMM diagnostics, and consider ABC extensions only for a concrete use case

## Notes for Future Branches

- prefer small, vertically complete slices
- each extension should usually land with:
  - Rust implementation
  - Python binding
  - direct tests
  - one focused vignette or API example
- if an idea starts demanding a large dependency just for ergonomics, it probably does not belong here
