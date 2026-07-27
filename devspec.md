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

The `refactor` branch starts from `origin/master` 0.7.1. The original dirty 0.5.1 audit tree is preserved on `pre-refactor-audit-snapshot`. The correctness pass has completed every P0 and P1 item from `evaluation-review.qmd`:

- correct weighted TwoSLS design and covariance behavior
- identified and objective-matched likelihood and M-estimator inference
- explicit no-inference contracts for predictive regularized estimators
- seeded shuffled DML folds and treatment-stratified AIPW folds
- absorbed fixed-effect rank in residual degrees of freedom
- strict covariance-diagonal validation
- explicit convergence semantics across iterative estimators and optimizers
- guarded, prediction-only `BaggedPolynomialRegressor` with OOB diagnostics

New estimator work should preserve these contracts. In particular, a summary must not expose standard errors that do not correspond to the fitted objective, and iterative estimators must not equate budget exhaustion with convergence.

### Estimator Math And API Audit (2026-07-11)

The follow-up audit traced every public estimator through its Rust implementation and documented its implemented criterion or estimating equation, inference method, prediction contract, and performance characteristics in the API references. It also:

- corrected Poisson validation so invalid outcomes, penalties, tolerances, iteration budgets, and nonfinite designs fail before optimization
- corrected `PCA` explained variance, full-variance ratios, and whitened inverse transformation, with NumPy SVD parity tests
- added grouped first-class reference coverage for all five public transform classes
- labeled `FTRL` experimental because its current fit is one full-batch proximal update with fresh state rather than a persistent online algorithm
- documented that `AndersenGill` cannot produce subject-clustered recurrent-event covariance without a subject identifier

The next estimator branch should resolve those last two public-API questions before adding another broad estimator family.

The audit branch was squash-merged as PR #16 and released as `v0.8.0` on 2026-07-11. The release includes Linux and macOS wheels for Python 3.10 through 3.14, a small docs-excluded sdist, and the separately deployed full Quarto site.

## Current Extension Status

### Active branch: Cressie-Read balancing

The `feature/cressie-read-balancing` branch extends the existing `BalancingWeights`
primitive with Cressie-Read / power-divergence calibration. The branch adds
`objective="cressie_read"` with alias `objective="power_divergence"`,
`divergence_power` for the Cressie-Read index, `dual_ridge` for optional
dual-side stabilization, and `solver="lbfgs"` as an explicit solver path used by
automatic fallback.

The source docs live in `docs/reference/BalancingWeights.qmd` and
`docs/examples/cressie-read-balancing.qmd`. The example treats Renyi divergence
as a diagnostic mapping through the shared power moment with
`lambda = alpha - 1`; it does not claim a separate `objective="renyi"` optimizer.
Rendered review output belongs on the Hetzner draft preview or `gh-pages`, not
on the source branch.

### Recently landed: randomized linear algebra, hypothesis tests, ABC OLS, anytime-valid OLS, MLE prediction, survival, and v0.8.0

The 2026-05 through 2026-06 extension sequence landed several items that used to be active or pending in this file:

- Randomized linear algebra / PR #8 is no longer an active PR. It added native randomized range finding, SVD, QR, QR solve, CountSketch OLS, `OLS.fit_sketch(...)`, `TwoSLS.fit_sketch(...)`, `GMM.fit_sketch(...)`, randomized SVD paths in `MatrixCompletion` / `InteractiveFixedEffects`, and the reusable `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA` transformers.
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
  - active branch extension: Cressie-Read / power-divergence calibration through `objective="cressie_read"`, with `divergence_power`, `dual_ridge`, and `solver="lbfgs"`
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

### 0. FTRL public API decision

Status: design decision required before promotion

Why it matters:

- the current `fit` constructs a fresh optimizer, evaluates one full-batch logistic gradient, applies one proximal update, and returns those coefficients
- there is no intercept, repeated optimization, convergence criterion, `partial_fit`, or persistent optimizer state
- the class name and standard `fit` shape imply a trained estimator that the implementation does not provide

Scope:

- preferred path: retain optimizer accumulators in the Python object and expose `partial_fit` for streaming mini-batches
- define intercept handling, feature-count checks, deterministic zero initialization, and explicit reset semantics
- make `fit` either run a documented epoch/batch schedule or remove it in favor of the honest online method
- alternative path: remove `FTRL` from the first-tier public surface if there is no streaming use case

Success condition:

- repeated calls have defined stateful semantics and match a hand-calculated FTRL-Proximal update sequence
- `fit` no longer suggests convergence after one accidental update
- the class has tests for sparsity, intercept behavior, reset/persistence, and online ordering

### 1. Andersen--Gill subject-clustered inference and risk-set performance

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

### 2. Difference-in-differences and event-study estimators

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

### 3. More likelihood methods

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

### 4. Abundance-based constraints beyond OLS

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

### 5. Weighted nonlinear estimators and weighted GMM

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

### 6. More IV / GMM diagnostics

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

### 7. Likelihood polish after NB2

Status: not started

Notes:

- grouped binomial / trials, complementary-log-log discrete-time hazards, probit, and Cox baseline survival are all useful, but none should jump ahead of the first new count-family slice unless there is a concrete downstream need
- keep each addition vertically complete rather than starting a broad unfinished GLM framework

### 8. Transformer composition only when demanded by workflows

Status: documentation complete; composition not started

Notes:

- public Python transform classes are `PCA`, `KernelBasis`, `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA`
- grouped reference pages now document the objective, transform, numerical method, output shape, and scaling of all five classes
- add lightweight composition helpers only if real downstream workflows demonstrate repeated boilerplate
- avoid sklearn-style pipeline sprawl

### 9. MEstimator diagnostics cleanup

Status: partial

Notes:

- current `MEstimator` uses a numerical score Jacobian for the sandwich bread and empirical score outer products for the meat, but its callback-heavy path and solver diagnostics remain thin
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

1. decide whether to rebuild `FTRL` as a persistent online estimator or remove it from the first-tier API
2. add subject IDs and clustered robust inference to `AndersenGill`, then optimize Cox-family risk-set accumulation
3. make convergence status explicit for iterative estimators that retain budget-exhausted results
4. add DiD / event-study support on top of the fixed-effects foundation
5. add the first new likelihood family, preferably NB2, under the likelihood-methods plan
6. add weighted nonlinear estimators and weighted GMM
7. add richer IV / GMM diagnostics, and consider ABC extensions only for a concrete use case

## Notes for Future Branches

- prefer small, vertically complete slices
- each extension should usually land with:
  - Rust implementation
  - Python binding
  - direct tests
  - one focused vignette or API example
- if an idea starts demanding a large dependency just for ergonomics, it probably does not belong here
