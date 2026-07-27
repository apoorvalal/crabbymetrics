# Devlog (crabbymetrics)

## Snapshot

`crabbymetrics` is a Rust-backed econometrics library exposed to Python through `pyo3` and `maturin`. The project is intentionally narrow:

- Rust owns the numerical work
- Python stays NumPy-only and scikit-adjacent
- docs are checked in as a Quarto site under `docs/`
- the current surface is stronger on econometrics estimators and inference than on generic ML breadth

Current release state: `v0.8.0` is published to PyPI and GitHub Releases. It contains the estimator-correctness remediation, complete estimator math/API audit, guarded `BaggedPolynomialRegressor`, and PCA/Poisson corrections merged in PR #16, in addition to the earlier anytime-valid OLS, likelihood, and survival work.

This file is meant to record the current architecture and the design choices that matter for future work.

## Current Branch: Cressie-Read Balancing (2026-07-27)

The `feature/cressie-read-balancing` branch extends `BalancingWeights` in
`src/estimators/balancing.rs` with a Cressie-Read / power-divergence calibration
map:

- `objective="cressie_read"` and alias `objective="power_divergence"`
- `divergence_power` for the Cressie-Read index lambda
- `dual_ridge` for optional ridge stabilization on non-intercept dual
  coefficients
- `solver="lbfgs"` plus automatic fallback from Gauss-Newton to L-BFGS to BFGS

The public docs are updated in `docs/reference/BalancingWeights.qmd`, with a
worked ATT reweighting example in `docs/examples/cressie-read-balancing.qmd` and
navigation links in `docs/_quarto.yml` / `docs/index.qmd`. The Renyi material in
the example is a diagnostic mapping through the shared power moment
`lambda = alpha - 1`, not a separate `objective="renyi"` optimizer. Regression
coverage lives in `tests/test_balancing_weights.py`.

## Refactor Correctness And Documentation Pass (2026-07-10 through 2026-07-11)

The `refactor` branch was created from `origin/master` 0.7.0 after preserving the audited dirty 0.5.1 tree on `pre-refactor-audit-snapshot`, then merged with current `master` 0.7.1. It fixes the review P0/P1 set: weighted TwoSLS, generic M-estimator covariance, identified multinomial inference, penalized-estimator inference contracts, shuffled/stratified cross-fitting, absorbed fixed-effect degrees of freedom, covariance-diagonal validation, and convergence semantics.

`BaggedPolynomialRegressor` is now a guarded prediction-only class in `regularized.rs`. It standardizes polynomial columns inside each learner, stores random subspaces and resolved dimensions, reports OOB MSE/coverage, enforces dense-design limits, and has direct scikit-learn parity tests. The public docs include a reference page and a leakage-free repeated-draw demo; both leave executable code visible by default. The repository-level baseline audit and remediation record is `evaluation-review.qmd`.

The follow-up source audit covers every public estimator and all five public transforms. Each API reference now states the implemented objective or estimating equations, covariance or inferential method, prediction contract, and the relevant dense computational scaling. The audit also corrected Poisson input validation and the PCA explained-variance and whitening-inverse formulas. FTRL is now explicitly labeled experimental because its current Python fit performs only one full-batch proximal update; Andersen--Gill is explicitly limited to inverse-information covariance because the API has no subject identifier for clustered recurrent-event inference.

Current validation is 143 passing Python tests and 3 passing Rust tests. The `v0.8.0` release-tag build renders all 94 Quarto pages with live Python execution, including the API overview, all 31 reference pages, and the Ding pages. The earlier Ding failure was a local path problem: `ding_w_source` pointed at the replication directory while the pages correctly append `repl/`; the symlink now points at the parent `Ding_CausalInference` directory. Strict Clippy is improved by removing PyO3 deprecations but still reports 30 pre-existing structural style lints.

## Repository Layout

```text
crabbymetrics/
  Cargo.toml
  pyproject.toml
  devlog.md
  devspec.md
  AGENTS.md
  docs/
  tests/
  src/
    abc.rs
    hyptests.rs
    lib.rs
    rla.rs
    utils.rs
    optimizers.rs
    estimators/
      mod.rs
      linear.rs
      regularized.rs  # includes BaggedPolynomialRegressor
      mle.rs
      gmm.rs
      balancing.rs
      semiparametric.rs
      survival.rs
      transforms.rs
```

Key points:

- `src/lib.rs` is the pyo3 entrypoint and class registry.
- `src/estimators/` is now split by estimator family instead of the old monolithic file layout.
- `src/abc.rs` holds the abundance-based-constraints / weighted-effect-coding OLS estimator.
- `src/hyptests.rs` holds module-level Wald, likelihood-ratio, and IV test helpers.
- `src/rla.rs` holds randomized range/SVD/QR and CountSketch least-squares utilities.
- `src/utils.rs` holds shared array conversion, least-squares helpers, covariance helpers, weighting helpers, and bootstrap utilities.
- `docs/` is a checked-in Quarto website, not just notebook scraps.
- `docs/ding/` holds the translated Peng Ding chapter pages plus grouping pages for the `First Course Ding` section.
- `tests/` is the main regression suite; examples are documented through the Quarto site rather than loose scripts.

## Build And Dev Workflow

- Python environment management uses `uv`.
- Native extension builds use `maturin develop`.
- Rust formatting uses `cargo fmt`; the repo-local pre-commit hook checks that formatting.
- On `master`, the repo-local pre-commit hook also blocks staged rendered docs artifacts (`docs/**/*.html`, `docs/search.json`, `docs/site_libs/`, `*_files/`, `.quarto_ipynb`, and `.jupyter_cache`) so rendered output has to go through `gh-pages` instead.
- `pyproject.toml` includes `tool.uv.cache-keys` for Rust sources so `uv run pytest ...` sees fresh extension builds after Rust changes.
- docs extras are tracked in `pyproject.toml` under `project.optional-dependencies.docs`
  - currently `jupyter`
  - currently `matplotlib`
  - currently `jupyter-cache`
  - currently `pandas`
- test extras are tracked separately and include `lifelines`, `pyfixest`, `scipy`, and `statsmodels` for parity checks and docs-only comparisons.

Useful commands:

```bash
uv run maturin develop
uv run pytest tests -q
uv sync --extra docs
uv run quarto render docs
cargo check
cargo fmt --all --check
```

For notebook-heavy page verification, Quarto execution has recently been more reliable with one-shot kernels:

```bash
uv run quarto render docs/ding/ch11-propensity-score.qmd --execute-daemon 0
```

The full site can still be rendered, but the most stable review path for branches that touch many notebook pages is to rerender the changed pages individually with `--execute-daemon 0`.

## Public Surface

The current Python module exports:

- linear and IV:
  - `OLS`
  - `ABCOLS`
  - `FixedEffectsOLS`
  - `TwoSLS`
  - `HorizontalPanelRidge`
  - `SyntheticControl`
  - `SyntheticDID`
  - `MatrixCompletion`
  - `InteractiveFixedEffects`
- regularized / online:
  - `Ridge`
  - `ElasticNet`
  - `FTRL`
  - `BaggedPolynomialRegressor`
- likelihood / generic:
  - `Logit`
  - `MultinomialLogit`
  - `Poisson`
  - `MEstimator`
- survival / event-time:
  - `ExponentialPH`
  - `WeibullPH`
  - `CoxPH`
  - `AndersenGill`
- moment / semiparametric:
  - `GMM`
  - `BalancingWeights`
  - `EPLM`
  - `AverageDerivative`
  - `PartiallyLinearDML`
  - `AIPW`
- transforms:
  - `PCA` (Rust type `PcaTransformer`)
  - `KernelBasis`
  - `NystromBasis`
  - `RandomFourierFeatures`
  - `RandomizedPCA`
- lower-level optimization surface:
  - `Optimizers`
- module-level helpers:
  - `panel_factor`
  - `panel_fe`
  - `randomized_range`
  - `randomized_qr`
  - `randomized_svd`
  - `qr_solve`
  - `sketch_ols`
  - `av`
  - `optimal_g`
  - `wald_test`
  - `likelihood_ratio_test`
  - `lr_test`

Not every class exposes every method. The broad pattern is still scikit-adjacent, but semiparametric estimators are mostly `fit(...)` plus `summary(...)`, with no meaningful `predict(...)`.

Panel causal API state on `master`:

- `HorizontalPanelRidge`, `SyntheticDID`, and `MatrixCompletion` now use the common public contract `fit(Y, W)`.
- `Y` is a balanced `(n_units, n_periods)` outcome matrix; `W` is a same-shaped binary absorbing treatment matrix.
- The estimators infer treated units, first-treatment cohorts, never-treated donors, and event-time structure internally.
- Fitted summaries now include the modal causal outputs needed for plots/tables: `att`, `counterfactual`, `treatment_effect`, `event_study`, and `group_means`.
- Event-study summaries include both `unweighted` and treated-count `weighted` aggregations with normal-approximation CI columns; group means retain cohort/event-time rows plus weighted and unweighted event-time aggregates for direct plotting.
- `SyntheticControl` and `BalancingWeights` remain public low-level weight APIs, but the paved panel path is estimator-first rather than balancing-weights-first.
- Same Root, panel-DGP, and synthetic-DID docs/examples were migrated to the new `fit(Y, W)` panel API; their Quarto freeze/cache artifacts were flushed and regenerated.

## Docs Surface

The docs site now includes a dedicated `First Course Ding` section alongside the estimator examples and ablations.

Docs-source policy:

- `master` should track Quarto source (`.qmd`), code, and supporting assets only.
- Rendered `.html`, `docs/search.json`, `.quarto_ipynb`, and Jupyter cache artifacts belong in local review builds or the published `gh-pages` branch, not on source branches.

Current Ding coverage:

- Chapters 1 through 8
- Chapter 9 through the cached bridging ablation
- Chapters 11 through 13
- Chapters 21, 23, and 27

The latest Ding pass filled in several places where the first Python pages were too thin relative to the companion R scripts:

- Chapter 5 now includes the post-stratification arithmetic section, not just blocked Penn data.
- Chapter 7 now includes both the Darwin exact sign-flip example and the television matched-pair regression/FRT example.
- Chapter 11 now includes propensity-score stratification over multiple stratum counts, IPW truncation, and balance diagnostics.
- Chapter 13 now includes the ATT doubly robust formula alongside odds weighting and balancing weights.
- Chapter 21 now includes the JOBS one-sided noncompliance example.
- Chapter 23 now includes an Anderson Rubin grid in addition to `TwoSLS`, `GMM`, and the control-function view.
- Chapter 27 now spells out the simulation DGPs before the NDE/NIE histograms.

For estimator docs, `ABCOLS` now has both a worked example page (`docs/examples/abc-ols.qmd`) and a class reference page (`docs/reference/ABCOLS.qmd`), matching the project rule that public classes should not rely on nav tables alone.

Anytime-valid OLS is documented through a worked page (`docs/examples/anytime-valid-ols.qmd`) rather than a class reference page, because the public surface is `OLS.summary(..., anytime_valid=True, ...)` plus module-level helpers `av(...)` and `optimal_g(...)`.

The v0.8.x docs architecture uses `docs/api.qmd` as the full API landing page and keeps the navbar slimmer:

- `API` links to grouped anchors for regression/GLMs, survival/event-time models, causal inference/panels, hypothesis testing, transforms, and estimation interfaces.
- `Regression And GLMs` includes the `mle-prediction-interface` vignette documenting the layered MLE prediction contract.
- `Survival / Time-to-Event / Recurrent Events` has both a richer worked example and compact class reference pages for `ExponentialPH`, `WeibullPH`, `CoxPH`, and `AndersenGill`.
- `Transforms` includes grouped PCA and kernel-transform reference pages covering `PCA`, `RandomizedPCA`, `KernelBasis`, `NystromBasis`, and `RandomFourierFeatures`.

Important docs deployment note: the PyPI release workflow does not deploy the Quarto site. The full `v0.8.0` site is rendered separately from the release tag and published through the `gh-pages` clone workflow; rendered HTML remains off `master`.

The translation rule for that section is:

- keep dependencies to `numpy`, `matplotlib`, and `pandas` only when real data reads require it
- use `crabbymetrics` estimators directly where the chapter logic calls for estimation
- prefer one Quarto page per chapter, with a few grouping pages to keep the navbar manageable

## Inference Surface

### Linear estimators

`OLS`, `Ridge`, `FixedEffectsOLS`, and `TwoSLS` now share the same covariance surface:

- `summary(vcov="vanilla")`
- `summary(vcov="hc1")`
- `summary(vcov="newey_west", lags=...)`
- `summary(vcov="cluster", clusters=...)`

This is one of the main extension-branch improvements. The same shared Rust helpers now build bread / meat / sandwich objects across the linear family.

### Anytime-valid OLS

`OLS.summary(...)` accepts:

- `anytime_valid=True`
- `g=...`
- `level=...`

When enabled, the fitted OLS coefficients remain unchanged, while the summary adds:

- `estimate`
- `std_error`
- `t_value`
- `p_value`
- `confint`
- `confint_level`
- `g`
- `f_statistic`, `f_p_value`, `df_model`, and `df_resid` when an omnibus non-intercept test is defined

The module-level helper `optimal_g(n, number_of_coefficients, alpha)` chooses the mixture precision for a target sample size / confidence level, and `av(model, g=..., vcov="vanilla", level=0.95)` is a thin wrapper around the same summary path.

Tests compare the mtcars example against `avlm` reference values for coefficient p-values, confidence intervals, omnibus F tests, robust covariance paths, helper parity, and validation behavior.

### Poisson

`Poisson.summary(...)` supports:

- `vcov="vanilla"`
- `vcov="sandwich"`
- `vcov="qmle"` is treated as the sandwich path in the user-facing docs

### Likelihood prediction contract

The MLE-style prediction contract was standardized in the v0.7.0 pass:

- `Logit.predict_lin(...)` returns the latent logit index, `predict(...)` returns probabilities, and `predict_label(...)` applies a cutoff.
- `MultinomialLogit.predict_lin(...)` returns classwise scores, `predict(...)` returns softmax probabilities, and `predict_label(...)` returns argmax labels.
- `Poisson.predict_lin(...)` returns log conditional means, while `predict(...)` returns mean-scale counts.
- The `docs/examples/mle-prediction-interface.qmd` page documents these identities with direct checks.

### Survival / event-time

The first survival module is landed and tested:

- `ExponentialPH` fits a parametric proportional-hazards model with constant baseline hazard and exposes log hazard, hazard, cumulative hazard, and survival predictions.
- `WeibullPH` fits a parametric proportional-hazards model with a Weibull baseline and exposes the same absolute prediction surfaces.
- `CoxPH` fits a semiparametric Cox partial-likelihood model and defaults to relative-risk predictions because the baseline hazard is not yet estimated.
- `AndersenGill` extends the Cox risk-set calculation to counting-process intervals `(start, stop]` for recurrent events or split records.

Tests compare Cox, exponential, and Weibull fits against `lifelines` parameterizations and check prediction-surface identities.

### GMM

`GMM.summary(...)` supports:

- `vcov="vanilla"`
- `vcov="sandwich"`

and separate moment-covariance choices:

- `omega="iid"`
- `omega="newey_west"`
- `omega="cluster"`

### Semiparametric estimators

`EPLM`, `AverageDerivative`, `PartiallyLinearDML`, and `AIPW` use exact-identified influence-function covariance calculations with:

- `vcov="vanilla"`
- `vcov="hc1"`
- `vcov="newey_west"`
- `vcov="cluster"`

Defaults are conservative:

- `hc1` for the semiparametric classes
- explicit clipping in `AIPW` to stabilize the ridge-based propensity nuisance

### Balancing weights

`BalancingWeights` is not yet a full inference object. Its `summary()` returns:

- fitted weights
- mean-balance diagnostics
- effective sample size
- optimization diagnostics

The estimator supports quadratic, entropy, and active-branch Cressie-Read
calibration geometries. Cressie-Read uses the dual map controlled by
`divergence_power`; Renyi divergence is documented as a fitted-weight diagnostic,
not as its own optimizer. The estimator is currently best viewed as a weighting
primitive with strong diagnostics rather than a one-stop causal-inference summary
object.

## Key Implementation Decisions

### 1. Prefer least-squares decompositions over explicit inverse formulas

Linear estimation paths now lean on QR / least-squares style solves analogous to `np.linalg.lstsq`. This especially matters for:

- `OLS`
- `Ridge`
- `TwoSLS`

The public formulas in docs still use the familiar econometrics notation, but the implementation avoids building the estimator through raw normal-equation inversion when a stable solve is cleaner.

### 2. Keep linear IV closed-form

`TwoSLS` is a real estimator class, not a special case of generic optimizer-driven GMM. It supports:

- multiple endogenous regressors
- multiple excluded instruments
- weighted fits through `fit_weighted(...)`

This is the fast path. Generic `GMM` exists for stacked moments and nonlinear score systems, not to replace closed-form linear IV.

### 3. Keep GMM minimal

The current `GMM` scope is intentionally narrow:

- just-identified moment systems
- two-step overidentified GMM
- stacked moment conditions

No continuously updated weighting and no large optimizer zoo were added. The design choice was to stop at the point where there was a real application.

### 4. Weighted linear fits use square-root row scaling

Weighted linear estimators are implemented by transforming the design and outcome with `sqrt(w)` where that algebra is exact. This currently covers:

- `OLS`
- `Ridge`
- `FixedEffectsOLS`
- `TwoSLS`

The nonlinear families do not yet have weighted fits.

### 5. Semiparametric estimators are narrow by design

The semiparametric module is intentionally opinionated:

- `EPLM` is a stacked-moment partially linear E-estimator
- `AverageDerivative` implements OB / IPW / DR variants for a scalar continuous treatment
- `PartiallyLinearDML` uses cross-fit ridge nuisances
- `AIPW` uses cross-fit ridge outcome models and a clipped ridge propensity nuisance

This is not a general DML framework. The choices are explicit so the Rust layer stays compact and testable.

### 6. Fold splitting is seeded and reproducible

Cross-fit estimators use a deterministic seeded hash shuffle. `AIPW` assigns folds within treatment strata so each nuisance-training split retains both treatment arms when the data permit. This makes:

- tests exact and reproducible
- debugging straightforward
- docs examples stable across rebuilds
- different seeds change fold membership rather than only relabeling fixed row blocks

The split is random-looking but reproducible for a fixed seed; row order is not used as the fold boundary.

### 7. Docs are part of the product

The Quarto site is checked in and is expected to stay coherent. Important conventions now in force:

- `embed-resources: true`
- full-width pages
- estimator API, reference, and worked-example code visible by default
- setup, utility, and ablation code folded only where the page configuration calls for it
- ablation pages under `docs/ablations/` use `execute.cache: true`
- ablation pages under `docs/ablations/` use `freeze: auto`

Heavy pages are expected to render once and then be reused from cache/freeze on later site renders.

## Current Docs Structure

The docs site is organized around:

- `Home`
- `API`
- `Binding Crash Course`
- `Regression And GLMs`
- `Causal Inference`
- `Transforms`
- `Ablations`
- `Optimization`
- `Ding: First Course`

Important likelihood and event-time pages now in the nav:

- MLE prediction interface
- logit
- multinomial logit
- Poisson
- survival / time-to-event / recurrent events
- compact survival models overview

Important causal and semiparametric pages now in the nav:

- balancing weights
- EPLM
- average derivative
- double ML and AIPW
- mediation through the Ding Chapter 27 page

Important cached ablations:

- variance-estimator comparisons
- semiparametric estimator comparisons
- panel-DGP estimator comparisons
- Same Root Basque/California panel case studies with semi-synthetic simulation, the first-class `HorizontalPanelRidge` estimator, vertical ridge, synthetic DID, matrix completion, HAC intervals, and donor-placebo inference

## Testing State

Current expectations:

- `uv run pytest tests -q` is the main Python-side regression suite
- `cargo check` is the main Rust sanity pass
- `uv run quarto render docs` is part of verification for any docs-heavy branch

As of `v0.8.0`, the Python regression suite has 143 passing tests.

The test suite now covers:

- exact numerical matches for many linear, IV, GMM, likelihood, survival, and semiparametric formulas
- weighted linear estimators
- balancing-weight diagnostics
- semiparametric failure modes and constructor validation
- randomized linear algebra, randomized estimator paths, and randomized transformers
- survival parity against `lifelines` for Cox, exponential, and Weibull parameterizations
- anytime-valid OLS parity against `avlm` reference calculations

## Known Caveats

### Public docs deployment is separate from package release

`v0.8.0` is published to PyPI, but GitHub Pages is served from `gh-pages`, not from `master`. The release site is therefore rebuilt and pushed separately; generated HTML, search indexes, and Quarto execution artifacts stay off the source branch.

### Transform math and numerical contracts are now explicit

The grouped PCA and kernel-transform references now cover all five exported classes. The audit also fixed `PCA` to report $s_j^2/(n-1)$, compute explained-variance ratios against total centered variance, and correctly undo whitening in `inverse_transform`. The Gaussian kernel page now matches the implementation's bandwidth convention, $\exp(-\lVert x-z\rVert^2/h)$.

### Weighted support is still incomplete

Weighted fits are only in the linear family so far. `Logit`, `Poisson`, and `GMM` still need a weighted story if that becomes a priority again.

### `AIPW` uses ridge for the propensity nuisance

This is intentional and keeps the dependency story clean, but it means:

- it is not a literal logistic-propensity implementation
- clipping is required for finite-sample stability
- some designs will favor direct balancing weights instead

### `MEstimator` is still the least polished estimator surface

It is useful as an escape hatch, but it still has limitations:

- covariance uses a numerical Jacobian for the bread and empirical score outer products for the meat
- Python callbacks keep it outside the all-Rust hot-path design
- solver diagnostics are thin
- it is best treated as a low-level custom hook, not as the flagship inference path

### `FTRL` is not yet a coherent online estimator API

The current wrapper initializes a fresh upstream state, computes one aggregate full-batch logistic gradient, applies one proximal update, and discards the optimizer state. There is no intercept, `partial_fit`, iteration control, or convergence target. It should remain experimental until it becomes a persistent online estimator or is removed from the first-tier public surface.

### `AndersenGill` lacks recurrent-event robust inference

The point estimator is the counting-process Cox partial likelihood with Breslow ties, but the public API has no subject identifier. Its inverse-information covariance is not the subject-clustered sandwich variance generally needed when subjects contribute recurrent events. Both `CoxPH` and `AndersenGill` also rescan all rows for every event and build dense second moments, costing approximately $O(Enp^2)$ per Newton iteration.

## Current Direction

After the correctness and class-by-class API audit, the most plausible next branch directions are:

1. decide whether to redesign `FTRL` around persistent `partial_fit` state or remove it from the first-tier estimator surface
2. add a subject identifier and clustered score covariance to `AndersenGill`, then optimize Cox-family risk-set accumulation
3. expose convergence status for iterative estimators that currently retain budget-exhausted results, especially `MatrixCompletion` and survival fits
4. difference-in-differences / event-study support
5. more likelihood methods, starting with negative binomial and grouped/binomial likelihoods
6. weighted nonlinear estimators, weighted GMM, and richer IV / GMM diagnostics

That is the current state the next extension branch should assume.

## 2026-05-03 Docs Taxonomy And Panel Examples

- Refactored the docs navigation away from the old `Supervised Learning`, `Semiparametrics`, and `Unsupervised Learning` buckets.
- The site now uses `Regression And GLMs`, `Causal Inference`, and `Transforms`; causal pages include semiparametric estimators, IV, synthetic control, synthetic DID, horizontal panel ridge, matrix completion, and interactive fixed effects.
- Added self-contained panel-data example pages for `HorizontalPanelRidge`, `MatrixCompletion`, and `InteractiveFixedEffects`.
- Added an `AGENTS.md` docs contract: each public-facing class exposed from `src/lib.rs` should have a clean example/docs page linked from the site navigation; the generated API table is not enough by itself.

## 2026-05-03 Staggered Panel Event-Study Vignette

- Added a causal-inference docs vignette for the Hainmueller--Hangartner 2019 municipal naturalization panel from the local `FTestEventStudy` sandbox.
- The vignette builds balanced `Y` and absorbing `W` matrices, fits `HorizontalPanelRidge`, `MatrixCompletion`, and `SyntheticDID` via `fit(Y, W)`, and plots the resulting weighted event-study summaries.
- It also uses local-docs-only `pyfixest` to compare a vanilla binned TWFE event study with a saturated/Sun-Abraham-style event study; no package dependency metadata was added.
- The page includes explicit source code for each panel of the combined adoption/event-study figure.

## 2026-05-03: SyntheticDID time-weight ATT patch

- Patched `SyntheticDID` so scalar `summary()["att"]` uses the cohort-specific synthetic difference-in-differences matrix contrast from the local `synthlearners/notebooks/cvxpy_reference.ipynb` reference: unit weights `[-omega, 1/N1]` and time weights `[-lambda, 1/T1]`.
- Kept `counterfactual`, `treatment_effect`, and `event_study` as the period-by-period unit-weighted synthetic-control gap path; time weights now enter the scalar ATT rather than rewriting the plotted dynamic path.
- Added a regression test where the SDID ATT differs from the plain post-treatment synthetic-control gap average.

## 2026-05-08 Randomized Linear Algebra Sketches PR

- Branch: `feature/randomized-linear-algebra`, PR #8.
- Added native Rust randomized linear algebra helpers in `src/rla.rs` using `ndarray` and `nalgebra` rather than vendoring external code.
- Exposed Python functions `randomized_range`, `randomized_svd`, and `sketch_ols`.
- Added `OLS.fit_sketch(...)` backed by a CountSketch row embedding for tall least-squares designs.
- Added regression tests in `tests/test_randomized_linear_algebra.py` covering randomized range orthonormality, randomized SVD reconstruction, standalone `sketch_ols`, `OLS.fit_sketch`, and sketch-size validation.
- Moved the sketching OLS ablation into `docs/ablations/randomized-sketching-ols.qmd` with rendered HTML and freeze outputs, and linked it from the docs navbar and index.
- Rendered review copy: `https://lalten.org/drafts/crabbymetrics-randomized-sketching-ols-pr8.html`.
- Local gates run on the branch: `cargo check`, `uv run maturin develop`, `uv run pytest`, `QUARTO_PYTHON=.venv/bin/python quarto render docs/ablations/randomized-sketching-ols.qmd`, and `QUARTO_PYTHON=.venv/bin/python quarto render docs/index.qmd`.

## 2026-05-08 Randomized QR And IV Sketching Follow-up

- Extended PR #8 beyond randomized SVD/range and CountSketch OLS.
- Added `randomized_qr(...)` and `qr_solve(...)` Python exports backed by the native Rust randomized range/QR helpers.
- Added `TwoSLS.fit_sketch(...)`, which applies one CountSketch embedding jointly to the IV regressor design, instrument design, and outcome before solving the compressed 2SLS problem.
- Cleaned the randomized sketching docs page so it is official-package math plus OLS/IV ablations, without upstream-package references or development-roadmap language.

## 2026-05-09 Estimator-Level RLA Integrations

- Completed the follow-on sketching sequence on `feature/randomized-linear-algebra` as opt-in estimator integrations rather than changing exact defaults.
- Added randomized SVD paths to `MatrixCompletion` and panel-factor extraction (`panel_factor` / `InteractiveFixedEffects`) with explicit rank/oversampling/power-iteration/seed controls.
- Added reusable transform classes `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA` for kernel approximations and wide-feature compression before downstream estimators such as `Ridge` and `BalancingWeights`.
- Added conservative `GMM.fit_sketch(...)` for many-moment systems using a fixed Rademacher projection over moments/Jacobians.
- Fixed weighted `TwoSLS.summary(...)` so weighted summaries use the same transformed-design convention as weighted fitting.
- Local gates: `cargo fmt`, `cargo check`, `uv run maturin develop`, targeted estimator tests, and full `uv run pytest -q` (`94 passed`).

## 2026-05-23 Hypothesis-Test Helpers

- Branch: `hyptests`.
- Added estimator-level `wald_test(...)` methods for `OLS`, `FixedEffectsOLS`, `TwoSLS`, `Logit`, `Poisson`, and `GMM` so fitted objects own the covariance calculation for joint linear restrictions `R beta = q`.
- Kept module-level `wald_test(coef, vcov, r, q=None)` as an array-level primitive for manual workflows.
- Added `likelihood_ratio_test(unrestricted_loglik, restricted_loglik, df)` plus `lr_test(...)` alias for nested likelihood comparisons.
- Summary dictionaries for `OLS`, `FixedEffectsOLS`, `TwoSLS`, `Logit`, and `Poisson` now include the full covariance matrix as `vcov`, making array-level Wald tests usable directly from fitted summaries.
- Added focused tests in `tests/test_hyptests.py` and a short docs page at `docs/reference/HypothesisTests.qmd`.
- Added `TwoSLS.anderson_rubin_test(beta=0.0, vcov="hc1", ...)` for weak-IV-robust scalar endogenous-regressor tests, with reduced-form F-test validation against `statsmodels`.

## 2026-05-24 ABC OLS, Release, And Packaging Guard

- Merged PR #11, `Add ABC OLS estimator and memo`, into `master`.
- Added `cm.ABCOLS()` for abundance-based constraints / weighted effect coding in OLS, covering categorical main effects, continuous-by-categorical interactions, and categorical-by-categorical interactions.
- Added `docs/examples/abc-ols.qmd` with the overcomplete design, constraints, hand-written null-space comparison, `ABCOLS` fit, true DGP targets, coefficient tables comparing vanilla one-hot/reference-coded OLS to ABC, and the Lin (2013) / Kowal interpretation upshot.
- Released `v0.6.7` after `v0.6.6` exposed a PyPI sdist-size failure. The underlying issue was rendered Quarto docs being included in source distributions: `0.5.1` was already 84.2 MB / 80.3 MiB, `0.6.5` reached 102.2 MB / 97.5 MiB, and `0.6.6` crossed the effective PyPI upload limit. `pyproject.toml` now excludes `docs/**/*` from sdists, and `0.6.7` published with a small ~0.4 MB sdist.
- Rendered and deployed the public docs site to `gh-pages`; spot-check confirmed the live ABC OLS page includes `ABCOLS`, Lin (2013), and the vanilla one-hot comparison.
- Added `AGENTS.md` release-packaging guidance requiring sdist size checks before tagging when docs changed.

## 2026-05-25 v0.7.0 Docs Architecture And MLE Prediction Interface

- Released `v0.7.0` and overhauled the docs architecture around a slimmer navbar plus a fuller `docs/api.qmd` landing page.
- Added `docs/examples/mle-prediction-interface.qmd`, documenting the layered prediction contract for MLE-style estimators: `predict_lin(...)` for latent scores, `predict(...)` for natural mean-scale predictions, and `predict_label(...)` only for classifiers.
- Updated `Logit`, `MultinomialLogit`, and `Poisson` docs/examples to use the new contract explicitly.
- Added source-branch hygiene around rendered docs: source branches carry Quarto source and assets, while rendered output is for local review or `gh-pages`.

## 2026-05-25 Survival Estimators

- Added `src/estimators/survival.rs` and exported `ExponentialPH`, `WeibullPH`, `CoxPH`, and `AndersenGill`.
- Parametric PH models now expose hazard, cumulative-hazard, and survival predictions; Cox-style models expose relative-risk predictions.
- Added class reference pages for all four survival estimators plus two worked vignettes: `docs/examples/survival-time-to-event-and-recurrent-events.qmd` and `docs/examples/survival-models.qmd`.
- Added `tests/test_survival.py` with simulated-DGP recovery, `lifelines` parity checks, Cox/Andersen-Gill split-row equivalence, and prediction-surface identity checks.

## 2026-06-27 v0.7.1 Anytime-Valid OLS Release

- PR #14 landed via squash merge as `895d25a Add anytime-valid OLS summaries (#14)`.
- Added anytime-valid OLS inference through `OLS.summary(vcov=..., anytime_valid=True, g=..., level=...)`, preserving ordinary OLS point estimates while adding anytime-valid p-values, confidence intervals, and omnibus F-test fields.
- Exported module-level helpers `optimal_g(...)` and `av(...)`.
- Added `docs/examples/anytime-valid-ols.qmd`, README snippet, API overview links, navbar entry, and `docs/llms.txt` guidance for the new summary path.
- Added `tests/test_anytime_valid.py` with 7 tests covering `avlm` mtcars parity values, robust covariance behavior, helper parity, confidence-radius optimization, and input validation.
- Released `v0.7.1` through the `Build wheels` workflow. The workflow bumped `Cargo.toml` to `0.7.1`, created tag `v0.7.1`, passed test jobs for Python 3.10 and 3.12, built Linux/macOS wheels for Python 3.10 through 3.14, published the GitHub Release, and published to PyPI.
- Local pre-release sdist sanity check produced a small source distribution (`crabbymetrics-0.7.0.tar.gz` at roughly 370 KiB before the workflow bumped the version); the published `v0.7.1` GitHub Release sdist is similarly small (`crabbymetrics-0.7.1.tar.gz`, 377,323 bytes).
- Current release links: PyPI latest is `0.7.1`; GitHub Release is `https://github.com/apoorvalal/crabbymetrics/releases/tag/v0.7.1`; release workflow run is `https://github.com/apoorvalal/crabbymetrics/actions/runs/28292326463`.
- Follow-up still pending: deploy rendered docs to `gh-pages` if the public docs site should show the new anytime-valid OLS page immediately.

## 2026-07-11 v0.8.0 Estimator Audit Release

- PR #16 landed via squash merge as `bbf7868 Correct estimator inference, add bagged polynomial regression, and audit APIs (#16)`.
- Released `v0.8.0` through the `Build wheels` workflow at `https://github.com/apoorvalal/crabbymetrics/actions/runs/29166538588`.
- The workflow passed Python 3.10 and 3.12 test jobs, built Linux and macOS wheels for Python 3.10 through 3.14, published 10 wheels plus a 410,390-byte sdist, created the GitHub Release, and published to PyPI.
- Local release gates passed with 143 Python tests, 3 Rust tests, Rust formatting, and a clean-checkout 404 KiB sdist containing neither `docs/` nor the untracked local `ding_ci` notebooks.
- Rebuilt all 94 public Quarto pages from the `v0.8.0` tag. The stale local `ding_w_source` symlink had pointed one directory too deep; pointing it at the parent `Ding_CausalInference` directory restored every external replication-data path.
- Release links: PyPI version `0.8.0`; GitHub Release `https://github.com/apoorvalal/crabbymetrics/releases/tag/v0.8.0`; public docs `https://apoorvalal.github.io/crabbymetrics/`.
