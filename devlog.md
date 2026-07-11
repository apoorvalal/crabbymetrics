# Devlog (crabbymetrics)

## Snapshot

`crabbymetrics` is a Rust-backed econometrics library exposed to Python through `pyo3` and `maturin`. The project is intentionally narrow:

- Rust owns the numerical work
- Python stays NumPy-only and scikit-adjacent
- docs are checked in as a Quarto site under `docs/`
- the current surface is stronger on econometrics estimators and inference than on generic ML breadth

Current release state: `v0.7.1` is published to PyPI and GitHub Releases. The release contains the anytime-valid OLS surface added in PR #14, plus the broader `v0.7.0` docs / likelihood / survival work.

This file is meant to record the current architecture and the design choices that matter for future work.

## Refactor Correctness Pass (2026-07-10)

The `refactor` branch was created from `origin/master` 0.7.0 after preserving the audited dirty 0.5.1 tree on `pre-refactor-audit-snapshot`, then merged with current `master` 0.7.1. It fixes the review P0/P1 set: weighted TwoSLS, generic M-estimator covariance, identified multinomial inference, penalized-estimator inference contracts, shuffled/stratified cross-fitting, absorbed fixed-effect degrees of freedom, covariance-diagonal validation, and convergence semantics.

`BaggedPolynomialRegressor` is now a guarded prediction-only class in `regularized.rs`. It standardizes polynomial columns inside each learner, stores random subspaces and resolved dimensions, reports OOB MSE/coverage, enforces dense-design limits, and has direct scikit-learn parity tests. The public docs include a reference page and a leakage-free repeated-draw demo; both leave executable code visible by default. The repository-level baseline audit and remediation record is `evaluation-review.qmd`.

Current validation is 141 passing Python tests and 3 passing Rust tests. All changed executable Quarto pages render. A 94-page no-execute site assembly passes; the full executing site remains blocked by the pre-existing absent `ding_w_source/repl/nhanes_bmi.csv`. Strict Clippy is improved by removing PyO3 deprecations but still reports 30 pre-existing structural style lints.

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

The v0.7.x docs architecture uses `docs/api.qmd` as the full API landing page and keeps the navbar slimmer:

- `API` links to grouped anchors for regression/GLMs, survival/event-time models, causal inference/panels, hypothesis testing, transforms, and estimation interfaces.
- `Regression And GLMs` includes the `mle-prediction-interface` vignette documenting the layered MLE prediction contract.
- `Survival / Time-to-Event / Recurrent Events` has both a richer worked example and compact class reference pages for `ExponentialPH`, `WeibullPH`, `CoxPH`, and `AndersenGill`.
- `Transforms` includes the PCA/kernel page, while randomized transformers are represented in the API surface and RLA ablation until they get deeper standalone docs.

Important docs deployment note: the PyPI release workflow does not deploy the Quarto site. After `v0.7.1`, source `master` contains the anytime-valid OLS page and nav/API links, but public GitHub Pages still needs the separate `gh-pages` publish path before the live site reflects those source docs.

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

The estimator is currently best viewed as a weighting primitive with strong diagnostics rather than a one-stop causal-inference summary object.

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

### 6. Fold splitting is deterministic

Cross-fit estimators use deterministic fold assignment by row order plus a seed offset. This makes:

- tests exact and reproducible
- debugging straightforward
- docs examples stable across rebuilds

This is a deliberate tradeoff in favor of reproducibility over random fold shuffling.

### 7. Docs are part of the product

The Quarto site is checked in and is expected to stay coherent. Important conventions now in force:

- `embed-resources: true`
- full-width pages
- code kept in the doc but folded where appropriate
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

As of `v0.7.1`, the Python regression suite has 126 test functions across 16 test files.

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

`v0.7.1` is published to PyPI, but GitHub Pages is served from `gh-pages`, not from `master`. Docs changes such as `docs/examples/anytime-valid-ols.qmd` need the separate rendered-site deployment workflow before they are visible at the public docs URL.

### Transform docs are thinner than the Python surface

The Python module exports `PCA`, `KernelBasis`, `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA`, but `docs/api.qmd` and the current transform nav still foreground only PCA/kernel examples. The randomized/Nystrom/RFF transforms are tested and referenced by the RLA ablation, but they still need deeper first-class reference/example docs if they become user-facing priorities.

### Weighted support is still incomplete

Weighted fits are only in the linear family so far. `Logit`, `Poisson`, and `GMM` still need a weighted story if that becomes a priority again.

### `AIPW` uses ridge for the propensity nuisance

This is intentional and keeps the dependency story clean, but it means:

- it is not a literal logistic-propensity implementation
- clipping is required for finite-sample stability
- some designs will favor direct balancing weights instead

### `MEstimator` is still the least polished estimator surface

It is useful as an escape hatch, but it still has limitations:

- variance uses a score-outer-product approximation for both bread and meat
- solver diagnostics are thin
- it is best treated as a low-level custom hook, not as the flagship inference path

### PyO3 deprecation warnings remain

The codebase still emits PyO3 deprecation warnings around `with_gil` and older downcast helpers. They do not currently block builds, but the warning debt is real.

## Current Direction

After the randomized linear algebra, ABC OLS, anytime-valid OLS, hypothesis-test, MLE prediction, and survival work, the most plausible next branch directions are:

1. docs housekeeping: publish the rendered `v0.7.1` docs site and fill in transform docs for `NystromBasis`, `RandomFourierFeatures`, and `RandomizedPCA`
2. difference-in-differences / event-study support
3. more likelihood methods, starting with negative binomial and grouped/binomial likelihoods
4. weighted nonlinear estimators and weighted GMM
5. richer IV / GMM diagnostics
6. Cox baseline-hazard / survival-curve follow-ons

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
