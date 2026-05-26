# Devlog (crabbymetrics)

## Snapshot

`crabbymetrics` is a Rust-backed econometrics library exposed to Python through `pyo3` and `maturin`. The project is intentionally narrow:

- Rust owns the numerical work
- Python stays NumPy-only and scikit-adjacent
- docs are checked in as a Quarto site under `docs/`
- the current surface is stronger on econometrics estimators and inference than on generic ML breadth

This file is meant to record the current architecture and the design choices that matter for future work.

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
    lib.rs
    utils.rs
    optimizers.rs
    estimators/
      mod.rs
      linear.rs
      regularized.rs
      mle.rs
      gmm.rs
      balancing.rs
      semiparametric.rs
      transforms.rs
```

Key points:

- `src/lib.rs` is the pyo3 entrypoint and class registry.
- `src/estimators/` is now split by estimator family instead of the old monolithic file layout.
- `src/utils.rs` holds shared array conversion, least-squares helpers, covariance helpers, weighting helpers, and bootstrap utilities.
- `docs/` is a checked-in Quarto website, not just notebook scraps.
- `docs/ding/` holds the translated Peng Ding chapter pages plus grouping pages for the `First Course Ding` section.
- `tests/` is the main regression suite; examples are documented through the Quarto site rather than loose scripts.

## Build And Dev Workflow

- Python environment management uses `uv`.
- Native extension builds use `maturin develop`.
- Rust formatting uses `cargo fmt`; the repo-local pre-commit hook checks that formatting.
- `pyproject.toml` includes `tool.uv.cache-keys` for Rust sources so `uv run pytest ...` sees fresh extension builds after Rust changes.
- docs extras are tracked in `pyproject.toml` under `project.optional-dependencies.docs`
  - currently `matplotlib`
  - currently `jupyter-cache`
  - currently `pandas`

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
- regularized / online:
  - `Ridge`
  - `ElasticNet`
  - `FTRL`
- likelihood / generic:
  - `Logit`
  - `MultinomialLogit`
  - `Poisson`
  - `MEstimator`
- moment / semiparametric:
  - `GMM`
  - `BalancingWeights`
  - `EPLM`
  - `AverageDerivative`
  - `PartiallyLinearDML`
  - `AIPW`
- transforms:
  - `PcaTransformer`
  - `KernelBasis`
- lower-level optimization surface:
  - `Optimizers`

Not every class exposes every method. The broad pattern is still scikit-adjacent, but semiparametric estimators are mostly `fit(...)` plus `summary(...)`, with no meaningful `predict(...)`.

Panel causal API update on the matrix-completion branch:

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

### Poisson

`Poisson.summary(...)` supports:

- `vcov="vanilla"`
- `vcov="sandwich"`
- `vcov="qmle"` is treated as the sandwich path in the user-facing docs

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
- `Supervised Learning`
- `Semiparametrics`
- `Unsupervised Learning`
- `Ablations`
- `Optimization`

Important semiparametric pages now in the nav:

- balancing weights
- EPLM
- average derivative
- double ML and AIPW
- mediation via a doc-level Baron-Kenny translation

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

The test suite now covers:

- exact numerical matches for many linear, IV, GMM, and semiparametric formulas
- weighted linear estimators
- balancing-weight diagnostics
- semiparametric failure modes and constructor validation

## Known Caveats

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

After the semiparametric and balancing work, the most plausible next branch directions are:

1. difference-in-differences / event-study support
2. negative binomial regression
3. linear restriction / Wald test helpers
4. weighted nonlinear estimators and weighted GMM
5. richer IV / GMM diagnostics

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
