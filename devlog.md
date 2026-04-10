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
  - `FixedEffectsOLS`
  - `TwoSLS`
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

## Docs Surface

The docs site now includes a dedicated `First Course Ding` section alongside the estimator examples and ablations.

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
- Chapter 23 now includes an Anderson-Rubin grid in addition to `TwoSLS`, `GMM`, and the control-function view.
- Chapter 27 now spells out the simulation DGPs before the NDE/NIE histograms.

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
