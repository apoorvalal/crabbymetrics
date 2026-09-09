# Estimator scaling benchmark

This benchmark inventories every estimator class exported by `crabbymetrics`,
maps it to a credible open-source reference, and measures scaling in an isolated
process. It interprets “10e3 to 10e7” as the conventional $10^3$ through $10^7$
range. The default sample grid is therefore
`1_000, 10_000, 100_000, 1_000_000, 10_000_000`, crossed with
`k = 5, 10, 20, 50, 100`.

The comparison order follows the project rule: scikit-learn when the estimator
has a matching API/estimand, then PyFixest, then a canonical R implementation.
A reference URL is not automatically a timing comparator. The registry marks a
method runnable only when the adapter is semantically defensible on the common
grid. “Nearest” references document related code, but are never plotted or
described as equivalent.

## Safety and reproducibility

Every cell runs in a fresh subprocess. The driver:

- estimates peak allocation before launch and records `preflight_oom` when the
  cell would exceed the cap;
- watches the aggregate RSS of the child and all descendants, terminating the
  tree at the hard cap;
- applies a per-cell wall-clock timeout;
- pins BLAS/OpenMP/Rcpp thread counts to one;
- prunes all larger `n` values for the same estimator, implementation, and `k`
  after a timeout, hard-RSS kill, preflight rejection, or execution error;
- writes both a long-form CSV and a host/configuration JSON file.

Grid dimensions are validated, deduplicated, and sorted before execution so
pruning always proceeds from smaller to larger samples. Missing executables and
dependencies are recorded as `missing_dependency` and prune the remaining path.
Unknown or provenance-only implementation names fail before launching the grid.

Results are persisted after each cell. `--append` preserves existing CSV column
order, extends the schema when new fields appear, and retains prior host/run
configurations in `previous_runs`. Output is spooled to temporary files while
the parent monitors the process, with bounded tails retained in memory. On POSIX
systems, cleanup terminates the cell's process group, including surviving workers.
Nonzero exits and invalid/nonfinite result payloads cannot be recorded as success.

The automatic cap is the smaller of 40% of physical RAM and currently available
RAM minus a 4 GiB system reserve. Override it only deliberately with
`--memory-gib`. A `10_000_000 x 100` `float64` matrix is 8 GB before copies or
solver workspaces, so large cells are expected to be skipped on ordinary hosts.
An explicit cap never exceeds available memory minus the reserve; when that
headroom is exhausted, all cells are rejected before launch.

## Adapter revisions

New rows and host metadata carry `adapter_revision=2` and the data-generation
seed. The committed August 2026 results predate this revision and are retained
as historical measurements, not silently relabeled as corrected comparisons.
Rerun affected paths before using their timings for comparative conclusions:

- sklearn logit and multinomial logit now explicitly use `C=np.inf`; Poisson
  uses `alpha=0.0`, matching the native unpenalized objectives.
- Both elastic-net adapters center the same design and response before fitting,
  matching the centered-design parity contract.
- Horizontal panel ridge now compares the native panel fit with the actual
  sklearn donor regression on the same panel. The previous sklearn adapter
  incorrectly fit an ordinary tabular ridge problem.
- Native and PyFixest 2SLS now share outcome, treatment, and instrument draws.
- R timing now excludes data-frame construction, package loading, and checksum
  calculation, like the Python fit timer. R and NumPy still use independent RNGs:
  these are distribution-matched designs, not identical sample realizations.

The Python timer is per cell rather than module-global. The requested seed also
initializes NumPy's legacy RNG for reproducible DoubleML fold draws; estimator
adapters with an explicit internal seed retain the documented fixed value 1729.
Different adapter revisions should not be pooled into a single timing summary.

Run the full guarded grid:

```bash
uv run python benchmarks/scaling/run_grid.py
```

The committed 2026-08-25 run used `--timeout 15` on a 16 GiB Apple-silicon
host. Its automatically selected hard cap was 1,960,706,048 bytes (1.83 GiB),
leaving a 4 GiB operating-system reserve at launch.

Run a smoke grid:

```bash
uv run python benchmarks/scaling/run_grid.py \
  --n 1000,10000 --k 5,20 --timeout 30 \
  --output /tmp/crabbymetrics-scaling-smoke.csv
```

Run native implementations only:

```bash
uv run python benchmarks/scaling/run_grid.py --implementations native
```

The committed analysis page is
[`docs/ablations/estimator-scaling.qmd`](../../docs/ablations/estimator-scaling.qmd).
It documents the DGP, exact fitted specification, reference interpretation,
completion frontier, failure counts, observed RSS, and log-log runtime slope for
each estimator separately. Those descriptions live in
[`report_metadata.py`](report_metadata.py), and the registry coverage test
requires one complete description for every exported estimator.

## Reference parity gate

The scaling adapters are backed by small deterministic solution-parity tests in
[`tests/test_external_reference_parity.py`](../../tests/test_external_reference_parity.py).
The suite checks OLS, ridge, centered-design elastic net, binary and multinomial
logit, Poisson, and horizontal panel ridge against scikit-learn; one-way fixed
effects and overidentified 2SLS under IID and HC1 covariance against PyFixest;
and Andersen--Gill against lifelines' time-varying Cox fitter. The existing
survival and polynomial-regression tests add Cox, exponential, Weibull, and
polynomial-pipeline reference coverage.

Install every reference package and run the gate with:

```bash
uv run --group test pytest -q tests/test_external_reference_parity.py
```

The runner and adapter regression gate is `tests/test_scaling_runner.py`. It
checks process cleanup, output capture, CSV persistence, pruning, budget
validation, and the exact data/objectives passed to the corrected adapters.

## Estimator inventory and references

| Crabbymetrics estimator | Preferred reference(s) | Generic-grid status |
|---|---|---|
| `ABCOLS` | [`wec`](https://github.com/cran/wec) | provenance; weighted-effect coding is the closest public R implementation |
| `OLS` | [scikit-learn](https://github.com/scikit-learn/scikit-learn) `LinearRegression` | runnable |
| `FixedEffectsOLS` | [PyFixest](https://github.com/py-econometrics/pyfixest), [fixest](https://github.com/lrberge/fixest) | runnable |
| `ElasticNet` | [scikit-learn](https://github.com/scikit-learn/scikit-learn) `ElasticNet` | runnable |
| `Ridge` | [scikit-learn](https://github.com/scikit-learn/scikit-learn) `Ridge` | runnable |
| `BaggedPolynomialRegressor` | scikit-learn `PolynomialFeatures` + `Ridge` + `BaggingRegressor` | runnable, matched pipeline |
| `Logit` | scikit-learn `LogisticRegression` | runnable |
| `MultinomialLogit` | scikit-learn `LogisticRegression` | runnable |
| `Poisson` | scikit-learn `PoissonRegressor` | runnable |
| `ExponentialPH` | [`flexsurv`](https://github.com/cran/flexsurv) | provenance; parameterization differs |
| `WeibullPH` | [`flexsurv`](https://github.com/cran/flexsurv) | provenance; parameterization differs |
| `CoxPH` | [lifelines](https://github.com/CamDavidsonPilon/lifelines), [survival](https://github.com/cran/survival) | runnable |
| `AndersenGill` | [survival](https://github.com/cran/survival) counting-process `coxph` | runnable |
| `TwoSLS` | [PyFixest](https://github.com/py-econometrics/pyfixest), [`ivreg`](https://github.com/cran/ivreg) | PyFixest runnable |
| `HorizontalPanelRidge` | scikit-learn `Ridge` by cohort | runnable building block |
| `SyntheticControl` | [`Synth`](https://github.com/cran/Synth) | provenance; preprocessing interface is not generic-grid equivalent |
| `SyntheticDID` | [`synthdid`](https://github.com/synth-inference/synthdid) | provenance; exact external implementation |
| `AugmentedBalancing` | [`augsynth`](https://github.com/ebenmichael/augsynth), independent `quadprog` fixture | provenance/parity fixture |
| `MatrixCompletion` | [`fect`](https://github.com/xuyiqing/fect) `method="mc"` | provenance; exact family |
| `InteractiveFixedEffects` | [`fect`](https://github.com/xuyiqing/fect) `method="ife"`, [`gsynth`](https://github.com/xuyiqing/gsynth) | provenance; exact family |
| `BalancingWeights` | [`ebal`](https://github.com/apoorvalal/ebal), [`WeightIt`](https://github.com/cran/WeightIt), [`CBPS`](https://github.com/cran/CBPS) | provenance; objectives differ |
| `MEstimator` | [`geex`](https://github.com/bsaul/geex) | provenance; arbitrary callback API |
| `GMM` | [`gmm`](https://github.com/cran/gmm), [statsmodels](https://github.com/statsmodels/statsmodels) | provenance; arbitrary moment API |
| `EPLM` | [DoubleML PLR](https://github.com/DoubleML/doubleml-for-py) | nearest only; estimand/orthogonalization differ |
| `AverageDerivative` | [`np`](https://github.com/cran/np) gradient routines | nearest only |
| `PartiallyLinearDML` | [DoubleML PLR](https://github.com/DoubleML/doubleml-for-py) | runnable |
| `AIPW` | [DoubleML IRM](https://github.com/DoubleML/doubleml-for-py), [EconML DRLearner](https://github.com/py-why/EconML) | DoubleML runnable |
| `DynamicCovariateBalance` | no exact public implementation found | native-only |
| `ParallelTrendsSNMM` | [`gesttools`](https://github.com/danieltompsett/gesttools) | nearest only; not parallel-trends SNMM |
| `RegressionBlip` | [`DTRreg`](https://github.com/cran/DTRreg), `gesttools` | nearest only |

Searches were performed with GitHub code and repository search. In particular,
no exact public code match was found for the Shahn et al. parallel-trends SNMM,
the Viviano–Bradic dynamic covariate-balancing estimator, or Crabbymetrics'
abundance-based constrained least squares. Those absences are recorded rather
than filled by misleading stand-ins.

## Dimension conventions

- Tabular, IV, survival, moment, balancing, and semiparametric estimators use
  `n` rows and `k` covariates/instruments.
- Panel estimators use `n` periods and `k` units/donors. This is deliberately
  explicit because they do not have an ordinary cross-sectional covariate count.
- Dynamic estimators use approximately `n / 4` units, four decision periods,
  and `k` history covariates.
- Synthetic-control `k` is the donor count and `n` is the pre-treatment length.

These conventions make the stress dimension transparent without pretending that
all 30 estimators consume the same mathematical object.
