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

The automatic cap is the smaller of 40% of physical RAM and currently available
RAM minus a 4 GiB system reserve. Override it only deliberately with
`--memory-gib`. A `10_000_000 x 100` `float64` matrix is 8 GB before copies or
solver workspaces, so large cells are expected to be skipped on ordinary hosts.

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
