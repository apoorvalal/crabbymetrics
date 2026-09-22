# Yiqing Xu panel lectures: empirical replication audit

This directory records **attempted replications**, not a claim that every result
in the six decks has been reproduced. It exercises unmodified CrabbyMetrics
0.9.0 and records independent R references, statistical conventions, and missing
estimator/workflow capabilities. See [SPEC.md](SPEC.md) for proposed changes and
[coverage.csv](coverage.csv) for the empirical-family inventory.

## Source revisions and scope

- Lectures: `xuyiqing/panel-lectures`, commit
  `891412f174fc0283677de84d612622ff602dd1d3`.
- CrabbyMetrics baseline: `0b56835`, version 0.9.0.
- Reanalysis archive: DOI `10.7910/DVN/9RJFZF`, version 1.0, file 11026794.
  The 581,990,430-byte archive is verified against Dataverse's MD5 and its SHA256
  is recorded in `results/archive-provenance.json`.
- Ten initial data sources are checksum-pinned in `acquire.py`. The archive adds
  raw inputs and study scripts for the 49 reanalyses. Inputs are downloaded to
  ignored `.cache/`; no third-party dataset is vendored in this branch.

The initial battery comprises 59 regression specifications and 102 reported
coefficients. A second battery runs 47 of the 49 archive datasets: 45 use the
source's point-regression specification, two are explicitly labeled baselines
without the source's unit-specific trends. Hall–Yoder and Sanford are not loaded:
the archive metadata reports approximately 89m and 158m observations. This is a
conservative memory preflight on a 16-GiB machine, **not** a measured estimator
failure. The 47 runs are not 47 replications of every method, diagnostic, or CI.

Additional tests cover five explicit-dummy imputation fits, outcome-only SCM,
synthetic DID, famine AIPW, and actual missing/reversing treatment panels. R
re-estimates the famine estimators, Bacon decomposition, augmented SCM,
canonical predictor-SCM/placebos, and selected FEct/IFEct references. Those R-only
results are labeled as such, not counted as CrabbyMetrics implementations.

## Reproduce

Use a working CrabbyMetrics development environment (Python 3.12, NumPy,
pandas, SciPy, Matplotlib), R, and the reference packages recorded in
`results/reference-packages.csv`. `install_references.R` installs references
into the task-local cache, never the global R library. GitHub packages are
pinned to their recorded source SHA; CRAN packages to their recorded version.

```bash
gh repo clone xuyiqing/panel-lectures /path/to/panel-lectures
git -C /path/to/panel-lectures checkout 891412f174fc0283677de84d612622ff602dd1d3
cd /path/to/crabbymetrics
python -m maturin develop --release
cd replications/yqx_panel
Rscript install_references.R .
python -B acquire.py --lectures /path/to/panel-lectures
python -B run_linear.py
Rscript reference_linear.R .
Rscript reference_panels.R . divorce
Rscript reference_panels.R . fdid
Rscript reference_panels.R . smoking
Rscript reference_panels.R . fect
python -B run_panels.py
python -B fdid_variance.py
python -B acquire_archive.py
Rscript export_archive.R .
python -B run_archive.py
Rscript reference_schafer.R .
python -B diagnose_schafer.py
Rscript reference_modern.R .
python -B run_panels.py --modern-only
Rscript reference_synth.R .
python -B summarize.py
```

Set `OPENBLAS_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1` for comparable
single-threaded runs. The archive download/extraction and reference libraries
use roughly 1.6 GiB. The existing project environment and Rust build are extra.
The scripts log case-level errors and preserve partial results; inspect those
logs and `verification.json`, rather than treating process completion as proof
of complete coverage. The R reference runs do not execute the archive's
unbounded 30-worker/1,000-bootstrap pipelines.

Render `docs/ablations/yqx-panel-lectures.qmd` after the numerical runs. Its
executed code audits the recorded comparisons and runs a small public-API SCM
example. Rendering is **not** a rerun of all reference packages.

## Reading the results correctly

1. `linear-comparison.csv`: CM versus the *identical exported design* in fixest.
   The source-formula divorce check is independent of that design exporter.
2. `archive-comparison.csv`: fresh CM versus fresh R and, separately, the
   archive's saved aggregate table. Matching the first does not imply matching
   the second. Inspect `scope` and `archive-specifications.json`.
3. `panels-crabby.csv`: native calls, explicit-dummy compositions, and a SciPy
   optimizer audit; the `status` field identifies each.
4. `*-reference.csv`, `augsynth-paths.csv`, `synth-canonical-*.csv`: R references.
   The canonical Synth recipe is reconstructed from ADH conventions; the
   lecture's original `ca1`–`ca8` script is not distributed.
5. `reanalysis-published-summary.csv`: extracted saved estimates, **not newly
   estimated outcomes**. Its ratio mean 1.02196 and median 1.00633 reconstruct
   the archive's figure; the median rounds to 1.01, not the lecture text's 1.00.
6. `capability-probes.json`: genuine API rejections on actual data, not toy
   assertions of support. No missing outcomes are filled with zero and no
   treatment reversals are silently converted into absorbing adoption.

### Important discrepancies

- CrabbyMetrics and R's small-sample/absorbed-FE corrections differ by default.
  The fatalities TWFE SE is 0.385787 in CM, 0.357078 in default fixest, and
  0.350150 in the lecture's plm HC1 convention. The coefficient is the same.
- The standalone SCM optimizer produces pre-RMSE 2.399913 versus the
  same-objective R/SciPy optimum 1.656400, at iteration budgets 50, 500 and
  5,000, while reporting `converged=True`. This is a numerical issue, not a
  statistical-specification difference.
- Schäfer's three-way-FE comparison differs by 2.06e-5 at fixest's default
  tolerance. Tightening fixest to 1e-10 gives agreement within 2e-10; permuting
  CM's FE columns leaves its estimate unchanged to about 1e-12. Do not label
  the initial difference a CrabbyMetrics bug.
- The archive's `Eckhouse` and `Esberg` saved-summary labels/values do not align
  with fresh fits of the correspondingly named shipped data and scripts.
  Preserve and expose the discrepancy; do not silently swap rows.
- The Brazil archive summary uses the trimmed `chris_sub` result. The full
  sample is a different estimand/sample: FEct is -0.119774 versus +0.044482 on
  the documented trimmed sample. The source-based baseline run is full sample.
- Coethnic-donation references use the two other-candidate covariates from the
  archive and yield FEct 0.127749. The current data retain 118 treated cells;
  the lecture mentions 74. Exact figure-level sample provenance remains open.
- Famine `fdid` OLS2 adds a superpopulation correction to HC3. Its source
  comment says HC0, but `car::hccm(lm)` defaults to HC3. Composing the actual
  formula around CM reproduces its SE 0.812714; ordinary HC1 alone does not.
- CM's ridge/logit AIPW and fdid's GRF AIPW are different nuisance-learning
  pipelines. Their estimates (0.264 versus -2.370 here) are not a parity test.

## License / provenance

CrabbyMetrics changes are original audit code and documentation. Lectures retain
their CC BY / code-license distinctions; third-party figures retain their own
rights. Do not redistribute the lecture figures or archive microdata as part
of a release. Acquisition scripts retain URLs and checksums. Source snippets
used to prepare archive designs are derived locally from the verified archive.
