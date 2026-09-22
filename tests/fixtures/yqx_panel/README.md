# Original R reference fixtures

Regenerate from the repository root with:

```sh
Rscript tests/fixtures/yqx_panel/generate.R
```

No downloaded data or contributed R package is used. `panel.csv` is a small
deterministic panel with reversals, missing observations, weights and covariates.
Base R `lm` estimates untreated-only counterfactuals; a full dummy-variable design
and cluster-summed scores independently produce the covariance references.
`scm.csv` is generated separately; enumerating every nonempty face of its
six-donor simplex gives the reference constrained optimum, including zero weights.

`test_yqx_panel_fixes.py` reads these fixtures without requiring R at test time.
Actual lecture datasets are not vendored here. Their optional tests and acquisition
instructions are in `replications/yqx_panel/`.
