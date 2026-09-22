# Lecture-motivated CrabbyMetrics specification

Status: **partially implemented on the replication branch**. Audit baseline 0.9.0 / `0b56835`.

Implemented in the follow-up: standalone SCM active-set QP and optimality
diagnostics; FE absorption controls, compact IDs, FE-only fits and identified
one/two-way level prediction; explicit FE/cluster SSC policies in summaries and
Wald tests; native `FEImputation` with observed/treatment masks, covariates,
weights, support checks and unit/time aggregation. Counterfactual nuisance
identification is checked on the untreated sample. R-generated offline fixtures
and optional actual lecture tests cover these paths; `results/fixes/` records
the post-fix empirical replay. Original audit outputs below remain historical.

Not implemented: imputation inference, causal IFE/MC, varying slopes, multiway
clustering, heterogeneous-DID estimators, joint event-study diagnostics, augmented
SCM and FDID orchestration. The provisional broad ImputationEstimator contract
below remains a roadmap; the shipped class is deliberately named FEImputation.
Evidence lives beside this file. Priorities distinguish a confirmed numerical
defect from missing interfaces, statistical conventions, and missing inputs.
Rust should own numerical kernels; public Python APIs should remain NumPy-first.
No pandas/formula/R dependency is proposed for the installed package.

## P0 — Repair standalone synthetic-control optimization

**Evidence:** Proposition 99 outcome-only SCM returns ATT -18.998496 and
pre-RMSE 2.399913; R augsynth and independent SciPy SLSQP agree on ATT -19.513630
and RMSE 1.656400. Budgets 50/500/5000 return the same suboptimal solution.
`SyntheticControl.summary()` unconditionally reports convergence.

**Change:** route `SyntheticControl` through the simplex quadratic-programming
solver already used by newer synthetic/augmented balancing paths, or implement
an equivalently verified constrained solver. Preserve exact boundary zeros.
Return objective, feasibility error, projected-gradient/KKT residual, iteration
count, stop reason and truthful convergence. Do not report success solely because
an optimizer returned a parameter vector. Surface numerical/identification
failures instead of silently accepting a flat softmax-gradient solution.

**Acceptance:** on checksum-pinned Prop99, pre-RMSE within 1e-6 of 1.656400207 and
ATT within 1e-4 of -19.5136299, simplex feasibility <1e-10, correct nonconvergence
with deliberately tiny budgets, donor permutation invariance. Add a small
original boundary-optimum fixture, not just a regression of current behavior.
Keep SDID and AugmentedBalancing parity checks intact.

## P0 — A treatment-aware panel contract and counterfactual estimator

**Evidence:** actual Grumbach–Sahn data are incomplete and contain on/off
treatment. MC and SDID reject these; `InteractiveFixedEffects.fit(Y)` accepts
only a fully observed outcome matrix and is not an IFEct estimator. Matrix
orientation also differs: IFE is T×N; MC/SDID are N×T.

**Proposed contract:** standardize new APIs on N×T, with separate `observed`
and `treatment` masks, optional covariates N×T×K, unit/time IDs or an external
long-to-panel adapter, sample weights, and an explicit training mask. Never
interpret zero as missing. Keep absorbing-only estimator restrictions where
mathematically required; a broader container must not silently broaden SDID's
estimand. Report duplicate cells, irregular time spacing, treatment reversals,
unsupported cells, never/always-treated units and sample exclusions.

Add an `ImputationEstimator(method='fe'|'ife'|'mc', ...)` (name provisional):
fit outcome models using untreated observed cells only; store nuisance levels;
predict treated counterfactuals; aggregate by observed treated cells, units,
cohorts, calendar time and event/spell time with explicit weights. Reject or
flag unidentified predictions (no untreated support / disconnected FE graph).
Preserve treated outcomes exclusively for effect evaluation. Never market the
current full-Y low-rank reconstruction as a causal counterfactual.

**Acceptance:** reproduce R FEct point estimates and surfaces on archived
Grumbach–Sahn with other-candidate covariates, Bischof, and the small turnout
panel. Current explicit-dummy CM estimates already establish the FE reference.
Verify no counterfactual changes if treated outcomes are perturbed, but ATT
changes appropriately. Include missing cells, reversals, unit exits and unequal
event-time support. IFE must have rank selection and convergence metadata;
turnout rank-2 R IFEct ATT 5.005401 is a fixed-rank reference, not a CV result.

## P1 — FE model controls, level prediction and covariance conventions

Expose absorption tolerance/iteration limits and residualization diagnostics;
support zero-covariate FE-only fits, fitted unit/time effects and level prediction
with unseen-level / connected-component checks. Implement varying slopes without
dense dummy expansion. Provide a documented df/SSC policy (including FE nested
in clusters and rank accounting); do not change the current default silently.
Multiway clustering should accept N×C cluster IDs; retain 1-D compatibility.

**Acceptance:** Swiss unit-linear and quadratic trends agree with source fixest
coefficients and exact-rank inference. Current trend SEs differ slightly even
after the generic all-FE correction because varying-slope rank conventions
differ. CK unbalanced levels DID 2.753606 must remain distinct from paired-change
DID 2.75. Reproduce the three fatalities SE conventions when explicitly selected.
Schäfer's 1.1m-row individual/year/age case must agree with tightly converged R,
not be forced to match R's looser default. Additional FE cases include
Fouirnaies–Hall's candidate-year effects and Jiang's province-year effects.

## P1 — Heterogeneity-robust DID and explicit event-study design

Provide common cohort/time metadata plus distinct Sun–Abraham and group-time ATT
estimators; distinguish never-treated and not-yet-treated controls, anticipation,
base period, balanced versus available-pair samples, and aggregation targets.
Stacked DID should expose stack weights and clustering instead of treating a
duplicated-control regression as automatically the target ATT.

**Acceptance:** Bischof's source six-method comparison; original decade-FE versus
year-FE baseline; Brazil full versus author-trimmed sample. Match independently
generated R cohort/event coefficients, full covariance, and aggregation weights.
Refuse these absorbing-adoption designs on treatment reversals. DID-multiple or
PanelMatch is a separate later estimator, not an alias for current SDID.

Event-study helpers must preserve endpoint bins in estimation even if hidden in
plots. The divorce source includes both -9 and +16 bins and uses -1 as reference;
dropping -9 changes the entire fitted curve. Record event-time conventions and
calendar spacing (election cycle versus year), not just integer positions.

## P1 — Diagnostics and inference, not only point estimates

Retain influence functions or joint bootstrap draws for all event/cohort effects.
Add cluster bootstrap with stable seeds/resampling units, joint pretrend tests,
held-out-preperiod placebo tests, treatment-exit carryover tests, and equivalence
tests with explicit margins. Distinguish pointwise from simultaneous bands.
Add Bacon-weight diagnostics for eligible binary absorbing designs.

**Acceptance:** divorce decomposition sums to -3.048894695 with weights summing
to one; reference groups and comparisons are explicit. Use the modern lecture
examples for pretrend/placebo/carryover diagnostics. Do not certify parallel
trends from non-rejection, especially on the small coethnic-treatment sample.
HonestDiD-style sensitivity needs the *joint event-study covariance* and a stated
restriction/target; it cannot be synthesized from independent error bars.
Full resampling parity remains unimplemented in this audit.

## P1 — Synthetic-panel workflow beyond a weight vector

Separate outcome-only SCM, predictor-V SCM, augmented SCM and SDID. Add predictor
aggregation windows / weights, or accept explicit predictor matrices and a
documented V vector. For augmented SCM, support ridge augmentation with training-
only tuning and weight extrapolation diagnostics. Include pre-fit RMSPE and
properly defined in-space placebo statistics; distinguish a placebo exercise
from calibrated confidence intervals. SDID's scalar ATT is a double-weighted
contrast, not necessarily the mean of its currently displayed gap curve.

**Acceptance:** the lecture's three augsynth fits on Prop99 (ATT -19.513629,
-15.952567, -12.710049 for current references), canonical predictor-SCM balance,
and documented placebo-ratio filters. Native SDID is -15.605398 versus R
-15.603828: record solver/objective tolerances and compare weights before
declaring exact parity. The independent 100-draw placebo SEs use different RNGs;
equal numeric seeds alone do not imply identical resampling draws.

## P2 — Factorial DID and nuisance-learning contracts

Add a small wrapper for baseline/event/post contrasts with explicit moderator
groups and target population (all, G=1, G=0). Support user-supplied cross-fitted
nuisance predictions and fold IDs, so the score can be validated independently
of ridge/logit versus causal-forest learners. Expose overlap and influence-score
diagnostics; document required identification assumptions for effect modification
versus causal moderation. Entropy balancing in current fdid is G=1-only and
must not be labeled the same all-population estimand.

**Acceptance:** 921-county famine DID/OLS1/OLS2 event and dynamic point estimates;
OLS2's actual HC3 + population variance correction (SE 0.812714147). The local
composition already matches that formula. Match external-score AIPW before
attempting a GRF clone. Preserve multivalued counts 412/254/255 and each pair's
target population. Seed-sensitive forest estimates are not exact slide pixels.

## P2 — Curricular completeness and scalable reproducibility

Swamy–Arora random effects is missing (fatalities -0.0520158 in R); add only if
it fits the package scope, rather than substituting pooled OLS. Dynamic-panel
IV/GMM and treatment-history methods need their own assumptions and tests.

Use fixture manifests with input hashes, source formulas, sample counts,
package/solver versions, variance conventions and explicit statuses. Scaling
work should preflight both observations and FE levels and bound memory/time.
For Hall–Yoder/Sanford, rerun on suitable hardware or prove an exact weighted
aggregation before loading; sampling is not an exact replication. The current
archive's saved-summary discrepancies and trimmed-sample choices must remain
visible. Re-drawing a published table is not re-estimating its entries.

## Suggested implementation sequence

1. Fix SCM numerical correctness and convergence metadata.
2. Add the panel mask/ID contract and efficient FE imputation with level prediction.
3. Add SSC policies, varying slopes, event-study metadata and joint inference.
4. Add group-time / interaction-weighted DID and their aggregation/diagnostics.
5. Add IFE/MC counterfactual fitting and training-only tuning, then augmented SCM.
6. Add FDID orchestration/external nuisances and the lower-priority curricular gaps.

Each stage should be a separately reviewable change with independent reference
fixtures. The initial audit commit changed no production estimator code. The implemented subset is listed at the top; remaining stages are proposals.
