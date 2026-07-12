#!/usr/bin/env python3
"""Generate first-pass class-centered Quarto reference page scaffolds.

Existing audited pages are skipped unless ``--force`` is supplied. The hand-written
objective, inference, performance, and implementation sections are authoritative.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = ROOT / "reference"
REF.mkdir(exist_ok=True)

COMMON_HEADER = """---
title: \"{title}\"
subtitle: \"{subtitle}\"
format:
  html:
    toc: true
    toc-depth: 3
    number-sections: true
    embed-resources: true
    page-layout: full
    code-fold: false
execute:
  echo: true
  warning: false
  message: false
jupyter: python3
---

```{{python}}
from _api_doc_utils import *
```
"""

def method_table(class_name: str) -> str:
    return f"""```{{python}}
cls = cm.{class_name}
display(HTML(html_table(["Public method"], public_methods(cls))))
```
"""

def format_python_snippet(snippet: str) -> str:
    snippet = snippet.strip()
    try:
        tree = ast.parse(snippet)
    except SyntaxError:
        return "\n".join(part.strip() for part in snippet.replace(";", "\n").splitlines() if part.strip())
    return "\n".join(ast.unparse(stmt) for stmt in tree.body)


def summary_table(class_name: str, example_expr: str) -> str:
    formatted = format_python_snippet(example_expr)
    return f"""```{{python}}
{formatted}
summary = model.summary()
display(HTML(html_table(["summary() key", "shape"], summary_shape_rows(summary))))
```
"""

PAGES = {
"OLS": dict(
 group="Regression", subtitle="Ordinary least squares with robust covariance options",
 math=r"""`OLS` estimates the linear projection

$$
y_i = \alpha + x_i'\beta + u_i
$$

by least squares. The class stores the fitted intercept and coefficient vector and exposes the common linear-model covariance surface: classical, HC1, Newey-West, and cluster-robust standard errors.""",
 api="""Use `fit(x, y)` for the standard estimator, `fit_weighted(x, y, sample_weight)` for weighted least squares, and `fit_sketch(x, y, sketch_size, seed=None)` when a randomized row sketch is acceptable for a large tall design. `predict(x)` returns fitted values. `bootstrap(B, seed=None)` returns bootstrap draws with the intercept in the first column.""",
 example="""rng = np.random.default_rng(1)
x = rng.normal(size=(200, 3))
y = 0.5 + x @ np.array([1.0, -0.7, 0.25]) + rng.normal(scale=0.3, size=200)
model = cm.OLS()
model.fit(x, y)
clusters = np.repeat(np.arange(20, dtype=np.int64), 10)
print(model.summary(vcov="hc1")["coef"])
print(model.summary(vcov="cluster", clusters=clusters)["coef_se"])
print(model.predict(x[:3]))""",
 setup="""rng = np.random.default_rng(101)
x = rng.normal(size=(80, 3))
y = 0.5 + x @ np.array([1.0, -0.7, 0.25]) + rng.normal(scale=0.3, size=80)
model = cm.OLS(); model.fit(x, y)"""),
"ABCOLS": dict(group="Regression", subtitle="Abundance-based constrained OLS for categorical modifiers",
 math=r"""`ABCOLS` is an OLS reparameterization for categorical main effects and categorical modifiers using abundance-based constraints / weighted effect coding.

Instead of treating one level as the omitted baseline, it estimates an overcomplete dummy/interactions design under linear constraints that force categorical effects to average to zero under empirical level frequencies. This makes the intercept and continuous main slopes sample-abundance-weighted averages rather than reference-category coefficients.""",
 api="""Call `fit(y, x, categories, cont_cat_interactions=None, cat_cat_interactions=None, center_continuous=True)`. Categorical inputs are zero-based dense `uint32` codes. `predict(x, categories)` returns fitted means for new rows under the same coding scheme. `summary()` reports constrained coefficients, standard errors, column names, constraint names, residual variance, residual degrees of freedom, rank, and a maximum-constraint-violation diagnostic.""",
 example="""rng = np.random.default_rng(2026)
group = np.repeat(np.array([0, 1, 2], dtype=np.uint32), [36, 54, 30])
sex = np.tile(np.array([0, 1], dtype=np.uint32), len(group) // 2)
categories = np.column_stack([group, sex]).astype(np.uint32)
x_raw = rng.normal(size=len(group))
x = x_raw[:, None]
x_centered = x_raw - x_raw.mean()
y = (
    1.25
    + 1.10 * x_centered
    + np.array([-0.75, 0.15, 0.95])[group]
    + np.array([0.45, -0.20, 0.10])[group] * x_centered
    + 0.35 * sex
    + rng.normal(scale=0.08, size=len(group))
)
model = cm.ABCOLS()
model.fit(y, x, categories, cont_cat_interactions=[(0, 0)], cat_cat_interactions=[(0, 1)])
print(model.summary()["column_names"])
print(model.summary()["coef"][:6])
print(model.predict(x[:3], categories[:3]))""",
 setup="""rng = np.random.default_rng(2126)
group = np.repeat(np.array([0, 1, 2], dtype=np.uint32), [24, 30, 18])
sex = np.tile(np.array([0, 1], dtype=np.uint32), len(group) // 2)
categories = np.column_stack([group, sex]).astype(np.uint32)
x_raw = rng.normal(size=len(group))
x = x_raw[:, None]
x_centered = x_raw - x_raw.mean()
y = (
    0.8
    + 0.9 * x_centered
    + np.array([-0.4, 0.1, 0.6])[group]
    + np.array([0.2, -0.1, 0.05])[group] * x_centered
    + 0.25 * sex
    + rng.normal(scale=0.1, size=len(group))
)
model = cm.ABCOLS()
model.fit(y, x, categories, cont_cat_interactions=[(0, 0)], cat_cat_interactions=[(0, 1)])"""),
"Ridge": dict(group="Regression", subtitle="L2-regularized least squares with optional CV",
 math=r"""`Ridge` solves

$$
\min_{\alpha,\beta} \sum_i (y_i - \alpha - x_i'\beta)^2 + \lambda \|\beta\|_2^2.
$$

A scalar penalty gives one ridge fit. A penalty grid with `cv` selects a penalty by cross-validation, stores the coefficient path, and refits on the full sample.""",
 api="""The main methods mirror `OLS`: `fit`, `fit_weighted`, `predict`, `summary`, and `bootstrap`. `summary()` includes the selected penalty and, for grid fits, cross-validation diagnostics and coefficient paths.""",
 example="""rng = np.random.default_rng(2)
x = rng.normal(size=(240, 5))
y = 0.3 + x @ np.array([1.0, -0.8, 0.0, 0.25, 0.1]) + rng.normal(scale=0.6, size=240)
model = cm.Ridge(penalty=np.array([0.0, 0.05, 0.2, 1.0]), cv=4)
model.fit(x, y)
print(model.summary()["penalty"])
print(model.summary()["coef"])
print(model.predict(x[:3]))""",
 setup="""rng = np.random.default_rng(102)
x = rng.normal(size=(90, 4)); y = 0.3 + x @ np.array([1, -.5, .2, 0]) + rng.normal(size=90)
model = cm.Ridge(penalty=np.array([0.0, 0.1, 1.0]), cv=3); model.fit(x, y)"""),
"FixedEffectsOLS": dict(group="Regression", subtitle="Within-estimator for high-dimensional fixed effects",
 math=r"""`FixedEffectsOLS` partials out one or more categorical fixed effects, then runs least squares on residualized variables:

$$
M_F y = M_F X\beta + M_F u.
$$

The fixed-effect matrix `fe` is a 2D `uint32` array of zero-based category codes, one column per fixed-effect dimension.""",
 api="""Call `fit(x, fe, y)` or `fit_weighted(x, fe, y, sample_weight)`. There is no `predict()` because the class is estimation-first and does not materialize fixed-effect coefficients. `summary()` supports the same covariance options as the other linear estimators.""",
 example="""rng = np.random.default_rng(3)
n = 300
x = rng.normal(size=(n, 2))
worker = rng.integers(0, 30, size=n, dtype=np.uint32)
firm = rng.integers(0, 12, size=n, dtype=np.uint32)
fe = np.column_stack([worker, firm]).astype(np.uint32)
y = x @ np.array([0.8, -0.5]) + rng.normal(size=30)[worker] + rng.normal(size=12)[firm] + rng.normal(scale=0.2, size=n)
model = cm.FixedEffectsOLS(); model.fit(x, fe, y)
print(model.summary(vcov="cluster", clusters=worker.astype(np.int64)))""",
 setup="""rng = np.random.default_rng(103)
n=100; x=rng.normal(size=(n,2)); g=rng.integers(0,10,size=n,dtype=np.uint32); h=rng.integers(0,5,size=n,dtype=np.uint32)
fe=np.column_stack([g,h]).astype(np.uint32); y=x@np.array([.8,-.5])+rng.normal(size=10)[g]+rng.normal(size=n)*.2
model=cm.FixedEffectsOLS(); model.fit(x, fe, y)"""),
"ElasticNet": dict(group="Regression", subtitle="Coordinate-descent elastic net regression",
 math=r"""`ElasticNet` estimates a penalized linear model with a convex combination of L1 and L2 penalties. The delegated implementation centers the outcome but not the feature columns:

$$
\begin{aligned}
\min_{\beta}\quad
&\frac{1}{2n}\|y-\bar y\mathbf1-X\beta\|_2^2 \\
&+\lambda\rho\|\beta\|_1
+\frac{\lambda(1-\rho)}{2}\|\beta\|_2^2.
\end{aligned}
$$

Prediction is $\bar y+X\hat\beta$. The wrapper recomputes the final duality gap and rejects budget-exhausted, nonconverged fits.""",
 api="""Use `ElasticNet(penalty, l1_ratio, tolerance, max_iterations)`, then `fit(x, y)`, `predict(x)`, `summary()`, and optionally `bootstrap(B, seed=None)`. The summary reports point estimates, convergence diagnostics, and the duality gap; analytic inference is unavailable.""",
 example="""rng = np.random.default_rng(4)
x = rng.normal(size=(180, 8))
y = 0.4 + x[:, :3] @ np.array([1.0, -0.8, 0.5]) + rng.normal(scale=0.5, size=180)
model = cm.ElasticNet(penalty=0.05, l1_ratio=0.7)
model.fit(x, y)
print(model.summary()["coef"])
print(model.predict(x[:3]))""",
 setup="""rng=np.random.default_rng(104); x=rng.normal(size=(90,5)); y=.4+x[:,:2]@np.array([1,-.8])+rng.normal(size=90)*.3
model=cm.ElasticNet(penalty=.05,l1_ratio=.7); model.fit(x,y)"""),
"Logit": dict(group="Regression", subtitle="Binary logistic regression",
 math=r"""`Logit` models a binary outcome through

$$
\Pr(Y_i=1\mid X_i=x_i)=\Lambda(\alpha+x_i'\beta),
$$

 with optional L2 regularization controlled by `alpha`. The native L-BFGS fit uses stable softplus evaluation and rejects nonconverged results. `predict(x)` returns probabilities for label 1.""",
 api="""Fit with 0/1 integer labels using `fit(x, y_int32)`. `summary()` returns fit diagnostics and intercept/coefficient estimates. Fisher-information inference is available only at `alpha=0`; `bootstrap()` resamples observations and refits the classifier.""",
 example="""rng=np.random.default_rng(5)
x=rng.normal(size=(220,3)); eta=-0.2+x@np.array([.7,-.4,.9]); p=1/(1+np.exp(-eta))
y=rng.binomial(1,p,size=220).astype(np.int32)
model=cm.Logit(max_iterations=200); model.fit(x,y)
print(model.summary()["coef"])
print(model.predict(x[:5]))""",
 setup="""rng=np.random.default_rng(105); x=rng.normal(size=(90,3)); p=1/(1+np.exp(-(x@np.array([.7,-.4,.9])))); y=rng.binomial(1,p,size=90).astype(np.int32)
model=cm.Logit(max_iterations=200); model.fit(x,y)"""),
"MultinomialLogit": dict(group="Regression", subtitle="Multiclass logistic regression",
 math=r"""`MultinomialLogit` generalizes binary logit to $K$ classes with softmax probabilities:

$$
\Pr(Y_i=k\mid X_i=x_i)=\frac{\exp(\alpha_k+x_i'\beta_k)}{\sum_\ell \exp(\alpha_\ell+x_i'\beta_\ell)}.
$$

 The native L-BFGS fit uses stable log-sum-exp evaluation and rejects nonconverged results. The summary reports identifiable class-versus-last-class contrasts.""",
 api="""Use integer class labels in `fit(x, y_int32)`. `predict(x)` returns class probabilities, `predict_lin(x)` returns logits, and `predict_label(x)` returns labels. Fisher-information inference is available only at `alpha=0`.""",
 example="""rng=np.random.default_rng(6)
x=rng.normal(size=(240,2)); logits=x@np.array([[.6,-.3],[-.4,.5],[.2,.2]]).T + np.array([.1,-.2,0.])
p=np.exp(logits-logits.max(axis=1,keepdims=True)); p=p/p.sum(axis=1,keepdims=True)
y=np.array([rng.choice(3,p=row) for row in p], dtype=np.int32)
model=cm.MultinomialLogit(max_iterations=200); model.fit(x,y)
print(model.summary()["coef"])
print(model.predict(x[:5]))""",
 setup="""rng=np.random.default_rng(106); x=rng.normal(size=(100,2)); logits=x@np.array([[.6,-.3],[-.4,.5],[.2,.2]]).T; p=np.exp(logits-logits.max(1,keepdims=True)); p=p/p.sum(1,keepdims=True); y=np.array([rng.choice(3,p=row) for row in p],dtype=np.int32)
model=cm.MultinomialLogit(max_iterations=200); model.fit(x,y)"""),
"Poisson": dict(group="Regression", subtitle="Poisson GLM for count outcomes",
 math=r"""`Poisson` fits

$$
\mathbb E[Y_i\mid X_i=x_i]=\exp(\alpha+x_i'\beta).
$$

The `alpha` constructor argument is an L2 penalty, not the intercept. `summary(vcov='vanilla')` reports Fisher-information standard errors; `summary(vcov='sandwich')` reports robust QMLE-style standard errors.""",
 api="""Use `fit(x, y)` with nonnegative count-like outcomes. `predict(x)` returns fitted conditional means. `summary()` reports optimizer diagnostics and inference is available only at `alpha=0`. The class also supports `bootstrap(B, seed=None)`.""",
 example="""rng=np.random.default_rng(7)
x=rng.normal(size=(250,2)); mu=np.exp(.2+x@np.array([.4,-.25])); y=rng.poisson(mu).astype(float)
model=cm.Poisson(max_iterations=200, tolerance=1e-8); model.fit(x,y)
print(model.summary(vcov="vanilla")["coef"])
print(model.summary(vcov="sandwich")["coef_se"])
print(model.predict(x[:3]))""",
 setup="""rng=np.random.default_rng(107); x=rng.normal(size=(100,2)); y=rng.poisson(np.exp(.2+x@np.array([.4,-.25]))).astype(float)
model=cm.Poisson(max_iterations=200); model.fit(x,y)"""),
"TwoSLS": dict(group="Causal inference", subtitle="Closed-form linear IV / two-stage least squares",
 math=r"""`TwoSLS` estimates linear instrumental-variables models. With endogenous regressors $X_e$, exogenous controls $X_c$, instruments $Z$, and outcome $y$, it runs the projection of the second-stage design on the instrument span and then estimates

$$
y = \alpha + X_e\beta_e + X_c\beta_c + u.
$$

It supports multiple endogenous regressors and excluded instruments.""",
 api="""Call `fit(x_endog, x_exog, z, y)`. The `predict(x)` method expects a second-stage design matrix matching the fitted endogenous+exogenous columns. The robust linear covariance options match `OLS`: `vanilla`, `hc1`, `newey_west`, and `cluster`.""",
 example="""rng=np.random.default_rng(9); n=400
z=rng.normal(size=(n,2)); x_exog=rng.normal(size=(n,1)); v=rng.normal(size=(n,1)); u=.6*v[:,0]+rng.normal(scale=.4,size=n)
x_endog=z@np.array([[.8],[-.4]])+.2*x_exog+v; y=.5+1.2*x_endog[:,0]-.7*x_exog[:,0]+u
model=cm.TwoSLS(); model.fit(x_endog,x_exog,z,y)
print(model.summary()["coef"])
print(model.summary(vcov="newey_west", lags=3)["coef_se"])""",
 setup="""rng=np.random.default_rng(109); n=120; z=rng.normal(size=(n,2)); x_exog=rng.normal(size=(n,1)); v=rng.normal(size=(n,1)); x_endog=z@np.array([[.8],[-.4]])+.2*x_exog+v; y=.5+1.2*x_endog[:,0]-.7*x_exog[:,0]+.6*v[:,0]+rng.normal(size=n)*.4
model=cm.TwoSLS(); model.fit(x_endog,x_exog,z,y)"""),
"BalancingWeights": dict(group="Causal inference", subtitle="Calibration weights for covariate balance",
 math=r"""`BalancingWeights` chooses weights for a source sample so weighted source covariate means match a target sample:

$$
\sum_i w_i x_i / \sum_i w_i \approx \sum_j q_j x_j / \sum_j q_j.
$$

The objective can be quadratic or entropy-like, with optional lower/upper bounds and ridge stabilization.""",
 api="""Use `fit(covariates, target_covariates, baseline_weights=None, target_weights=None)`. `solver_converged` describes the scaled calibration solve; `success` additionally checks weight feasibility. `summary()` separates scaled solver diagnostics from original-unit balance diagnostics.""",
 example="""rng=np.random.default_rng(10)
x0=rng.normal(size=(180,3)); x1=rng.normal(loc=np.array([.4,-.2,.1]), size=(80,3))
model=cm.BalancingWeights(objective="quadratic", max_iterations=500)
model.fit(x0, x1)
print(model.summary()["mean_diff"])
print(model.summary()["effective_sample_size"])
print(model.get_weights()[:5])""",
 setup="""rng=np.random.default_rng(110); x0=rng.normal(size=(80,3)); x1=rng.normal(loc=np.array([.4,-.2,.1]), size=(40,3))
model=cm.BalancingWeights(); model.fit(x0,x1)"""),
"EPLM": dict(group="Causal inference", subtitle="Robins-Newey partially linear E-estimator",
 math=r"""`EPLM` targets a scalar treatment effect in a partially linear model. It combines an outcome equation with a working model for $E[D\mid W]$ and solves the resulting stacked moment system.

The intended estimand is the coefficient on the scalar treatment after accounting for controls $W$ without treating the nuisance regression as the object of interest.""",
 api="""Call `fit(y, d, w)` with scalar treatment `d` and 2D controls `w`. `summary(vcov=None, lags=None, clusters=None)` returns the coefficient, standard error, covariance matrix, and nuisance coefficients. There is no `predict()` method.""",
 example="""rng=np.random.default_rng(11)
w=rng.normal(size=(300,3)); d=.4+w@np.array([.6,-.2,.3])+rng.normal(size=300); y=1.1*d+w@np.array([.2,.1,-.2])+rng.normal(scale=.5,size=300)
model=cm.EPLM(); model.fit(y,d,w)
print(model.summary()["coef"])
print(model.summary()["se"])""",
 setup="""rng=np.random.default_rng(111); w=rng.normal(size=(120,3)); d=.4+w@np.array([.6,-.2,.3])+rng.normal(size=120); y=1.1*d+w@np.array([.2,.1,-.2])+rng.normal(size=120)*.5
model=cm.EPLM(); model.fit(y,d,w)"""),
"AverageDerivative": dict(group="Causal inference", subtitle="Average derivative estimator for continuous treatments",
 math=r"""`AverageDerivative` targets an average marginal effect of a scalar continuous treatment. The class exposes three related estimating equations through `method='ob'`, `'ipw'`, or `'dr'`.

The doubly robust option combines outcome-bridge and weighting components, while the other options expose the individual pieces.""",
 api="""Call `fit(y, d, w)`. The summary reports `method`, `coef`, `se`, and `vcov`. There is no `predict()` method because the object is a semiparametric target rather than a full conditional mean model.""",
 example="""rng=np.random.default_rng(12)
w=rng.normal(size=(320,2)); d=.2+w@np.array([.5,-.3])+rng.normal(scale=.7,size=320); y=.8*d+w@np.array([.2,-.1])+rng.normal(scale=.5,size=320)
model=cm.AverageDerivative(method="dr"); model.fit(y,d,w)
print(model.summary())""",
 setup="""rng=np.random.default_rng(112); w=rng.normal(size=(120,2)); d=.2+w@np.array([.5,-.3])+rng.normal(size=120)*.7; y=.8*d+w@np.array([.2,-.1])+rng.normal(size=120)*.5
model=cm.AverageDerivative(method='dr'); model.fit(y,d,w)"""),
"PartiallyLinearDML": dict(group="Causal inference", subtitle="Cross-fit partially linear Double ML",
 math=r"""`PartiallyLinearDML` estimates the treatment coefficient in

$$
y = \theta d + g(x) + u, \qquad d = m(x) + v,
$$

using cross-fitted ridge nuisance regressions. The final coefficient is estimated from the orthogonalized residual-on-residual score.""",
 api="""Use `PartiallyLinearDML(penalty=None, cv=5, n_folds=5, seed=42)`, then `fit(y, d, x)`. `summary()` reports the coefficient, robust standard error, covariance, and selected nuisance penalties by fold.""",
 example="""rng=np.random.default_rng(13)
x=rng.normal(size=(400,4)); d=.3+x@np.array([.5,-.4,.2,.1])+rng.normal(scale=.8,size=400); y=1.3*d+x@np.array([.4,-.2,.1,.3])+rng.normal(scale=.6,size=400)
model=cm.PartiallyLinearDML(penalty=np.logspace(-4,1,10), cv=3, n_folds=4, seed=1); model.fit(y,d,x)
print(model.summary()["coef"])
print(model.summary()["outcome_penalties"][:2])""",
 setup="""rng=np.random.default_rng(113); x=rng.normal(size=(160,4)); d=.3+x@np.array([.5,-.4,.2,.1])+rng.normal(size=160)*.8; y=1.3*d+x@np.array([.4,-.2,.1,.3])+rng.normal(size=160)*.6
model=cm.PartiallyLinearDML(penalty=np.logspace(-4,1,6),cv=3,n_folds=4,seed=1); model.fit(y,d,x)"""),
"AIPW": dict(group="Causal inference", subtitle="Cross-fit augmented inverse-probability weighting",
 math=r"""`AIPW` estimates a binary-treatment ATE by combining outcome regressions and a propensity model:

$$
\hat\tau = n^{-1}\sum_i \left[\hat\mu_1(x_i)-\hat\mu_0(x_i) + \frac{d_i(y_i-\hat\mu_1(x_i))}{\hat e(x_i)} - \frac{(1-d_i)(y_i-\hat\mu_0(x_i))}{1-\hat e(x_i)}\right].
$$

The nuisance functions are cross-fit ridge models.""",
 api="""Call `fit(y, d, x)` with binary treatment `d`. `summary()` reports `ate`, `se`, `vcov`, and selected penalties for the outcome and propensity nuisance models.""",
 example="""rng=np.random.default_rng(14)
x=rng.normal(size=(420,3)); pi=1/(1+np.exp(-(.1+x@np.array([.6,-.3,.2])))); d=rng.binomial(1,pi,size=420).astype(float); y=.5+x@np.array([.2,-.1,.3])+1.0*d+rng.normal(size=420)
model=cm.AIPW(penalty=np.logspace(-4,1,10), cv=3, n_folds=4, seed=2); model.fit(y,d,x)
print(model.summary()["ate"])
print(model.summary()["se"])""",
 setup="""rng=np.random.default_rng(114); x=rng.normal(size=(160,3)); pi=1/(1+np.exp(-(.1+x@np.array([.6,-.3,.2])))); d=rng.binomial(1,pi,size=160).astype(float); y=.5+x@np.array([.2,-.1,.3])+d+rng.normal(size=160)
model=cm.AIPW(penalty=np.logspace(-4,1,6),cv=3,n_folds=4,seed=2); model.fit(y,d,x)"""),
"SyntheticControl": dict(group="Causal inference", subtitle="Single-treated-unit donor weighting",
 math=r"""`SyntheticControl` fits nonnegative donor weights that sum to one, minimizing pre-treatment imbalance between the treated path and a convex combination of donor paths:

$$
\min_{w\ge 0,\;1'w=1}\|y_{\mathrm{treated,pre}} - Y_{\mathrm{donor,pre}}w\|_2^2.
$$

It is the lower-level single-path API; the panel estimators use the newer `fit(Y, W)` contract.""",
 api="""Call `fit(donors, treated)` where `donors` is `(n_periods, n_donors)` and `treated` is the treated pre-period vector. `predict(donors)` applies the learned weights to a donor matrix. `summary()` reports weights and pre-fit RMSE.""",
 example="""rng=np.random.default_rng(15)
donors=rng.normal(size=(40,4)); w_true=np.array([.45,.25,.2,.1]); treated=donors@w_true+rng.normal(scale=.02,size=40)
model=cm.SyntheticControl(max_iterations=500); model.fit(donors, treated)
print(model.summary()["weights"])
print(model.predict(donors[-3:]))""",
 setup="""rng=np.random.default_rng(115); donors=rng.normal(size=(30,4)); treated=donors@np.array([.45,.25,.2,.1])
model=cm.SyntheticControl(max_iterations=300); model.fit(donors,treated)"""),
"HorizontalPanelRidge": dict(group="Causal inference", subtitle="Horizontal ridge counterfactuals for panel treatment effects",
 math=r"""`HorizontalPanelRidge` implements a horizontal panel-prediction design. For each adoption cohort, never-treated donor outcomes at time $t$ become features for treated outcomes at time $t$ in the pre-period. Ridge then extrapolates counterfactual treated paths into the treated post-period.

The public panel contract is `fit(Y, W)`: balanced outcomes plus a same-shaped absorbing treatment matrix.""",
 api="""After `fit(y, w)`, `predict()` returns treated-unit counterfactuals, `treatment_effect()` returns observed-minus-counterfactual effects, and `summary()` returns ATT, event-study, group means, fitted coefficients, cohorts, and diagnostics.""",
 example="""rng=np.random.default_rng(16)
y=rng.normal(size=(10,14)); w=np.zeros_like(y); w[7:,9:]=1; y[7:,9:]+=1.0
model=cm.HorizontalPanelRidge(penalty=1.0); model.fit(y,w)
print(model.summary()["att"])
print(list(model.summary()["event_study"].items())[:3])""",
 setup="""rng=np.random.default_rng(116); y=rng.normal(size=(8,12)); w=np.zeros_like(y); w[6:,8:]=1; y[6:,8:]+=.8
model=cm.HorizontalPanelRidge(); model.fit(y,w)"""),
"SyntheticDID": dict(group="Causal inference", subtitle="Synthetic difference-in-differences for balanced panels",
 math=r"""`SyntheticDID` combines donor-unit weights and pre-period time weights. It estimates counterfactual treated outcomes by reweighting both units and periods, then reports ATT and event-time summaries under the common `fit(Y, W)` panel contract.""",
 api="""Use `SyntheticDID(zeta_omega=None, zeta_lambda=None, max_iterations=1000)`. `fit(y, w)` infers cohorts and donors. `predict()`, `treatment_effect()`, `summary()`, `vcov()`, and `se()` expose fitted counterfactuals and uncertainty helpers.""",
 example="""rng=np.random.default_rng(17)
y=rng.normal(size=(9,13)); w=np.zeros_like(y); w[6:,8:]=1; y[6:,8:]+=0.7
model=cm.SyntheticDID(max_iterations=500); model.fit(y,w)
print(model.summary()["att"])
print(model.treatment_effect().shape)""",
 setup="""rng=np.random.default_rng(117); y=rng.normal(size=(8,12)); w=np.zeros_like(y); w[6:,8:]=1; y[6:,8:]+=.7
model=cm.SyntheticDID(max_iterations=300); model.fit(y,w)"""),
"MatrixCompletion": dict(group="Causal inference", subtitle="Nuclear-norm panel counterfactual completion",
 math=r"""`MatrixCompletion` treats untreated cells as observed entries and treated cells as missing counterfactuals. It estimates a low-rank untreated-outcome surface, optionally with unit and time effects, using nuclear-norm style shrinkage.

The completed values in treated cells become counterfactual outcomes for ATT and event-study summaries.""",
 api="""Call `MatrixCompletion(...).fit(y, w)`. `predict()` returns completed/counterfactual values. `summary()` reports ATT, low-rank components, histories, and explicit convergence diagnostics; a budget-exhausted final iterate is retained with `converged=False`.""",
 example="""rng=np.random.default_rng(18)
load=rng.normal(size=(10,2)); fac=rng.normal(size=(2,14)); y=load@fac+rng.normal(scale=.1,size=(10,14)); w=np.zeros_like(y); w[7:,9:]=1; y[7:,9:]+=1
model=cm.MatrixCompletion(max_iterations=100, tolerance=1e-5); model.fit(y,w)
print(model.summary()["att"])
print(model.predict().shape)""",
 setup="""rng=np.random.default_rng(118); y=rng.normal(size=(8,10)); w=np.zeros_like(y); w[6:,7:]=1; y[6:,7:]+=.8
model=cm.MatrixCompletion(max_iterations=80,tolerance=1e-5); model.fit(y,w)"""),
"InteractiveFixedEffects": dict(group="Causal inference", subtitle="Factor-model panel counterfactual helper",
 math=r"""`InteractiveFixedEffects` estimates a low-rank factor structure in a balanced panel. It is closest to a lightweight `fect` helper: remove additive components according to `force`, estimate factors, and reconstruct fitted untreated outcomes.""",
 api="""Use `InteractiveFixedEffects(rank=0, force=3, ...)`, then `fit(y)`. `predict()` reconstructs the fitted panel. `summary()` reports low-rank pieces, additive effects, singular values, chosen rank, and diagnostics.""",
 example="""rng=np.random.default_rng(19)
y=rng.normal(size=(12,16)) + rng.normal(size=(12,1)) + rng.normal(size=(1,16))
model=cm.InteractiveFixedEffects(rank=2); model.fit(y)
print(model.summary()["rank"])
print(model.predict().shape)""",
 setup="""rng=np.random.default_rng(119); y=rng.normal(size=(10,12))+rng.normal(size=(10,1))+rng.normal(size=(1,12))
model=cm.InteractiveFixedEffects(rank=2); model.fit(y)"""),
"StaggeredPanelEventStudy": None,
"GMM": dict(group="Estimation interfaces", subtitle="Callback-driven generalized method of moments",
 math=r"""`GMM` solves moment restrictions of the form

$$
\mathbb E[g_i(\theta)] = 0.
$$

The user supplies a Python callback returning the per-observation moment matrix. In exactly identified cases the class can solve by Gauss-Newton; in overidentified cases it can use identity or two-step weighting and report sandwich covariance estimates.""",
 api="""Construct with `GMM(moment_fn, jacobian_fn=None, max_iterations=100, tolerance=1e-6, ridge=1e-8, fd_eps=1e-6)`. `fit(data, theta0, weighting='auto')` stores the fitted parameters. `fit_sketch(...)` projects the moment columns. `summary(vcov='sandwich', omega='iid', lags=None, clusters=None)` controls inference.""",
 example="""def moments(theta, data):
    resid = data["y"] - data["x"] * theta[0]
    return data["z"] * resid[:, None]

def jac(theta, data):
    return -(data["z"].T @ data["x"][:, None]) / data["x"].shape[0]

rng=np.random.default_rng(20); n=300; z=rng.normal(size=(n,3)); v=rng.normal(size=n); x=z@np.array([.9,.4,-.3])+v; y=1.2*x+.5*v+rng.normal(size=n)*.3
model=cm.GMM(moments, jacobian_fn=jac, max_iterations=200); model.fit({"x":x,"y":y,"z":z}, np.array([0.0]), weighting="identity")
print(model.summary()["coef"])
print(model.summary()["j_stat"])""",
 setup="""def moments(theta, data):
    resid=data['y']-data['x']*theta[0]
    return data['z']*resid[:,None]
def jac(theta,data):
    return -(data['z'].T@data['x'][:,None])/data['x'].shape[0]
rng=np.random.default_rng(120); n=120; z=rng.normal(size=(n,3)); v=rng.normal(size=n); x=z@np.array([.9,.4,-.3])+v; y=1.2*x+.5*v+rng.normal(size=n)*.3
model=cm.GMM(moments,jacobian_fn=jac,max_iterations=200); model.fit({'x':x,'y':y,'z':z},np.array([0.0]),weighting='identity')"""),
"MEstimator": dict(group="Estimation interfaces", subtitle="Low-level objective-plus-score M-estimation",
 math=r"""`MEstimator` is the lowest-level public estimation interface. It minimizes a user-supplied objective with gradient and uses a user-supplied per-observation score matrix for covariance estimation:

$$
\hat\theta = \arg\min_\theta Q_n(\theta), \qquad \widehat V = A^{-1} B A^{-T}/n.
$$

The bread is the numerical Jacobian of the mean score and the meat is the empirical score outer product. Nonconverged L-BFGS results are rejected.""",
 api="""Construct with `MEstimator(objective_fn, score_fn, max_iterations=100, tolerance=1e-6, derivative_step=1e-6)`. `objective_fn(theta, data)` must return `(objective, gradient)`. `score_fn(theta, data)` must return an `(n, p)` matrix. `summary()` includes common fit diagnostics.""",
 example="""def obj(theta, data):
    X, y = data["X"], data["y"]
    idx = data.get("indices", np.arange(len(y)))
    r = y[idx] - X[idx] @ theta
    return 0.5*np.sum(r*r), -(X[idx].T @ r)

def score(theta, data):
    r = data["y"] - data["X"] @ theta
    return -data["X"] * r[:, None]

rng=np.random.default_rng(21); X=rng.normal(size=(180,2)); y=X@np.array([1.0,-.5])+rng.normal(scale=.2,size=180)
model=cm.MEstimator(obj, score, max_iterations=200); model.fit({"X":X,"y":y,"n":len(y)}, np.zeros(2))
print(model.summary())""",
 setup="""def obj(theta, data):
    X,y=data['X'],data['y']; idx=data.get('indices',np.arange(len(y))); r=y[idx]-X[idx]@theta; return .5*np.sum(r*r), -(X[idx].T@r)
def score(theta,data):
    r=data['y']-data['X']@theta; return -data['X']*r[:,None]
rng=np.random.default_rng(121); X=rng.normal(size=(100,2)); y=X@np.array([1,-.5])+rng.normal(size=100)*.2
model=cm.MEstimator(obj,score,max_iterations=200); model.fit({'X':X,'y':y,'n':len(y)},np.zeros(2))"""),
"PCA": dict(group="Transforms", subtitle="Principal-components transformer",
 math=r"""`PCA` learns an orthogonal low-rank basis from a design matrix. It centers the training data, computes principal directions, and maps observations to component scores:

$$
Z = (X - \bar X) V_k.
$$

With `whiten=True`, scores are rescaled by the singular values.""",
 api="""Use `PCA(n_components, whiten=False)`. `fit(x)` stores the basis, `transform(x)` maps new data to scores, `fit_transform(x)` combines both, and `inverse_transform(scores)` reconstructs approximate original features. `summary()` exposes components, means, explained variance, ratios, and singular values.""",
 example="""rng=np.random.default_rng(22)
x=rng.normal(size=(150,5)) @ np.array([[1,.2,.1,0,0],[0,.8,.3,.1,0],[0,0,.5,.2,.1],[0,0,0,.3,.2],[0,0,0,0,.1]])
model=cm.PCA(n_components=2); scores=model.fit_transform(x)
print(scores.shape)
print(model.summary()["explained_variance_ratio"])
print(model.inverse_transform(scores[:2]))""",
 setup="""rng=np.random.default_rng(122); x=rng.normal(size=(80,5)); model=cm.PCA(n_components=2); model.fit(x)"""),
"KernelBasis": dict(group="Transforms", subtitle="Kernel feature transformer against the training basis",
 math=r"""`KernelBasis` stores a training design and transforms new rows into kernel similarities against that basis. For a Gaussian kernel, the transformed feature for training row $j$ is

$$
\phi_j(x) = \exp\{-\|x-x_j\|^2/h\}.
$$

The resulting feature matrix can be fed into any downstream regression estimator.""",
 api="""Use `KernelBasis(kernel='gaussian', bandwidth=0.5, coef0=1.0, degree=2.0)`. `fit(x)` stores training rows. `transform(x)` returns kernel features. `summary()` reports the chosen kernel and basis dimensions.""",
 example="""rng=np.random.default_rng(23)
x=rng.normal(size=(80,2)); y=np.sin(x[:,0])+rng.normal(scale=.1,size=80)
basis=cm.KernelBasis(kernel="gaussian", bandwidth=0.8); z=basis.fit_transform(x)
reg=cm.Ridge(penalty=0.1); reg.fit(z,y)
print(basis.summary())
print(reg.predict(basis.transform(x[:3])))""",
 setup="""rng=np.random.default_rng(123); x=rng.normal(size=(50,2)); model=cm.KernelBasis(kernel='gaussian',bandwidth=.8); model.fit(x)"""),
"Optimizers": dict(group="Estimation interfaces", subtitle="Static optimization routines for Python callbacks",
 math=r"""`Optimizers` is a small namespace for callback-driven numerical optimization. It is not an estimator; it exposes reusable routines for smooth objectives, nonlinear least squares, and a simple stochastic global search.""",
 api="""The methods are static and return plain dictionaries with scipy-like keys: `x`, `fun`, `nit`, `success`, `message`, and `method`. Smooth minimizers require objective and gradient callbacks; Gauss-Newton requires residual and Jacobian callbacks; simulated annealing only requires the objective.""",
 example="""def fun(theta):
    return float(np.sum((theta - np.array([1.0, -2.0]))**2))

def grad(theta):
    return 2.0 * (theta - np.array([1.0, -2.0]))

res = cm.Optimizers.minimize_bfgs(fun, np.zeros(2), grad, max_iterations=100)
print(res)""",
 setup="""class _Dummy:
    def summary(self):
        return {'note':'Optimizers has static methods, not fitted state'}
model=_Dummy()"""),
}

# remove placeholder
PAGES.pop("StaggeredPanelEventStudy", None)

def restore_latex_escapes(text: str) -> str:
    # A few LaTeX commands in ordinary Python strings contain escape-prefix
    # characters like \t, \f, \r, \a, or \b. Restore them before writing qmd.
    return (
        text.replace("\t", r"\t")
        .replace("\f", r"\f")
        .replace("\r", r"\r")
        .replace("\a", r"\a")
        .replace("\b", r"\b")
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--force",
    action="store_true",
    help="overwrite existing reference pages with first-pass scaffolds",
)
args = parser.parse_args()

written = 0
skipped = 0
for class_name, meta in PAGES.items():
    output_path = REF / f"{class_name}.qmd"
    if output_path.exists() and not args.force:
        skipped += 1
        continue
    text = COMMON_HEADER.format(title=class_name, subtitle=meta["subtitle"])
    text += f"\n## Where it fits\n\n**Group:** {meta['group']}\n\n{meta['math']}\n\n"
    text += f"## Python API\n\nConstructor: `cm.{class_name}`\n\n{meta['api']}\n\n"
    text += f"```{{python}}\nprint(inspect.signature(cm.{class_name}))\n```\n\n"
    text += method_table(class_name)
    formatted_example = format_python_snippet(meta["example"])
    text += f"\n## Minimal example\n\n```{{python}}\n#| code-fold: false\n{formatted_example}\n```\n\n"
    text += "## `summary()` contract\n\nThe table below is generated by fitting the live class in this repository and then inspecting `summary()`. Shapes are shown because most values are plain NumPy arrays or scalars.\n\n"
    text += summary_table(class_name, meta["setup"])
    output_path.write_text(restore_latex_escapes(text))
    written += 1

print(f"Wrote {written} class page scaffolds under {REF}; skipped {skipped} existing pages")
