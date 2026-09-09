"""Human-readable numerical-experiment descriptions for the scaling report."""

from __future__ import annotations

from typing import TypedDict


class ExperimentDetail(TypedDict):
    dgp: str
    fit: str
    comparison: str


EXPERIMENT_DETAILS: dict[str, ExperimentDetail] = {
    "ABCOLS": {
        "dgp": "Gaussian linear outcome with n rows and k standard-normal continuous covariates; two deterministic categorical variables cycle through 8 and 5 levels.",
        "fit": "ABCOLS with centered continuous covariates and categorical main effects; the scaling run does not add continuous-by-category or category-by-category interactions.",
        "comparison": "No exact public implementation was found. R wec is retained as weighted-effect-coding provenance, not timed as an equivalent estimator.",
    },
    "OLS": {
        "dgp": "Gaussian linear outcome y = X beta + epsilon with X of shape n by k, beta_j increasing from 0.2 to 1 and normalized by sqrt(k), and unit-variance noise.",
        "fit": "Unpenalized Crabbymetrics OLS versus scikit-learn LinearRegression, both with their normal intercept conventions.",
        "comparison": "Shape- and estimand-matched timing comparison.",
    },
    "FixedEffectsOLS": {
        "dgp": "The Gaussian linear design plus one fixed-effect identifier cycling over min(1000, n/20) groups.",
        "fit": "Crabbymetrics within-transformed FixedEffectsOLS versus PyFixest feols and R fixest feols with iid covariance work requested.",
        "comparison": "Same outcome, covariates, and one-way fixed-effect partition; implementation overhead and covariance defaults can still differ.",
    },
    "ElasticNet": {
        "dgp": "The Gaussian linear design with n rows and k covariates.",
        "fit": "Penalty 0.01, l1 ratio 0.5, and at most 300 iterations in Crabbymetrics and scikit-learn.",
        "comparison": "Matched regularization family and tuning values; stopping criteria are library-specific.",
    },
    "Ridge": {
        "dgp": "The Gaussian linear design with n rows and k covariates.",
        "fit": "Unit ridge penalty in Crabbymetrics and scikit-learn.",
        "comparison": "Matched penalized least-squares problem modulo intercept and solver conventions.",
    },
    "BaggedPolynomialRegressor": {
        "dgp": "The Gaussian linear design is deliberately used even though the fitted basis is quadratic, isolating feature-expansion and bagging costs from a changing signal model.",
        "fit": "Ten degree-2 ridge learners, at most 12 raw features per learner and at most 100,000 sampled rows, compared with a scikit-learn PolynomialFeatures/StandardScaler/Ridge BaggingRegressor pipeline.",
        "comparison": "Matched computational pipeline, but bootstrap draws and standardization details are implementation-specific.",
    },
    "Logit": {
        "dgp": "Binary response 1[X beta plus a standard logistic shock exceeds zero].",
        "fit": "Unpenalized Logit with at most 100 iterations versus scikit-learn LogisticRegression using lbfgs and 100 iterations.",
        "comparison": "Matched binary-logit family; convergence tolerances and regularization conventions differ slightly.",
    },
    "MultinomialLogit": {
        "dgp": "Three response classes obtained by thresholding X beta plus a standard-normal shock at -0.5 and 0.5.",
        "fit": "Crabbymetrics multinomial logit versus scikit-learn multinomial LogisticRegression/lbfgs, each capped at 100 iterations.",
        "comparison": "Matched model family; coefficient normalization and convergence checks are library-specific.",
    },
    "Poisson": {
        "dgp": "Poisson response with mean exp(X beta), clipping the linear predictor to [-1.5, 1.5] to avoid pathological counts.",
        "fit": "Unpenalized Poisson regression with at most 100 iterations versus scikit-learn PoissonRegressor.",
        "comparison": "Matched log-link Poisson mean model with library-specific numerical solvers.",
    },
    "ExponentialPH": {
        "dgp": "Survival time is exponential with scale exp(-clip(X beta, -1, 1)); censoring/event indicator is Bernoulli(0.8).",
        "fit": "Crabbymetrics ExponentialPH with k covariates.",
        "comparison": "R flexsurv is documented but not timed because its generic regression parameterization is not a clean drop-in PH comparator here.",
    },
    "WeibullPH": {
        "dgp": "The same censored proportional-hazards stress design used for the other survival estimators.",
        "fit": "Crabbymetrics WeibullPH with k covariates.",
        "comparison": "R flexsurv is documented as the canonical reference; parameterization differences keep it out of the generic timing grid.",
    },
    "CoxPH": {
        "dgp": "The censored proportional-hazards design with exponential baseline time and k Gaussian covariates.",
        "fit": "Crabbymetrics CoxPH versus lifelines CoxPHFitter and R survival::coxph with Breslow ties.",
        "comparison": "Matched partial-likelihood family; risk-set algorithms and default inference work differ.",
    },
    "AndersenGill": {
        "dgp": "Counting-process rows with uniform starts, exponentially distributed positive interval lengths, Bernoulli(0.7) events, and k Gaussian covariates.",
        "fit": "Crabbymetrics AndersenGill versus R survival::coxph on Surv(start, stop, event) with Breslow ties.",
        "comparison": "Matched counting-process partial likelihood without subject-clustered covariance in the timing call.",
    },
    "TwoSLS": {
        "dgp": "k exogenous controls X and k excluded instruments Z; one endogenous regressor d = 0.7 Z_1 plus noise; y = d + X beta plus noise.",
        "fit": "Crabbymetrics TwoSLS versus PyFixest's IV formula with the same k controls and k excluded instruments.",
        "comparison": "Matched linear IV dimensions and estimand; formula construction and covariance bookkeeping differ.",
    },
    "HorizontalPanelRidge": {
        "dgp": "Low-rank panel with n periods and k units; control outcomes load on two common factors and treated outcomes are noisy convex combinations of controls, treated in the final third.",
        "fit": "Crabbymetrics HorizontalPanelRidge with unit penalty versus the corresponding scikit-learn Ridge donor-regression building block.",
        "comparison": "The scikit-learn row is a matched donor-regression kernel, not the complete cohort orchestration.",
    },
    "SyntheticControl": {
        "dgp": "n pre-treatment periods by k independent Gaussian donor series; the treated series is the equal-weight donor average plus small noise.",
        "fit": "Crabbymetrics SyntheticControl with at most 300 simplex iterations.",
        "comparison": "R Synth is the canonical reference but its data-preparation/optimization interface is not timed on the generic grid.",
    },
    "SyntheticDID": {
        "dgp": "Low-rank n-period by k-unit panel; treated units are convex mixtures of controls and treatment starts after two thirds of periods.",
        "fit": "SyntheticDID with unit and time penalties fixed at 0.01 and at most 3,000 simplex iterations.",
        "comparison": "R synthdid is the exact external family but is retained as provenance because the package was unavailable in the benchmark environment.",
    },
    "AugmentedBalancing": {
        "dgp": "The same low-rank staggered panel used for SyntheticDID, with no supplied outcome surface so runtime covers raw double balancing.",
        "fit": "Double AugmentedBalancing with unit and time penalties 0.01 and at most 3,000 iterations.",
        "comparison": "R augsynth and the repository's independent quadprog parity fixture are references; neither is substituted for the exact Crabbymetrics configuration in timing plots.",
    },
    "MatrixCompletion": {
        "dgp": "The low-rank treated panel with n periods and k units.",
        "fit": "MatrixCompletion with at most 50 outer iterations and SVD rank min(6, k-1), retaining unit and time effects.",
        "comparison": "R fect method=mc is the canonical family reference, but its full long-panel interface is not treated as a generic-grid drop-in.",
    },
    "InteractiveFixedEffects": {
        "dgp": "The untreated low-rank n-period by k-unit outcome matrix from the panel generator.",
        "fit": "InteractiveFixedEffects with rank min(2, k-1) and the default force specification.",
        "comparison": "R fect method=ife and gsynth are canonical family references, not timed substitutes.",
    },
    "BalancingWeights": {
        "dgp": "Gaussian X with the first 80 percent of rows as the source sample and the final 20 percent as the target sample.",
        "fit": "Quadratic BalancingWeights, default box constraints, and at most 100 iterations, balancing k raw covariate means.",
        "comparison": "ebal, WeightIt, and CBPS are provenance because their objectives and constraints are not identical.",
    },
    "MEstimator": {
        "dgp": "Gaussian linear outcome with an explicit intercept-augmented n by (k+1) design.",
        "fit": "Generic MEstimator receives a least-squares objective/analytic gradient callback and observation-level score callback, starting at zero for at most 30 iterations.",
        "comparison": "R geex is the callback-framework reference; arbitrary user callbacks prevent a single universal external timing adapter.",
    },
    "GMM": {
        "dgp": "Exactly identified linear moments X_i(y_i - X_i' theta) on the intercept-augmented Gaussian linear design.",
        "fit": "Identity-weighted GMM with analytic Jacobian, zero initialization, and at most 30 iterations.",
        "comparison": "R gmm and statsmodels are framework references; their generic callback and covariance work is not forced into a misleading one-size timing row.",
    },
    "EPLM": {
        "dgp": "Continuous treatment d = 0.25 X beta plus noise and outcome y = 0.8 d + X beta plus noise.",
        "fit": "Crabbymetrics EPLM with its default finite-difference epsilon.",
        "comparison": "DoubleML PLR is related but uses different orthogonalization/cross-fitting, so it is provenance rather than an EPLM speed comparator.",
    },
    "AverageDerivative": {
        "dgp": "The continuous-treatment partially linear design used for EPLM.",
        "fit": "Doubly robust AverageDerivative with default finite-difference epsilon.",
        "comparison": "R np gradient routines are the nearest public reference, not an identical estimator.",
    },
    "PartiallyLinearDML": {
        "dgp": "Continuous treatment d = 0.25 X beta plus noise and outcome y = 0.8 d + X beta plus noise.",
        "fit": "Crabbymetrics PartiallyLinearDML with ridge penalty 0.1 and two folds versus DoubleML PLR with two-fold ridge nuisance learners.",
        "comparison": "Matched partially linear orthogonal-score family with library-specific fold draws and nuisance conventions.",
    },
    "AIPW": {
        "dgp": "Binary treatment obtained by median-splitting 0.25 X beta plus noise; y = 0.8 d + X beta plus noise.",
        "fit": "Crabbymetrics AIPW with ridge penalty 0.1, two folds, and propensity clipping 0.02 versus DoubleML IRM with ridge/logit nuisances and the same clipping threshold.",
        "comparison": "Matched ATE/AIPW family; fold construction and nuisance-standardization details differ.",
    },
    "DynamicCovariateBalance": {
        "dgp": "Approximately n/4 units over four decision periods, k Gaussian history variables per period, logistic treatment driven by the first history coordinate, and a random-walk outcome with additive 0.5 treatment effects.",
        "fit": "DynamicCovariateBalance targets the all-zero four-period path with at most 100 iterations.",
        "comparison": "No exact public Viviano-Bradic implementation was found; the estimator is native-only in the timing grid.",
    },
    "ParallelTrendsSNMM": {
        "dgp": "The four-period dynamic-treatment design with a five-column outcome path including baseline.",
        "fit": "ParallelTrendsSNMM with maximum horizon 1, blip treatment mode, two nuisance folds, and fixed seed 1729.",
        "comparison": "gesttools is a nearby SNMM reference but not a parallel-trends SNMM, so it is not timed as equivalent.",
    },
    "RegressionBlip": {
        "dgp": "The four-period dynamic-treatment design with period outcomes aligned to treatment and k history covariates.",
        "fit": "RegressionBlip with one treatment lag and time effects.",
        "comparison": "DTRreg and gesttools are related blip/SNMM references, but neither matches the shipped regression-blip contract exactly.",
    },
}
