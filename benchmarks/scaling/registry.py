"""Estimator inventory and external-reference registry for scaling benchmarks."""

from __future__ import annotations

from typing import TypedDict

# Increment when adapter semantics change; old timings are not interchangeable.
ADAPTER_REVISION = 2


class EstimatorSpec(TypedDict):
    family: str
    references: tuple[str, ...]
    k_semantics: str
    memory_factor: float


# This is intentionally explicit: a missing public estimator should fail the inventory test.
ESTIMATORS: dict[str, EstimatorSpec] = {
    "ABCOLS": {
        "family": "tabular",
        "references": ("r-wec",),
        "k_semantics": "continuous covariates (plus two categorical variables)",
        "memory_factor": 8.0,
    },
    "OLS": {
        "family": "tabular",
        "references": ("sklearn-linear-regression",),
        "k_semantics": "covariates",
        "memory_factor": 5.0,
    },
    "FixedEffectsOLS": {
        "family": "fixed-effects",
        "references": ("pyfixest-feols", "r-fixest-feols"),
        "k_semantics": "covariates",
        "memory_factor": 8.0,
    },
    "ElasticNet": {
        "family": "tabular",
        "references": ("sklearn-elastic-net",),
        "k_semantics": "covariates",
        "memory_factor": 6.0,
    },
    "Ridge": {
        "family": "tabular",
        "references": ("sklearn-ridge",),
        "k_semantics": "covariates",
        "memory_factor": 5.0,
    },
    "BaggedPolynomialRegressor": {
        "family": "polynomial",
        "references": ("sklearn-bagged-polynomial",),
        "k_semantics": "raw covariates",
        "memory_factor": 25.0,
    },
    "Logit": {
        "family": "glm",
        "references": ("sklearn-logistic-regression",),
        "k_semantics": "covariates",
        "memory_factor": 7.0,
    },
    "MultinomialLogit": {
        "family": "glm",
        "references": ("sklearn-multinomial-logit",),
        "k_semantics": "covariates",
        "memory_factor": 12.0,
    },
    "Poisson": {
        "family": "glm",
        "references": ("sklearn-poisson-regressor",),
        "k_semantics": "covariates",
        "memory_factor": 7.0,
    },
    "ExponentialPH": {
        "family": "survival",
        "references": ("r-flexsurv-exponential",),
        "k_semantics": "covariates",
        "memory_factor": 8.0,
    },
    "WeibullPH": {
        "family": "survival",
        "references": ("r-flexsurv-weibull-ph",),
        "k_semantics": "covariates",
        "memory_factor": 9.0,
    },
    "CoxPH": {
        "family": "survival",
        "references": ("lifelines-cox-ph", "r-survival-coxph"),
        "k_semantics": "covariates",
        "memory_factor": 12.0,
    },
    "AndersenGill": {
        "family": "survival",
        "references": ("r-survival-andersen-gill",),
        "k_semantics": "covariates",
        "memory_factor": 12.0,
    },
    "TwoSLS": {
        "family": "iv",
        "references": ("pyfixest-iv", "r-ivreg"),
        "k_semantics": "exogenous covariates and excluded instruments",
        "memory_factor": 12.0,
    },
    "HorizontalPanelRidge": {
        "family": "panel",
        "references": ("sklearn-ridge",),
        "k_semantics": "donor units",
        "memory_factor": 6.0,
    },
    "SyntheticControl": {
        "family": "synthetic-control",
        "references": ("r-synth",),
        "k_semantics": "donor units",
        "memory_factor": 8.0,
    },
    "SyntheticDID": {
        "family": "panel",
        "references": ("r-synthdid",),
        "k_semantics": "units; n is periods",
        "memory_factor": 12.0,
    },
    "AugmentedBalancing": {
        "family": "panel",
        "references": ("r-augsynth", "r-independent-quadprog"),
        "k_semantics": "units; n is periods",
        "memory_factor": 16.0,
    },
    "MatrixCompletion": {
        "family": "panel",
        "references": ("r-fect-mc",),
        "k_semantics": "units; n is periods",
        "memory_factor": 16.0,
    },
    "InteractiveFixedEffects": {
        "family": "panel",
        "references": ("r-fect-ife", "r-gsynth"),
        "k_semantics": "units; n is periods",
        "memory_factor": 12.0,
    },
    "BalancingWeights": {
        "family": "balancing",
        "references": ("r-ebal", "r-weightit", "r-cbps"),
        "k_semantics": "balance functions",
        "memory_factor": 12.0,
    },
    "MEstimator": {
        "family": "moments",
        "references": ("r-geex",),
        "k_semantics": "parameters/covariates",
        "memory_factor": 10.0,
    },
    "GMM": {
        "family": "moments",
        "references": ("r-gmm", "statsmodels-gmm"),
        "k_semantics": "parameters/instruments",
        "memory_factor": 14.0,
    },
    "EPLM": {
        "family": "semiparametric",
        "references": ("doubleml-plr-nearest",),
        "k_semantics": "controls",
        "memory_factor": 18.0,
    },
    "AverageDerivative": {
        "family": "semiparametric",
        "references": ("r-np-gradient-nearest",),
        "k_semantics": "controls",
        "memory_factor": 22.0,
    },
    "PartiallyLinearDML": {
        "family": "semiparametric",
        "references": ("doubleml-plr",),
        "k_semantics": "controls",
        "memory_factor": 24.0,
    },
    "AIPW": {
        "family": "semiparametric",
        "references": ("doubleml-irm", "econml-drlearner"),
        "k_semantics": "controls",
        "memory_factor": 24.0,
    },
    "DynamicCovariateBalance": {
        "family": "dynamic",
        "references": ("native-only-no-public-exact-match",),
        "k_semantics": "history covariates",
        "memory_factor": 18.0,
    },
    "ParallelTrendsSNMM": {
        "family": "dynamic",
        "references": ("r-gesttools-nearest",),
        "k_semantics": "history covariates",
        "memory_factor": 22.0,
    },
    "RegressionBlip": {
        "family": "dynamic",
        "references": ("r-dtrreg-nearest", "r-gesttools-nearest"),
        "k_semantics": "history covariates",
        "memory_factor": 14.0,
    },
}


REFERENCE_URLS = {
    "sklearn-linear-regression": "https://github.com/scikit-learn/scikit-learn",
    "sklearn-ridge": "https://github.com/scikit-learn/scikit-learn",
    "sklearn-elastic-net": "https://github.com/scikit-learn/scikit-learn",
    "sklearn-bagged-polynomial": "https://github.com/scikit-learn/scikit-learn",
    "sklearn-logistic-regression": "https://github.com/scikit-learn/scikit-learn",
    "sklearn-multinomial-logit": "https://github.com/scikit-learn/scikit-learn",
    "sklearn-poisson-regressor": "https://github.com/scikit-learn/scikit-learn",
    "pyfixest-feols": "https://github.com/py-econometrics/pyfixest",
    "pyfixest-iv": "https://github.com/py-econometrics/pyfixest",
    "lifelines-cox-ph": "https://github.com/CamDavidsonPilon/lifelines",
    "statsmodels-gmm": "https://github.com/statsmodels/statsmodels",
    "doubleml-plr-nearest": "https://github.com/DoubleML/doubleml-for-py",
    "doubleml-plr": "https://github.com/DoubleML/doubleml-for-py",
    "doubleml-irm": "https://github.com/DoubleML/doubleml-for-py",
    "econml-drlearner": "https://github.com/py-why/EconML",
    "r-wec": "https://github.com/cran/wec",
    "r-fixest-feols": "https://github.com/lrberge/fixest",
    "r-flexsurv-exponential": "https://github.com/cran/flexsurv",
    "r-flexsurv-weibull-ph": "https://github.com/cran/flexsurv",
    "r-survival-coxph": "https://github.com/cran/survival",
    "r-survival-andersen-gill": "https://github.com/cran/survival",
    "r-ivreg": "https://github.com/cran/ivreg",
    "r-synth": "https://github.com/cran/Synth",
    "r-synthdid": "https://github.com/synth-inference/synthdid",
    "r-augsynth": "https://github.com/ebenmichael/augsynth",
    "r-independent-quadprog": "../../tests/fixtures/generate_r_augmented_balancing_reference.R",
    "r-fect-mc": "https://github.com/xuyiqing/fect",
    "r-fect-ife": "https://github.com/xuyiqing/fect",
    "r-gsynth": "https://github.com/xuyiqing/gsynth",
    "r-ebal": "https://github.com/apoorvalal/ebal",
    "r-weightit": "https://github.com/cran/WeightIt",
    "r-cbps": "https://github.com/cran/CBPS",
    "r-geex": "https://github.com/bsaul/geex",
    "r-gmm": "https://github.com/cran/gmm",
    "r-np-gradient-nearest": "https://github.com/cran/np",
    "r-gesttools-nearest": "https://github.com/danieltompsett/gesttools",
    "r-dtrreg-nearest": "https://github.com/cran/DTRreg",
    "native-only-no-public-exact-match": "",
}


# These have a cell runner in this repository. The remaining references are
# provenance/cross-check targets whose interface is not shape-compatible with
# the generic grid, or for which no exact public implementation was found.
RUNNABLE_REFERENCES = {
    "sklearn-linear-regression",
    "sklearn-ridge",
    "sklearn-elastic-net",
    "sklearn-bagged-polynomial",
    "sklearn-logistic-regression",
    "sklearn-multinomial-logit",
    "sklearn-poisson-regressor",
    "pyfixest-feols",
    "pyfixest-iv",
    "lifelines-cox-ph",
    "doubleml-plr",
    "doubleml-irm",
    "r-fixest-feols",
    "r-survival-coxph",
    "r-survival-andersen-gill",
}


def implementations(estimator: str) -> tuple[str, ...]:
    """Return the native implementation followed by registered references."""

    return ("crabbymetrics", *ESTIMATORS[estimator]["references"])


def runnable_implementations(estimator: str) -> tuple[str, ...]:
    """Return implementations backed by an executable generic-grid adapter."""

    return (
        "crabbymetrics",
        *(
            ref
            for ref in ESTIMATORS[estimator]["references"]
            if ref in RUNNABLE_REFERENCES
        ),
    )
