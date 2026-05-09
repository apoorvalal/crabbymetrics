mod estimators;
mod optimizers;
mod rla;
mod utils;

use crate::estimators::{
    AverageDerivative, BalancingWeights, ElasticNet, FixedEffectsOLS, HorizontalPanelRidge,
    InteractiveFixedEffects, KernelBasis, Logit, MEstimator, MatrixCompletion, MultinomialLogit,
    PartiallyLinearDML, PcaTransformer, Poisson, Ridge, SyntheticControl, SyntheticDID, TwoSLS,
    AIPW, EPLM, FTRL, GMM, OLS,
};
use crate::optimizers::Optimizers;
use pyo3::prelude::*;

#[pymodule]
fn crabbymetrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OLS>()?;
    m.add_class::<FixedEffectsOLS>()?;
    m.add_class::<ElasticNet>()?;
    m.add_class::<Ridge>()?;
    m.add_class::<Logit>()?;
    m.add_class::<MultinomialLogit>()?;
    m.add_class::<Poisson>()?;
    m.add_class::<TwoSLS>()?;
    m.add_class::<HorizontalPanelRidge>()?;
    m.add_class::<SyntheticControl>()?;
    m.add_class::<SyntheticDID>()?;
    m.add_class::<MatrixCompletion>()?;
    m.add_class::<InteractiveFixedEffects>()?;
    m.add_function(wrap_pyfunction!(crate::estimators::panel_factor, m)?)?;
    m.add_function(wrap_pyfunction!(crate::estimators::panel_fe, m)?)?;
    m.add_class::<BalancingWeights>()?;
    m.add_class::<FTRL>()?;
    m.add_class::<MEstimator>()?;
    m.add_class::<GMM>()?;
    m.add_class::<EPLM>()?;
    m.add_class::<AverageDerivative>()?;
    m.add_class::<PartiallyLinearDML>()?;
    m.add_class::<AIPW>()?;
    m.add_class::<PcaTransformer>()?;
    m.add_class::<KernelBasis>()?;
    m.add_class::<Optimizers>()?;
    m.add_function(wrap_pyfunction!(crate::rla::randomized_range, m)?)?;
    m.add_function(wrap_pyfunction!(crate::rla::randomized_svd, m)?)?;
    m.add_function(wrap_pyfunction!(crate::rla::sketch_ols, m)?)?;
    Ok(())
}
