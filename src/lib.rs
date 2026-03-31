mod estimators;
mod optimizers;
mod utils;

use crate::estimators::{
    ElasticNet, FixedEffectsOLS, KernelBasis, Logit, MEstimator, MultinomialLogit, PcaTransformer,
    Poisson, Ridge, SyntheticControl, TwoSLS, FTRL, GMM, OLS,
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
    m.add_class::<SyntheticControl>()?;
    m.add_class::<FTRL>()?;
    m.add_class::<MEstimator>()?;
    m.add_class::<GMM>()?;
    m.add_class::<PcaTransformer>()?;
    m.add_class::<KernelBasis>()?;
    m.add_class::<Optimizers>()?;
    Ok(())
}
