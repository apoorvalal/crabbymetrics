mod estimators;
mod utils;

use crate::estimators::{ElasticNet, FTRL, Logit, MultinomialLogit, OLS, Poisson, TwoSLS};
use pyo3::prelude::*;

#[pymodule]
fn crabbymetrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OLS>()?;
    m.add_class::<ElasticNet>()?;
    m.add_class::<Logit>()?;
    m.add_class::<MultinomialLogit>()?;
    m.add_class::<Poisson>()?;
    m.add_class::<TwoSLS>()?;
    m.add_class::<FTRL>()?;
    Ok(())
}
