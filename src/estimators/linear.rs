use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, hc1_cov, pyarray1_from_f64, pyarray2_from_f64,
    take_rows, take_rows_u32, take_rows_vec, to_array1, to_array2, to_array2_u32,
};
use linfa::prelude::{Fit, Predict};
use linfa::Dataset;
use linfa_linear::LinearRegression as LinfaLinearRegression;
use ndarray::{concatenate, s, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use within::{Solver as WithinSolver, SolverParams as WithinSolverParams};

struct FixedEffectsOlsFitResult {
    coef: Array1<f64>,
    x_resid: Array2<f64>,
    y_resid: Array1<f64>,
}

fn fit_fixed_effects_ols(
    x: &Array2<f64>,
    y: &Array1<f64>,
    fe: &Array2<u32>,
) -> PyResult<FixedEffectsOlsFitResult> {
    if x.nrows() != y.len() || fe.nrows() != y.len() {
        return Err(PyValueError::new_err("row count mismatch"));
    }
    if x.ncols() == 0 {
        return Err(PyValueError::new_err("x must have at least one column"));
    }
    if fe.ncols() == 0 {
        return Err(PyValueError::new_err("fe must have at least one column"));
    }

    let params = WithinSolverParams::default();
    let solver = WithinSolver::new(fe.view(), None, &params, None)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let mut rhs_owned = Vec::with_capacity(x.ncols() + 1);
    rhs_owned.push(y.clone());
    for col in x.columns() {
        rhs_owned.push(col.to_owned());
    }
    let rhs: Vec<&[f64]> = rhs_owned
        .iter()
        .map(|column| column.as_slice().expect("owned arrays are contiguous"))
        .collect();

    let partialled = solver
        .solve_batch(&rhs)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    if !partialled.converged().iter().all(|flag| *flag) {
        return Err(PyValueError::new_err(format!(
            "fixed-effects partialling-out did not converge for all right-hand sides: converged={:?}, residual={:?}",
            partialled.converged(),
            partialled.final_residual()
        )));
    }

    let y_resid = Array1::from_vec(partialled.demeaned(0).to_vec());
    let mut x_resid = Array2::<f64>::zeros((x.nrows(), x.ncols()));
    for j in 0..x.ncols() {
        let col = Array1::from_vec(partialled.demeaned(j + 1).to_vec());
        x_resid.column_mut(j).assign(&col);
    }

    let dataset = Dataset::new(x_resid.clone(), y_resid.clone());
    let model = LinfaLinearRegression::new()
        .with_intercept(false)
        .fit(&dataset)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(FixedEffectsOlsFitResult {
        coef: model.params().to_owned(),
        x_resid,
        y_resid,
    })
}

#[pyclass]
pub struct OLS {
    fit_intercept: bool,
    model: Option<linfa_linear::FittedLinearRegression<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

#[pymethods]
impl OLS {
    #[new]
    fn new() -> Self {
        Self {
            fit_intercept: true,
            model: None,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let dataset = Dataset::new(x.clone(), y.clone());
        let model = LinfaLinearRegression::new()
            .with_intercept(self.fit_intercept)
            .fit(&dataset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.model = Some(model);
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("OLS model is not fitted"))?;
        let x = to_array2(&x);
        let pred = model.predict(&x);
        Ok(pyarray1_from_f64(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("OLS model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let y_hat = model.predict(x);
        let residuals = y - &y_hat;
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = hc1_cov(&design, &residuals).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept, coef, intercept_se, coef_se) = if self.fit_intercept {
            (
                model.intercept(),
                model.params().to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, model.params().to_owned(), None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, &coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((
            n_bootstrap,
            x.ncols() + if self.fit_intercept { 1 } else { 0 },
        ));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let dataset = Dataset::new(xb, yb);
            let model = LinfaLinearRegression::new()
                .with_intercept(self.fit_intercept)
                .fit(&dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            if self.fit_intercept {
                out[[i, 0]] = model.intercept();
                out.row_mut(i).slice_mut(s![1..]).assign(&model.params());
            } else {
                out.row_mut(i).assign(&model.params());
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct FixedEffectsOLS {
    coef: Option<Array1<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
    fe: Option<Array2<u32>>,
    x_resid: Option<Array2<f64>>,
    y_resid: Option<Array1<f64>>,
}

#[pymethods]
impl FixedEffectsOLS {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            x: None,
            y: None,
            fe: None,
            x_resid: None,
            y_resid: None,
        }
    }

    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        fe: PyReadonlyArray2<u32>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let x = to_array2(&x);
        let fe = to_array2_u32(&fe);
        let y = to_array1(&y);
        let fit = fit_fixed_effects_ols(&x, &y, &fe)?;

        self.coef = Some(fit.coef);
        self.x = Some(x);
        self.y = Some(y);
        self.fe = Some(fe);
        self.x_resid = Some(fit.x_resid);
        self.y_resid = Some(fit.y_resid);
        Ok(())
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FixedEffectsOLS model is not fitted"))?;
        let x_resid = self
            .x_resid
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No residualized design stored"))?;
        let y_resid = self
            .y_resid
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No residualized response stored"))?;

        let residuals = y_resid - &x_resid.dot(coef);
        let cov = hc1_cov(x_resid, &residuals).map_err(PyValueError::new_err)?;
        let coef_se = diag_sqrt(&cov);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let fe = self
            .fe
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((n_bootstrap, x.ncols()));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let feb = take_rows_u32(fe, idx);
            let fit = fit_fixed_effects_ols(&xb, &yb, &feb)?;
            out.row_mut(i).assign(&fit.coef);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct TwoSLS {
    fit_intercept: bool,
    coef: Option<Array1<f64>>,
    intercept: f64,
    x_endog: Option<Array2<f64>>,
    x_exog: Option<Array2<f64>>,
    z: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

#[pymethods]
impl TwoSLS {
    #[new]
    fn new() -> Self {
        Self {
            fit_intercept: true,
            coef: None,
            intercept: 0.0,
            x_endog: None,
            x_exog: None,
            z: None,
            y: None,
        }
    }

    fn fit(
        &mut self,
        x_endog: PyReadonlyArray2<f64>,
        x_exog: PyReadonlyArray2<f64>,
        z: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let x_endog = to_array2(&x_endog);
        let x_exog = to_array2(&x_exog);
        let z = to_array2(&z);
        let y = to_array1(&y);

        if x_endog.nrows() != y.len() || x_exog.nrows() != y.len() || z.nrows() != y.len() {
            return Err(PyValueError::new_err("row count mismatch"));
        }

        let z_full = if x_exog.ncols() > 0 {
            concatenate(Axis(1), &[x_exog.view(), z.view()])
                .map_err(|_| PyValueError::new_err("failed to concat instruments"))?
        } else {
            z.clone()
        };

        let stage1 = Dataset::new(z_full.clone(), x_endog.column(0).to_owned());
        let stage1_model = LinfaLinearRegression::new()
            .with_intercept(self.fit_intercept)
            .fit(&stage1)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let x_endog_hat = stage1_model.predict(&z_full);

        let x_hat = if x_exog.ncols() > 0 {
            concatenate(
                Axis(1),
                &[x_endog_hat.view().insert_axis(Axis(1)), x_exog.view()],
            )
            .map_err(|_| PyValueError::new_err("failed to concat endog/exog"))?
        } else {
            x_endog_hat.view().insert_axis(Axis(1)).to_owned()
        };

        let stage2 = Dataset::new(x_hat.clone(), y.clone());
        let stage2_model = LinfaLinearRegression::new()
            .with_intercept(self.fit_intercept)
            .fit(&stage2)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        self.coef = Some(stage2_model.params().to_owned());
        self.intercept = stage2_model.intercept();
        self.x_endog = Some(x_endog);
        self.x_exog = Some(x_exog);
        self.z = Some(z);
        self.y = Some(y);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("TwoSLS model is not fitted"))?;
        let x = to_array2(&x);
        let pred: Array1<f64> = x.dot(coef) + self.intercept;
        Ok(pyarray1_from_f64(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("TwoSLS model is not fitted"))?;
        let x_endog = self
            .x_endog
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let x_exog = self
            .x_exog
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let z = self
            .z
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let z_full = if x_exog.ncols() > 0 {
            concatenate(Axis(1), &[x_exog.view(), z.view()])
                .map_err(|_| PyValueError::new_err("failed to concat instruments"))?
        } else {
            z.clone()
        };
        let stage1 = Dataset::new(z_full.clone(), x_endog.column(0).to_owned());
        let stage1_model = LinfaLinearRegression::new()
            .with_intercept(self.fit_intercept)
            .fit(&stage1)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let x_endog_hat = stage1_model.predict(&z_full);
        let x_hat = if x_exog.ncols() > 0 {
            concatenate(
                Axis(1),
                &[x_endog_hat.view().insert_axis(Axis(1)), x_exog.view()],
            )
            .map_err(|_| PyValueError::new_err("failed to concat endog/exog"))?
        } else {
            x_endog_hat.view().insert_axis(Axis(1)).to_owned()
        };

        let y_hat = x_hat.dot(coef) + self.intercept;
        let residuals = y - &y_hat;
        let design = if self.fit_intercept {
            add_intercept(&x_hat)
        } else {
            x_hat.clone()
        };
        let cov = hc1_cov(&design, &residuals).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept_se, coef_se) = if self.fit_intercept {
            (Some(se_all[0]), se_all.slice(s![1..]).to_owned())
        } else {
            (None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("TwoSLS model is not fitted"))?;
        let x = self
            .x_endog
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let x_exog = self
            .x_exog
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let z = self
            .z
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((
            n_bootstrap,
            coef.len() + if self.fit_intercept { 1 } else { 0 },
        ));
        for (i, idx) in idxs.iter().enumerate() {
            let x_endog_b = take_rows(x, idx);
            let x_exog_b = take_rows(x_exog, idx);
            let z_b = take_rows(z, idx);
            let yb = take_rows_vec(y, idx);

            let z_full = if x_exog_b.ncols() > 0 {
                concatenate(Axis(1), &[x_exog_b.view(), z_b.view()])
                    .map_err(|_| PyValueError::new_err("failed to concat instruments"))?
            } else {
                z_b
            };
            let stage1 = Dataset::new(z_full.clone(), x_endog_b.column(0).to_owned());
            let stage1_model = LinfaLinearRegression::new()
                .with_intercept(self.fit_intercept)
                .fit(&stage1)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let x_endog_hat = stage1_model.predict(&z_full);
            let x_hat = if x_exog_b.ncols() > 0 {
                concatenate(
                    Axis(1),
                    &[x_endog_hat.view().insert_axis(Axis(1)), x_exog_b.view()],
                )
                .map_err(|_| PyValueError::new_err("failed to concat endog/exog"))?
            } else {
                x_endog_hat.view().insert_axis(Axis(1)).to_owned()
            };
            let stage2 = Dataset::new(x_hat, yb);
            let model = LinfaLinearRegression::new()
                .with_intercept(self.fit_intercept)
                .fit(&stage2)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            if self.fit_intercept {
                out[[i, 0]] = model.intercept();
                out.row_mut(i).slice_mut(s![1..]).assign(&model.params());
            } else {
                out.row_mut(i).assign(&model.params());
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}
