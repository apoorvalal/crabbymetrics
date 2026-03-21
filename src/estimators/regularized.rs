use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, fisher_cov_binary, hc1_cov, pyarray1_from_f64,
    pyarray2_from_f64, take_rows, take_rows_vec, to_array1, to_array1_i32, to_array2,
};
use linfa::prelude::{Fit, FitWith, Predict};
use linfa::Dataset;
use linfa_elasticnet::ElasticNet as LinfaElasticNet;
use linfa_ftrl::Ftrl as LinfaFtrl;
use ndarray::{s, Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
pub struct ElasticNet {
    penalty: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: u32,
    model: Option<linfa_elasticnet::ElasticNet<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

#[pymethods]
impl ElasticNet {
    #[new]
    #[pyo3(signature = (penalty=1.0, l1_ratio=0.5, tolerance=1e-4, max_iterations=1000))]
    fn new(penalty: f64, l1_ratio: f64, tolerance: f64, max_iterations: u32) -> Self {
        Self {
            penalty,
            l1_ratio,
            fit_intercept: true,
            tolerance,
            max_iterations,
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
        let params = LinfaElasticNet::params()
            .penalty(self.penalty)
            .l1_ratio(self.l1_ratio)
            .with_intercept(self.fit_intercept)
            .tolerance(self.tolerance)
            .max_iterations(self.max_iterations);
        let model = params
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
            .ok_or_else(|| PyValueError::new_err("ElasticNet model is not fitted"))?;
        let x = to_array2(&x);
        let pred = model.predict(&x);
        Ok(pyarray1_from_f64(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ElasticNet model is not fitted"))?;
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
                model.hyperplane().to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, model.hyperplane().to_owned(), None, se_all)
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
            let params = LinfaElasticNet::params()
                .penalty(self.penalty)
                .l1_ratio(self.l1_ratio)
                .with_intercept(self.fit_intercept)
                .tolerance(self.tolerance)
                .max_iterations(self.max_iterations);
            let model = params
                .fit(&dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            if self.fit_intercept {
                out[[i, 0]] = model.intercept();
                out.row_mut(i)
                    .slice_mut(s![1..])
                    .assign(&model.hyperplane());
            } else {
                out.row_mut(i).assign(&model.hyperplane());
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct FTRL {
    alpha: f64,
    beta: f64,
    l1_ratio: f64,
    l2_ratio: f64,
    model: Option<LinfaFtrl<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<bool>>,
}

#[pymethods]
impl FTRL {
    #[new]
    #[pyo3(signature = (alpha=0.1, beta=1.0, l1_ratio=1.0, l2_ratio=1.0))]
    fn new(alpha: f64, beta: f64, l1_ratio: f64, l2_ratio: f64) -> Self {
        Self {
            alpha,
            beta,
            l1_ratio,
            l2_ratio,
            model: None,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i32>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1_i32(&y).mapv(|v| v != 0);
        let dataset = Dataset::new(x.clone(), y.clone());
        let params = linfa_ftrl::Ftrl::params()
            .alpha(self.alpha)
            .beta(self.beta)
            .l1_ratio(self.l1_ratio)
            .l2_ratio(self.l2_ratio);
        let model = params
            .fit_with(None, &dataset)
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
            .ok_or_else(|| PyValueError::new_err("FTRL model is not fitted"))?;
        let x = to_array2(&x);
        let probs = model.predict(&x).mapv(|v| f64::from(*v));
        Ok(pyarray1_from_f64(py, &probs))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FTRL model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let weights = model.get_weights();
        let probs = model.predict(x).mapv(|v| f64::from(*v));
        let cov = fisher_cov_binary(x, &probs).map_err(PyValueError::new_err)?;
        let se = diag_sqrt(&cov);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, &weights))?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &se))?;
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
        let mut out = Array2::<f64>::zeros((n_bootstrap, x.ncols()));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let mut yb = Array1::from_elem(idx.len(), false);
            for (j, &row) in idx.iter().enumerate() {
                yb[j] = y[row];
            }
            let dataset = Dataset::new(xb, yb);
            let params = linfa_ftrl::Ftrl::params()
                .alpha(self.alpha)
                .beta(self.beta)
                .l1_ratio(self.l1_ratio)
                .l2_ratio(self.l2_ratio);
            let model = params
                .fit_with(None, &dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            out.row_mut(i).assign(&model.get_weights());
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}
