use super::*;
#[pyclass]
pub struct SyntheticControl {
    max_iterations: u64,
    weights: Option<Array1<f64>>,
    donors: Option<Array2<f64>>,
    treated: Option<Array1<f64>>,
}

#[pymethods]
impl SyntheticControl {
    #[new]
    #[pyo3(signature = (max_iterations=500))]
    fn new(max_iterations: u64) -> Self {
        Self {
            max_iterations,
            weights: None,
            donors: None,
            treated: None,
        }
    }

    fn fit(
        &mut self,
        donors: PyReadonlyArray2<f64>,
        treated: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        let donors = to_array2(&donors);
        let treated = to_array1(&treated);
        let weights = fit_synthetic_control_weights(&donors, &treated, self.max_iterations)?;

        self.weights = Some(weights);
        self.donors = Some(donors);
        self.treated = Some(treated);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        donors: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticControl model is not fitted"))?;
        let donors = to_array2(&donors);
        if donors.ncols() != weights.len() {
            return Err(PyValueError::new_err(
                "donor columns must match number of fitted weights",
            ));
        }
        let pred = donors.dot(weights);
        Ok(pyarray1_from_f64(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticControl model is not fitted"))?;
        let donors = self
            .donors
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let treated = self
            .treated
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("weights", pyarray1_from_f64(py, weights))?;
        dict.set_item("pre_rmse", synthetic_control_rmse(donors, treated, weights))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let donors = self
            .donors
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let treated = self
            .treated
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let idxs = bootstrap_indices(donors.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((n_bootstrap, donors.ncols()));
        for (i, idx) in idxs.iter().enumerate() {
            let donors_b = take_rows(donors, idx);
            let treated_b = take_rows_vec(treated, idx);
            let weights_b =
                fit_synthetic_control_weights(&donors_b, &treated_b, self.max_iterations)?;
            out.row_mut(i).assign(&weights_b);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}
