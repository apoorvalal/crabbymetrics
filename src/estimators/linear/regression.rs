use super::*;
#[pyclass]
pub struct OLS {
    fit_intercept: bool,
    intercept: f64,
    coef: Option<Array1<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
    sample_weight: Option<Array1<f64>>,
}

#[pymethods]
impl OLS {
    #[new]
    fn new() -> Self {
        Self {
            fit_intercept: true,
            intercept: 0.0,
            coef: None,
            x: None,
            y: None,
            sample_weight: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let params =
            fit_linear_params(&x, &y, self.fit_intercept, None).map_err(PyValueError::new_err)?;
        let (intercept, coef) = split_params(&params, self.fit_intercept);
        self.intercept = intercept;
        self.coef = Some(coef);
        self.x = Some(x);
        self.y = Some(y);
        self.sample_weight = None;
        Ok(())
    }

    fn fit_weighted(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Vec<f64>,
    ) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let sample_weight = Array1::from_vec(sample_weight);
        let params = fit_linear_params(&x, &y, self.fit_intercept, Some(&sample_weight))
            .map_err(PyValueError::new_err)?;
        let (intercept, coef) = split_params(&params, self.fit_intercept);
        self.intercept = intercept;
        self.coef = Some(coef);
        self.x = Some(x);
        self.y = Some(y);
        self.sample_weight = Some(sample_weight);
        Ok(())
    }

    #[pyo3(signature = (x, y, sketch_size, seed=None))]
    fn fit_sketch(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sketch_size: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let params = sketch_ols_params(&x, &y, self.fit_intercept, sketch_size, seed)?;
        let (intercept, coef) = split_params(&params, self.fit_intercept);
        self.intercept = intercept;
        self.coef = Some(coef);
        self.x = Some(x);
        self.y = Some(y);
        self.sample_weight = None;
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
            .ok_or_else(|| PyValueError::new_err("OLS model is not fitted"))?;
        let x = to_array2(&x);
        let pred = x.dot(coef) + self.intercept;
        Ok(pyarray1_from_f64(py, &pred))
    }

    #[pyo3(signature = (vcov="hc1", lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
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
        let sample_weight = self.sample_weight.as_ref();

        let mut params = Array1::<f64>::zeros(x.ncols() + if self.fit_intercept { 1 } else { 0 });
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        if self.fit_intercept {
            params[0] = self.intercept;
            params.slice_mut(s![1..]).assign(coef);
        } else {
            params.assign(coef);
        }
        let fitted = design.dot(&params);
        let residuals = y - &fitted;
        let (design_work, residuals_work) = apply_sqrt_weights(&design, &residuals, sample_weight)
            .map_err(PyValueError::new_err)?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = linear_covariance(
            &design_work,
            &residuals_work,
            vcov,
            lags,
            cluster_ids.as_ref(),
        )
        .map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept, coef, intercept_se, coef_se) = if self.fit_intercept {
            (
                self.intercept,
                coef.to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, coef.to_owned(), None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, &coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        dict.set_item("vcov_type", vcov)?;
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
        let sample_weight = self.sample_weight.as_ref();
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((
            n_bootstrap,
            x.ncols() + if self.fit_intercept { 1 } else { 0 },
        ));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let wb = sample_weight.map(|weights| take_rows_vec(weights, idx));
            let params = fit_linear_params(&xb, &yb, self.fit_intercept, wb.as_ref())
                .map_err(PyValueError::new_err)?;
            if self.fit_intercept {
                out[[i, 0]] = params[0];
                out.row_mut(i)
                    .slice_mut(s![1..])
                    .assign(&params.slice(s![1..]));
            } else {
                out.row_mut(i).assign(&params);
            }
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
    sample_weight: Option<Array1<f64>>,
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
            sample_weight: None,
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

        let fit = fit_two_sls_closed_form(&x_endog, &x_exog, &z, &y, self.fit_intercept, None)?;
        let (intercept, coef) = split_params(&fit.params, self.fit_intercept);

        self.coef = Some(coef);
        self.intercept = intercept;
        self.x_endog = Some(x_endog);
        self.x_exog = Some(x_exog);
        self.z = Some(z);
        self.y = Some(y);
        self.sample_weight = None;
        Ok(())
    }

    fn fit_weighted(
        &mut self,
        x_endog: PyReadonlyArray2<f64>,
        x_exog: PyReadonlyArray2<f64>,
        z: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Vec<f64>,
    ) -> PyResult<()> {
        let x_endog = to_array2(&x_endog);
        let x_exog = to_array2(&x_exog);
        let z = to_array2(&z);
        let y = to_array1(&y);
        let sample_weight = Array1::from_vec(sample_weight);

        let fit = fit_two_sls_closed_form(
            &x_endog,
            &x_exog,
            &z,
            &y,
            self.fit_intercept,
            Some(&sample_weight),
        )?;
        let (intercept, coef) = split_params(&fit.params, self.fit_intercept);

        self.coef = Some(coef);
        self.intercept = intercept;
        self.x_endog = Some(x_endog);
        self.x_exog = Some(x_exog);
        self.z = Some(z);
        self.y = Some(y);
        self.sample_weight = Some(sample_weight);
        Ok(())
    }

    #[pyo3(signature = (x_endog, x_exog, z, y, sketch_size, seed=None))]
    fn fit_sketch(
        &mut self,
        x_endog: PyReadonlyArray2<f64>,
        x_exog: PyReadonlyArray2<f64>,
        z: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sketch_size: usize,
        seed: Option<u64>,
    ) -> PyResult<()> {
        let x_endog = to_array2(&x_endog);
        let x_exog = to_array2(&x_exog);
        let z = to_array2(&z);
        let y = to_array1(&y);
        let fit = fit_two_sls_sketch(
            &x_endog,
            &x_exog,
            &z,
            &y,
            self.fit_intercept,
            sketch_size,
            seed,
        )?;
        let (intercept, coef) = split_params(&fit.params, self.fit_intercept);

        self.coef = Some(coef);
        self.intercept = intercept;
        self.x_endog = Some(x_endog);
        self.x_exog = Some(x_exog);
        self.z = Some(z);
        self.y = Some(y);
        self.sample_weight = None;
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

    #[pyo3(signature = (vcov="hc1", lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let fitted_coef = self
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
        let sample_weight = self.sample_weight.as_ref();

        let mut params =
            Array1::<f64>::zeros(fitted_coef.len() + if self.fit_intercept { 1 } else { 0 });
        if self.fit_intercept {
            params[0] = self.intercept;
            params.slice_mut(s![1..]).assign(fitted_coef);
        } else {
            params.assign(fitted_coef);
        }
        let (x_design_work, z_design_work, y_work) = match sample_weight {
            Some(weights) => {
                let sqrt_weight = sqrt_sample_weight(Some(weights), y.len())
                    .map_err(PyValueError::new_err)?
                    .expect("weights were provided");
                let x_endog_work =
                    scale_rows(x_endog, &sqrt_weight).map_err(PyValueError::new_err)?;
                let x_exog_work =
                    scale_rows(x_exog, &sqrt_weight).map_err(PyValueError::new_err)?;
                let z_work = scale_rows(z, &sqrt_weight).map_err(PyValueError::new_err)?;
                let y_work = scale_vec(y, &sqrt_weight).map_err(PyValueError::new_err)?;
                let (x_design_work, z_design_work) =
                    build_iv_designs(&x_endog_work, &x_exog_work, &z_work, self.fit_intercept)?;
                (x_design_work, z_design_work, y_work)
            }
            None => {
                let (x_design, z_design) =
                    build_iv_designs(x_endog, x_exog, z, self.fit_intercept)?;
                (x_design, z_design, y.clone())
            }
        };
        let residuals = y_work - &x_design_work.dot(&params);
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = twosls_covariance(
            &x_design_work,
            &z_design_work,
            &residuals,
            vcov,
            lags,
            cluster_ids.as_ref(),
        )
        .map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept_se, coef_se) = if self.fit_intercept {
            (Some(se_all[0]), se_all.slice(s![1..]).to_owned())
        } else {
            (None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, fitted_coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        dict.set_item("vcov_type", vcov)?;
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
        let sample_weight = self.sample_weight.as_ref();
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
            let wb = sample_weight.map(|weights| take_rows_vec(weights, idx));

            let fit = fit_two_sls_closed_form(
                &x_endog_b,
                &x_exog_b,
                &z_b,
                &yb,
                self.fit_intercept,
                wb.as_ref(),
            )?;
            if self.fit_intercept {
                out[[i, 0]] = fit.params[0];
                out.row_mut(i)
                    .slice_mut(s![1..])
                    .assign(&fit.params.slice(s![1..]));
            } else {
                out.row_mut(i).assign(&fit.params);
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}
