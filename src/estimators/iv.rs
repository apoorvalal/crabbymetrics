use super::linear::{linear_covariance, split_params};
use crate::hyptests::{f_sf, wald_test_arrays};
use crate::rla::count_sketch_joint;
use crate::utils::{
    add_intercept, apply_sqrt_weights, bootstrap_indices, diag_sqrt, invert_matrix,
    pyarray1_from_f64, pyarray2_from_f64, sandwich_cov_from_parameter_scores, scale_rows,
    scale_vec, solve_least_squares_mat, solve_least_squares_vec, sqrt_sample_weight, take_rows,
    take_rows_vec, to_array1, to_array1_i64, to_array2,
};
use ndarray::{concatenate, s, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct TwoSlsFitResult {
    params: Array1<f64>,
}

fn combine_endog_exog(x_endog: &Array2<f64>, x_exog: &Array2<f64>) -> PyResult<Array2<f64>> {
    if x_endog.ncols() == 0 {
        return Err(PyValueError::new_err(
            "x_endog must have at least one column",
        ));
    }

    if x_exog.ncols() > 0 {
        concatenate(Axis(1), &[x_endog.view(), x_exog.view()])
            .map_err(|_| PyValueError::new_err("failed to concat endog/exog"))
    } else {
        Ok(x_endog.clone())
    }
}

fn build_iv_designs(
    x_endog: &Array2<f64>,
    x_exog: &Array2<f64>,
    z: &Array2<f64>,
    fit_intercept: bool,
) -> PyResult<(Array2<f64>, Array2<f64>)> {
    if x_endog.nrows() != x_exog.nrows() || x_endog.nrows() != z.nrows() {
        return Err(PyValueError::new_err("row count mismatch"));
    }
    if z.ncols() < x_endog.ncols() {
        return Err(PyValueError::new_err(
            "need at least as many excluded instruments as endogenous regressors",
        ));
    }

    let x_rhs = combine_endog_exog(x_endog, x_exog)?;
    let z_rhs = if x_exog.ncols() > 0 {
        concatenate(Axis(1), &[x_exog.view(), z.view()])
            .map_err(|_| PyValueError::new_err("failed to concat instruments"))?
    } else {
        z.clone()
    };

    let x_design = if fit_intercept {
        add_intercept(&x_rhs)
    } else {
        x_rhs
    };
    let z_design = if fit_intercept {
        add_intercept(&z_rhs)
    } else {
        z_rhs
    };

    if z_design.ncols() < x_design.ncols() {
        return Err(PyValueError::new_err(
            "model is underidentified: instrument count is smaller than regressor count",
        ));
    }

    Ok((x_design, z_design))
}

fn twosls_covariance(
    x_design: &Array2<f64>,
    z_design: &Array2<f64>,
    residuals: &Array1<f64>,
    vcov: &str,
    lags: Option<usize>,
    clusters: Option<&Array1<i64>>,
) -> Result<Array2<f64>, String> {
    let n = x_design.nrows();
    let p = x_design.ncols();
    if residuals.len() != n || z_design.nrows() != n {
        return Err("residual length mismatch".to_string());
    }

    let n_f64 = n as f64;
    let weight_base = z_design.t().dot(z_design) / n_f64;
    let weight = invert_matrix(&weight_base)?;
    let jacobian = -(z_design.t().dot(x_design)) / n_f64;
    let a_matrix = jacobian.t().dot(&weight).dot(&jacobian);
    let a_inv = invert_matrix(&a_matrix)?;

    match vcov {
        "vanilla" => {
            if n <= p {
                return Err("need more observations than regressors".to_string());
            }
            let sigma2 = residuals.dot(residuals) / ((n - p) as f64);
            Ok(a_inv.mapv(|value| value * sigma2 / n_f64))
        }
        "hc1" | "newey_west" | "cluster" => {
            let mut moment_scores = z_design.clone();
            for i in 0..n {
                let scale = residuals[i];
                moment_scores.row_mut(i).mapv_inplace(|value| value * scale);
            }
            let transform = weight.dot(&jacobian).dot(&a_inv) / n_f64;
            let param_scores = moment_scores.dot(&transform);
            sandwich_cov_from_parameter_scores(
                &param_scores,
                vcov,
                n_f64 - p as f64,
                lags,
                clusters,
            )
        }
        _ => Err("vcov must be one of {'hc1', 'vanilla', 'newey_west', 'cluster'}".to_string()),
    }
}

fn fit_two_sls_sketch(
    x_endog: &Array2<f64>,
    x_exog: &Array2<f64>,
    z: &Array2<f64>,
    y: &Array1<f64>,
    fit_intercept: bool,
    sketch_size: usize,
    seed: Option<u64>,
) -> PyResult<TwoSlsFitResult> {
    if x_endog.nrows() != y.len() || x_exog.nrows() != y.len() || z.nrows() != y.len() {
        return Err(PyValueError::new_err("row count mismatch"));
    }
    let x_rhs = combine_endog_exog(x_endog, x_exog)?;
    let z_rhs = if x_exog.ncols() > 0 {
        concatenate(Axis(1), &[x_exog.view(), z.view()])
            .map_err(|_| PyValueError::new_err("failed to concat instruments"))?
    } else {
        z.clone()
    };
    let x_design = if fit_intercept {
        add_intercept(&x_rhs)
    } else {
        x_rhs
    };
    let z_design = if fit_intercept {
        add_intercept(&z_rhs)
    } else {
        z_rhs
    };
    if z_design.ncols() < x_design.ncols() {
        return Err(PyValueError::new_err(
            "model is underidentified: instrument count is smaller than regressor count",
        ));
    }
    if sketch_size < z_design.ncols().max(x_design.ncols()) {
        return Err(PyValueError::new_err(
            "sketch_size must be at least the larger of design and instrument columns",
        ));
    }

    let (sketched_mats, sketched_vecs) =
        count_sketch_joint(&[&x_design, &z_design], &[y], sketch_size, seed)?;
    let sx = &sketched_mats[0];
    let sz = &sketched_mats[1];
    let sy = &sketched_vecs[0];
    let x_hat = solve_least_squares_mat(sz, sx)
        .map(|pi_hat| sz.dot(&pi_hat))
        .map_err(PyValueError::new_err)?;
    let params = solve_least_squares_vec(&x_hat, sy).map_err(PyValueError::new_err)?;
    Ok(TwoSlsFitResult { params })
}

fn fit_two_sls_closed_form(
    x_endog: &Array2<f64>,
    x_exog: &Array2<f64>,
    z: &Array2<f64>,
    y: &Array1<f64>,
    fit_intercept: bool,
    sample_weight: Option<&Array1<f64>>,
) -> PyResult<TwoSlsFitResult> {
    if x_endog.nrows() != y.len() || x_exog.nrows() != y.len() || z.nrows() != y.len() {
        return Err(PyValueError::new_err("row count mismatch"));
    }

    let (x_design, z_design) = build_iv_designs(x_endog, x_exog, z, fit_intercept)?;
    let sqrt_weight = sqrt_sample_weight(sample_weight, y.len()).map_err(PyValueError::new_err)?;
    let (x_work, z_work, y_work) = match sqrt_weight.as_ref() {
        Some(scale) => (
            scale_rows(&x_design, scale).map_err(PyValueError::new_err)?,
            scale_rows(&z_design, scale).map_err(PyValueError::new_err)?,
            scale_vec(y, scale).map_err(PyValueError::new_err)?,
        ),
        None => (x_design, z_design, y.clone()),
    };

    let x_hat = solve_least_squares_mat(&z_work, &x_work)
        .map(|pi_hat| z_work.dot(&pi_hat))
        .map_err(PyValueError::new_err)?;
    let params = solve_least_squares_vec(&x_hat, &y_work).map_err(PyValueError::new_err)?;

    Ok(TwoSlsFitResult { params })
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
        let (x_design, z_design) = build_iv_designs(x_endog, x_exog, z, self.fit_intercept)?;
        let sqrt_weight =
            sqrt_sample_weight(sample_weight, y.len()).map_err(PyValueError::new_err)?;
        let (x_design_work, z_design_work, y_work) = match sqrt_weight.as_ref() {
            Some(scale) => (
                scale_rows(&x_design, scale).map_err(PyValueError::new_err)?,
                scale_rows(&z_design, scale).map_err(PyValueError::new_err)?,
                scale_vec(y, scale).map_err(PyValueError::new_err)?,
            ),
            None => (x_design, z_design, y.clone()),
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
        let se_all = diag_sqrt(&cov).map_err(PyValueError::new_err)?;

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
        dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        dict.set_item("vcov_type", vcov)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (r, q=None, vcov=None, lags=None, clusters=None))]
    fn wald_test<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<f64>,
        q: Option<PyReadonlyArray1<f64>>,
        vcov: Option<&str>,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
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
        let vcov = vcov.unwrap_or("hc1");
        let mut params =
            Array1::<f64>::zeros(fitted_coef.len() + if self.fit_intercept { 1 } else { 0 });
        if self.fit_intercept {
            params[0] = self.intercept;
            params.slice_mut(s![1..]).assign(fitted_coef);
        } else {
            params.assign(fitted_coef);
        }
        let (x_design, z_design) = build_iv_designs(x_endog, x_exog, z, self.fit_intercept)?;
        let sqrt_weight = sqrt_sample_weight(self.sample_weight.as_ref(), y.len())
            .map_err(PyValueError::new_err)?;
        let (x_design_work, z_design_work, y_work) = match sqrt_weight.as_ref() {
            Some(scale) => (
                scale_rows(&x_design, scale).map_err(PyValueError::new_err)?,
                scale_rows(&z_design, scale).map_err(PyValueError::new_err)?,
                scale_vec(y, scale).map_err(PyValueError::new_err)?,
            ),
            None => (x_design, z_design, y.clone()),
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
        let rmat = to_array2(&r);
        let qvec = q.as_ref().map(to_array1);
        wald_test_arrays(py, &params, &cov, &rmat, qvec.as_ref())
    }

    #[pyo3(signature = (beta=0.0, vcov="hc1", lags=None, clusters=None))]
    fn anderson_rubin_test<'py>(
        &self,
        py: Python<'py>,
        beta: f64,
        vcov: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let x_endog = self
            .x_endog
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("TwoSLS model is not fitted"))?;
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

        if x_endog.ncols() != 1 {
            return Err(PyValueError::new_err(
                "anderson_rubin_test currently supports exactly one endogenous regressor",
            ));
        }
        if !beta.is_finite() {
            return Err(PyValueError::new_err("beta must be finite"));
        }

        let y_null = y - &(x_endog.column(0).to_owned() * beta);
        let z_rhs = if x_exog.ncols() > 0 {
            concatenate(Axis(1), &[x_exog.view(), z.view()])
                .map_err(|_| PyValueError::new_err("failed to concat instruments"))?
        } else {
            z.clone()
        };
        let design = if self.fit_intercept {
            add_intercept(&z_rhs)
        } else {
            z_rhs
        };
        if design.nrows() <= design.ncols() {
            return Err(PyValueError::new_err(
                "not enough residual degrees of freedom for Anderson Rubin test",
            ));
        }

        let (design_work, y_work) =
            apply_sqrt_weights(&design, &y_null, self.sample_weight.as_ref())
                .map_err(PyValueError::new_err)?;
        let params =
            solve_least_squares_vec(&design_work, &y_work).map_err(PyValueError::new_err)?;
        let residuals = y_work - &design_work.dot(&params);
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = linear_covariance(
            &design_work,
            &residuals,
            vcov,
            lags,
            cluster_ids.as_ref(),
            None,
        )
        .map_err(PyValueError::new_err)?;

        let df1 = z.ncols();
        let df2 = design_work.nrows() - design_work.ncols();
        let start = if self.fit_intercept { 1 } else { 0 } + x_exog.ncols();
        let mut rmat = Array2::<f64>::zeros((df1, params.len()));
        for j in 0..df1 {
            rmat[[j, start + j]] = 1.0;
        }
        let diff = rmat.dot(&params);
        let rcov = rmat.dot(&cov).dot(&rmat.t());
        let rcov_inv = invert_matrix(&rcov).map_err(PyValueError::new_err)?;
        let tmp = rcov_inv.dot(&diff.clone().insert_axis(Axis(1)));
        let wald_statistic = diff.dot(&tmp.column(0)).max(0.0);
        let statistic = wald_statistic / df1 as f64;
        let p_value = f_sf(statistic, df1, df2)?;

        let out = PyDict::new(py);
        out.set_item("statistic", statistic)?;
        out.set_item("wald_statistic", wald_statistic)?;
        out.set_item("df_num", df1)?;
        out.set_item("df_denom", df2)?;
        out.set_item("p_value", p_value)?;
        out.set_item("beta", beta)?;
        out.set_item("vcov_type", vcov)?;
        out.set_item("test", "Anderson Rubin")?;
        Ok(out)
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
