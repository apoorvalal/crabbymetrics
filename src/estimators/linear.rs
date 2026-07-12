use crate::hyptests::wald_test_arrays;
use crate::rla::sketch_ols_params;
use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, invert_matrix, pyarray1_from_f64,
    pyarray2_from_f64, sandwich_cov_from_parameter_scores, scale_rows, scale_vec,
    solve_least_squares_vec, sqrt_sample_weight, take_rows, take_rows_u32, take_rows_vec,
    to_array1, to_array1_i64, to_array2, to_array2_u32, validate_sample_weight,
};
use ndarray::{s, Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, BTreeSet};
use within::{Solver as WithinSolver, SolverParams as WithinSolverParams};

struct FixedEffectsOlsFitResult {
    coef: Array1<f64>,
    x_resid: Array2<f64>,
    y_resid: Array1<f64>,
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent != value {
            self.parent[value] = self.find(parent);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root != right_root {
            self.parent[right_root] = left_root;
        }
    }
}

fn observed_level_map(fe: &Array2<u32>, column: usize) -> BTreeMap<u32, usize> {
    let levels: BTreeSet<u32> = fe.column(column).iter().copied().collect();
    levels
        .into_iter()
        .enumerate()
        .map(|(index, level)| (level, index))
        .collect()
}

fn absorbed_fe_rank(fe: &Array2<u32>) -> Result<(usize, &'static str), String> {
    match fe.ncols() {
        0 => Err("fe must have at least one column".to_string()),
        1 => Ok((observed_level_map(fe, 0).len(), "exact_one_way")),
        2 => {
            let first = observed_level_map(fe, 0);
            let second = observed_level_map(fe, 1);
            let offset = first.len();
            let mut components = UnionFind::new(first.len() + second.len());
            for row in 0..fe.nrows() {
                components.union(first[&fe[[row, 0]]], offset + second[&fe[[row, 1]]]);
            }
            let n_components = (0..(first.len() + second.len()))
                .map(|index| components.find(index))
                .collect::<BTreeSet<_>>()
                .len();
            Ok((first.len() + second.len() - n_components, "exact_two_way"))
        }
        n_dimensions => {
            let total_levels: usize = (0..n_dimensions)
                .map(|column| observed_level_map(fe, column).len())
                .sum();
            Ok((
                total_levels.saturating_sub(n_dimensions - 1),
                "conservative_multiway",
            ))
        }
    }
}

pub(crate) fn split_params(params: &Array1<f64>, fit_intercept: bool) -> (f64, Array1<f64>) {
    if fit_intercept {
        (params[0], params.slice(s![1..]).to_owned())
    } else {
        (0.0, params.clone())
    }
}

pub(crate) fn apply_sqrt_weights(
    design: &Array2<f64>,
    values: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    let sqrt_weight = sqrt_sample_weight(sample_weight, design.nrows())?;
    match sqrt_weight.as_ref() {
        Some(scale) => Ok((scale_rows(design, scale)?, scale_vec(values, scale)?)),
        None => Ok((design.clone(), values.clone())),
    }
}

fn fit_linear_params_from_design(
    design: &Array2<f64>,
    y: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let (design_work, y_work) = apply_sqrt_weights(design, y, sample_weight)?;
    solve_least_squares_vec(&design_work, &y_work)
}

fn fit_linear_params(
    x: &Array2<f64>,
    y: &Array1<f64>,
    fit_intercept: bool,
    sample_weight: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let design = if fit_intercept {
        add_intercept(x)
    } else {
        x.clone()
    };
    fit_linear_params_from_design(&design, y, sample_weight)
}

fn linear_parameter_scores(
    design: &Array2<f64>,
    residuals: &Array1<f64>,
    bread: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if design.nrows() != residuals.len() {
        return Err("residual length mismatch".to_string());
    }

    let mut raw_scores = design.clone();
    for i in 0..design.nrows() {
        raw_scores
            .row_mut(i)
            .mapv_inplace(|value| value * residuals[i]);
    }
    Ok(raw_scores.dot(bread))
}

fn ols_hc_covariance(
    design: &Array2<f64>,
    residuals: &Array1<f64>,
    kind: &str,
    residual_df: f64,
) -> Result<Array2<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if residuals.len() != n {
        return Err("residual length mismatch".to_string());
    }
    if n <= p {
        return Err("need more observations than regressors".to_string());
    }

    let xtx = design.t().dot(design);
    let xtx_inv = invert_matrix(&xtx)?;
    let mut weighted_design = design.clone();
    for i in 0..n {
        let leverage = design.row(i).dot(&xtx_inv.dot(&design.row(i).to_owned()));
        let denom = (1.0 - leverage).max(1e-12);
        let mut weight = residuals[i] * residuals[i];
        match kind {
            "hc0" => {}
            "hc1" => weight *= n as f64 / residual_df,
            "hc2" => weight /= denom,
            "hc3" => weight /= denom * denom,
            _ => return Err("vcov must be one of {'hc0', 'hc1', 'hc2', 'hc3'}".to_string()),
        }
        weighted_design
            .row_mut(i)
            .mapv_inplace(|value| value * weight);
    }
    let meat = design.t().dot(&weighted_design);
    Ok(xtx_inv.dot(&meat).dot(&xtx_inv))
}

pub(crate) fn linear_covariance(
    design: &Array2<f64>,
    residuals: &Array1<f64>,
    vcov: &str,
    lags: Option<usize>,
    clusters: Option<&Array1<i64>>,
    residual_df: Option<f64>,
) -> Result<Array2<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if residuals.len() != n {
        return Err("residual length mismatch".to_string());
    }
    let df_resid = residual_df.unwrap_or(n as f64 - p as f64);
    if df_resid <= 0.0 {
        return Err("need positive residual degrees of freedom".to_string());
    }

    match vcov {
        "vanilla" => {
            let xtx_inv = invert_matrix(&design.t().dot(design))?;
            let sigma2 = residuals.dot(residuals) / df_resid;
            Ok(xtx_inv.mapv(|value| value * sigma2))
        }
        "hc0" | "hc1" | "hc2" | "hc3" => ols_hc_covariance(design, residuals, vcov, df_resid),
        "newey_west" | "cluster" => {
            let xtx = design.t().dot(design);
            let bread = invert_matrix(&xtx)?;
            let param_scores = linear_parameter_scores(design, residuals, &bread)?;
            sandwich_cov_from_parameter_scores(&param_scores, vcov, df_resid, lags, clusters)
        }
        _ => Err(
            "vcov must be one of {'vanilla', 'hc0', 'hc1', 'hc2', 'hc3', 'newey_west', 'cluster'}"
                .to_string(),
        ),
    }
}

fn av_t_radius(g: f64, n: usize, number_of_coefficients: usize, alpha: f64) -> PyResult<f64> {
    if g <= 0.0 || !g.is_finite() {
        return Err(PyValueError::new_err("g must be positive and finite"));
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(PyValueError::new_err("alpha must be in (0, 1)"));
    }
    if n <= number_of_coefficients {
        return Err(PyValueError::new_err(
            "n must be greater than number_of_coefficients",
        ));
    }

    let nu = (n - number_of_coefficients) as f64;
    let d = 1.0;
    let t = g / (g + n as f64);
    let powered = (t * alpha.powi(2)).powf(1.0 / (nu + d));
    let denominator = powered - t;
    if denominator <= 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok((nu * (1.0 - powered) / denominator).sqrt())
}

fn av_log_g_t(t2: f64, nu: f64, n: usize, g: f64) -> f64 {
    let r = g / (g + n as f64);
    0.5 * r.ln() + 0.5 * (nu + 1.0) * ((1.0 + t2 / nu).ln() - (1.0 + r * t2 / nu).ln())
}

fn av_p_from_log_g(log_g: f64) -> f64 {
    (-log_g).exp().min(1.0)
}

fn av_log_g_f(f: f64, d: f64, nu: f64, n: usize, g: f64) -> f64 {
    let r = g / (g + n as f64);
    0.5 * d * r.ln() + 0.5 * (nu + d) * ((1.0 + (d / nu) * f).ln() - (1.0 + r * (d / nu) * f).ln())
}

fn fit_fixed_effects_ols(
    x: &Array2<f64>,
    y: &Array1<f64>,
    fe: &Array2<u32>,
    sample_weight: Option<&Array1<f64>>,
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
    if let Some(weights) = sample_weight {
        validate_sample_weight(weights, y.len()).map_err(PyValueError::new_err)?;
    }

    let params = WithinSolverParams::default();
    let solver = WithinSolver::new(
        fe.view(),
        sample_weight.and_then(|weights| weights.as_slice()),
        &params,
        None,
    )
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

    let coef = fit_linear_params_from_design(&x_resid, &y_resid, sample_weight)
        .map_err(PyValueError::new_err)?;

    Ok(FixedEffectsOlsFitResult {
        coef,
        x_resid,
        y_resid,
    })
}

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

    #[pyo3(signature = (vcov="hc1", lags=None, clusters=None, anytime_valid=false, g=1.0, level=0.95))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
        anytime_valid: bool,
        g: f64,
        level: f64,
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
        let vcov_normalized = vcov.to_ascii_lowercase();
        let cov = linear_covariance(
            &design_work,
            &residuals_work,
            &vcov_normalized,
            lags,
            cluster_ids.as_ref(),
            None,
        )
        .map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov).map_err(PyValueError::new_err)?;

        let (intercept, coef, intercept_se, coef_se) = if self.fit_intercept {
            (
                self.intercept,
                coef.to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, coef.to_owned(), None, se_all.clone())
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, &coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        dict.set_item("vcov_type", vcov_normalized.as_str())?;
        dict.set_item("anytime_valid", anytime_valid)?;

        if anytime_valid {
            if !(0.0..1.0).contains(&level) {
                return Err(PyValueError::new_err("level must be in (0, 1)"));
            }
            let n = design.nrows();
            let p = design.ncols();
            let nu = (n - p) as f64;
            let t_values = &params / &se_all;
            let p_values = t_values.mapv(|t| av_p_from_log_g(av_log_g_t(t * t, nu, n, g)));
            let radius = av_t_radius(g, n, params.len(), 1.0 - level)?;
            let mut confint = Array2::<f64>::zeros((params.len(), 2));
            for i in 0..params.len() {
                confint[[i, 0]] = params[i] - radius * se_all[i];
                confint[[i, 1]] = params[i] + radius * se_all[i];
            }

            dict.set_item("estimate", pyarray1_from_f64(py, &params))?;
            dict.set_item("std_error", pyarray1_from_f64(py, &se_all))?;
            dict.set_item("t_value", pyarray1_from_f64(py, &t_values))?;
            dict.set_item("p_value", pyarray1_from_f64(py, &p_values))?;
            dict.set_item("confint", pyarray2_from_f64(py, &confint))?;
            dict.set_item("confint_level", level)?;
            dict.set_item("g", g)?;

            let d = if self.fit_intercept { p - 1 } else { p };
            if d > 0 {
                let start = if self.fit_intercept { 1 } else { 0 };
                let beta = params.slice(s![start..]).to_owned();
                let f_value = if vcov_normalized == "vanilla" {
                    let cov_sub = cov.slice(s![start.., start..]).to_owned();
                    let precision = invert_matrix(&cov_sub).map_err(PyValueError::new_err)?;
                    beta.dot(&precision.dot(&beta)) / d as f64
                } else {
                    let precision = invert_matrix(&cov).map_err(PyValueError::new_err)?;
                    let precision_sub = precision.slice(s![start.., start..]).to_owned();
                    beta.dot(&precision_sub.dot(&beta))
                };
                let f_p_value = av_p_from_log_g(av_log_g_f(f_value, d as f64, nu, n, g));
                dict.set_item("f_statistic", f_value)?;
                dict.set_item("f_p_value", f_p_value)?;
                dict.set_item("df_model", d)?;
                dict.set_item("df_resid", n - p)?;
            }
        }

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
            .ok_or_else(|| PyValueError::new_err("No response stored"))?;
        let vcov = vcov.unwrap_or("hc1");
        let mut params = Array1::<f64>::zeros(coef.len() + if self.fit_intercept { 1 } else { 0 });
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
        let residuals = y - &design.dot(&params);
        let (design_work, residuals_work) =
            apply_sqrt_weights(&design, &residuals, self.sample_weight.as_ref())
                .map_err(PyValueError::new_err)?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = linear_covariance(
            &design_work,
            &residuals_work,
            vcov,
            lags,
            cluster_ids.as_ref(),
            None,
        )
        .map_err(PyValueError::new_err)?;
        let rmat = to_array2(&r);
        let qvec = q.as_ref().map(to_array1);
        wald_test_arrays(py, &params, &cov, &rmat, qvec.as_ref())
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

#[pyfunction]
#[pyo3(signature = (model, g=1.0, vcov="vanilla", level=0.95))]
pub fn av<'py>(
    py: Python<'py>,
    model: PyRef<'_, OLS>,
    g: f64,
    vcov: &str,
    level: f64,
) -> PyResult<Py<PyAny>> {
    model.summary(py, vcov, None, None, true, g, level)
}

#[pyfunction]
pub fn optimal_g(n: usize, number_of_coefficients: usize, alpha: f64) -> PyResult<f64> {
    if n <= number_of_coefficients {
        return Err(PyValueError::new_err(
            "n must be greater than number_of_coefficients",
        ));
    }
    if !(0.0..1.0).contains(&alpha) {
        return Err(PyValueError::new_err("alpha must be in (0, 1)"));
    }

    let nu = (n - number_of_coefficients) as f64;
    let upper = n as f64 * alpha.powf(2.0 / nu) / (1.0 - alpha.powf(2.0 / nu));
    let lower = 1.0_f64;
    if upper <= lower || !upper.is_finite() {
        return Ok(lower);
    }

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let resphi = 2.0 - phi;
    let mut a = lower;
    let mut b = upper;
    let mut c = a + resphi * (b - a);
    let mut d = b - resphi * (b - a);
    let mut fc = av_t_radius(c, n, number_of_coefficients, alpha)?;
    let mut fd = av_t_radius(d, n, number_of_coefficients, alpha)?;

    for _ in 0..160 {
        if (b - a).abs() <= 1e-10 * (1.0 + c.abs() + d.abs()) {
            break;
        }
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = a + resphi * (b - a);
            fc = av_t_radius(c, n, number_of_coefficients, alpha)?;
        } else {
            a = c;
            c = d;
            fc = fd;
            d = b - resphi * (b - a);
            fd = av_t_radius(d, n, number_of_coefficients, alpha)?;
        }
    }

    Ok(0.5 * (a + b))
}

#[pyclass]
pub struct FixedEffectsOLS {
    coef: Option<Array1<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
    fe: Option<Array2<u32>>,
    sample_weight: Option<Array1<f64>>,
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
            sample_weight: None,
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
        let fit = fit_fixed_effects_ols(&x, &y, &fe, None)?;

        self.coef = Some(fit.coef);
        self.x = Some(x);
        self.y = Some(y);
        self.fe = Some(fe);
        self.sample_weight = None;
        self.x_resid = Some(fit.x_resid);
        self.y_resid = Some(fit.y_resid);
        Ok(())
    }

    fn fit_weighted(
        &mut self,
        x: PyReadonlyArray2<f64>,
        fe: PyReadonlyArray2<u32>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Vec<f64>,
    ) -> PyResult<()> {
        let x = to_array2(&x);
        let fe = to_array2_u32(&fe);
        let y = to_array1(&y);
        let sample_weight = Array1::from_vec(sample_weight);
        let fit = fit_fixed_effects_ols(&x, &y, &fe, Some(&sample_weight))?;

        self.coef = Some(fit.coef);
        self.x = Some(x);
        self.y = Some(y);
        self.fe = Some(fe);
        self.sample_weight = Some(sample_weight);
        self.x_resid = Some(fit.x_resid);
        self.y_resid = Some(fit.y_resid);
        Ok(())
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
            .ok_or_else(|| PyValueError::new_err("FixedEffectsOLS model is not fitted"))?;
        let x_resid = self
            .x_resid
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No residualized design stored"))?;
        let y_resid = self
            .y_resid
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No residualized response stored"))?;
        let fe = self
            .fe
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fixed effects stored"))?;
        let (absorbed_df, absorbed_df_method) =
            absorbed_fe_rank(fe).map_err(PyValueError::new_err)?;
        let residual_df = y_resid.len() as f64 - x_resid.ncols() as f64 - absorbed_df as f64;
        if residual_df <= 0.0 {
            return Err(PyValueError::new_err(
                "absorbed fixed effects leave no residual degrees of freedom",
            ));
        }
        let sample_weight = self.sample_weight.as_ref();

        let residuals = y_resid - &x_resid.dot(coef);
        let (design_work, residuals_work) = apply_sqrt_weights(x_resid, &residuals, sample_weight)
            .map_err(PyValueError::new_err)?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = linear_covariance(
            &design_work,
            &residuals_work,
            vcov,
            lags,
            cluster_ids.as_ref(),
            Some(residual_df),
        )
        .map_err(PyValueError::new_err)?;
        let coef_se = diag_sqrt(&cov).map_err(PyValueError::new_err)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        dict.set_item("vcov_type", vcov)?;
        dict.set_item("absorbed_df", absorbed_df)?;
        dict.set_item("residual_df", residual_df)?;
        dict.set_item("absorbed_df_method", absorbed_df_method)?;
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
        let fe = self
            .fe
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fixed effects stored"))?;
        let (absorbed_df, _) = absorbed_fe_rank(fe).map_err(PyValueError::new_err)?;
        let residual_df = y_resid.len() as f64 - x_resid.ncols() as f64 - absorbed_df as f64;
        if residual_df <= 0.0 {
            return Err(PyValueError::new_err(
                "absorbed fixed effects leave no residual degrees of freedom",
            ));
        }
        let vcov = vcov.unwrap_or("hc1");
        let residuals = y_resid - &x_resid.dot(coef);
        let (design_work, residuals_work) =
            apply_sqrt_weights(x_resid, &residuals, self.sample_weight.as_ref())
                .map_err(PyValueError::new_err)?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = linear_covariance(
            &design_work,
            &residuals_work,
            vcov,
            lags,
            cluster_ids.as_ref(),
            Some(residual_df),
        )
        .map_err(PyValueError::new_err)?;
        let rmat = to_array2(&r);
        let qvec = q.as_ref().map(to_array1);
        wald_test_arrays(py, coef, &cov, &rmat, qvec.as_ref())
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
        let sample_weight = self.sample_weight.as_ref();

        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((n_bootstrap, x.ncols()));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let feb = take_rows_u32(fe, idx);
            let wb = sample_weight.map(|weights| take_rows_vec(weights, idx));
            let fit = fit_fixed_effects_ols(&xb, &yb, &feb, wb.as_ref())?;
            out.row_mut(i).assign(&fit.coef);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}
