use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, invert_matrix, ols_vanilla_cov, pyarray1_from_f64,
    pyarray2_from_f64, sandwich_cov_from_parameter_scores, scale_rows, scale_vec,
    solve_least_squares_mat, solve_least_squares_vec, sqrt_sample_weight, take_rows, take_rows_u32,
    take_rows_vec, to_array1, to_array1_i64, to_array2, to_array2_u32, validate_sample_weight,
};
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use within::{Solver as WithinSolver, SolverParams as WithinSolverParams};

struct FixedEffectsOlsFitResult {
    coef: Array1<f64>,
    x_resid: Array2<f64>,
    y_resid: Array1<f64>,
}

struct TwoSlsFitResult {
    params: Array1<f64>,
    x_design: Array2<f64>,
    z_design: Array2<f64>,
    residuals: Array1<f64>,
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

fn split_params(params: &Array1<f64>, fit_intercept: bool) -> (f64, Array1<f64>) {
    if fit_intercept {
        (params[0], params.slice(s![1..]).to_owned())
    } else {
        (0.0, params.clone())
    }
}

fn apply_sqrt_weights(
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

fn linear_covariance(
    design: &Array2<f64>,
    residuals: &Array1<f64>,
    vcov: &str,
    lags: Option<usize>,
    clusters: Option<&Array1<i64>>,
) -> Result<Array2<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if residuals.len() != n {
        return Err("residual length mismatch".to_string());
    }

    match vcov {
        "vanilla" => ols_vanilla_cov(design, residuals),
        "hc1" | "newey_west" | "cluster" => {
            let xtx = design.t().dot(design);
            let bread = invert_matrix(&xtx)?;
            let param_scores = linear_parameter_scores(design, residuals, &bread)?;
            sandwich_cov_from_parameter_scores(
                &param_scores,
                vcov,
                n as f64 - p as f64,
                lags,
                clusters,
            )
        }
        _ => Err("vcov must be one of {'hc1', 'vanilla', 'newey_west', 'cluster'}".to_string()),
    }
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

    let sqrt_weight = sqrt_sample_weight(sample_weight, y.len()).map_err(PyValueError::new_err)?;
    let x_endog_work = match sqrt_weight.as_ref() {
        Some(scale) => scale_rows(x_endog, scale).map_err(PyValueError::new_err)?,
        None => x_endog.clone(),
    };
    let x_exog_work = match sqrt_weight.as_ref() {
        Some(scale) => scale_rows(x_exog, scale).map_err(PyValueError::new_err)?,
        None => x_exog.clone(),
    };
    let z_work = match sqrt_weight.as_ref() {
        Some(scale) => scale_rows(z, scale).map_err(PyValueError::new_err)?,
        None => z.clone(),
    };
    let y_work = match sqrt_weight.as_ref() {
        Some(scale) => scale_vec(y, scale).map_err(PyValueError::new_err)?,
        None => y.clone(),
    };

    let (x_design, z_design) =
        build_iv_designs(&x_endog_work, &x_exog_work, &z_work, fit_intercept)?;
    let x_endog_hat =
        solve_least_squares_mat(&z_design, &x_endog_work).map(|pi_hat| z_design.dot(&pi_hat));
    let x_endog_hat = x_endog_hat.map_err(PyValueError::new_err)?;
    let x_hat_rhs = if x_exog_work.ncols() > 0 {
        concatenate(Axis(1), &[x_endog_hat.view(), x_exog_work.view()])
            .map_err(|_| PyValueError::new_err("failed to concat endog/exog"))?
    } else {
        x_endog_hat
    };
    let x_hat_design = if fit_intercept {
        add_intercept(&x_hat_rhs)
    } else {
        x_hat_rhs
    };
    let params = solve_least_squares_vec(&x_hat_design, &y_work).map_err(PyValueError::new_err)?;

    let fitted = x_design.dot(&params);
    let residuals = y_work - &fitted;

    Ok(TwoSlsFitResult {
        params,
        x_design,
        z_design,
        residuals,
    })
}

fn softmax_weights(theta: &Array1<f64>) -> Array1<f64> {
    let max_theta = theta
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
    let exp_shifted = theta.mapv(|value| (value - max_theta).exp());
    let sum = exp_shifted.sum();
    exp_shifted / sum
}

fn synthetic_control_rmse(
    donors: &Array2<f64>,
    treated: &Array1<f64>,
    weights: &Array1<f64>,
) -> f64 {
    let residual = donors.dot(weights) - treated;
    (residual.mapv(|value| value * value).mean().unwrap_or(0.0)).sqrt()
}

struct SyntheticControlProblem<'a> {
    donors: ArrayView2<'a, f64>,
    treated: ArrayView1<'a, f64>,
}

impl CostFunction for SyntheticControlProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let weights = softmax_weights(theta);
        let residual = self.donors.dot(&weights) - &self.treated;
        let mse = 0.5 * residual.dot(&residual) / (self.donors.nrows() as f64);
        Ok(mse)
    }
}

impl Gradient for SyntheticControlProblem<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(
        &self,
        theta: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        let weights = softmax_weights(theta);
        let residual = self.donors.dot(&weights) - &self.treated;
        let grad_weights = self.donors.t().dot(&residual) / (self.donors.nrows() as f64);
        let centered = &grad_weights - weights.dot(&grad_weights);
        Ok(weights * centered)
    }
}

fn fit_synthetic_control_weights(
    donors: &Array2<f64>,
    treated: &Array1<f64>,
    max_iterations: u64,
) -> PyResult<Array1<f64>> {
    if donors.nrows() != treated.len() {
        return Err(PyValueError::new_err(
            "donor rows must match treated length",
        ));
    }
    if donors.nrows() == 0 {
        return Err(PyValueError::new_err(
            "need at least one pre-treatment period",
        ));
    }
    if donors.ncols() == 0 {
        return Err(PyValueError::new_err("need at least one donor series"));
    }
    if donors.ncols() == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }

    let problem = SyntheticControlProblem {
        donors: donors.view(),
        treated: treated.view(),
    };
    let theta0 = Array1::<f64>::zeros(donors.ncols());
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let mut result = Executor::new(problem, solver)
        .configure(|state| state.param(theta0).max_iters(max_iterations))
        .run()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let theta = result
        .state
        .take_best_param()
        .ok_or_else(|| PyValueError::new_err("synthetic control optimization failed"))?;

    Ok(softmax_weights(&theta))
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
        )
        .map_err(PyValueError::new_err)?;
        let coef_se = diag_sqrt(&cov);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
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
        self.coef
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

        let fit =
            fit_two_sls_closed_form(x_endog, x_exog, z, y, self.fit_intercept, sample_weight)?;
        let (intercept, coef) = split_params(&fit.params, self.fit_intercept);
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = twosls_covariance(
            &fit.x_design,
            &fit.z_design,
            &fit.residuals,
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
