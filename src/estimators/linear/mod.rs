use crate::rla::{count_sketch_joint, randomized_svd_impl, sketch_ols_params};
use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, invert_matrix, ols_vanilla_cov, pyarray1_from_f64,
    pyarray2_from_f64, sandwich_cov_from_parameter_scores, scale_rows, scale_vec,
    solve_least_squares_mat, solve_least_squares_vec, sqrt_sample_weight, take_rows, take_rows_u32,
    take_rows_vec, to_array1, to_array1_i64, to_array2, to_array2_u32, validate_sample_weight,
};
use argmin::core::{CostFunction, Executor, Gradient};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::DMatrix;
use ndarray::{concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use within::{Solver as WithinSolver, SolverParams as WithinSolverParams};

struct FixedEffectsOlsFitResult {
    coef: Array1<f64>,
    x_resid: Array2<f64>,
    y_resid: Array1<f64>,
}

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

    let (_x_design, z_design) =
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

    Ok(TwoSlsFitResult { params })
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

struct SimplexLeastSquaresProblem<'a> {
    design: ArrayView2<'a, f64>,
    target: ArrayView1<'a, f64>,
    zeta: f64,
    intercept: bool,
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

impl SimplexLeastSquaresProblem<'_> {
    fn residual(&self, weights: &Array1<f64>) -> Array1<f64> {
        let mut residual = self.design.dot(weights) - self.target;
        if self.intercept {
            let mean = residual.mean().unwrap_or(0.0);
            residual.mapv_inplace(|value| value - mean);
        }
        residual
    }
}

impl CostFunction for SimplexLeastSquaresProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let weights = softmax_weights(theta);
        let residual = self.residual(&weights);
        let n = self.design.nrows() as f64;
        let fit = residual.dot(&residual) / n;
        let penalty = self.zeta * self.zeta * weights.dot(&weights);
        Ok(0.5 * (fit + penalty))
    }
}

impl Gradient for SimplexLeastSquaresProblem<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(
        &self,
        theta: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        let weights = softmax_weights(theta);
        let residual = self.residual(&weights);
        let n = self.design.nrows() as f64;
        let grad_weights = self.design.t().dot(&residual) / n
            + weights.mapv(|value| self.zeta * self.zeta * value);
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

fn fit_simplex_least_squares_weights(
    design: &Array2<f64>,
    target: &Array1<f64>,
    zeta: f64,
    intercept: bool,
    max_iterations: u64,
) -> PyResult<Array1<f64>> {
    if design.nrows() != target.len() {
        return Err(PyValueError::new_err(
            "design rows must match target length",
        ));
    }
    if design.nrows() == 0 {
        return Err(PyValueError::new_err("need at least one observation"));
    }
    if design.ncols() == 0 {
        return Err(PyValueError::new_err("need at least one simplex weight"));
    }
    if !zeta.is_finite() || zeta < 0.0 {
        return Err(PyValueError::new_err("zeta must be finite and nonnegative"));
    }
    if design.ncols() == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }

    let problem = SimplexLeastSquaresProblem {
        design: design.view(),
        target: target.view(),
        zeta,
        intercept,
    };
    let theta0 = Array1::<f64>::zeros(design.ncols());
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 7);

    let mut result = Executor::new(problem, solver)
        .configure(|state| state.param(theta0).max_iters(max_iterations))
        .run()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let theta = result
        .state
        .take_best_param()
        .ok_or_else(|| PyValueError::new_err("simplex least-squares optimization failed"))?;

    Ok(softmax_weights(&theta))
}

fn simplex_intercept(design: &Array2<f64>, target: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let fitted = design.dot(weights);
    (target - &fitted).mean().unwrap_or(0.0)
}

mod panel;
mod regression;
mod synthetic;

pub use panel::{
    panel_factor, panel_fe, FixedEffectsOLS, HorizontalPanelRidge, InteractiveFixedEffects,
    MatrixCompletion, SyntheticDID,
};
pub use regression::{TwoSLS, OLS};
pub use synthetic::SyntheticControl;
