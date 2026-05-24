use crate::hyptests::{f_sf, wald_test_arrays};
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

fn validate_finite_matrix(name: &str, values: &Array2<f64>) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{} must contain only finite values",
            name
        )));
    }
    Ok(())
}

fn fit_ridge_with_intercept(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalty: f64,
) -> PyResult<(f64, Array1<f64>)> {
    if x.nrows() != y.len() {
        return Err(PyValueError::new_err("x rows must match y length"));
    }
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(PyValueError::new_err(
            "penalty must be finite and nonnegative",
        ));
    }
    let design = add_intercept(x);
    let mut gram = design.t().dot(&design);
    for j in 1..gram.ncols() {
        gram[[j, j]] += penalty;
    }
    let rhs = design.t().dot(y);
    let params = invert_matrix(&gram)
        .map_err(PyValueError::new_err)?
        .dot(&rhs);
    Ok((params[0], params.slice(s![1..]).to_owned()))
}

fn center_effects(row_effects: &mut Array1<f64>, col_effects: &mut Array1<f64>) {
    let row_mean = row_effects.mean().unwrap_or(0.0);
    row_effects.mapv_inplace(|value| value - row_mean);
    col_effects.mapv_inplace(|value| value + row_mean);

    let col_mean = col_effects.mean().unwrap_or(0.0);
    col_effects.mapv_inplace(|value| value - col_mean);
    row_effects.mapv_inplace(|value| value + col_mean);
}

fn svt(matrix: &Array2<f64>, threshold: f64) -> PyResult<(Array2<f64>, Array1<f64>)> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let data: Vec<f64> = matrix.iter().copied().collect();
    let dm = DMatrix::from_row_slice(rows, cols, &data);
    let svd = dm.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| PyValueError::new_err("SVD failed to return left singular vectors"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| PyValueError::new_err("SVD failed to return right singular vectors"))?;
    let k = svd.singular_values.len();
    let mut shrunk = Array1::<f64>::zeros(k);
    let mut diag = DMatrix::<f64>::zeros(k, k);
    for j in 0..k {
        let value = (svd.singular_values[j] - threshold).max(0.0);
        shrunk[j] = value;
        diag[(j, j)] = value;
    }
    let reconstructed = u * diag * vt;
    let mut out = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = reconstructed[(i, j)];
        }
    }
    Ok((out, shrunk))
}

fn svt_randomized(
    matrix: &Array2<f64>,
    threshold: f64,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<(Array2<f64>, Array1<f64>)> {
    let min_dim = matrix.nrows().min(matrix.ncols());
    if rank == 0 || rank > min_dim {
        return Err(PyValueError::new_err(
            "svd_rank must be between 1 and min(Y.shape)",
        ));
    }
    let result = randomized_svd_impl(matrix, rank, oversamples, power_iter, seed)?;
    let k = result.singular_values.len();
    let mut shrunk = Array1::<f64>::zeros(k);
    let mut scaled_u = result.u.clone();
    for j in 0..k {
        let value = (result.singular_values[j] - threshold).max(0.0);
        shrunk[j] = value;
        for i in 0..scaled_u.nrows() {
            scaled_u[[i, j]] *= value;
        }
    }
    let reconstructed = scaled_u.dot(&result.vt);
    Ok((reconstructed, shrunk))
}

fn svt_with_method(
    matrix: &Array2<f64>,
    threshold: f64,
    svd_method: &str,
    svd_rank: Option<usize>,
    svd_oversamples: usize,
    svd_power_iter: usize,
    svd_seed: Option<u64>,
) -> PyResult<(Array2<f64>, Array1<f64>)> {
    match svd_method {
        "exact" => svt(matrix, threshold),
        "randomized" => {
            let min_dim = matrix.nrows().min(matrix.ncols());
            let rank = svd_rank.unwrap_or(min_dim);
            svt_randomized(
                matrix,
                threshold,
                rank,
                svd_oversamples,
                svd_power_iter,
                svd_seed,
            )
        }
        _ => Err(PyValueError::new_err(
            "svd_method must be either 'exact' or 'randomized'",
        )),
    }
}

struct PanelFactorFit {
    factor: Array2<f64>,
    loading: Array2<f64>,
    vnt: Array2<f64>,
    fixed_effect: Array2<f64>,
}

fn dmatrix_to_array2(matrix: &DMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((matrix.nrows(), matrix.ncols()));
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            out[[i, j]] = matrix[(i, j)];
        }
    }
    out
}

fn array2_to_dmatrix(matrix: &Array2<f64>) -> DMatrix<f64> {
    let data: Vec<f64> = matrix.iter().copied().collect();
    DMatrix::from_row_slice(matrix.nrows(), matrix.ncols(), &data)
}

fn panel_factor_fit(e: &Array2<f64>, rank: usize) -> PyResult<PanelFactorFit> {
    let t = e.nrows();
    let n = e.ncols();
    if rank > t.min(n) {
        return Err(PyValueError::new_err(
            "rank must be <= min(n_periods, n_units)",
        ));
    }
    if rank == 0 {
        return Ok(PanelFactorFit {
            factor: Array2::<f64>::zeros((t, 0)),
            loading: Array2::<f64>::zeros((n, 0)),
            vnt: Array2::<f64>::zeros((0, 0)),
            fixed_effect: Array2::<f64>::zeros((t, n)),
        });
    }

    let e_dm = array2_to_dmatrix(e);
    let scale = (n * t) as f64;
    let (factor_dm, loading_dm, singular_values) = if t < n {
        let ee = (&e_dm * e_dm.transpose()) / scale;
        let svd = ee.svd(true, false);
        let u = svd
            .u
            .ok_or_else(|| PyValueError::new_err("SVD failed to return factor vectors"))?;
        let factor = u.columns(0, rank).into_owned() * (t as f64).sqrt();
        let loading = e_dm.transpose() * &factor / (t as f64);
        (factor, loading, svd.singular_values)
    } else {
        let ee = (e_dm.transpose() * &e_dm) / scale;
        let svd = ee.svd(true, false);
        let u = svd
            .u
            .ok_or_else(|| PyValueError::new_err("SVD failed to return loading vectors"))?;
        let loading = u.columns(0, rank).into_owned() * (n as f64).sqrt();
        let factor = e_dm * &loading / (n as f64);
        (factor, loading, svd.singular_values)
    };

    let fixed_effect_dm = &factor_dm * loading_dm.transpose();
    let mut vnt = Array2::<f64>::zeros((rank, rank));
    for j in 0..rank {
        vnt[[j, j]] = singular_values[j];
    }

    Ok(PanelFactorFit {
        factor: dmatrix_to_array2(&factor_dm),
        loading: dmatrix_to_array2(&loading_dm),
        vnt,
        fixed_effect: dmatrix_to_array2(&fixed_effect_dm),
    })
}

fn panel_factor_fit_randomized(
    e: &Array2<f64>,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<PanelFactorFit> {
    let t = e.nrows();
    let n = e.ncols();
    if rank > t.min(n) {
        return Err(PyValueError::new_err(
            "rank must be <= min(n_periods, n_units)",
        ));
    }
    if rank == 0 {
        return Ok(PanelFactorFit {
            factor: Array2::<f64>::zeros((t, 0)),
            loading: Array2::<f64>::zeros((n, 0)),
            vnt: Array2::<f64>::zeros((0, 0)),
            fixed_effect: Array2::<f64>::zeros((t, n)),
        });
    }

    let svd = randomized_svd_impl(e, rank, oversamples, power_iter, seed)?;
    let mut factor = svd.u.clone();
    let factor_scale = (t as f64).sqrt();
    factor.mapv_inplace(|value| value * factor_scale);

    let mut loading = Array2::<f64>::zeros((n, rank));
    for k in 0..rank {
        let scale = svd.singular_values[k] / factor_scale;
        for j in 0..n {
            loading[[j, k]] = svd.vt[[k, j]] * scale;
        }
    }
    let fixed_effect = factor.dot(&loading.t());
    let mut vnt = Array2::<f64>::zeros((rank, rank));
    let denom = (n * t) as f64;
    for j in 0..rank {
        vnt[[j, j]] = svd.singular_values[j] * svd.singular_values[j] / denom;
    }

    Ok(PanelFactorFit {
        factor,
        loading,
        vnt,
        fixed_effect,
    })
}

fn panel_factor_fit_with_method(
    e: &Array2<f64>,
    rank: usize,
    factor_method: &str,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<PanelFactorFit> {
    match factor_method {
        "exact" => panel_factor_fit(e, rank),
        "randomized" => panel_factor_fit_randomized(e, rank, oversamples, power_iter, seed),
        _ => Err(PyValueError::new_err(
            "factor_method must be either 'exact' or 'randomized'",
        )),
    }
}

fn panel_fe_fect(e: &Array2<f64>, lambda: f64, hard: bool) -> PyResult<(Array2<f64>, Array1<f64>)> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(PyValueError::new_err(
            "lambda must be finite and nonnegative",
        ));
    }
    let t = e.nrows();
    let n = e.ncols();
    let scale = (t * n) as f64;
    let scaled = e.mapv(|value| value / scale);
    let data: Vec<f64> = scaled.iter().copied().collect();
    let dm = DMatrix::from_row_slice(t, n, &data);
    let svd = dm.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| PyValueError::new_err("SVD failed to return left singular vectors"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| PyValueError::new_err("SVD failed to return right singular vectors"))?;
    let k = svd.singular_values.len();
    let mut shrunk = Array1::<f64>::zeros(k);
    let mut diag = DMatrix::<f64>::zeros(k, k);
    for j in 0..k {
        let value = if svd.singular_values[j] > lambda {
            if hard {
                svd.singular_values[j]
            } else {
                svd.singular_values[j] - lambda
            }
        } else {
            0.0
        };
        shrunk[j] = value;
        diag[(j, j)] = value;
    }
    let reconstructed = u * diag * vt * scale;
    Ok((dmatrix_to_array2(&reconstructed), shrunk))
}

fn additive_demean_balanced(
    y: &Array2<f64>,
    force: i32,
) -> PyResult<(Array2<f64>, f64, Array1<f64>, Array1<f64>)> {
    if !(0..=3).contains(&force) {
        return Err(PyValueError::new_err("force must be one of {0, 1, 2, 3}"));
    }
    validate_finite_matrix("y", y)?;
    let mut yy = y.clone();
    let mu = yy.mean().unwrap_or(0.0);
    yy.mapv_inplace(|value| value - mu);
    let mut alpha = Array1::<f64>::zeros(y.ncols());
    let mut xi = Array1::<f64>::zeros(y.nrows());

    if force == 1 || force == 3 {
        alpha = yy
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::<f64>::zeros(y.ncols()));
        for i in 0..yy.nrows() {
            for j in 0..yy.ncols() {
                yy[[i, j]] -= alpha[j];
            }
        }
    }
    if force == 2 || force == 3 {
        xi = yy
            .mean_axis(Axis(1))
            .unwrap_or_else(|| Array1::<f64>::zeros(y.nrows()));
        for i in 0..yy.nrows() {
            for j in 0..yy.ncols() {
                yy[[i, j]] -= xi[i];
            }
        }
    }
    Ok((yy, mu, alpha, xi))
}

fn matrix_completion_update_effects(
    y: &Array2<f64>,
    mask: &Array2<bool>,
    low_rank: &Array2<f64>,
    row_effects: &mut Array1<f64>,
    col_effects: &mut Array1<f64>,
    fit_unit_effects: bool,
    fit_time_effects: bool,
    effect_iterations: usize,
) {
    for _ in 0..effect_iterations {
        if fit_unit_effects {
            for i in 0..y.nrows() {
                let mut sum = 0.0;
                let mut count = 0usize;
                for t in 0..y.ncols() {
                    if mask[[i, t]] {
                        sum += y[[i, t]] - low_rank[[i, t]] - col_effects[t];
                        count += 1;
                    }
                }
                if count > 0 {
                    row_effects[i] = sum / count as f64;
                }
            }
        }
        if fit_time_effects {
            for t in 0..y.ncols() {
                let mut sum = 0.0;
                let mut count = 0usize;
                for i in 0..y.nrows() {
                    if mask[[i, t]] {
                        sum += y[[i, t]] - low_rank[[i, t]] - row_effects[i];
                        count += 1;
                    }
                }
                if count > 0 {
                    col_effects[t] = sum / count as f64;
                }
            }
        }
        center_effects(row_effects, col_effects);
    }
}

fn matrix_completion_objective(
    y: &Array2<f64>,
    mask: &Array2<bool>,
    low_rank: &Array2<f64>,
    row_effects: &Array1<f64>,
    col_effects: &Array1<f64>,
    singular_values: &Array1<f64>,
    lambda_l: f64,
) -> f64 {
    let mut rss = 0.0;
    let mut n_obs = 0usize;
    for i in 0..y.nrows() {
        for t in 0..y.ncols() {
            if mask[[i, t]] {
                let residual = low_rank[[i, t]] + row_effects[i] + col_effects[t] - y[[i, t]];
                rss += residual * residual;
                n_obs += 1;
            }
        }
    }
    rss / (n_obs.max(1) as f64) + lambda_l * singular_values.sum()
}

fn matrix_completion_lambda_max_internal(
    y: &Array2<f64>,
    mask: &Array2<bool>,
    fit_unit_effects: bool,
    fit_time_effects: bool,
) -> PyResult<f64> {
    let mut row_effects = Array1::<f64>::zeros(y.nrows());
    let mut col_effects = Array1::<f64>::zeros(y.ncols());
    let low_rank = Array2::<f64>::zeros(y.raw_dim());
    matrix_completion_update_effects(
        y,
        mask,
        &low_rank,
        &mut row_effects,
        &mut col_effects,
        fit_unit_effects,
        fit_time_effects,
        20,
    );

    let mut residual = Array2::<f64>::zeros(y.raw_dim());
    let mut n_obs = 0usize;
    for i in 0..y.nrows() {
        for t in 0..y.ncols() {
            if mask[[i, t]] {
                residual[[i, t]] = y[[i, t]] - row_effects[i] - col_effects[t];
                n_obs += 1;
            }
        }
    }
    let (_, singular_values) = svt(&residual, 0.0)?;
    Ok(2.0 * singular_values[0] / (n_obs.max(1) as f64))
}

fn sdid_sigma_estimator(y_reordered: &Array2<f64>, n_control: usize, t_pre: usize) -> f64 {
    if n_control == 0 || t_pre < 2 {
        return 0.0;
    }

    let mut diffs = Vec::with_capacity(n_control * (t_pre - 1));
    for i in 0..n_control {
        for t in 1..t_pre {
            diffs.push(y_reordered[[i, t]] - y_reordered[[i, t - 1]]);
        }
    }
    if diffs.len() < 2 {
        return 0.0;
    }

    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let var = diffs
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (diffs.len() - 1) as f64;
    var.sqrt()
}

#[derive(Clone)]
struct PanelTreatmentInfo {
    first_treat: Vec<Option<usize>>,
    ever_treated: Vec<usize>,
    never_treated: Vec<usize>,
    cohorts: Vec<usize>,
}

struct PanelEventSummary {
    event_time: Array1<f64>,
    estimate: Array1<f64>,
    n: Array1<f64>,
}

struct PanelGroupMeans {
    cohort: Array1<f64>,
    event_time: Array1<f64>,
    n_treated: Array1<f64>,
    treated_mean: Array1<f64>,
    counterfactual_mean: Array1<f64>,
    effect: Array1<f64>,
}

struct PanelEffectSummaries {
    group_means: PanelGroupMeans,
    event_unweighted: PanelEventSummary,
    event_weighted: PanelEventSummary,
}

fn infer_panel_treatment(y: &Array2<f64>, w: &Array2<f64>) -> PyResult<PanelTreatmentInfo> {
    if y.nrows() == 0 || y.ncols() == 0 {
        return Err(PyValueError::new_err("Y must be a non-empty 2D matrix"));
    }
    if w.raw_dim() != y.raw_dim() {
        return Err(PyValueError::new_err("W must have the same shape as Y"));
    }
    validate_finite_matrix("Y", y)?;
    validate_finite_matrix("W", w)?;

    let mut first_treat = Vec::with_capacity(w.nrows());
    let mut ever_treated = Vec::new();
    let mut never_treated = Vec::new();
    let mut cohorts = Vec::new();

    for i in 0..w.nrows() {
        let mut first: Option<usize> = None;
        for t in 0..w.ncols() {
            let value = w[[i, t]];
            if (value - 0.0).abs() > 1e-10 && (value - 1.0).abs() > 1e-10 {
                return Err(PyValueError::new_err(
                    "W must be a binary 0/1 treatment indicator matrix",
                ));
            }
            if value > 0.5 && first.is_none() {
                first = Some(t);
            }
        }
        if let Some(g) = first {
            for t in g..w.ncols() {
                if w[[i, t]] < 0.5 {
                    return Err(PyValueError::new_err(
                        "W must be absorbing: once treated, a unit must remain treated",
                    ));
                }
            }
            ever_treated.push(i);
            if !cohorts.contains(&g) {
                cohorts.push(g);
            }
        } else {
            never_treated.push(i);
        }
        first_treat.push(first);
    }

    if ever_treated.is_empty() {
        return Err(PyValueError::new_err(
            "W must mark at least one ever-treated unit",
        ));
    }
    cohorts.sort_unstable();
    Ok(PanelTreatmentInfo {
        first_treat,
        ever_treated,
        never_treated,
        cohorts,
    })
}

fn ensure_panel_has_never_treated(info: &PanelTreatmentInfo) -> PyResult<()> {
    if info.never_treated.is_empty() {
        return Err(PyValueError::new_err(
            "this estimator currently requires at least one never-treated donor unit",
        ));
    }
    Ok(())
}

fn cohort_units(info: &PanelTreatmentInfo, cohort: usize) -> Vec<usize> {
    info.first_treat
        .iter()
        .enumerate()
        .filter_map(|(idx, first)| match first {
            Some(value) if *value == cohort => Some(idx),
            _ => None,
        })
        .collect()
}

fn finite_mean(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        if value.is_finite() {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn event_summary_from_groups(rows: &[(i64, f64, f64)]) -> PanelEventSummary {
    let mut event_times: Vec<i64> = rows.iter().map(|row| row.0).collect();
    event_times.sort_unstable();
    event_times.dedup();

    let mut out_event = Vec::with_capacity(event_times.len());
    let mut out_est = Vec::with_capacity(event_times.len());
    let mut out_n = Vec::with_capacity(event_times.len());

    for event in event_times {
        let mut values = Vec::new();
        let mut weights = Vec::new();
        for (row_event, value, weight) in rows {
            if *row_event == event && value.is_finite() && weight.is_finite() && *weight > 0.0 {
                values.push(*value);
                weights.push(*weight);
            }
        }
        let total_weight = weights.iter().sum::<f64>();
        let estimate = if total_weight > 0.0 {
            values
                .iter()
                .zip(weights.iter())
                .map(|(value, weight)| value * weight)
                .sum::<f64>()
                / total_weight
        } else {
            f64::NAN
        };
        out_event.push(event as f64);
        out_est.push(estimate);
        out_n.push(total_weight);
    }

    PanelEventSummary {
        event_time: Array1::from_vec(out_event),
        estimate: Array1::from_vec(out_est),
        n: Array1::from_vec(out_n),
    }
}

fn summarize_panel_effects(
    y: &Array2<f64>,
    counterfactual: &Array2<f64>,
    info: &PanelTreatmentInfo,
) -> PanelEffectSummaries {
    let mut group_cohort = Vec::new();
    let mut group_event_time = Vec::new();
    let mut group_n = Vec::new();
    let mut group_treated_mean = Vec::new();
    let mut group_counterfactual_mean = Vec::new();
    let mut group_effect = Vec::new();
    let mut unweighted_rows = Vec::new();
    let mut weighted_rows = Vec::new();

    for cohort in &info.cohorts {
        let units = cohort_units(info, *cohort);
        for t in 0..y.ncols() {
            let event = t as i64 - *cohort as i64;
            let treated_values: Vec<f64> = units.iter().map(|idx| y[[*idx, t]]).collect();
            let cf_values: Vec<f64> = units.iter().map(|idx| counterfactual[[*idx, t]]).collect();
            let effects: Vec<f64> = units
                .iter()
                .map(|idx| y[[*idx, t]] - counterfactual[[*idx, t]])
                .filter(|value| value.is_finite())
                .collect();
            if effects.is_empty() {
                continue;
            }
            let effect = finite_mean(&effects);
            let n = effects.len() as f64;
            group_cohort.push(*cohort as f64);
            group_event_time.push(event as f64);
            group_n.push(n);
            group_treated_mean.push(finite_mean(&treated_values));
            group_counterfactual_mean.push(finite_mean(&cf_values));
            group_effect.push(effect);
            unweighted_rows.push((event, effect, 1.0));
            weighted_rows.push((event, effect, n));
        }
    }

    PanelEffectSummaries {
        group_means: PanelGroupMeans {
            cohort: Array1::from_vec(group_cohort),
            event_time: Array1::from_vec(group_event_time),
            n_treated: Array1::from_vec(group_n),
            treated_mean: Array1::from_vec(group_treated_mean),
            counterfactual_mean: Array1::from_vec(group_counterfactual_mean),
            effect: Array1::from_vec(group_effect),
        },
        event_unweighted: event_summary_from_groups(&unweighted_rows),
        event_weighted: event_summary_from_groups(&weighted_rows),
    }
}

fn event_summary_to_dict<'py>(
    py: Python<'py>,
    summary: &PanelEventSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("event_time", pyarray1_from_f64(py, &summary.event_time))?;
    dict.set_item("estimate", pyarray1_from_f64(py, &summary.estimate))?;
    dict.set_item("n", pyarray1_from_f64(py, &summary.n))?;
    Ok(dict)
}

fn aggregate_group_means_to_dict<'py>(
    py: Python<'py>,
    group: &PanelGroupMeans,
    weighted: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut event_times: Vec<i64> = group
        .event_time
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| *value as i64)
        .collect();
    event_times.sort_unstable();
    event_times.dedup();

    let mut out_event = Vec::with_capacity(event_times.len());
    let mut out_n = Vec::with_capacity(event_times.len());
    let mut out_treated = Vec::with_capacity(event_times.len());
    let mut out_cf = Vec::with_capacity(event_times.len());
    let mut out_effect = Vec::with_capacity(event_times.len());

    for event in event_times {
        let mut treated_sum = 0.0;
        let mut cf_sum = 0.0;
        let mut effect_sum = 0.0;
        let mut weight_sum = 0.0;
        for idx in 0..group.event_time.len() {
            if group.event_time[idx] as i64 != event {
                continue;
            }
            let weight = if weighted { group.n_treated[idx] } else { 1.0 };
            if weight <= 0.0 || !weight.is_finite() {
                continue;
            }
            if group.treated_mean[idx].is_finite() {
                treated_sum += group.treated_mean[idx] * weight;
            }
            if group.counterfactual_mean[idx].is_finite() {
                cf_sum += group.counterfactual_mean[idx] * weight;
            }
            if group.effect[idx].is_finite() {
                effect_sum += group.effect[idx] * weight;
            }
            weight_sum += weight;
        }
        out_event.push(event as f64);
        out_n.push(weight_sum);
        if weight_sum > 0.0 {
            out_treated.push(treated_sum / weight_sum);
            out_cf.push(cf_sum / weight_sum);
            out_effect.push(effect_sum / weight_sum);
        } else {
            out_treated.push(f64::NAN);
            out_cf.push(f64::NAN);
            out_effect.push(f64::NAN);
        }
    }

    let dict = PyDict::new(py);
    dict.set_item(
        "event_time",
        pyarray1_from_f64(py, &Array1::from_vec(out_event)),
    )?;
    dict.set_item("n", pyarray1_from_f64(py, &Array1::from_vec(out_n)))?;
    dict.set_item(
        "treated_mean",
        pyarray1_from_f64(py, &Array1::from_vec(out_treated)),
    )?;
    dict.set_item(
        "counterfactual_mean",
        pyarray1_from_f64(py, &Array1::from_vec(out_cf)),
    )?;
    dict.set_item(
        "effect",
        pyarray1_from_f64(py, &Array1::from_vec(out_effect)),
    )?;
    Ok(dict)
}

fn group_means_to_dict<'py>(
    py: Python<'py>,
    group: &PanelGroupMeans,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("cohort", pyarray1_from_f64(py, &group.cohort))?;
    dict.set_item("event_time", pyarray1_from_f64(py, &group.event_time))?;
    dict.set_item("n_treated", pyarray1_from_f64(py, &group.n_treated))?;
    dict.set_item("treated_mean", pyarray1_from_f64(py, &group.treated_mean))?;
    dict.set_item(
        "counterfactual_mean",
        pyarray1_from_f64(py, &group.counterfactual_mean),
    )?;
    dict.set_item("effect", pyarray1_from_f64(py, &group.effect))?;
    dict.set_item(
        "unweighted",
        aggregate_group_means_to_dict(py, group, false)?,
    )?;
    dict.set_item("weighted", aggregate_group_means_to_dict(py, group, true)?)?;
    Ok(dict)
}

fn panel_summaries_to_dict<'py>(
    py: Python<'py>,
    summaries: &PanelEffectSummaries,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let event_study = PyDict::new(py);
    event_study.set_item(
        "unweighted",
        event_summary_to_dict(py, &summaries.event_unweighted)?,
    )?;
    event_study.set_item(
        "weighted",
        event_summary_to_dict(py, &summaries.event_weighted)?,
    )?;
    let group_means = group_means_to_dict(py, &summaries.group_means)?;
    Ok((event_study, group_means))
}

fn panel_att_from_effects(w: &Array2<f64>, effects: &Array2<f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..w.nrows() {
        for t in 0..w.ncols() {
            let effect = effects[[i, t]];
            if w[[i, t]] > 0.5 && effect.is_finite() {
                sum += effect;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn panel_group_pre_rmse(effects: &Array2<f64>, info: &PanelTreatmentInfo) -> f64 {
    let mut sq = Vec::new();
    for cohort in &info.cohorts {
        let units = cohort_units(info, *cohort);
        for t in 0..*cohort {
            let values: Vec<f64> = units.iter().map(|idx| effects[[*idx, t]]).collect();
            let mean = finite_mean(&values);
            if mean.is_finite() {
                sq.push(mean * mean);
            }
        }
    }
    if sq.is_empty() {
        f64::NAN
    } else {
        (sq.iter().sum::<f64>() / sq.len() as f64).sqrt()
    }
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
pub struct InteractiveFixedEffects {
    rank: usize,
    force: i32,
    factor_method: String,
    factor_oversamples: usize,
    factor_power_iter: usize,
    factor_seed: Option<u64>,
    fit: Option<Array2<f64>>,
    residuals: Option<Array2<f64>>,
    mu: Option<f64>,
    alpha: Option<Array1<f64>>,
    xi: Option<Array1<f64>>,
    factor: Option<Array2<f64>>,
    loading: Option<Array2<f64>>,
    vnt: Option<Array2<f64>>,
}

#[pymethods]
impl InteractiveFixedEffects {
    #[new]
    #[pyo3(signature = (rank=0, force=3, factor_method="exact".to_string(), factor_oversamples=10, factor_power_iter=1, factor_seed=None))]
    fn new(
        rank: usize,
        force: i32,
        factor_method: String,
        factor_oversamples: usize,
        factor_power_iter: usize,
        factor_seed: Option<u64>,
    ) -> PyResult<Self> {
        if !(0..=3).contains(&force) {
            return Err(PyValueError::new_err("force must be one of {0, 1, 2, 3}"));
        }
        if factor_method != "exact" && factor_method != "randomized" {
            return Err(PyValueError::new_err(
                "factor_method must be either 'exact' or 'randomized'",
            ));
        }
        if factor_power_iter > 10 {
            return Err(PyValueError::new_err("factor_power_iter must be <= 10"));
        }
        Ok(Self {
            rank,
            force,
            factor_method,
            factor_oversamples,
            factor_power_iter,
            factor_seed,
            fit: None,
            residuals: None,
            mu: None,
            alpha: None,
            xi: None,
            factor: None,
            loading: None,
            vnt: None,
        })
    }

    fn fit(&mut self, y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let y = to_array2(&y);
        if y.nrows() == 0 || y.ncols() == 0 {
            return Err(PyValueError::new_err("y must be a non-empty 2D matrix"));
        }
        let (demeaned, mu, alpha, xi) = additive_demean_balanced(&y, self.force)?;
        let pf = panel_factor_fit_with_method(
            &demeaned,
            self.rank,
            &self.factor_method,
            self.factor_oversamples,
            self.factor_power_iter,
            self.factor_seed,
        )?;
        let mut fitted = pf.fixed_effect.clone();
        for i in 0..fitted.nrows() {
            for j in 0..fitted.ncols() {
                fitted[[i, j]] += mu + alpha[j] + xi[i];
            }
        }
        let residuals = &y - &fitted;

        self.fit = Some(fitted);
        self.residuals = Some(residuals);
        self.mu = Some(mu);
        self.alpha = Some(alpha);
        self.xi = Some(xi);
        self.factor = Some(pf.factor);
        self.loading = Some(pf.loading);
        self.vnt = Some(pf.vnt);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let fit = self
            .fit
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        Ok(pyarray2_from_f64(py, fit))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let fit = self
            .fit
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        let residuals = self
            .residuals
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        let alpha = self
            .alpha
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        let xi = self
            .xi
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        let factor = self
            .factor
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        let loading = self
            .loading
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;
        let vnt = self
            .vnt
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("InteractiveFixedEffects model is not fitted"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("fit", pyarray2_from_f64(py, fit))?;
        dict.set_item("residuals", pyarray2_from_f64(py, residuals))?;
        dict.set_item("mu", self.mu)?;
        dict.set_item("alpha", pyarray1_from_f64(py, alpha))?;
        dict.set_item("xi", pyarray1_from_f64(py, xi))?;
        dict.set_item("factor", pyarray2_from_f64(py, factor))?;
        dict.set_item("loading", pyarray2_from_f64(py, loading))?;
        dict.set_item("vnt", pyarray2_from_f64(py, vnt))?;
        dict.set_item("rank", self.rank)?;
        dict.set_item("force", self.force)?;
        dict.set_item("factor_method", self.factor_method.clone())?;
        dict.set_item("factor_oversamples", self.factor_oversamples)?;
        dict.set_item("factor_power_iter", self.factor_power_iter)?;
        Ok(dict.into())
    }
}

#[pyfunction]
#[pyo3(signature = (e, rank, factor_method="exact".to_string(), oversamples=10, power_iter=1, seed=None))]
pub fn panel_factor<'py>(
    py: Python<'py>,
    e: PyReadonlyArray2<f64>,
    rank: usize,
    factor_method: String,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let e = to_array2(&e);
    validate_finite_matrix("e", &e)?;
    let pf = panel_factor_fit_with_method(&e, rank, &factor_method, oversamples, power_iter, seed)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("factor", pyarray2_from_f64(py, &pf.factor))?;
    dict.set_item("loading", pyarray2_from_f64(py, &pf.loading))?;
    dict.set_item("vnt", pyarray2_from_f64(py, &pf.vnt))?;
    dict.set_item("fe", pyarray2_from_f64(py, &pf.fixed_effect))?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (e, lambda, hard=false))]
pub fn panel_fe<'py>(
    py: Python<'py>,
    e: PyReadonlyArray2<f64>,
    lambda: f64,
    hard: bool,
) -> PyResult<Py<PyAny>> {
    let e = to_array2(&e);
    validate_finite_matrix("e", &e)?;
    let (fe, singular_values) = panel_fe_fect(&e, lambda, hard)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("fe", pyarray2_from_f64(py, &fe))?;
    dict.set_item("singular_values", pyarray1_from_f64(py, &singular_values))?;
    Ok(dict.into())
}

#[pyclass]
pub struct MatrixCompletion {
    lambda_l: Option<f64>,
    lambda_fraction: f64,
    fit_unit_effects: bool,
    fit_time_effects: bool,
    max_iterations: usize,
    effect_iterations: usize,
    tolerance: f64,
    svd_method: String,
    svd_rank: Option<usize>,
    svd_oversamples: usize,
    svd_power_iter: usize,
    svd_seed: Option<u64>,
    completed: Option<Array2<f64>>,
    low_rank: Option<Array2<f64>>,
    unit_effects: Option<Array1<f64>>,
    time_effects: Option<Array1<f64>>,
    singular_values: Option<Array1<f64>>,
    fitted_lambda_l: Option<f64>,
    objective: Option<f64>,
    iterations: Option<usize>,
    history_objective: Vec<f64>,
    history_rmse: Vec<f64>,
    y: Option<Array2<f64>>,
    w: Option<Array2<f64>>,
    treatment_info: Option<PanelTreatmentInfo>,
    treatment_effect: Option<Array2<f64>>,
    att: Option<f64>,
}

#[pymethods]
impl MatrixCompletion {
    #[new]
    #[pyo3(signature = (lambda_l=None, lambda_fraction=0.25, fit_unit_effects=true, fit_time_effects=true, max_iterations=500, effect_iterations=2, tolerance=1e-6, svd_method="exact".to_string(), svd_rank=None, svd_oversamples=10, svd_power_iter=1, svd_seed=None))]
    fn new(
        lambda_l: Option<f64>,
        lambda_fraction: f64,
        fit_unit_effects: bool,
        fit_time_effects: bool,
        max_iterations: usize,
        effect_iterations: usize,
        tolerance: f64,
        svd_method: String,
        svd_rank: Option<usize>,
        svd_oversamples: usize,
        svd_power_iter: usize,
        svd_seed: Option<u64>,
    ) -> PyResult<Self> {
        if let Some(value) = lambda_l {
            if !value.is_finite() || value < 0.0 {
                return Err(PyValueError::new_err(
                    "lambda_l must be finite and nonnegative",
                ));
            }
        }
        if !lambda_fraction.is_finite() || lambda_fraction < 0.0 {
            return Err(PyValueError::new_err(
                "lambda_fraction must be finite and nonnegative",
            ));
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be finite and nonnegative",
            ));
        }
        if svd_method != "exact" && svd_method != "randomized" {
            return Err(PyValueError::new_err(
                "svd_method must be either 'exact' or 'randomized'",
            ));
        }
        if matches!(svd_rank, Some(0)) {
            return Err(PyValueError::new_err("svd_rank must be positive"));
        }
        if svd_power_iter > 10 {
            return Err(PyValueError::new_err("svd_power_iter must be <= 10"));
        }
        Ok(Self {
            lambda_l,
            lambda_fraction,
            fit_unit_effects,
            fit_time_effects,
            max_iterations,
            effect_iterations,
            tolerance,
            svd_method,
            svd_rank,
            svd_oversamples,
            svd_power_iter,
            svd_seed,
            completed: None,
            low_rank: None,
            unit_effects: None,
            time_effects: None,
            singular_values: None,
            fitted_lambda_l: None,
            objective: None,
            iterations: None,
            history_objective: Vec::new(),
            history_rmse: Vec::new(),
            y: None,
            w: None,
            treatment_info: None,
            treatment_effect: None,
            att: None,
        })
    }

    fn fit(&mut self, y: PyReadonlyArray2<f64>, w: PyReadonlyArray2<f64>) -> PyResult<()> {
        let y_input = to_array2(&y);
        let w_input = to_array2(&w);
        let treatment_info = infer_panel_treatment(&y_input, &w_input)?;

        let mask_arr = w_input.mapv(|value| value < 0.5);
        if !mask_arr.iter().any(|value| *value) {
            return Err(PyValueError::new_err(
                "W leaves no untreated observed entries",
            ));
        }

        let mut y_work = Array2::<f64>::zeros(y_input.raw_dim());
        for i in 0..y_input.nrows() {
            for t in 0..y_input.ncols() {
                if mask_arr[[i, t]] {
                    let value = y_input[[i, t]];
                    if !value.is_finite() {
                        return Err(PyValueError::new_err("observed y entries must be finite"));
                    }
                    y_work[[i, t]] = value;
                }
            }
        }

        let lambda_l = match self.lambda_l {
            Some(value) => value,
            None => {
                self.lambda_fraction
                    * matrix_completion_lambda_max_internal(
                        &y_work,
                        &mask_arr,
                        self.fit_unit_effects,
                        self.fit_time_effects,
                    )?
            }
        };
        let n_obs = mask_arr.iter().filter(|value| **value).count().max(1) as f64;
        let threshold = lambda_l * n_obs / 2.0;

        let mut low_rank = Array2::<f64>::zeros(y_work.raw_dim());
        let mut row_effects = Array1::<f64>::zeros(y_work.nrows());
        let mut col_effects = Array1::<f64>::zeros(y_work.ncols());
        let mut singular_values = Array1::<f64>::zeros(y_work.nrows().min(y_work.ncols()));
        let mut previous_obj: Option<f64> = None;
        self.history_objective.clear();
        self.history_rmse.clear();
        let mut final_iteration = 0usize;

        for iteration in 0..self.max_iterations {
            matrix_completion_update_effects(
                &y_work,
                &mask_arr,
                &low_rank,
                &mut row_effects,
                &mut col_effects,
                self.fit_unit_effects,
                self.fit_time_effects,
                self.effect_iterations,
            );

            let mut projected = low_rank.clone();
            let mut rss = 0.0;
            for i in 0..y_work.nrows() {
                for t in 0..y_work.ncols() {
                    if mask_arr[[i, t]] {
                        let fitted = low_rank[[i, t]] + row_effects[i] + col_effects[t];
                        let residual = y_work[[i, t]] - fitted;
                        projected[[i, t]] += residual;
                        rss += residual * residual;
                    }
                }
            }
            let seed = self
                .svd_seed
                .map(|value| value.wrapping_add(iteration as u64));
            let (updated_low_rank, updated_singular_values) = svt_with_method(
                &projected,
                threshold,
                &self.svd_method,
                self.svd_rank,
                self.svd_oversamples,
                self.svd_power_iter,
                seed,
            )?;
            low_rank = updated_low_rank;
            singular_values = updated_singular_values;

            let obj = matrix_completion_objective(
                &y_work,
                &mask_arr,
                &low_rank,
                &row_effects,
                &col_effects,
                &singular_values,
                lambda_l,
            );
            self.history_objective.push(obj);
            self.history_rmse.push((rss / n_obs).sqrt());
            final_iteration = iteration + 1;
            if let Some(prev) = previous_obj {
                let rel = (prev - obj).abs() / (prev.abs() + 1e-12);
                if rel < self.tolerance {
                    break;
                }
            }
            previous_obj = Some(obj);
        }

        let mut completed = low_rank.clone();
        for i in 0..completed.nrows() {
            for t in 0..completed.ncols() {
                completed[[i, t]] += row_effects[i] + col_effects[t];
            }
        }

        let mut treatment_effect = Array2::<f64>::from_elem(y_input.raw_dim(), f64::NAN);
        for i in 0..y_input.nrows() {
            for t in 0..y_input.ncols() {
                if treatment_info.first_treat[i].is_some() {
                    treatment_effect[[i, t]] = y_input[[i, t]] - completed[[i, t]];
                }
            }
        }
        let att = panel_att_from_effects(&w_input, &treatment_effect);

        self.completed = Some(completed);
        self.low_rank = Some(low_rank);
        self.unit_effects = Some(row_effects);
        self.time_effects = Some(col_effects);
        self.singular_values = Some(singular_values);
        self.fitted_lambda_l = Some(lambda_l);
        self.objective = self.history_objective.last().copied();
        self.iterations = Some(final_iteration);
        self.y = Some(y_input);
        self.w = Some(w_input);
        self.treatment_info = Some(treatment_info);
        self.treatment_effect = Some(treatment_effect);
        self.att = Some(att);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let completed = self
            .completed
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        Ok(pyarray2_from_f64(py, completed))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let completed = self
            .completed
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let low_rank = self
            .low_rank
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let unit_effects = self
            .unit_effects
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let time_effects = self
            .time_effects
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let singular_values = self
            .singular_values
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let treatment_info = self
            .treatment_info
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MatrixCompletion model is not fitted"))?;
        let summaries = summarize_panel_effects(y, completed, treatment_info);
        let (event_study, group_means) = panel_summaries_to_dict(py, &summaries)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("completed", pyarray2_from_f64(py, completed))?;
        dict.set_item("low_rank", pyarray2_from_f64(py, low_rank))?;
        dict.set_item("unit_effects", pyarray1_from_f64(py, unit_effects))?;
        dict.set_item("time_effects", pyarray1_from_f64(py, time_effects))?;
        dict.set_item("singular_values", pyarray1_from_f64(py, singular_values))?;
        dict.set_item("lambda_l", self.fitted_lambda_l)?;
        dict.set_item("objective", self.objective)?;
        dict.set_item("iterations", self.iterations)?;
        dict.set_item("history_objective", self.history_objective.clone())?;
        dict.set_item("history_rmse", self.history_rmse.clone())?;
        dict.set_item("svd_method", self.svd_method.clone())?;
        dict.set_item("svd_rank", self.svd_rank)?;
        dict.set_item("svd_oversamples", self.svd_oversamples)?;
        dict.set_item("svd_power_iter", self.svd_power_iter)?;
        dict.set_item("att", self.att)?;
        dict.set_item("counterfactual", pyarray2_from_f64(py, completed))?;
        dict.set_item("treatment_effect", pyarray2_from_f64(py, treatment_effect))?;
        dict.set_item("event_study", event_study)?;
        dict.set_item("group_means", group_means)?;
        dict.set_item("control_units", treatment_info.never_treated.clone())?;
        dict.set_item("treated_units", treatment_info.ever_treated.clone())?;
        dict.set_item("cohorts", treatment_info.cohorts.clone())?;
        Ok(dict.into())
    }
}

#[pyclass]
pub struct HorizontalPanelRidge {
    penalty: f64,
    cohort_intercepts: Option<Array1<f64>>,
    cohort_coef: Option<Array2<f64>>,
    counterfactual: Option<Array2<f64>>,
    treatment_effect: Option<Array2<f64>>,
    att: Option<f64>,
    pre_rmse: Option<f64>,
    control_units: Option<Vec<usize>>,
    treated_units: Option<Vec<usize>>,
    cohorts: Option<Vec<usize>>,
    treatment_info: Option<PanelTreatmentInfo>,
    y: Option<Array2<f64>>,
    w: Option<Array2<f64>>,
}

#[pyclass]
pub struct SyntheticControl {
    max_iterations: u64,
    weights: Option<Array1<f64>>,
    donors: Option<Array2<f64>>,
    treated: Option<Array1<f64>>,
}

#[pyclass]
pub struct SyntheticDID {
    zeta_omega: Option<f64>,
    zeta_lambda: Option<f64>,
    max_iterations: u64,
    att: Option<f64>,
    unit_weights: Option<Array2<f64>>,
    time_weights: Option<Array2<f64>>,
    counterfactual: Option<Array2<f64>>,
    treatment_effect: Option<Array2<f64>>,
    pre_rmse: Option<f64>,
    unit_intercept: Option<Array1<f64>>,
    time_intercept: Option<Array1<f64>>,
    fitted_zeta_omega: Option<Array1<f64>>,
    fitted_zeta_lambda: Option<Array1<f64>>,
    control_units: Option<Vec<usize>>,
    treated_units: Option<Vec<usize>>,
    cohorts: Option<Vec<usize>>,
    treatment_info: Option<PanelTreatmentInfo>,
    y: Option<Array2<f64>>,
    w: Option<Array2<f64>>,
}

#[pymethods]
impl HorizontalPanelRidge {
    #[new]
    #[pyo3(signature = (penalty=1.0))]
    fn new(penalty: f64) -> PyResult<Self> {
        if !penalty.is_finite() || penalty < 0.0 {
            return Err(PyValueError::new_err(
                "penalty must be finite and nonnegative",
            ));
        }
        Ok(Self {
            penalty,
            cohort_intercepts: None,
            cohort_coef: None,
            counterfactual: None,
            treatment_effect: None,
            att: None,
            pre_rmse: None,
            control_units: None,
            treated_units: None,
            cohorts: None,
            treatment_info: None,
            y: None,
            w: None,
        })
    }

    fn fit(&mut self, y: PyReadonlyArray2<f64>, w: PyReadonlyArray2<f64>) -> PyResult<()> {
        let y = to_array2(&y);
        let w = to_array2(&w);
        let treatment_info = infer_panel_treatment(&y, &w)?;
        ensure_panel_has_never_treated(&treatment_info)?;
        if treatment_info.cohorts.iter().any(|cohort| *cohort == 0) {
            return Err(PyValueError::new_err(
                "HorizontalPanelRidge needs at least one pre-treatment period for every treated cohort",
            ));
        }

        let n_units = y.nrows();
        let n_periods = y.ncols();
        let n_cohorts = treatment_info.cohorts.len();
        let mut counterfactual = Array2::<f64>::from_elem((n_units, n_periods), f64::NAN);
        let mut treatment_effect = Array2::<f64>::from_elem((n_units, n_periods), f64::NAN);
        let mut cohort_intercepts = Array1::<f64>::zeros(n_cohorts);
        let mut cohort_coef = Array2::<f64>::zeros((n_cohorts, n_units));

        for (c_idx, cohort) in treatment_info.cohorts.iter().enumerate() {
            let treated_units = cohort_units(&treatment_info, *cohort);
            let control_units = &treatment_info.never_treated;
            let control_panel = y.select(Axis(0), control_units);
            let treated_panel = y.select(Axis(0), &treated_units);
            let treated_mean = treated_panel
                .mean_axis(Axis(0))
                .ok_or_else(|| PyValueError::new_err("failed to average treated outcomes"))?;
            let x_pre = control_panel.slice(s![.., 0..*cohort]).t().to_owned();
            let y_pre = treated_mean.slice(s![0..*cohort]).to_owned();
            let (intercept, coef) = fit_ridge_with_intercept(&x_pre, &y_pre, self.penalty)?;
            let x_all = control_panel.t().to_owned();
            let cohort_counterfactual = x_all.dot(&coef) + intercept;

            cohort_intercepts[c_idx] = intercept;
            for (j, unit) in control_units.iter().enumerate() {
                cohort_coef[[c_idx, *unit]] = coef[j];
            }
            for unit in treated_units {
                for t in 0..n_periods {
                    counterfactual[[unit, t]] = cohort_counterfactual[t];
                    treatment_effect[[unit, t]] = y[[unit, t]] - cohort_counterfactual[t];
                }
            }
        }

        let att = panel_att_from_effects(&w, &treatment_effect);
        let pre_rmse = panel_group_pre_rmse(&treatment_effect, &treatment_info);

        self.cohort_intercepts = Some(cohort_intercepts);
        self.cohort_coef = Some(cohort_coef);
        self.counterfactual = Some(counterfactual);
        self.treatment_effect = Some(treatment_effect);
        self.att = Some(att);
        self.pre_rmse = Some(pre_rmse);
        self.control_units = Some(treatment_info.never_treated.clone());
        self.treated_units = Some(treatment_info.ever_treated.clone());
        self.cohorts = Some(treatment_info.cohorts.clone());
        self.treatment_info = Some(treatment_info);
        self.y = Some(y);
        self.w = Some(w);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        Ok(pyarray2_from_f64(py, counterfactual))
    }

    fn treatment_effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        Ok(pyarray2_from_f64(py, treatment_effect))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let att = self
            .att
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let cohort_intercepts = self
            .cohort_intercepts
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let cohort_coef = self
            .cohort_coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let treatment_info = self
            .treatment_info
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let summaries = summarize_panel_effects(y, counterfactual, treatment_info);
        let (event_study, group_means) = panel_summaries_to_dict(py, &summaries)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("att", att)?;
        dict.set_item("intercept", cohort_intercepts[0])?;
        dict.set_item(
            "coef",
            pyarray1_from_f64(py, &cohort_coef.row(0).to_owned()),
        )?;
        dict.set_item(
            "cohort_intercepts",
            pyarray1_from_f64(py, cohort_intercepts),
        )?;
        dict.set_item("cohort_coef", pyarray2_from_f64(py, cohort_coef))?;
        dict.set_item("counterfactual", pyarray2_from_f64(py, counterfactual))?;
        dict.set_item("treatment_effect", pyarray2_from_f64(py, treatment_effect))?;
        dict.set_item("event_study", event_study)?;
        dict.set_item("group_means", group_means)?;
        dict.set_item("pre_rmse", self.pre_rmse)?;
        dict.set_item("penalty", self.penalty)?;
        dict.set_item("control_units", self.control_units.clone())?;
        dict.set_item("treated_units", self.treated_units.clone())?;
        dict.set_item("cohorts", self.cohorts.clone())?;
        Ok(dict.into())
    }
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

struct SyntheticDidFitResult {
    att: f64,
    unit_weights: Array2<f64>,
    time_weights: Array2<f64>,
    counterfactual: Array2<f64>,
    treatment_effect: Array2<f64>,
    pre_rmse: f64,
    unit_intercept: Array1<f64>,
    time_intercept: Array1<f64>,
    fitted_zeta_omega: Array1<f64>,
    fitted_zeta_lambda: Array1<f64>,
    control_units: Vec<usize>,
    treated_units: Vec<usize>,
    cohorts: Vec<usize>,
    treatment_info: PanelTreatmentInfo,
}

fn fit_synthetic_did_panel(
    y: &Array2<f64>,
    w: &Array2<f64>,
    zeta_omega_opt: Option<f64>,
    zeta_lambda_opt: Option<f64>,
    max_iterations: u64,
) -> PyResult<SyntheticDidFitResult> {
    let treatment_info = infer_panel_treatment(y, w)?;
    ensure_panel_has_never_treated(&treatment_info)?;
    if treatment_info.cohorts.iter().any(|cohort| *cohort == 0) {
        return Err(PyValueError::new_err(
            "SyntheticDID needs at least one pre-treatment period for every treated cohort",
        ));
    }

    let n_units = y.nrows();
    let n_periods = y.ncols();
    let n_cohorts = treatment_info.cohorts.len();
    let mut counterfactual = Array2::<f64>::from_elem((n_units, n_periods), f64::NAN);
    let mut treatment_effect = Array2::<f64>::from_elem((n_units, n_periods), f64::NAN);
    let mut unit_weight_mat = Array2::<f64>::zeros((n_cohorts, n_units));
    let mut time_weight_mat = Array2::<f64>::zeros((n_cohorts, n_periods));
    let mut unit_intercepts = Array1::<f64>::zeros(n_cohorts);
    let mut time_intercepts = Array1::<f64>::zeros(n_cohorts);
    let mut zeta_omegas = Array1::<f64>::zeros(n_cohorts);
    let mut zeta_lambdas = Array1::<f64>::zeros(n_cohorts);
    let mut att_sum = 0.0;
    let mut att_weight = 0.0;

    for (c_idx, cohort) in treatment_info.cohorts.iter().enumerate() {
        let treated_units = cohort_units(&treatment_info, *cohort);
        let control_units = &treatment_info.never_treated;
        let mut order = control_units.clone();
        order.extend_from_slice(&treated_units);
        let y_reordered = y.select(Axis(0), &order);
        let n_control = control_units.len();
        let n_treated = treated_units.len();
        let t_pre = *cohort;
        let t_post = n_periods - t_pre;

        let sigma = sdid_sigma_estimator(&y_reordered, n_control, t_pre);
        let zeta_omega = match zeta_omega_opt {
            Some(value) if value.is_finite() && value >= 0.0 => value,
            Some(_) => {
                return Err(PyValueError::new_err(
                    "zeta_omega must be finite and nonnegative",
                ))
            }
            None => ((n_treated * t_post) as f64).powf(0.25) * sigma,
        };
        let zeta_lambda = match zeta_lambda_opt {
            Some(value) if value.is_finite() && value >= 0.0 => value,
            Some(_) => {
                return Err(PyValueError::new_err(
                    "zeta_lambda must be finite and nonnegative",
                ))
            }
            None => 1e-6 * sigma,
        };

        let y_control_pre = y_reordered.slice(s![0..n_control, 0..t_pre]).to_owned();
        let y_control_post = y_reordered.slice(s![0..n_control, t_pre..]).to_owned();
        let y_treated_pre = y_reordered.slice(s![n_control.., 0..t_pre]).to_owned();

        let control_post_mean = y_control_post
            .mean_axis(Axis(1))
            .ok_or_else(|| PyValueError::new_err("failed to average control post outcomes"))?;
        let treated_pre_mean = y_treated_pre
            .mean_axis(Axis(0))
            .ok_or_else(|| PyValueError::new_err("failed to average treated pre outcomes"))?;

        let lambda_weights = fit_simplex_least_squares_weights(
            &y_control_pre,
            &control_post_mean,
            zeta_lambda,
            true,
            max_iterations,
        )?;
        let omega_design = y_control_pre.t().to_owned();
        let unit_weights = fit_simplex_least_squares_weights(
            &omega_design,
            &treated_pre_mean,
            zeta_omega,
            true,
            max_iterations,
        )?;

        let unit_intercept = simplex_intercept(&omega_design, &treated_pre_mean, &unit_weights);
        let time_intercept = simplex_intercept(&y_control_pre, &control_post_mean, &lambda_weights);
        let control_panel = y.select(Axis(0), control_units);
        let cohort_counterfactual = control_panel.t().dot(&unit_weights) + unit_intercept;

        let mut unit_weight_vec = Array1::<f64>::zeros(n_control + n_treated);
        for j in 0..n_control {
            unit_weight_vec[j] = -unit_weights[j];
        }
        for j in 0..n_treated {
            unit_weight_vec[n_control + j] = 1.0 / n_treated as f64;
        }
        let mut time_weight_vec = Array1::<f64>::zeros(n_periods);
        for t in 0..t_pre {
            time_weight_vec[t] = -lambda_weights[t];
        }
        for t in t_pre..n_periods {
            time_weight_vec[t] = 1.0 / t_post as f64;
        }
        let cohort_att = unit_weight_vec.dot(&y_reordered.dot(&time_weight_vec));
        let cohort_weight = (n_treated * t_post) as f64;
        att_sum += cohort_att * cohort_weight;
        att_weight += cohort_weight;

        for (j, unit) in control_units.iter().enumerate() {
            unit_weight_mat[[c_idx, *unit]] = unit_weights[j];
        }
        for t in 0..t_pre {
            time_weight_mat[[c_idx, t]] = lambda_weights[t];
        }
        unit_intercepts[c_idx] = unit_intercept;
        time_intercepts[c_idx] = time_intercept;
        zeta_omegas[c_idx] = zeta_omega;
        zeta_lambdas[c_idx] = zeta_lambda;

        for unit in treated_units {
            for t in 0..n_periods {
                counterfactual[[unit, t]] = cohort_counterfactual[t];
                treatment_effect[[unit, t]] = y[[unit, t]] - cohort_counterfactual[t];
            }
        }
    }

    let att = if att_weight > 0.0 {
        att_sum / att_weight
    } else {
        f64::NAN
    };
    let pre_rmse = panel_group_pre_rmse(&treatment_effect, &treatment_info);
    Ok(SyntheticDidFitResult {
        att,
        unit_weights: unit_weight_mat,
        time_weights: time_weight_mat,
        counterfactual,
        treatment_effect,
        pre_rmse,
        unit_intercept: unit_intercepts,
        time_intercept: time_intercepts,
        fitted_zeta_omega: zeta_omegas,
        fitted_zeta_lambda: zeta_lambdas,
        control_units: treatment_info.never_treated.clone(),
        treated_units: treatment_info.ever_treated.clone(),
        cohorts: treatment_info.cohorts.clone(),
        treatment_info,
    })
}

fn finite_sample_sd(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let n = finite.len();
    if n <= 1 {
        return f64::NAN;
    }
    let mean = finite.iter().sum::<f64>() / n as f64;
    let var = finite
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / ((n - 1) as f64);
    var.sqrt()
}

fn sdid_bootstrap_se(
    y: &Array2<f64>,
    w: &Array2<f64>,
    zeta_omega: Option<f64>,
    zeta_lambda: Option<f64>,
    max_iterations: u64,
    replications: usize,
    seed: Option<u64>,
) -> PyResult<f64> {
    if replications < 2 {
        return Err(PyValueError::new_err("replications must be at least 2"));
    }
    let info = infer_panel_treatment(y, w)?;
    if info.ever_treated.len() == 1 {
        return Ok(f64::NAN);
    }
    let idxs = bootstrap_indices(y.nrows(), replications, seed);
    let mut estimates = Vec::new();
    for idx in idxs {
        let y_b = take_rows(y, &idx);
        let w_b = take_rows(w, &idx);
        if let Ok(fit) =
            fit_synthetic_did_panel(&y_b, &w_b, zeta_omega, zeta_lambda, max_iterations)
        {
            if fit.att.is_finite() {
                estimates.push(fit.att);
            }
        }
    }
    if estimates.len() <= 1 {
        return Ok(f64::NAN);
    }
    Ok(((replications as f64 - 1.0) / replications as f64).sqrt() * finite_sample_sd(&estimates))
}

fn sdid_jackknife_se(
    y: &Array2<f64>,
    w: &Array2<f64>,
    zeta_omega: Option<f64>,
    zeta_lambda: Option<f64>,
    max_iterations: u64,
) -> PyResult<f64> {
    let info = infer_panel_treatment(y, w)?;
    if info.ever_treated.len() == 1 {
        return Ok(f64::NAN);
    }
    let n = y.nrows();
    if n <= 2 {
        return Ok(f64::NAN);
    }
    let mut estimates = Vec::with_capacity(n);
    for drop_i in 0..n {
        let idx: Vec<usize> = (0..n).filter(|i| *i != drop_i).collect();
        let y_j = take_rows(y, &idx);
        let w_j = take_rows(w, &idx);
        let fit = match fit_synthetic_did_panel(&y_j, &w_j, zeta_omega, zeta_lambda, max_iterations)
        {
            Ok(fit) => fit,
            Err(_) => return Ok(f64::NAN),
        };
        if !fit.att.is_finite() {
            return Ok(f64::NAN);
        }
        estimates.push(fit.att);
    }
    let mean = estimates.iter().sum::<f64>() / n as f64;
    let sumsq = estimates
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>();
    Ok((((n - 1) as f64 / n as f64) * sumsq).sqrt())
}

fn sdid_placebo_se(
    y: &Array2<f64>,
    w: &Array2<f64>,
    zeta_omega: Option<f64>,
    zeta_lambda: Option<f64>,
    max_iterations: u64,
    replications: usize,
    seed: Option<u64>,
) -> PyResult<f64> {
    if replications < 2 {
        return Err(PyValueError::new_err("replications must be at least 2"));
    }
    let info = infer_panel_treatment(y, w)?;
    let n_control = info.never_treated.len();
    let n_treated = info.ever_treated.len();
    if n_control <= n_treated {
        return Err(PyValueError::new_err(
            "must have more controls than treated units to use the placebo se",
        ));
    }
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    let mut estimates = Vec::new();
    for _ in 0..replications {
        let mut controls = info.never_treated.clone();
        controls.shuffle(&mut rng);
        let y_p = y.select(Axis(0), &controls);
        let mut w_p = Array2::<f64>::zeros((n_control, y.ncols()));
        let placebo_start = n_control - n_treated;
        for (j, treated_unit) in info.ever_treated.iter().enumerate() {
            w_p.row_mut(placebo_start + j).assign(&w.row(*treated_unit));
        }
        if let Ok(fit) =
            fit_synthetic_did_panel(&y_p, &w_p, zeta_omega, zeta_lambda, max_iterations)
        {
            if fit.att.is_finite() {
                estimates.push(fit.att);
            }
        }
    }
    if estimates.len() <= 1 {
        return Ok(f64::NAN);
    }
    Ok(((replications as f64 - 1.0) / replications as f64).sqrt() * finite_sample_sd(&estimates))
}

#[pymethods]
impl SyntheticDID {
    #[new]
    #[pyo3(signature = (zeta_omega=None, zeta_lambda=None, max_iterations=1000))]
    fn new(zeta_omega: Option<f64>, zeta_lambda: Option<f64>, max_iterations: u64) -> Self {
        Self {
            zeta_omega,
            zeta_lambda,
            max_iterations,
            att: None,
            unit_weights: None,
            time_weights: None,
            counterfactual: None,
            treatment_effect: None,
            pre_rmse: None,
            unit_intercept: None,
            time_intercept: None,
            fitted_zeta_omega: None,
            fitted_zeta_lambda: None,
            control_units: None,
            treated_units: None,
            cohorts: None,
            treatment_info: None,
            y: None,
            w: None,
        }
    }

    fn fit(&mut self, y: PyReadonlyArray2<f64>, w: PyReadonlyArray2<f64>) -> PyResult<()> {
        let y = to_array2(&y);
        let w = to_array2(&w);
        let fit = fit_synthetic_did_panel(
            &y,
            &w,
            self.zeta_omega,
            self.zeta_lambda,
            self.max_iterations,
        )?;

        self.att = Some(fit.att);
        self.unit_weights = Some(fit.unit_weights);
        self.time_weights = Some(fit.time_weights);
        self.counterfactual = Some(fit.counterfactual);
        self.treatment_effect = Some(fit.treatment_effect);
        self.pre_rmse = Some(fit.pre_rmse);
        self.unit_intercept = Some(fit.unit_intercept);
        self.time_intercept = Some(fit.time_intercept);
        self.fitted_zeta_omega = Some(fit.fitted_zeta_omega);
        self.fitted_zeta_lambda = Some(fit.fitted_zeta_lambda);
        self.control_units = Some(fit.control_units);
        self.treated_units = Some(fit.treated_units);
        self.cohorts = Some(fit.cohorts);
        self.treatment_info = Some(fit.treatment_info);
        self.y = Some(y);
        self.w = Some(w);
        Ok(())
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let att = self
            .att
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let unit_weights = self
            .unit_weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let time_weights = self
            .time_weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let treatment_info = self
            .treatment_info
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let summaries = summarize_panel_effects(y, counterfactual, treatment_info);
        let (event_study, group_means) = panel_summaries_to_dict(py, &summaries)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("att", att)?;
        dict.set_item("unit_weights", pyarray2_from_f64(py, unit_weights))?;
        dict.set_item("time_weights", pyarray2_from_f64(py, time_weights))?;
        dict.set_item("counterfactual", pyarray2_from_f64(py, counterfactual))?;
        dict.set_item("synthetic_outcome", pyarray2_from_f64(py, counterfactual))?;
        dict.set_item("treatment_effect", pyarray2_from_f64(py, treatment_effect))?;
        dict.set_item("event_study", event_study)?;
        dict.set_item("group_means", group_means)?;
        dict.set_item("pre_rmse", self.pre_rmse)?;
        dict.set_item(
            "unit_intercept",
            pyarray1_from_f64(py, self.unit_intercept.as_ref().unwrap()),
        )?;
        dict.set_item(
            "time_intercept",
            pyarray1_from_f64(py, self.time_intercept.as_ref().unwrap()),
        )?;
        dict.set_item(
            "zeta_omega",
            pyarray1_from_f64(py, self.fitted_zeta_omega.as_ref().unwrap()),
        )?;
        dict.set_item(
            "zeta_lambda",
            pyarray1_from_f64(py, self.fitted_zeta_lambda.as_ref().unwrap()),
        )?;
        dict.set_item("control_units", self.control_units.clone())?;
        dict.set_item("treated_units", self.treated_units.clone())?;
        dict.set_item("cohorts", self.cohorts.clone())?;
        Ok(dict.into())
    }

    #[pyo3(signature = (method="bootstrap", replications=200, seed=None))]
    fn vcov<'py>(
        &self,
        py: Python<'py>,
        method: &str,
        replications: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let se = self.se(method, replications, seed)?;
        let out = Array2::<f64>::from_elem((1, 1), se * se);
        Ok(pyarray2_from_f64(py, &out))
    }

    #[pyo3(signature = (method="bootstrap", replications=200, seed=None))]
    fn se(&self, method: &str, replications: usize, seed: Option<u64>) -> PyResult<f64> {
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let w = self
            .w
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        match method {
            "bootstrap" => sdid_bootstrap_se(
                y,
                w,
                self.zeta_omega,
                self.zeta_lambda,
                self.max_iterations,
                replications,
                seed,
            ),
            "jackknife" => {
                sdid_jackknife_se(y, w, self.zeta_omega, self.zeta_lambda, self.max_iterations)
            }
            "placebo" => sdid_placebo_se(
                y,
                w,
                self.zeta_omega,
                self.zeta_lambda,
                self.max_iterations,
                replications,
                seed,
            ),
            _ => Err(PyValueError::new_err(
                "method must be 'bootstrap', 'jackknife', or 'placebo'",
            )),
        }
    }

    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        Ok(pyarray2_from_f64(py, counterfactual))
    }

    fn treatment_effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        Ok(pyarray2_from_f64(py, treatment_effect))
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
        let (x_design_work, z_design_work, y_work) = match self.sample_weight.as_ref() {
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
                "not enough residual degrees of freedom for Anderson-Rubin test",
            ));
        }

        let (design_work, y_work) =
            apply_sqrt_weights(&design, &y_null, self.sample_weight.as_ref())
                .map_err(PyValueError::new_err)?;
        let params =
            solve_least_squares_vec(&design_work, &y_work).map_err(PyValueError::new_err)?;
        let residuals = y_work - &design_work.dot(&params);
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = linear_covariance(&design_work, &residuals, vcov, lags, cluster_ids.as_ref())
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
        out.set_item("test", "anderson_rubin")?;
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
