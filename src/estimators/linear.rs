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
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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

    let mut row_std = Vec::with_capacity(n_control);
    for i in 0..n_control {
        let mut diffs = Vec::with_capacity(t_pre - 1);
        for t in 1..t_pre {
            diffs.push(y_reordered[[i, t]] - y_reordered[[i, t - 1]]);
        }
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let var = diffs
            .iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>()
            / diffs.len() as f64;
        row_std.push(var.sqrt());
    }

    let mean = row_std.iter().sum::<f64>() / row_std.len() as f64;
    let var = row_std
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / row_std.len() as f64;
    var.sqrt()
}

fn reorder_panel_for_treated(
    y: &Array2<f64>,
    treated_units: &[usize],
) -> PyResult<(Array2<f64>, Vec<usize>, Vec<usize>)> {
    let n_units = y.nrows();
    if treated_units.is_empty() {
        return Err(PyValueError::new_err(
            "treated_units must contain at least one unit",
        ));
    }

    let mut is_treated = vec![false; n_units];
    for &idx in treated_units {
        if idx >= n_units {
            return Err(PyValueError::new_err("treated unit index out of range"));
        }
        if is_treated[idx] {
            return Err(PyValueError::new_err(
                "treated_units must not contain duplicates",
            ));
        }
        is_treated[idx] = true;
    }

    let control_units: Vec<usize> = (0..n_units).filter(|idx| !is_treated[*idx]).collect();
    if control_units.is_empty() {
        return Err(PyValueError::new_err(
            "need at least one untreated control unit",
        ));
    }

    let mut order = control_units.clone();
    order.extend_from_slice(treated_units);
    let y_reordered = y.select(Axis(0), &order);
    Ok((y_reordered, control_units, treated_units.to_vec()))
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
pub struct InteractiveFixedEffects {
    rank: usize,
    force: i32,
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
    #[pyo3(signature = (rank=0, force=3))]
    fn new(rank: usize, force: i32) -> PyResult<Self> {
        if !(0..=3).contains(&force) {
            return Err(PyValueError::new_err("force must be one of {0, 1, 2, 3}"));
        }
        Ok(Self {
            rank,
            force,
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
        let pf = panel_factor_fit(&demeaned, self.rank)?;
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
        Ok(dict.into())
    }
}

#[pyfunction]
#[pyo3(signature = (e, rank))]
pub fn panel_factor<'py>(
    py: Python<'py>,
    e: PyReadonlyArray2<f64>,
    rank: usize,
) -> PyResult<Py<PyAny>> {
    let e = to_array2(&e);
    validate_finite_matrix("e", &e)?;
    let pf = panel_factor_fit(&e, rank)?;
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
}

#[pymethods]
impl MatrixCompletion {
    #[new]
    #[pyo3(signature = (lambda_l=None, lambda_fraction=0.25, fit_unit_effects=true, fit_time_effects=true, max_iterations=500, effect_iterations=2, tolerance=1e-6))]
    fn new(
        lambda_l: Option<f64>,
        lambda_fraction: f64,
        fit_unit_effects: bool,
        fit_time_effects: bool,
        max_iterations: usize,
        effect_iterations: usize,
        tolerance: f64,
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
        Ok(Self {
            lambda_l,
            lambda_fraction,
            fit_unit_effects,
            fit_time_effects,
            max_iterations,
            effect_iterations,
            tolerance,
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
        })
    }

    #[pyo3(signature = (y, mask=None))]
    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        mask: Option<PyReadonlyArray2<bool>>,
    ) -> PyResult<()> {
        let y_input = to_array2(&y);
        if y_input.nrows() == 0 || y_input.ncols() == 0 {
            return Err(PyValueError::new_err("y must be a non-empty 2D matrix"));
        }

        let mask_arr = match mask {
            Some(mask_py) => {
                let shape = mask_py.shape();
                if shape[0] != y_input.nrows() || shape[1] != y_input.ncols() {
                    return Err(PyValueError::new_err("mask must have the same shape as y"));
                }
                let data: Vec<bool> = mask_py.as_array().iter().copied().collect();
                Array2::from_shape_vec((shape[0], shape[1]), data)
                    .map_err(|_| PyValueError::new_err("invalid mask shape"))?
            }
            None => y_input.mapv(|value| value.is_finite()),
        };
        if !mask_arr.iter().any(|value| *value) {
            return Err(PyValueError::new_err("mask contains no observed entries"));
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
            let (updated_low_rank, updated_singular_values) = svt(&projected, threshold)?;
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

        self.completed = Some(completed);
        self.low_rank = Some(low_rank);
        self.unit_effects = Some(row_effects);
        self.time_effects = Some(col_effects);
        self.singular_values = Some(singular_values);
        self.fitted_lambda_l = Some(lambda_l);
        self.objective = self.history_objective.last().copied();
        self.iterations = Some(final_iteration);
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
        Ok(dict.into())
    }
}

#[pyclass]
pub struct HorizontalPanelRidge {
    penalty: f64,
    intercept: Option<f64>,
    coef: Option<Array1<f64>>,
    counterfactual: Option<Array1<f64>>,
    treated_outcome: Option<Array1<f64>>,
    treatment_effect: Option<Array1<f64>>,
    att: Option<f64>,
    pre_rmse: Option<f64>,
    control_units: Option<Vec<usize>>,
    treated_units: Option<Vec<usize>>,
    t_pre: Option<usize>,
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
    unit_weights: Option<Array1<f64>>,
    time_weights: Option<Array1<f64>>,
    treated_outcome: Option<Array1<f64>>,
    synthetic_outcome: Option<Array1<f64>>,
    treatment_effect: Option<Array1<f64>>,
    pre_rmse: Option<f64>,
    unit_intercept: Option<f64>,
    time_intercept: Option<f64>,
    fitted_zeta_omega: Option<f64>,
    fitted_zeta_lambda: Option<f64>,
    control_units: Option<Vec<usize>>,
    treated_units: Option<Vec<usize>>,
    t_pre: Option<usize>,
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
            intercept: None,
            coef: None,
            counterfactual: None,
            treated_outcome: None,
            treatment_effect: None,
            att: None,
            pre_rmse: None,
            control_units: None,
            treated_units: None,
            t_pre: None,
        })
    }

    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        treated_units: Vec<usize>,
        t_pre: usize,
    ) -> PyResult<()> {
        let y = to_array2(&y);
        validate_finite_matrix("y", &y)?;
        if y.nrows() < 2 {
            return Err(PyValueError::new_err("y must contain at least two units"));
        }
        if t_pre == 0 || t_pre >= y.ncols() {
            return Err(PyValueError::new_err(
                "t_pre must be positive and smaller than the number of periods",
            ));
        }

        let (y_reordered, control_units, treated_units) =
            reorder_panel_for_treated(&y, &treated_units)?;
        let n_control = control_units.len();
        let t_post = y.ncols() - t_pre;

        let control_panel = y_reordered.slice(s![0..n_control, ..]).to_owned();
        let treated_outcome = y_reordered
            .slice(s![n_control.., ..])
            .mean_axis(Axis(0))
            .ok_or_else(|| PyValueError::new_err("failed to average treated outcomes"))?;

        let x_pre = control_panel.slice(s![.., 0..t_pre]).t().to_owned();
        let y_pre = treated_outcome.slice(s![0..t_pre]).to_owned();
        let (intercept, coef) = fit_ridge_with_intercept(&x_pre, &y_pre, self.penalty)?;
        let x_all = control_panel.t().to_owned();
        let counterfactual = x_all.dot(&coef) + intercept;
        let treatment_effect = &treated_outcome - &counterfactual;
        let post_effect = treatment_effect.slice(s![t_pre..]).to_owned();
        let att = post_effect.mean().unwrap_or(0.0);
        let pre_rmse = treatment_effect
            .slice(s![0..t_pre])
            .mapv(|value| value * value)
            .mean()
            .unwrap_or(0.0)
            .sqrt();

        debug_assert_eq!(post_effect.len(), t_post);
        self.intercept = Some(intercept);
        self.coef = Some(coef);
        self.counterfactual = Some(counterfactual);
        self.treated_outcome = Some(treated_outcome);
        self.treatment_effect = Some(treatment_effect);
        self.att = Some(att);
        self.pre_rmse = Some(pre_rmse);
        self.control_units = Some(control_units);
        self.treated_units = Some(treated_units);
        self.t_pre = Some(t_pre);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        Ok(pyarray1_from_f64(py, counterfactual))
    }

    fn treatment_effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        Ok(pyarray1_from_f64(py, treatment_effect))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let att = self
            .att
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let treated_outcome = self
            .treated_outcome
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("HorizontalPanelRidge model is not fitted"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("att", att)?;
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("counterfactual", pyarray1_from_f64(py, counterfactual))?;
        dict.set_item("treated_outcome", pyarray1_from_f64(py, treated_outcome))?;
        dict.set_item("treatment_effect", pyarray1_from_f64(py, treatment_effect))?;
        dict.set_item("pre_rmse", self.pre_rmse)?;
        dict.set_item("penalty", self.penalty)?;
        dict.set_item("control_units", self.control_units.clone())?;
        dict.set_item("treated_units", self.treated_units.clone())?;
        dict.set_item("t_pre", self.t_pre)?;
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
            treated_outcome: None,
            synthetic_outcome: None,
            treatment_effect: None,
            pre_rmse: None,
            unit_intercept: None,
            time_intercept: None,
            fitted_zeta_omega: None,
            fitted_zeta_lambda: None,
            control_units: None,
            treated_units: None,
            t_pre: None,
        }
    }

    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        treated_units: Vec<usize>,
        t_pre: usize,
    ) -> PyResult<()> {
        let y = to_array2(&y);
        validate_finite_matrix("y", &y)?;
        if y.nrows() < 2 {
            return Err(PyValueError::new_err("y must contain at least two units"));
        }
        if t_pre == 0 || t_pre >= y.ncols() {
            return Err(PyValueError::new_err(
                "t_pre must be positive and smaller than the number of periods",
            ));
        }

        let (y_reordered, control_units, treated_units) =
            reorder_panel_for_treated(&y, &treated_units)?;
        let n_control = control_units.len();
        let n_treated = treated_units.len();
        let t_post = y.ncols() - t_pre;

        let sigma = sdid_sigma_estimator(&y_reordered, n_control, t_pre);
        let zeta_omega = match self.zeta_omega {
            Some(value) if value.is_finite() && value >= 0.0 => value,
            Some(_) => {
                return Err(PyValueError::new_err(
                    "zeta_omega must be finite and nonnegative",
                ))
            }
            None => ((n_treated * t_post) as f64).powf(0.25) * sigma,
        };
        let zeta_lambda = match self.zeta_lambda {
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
            self.max_iterations,
        )?;
        let omega_design = y_control_pre.t().to_owned();
        let unit_weights = fit_simplex_least_squares_weights(
            &omega_design,
            &treated_pre_mean,
            zeta_omega,
            true,
            self.max_iterations,
        )?;

        let unit_intercept = simplex_intercept(&omega_design, &treated_pre_mean, &unit_weights);
        let time_intercept = simplex_intercept(&y_control_pre, &control_post_mean, &lambda_weights);

        let mut unit_weight_vec = Array1::<f64>::zeros(y.nrows());
        for i in 0..n_control {
            unit_weight_vec[i] = -unit_weights[i];
        }
        for i in n_control..y.nrows() {
            unit_weight_vec[i] = 1.0 / n_treated as f64;
        }

        let mut time_weight_vec = Array1::<f64>::zeros(y.ncols());
        for t in 0..t_pre {
            time_weight_vec[t] = -lambda_weights[t];
        }
        for t in t_pre..y.ncols() {
            time_weight_vec[t] = 1.0 / t_post as f64;
        }

        let att = unit_weight_vec.dot(&y_reordered.dot(&time_weight_vec));
        let treated_outcome = y_reordered
            .slice(s![n_control.., ..])
            .mean_axis(Axis(0))
            .ok_or_else(|| PyValueError::new_err("failed to average treated outcomes"))?;
        let control_panel = y_reordered.slice(s![0..n_control, ..]).to_owned();
        let synthetic_outcome = control_panel.t().dot(&unit_weights) + unit_intercept;
        let treatment_effect = &treated_outcome - &synthetic_outcome;
        let pre_rmse = treatment_effect
            .slice(s![0..t_pre])
            .mapv(|value| value * value)
            .mean()
            .unwrap_or(0.0)
            .sqrt();

        self.att = Some(att);
        self.unit_weights = Some(unit_weights);
        self.time_weights = Some(lambda_weights);
        self.treated_outcome = Some(treated_outcome);
        self.synthetic_outcome = Some(synthetic_outcome);
        self.treatment_effect = Some(treatment_effect);
        self.pre_rmse = Some(pre_rmse);
        self.unit_intercept = Some(unit_intercept);
        self.time_intercept = Some(time_intercept);
        self.fitted_zeta_omega = Some(zeta_omega);
        self.fitted_zeta_lambda = Some(zeta_lambda);
        self.control_units = Some(control_units);
        self.treated_units = Some(treated_units);
        self.t_pre = Some(t_pre);
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
        let treated_outcome = self
            .treated_outcome
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let synthetic_outcome = self
            .synthetic_outcome
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("att", att)?;
        dict.set_item("unit_weights", pyarray1_from_f64(py, unit_weights))?;
        dict.set_item("time_weights", pyarray1_from_f64(py, time_weights))?;
        dict.set_item("treated_outcome", pyarray1_from_f64(py, treated_outcome))?;
        dict.set_item(
            "synthetic_outcome",
            pyarray1_from_f64(py, synthetic_outcome),
        )?;
        dict.set_item("treatment_effect", pyarray1_from_f64(py, treatment_effect))?;
        dict.set_item("pre_rmse", self.pre_rmse)?;
        dict.set_item("unit_intercept", self.unit_intercept)?;
        dict.set_item("time_intercept", self.time_intercept)?;
        dict.set_item("zeta_omega", self.fitted_zeta_omega)?;
        dict.set_item("zeta_lambda", self.fitted_zeta_lambda)?;
        dict.set_item("control_units", self.control_units.clone())?;
        dict.set_item("treated_units", self.treated_units.clone())?;
        dict.set_item("t_pre", self.t_pre)?;
        Ok(dict.into())
    }

    fn treatment_effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let treatment_effect = self
            .treatment_effect
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("SyntheticDID model is not fitted"))?;
        Ok(pyarray1_from_f64(py, treatment_effect))
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
