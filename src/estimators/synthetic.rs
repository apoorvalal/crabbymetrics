use super::panel::{
    cohort_units, ensure_panel_has_never_treated, infer_panel_treatment, panel_effect_dicts,
    panel_group_pre_rmse, PanelTreatmentInfo,
};
use crate::fit::optimization_success;
use crate::utils::{
    bootstrap_indices, pyarray1_from_f64, pyarray2_from_f64, take_rows, take_rows_vec, to_array1,
    to_array2,
};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

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
    if max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
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
    let solver = LBFGS::new(linesearch, 7)
        .with_tolerance_grad(1e-8)
        .map_err(|err| PyValueError::new_err(err.to_string()))?
        .with_tolerance_cost(1e-12)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let mut result = Executor::new(problem, solver)
        .configure(|state| state.param(theta0).max_iters(max_iterations))
        .run()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    if !optimization_success(result.state.get_termination_status()) {
        return Err(PyValueError::new_err(format!(
            "simplex optimization did not converge: {}",
            result.state.get_termination_status()
        )));
    }
    let theta = result
        .state
        .take_best_param()
        .ok_or_else(|| PyValueError::new_err("synthetic control optimization failed"))?;

    Ok(softmax_weights(&theta))
}

pub(crate) fn fit_simplex_least_squares_weights(
    design: &Array2<f64>,
    target: &Array1<f64>,
    zeta: f64,
    intercept: bool,
    max_iterations: u64,
) -> PyResult<Array1<f64>> {
    if max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
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

    let n_observations = design.nrows() as f64;
    let n_weights = design.ncols();
    let mut centered_design = design.clone();
    let mut centered_target = target.clone();
    if intercept {
        for column in 0..n_weights {
            let mean = centered_design.column(column).mean().unwrap_or(0.0);
            centered_design
                .column_mut(column)
                .mapv_inplace(|value| value - mean);
        }
        let mean = centered_target.mean().unwrap_or(0.0);
        centered_target.mapv_inplace(|value| value - mean);
    }

    let mut hessian = centered_design.t().dot(&centered_design) / n_observations;
    for index in 0..n_weights {
        hessian[[index, index]] += zeta * zeta;
    }
    let linear = centered_design.t().dot(&centered_target) / n_observations;
    solve_simplex_quadratic(&hessian, &linear, max_iterations)
}

fn equality_constrained_direction(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
    free: &[usize],
) -> PyResult<(Array1<f64>, f64)> {
    let dimension = free.len() + 1;
    let mut kkt = DMatrix::<f64>::zeros(dimension, dimension);
    let mut rhs = DVector::<f64>::zeros(dimension);
    for (row, source_row) in free.iter().enumerate() {
        rhs[row] = -gradient[*source_row];
        for (column, source_column) in free.iter().enumerate() {
            kkt[(row, column)] = hessian[[*source_row, *source_column]];
        }
        kkt[(row, dimension - 1)] = 1.0;
        kkt[(dimension - 1, row)] = 1.0;
    }

    let solution = kkt
        .clone()
        .lu()
        .solve(&rhs)
        .or_else(|| kkt.svd(true, true).solve(&rhs, 1e-12).ok())
        .ok_or_else(|| PyValueError::new_err("simplex quadratic KKT system could not be solved"))?;
    let mut direction = Array1::<f64>::zeros(hessian.nrows());
    for (index, source) in free.iter().enumerate() {
        direction[*source] = solution[index];
    }
    Ok((direction, solution[dimension - 1]))
}

fn solve_simplex_quadratic(
    hessian: &Array2<f64>,
    linear: &Array1<f64>,
    max_iterations: u64,
) -> PyResult<Array1<f64>> {
    let n_weights = hessian.nrows();
    let mut weights = Array1::<f64>::from_elem(n_weights, 1.0 / n_weights as f64);
    let mut is_free = vec![true; n_weights];
    let tolerance = 1e-10;

    for _ in 0..max_iterations {
        let gradient = hessian.dot(&weights) - linear;
        let free: Vec<usize> = is_free
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.then_some(index))
            .collect();
        let (direction, multiplier) = equality_constrained_direction(hessian, &gradient, &free)?;
        let direction_norm = direction.dot(&direction).sqrt();

        if direction_norm <= tolerance {
            let candidate = is_free
                .iter()
                .enumerate()
                .filter(|(_, value)| !**value)
                .map(|(index, _)| (index, gradient[index] + multiplier))
                .min_by(|left, right| left.1.total_cmp(&right.1));
            match candidate {
                Some((index, reduced_gradient)) if reduced_gradient < -tolerance => {
                    is_free[index] = true;
                    continue;
                }
                _ => {
                    weights.mapv_inplace(|value| value.max(0.0));
                    let total = weights.sum();
                    if total <= 0.0 || !total.is_finite() {
                        return Err(PyValueError::new_err(
                            "simplex quadratic solver produced invalid weights",
                        ));
                    }
                    weights /= total;
                    return Ok(weights);
                }
            }
        }

        let mut step = 1.0_f64;
        for index in &free {
            if direction[*index] < 0.0 {
                step = step.min(-weights[*index] / direction[*index]);
            }
        }
        for index in 0..n_weights {
            weights[index] += step * direction[index];
            if weights[index].abs() <= tolerance {
                weights[index] = 0.0;
            }
        }
        if step < 1.0 - tolerance {
            for index in &free {
                if weights[*index] == 0.0 && direction[*index] < 0.0 {
                    is_free[*index] = false;
                }
            }
        }
    }

    Err(PyValueError::new_err(
        "simplex quadratic optimization did not converge within max_iterations",
    ))
}

fn simplex_intercept(design: &Array2<f64>, target: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let fitted = design.dot(weights);
    (target - &fitted).mean().unwrap_or(0.0)
}

pub(crate) fn sdid_sigma_estimator(
    y_reordered: &Array2<f64>,
    n_control: usize,
    t_pre: usize,
) -> f64 {
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
        dict.set_item("converged", true)?;
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
        let (event_study, group_means) = panel_effect_dicts(py, y, counterfactual, treatment_info)?;

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
        dict.set_item("converged", true)?;
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
