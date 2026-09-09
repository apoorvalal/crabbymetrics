use crate::fit::{optimization_success, FitDiagnostics};
use crate::utils::{pyarray1_from_f64, scale_rows, solve_least_squares_vec, to_array2};
use crate::validation::validate_weights;
use argmin::core::{CostFunction, Error as ArgminError, Executor, Gradient, State};
use argmin::solver::{
    linesearch::MoreThuenteLineSearch,
    quasinewton::{BFGS, LBFGS},
};
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BalanceObjective {
    Entropy,
    Quadratic,
    CressieRead,
}

impl BalanceObjective {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "entropy" => Ok(Self::Entropy),
            "quadratic" => Ok(Self::Quadratic),
            "cressie_read" | "power_divergence" => Ok(Self::CressieRead),
            _ => {
                Err("objective must be one of {'entropy', 'quadratic', 'cressie_read'}".to_string())
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SolveMode {
    Auto,
    GaussNewton,
    Lbfgs,
    Bfgs,
}

impl SolveMode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "auto" => Ok(Self::Auto),
            "gauss_newton" => Ok(Self::GaussNewton),
            "lbfgs" => Ok(Self::Lbfgs),
            "bfgs" => Ok(Self::Bfgs),
            _ => Err("solver must be one of {'auto', 'gauss_newton', 'lbfgs', 'bfgs'}".to_string()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EntropyPhase {
    Relaxed,
    Bounded,
}

fn normalize_weights(name: &str, weights: &Array1<f64>) -> Result<Array1<f64>, String> {
    validate_weights(name, weights, weights.len())?;
    let total = weights.sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(format!("{} must sum to a positive finite value", name));
    }
    Ok(weights / total)
}

fn weighted_mean(
    x: &Array2<f64>,
    weights: Option<&Array1<f64>>,
    weight_name: &str,
) -> Result<Array1<f64>, String> {
    if x.nrows() == 0 {
        return Err("need at least one observation".to_string());
    }

    match weights {
        Some(weights) => {
            validate_weights(weight_name, weights, x.nrows())?;
            let total = weights.sum();
            if total <= 0.0 {
                return Err(format!("{} must sum to a positive value", weight_name));
            }
            Ok(x.t().dot(weights) / total)
        }
        None => x
            .mean_axis(Axis(0))
            .ok_or_else(|| "failed to compute mean".to_string()),
    }
}

fn minmax_scale_fit(
    covariates: &Array2<f64>,
    target_covariates: &Array2<f64>,
) -> (Array2<f64>, Array2<f64>) {
    let mut cov_scaled = Array2::<f64>::zeros(covariates.raw_dim());
    let mut target_scaled = Array2::<f64>::zeros(target_covariates.raw_dim());

    for j in 0..covariates.ncols() {
        let source_col = covariates.column(j);
        let min_value = source_col
            .iter()
            .fold(f64::INFINITY, |acc, value| acc.min(*value));
        let max_value = source_col
            .iter()
            .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));
        let range = max_value - min_value;

        if !range.is_finite() || range.abs() <= 1e-12 {
            continue;
        }

        for i in 0..covariates.nrows() {
            cov_scaled[[i, j]] = (covariates[[i, j]] - min_value) / range;
        }
        for i in 0..target_covariates.nrows() {
            target_scaled[[i, j]] = (target_covariates[[i, j]] - min_value) / range;
        }
    }

    (cov_scaled, target_scaled)
}

fn balance_design(covariates: &Array2<f64>, target_mean: &Array1<f64>) -> Array2<f64> {
    let n = covariates.nrows();
    let p = covariates.ncols();
    let mut z = Array2::<f64>::zeros((n, p + 1));
    z.column_mut(0).fill(1.0);
    for j in 0..p {
        for i in 0..n {
            z[[i, j + 1]] = covariates[[i, j]] - target_mean[j];
        }
    }
    z
}

fn initial_beta(
    z: &Array2<f64>,
    objective: BalanceObjective,
    baseline_weights: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let rhs = {
        let mut rhs = Array1::<f64>::zeros(z.ncols());
        rhs[0] = 1.0;
        rhs
    };

    match objective {
        BalanceObjective::Entropy => Ok(Array1::<f64>::zeros(z.ncols())),
        BalanceObjective::CressieRead => Ok(Array1::<f64>::zeros(z.ncols())),
        BalanceObjective::Quadratic => match baseline_weights {
            Some(baseline_weights) => {
                let weighted_z = scale_rows(z, baseline_weights)?;
                let lhs = z.t().dot(&weighted_z);
                let rhs_shift = &rhs - &z.t().dot(baseline_weights);
                solve_least_squares_vec(&lhs, &rhs_shift)
            }
            None => {
                let lhs = z.t().dot(z);
                solve_least_squares_vec(&lhs, &rhs)
            }
        },
    }
}

fn slack_vector(beta_tail: &Array1<f64>, l2_norm: f64) -> Array1<f64> {
    if l2_norm <= 0.0 || beta_tail.is_empty() {
        return Array1::<f64>::zeros(beta_tail.len());
    }
    let norm = beta_tail.dot(beta_tail).sqrt();
    if norm <= 1e-12 {
        return Array1::<f64>::zeros(beta_tail.len());
    }
    beta_tail.mapv(|value| l2_norm * value / norm)
}

fn slack_jacobian(beta_tail: &Array1<f64>, l2_norm: f64) -> Array2<f64> {
    let p = beta_tail.len();
    if l2_norm <= 0.0 || p == 0 {
        return Array2::<f64>::zeros((p, p));
    }

    let norm = beta_tail.dot(beta_tail).sqrt();
    if norm <= 1e-12 {
        return Array2::<f64>::zeros((p, p));
    }

    let mut jac = Array2::<f64>::zeros((p, p));
    for i in 0..p {
        jac[[i, i]] = l2_norm / norm;
    }

    let outer = beta_tail
        .to_owned()
        .insert_axis(Axis(1))
        .dot(&beta_tail.to_owned().insert_axis(Axis(0)));
    jac - outer * (l2_norm / norm.powi(3))
}

struct CalibrationSystem {
    z: Array2<f64>,
    objective: BalanceObjective,
    baseline_weights: Option<Array1<f64>>,
    min_weight: f64,
    max_weight: f64,
    l2_norm: f64,
    divergence_power: f64,
    dual_ridge: f64,
    entropy_phase: EntropyPhase,
}

impl CalibrationSystem {
    fn baseline_value(&self, idx: usize) -> f64 {
        self.baseline_weights
            .as_ref()
            .map(|weights| weights[idx])
            .unwrap_or(1.0 / self.z.nrows() as f64)
    }

    fn cressie_read_link(&self, linear: f64) -> (f64, f64) {
        let lambda = self.divergence_power;
        if lambda.abs() <= 1e-10 {
            let exponent = linear.clamp(-745.0, 700.0);
            let value = exponent.exp();
            return (value, value);
        }

        let base = 1.0 + lambda * linear;
        if base <= 0.0 || !base.is_finite() {
            return (0.0, 0.0);
        }
        let value = base.powf(1.0 / lambda);
        let slope = value / base;
        (value, slope)
    }

    fn weights_and_slope(&self, beta: &Array1<f64>) -> Result<(Array1<f64>, Array1<f64>), String> {
        let linear = self.z.dot(beta);
        let mut weights = Array1::<f64>::zeros(linear.len());
        let mut slope = Array1::<f64>::zeros(linear.len());

        match self.objective {
            BalanceObjective::Entropy => {
                let upper_log = match self.entropy_phase {
                    EntropyPhase::Relaxed => 1e8f64.ln(),
                    EntropyPhase::Bounded => self.max_weight.ln(),
                };
                let lower_log = match self.entropy_phase {
                    EntropyPhase::Relaxed => f64::NEG_INFINITY,
                    EntropyPhase::Bounded => {
                        if self.min_weight > 0.0 {
                            self.min_weight.ln()
                        } else {
                            f64::NEG_INFINITY
                        }
                    }
                };

                for i in 0..linear.len() {
                    let raw_log = match self.baseline_weights.as_ref() {
                        Some(baseline) => baseline[i].ln() + linear[i] - 1.0,
                        None => linear[i],
                    };
                    let clipped = raw_log.clamp(lower_log, upper_log);
                    let weight = clipped.exp();
                    weights[i] = weight;
                    if raw_log > lower_log && raw_log < upper_log {
                        slope[i] = weight;
                    }
                }
            }
            BalanceObjective::CressieRead => {
                for i in 0..linear.len() {
                    let baseline = self.baseline_value(i);
                    let (link, link_slope) = self.cressie_read_link(linear[i]);
                    let raw = baseline * link;
                    let clipped = raw.clamp(self.min_weight, self.max_weight);
                    weights[i] = clipped;
                    if raw > self.min_weight && raw < self.max_weight {
                        slope[i] = baseline * link_slope;
                    }
                }
            }
            BalanceObjective::Quadratic => {
                for i in 0..linear.len() {
                    let raw = match self.baseline_weights.as_ref() {
                        Some(baseline) => baseline[i] * (linear[i] + 1.0),
                        None => linear[i],
                    };
                    let clipped = raw.clamp(self.min_weight, self.max_weight);
                    weights[i] = clipped;
                    if raw > self.min_weight && raw < self.max_weight {
                        slope[i] = match self.baseline_weights.as_ref() {
                            Some(baseline) => baseline[i],
                            None => 1.0,
                        };
                    }
                }
            }
        }

        Ok((weights, slope))
    }

    fn residual_and_jacobian(
        &self,
        beta: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>, Array1<f64>), String> {
        let (weights, slope) = self.weights_and_slope(beta)?;

        let mut residual = self.z.t().dot(&weights);
        residual[0] -= 1.0;

        let beta_tail = beta.slice(s![1..]).to_owned();
        let slack = slack_vector(&beta_tail, self.l2_norm);
        for j in 0..slack.len() {
            residual[j + 1] += slack[j];
        }

        let weighted_z = scale_rows(&self.z, &slope)?;
        let mut jacobian = self.z.t().dot(&weighted_z);
        let slack_j = slack_jacobian(&beta_tail, self.l2_norm);
        for r in 0..slack_j.nrows() {
            for c in 0..slack_j.ncols() {
                jacobian[[r + 1, c + 1]] += slack_j[[r, c]];
            }
        }

        Ok((residual, jacobian, weights))
    }

    fn objective(&self, beta: &Array1<f64>) -> Result<f64, String> {
        let (residual, _, _) = self.residual_and_jacobian(beta)?;
        let mut value = 0.5 * residual.dot(&residual);
        if self.dual_ridge > 0.0 && beta.len() > 1 {
            value += 0.5 * self.dual_ridge * beta.slice(s![1..]).dot(&beta.slice(s![1..]));
        }
        Ok(value)
    }

    fn residual_norm(&self, beta: &Array1<f64>) -> Result<f64, String> {
        let (residual, _, _) = self.residual_and_jacobian(beta)?;
        Ok(residual.dot(&residual).sqrt())
    }
}

struct CalibrationObjective<'a> {
    system: &'a CalibrationSystem,
}

impl CostFunction for CalibrationObjective<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, ArgminError> {
        self.system
            .objective(param)
            .map_err(|err| ArgminError::msg(err.to_string()))
    }
}

impl Gradient for CalibrationObjective<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        let (residual, jacobian, _) = self
            .system
            .residual_and_jacobian(param)
            .map_err(|err| ArgminError::msg(err.to_string()))?;
        let mut gradient = jacobian.t().dot(&residual);
        if self.system.dual_ridge > 0.0 && gradient.len() > 1 {
            for j in 1..gradient.len() {
                gradient[j] += self.system.dual_ridge * param[j];
            }
        }
        Ok(gradient)
    }
}

struct CalibrationFit {
    beta: Array1<f64>,
    weights: Array1<f64>,
    criterion: f64,
    residual_norm: f64,
    diagnostics: FitDiagnostics,
}

fn calibration_fit(
    beta: Array1<f64>,
    weights: Array1<f64>,
    criterion: f64,
    residual_norm: f64,
    iterations: usize,
    converged: bool,
    termination_reason: impl Into<String>,
) -> CalibrationFit {
    CalibrationFit {
        beta,
        weights,
        criterion,
        residual_norm,
        diagnostics: FitDiagnostics::new(
            converged,
            iterations as u64,
            termination_reason,
            Some(criterion),
        ),
    }
}

fn solve_gauss_newton(
    system: &CalibrationSystem,
    beta0: &Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<CalibrationFit, String> {
    let mut beta = beta0.clone();
    let mut iter = 0usize;

    loop {
        let (residual, jacobian, weights) = system.residual_and_jacobian(&beta)?;
        let current_criterion = system.objective(&beta)?;
        let residual_norm = residual.dot(&residual).sqrt();
        let mut normal = jacobian.t().dot(&jacobian);
        for i in 0..normal.nrows() {
            normal[[i, i]] += 1e-8;
        }
        let mut rhs = jacobian.t().dot(&residual);
        if system.dual_ridge > 0.0 {
            for j in 1..normal.nrows() {
                normal[[j, j]] += system.dual_ridge;
                rhs[j] += system.dual_ridge * beta[j];
            }
        }
        let step = solve_least_squares_vec(&normal, &rhs)?;
        let step_norm = step.dot(&step).sqrt();

        if residual_norm <= tolerance {
            return Ok(calibration_fit(
                beta,
                weights,
                current_criterion,
                residual_norm,
                iter,
                true,
                "Scaled residual tolerance reached",
            ));
        }

        if step_norm <= tolerance || iter >= max_iterations {
            let reason = if iter >= max_iterations {
                "Maximum number of iterations reached"
            } else {
                "Step tolerance reached without scaled residual convergence"
            };
            return Ok(calibration_fit(
                beta,
                weights,
                current_criterion,
                residual_norm,
                iter,
                false,
                reason,
            ));
        }

        let mut alpha = 1.0;
        let mut accepted_beta = None;
        let mut accepted_weights = None;
        let mut accepted_criterion = current_criterion;
        let mut accepted_residual_norm = residual_norm;

        while alpha >= 1e-8 {
            let candidate = &beta - &(step.mapv(|value| alpha * value));
            let (candidate_residual, _, candidate_weights) =
                system.residual_and_jacobian(&candidate)?;
            let candidate_criterion = system.objective(&candidate)?;
            if candidate_criterion < current_criterion {
                accepted_residual_norm = candidate_residual.dot(&candidate_residual).sqrt();
                accepted_beta = Some(candidate);
                accepted_weights = Some(candidate_weights);
                accepted_criterion = candidate_criterion;
                break;
            }
            alpha *= 0.5;
        }

        let next_beta = match accepted_beta {
            Some(candidate) => candidate,
            None => {
                return Ok(calibration_fit(
                    beta,
                    weights,
                    current_criterion,
                    residual_norm,
                    iter,
                    false,
                    "Line search failed before scaled residual convergence",
                ));
            }
        };

        iter += 1;
        if (current_criterion - accepted_criterion).abs() <= tolerance || iter >= max_iterations {
            let converged = accepted_residual_norm <= tolerance;
            let reason = if converged {
                "Scaled residual tolerance reached"
            } else if iter >= max_iterations {
                "Maximum number of iterations reached"
            } else {
                "Objective tolerance reached without scaled residual convergence"
            };
            return Ok(calibration_fit(
                next_beta,
                accepted_weights.expect("accepted weights missing"),
                accepted_criterion,
                accepted_residual_norm,
                iter,
                converged,
                reason,
            ));
        }

        beta = next_beta;
    }
}

fn solve_bfgs(
    system: &CalibrationSystem,
    beta0: &Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<CalibrationFit, String> {
    let problem = CalibrationObjective { system };
    let linesearch = MoreThuenteLineSearch::new();
    let solver = BFGS::new(linesearch)
        .with_tolerance_grad(tolerance)
        .map_err(|err| err.to_string())?
        .with_tolerance_cost(tolerance)
        .map_err(|err| err.to_string())?;
    let inv_hessian = identity_matrix(beta0.len());

    let mut result = Executor::new(problem, solver)
        .configure(|state| {
            state
                .param(beta0.clone())
                .inv_hessian(inv_hessian)
                .max_iters(max_iterations as u64)
        })
        .run()
        .map_err(|err| err.to_string())?;

    let nit = result.state.get_iter() as usize;
    let converged_by_status = optimization_success(result.state.get_termination_status());
    let status_message = result.state.get_termination_status().to_string();
    let beta = result
        .state
        .take_best_param()
        .unwrap_or_else(|| beta0.clone());
    let (_, _, weights) = system.residual_and_jacobian(&beta)?;
    let criterion = system.objective(&beta)?;
    let residual_norm = system.residual_norm(&beta)?;

    let converged = converged_by_status && residual_norm <= tolerance;
    let termination_reason = if converged {
        "Scaled residual tolerance reached".to_string()
    } else if converged_by_status {
        format!("{status_message}, but scaled residual tolerance was not reached")
    } else {
        status_message
    };
    Ok(calibration_fit(
        beta,
        weights,
        criterion,
        residual_norm,
        nit,
        converged,
        termination_reason,
    ))
}

fn solve_lbfgs(
    system: &CalibrationSystem,
    beta0: &Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<CalibrationFit, String> {
    let problem = CalibrationObjective { system };
    let linesearch = MoreThuenteLineSearch::new();
    let solver = LBFGS::new(linesearch, 10)
        .with_tolerance_grad(tolerance)
        .map_err(|err| err.to_string())?
        .with_tolerance_cost((tolerance * tolerance).max(f64::EPSILON))
        .map_err(|err| err.to_string())?;

    let mut result = Executor::new(problem, solver)
        .configure(|state| state.param(beta0.clone()).max_iters(max_iterations as u64))
        .run()
        .map_err(|err| err.to_string())?;

    let nit = result.state.get_iter() as usize;
    let converged_by_status = optimization_success(result.state.get_termination_status());
    let status_message = result.state.get_termination_status().to_string();
    let beta = result
        .state
        .take_best_param()
        .unwrap_or_else(|| beta0.clone());
    let (_, _, weights) = system.residual_and_jacobian(&beta)?;
    let criterion = system.objective(&beta)?;
    let residual_norm = system.residual_norm(&beta)?;

    let converged = converged_by_status && residual_norm <= tolerance;
    let termination_reason = if converged {
        "Scaled residual tolerance reached".to_string()
    } else if converged_by_status {
        format!("{status_message}, but scaled residual tolerance was not reached")
    } else {
        status_message
    };
    Ok(calibration_fit(
        beta,
        weights,
        criterion,
        residual_norm,
        nit,
        converged,
        termination_reason,
    ))
}

fn solve_system(
    system: &CalibrationSystem,
    solver: SolveMode,
    beta0: &Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> Result<(CalibrationFit, String), String> {
    match solver {
        SolveMode::GaussNewton => Ok((
            solve_gauss_newton(system, beta0, max_iterations, tolerance)?,
            "gauss_newton".to_string(),
        )),
        SolveMode::Lbfgs => Ok((
            solve_lbfgs(system, beta0, max_iterations, tolerance)?,
            "lbfgs".to_string(),
        )),
        SolveMode::Bfgs => Ok((
            solve_bfgs(system, beta0, max_iterations, tolerance)?,
            "bfgs".to_string(),
        )),
        SolveMode::Auto => {
            let gauss_newton = solve_gauss_newton(system, beta0, max_iterations, tolerance)?;
            if gauss_newton.diagnostics.converged {
                return Ok((gauss_newton, "gauss_newton".to_string()));
            }

            if let Ok(lbfgs) = solve_lbfgs(system, &gauss_newton.beta, max_iterations, tolerance) {
                if lbfgs.criterion <= gauss_newton.criterion {
                    return Ok((lbfgs, "lbfgs".to_string()));
                }
            }

            match solve_bfgs(system, &gauss_newton.beta, max_iterations, tolerance) {
                Ok(bfgs) if bfgs.criterion <= gauss_newton.criterion => {
                    Ok((bfgs, "bfgs".to_string()))
                }
                Ok(_) | Err(_) => Ok((gauss_newton, "gauss_newton".to_string())),
            }
        }
    }
}

fn effective_sample_size(weights: &Array1<f64>) -> f64 {
    let sum_sq = weights.mapv(|value| value * value).sum();
    if sum_sq <= 0.0 {
        0.0
    } else {
        1.0 / sum_sq
    }
}

pub(super) struct QuadraticCalibrationFit {
    pub weights: Array1<f64>,
    pub effective_sample_size: f64,
    pub max_abs_balance: f64,
    pub converged: bool,
    pub iterations: usize,
}

/// Internal exact quadratic-calibration path shared by dynamic covariate balancing.
///
/// This deliberately reuses the same scaled design, dual weight map, bounded solver,
/// and convergence checks as `BalancingWeights(objective="quadratic")`. The target
/// weights are normalized before fitting, and returned source weights sum to one.
pub(super) fn fit_quadratic_calibration(
    covariates: &Array2<f64>,
    target_covariates: &Array2<f64>,
    target_weights: Option<&Array1<f64>>,
    autoscale: bool,
    max_weight: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Result<QuadraticCalibrationFit, String> {
    if covariates.ncols() != target_covariates.ncols() {
        return Err(
            "covariates and target_covariates must have the same number of columns".to_string(),
        );
    }
    if covariates.nrows() == 0 || target_covariates.nrows() == 0 {
        return Err("covariates and target_covariates must both have at least one row".to_string());
    }
    if !max_weight.is_finite() || max_weight <= 0.0 || max_weight > 1.0 {
        return Err("max_weight must lie in (0, 1]".to_string());
    }
    let uniform_weight = 1.0 / covariates.nrows() as f64;
    if max_weight < uniform_weight {
        return Err(format!(
            "max_weight cannot be smaller than the uniform source weight {uniform_weight}"
        ));
    }

    let normalized_target_weights = match target_weights {
        Some(weights) => Some(normalize_weights("target_weights", weights)?),
        None => None,
    };
    if let Some(weights) = normalized_target_weights.as_ref() {
        validate_weights("target_weights", weights, target_covariates.nrows())?;
    }

    let (covariates_fit, target_covariates_fit) = if autoscale {
        minmax_scale_fit(covariates, target_covariates)
    } else {
        (covariates.clone(), target_covariates.clone())
    };
    let active_columns = (0..covariates_fit.ncols())
        .filter(|column| {
            let values = covariates_fit.column(*column);
            let minimum = values
                .iter()
                .fold(f64::INFINITY, |current, value| current.min(*value));
            let maximum = values
                .iter()
                .fold(f64::NEG_INFINITY, |current, value| current.max(*value));
            maximum - minimum > 1e-12
        })
        .collect::<Vec<_>>();
    let covariates_fit = covariates_fit.select(Axis(1), &active_columns);
    let target_covariates_fit = target_covariates_fit.select(Axis(1), &active_columns);
    let target_mean_fit = weighted_mean(
        &target_covariates_fit,
        normalized_target_weights.as_ref(),
        "target_weights",
    )?;
    let z = balance_design(&covariates_fit, &target_mean_fit);
    let beta0 = initial_beta(&z, BalanceObjective::Quadratic, None)?;
    let system = CalibrationSystem {
        z,
        objective: BalanceObjective::Quadratic,
        baseline_weights: None,
        min_weight: 0.0,
        max_weight,
        l2_norm: 0.0,
        divergence_power: 0.0,
        dual_ridge: 0.0,
        entropy_phase: EntropyPhase::Bounded,
    };
    let (fit, _) = solve_system(&system, SolveMode::Auto, &beta0, max_iterations, tolerance)?;

    let weights = normalize_weights("weights", &fit.weights)?;
    let weighted_mean_original = weighted_mean(covariates, Some(&weights), "weights")?;
    let target_mean_original = weighted_mean(
        target_covariates,
        normalized_target_weights.as_ref(),
        "target_weights",
    )?;
    let mean_diff = weighted_mean_original - target_mean_original;
    let max_abs_balance = mean_diff
        .iter()
        .fold(0.0_f64, |current, value| current.max(value.abs()));
    let converged = fit.diagnostics.converged
        && weights
            .iter()
            .all(|value| value.is_finite() && *value >= -1e-8 && *value <= max_weight + 1e-8);

    Ok(QuadraticCalibrationFit {
        effective_sample_size: effective_sample_size(&weights),
        max_abs_balance,
        converged,
        iterations: fit.diagnostics.iterations as usize,
        weights,
    })
}

fn identity_matrix(n: usize) -> Array2<f64> {
    let mut eye = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        eye[[i, i]] = 1.0;
    }
    eye
}

#[pyclass]
pub struct BalancingWeights {
    objective: String,
    solver: String,
    autoscale: bool,
    min_weight: f64,
    max_weight: f64,
    l2_norm: f64,
    max_iterations: usize,
    tolerance: f64,
    divergence_power: f64,
    dual_ridge: f64,
    weights: Option<Array1<f64>>,
    beta: Option<Array1<f64>>,
    covariates: Option<Array2<f64>>,
    target_covariates: Option<Array2<f64>>,
    weighted_mean: Option<Array1<f64>>,
    target_mean: Option<Array1<f64>>,
    success: bool,
    nit: usize,
    solver_used: Option<String>,
    criterion: Option<f64>,
    residual_norm: Option<f64>,
    diagnostics: Option<FitDiagnostics>,
}

#[pymethods]
impl BalancingWeights {
    #[new]
    #[pyo3(signature = (objective="quadratic", solver="auto", autoscale=false, min_weight=0.0, max_weight=1.0, l2_norm=0.0, max_iterations=200, tolerance=1e-8, divergence_power=0.0, dual_ridge=0.0))]
    fn new(
        objective: &str,
        solver: &str,
        autoscale: bool,
        min_weight: f64,
        max_weight: f64,
        l2_norm: f64,
        max_iterations: usize,
        tolerance: f64,
        divergence_power: f64,
        dual_ridge: f64,
    ) -> PyResult<Self> {
        BalanceObjective::parse(objective).map_err(PyValueError::new_err)?;
        SolveMode::parse(solver).map_err(PyValueError::new_err)?;
        if !min_weight.is_finite() || min_weight < 0.0 {
            return Err(PyValueError::new_err(
                "min_weight must be finite and nonnegative",
            ));
        }
        if !max_weight.is_finite() || max_weight <= 0.0 {
            return Err(PyValueError::new_err(
                "max_weight must be positive and finite",
            ));
        }
        if !l2_norm.is_finite() || l2_norm < 0.0 {
            return Err(PyValueError::new_err(
                "l2_norm must be finite and nonnegative",
            ));
        }
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be positive and finite",
            ));
        }
        if !divergence_power.is_finite() {
            return Err(PyValueError::new_err(
                "divergence_power must be a finite float",
            ));
        }
        if !dual_ridge.is_finite() || dual_ridge < 0.0 {
            return Err(PyValueError::new_err(
                "dual_ridge must be finite and nonnegative",
            ));
        }

        Ok(Self {
            objective: objective.to_string(),
            solver: solver.to_string(),
            autoscale,
            min_weight,
            max_weight,
            l2_norm,
            max_iterations,
            tolerance,
            divergence_power,
            dual_ridge,
            weights: None,
            beta: None,
            covariates: None,
            target_covariates: None,
            weighted_mean: None,
            target_mean: None,
            success: false,
            nit: 0,
            solver_used: None,
            criterion: None,
            residual_norm: None,
            diagnostics: None,
        })
    }

    #[pyo3(signature = (covariates, target_covariates, baseline_weights=None, target_weights=None))]
    fn fit(
        &mut self,
        covariates: PyReadonlyArray2<f64>,
        target_covariates: PyReadonlyArray2<f64>,
        baseline_weights: Option<Vec<f64>>,
        target_weights: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.weights = None;
        self.beta = None;
        self.covariates = None;
        self.target_covariates = None;
        self.weighted_mean = None;
        self.target_mean = None;
        self.success = false;
        self.nit = 0;
        self.solver_used = None;
        self.criterion = None;
        self.residual_norm = None;
        self.diagnostics = None;
        let covariates = to_array2(&covariates);
        let target_covariates = to_array2(&target_covariates);
        crate::validation::validate_finite("covariates", &covariates)
            .map_err(PyValueError::new_err)?;
        crate::validation::validate_finite("target_covariates", &target_covariates)
            .map_err(PyValueError::new_err)?;
        if covariates.ncols() != target_covariates.ncols() {
            return Err(PyValueError::new_err(
                "covariates and target_covariates must have the same number of columns",
            ));
        }
        if covariates.nrows() == 0 || target_covariates.nrows() == 0 {
            return Err(PyValueError::new_err(
                "covariates and target_covariates must both have at least one row",
            ));
        }
        let objective = BalanceObjective::parse(&self.objective).map_err(PyValueError::new_err)?;
        let solver = SolveMode::parse(&self.solver).map_err(PyValueError::new_err)?;

        let uniform_weight = 1.0 / covariates.nrows() as f64;
        if self.min_weight < 0.0 {
            return Err(PyValueError::new_err("min_weight must be nonnegative"));
        }
        if self.max_weight > 1.0 {
            return Err(PyValueError::new_err("max_weight must not exceed 1.0"));
        }
        if self.max_weight < uniform_weight {
            return Err(PyValueError::new_err(format!(
                "max_weight cannot be smaller than the uniform weight {}",
                uniform_weight
            )));
        }
        if self.min_weight > uniform_weight {
            return Err(PyValueError::new_err(format!(
                "min_weight cannot exceed the uniform weight {}",
                uniform_weight
            )));
        }
        if self.min_weight > self.max_weight {
            return Err(PyValueError::new_err(
                "min_weight must not exceed max_weight",
            ));
        }

        let baseline_weights = match baseline_weights {
            Some(values) => {
                let weights = Array1::from_vec(values);
                Some(
                    normalize_weights("baseline_weights", &weights)
                        .map_err(PyValueError::new_err)?,
                )
            }
            None => None,
        };
        if let Some(weights) = baseline_weights.as_ref() {
            validate_weights("baseline_weights", weights, covariates.nrows())
                .map_err(PyValueError::new_err)?;
        }

        let target_weights = match target_weights {
            Some(values) => {
                let weights = Array1::from_vec(values);
                Some(normalize_weights("target_weights", &weights).map_err(PyValueError::new_err)?)
            }
            None => None,
        };
        if let Some(weights) = target_weights.as_ref() {
            validate_weights("target_weights", weights, target_covariates.nrows())
                .map_err(PyValueError::new_err)?;
        }

        let (covariates_fit, target_covariates_fit) = if self.autoscale {
            minmax_scale_fit(&covariates, &target_covariates)
        } else {
            (covariates.clone(), target_covariates.clone())
        };

        let target_mean_fit = weighted_mean(
            &target_covariates_fit,
            target_weights.as_ref(),
            "target_weights",
        )
        .map_err(PyValueError::new_err)?;
        let z = balance_design(&covariates_fit, &target_mean_fit);
        let beta0 = initial_beta(&z, objective, baseline_weights.as_ref())
            .map_err(PyValueError::new_err)?;

        let system_relaxed = CalibrationSystem {
            z: z.clone(),
            objective,
            baseline_weights: baseline_weights.clone(),
            min_weight: self.min_weight,
            max_weight: self.max_weight,
            l2_norm: self.l2_norm,
            divergence_power: self.divergence_power,
            dual_ridge: self.dual_ridge,
            entropy_phase: if objective == BalanceObjective::Entropy {
                EntropyPhase::Relaxed
            } else {
                EntropyPhase::Bounded
            },
        };

        let (mut fit, mut solver_used) = solve_system(
            &system_relaxed,
            solver,
            &beta0,
            self.max_iterations,
            self.tolerance,
        )
        .map_err(PyValueError::new_err)?;

        if objective == BalanceObjective::Entropy
            && (fit
                .weights
                .iter()
                .any(|value| *value < self.min_weight - 1e-10)
                || fit
                    .weights
                    .iter()
                    .any(|value| *value > self.max_weight + 1e-10))
        {
            let system_bounded = CalibrationSystem {
                z: z.clone(),
                objective,
                baseline_weights: baseline_weights.clone(),
                min_weight: self.min_weight,
                max_weight: self.max_weight,
                l2_norm: self.l2_norm,
                divergence_power: self.divergence_power,
                dual_ridge: self.dual_ridge,
                entropy_phase: EntropyPhase::Bounded,
            };
            let bounded = solve_system(
                &system_bounded,
                solver,
                &fit.beta,
                self.max_iterations,
                self.tolerance,
            )
            .map_err(PyValueError::new_err)?;
            fit = bounded.0;
            solver_used = bounded.1;
        }

        let weighted_mean_original = weighted_mean(&covariates, Some(&fit.weights), "weights")
            .map_err(PyValueError::new_err)?;
        let target_mean_original = weighted_mean(
            &target_covariates,
            target_weights.as_ref(),
            "target_weights",
        )
        .map_err(PyValueError::new_err)?;
        let success = fit.diagnostics.converged
            && (fit.weights.sum() - 1.0).abs() <= 1e-6
            && fit
                .weights
                .iter()
                .all(|value| value.is_finite() && *value >= self.min_weight - 1e-8)
            && fit
                .weights
                .iter()
                .all(|value| *value <= self.max_weight + 1e-8);

        self.weights = Some(fit.weights);
        self.beta = Some(fit.beta);
        self.covariates = Some(covariates);
        self.target_covariates = Some(target_covariates);
        self.weighted_mean = Some(weighted_mean_original);
        self.target_mean = Some(target_mean_original);
        self.success = success;
        self.nit = fit.diagnostics.iterations as usize;
        self.solver_used = Some(solver_used);
        self.criterion = Some(fit.criterion);
        self.residual_norm = Some(fit.residual_norm);
        self.diagnostics = Some(fit.diagnostics);

        Ok(())
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BalancingWeights model is not fitted"))?;
        let beta = self
            .beta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BalancingWeights model is not fitted"))?;
        let weighted_mean = self
            .weighted_mean
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BalancingWeights model is not fitted"))?;
        let target_mean = self
            .target_mean
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BalancingWeights model is not fitted"))?;
        let solver_used = self
            .solver_used
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BalancingWeights model is not fitted"))?;
        let diagnostics = self
            .diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?;

        let mean_diff = weighted_mean - target_mean;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("weights", pyarray1_from_f64(py, weights))?;
        dict.set_item("beta", pyarray1_from_f64(py, beta))?;
        dict.set_item("objective", self.objective.clone())?;
        dict.set_item("solver", solver_used.clone())?;
        dict.set_item("success", self.success)?;
        dict.set_item("nit", self.nit)?;
        dict.set_item("criterion", self.criterion)?;
        dict.set_item("residual_norm", self.residual_norm)?;
        dict.set_item("solver_converged", diagnostics.converged)?;
        dict.set_item("scaled_residual_norm", self.residual_norm)?;
        dict.set_item("solver_objective", diagnostics.objective)?;
        diagnostics.write_status(&dict)?;
        dict.set_item("weight_sum", weights.sum())?;
        dict.set_item("weighted_mean", pyarray1_from_f64(py, weighted_mean))?;
        dict.set_item("target_mean", pyarray1_from_f64(py, target_mean))?;
        dict.set_item("mean_diff", pyarray1_from_f64(py, &mean_diff))?;
        dict.set_item("l2_diff", mean_diff.dot(&mean_diff).sqrt())?;
        dict.set_item("original_balance_l2", mean_diff.dot(&mean_diff).sqrt())?;
        dict.set_item(
            "max_abs_diff",
            mean_diff
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        )?;
        dict.set_item(
            "original_balance_max_abs",
            mean_diff
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs())),
        )?;
        dict.set_item("effective_sample_size", effective_sample_size(weights))?;
        dict.set_item("min_weight", self.min_weight)?;
        dict.set_item("max_weight", self.max_weight)?;
        dict.set_item("l2_norm", self.l2_norm)?;
        dict.set_item("divergence_power", self.divergence_power)?;
        dict.set_item("dual_ridge", self.dual_ridge)?;
        Ok(dict.into())
    }

    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("BalancingWeights model is not fitted"))?;
        Ok(pyarray1_from_f64(py, weights))
    }

    #[getter]
    fn success(&self) -> bool {
        self.success
    }

    #[getter]
    fn solver_converged(&self) -> bool {
        self.diagnostics
            .as_ref()
            .map(|diagnostics| diagnostics.converged)
            .unwrap_or(false)
    }

    #[getter]
    fn nit(&self) -> usize {
        self.nit
    }

    #[getter]
    fn solver_used(&self) -> Option<String> {
        self.solver_used.clone()
    }
}
