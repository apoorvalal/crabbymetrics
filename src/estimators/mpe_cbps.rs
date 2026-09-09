use crate::utils::{
    pyarray1_from_f64, solve_least_squares_vec, to_array1, to_array1_i32, to_array2,
};
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const ARMIJO: f64 = 1e-4;
const MIN_STEP: f64 = 5.960_464_477_539_063e-8;

#[derive(Clone)]
struct ArmFit {
    beta: Array1<f64>,
    weights: Array1<f64>,
    converged: bool,
    iterations: usize,
    objective: f64,
    gradient_norm: f64,
}

fn objective_gradient_hessian(
    design: &Array2<f64>,
    arm: &Array1<f64>,
    policy_derivative: &Array1<f64>,
    beta: &Array1<f64>,
    max_log_weight: f64,
) -> (f64, Array1<f64>, Array2<f64>) {
    let n = design.nrows() as f64;
    let p = design.ncols();
    let linear = design.dot(beta);
    let mut objective = 0.0;
    let mut gradient = Array1::<f64>::zeros(p);
    let mut hessian = Array2::<f64>::zeros((p, p));

    for i in 0..design.nrows() {
        let exp_neg = (-linear[i]).clamp(-max_log_weight, max_log_weight).exp();
        let dpi = policy_derivative[i];
        objective += dpi * (arm[i] * exp_neg + (1.0 - arm[i]) * linear[i]);
        let score_scale = dpi * ((1.0 - arm[i]) - arm[i] * exp_neg);
        let curvature = dpi * arm[i] * exp_neg;
        for j in 0..p {
            let zij = design[[i, j]];
            gradient[j] += zij * score_scale;
            for k in 0..p {
                hessian[[j, k]] += curvature * zij * design[[i, k]];
            }
        }
    }

    (objective / n, gradient / n, hessian / n)
}

fn objective_only(
    design: &Array2<f64>,
    arm: &Array1<f64>,
    policy_derivative: &Array1<f64>,
    beta: &Array1<f64>,
    max_log_weight: f64,
) -> f64 {
    objective_gradient_hessian(design, arm, policy_derivative, beta, max_log_weight).0
}

fn fit_arm(
    design: &Array2<f64>,
    arm: &Array1<f64>,
    policy_derivative: &Array1<f64>,
    max_iterations: usize,
    tolerance: f64,
    max_log_weight: f64,
) -> Result<ArmFit, String> {
    let mut beta = Array1::<f64>::zeros(design.ncols());
    let mut converged = false;
    let mut iterations = 0;
    let objective;
    let mut gradient_norm;

    for iteration in 0..max_iterations {
        let (current_objective, gradient, mut hessian) =
            objective_gradient_hessian(design, arm, policy_derivative, &beta, max_log_weight);
        gradient_norm = gradient
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        iterations = iteration;
        if gradient_norm <= tolerance {
            converged = true;
            break;
        }

        // A vanishing ridge only stabilizes rank-deficient trial systems. It is far below
        // the public convergence tolerance and does not alter the CBPS target moments.
        let ridge = 1e-12
            * hessian
                .diag()
                .iter()
                .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        for j in 0..hessian.nrows() {
            hessian[[j, j]] += ridge;
        }
        let step = solve_least_squares_vec(&hessian, &gradient)?;
        let directional_derivative = gradient.dot(&step);
        if !directional_derivative.is_finite() || directional_derivative <= 0.0 {
            return Err("MPE_CBPS Newton system did not produce a descent direction".to_string());
        }

        let mut step_size = 1.0;
        let mut accepted = false;
        while step_size >= MIN_STEP {
            let candidate = &beta - &(step_size * &step);
            let candidate_objective =
                objective_only(design, arm, policy_derivative, &candidate, max_log_weight);
            if candidate_objective
                <= current_objective - ARMIJO * step_size * directional_derivative
            {
                beta = candidate;
                accepted = true;
                break;
            }
            step_size *= 0.5;
        }
        if !accepted {
            return Err("MPE_CBPS line search failed to find a finite descent step".to_string());
        }
        iterations = iteration + 1;
    }

    let (final_objective, final_gradient, _) =
        objective_gradient_hessian(design, arm, policy_derivative, &beta, max_log_weight);
    objective = final_objective;
    gradient_norm = final_gradient
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    converged = converged || gradient_norm <= tolerance;

    let linear = design.dot(&beta);
    let weights = linear.mapv(|value| 1.0 + (-value).clamp(-max_log_weight, max_log_weight).exp());
    Ok(ArmFit {
        beta,
        weights,
        converged,
        iterations,
        objective,
        gradient_norm,
    })
}

fn effective_sample_size(
    weights: &Array1<f64>,
    arm: &Array1<f64>,
    policy_derivative: &Array1<f64>,
) -> f64 {
    let active = weights * arm * policy_derivative;
    let total = active.sum();
    let squared = active.dot(&active);
    if squared > 0.0 {
        total * total / squared
    } else {
        0.0
    }
}

fn arm_weighted_mean(
    covariates: &Array2<f64>,
    weights: &Array1<f64>,
    arm: &Array1<f64>,
    policy_derivative: &Array1<f64>,
) -> Array1<f64> {
    let active = weights * arm * policy_derivative;
    covariates.t().dot(&active) / active.sum()
}

/// Covariate-balancing weights for the marginal policy effect in the Chronos
/// long-term-value estimator of Qiu, Kuang, Liskovich, Rauh, and Wager (2026).
///
/// The two arm-specific convex programs use the released implementation's
/// inverse-logit weights, `1 + exp(-z @ beta)`. Optimization, standardization,
/// diagnostics, and policy-gradient aggregation are implemented in Rust.
#[pyclass(name = "MPE_CBPS")]
pub struct MpeCbps {
    standardize: bool,
    max_iterations: usize,
    tolerance: f64,
    max_log_weight: f64,
    covariates: Option<Array2<f64>>,
    treatment: Option<Array1<f64>>,
    policy_derivative: Option<Array1<f64>>,
    means: Option<Array1<f64>>,
    scales: Option<Array1<f64>>,
    target_mean: Option<Array1<f64>>,
    fit_zero: Option<ArmFit>,
    fit_one: Option<ArmFit>,
}

#[pymethods]
impl MpeCbps {
    #[new]
    #[pyo3(signature = (standardize=true, max_iterations=500, tolerance=1e-8, max_log_weight=50.0))]
    fn new(
        standardize: bool,
        max_iterations: usize,
        tolerance: f64,
        max_log_weight: f64,
    ) -> PyResult<Self> {
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be positive and finite",
            ));
        }
        if !max_log_weight.is_finite() || max_log_weight <= 0.0 || max_log_weight > 700.0 {
            return Err(PyValueError::new_err(
                "max_log_weight must be finite and in (0, 700]",
            ));
        }
        Ok(Self {
            standardize,
            max_iterations,
            tolerance,
            max_log_weight,
            covariates: None,
            treatment: None,
            policy_derivative: None,
            means: None,
            scales: None,
            target_mean: None,
            fit_zero: None,
            fit_one: None,
        })
    }

    /// Fit the two arm-specific CBPS systems.
    ///
    /// `policy_derivative` is the derivative of the treatment probability with
    /// respect to the policy perturbation. A constant vector only rescales each
    /// convex objective, but is retained for the final policy-gradient estimate.
    #[pyo3(signature = (covariates, treatment, policy_derivative=None))]
    fn fit(
        &mut self,
        covariates: PyReadonlyArray2<f64>,
        treatment: PyReadonlyArray1<i32>,
        policy_derivative: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.covariates = None;
        self.treatment = None;
        self.policy_derivative = None;
        self.means = None;
        self.scales = None;
        self.target_mean = None;
        self.fit_zero = None;
        self.fit_one = None;

        let covariates = to_array2(&covariates);
        let treatment_i32 = to_array1_i32(&treatment);
        let n = covariates.nrows();
        let p = covariates.ncols();
        if n == 0 || p == 0 {
            return Err(PyValueError::new_err(
                "covariates must have at least one row and one column",
            ));
        }
        if treatment_i32.len() != n {
            return Err(PyValueError::new_err(
                "treatment length must match the number of covariate rows",
            ));
        }
        if covariates.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "covariates must contain only finite values",
            ));
        }
        if treatment_i32.iter().any(|value| *value != 0 && *value != 1) {
            return Err(PyValueError::new_err("treatment must contain only 0 and 1"));
        }
        let treatment = treatment_i32.mapv(|value| value as f64);
        let n_one = treatment.sum() as usize;
        if n_one == 0 || n_one == n {
            return Err(PyValueError::new_err(
                "treatment must contain observations from both arms",
            ));
        }

        let policy_derivative = match policy_derivative {
            Some(values) => Array1::from_vec(values),
            None => Array1::ones(n),
        };
        if policy_derivative.len() != n {
            return Err(PyValueError::new_err(
                "policy_derivative length must match the number of observations",
            ));
        }
        if policy_derivative
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(
                "policy_derivative must contain only positive finite values",
            ));
        }

        let means = covariates
            .mean_axis(Axis(0))
            .ok_or_else(|| PyValueError::new_err("failed to compute covariate means"))?;
        let mut scales = Array1::<f64>::ones(p);
        if self.standardize {
            for j in 0..p {
                let variance = covariates
                    .column(j)
                    .iter()
                    .map(|value| (value - means[j]).powi(2))
                    .sum::<f64>()
                    / n as f64;
                let scale = variance.sqrt();
                if scale.is_finite() && scale > 0.0 {
                    scales[j] = scale;
                }
            }
        }

        let mut design = Array2::<f64>::ones((n, p + 1));
        for i in 0..n {
            for j in 0..p {
                design[[i, j + 1]] = if self.standardize {
                    (covariates[[i, j]] - means[j]) / scales[j]
                } else {
                    covariates[[i, j]]
                };
            }
        }
        let arm_one = treatment.clone();
        let arm_zero = treatment.mapv(|value| 1.0 - value);
        let fit_zero = fit_arm(
            &design,
            &arm_zero,
            &policy_derivative,
            self.max_iterations,
            self.tolerance,
            self.max_log_weight,
        )
        .map_err(PyValueError::new_err)?;
        let fit_one = fit_arm(
            &design,
            &arm_one,
            &policy_derivative,
            self.max_iterations,
            self.tolerance,
            self.max_log_weight,
        )
        .map_err(PyValueError::new_err)?;

        self.target_mean = Some(covariates.t().dot(&policy_derivative) / policy_derivative.sum());
        self.covariates = Some(covariates);
        self.treatment = Some(treatment);
        self.policy_derivative = Some(policy_derivative);
        self.means = Some(means);
        self.scales = Some(scales);
        self.fit_zero = Some(fit_zero);
        self.fit_one = Some(fit_one);
        Ok(())
    }

    /// Estimate the marginal policy effect from a cumulative future outcome.
    ///
    /// The default denominator is the number of observations. Set `denominator`
    /// to baseline value (for example total spend) to reproduce a normalized
    /// value elasticity such as the paper's reported estimand.
    #[pyo3(signature = (outcome, denominator=None))]
    fn estimate(&self, outcome: PyReadonlyArray1<f64>, denominator: Option<f64>) -> PyResult<f64> {
        let outcome = to_array1(&outcome);
        let treatment = self
            .treatment
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let policy_derivative = self
            .policy_derivative
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let fit_zero = self
            .fit_zero
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let fit_one = self
            .fit_one
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        if outcome.len() != treatment.len() {
            return Err(PyValueError::new_err(
                "outcome length must match the fitted sample",
            ));
        }
        if outcome.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "outcome must contain only finite values",
            ));
        }
        let denominator = denominator.unwrap_or(treatment.len() as f64);
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(PyValueError::new_err(
                "denominator must be positive and finite",
            ));
        }

        let mut numerator = 0.0;
        for i in 0..treatment.len() {
            numerator += policy_derivative[i]
                * (treatment[i] * fit_one.weights[i] - (1.0 - treatment[i]) * fit_zero.weights[i])
                * outcome[i];
        }
        Ok(numerator / denominator)
    }

    fn get_weights<'py>(&self, py: Python<'py>, arm: i32) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let fit = match arm {
            0 => self.fit_zero.as_ref(),
            1 => self.fit_one.as_ref(),
            _ => return Err(PyValueError::new_err("arm must be 0 or 1")),
        }
        .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        Ok(pyarray1_from_f64(py, &fit.weights))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let covariates = self
            .covariates
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let treatment = self
            .treatment
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let policy_derivative = self
            .policy_derivative
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let means = self
            .means
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let scales = self
            .scales
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let target_mean = self
            .target_mean
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let fit_zero = self
            .fit_zero
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let fit_one = self
            .fit_one
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MPE_CBPS model is not fitted"))?;
        let arm_one = treatment.clone();
        let arm_zero = treatment.mapv(|value| 1.0 - value);
        let weighted_mean_zero =
            arm_weighted_mean(covariates, &fit_zero.weights, &arm_zero, policy_derivative);
        let weighted_mean_one =
            arm_weighted_mean(covariates, &fit_one.weights, &arm_one, policy_derivative);
        let max_abs_balance_zero = (&weighted_mean_zero - target_mean)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));
        let max_abs_balance_one = (&weighted_mean_one - target_mean)
            .iter()
            .fold(0.0_f64, |acc, value| acc.max(value.abs()));

        let dict = PyDict::new(py);
        dict.set_item("success", fit_zero.converged && fit_one.converged)?;
        dict.set_item("converged_zero", fit_zero.converged)?;
        dict.set_item("converged_one", fit_one.converged)?;
        dict.set_item("iterations_zero", fit_zero.iterations)?;
        dict.set_item("iterations_one", fit_one.iterations)?;
        dict.set_item("objective_zero", fit_zero.objective)?;
        dict.set_item("objective_one", fit_one.objective)?;
        dict.set_item("gradient_norm_zero", fit_zero.gradient_norm)?;
        dict.set_item("gradient_norm_one", fit_one.gradient_norm)?;
        dict.set_item("beta_zero", pyarray1_from_f64(py, &fit_zero.beta))?;
        dict.set_item("beta_one", pyarray1_from_f64(py, &fit_one.beta))?;
        dict.set_item("weights_zero", pyarray1_from_f64(py, &fit_zero.weights))?;
        dict.set_item("weights_one", pyarray1_from_f64(py, &fit_one.weights))?;
        dict.set_item(
            "policy_derivative",
            pyarray1_from_f64(py, policy_derivative),
        )?;
        dict.set_item("covariate_mean", pyarray1_from_f64(py, means))?;
        dict.set_item("covariate_scale", pyarray1_from_f64(py, scales))?;
        dict.set_item("target_mean", pyarray1_from_f64(py, target_mean))?;
        dict.set_item(
            "weighted_mean_zero",
            pyarray1_from_f64(py, &weighted_mean_zero),
        )?;
        dict.set_item(
            "weighted_mean_one",
            pyarray1_from_f64(py, &weighted_mean_one),
        )?;
        dict.set_item("max_abs_balance_zero", max_abs_balance_zero)?;
        dict.set_item("max_abs_balance_one", max_abs_balance_one)?;
        dict.set_item("weight_sum_zero", (&fit_zero.weights * &arm_zero).sum())?;
        dict.set_item("weight_sum_one", (&fit_one.weights * &arm_one).sum())?;
        dict.set_item("target_policy_mass", policy_derivative.sum())?;
        dict.set_item(
            "policy_weighted_mass_zero",
            (&fit_zero.weights * &arm_zero * policy_derivative).sum(),
        )?;
        dict.set_item(
            "policy_weighted_mass_one",
            (&fit_one.weights * &arm_one * policy_derivative).sum(),
        )?;
        dict.set_item(
            "effective_sample_size_zero",
            effective_sample_size(&fit_zero.weights, &arm_zero, policy_derivative),
        )?;
        dict.set_item(
            "effective_sample_size_one",
            effective_sample_size(&fit_one.weights, &arm_one, policy_derivative),
        )?;
        dict.set_item("standardize", self.standardize)?;
        dict.set_item("max_log_weight", self.max_log_weight)?;
        Ok(dict.into())
    }

    #[getter]
    fn success(&self) -> bool {
        self.fit_zero
            .as_ref()
            .zip(self.fit_one.as_ref())
            .map(|(zero, one)| zero.converged && one.converged)
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn arm_solver_satisfies_the_released_balance_moment() {
        let design = array![
            [1.0, -1.2],
            [1.0, -0.5],
            [1.0, 0.1],
            [1.0, 0.4],
            [1.0, 0.8],
            [1.0, 1.3]
        ];
        let arm = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let dpi = Array1::ones(6);
        let fit = fit_arm(&design, &arm, &dpi, 100, 1e-10, 50.0).unwrap();
        let active = &fit.weights * &arm;
        let balanced = design.t().dot(&active);
        let target = design.sum_axis(Axis(0));
        assert!(fit.converged);
        assert!(
            (&balanced - &target)
                .iter()
                .fold(0.0_f64, |acc, value| acc.max(value.abs()))
                < 1e-8
        );
    }
}
