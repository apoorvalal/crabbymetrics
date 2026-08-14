use super::balancing::fit_quadratic_calibration;
use crate::utils::{
    add_intercept, diag_sqrt, invert_matrix, pyarray1_from_f64, pyarray2_from_f64,
    sandwich_cov_from_parameter_scores, solve_least_squares_mat, solve_least_squares_vec,
};
use ndarray::{Array1, Array2, Array3};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_array3(x: &PyReadonlyArray3<f64>) -> Array3<f64> {
    let shape = x.shape();
    let values = x.as_array().iter().copied().collect::<Vec<_>>();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
        .expect("invalid three-dimensional array shape")
}

fn validate_finite_matrix(name: &str, values: &Array2<f64>) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn validate_finite_cube(name: &str, values: &Array3<f64>) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn is_binary(values: &Array2<f64>) -> bool {
    values.iter().all(|value| *value == 0.0 || *value == 1.0)
}

fn ridge_augmented_design(x: &Array2<f64>, penalty: f64) -> Array2<f64> {
    let design = add_intercept(x);
    if penalty == 0.0 || x.ncols() == 0 {
        return design;
    }

    let n = design.nrows();
    let p = design.ncols();
    let mut augmented = Array2::<f64>::zeros((n + p - 1, p));
    augmented.slice_mut(ndarray::s![..n, ..]).assign(&design);
    for column in 1..p {
        augmented[[n + column - 1, column]] = penalty.sqrt();
    }
    augmented
}

fn ridge_fit_vec(x: &Array2<f64>, y: &Array1<f64>, penalty: f64) -> Result<Array1<f64>, String> {
    if x.nrows() != y.len() {
        return Err("ridge row count mismatch".to_string());
    }
    let augmented_x = ridge_augmented_design(x, penalty);
    let mut augmented_y = Array1::<f64>::zeros(augmented_x.nrows());
    augmented_y.slice_mut(ndarray::s![..y.len()]).assign(y);
    solve_least_squares_vec(&augmented_x, &augmented_y)
}

fn ridge_fit_mat(x: &Array2<f64>, y: &Array2<f64>, penalty: f64) -> Result<Array2<f64>, String> {
    if x.nrows() != y.nrows() {
        return Err("ridge row count mismatch".to_string());
    }
    let augmented_x = ridge_augmented_design(x, penalty);
    let mut augmented_y = Array2::<f64>::zeros((augmented_x.nrows(), y.ncols()));
    augmented_y
        .slice_mut(ndarray::s![..y.nrows(), ..])
        .assign(y);
    solve_least_squares_mat(&augmented_x, &augmented_y)
}

fn ridge_predict_vec(x: &Array2<f64>, coef: &Array1<f64>) -> Result<Array1<f64>, String> {
    let design = add_intercept(x);
    if design.ncols() != coef.len() {
        return Err("ridge coefficient dimension mismatch".to_string());
    }
    Ok(design.dot(coef))
}

fn ridge_predict_mat(x: &Array2<f64>, coef: &Array2<f64>) -> Result<Array2<f64>, String> {
    let design = add_intercept(x);
    if design.ncols() != coef.nrows() {
        return Err("ridge coefficient dimension mismatch".to_string());
    }
    Ok(design.dot(coef))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn unit_folds(n_units: usize, n_folds: usize, seed: u64) -> Result<Vec<usize>, String> {
    if n_units < 2 {
        return Err("at least two units are required".to_string());
    }
    if n_folds < 2 || n_folds > n_units {
        return Err("n_folds must lie between 2 and the number of units".to_string());
    }
    let mut units = (0..n_units).collect::<Vec<_>>();
    units.sort_by_key(|unit| splitmix64(seed ^ (*unit as u64)));
    let mut fold = vec![0usize; n_units];
    for (position, unit) in units.iter().enumerate() {
        fold[*unit] = position % n_folds;
    }
    Ok(fold)
}

fn history_matrix(history: Option<&Array3<f64>>, units: &[usize], time: usize) -> Array2<f64> {
    let p = history.map_or(0, |values| values.dim().2);
    let mut out = Array2::<f64>::zeros((units.len(), p));
    if let Some(values) = history {
        for (row, unit) in units.iter().enumerate() {
            for column in 0..p {
                out[[row, column]] = values[[*unit, time, column]];
            }
        }
    }
    out
}

fn treatment_coding(
    treatment: &Array2<f64>,
    mode: &str,
) -> Result<(Array2<f64>, Array2<bool>), String> {
    let (n, t) = treatment.dim();
    match mode {
        "blip" => Ok((treatment.clone(), Array2::from_elem((n, t), true))),
        "initiation" => {
            if !is_binary(treatment) {
                return Err("treatment must be binary when treatment_mode='initiation'".to_string());
            }
            let mut pulse = Array2::<f64>::zeros((n, t));
            let mut at_risk = Array2::from_elem((n, t), false);
            for unit in 0..n {
                let mut previously_treated = false;
                for time in 0..t {
                    let current = treatment[[unit, time]] == 1.0;
                    if previously_treated && !current {
                        return Err(
                            "treatment must be absorbing when treatment_mode='initiation'"
                                .to_string(),
                        );
                    }
                    at_risk[[unit, time]] = !previously_treated;
                    if current && !previously_treated {
                        pulse[[unit, time]] = 1.0;
                    }
                    previously_treated |= current;
                }
            }
            Ok((pulse, at_risk))
        }
        _ => Err("treatment_mode must be one of {'blip', 'initiation'}".to_string()),
    }
}

fn blip_design_difference(
    treatment: &Array2<f64>,
    unit: usize,
    m: usize,
    k: usize,
    max_horizon: usize,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(max_horizon);
    for horizon in 1..=max_horizon {
        if k >= horizon {
            let treatment_time = k - horizon;
            if treatment_time >= m && treatment_time < k {
                out[horizon - 1] += treatment[[unit, treatment_time]];
            }
        }
        if k > horizon {
            let treatment_time = k - 1 - horizon;
            if treatment_time >= m && treatment_time < k - 1 {
                out[horizon - 1] -= treatment[[unit, treatment_time]];
            }
        }
    }
    out
}

struct OrthogonalRow {
    unit: usize,
    instrument: usize,
    treatment_residual: f64,
    outcome_residual: f64,
    design_residual: Array1<f64>,
}

/// Dynamic covariate-balancing estimator for a target treatment path.
///
/// The estimator implements the full-interaction recursive potential-projection
/// construction of Viviano and Bradic. At each treatment time it calls the same
/// exact quadratic-calibration engine as `BalancingWeights`, restricting source
/// weights to units whose realized treatment prefix matches `target_path`.
#[pyclass]
pub struct DynamicCovariateBalance {
    nuisance_penalty: f64,
    autoscale: bool,
    max_weight: f64,
    max_iterations: usize,
    tolerance: f64,
    estimate: Option<f64>,
    target_path: Option<Array1<f64>>,
    weights: Option<Array2<f64>>,
    predictions: Option<Array2<f64>>,
    path_support: Option<Array1<f64>>,
    effective_sample_size: Option<Array1<f64>>,
    max_abs_balance: Option<Array1<f64>>,
    iterations: Option<Array1<f64>>,
    solver_converged: Option<Vec<bool>>,
    n_units: Option<usize>,
    n_periods: Option<usize>,
}

#[pymethods]
impl DynamicCovariateBalance {
    #[new]
    #[pyo3(signature = (nuisance_penalty=1e-6, autoscale=true, max_weight=1.0, max_iterations=300, tolerance=1e-6))]
    fn new(
        nuisance_penalty: f64,
        autoscale: bool,
        max_weight: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<Self> {
        if !nuisance_penalty.is_finite() || nuisance_penalty < 0.0 {
            return Err(PyValueError::new_err(
                "nuisance_penalty must be finite and nonnegative",
            ));
        }
        if !max_weight.is_finite() || max_weight <= 0.0 || max_weight > 1.0 {
            return Err(PyValueError::new_err("max_weight must lie in (0, 1]"));
        }
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be positive and finite",
            ));
        }
        Ok(Self {
            nuisance_penalty,
            autoscale,
            max_weight,
            max_iterations,
            tolerance,
            estimate: None,
            target_path: None,
            weights: None,
            predictions: None,
            path_support: None,
            effective_sample_size: None,
            max_abs_balance: None,
            iterations: None,
            solver_converged: None,
            n_units: None,
            n_periods: None,
        })
    }

    /// Estimate the mean final potential outcome under `target_path`.
    ///
    /// `outcome` contains the final-period outcome and has shape `(n_units,)`.
    /// `treatment` is a binary `(n_units, n_periods)` panel. `history[i, t, :]`
    /// contains variables observed before treatment at time `t`; it may include
    /// baseline covariates, prior treatment, time-varying covariates, and prior
    /// outcomes. `target_path` is a binary vector of length `n_periods`.
    fn fit(
        &mut self,
        outcome: PyReadonlyArray1<f64>,
        treatment: PyReadonlyArray2<f64>,
        history: PyReadonlyArray3<f64>,
        target_path: Vec<f64>,
    ) -> PyResult<()> {
        self.estimate = None;
        self.target_path = None;
        self.weights = None;
        self.predictions = None;
        self.path_support = None;
        self.effective_sample_size = None;
        self.max_abs_balance = None;
        self.iterations = None;
        self.solver_converged = None;
        self.n_units = None;
        self.n_periods = None;

        let outcome = Array1::from_iter(outcome.as_array().iter().copied());
        let treatment = Array2::from_shape_vec(
            (treatment.shape()[0], treatment.shape()[1]),
            treatment.as_array().iter().copied().collect(),
        )
        .expect("invalid treatment shape");
        let history = to_array3(&history);
        let target_path = Array1::from_vec(target_path);

        if outcome.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err(
                "outcome must contain only finite values",
            ));
        }
        validate_finite_matrix("treatment", &treatment).map_err(PyValueError::new_err)?;
        validate_finite_cube("history", &history).map_err(PyValueError::new_err)?;
        if !is_binary(&treatment) {
            return Err(PyValueError::new_err("treatment must be binary"));
        }
        if target_path
            .iter()
            .any(|value| *value != 0.0 && *value != 1.0)
        {
            return Err(PyValueError::new_err("target_path must be binary"));
        }

        let (n_units, n_periods) = treatment.dim();
        if n_units < 2 || n_periods == 0 {
            return Err(PyValueError::new_err(
                "need at least two units and one treatment period",
            ));
        }
        if outcome.len() != n_units {
            return Err(PyValueError::new_err(
                "outcome length must equal treatment.shape[0]",
            ));
        }
        if history.dim().0 != n_units || history.dim().1 != n_periods {
            return Err(PyValueError::new_err(
                "history must have shape (n_units, n_periods, n_features)",
            ));
        }
        if target_path.len() != n_periods {
            return Err(PyValueError::new_err(
                "target_path length must equal treatment.shape[1]",
            ));
        }

        let mut prefix_match = Array2::from_elem((n_units, n_periods), false);
        let mut path_support = Array1::<f64>::zeros(n_periods);
        for unit in 0..n_units {
            let mut matches = true;
            for time in 0..n_periods {
                matches &= treatment[[unit, time]] == target_path[time];
                prefix_match[[unit, time]] = matches;
                if matches {
                    path_support[time] += 1.0;
                }
            }
        }
        for time in 0..n_periods {
            if path_support[time] < 2.0 {
                return Err(PyValueError::new_err(format!(
                    "target_path has fewer than two matching units through period {time}"
                )));
            }
        }

        let all_units = (0..n_units).collect::<Vec<_>>();
        let mut predictions = Array2::<f64>::zeros((n_units, n_periods));
        let mut pseudo_outcome = outcome.clone();
        for time in (0..n_periods).rev() {
            let eligible = (0..n_units)
                .filter(|unit| prefix_match[[*unit, time]])
                .collect::<Vec<_>>();
            let x_train = history_matrix(Some(&history), &eligible, time);
            let y_train = Array1::from_iter(eligible.iter().map(|unit| pseudo_outcome[*unit]));
            let coefficient =
                ridge_fit_vec(&x_train, &y_train, self.nuisance_penalty).map_err(|error| {
                    PyValueError::new_err(format!(
                        "period {time} potential projection failed: {error}"
                    ))
                })?;
            let x_all = history_matrix(Some(&history), &all_units, time);
            let fitted = ridge_predict_vec(&x_all, &coefficient).map_err(PyValueError::new_err)?;
            predictions.column_mut(time).assign(&fitted);
            pseudo_outcome = fitted;
        }

        let mut weights = Array2::<f64>::zeros((n_units, n_periods));
        let mut previous_weights = Array1::from_elem(n_units, 1.0 / n_units as f64);
        let mut effective_sample_size = Array1::<f64>::zeros(n_periods);
        let mut max_abs_balance = Array1::<f64>::zeros(n_periods);
        let mut iterations = Array1::<f64>::zeros(n_periods);
        let mut solver_converged = Vec::<bool>::with_capacity(n_periods);

        for time in 0..n_periods {
            let source_units = (0..n_units)
                .filter(|unit| prefix_match[[*unit, time]])
                .collect::<Vec<_>>();
            let target_units = if time == 0 {
                all_units.clone()
            } else {
                (0..n_units)
                    .filter(|unit| prefix_match[[*unit, time - 1]])
                    .collect::<Vec<_>>()
            };
            let source_history = history_matrix(Some(&history), &source_units, time);
            let target_history = history_matrix(Some(&history), &target_units, time);
            let target_weights =
                Array1::from_iter(target_units.iter().map(|unit| previous_weights[*unit]));
            let fit = fit_quadratic_calibration(
                &source_history,
                &target_history,
                Some(&target_weights),
                self.autoscale,
                self.max_weight,
                self.max_iterations,
                self.tolerance,
            )
            .map_err(|error| {
                PyValueError::new_err(format!("period {time} balancing failed: {error}"))
            })?;
            let mut current_weights = Array1::<f64>::zeros(n_units);
            for (row, unit) in source_units.iter().enumerate() {
                current_weights[*unit] = fit.weights[row];
            }
            weights.column_mut(time).assign(&current_weights);
            effective_sample_size[time] = fit.effective_sample_size;
            max_abs_balance[time] = fit.max_abs_balance;
            iterations[time] = fit.iterations as f64;
            solver_converged.push(fit.converged);
            previous_weights = current_weights;
        }

        let uniform = Array1::from_elem(n_units, 1.0 / n_units as f64);
        let mut contribution = weights.column(n_periods - 1).to_owned() * &outcome;
        for time in 1..n_periods {
            let weight_difference =
                weights.column(time).to_owned() - weights.column(time - 1).to_owned();
            contribution -= &(weight_difference * predictions.column(time));
        }
        contribution -= &((weights.column(0).to_owned() - uniform) * predictions.column(0));
        let estimate = contribution.sum();
        if !estimate.is_finite() {
            return Err(PyValueError::new_err(
                "dynamic covariate-balance estimate is not finite",
            ));
        }

        self.estimate = Some(estimate);
        self.target_path = Some(target_path);
        self.weights = Some(weights);
        self.predictions = Some(predictions);
        self.path_support = Some(path_support);
        self.effective_sample_size = Some(effective_sample_size);
        self.max_abs_balance = Some(max_abs_balance);
        self.iterations = Some(iterations);
        self.solver_converged = Some(solver_converged);
        self.n_units = Some(n_units);
        self.n_periods = Some(n_periods);
        Ok(())
    }

    #[getter]
    fn potential_outcome(&self) -> PyResult<f64> {
        self.estimate
            .ok_or_else(|| PyValueError::new_err("DynamicCovariateBalance is not fitted"))
    }

    fn get_weights<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("DynamicCovariateBalance is not fitted"))?;
        Ok(pyarray2_from_f64(py, weights))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let estimate = self
            .estimate
            .ok_or_else(|| PyValueError::new_err("DynamicCovariateBalance is not fitted"))?;
        let target_path = self
            .target_path
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("DynamicCovariateBalance is not fitted"))?;
        let path_support = self
            .path_support
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("DynamicCovariateBalance is not fitted"))?;
        let effective_sample_size = self.effective_sample_size.as_ref().ok_or_else(|| {
            PyValueError::new_err("DynamicCovariateBalance diagnostics are unavailable")
        })?;
        let max_abs_balance = self.max_abs_balance.as_ref().ok_or_else(|| {
            PyValueError::new_err("DynamicCovariateBalance diagnostics are unavailable")
        })?;
        let iterations = self.iterations.as_ref().ok_or_else(|| {
            PyValueError::new_err("DynamicCovariateBalance diagnostics are unavailable")
        })?;
        let solver_converged = self.solver_converged.as_ref().ok_or_else(|| {
            PyValueError::new_err("DynamicCovariateBalance diagnostics are unavailable")
        })?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("estimator", "dynamic_covariate_balance")?;
        dict.set_item("potential_outcome", estimate)?;
        dict.set_item("target_path", pyarray1_from_f64(py, target_path))?;
        dict.set_item("path_support", pyarray1_from_f64(py, path_support))?;
        dict.set_item(
            "effective_sample_size",
            pyarray1_from_f64(py, effective_sample_size),
        )?;
        dict.set_item("max_abs_balance", pyarray1_from_f64(py, max_abs_balance))?;
        dict.set_item("iterations", pyarray1_from_f64(py, iterations))?;
        dict.set_item("solver_converged", solver_converged.clone())?;
        dict.set_item("success", solver_converged.iter().all(|value| *value))?;
        dict.set_item("n_units", self.n_units)?;
        dict.set_item("n_periods", self.n_periods)?;
        dict.set_item("nuisance_penalty", self.nuisance_penalty)?;
        dict.set_item("autoscale", self.autoscale)?;
        dict.set_item("max_weight", self.max_weight)?;
        Ok(dict.into())
    }
}

/// Additive structural nested mean model identified by time-varying parallel trends.
///
/// The implementation targets linear blip functions with a common coefficient at each
/// treatment-to-outcome horizon. Nuisance conditional means are fitted by ridge regression
/// and cross-fitted over units. `treatment_mode="initiation"` accepts a binary absorbing
/// treatment-status panel and internally converts it to first-treatment pulses; `"blip"`
/// treats the supplied panel as the period-specific treatment itself.
#[pyclass]
pub struct ParallelTrendsSNMM {
    max_horizon: usize,
    treatment_mode: String,
    n_folds: usize,
    nuisance_penalty: f64,
    propensity_clip: f64,
    seed: u64,
    coef: Option<Array1<f64>>,
    vcov: Option<Array2<f64>>,
    se: Option<Array1<f64>>,
    n_units: Option<usize>,
    n_periods: Option<usize>,
    n_moment_rows: Option<usize>,
    propensity_min: Option<f64>,
    propensity_max: Option<f64>,
    max_abs_moment: Option<f64>,
}

#[pymethods]
impl ParallelTrendsSNMM {
    #[new]
    #[pyo3(signature = (max_horizon=1, treatment_mode="blip", n_folds=2, nuisance_penalty=1e-6, propensity_clip=0.01, seed=42))]
    fn new(
        max_horizon: usize,
        treatment_mode: &str,
        n_folds: usize,
        nuisance_penalty: f64,
        propensity_clip: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if max_horizon == 0 {
            return Err(PyValueError::new_err("max_horizon must be positive"));
        }
        if treatment_mode != "blip" && treatment_mode != "initiation" {
            return Err(PyValueError::new_err(
                "treatment_mode must be one of {'blip', 'initiation'}",
            ));
        }
        if n_folds < 2 {
            return Err(PyValueError::new_err("n_folds must be at least 2"));
        }
        if !nuisance_penalty.is_finite() || nuisance_penalty < 0.0 {
            return Err(PyValueError::new_err(
                "nuisance_penalty must be finite and nonnegative",
            ));
        }
        if !(0.0..0.5).contains(&propensity_clip) {
            return Err(PyValueError::new_err(
                "propensity_clip must lie in [0, 0.5)",
            ));
        }
        Ok(Self {
            max_horizon,
            treatment_mode: treatment_mode.to_string(),
            n_folds,
            nuisance_penalty,
            propensity_clip,
            seed,
            coef: None,
            vcov: None,
            se: None,
            n_units: None,
            n_periods: None,
            n_moment_rows: None,
            propensity_min: None,
            propensity_max: None,
            max_abs_moment: None,
        })
    }

    /// Fit the model to wide panels.
    ///
    /// `y` has shape `(n_units, n_treat_periods + 1)` because treatment at period
    /// `m` affects outcomes from `m + 1` onward. `treatment` has shape
    /// `(n_units, n_treat_periods)`. Optional `history` has shape
    /// `(n_units, n_treat_periods, n_history_features)` and must contain only
    /// variables available when treatment is assigned.
    #[pyo3(signature = (y, treatment, history=None))]
    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        treatment: PyReadonlyArray2<f64>,
        history: Option<PyReadonlyArray3<f64>>,
    ) -> PyResult<()> {
        let y = Array2::from_shape_vec(
            (y.shape()[0], y.shape()[1]),
            y.as_array().iter().copied().collect(),
        )
        .expect("invalid outcome shape");
        let treatment = Array2::from_shape_vec(
            (treatment.shape()[0], treatment.shape()[1]),
            treatment.as_array().iter().copied().collect(),
        )
        .expect("invalid treatment shape");
        let history = history.as_ref().map(to_array3);

        validate_finite_matrix("y", &y).map_err(PyValueError::new_err)?;
        validate_finite_matrix("treatment", &treatment).map_err(PyValueError::new_err)?;
        if let Some(values) = &history {
            validate_finite_cube("history", values).map_err(PyValueError::new_err)?;
        }

        let (n_units, n_treat_periods) = treatment.dim();
        if y.nrows() != n_units || y.ncols() != n_treat_periods + 1 {
            return Err(PyValueError::new_err(
                "y must have shape (n_units, treatment.shape[1] + 1)",
            ));
        }
        if self.max_horizon > n_treat_periods {
            return Err(PyValueError::new_err(
                "max_horizon cannot exceed the number of treatment periods",
            ));
        }
        if let Some(values) = &history {
            if values.dim().0 != n_units || values.dim().1 != n_treat_periods {
                return Err(PyValueError::new_err(
                    "history must have shape (n_units, n_treat_periods, n_features)",
                ));
            }
        }

        let (blips, at_risk) =
            treatment_coding(&treatment, &self.treatment_mode).map_err(PyValueError::new_err)?;
        let binary_treatment = is_binary(&blips);
        let folds = unit_folds(n_units, self.n_folds, self.seed).map_err(PyValueError::new_err)?;
        let mut rows = Vec::<OrthogonalRow>::new();
        let mut propensity_min = f64::INFINITY;
        let mut propensity_max = f64::NEG_INFINITY;

        for fold in 0..self.n_folds {
            for m in 0..n_treat_periods {
                let train_units = (0..n_units)
                    .filter(|unit| folds[*unit] != fold && at_risk[[*unit, m]])
                    .collect::<Vec<_>>();
                let test_units = (0..n_units)
                    .filter(|unit| folds[*unit] == fold && at_risk[[*unit, m]])
                    .collect::<Vec<_>>();
                if train_units.is_empty() || test_units.is_empty() {
                    continue;
                }

                let x_train = history_matrix(history.as_ref(), &train_units, m);
                let x_test = history_matrix(history.as_ref(), &test_units, m);
                let a_train = Array1::from_iter(train_units.iter().map(|unit| blips[[*unit, m]]));
                let pi_coef = ridge_fit_vec(&x_train, &a_train, self.nuisance_penalty)
                    .map_err(PyValueError::new_err)?;
                let mut pi_test =
                    ridge_predict_vec(&x_test, &pi_coef).map_err(PyValueError::new_err)?;
                if binary_treatment {
                    pi_test.mapv_inplace(|value| {
                        value.clamp(self.propensity_clip, 1.0 - self.propensity_clip)
                    });
                }
                for value in &pi_test {
                    propensity_min = propensity_min.min(*value);
                    propensity_max = propensity_max.max(*value);
                }

                let available_horizons = self.max_horizon.min(n_treat_periods - m);
                for horizon in 1..=available_horizons {
                    let k = m + horizon;
                    let mut nuisance_targets =
                        Array2::<f64>::zeros((train_units.len(), 1 + self.max_horizon));
                    for (row, unit) in train_units.iter().enumerate() {
                        nuisance_targets[[row, 0]] = y[[*unit, k]] - y[[*unit, k - 1]];
                        let design_diff =
                            blip_design_difference(&blips, *unit, m, k, self.max_horizon);
                        for column in 0..self.max_horizon {
                            nuisance_targets[[row, 1 + column]] = design_diff[column];
                        }
                    }
                    let nuisance_coef =
                        ridge_fit_mat(&x_train, &nuisance_targets, self.nuisance_penalty)
                            .map_err(PyValueError::new_err)?;
                    let nuisance_prediction = ridge_predict_mat(&x_test, &nuisance_coef)
                        .map_err(PyValueError::new_err)?;

                    for (row, unit) in test_units.iter().enumerate() {
                        let outcome_trend = y[[*unit, k]] - y[[*unit, k - 1]];
                        let design_diff =
                            blip_design_difference(&blips, *unit, m, k, self.max_horizon);
                        let mut design_residual = Array1::<f64>::zeros(self.max_horizon);
                        for column in 0..self.max_horizon {
                            design_residual[column] =
                                design_diff[column] - nuisance_prediction[[row, 1 + column]];
                        }
                        rows.push(OrthogonalRow {
                            unit: *unit,
                            instrument: horizon - 1,
                            treatment_residual: blips[[*unit, m]] - pi_test[row],
                            outcome_residual: outcome_trend - nuisance_prediction[[row, 0]],
                            design_residual,
                        });
                    }
                }
            }
        }

        if rows.is_empty() {
            return Err(PyValueError::new_err(
                "no estimating-equation rows remain after risk-set and fold construction",
            ));
        }

        let p = self.max_horizon;
        let mut lhs = Array2::<f64>::zeros((p, p));
        let mut rhs = Array1::<f64>::zeros(p);
        for row in &rows {
            let z = row.treatment_residual;
            rhs[row.instrument] += z * row.outcome_residual;
            for column in 0..p {
                lhs[[row.instrument, column]] += z * row.design_residual[column];
            }
        }
        let coef = solve_least_squares_vec(&lhs, &rhs).map_err(|error| {
            PyValueError::new_err(format!(
                "SNMM estimating equations are not identified: {error}"
            ))
        })?;

        let mut unit_moments = Array2::<f64>::zeros((n_units, p));
        for row in &rows {
            let residual = row.outcome_residual - row.design_residual.dot(&coef);
            unit_moments[[row.unit, row.instrument]] += row.treatment_residual * residual;
        }
        let jacobian = lhs.mapv(|value| -value / n_units as f64);
        let inverse_jacobian = invert_matrix(&jacobian).map_err(|error| {
            PyValueError::new_err(format!("SNMM moment Jacobian is singular: {error}"))
        })?;
        let parameter_scores = unit_moments.dot(&inverse_jacobian.t()) / n_units as f64;
        let vcov = sandwich_cov_from_parameter_scores(
            &parameter_scores,
            "hc1",
            n_units as f64 - p as f64,
            None,
            None,
        )
        .map_err(PyValueError::new_err)?;
        let se = diag_sqrt(&vcov).map_err(PyValueError::new_err)?;
        let mean_moments = unit_moments
            .mean_axis(ndarray::Axis(0))
            .ok_or_else(|| PyValueError::new_err("empty unit moment matrix"))?;
        let max_abs_moment = mean_moments
            .iter()
            .fold(0.0_f64, |current, value| current.max(value.abs()));

        self.coef = Some(coef);
        self.vcov = Some(vcov);
        self.se = Some(se);
        self.n_units = Some(n_units);
        self.n_periods = Some(n_treat_periods);
        self.n_moment_rows = Some(rows.len());
        self.propensity_min = Some(propensity_min);
        self.propensity_max = Some(propensity_max);
        self.max_abs_moment = Some(max_abs_moment);
        Ok(())
    }

    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ParallelTrendsSNMM is not fitted"))?;
        Ok(pyarray1_from_f64(py, coef))
    }

    #[getter]
    fn standard_errors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let se = self
            .se
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ParallelTrendsSNMM is not fitted"))?;
        Ok(pyarray1_from_f64(py, se))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ParallelTrendsSNMM is not fitted"))?;
        let vcov = self
            .vcov
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ParallelTrendsSNMM is not fitted"))?;
        let se = self
            .se
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ParallelTrendsSNMM is not fitted"))?;
        let horizons = Array1::from_iter((1..=self.max_horizon).map(|value| value as f64));
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("estimator", "parallel_trends_snmm")?;
        dict.set_item("treatment_mode", self.treatment_mode.clone())?;
        dict.set_item("horizons", pyarray1_from_f64(py, &horizons))?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("se", pyarray1_from_f64(py, se))?;
        dict.set_item("vcov", pyarray2_from_f64(py, vcov))?;
        dict.set_item("n_units", self.n_units)?;
        dict.set_item("n_treatment_periods", self.n_periods)?;
        dict.set_item("n_moment_rows", self.n_moment_rows)?;
        dict.set_item("n_folds", self.n_folds)?;
        dict.set_item("nuisance_penalty", self.nuisance_penalty)?;
        dict.set_item("propensity_min", self.propensity_min)?;
        dict.set_item("propensity_max", self.propensity_max)?;
        dict.set_item("max_abs_moment", self.max_abs_moment)?;
        Ok(dict.into())
    }
}

fn regression_design(
    y: &Array2<f64>,
    treatment: &Array2<f64>,
    history: Option<&Array3<f64>>,
    lag: usize,
    time_effects: bool,
    previous_coef: &Array1<f64>,
) -> (Array2<f64>, Array1<f64>, Array1<i64>) {
    let (n_units, n_periods) = y.dim();
    let p_history = history.map_or(0, |values| values.dim().2);
    let n_source_periods = n_periods - lag - 1;
    let n_time_dummies = if time_effects {
        n_source_periods.saturating_sub(1)
    } else {
        0
    };
    let n_columns = 3 + p_history + n_time_dummies;
    let n_rows = n_units * n_source_periods;
    let mut x = Array2::<f64>::zeros((n_rows, n_columns));
    let mut target = Array1::<f64>::zeros(n_rows);
    let mut unit_ids = Array1::<i64>::zeros(n_rows);

    let mut row = 0usize;
    for unit in 0..n_units {
        for source_time in 1..(n_periods - lag) {
            let outcome_time = source_time + lag;
            let mut blipped = y[[unit, outcome_time]];
            for previous_lag in 0..lag {
                blipped -=
                    previous_coef[previous_lag] * treatment[[unit, outcome_time - previous_lag]];
            }
            target[row] = blipped;
            unit_ids[row] = unit as i64;
            x[[row, 0]] = treatment[[unit, source_time]];
            x[[row, 1]] = treatment[[unit, source_time - 1]];
            x[[row, 2]] = y[[unit, source_time - 1]];
            if let Some(values) = history {
                for column in 0..p_history {
                    x[[row, 3 + column]] = values[[unit, source_time, column]];
                }
            }
            if time_effects && source_time > 1 {
                x[[row, 3 + p_history + source_time - 2]] = 1.0;
            }
            row += 1;
        }
    }
    (x, target, unit_ids)
}

/// Sequential regression g-estimator for additive impulse-response blips.
///
/// This is the recursive regression estimator described by Blackwell and Glynn
/// (2018). At lag `j`, it subtracts already estimated shorter-lag blips and
/// regresses the resulting outcome on treatment at `t-j` and variables known at
/// that treatment time. The stagewise standard errors are unit-clustered but do
/// not propagate uncertainty from earlier recursive stages; use a unit block
/// bootstrap when joint inference is required.
#[pyclass]
pub struct RegressionBlip {
    max_lag: usize,
    time_effects: bool,
    coef: Option<Array1<f64>>,
    stage_se: Option<Array1<f64>>,
    n_units: Option<usize>,
    n_periods: Option<usize>,
}

#[pymethods]
impl RegressionBlip {
    #[new]
    #[pyo3(signature = (max_lag=1, time_effects=true))]
    fn new(max_lag: usize, time_effects: bool) -> Self {
        Self {
            max_lag,
            time_effects,
            coef: None,
            stage_se: None,
            n_units: None,
            n_periods: None,
        }
    }

    /// Fit sequential blip regressions to same-shaped outcome and treatment panels.
    ///
    /// Optional `history[i, t, :]` must contain only variables known when
    /// `treatment[i, t]` is assigned. The design automatically includes treatment
    /// and outcome at `t-1`; rows begin at treatment time one.
    #[pyo3(signature = (y, treatment, history=None))]
    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        treatment: PyReadonlyArray2<f64>,
        history: Option<PyReadonlyArray3<f64>>,
    ) -> PyResult<()> {
        let y = Array2::from_shape_vec(
            (y.shape()[0], y.shape()[1]),
            y.as_array().iter().copied().collect(),
        )
        .expect("invalid outcome shape");
        let treatment = Array2::from_shape_vec(
            (treatment.shape()[0], treatment.shape()[1]),
            treatment.as_array().iter().copied().collect(),
        )
        .expect("invalid treatment shape");
        let history = history.as_ref().map(to_array3);

        validate_finite_matrix("y", &y).map_err(PyValueError::new_err)?;
        validate_finite_matrix("treatment", &treatment).map_err(PyValueError::new_err)?;
        if y.dim() != treatment.dim() {
            return Err(PyValueError::new_err(
                "y and treatment must have the same shape",
            ));
        }
        let (n_units, n_periods) = y.dim();
        if n_periods < self.max_lag + 2 {
            return Err(PyValueError::new_err("need at least max_lag + 2 periods"));
        }
        if n_units < 2 {
            return Err(PyValueError::new_err("at least two units are required"));
        }
        if let Some(values) = &history {
            validate_finite_cube("history", values).map_err(PyValueError::new_err)?;
            if values.dim().0 != n_units || values.dim().1 != n_periods {
                return Err(PyValueError::new_err(
                    "history must have shape (n_units, n_periods, n_features)",
                ));
            }
        }

        let mut coef = Array1::<f64>::zeros(self.max_lag + 1);
        let mut stage_se = Array1::<f64>::zeros(self.max_lag + 1);
        for lag in 0..=self.max_lag {
            let (x, target, unit_ids) = regression_design(
                &y,
                &treatment,
                history.as_ref(),
                lag,
                self.time_effects,
                &coef,
            );
            let design = add_intercept(&x);
            if design.nrows() <= design.ncols() {
                return Err(PyValueError::new_err(format!(
                    "lag {lag} has too few rows for its regression design"
                )));
            }
            let params =
                solve_least_squares_vec(&design, &target).map_err(PyValueError::new_err)?;
            coef[lag] = params[1];

            let residual = &target - &design.dot(&params);
            let bread = invert_matrix(&design.t().dot(&design)).map_err(|error| {
                PyValueError::new_err(format!("lag {lag} regression is singular: {error}"))
            })?;
            let mut raw_scores = design.clone();
            for row in 0..design.nrows() {
                raw_scores
                    .row_mut(row)
                    .mapv_inplace(|value| value * residual[row]);
            }
            let parameter_scores = raw_scores.dot(&bread);
            let covariance = sandwich_cov_from_parameter_scores(
                &parameter_scores,
                "cluster",
                design.nrows() as f64 - design.ncols() as f64,
                None,
                Some(&unit_ids),
            )
            .map_err(PyValueError::new_err)?;
            stage_se[lag] = covariance[[1, 1]].abs().sqrt();
        }

        self.coef = Some(coef);
        self.stage_se = Some(stage_se);
        self.n_units = Some(n_units);
        self.n_periods = Some(n_periods);
        Ok(())
    }

    #[getter]
    fn coef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegressionBlip is not fitted"))?;
        Ok(pyarray1_from_f64(py, coef))
    }

    #[getter]
    fn stage_standard_errors<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
        let se = self
            .stage_se
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegressionBlip is not fitted"))?;
        Ok(pyarray1_from_f64(py, se))
    }

    fn blip_down<'py>(
        &self,
        py: Python<'py>,
        y: PyReadonlyArray2<f64>,
        treatment: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegressionBlip is not fitted"))?;
        if y.shape() != treatment.shape() {
            return Err(PyValueError::new_err(
                "y and treatment must have the same shape",
            ));
        }
        let mut out = Array2::from_shape_vec(
            (y.shape()[0], y.shape()[1]),
            y.as_array().iter().copied().collect(),
        )
        .expect("invalid outcome shape");
        let treatment = Array2::from_shape_vec(
            (treatment.shape()[0], treatment.shape()[1]),
            treatment.as_array().iter().copied().collect(),
        )
        .expect("invalid treatment shape");
        for unit in 0..out.nrows() {
            for time in 0..out.ncols() {
                for lag in 0..=self.max_lag.min(time) {
                    out[[unit, time]] -= coef[lag] * treatment[[unit, time - lag]];
                }
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegressionBlip is not fitted"))?;
        let se = self
            .stage_se
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RegressionBlip is not fitted"))?;
        let lags = Array1::from_iter((0..=self.max_lag).map(|value| value as f64));
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("estimator", "sequential_regression_blip")?;
        dict.set_item("lags", pyarray1_from_f64(py, &lags))?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("stage_se", pyarray1_from_f64(py, se))?;
        dict.set_item("stage_se_scope", "conditional_on_earlier_blip_estimates")?;
        dict.set_item("time_effects", self.time_effects)?;
        dict.set_item("n_units", self.n_units)?;
        dict.set_item("n_periods", self.n_periods)?;
        Ok(dict.into())
    }
}
