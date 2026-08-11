use crate::utils::{
    add_intercept, diag_sqrt, invert_matrix, pyarray1_from_f64, pyarray2_from_f64,
    sandwich_cov_from_parameter_scores, solve_least_squares_mat, solve_least_squares_vec,
};
use ndarray::{Array1, Array2, Array3};
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
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
