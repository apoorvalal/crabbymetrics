use super::panel::{
    cohort_units, ensure_panel_has_never_treated, infer_panel_treatment, panel_effect_dicts,
    panel_group_pre_rmse, PanelTreatmentInfo,
};
use super::synthetic::{fit_simplex_least_squares_weights, sdid_sigma_estimator};
use crate::utils::{pyarray1_from_f64, pyarray2_from_f64, to_array2};
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq)]
enum BalanceDimension {
    None,
    Unit,
    Time,
    Double,
}

impl BalanceDimension {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "none" => Ok(Self::None),
            "unit" => Ok(Self::Unit),
            "time" => Ok(Self::Time),
            "double" => Ok(Self::Double),
            _ => Err(PyValueError::new_err(
                "balance must be 'none', 'unit', 'time', or 'double'",
            )),
        }
    }

    fn uses_unit_weights(self) -> bool {
        matches!(self, Self::Unit | Self::Double)
    }

    fn uses_time_weights(self) -> bool {
        matches!(self, Self::Time | Self::Double)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum UnitTarget {
    Cohort,
    Individual,
}

impl UnitTarget {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "cohort" => Ok(Self::Cohort),
            "individual" => Ok(Self::Individual),
            _ => Err(PyValueError::new_err(
                "unit_target must be 'cohort' or 'individual'",
            )),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TimeTarget {
    All,
    Period,
}

impl TimeTarget {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "all" => Ok(Self::All),
            "period" => Ok(Self::Period),
            _ => Err(PyValueError::new_err(
                "time_target must be 'all' or 'period'",
            )),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BalanceData {
    Raw,
    Residual,
}

impl BalanceData {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "raw" => Ok(Self::Raw),
            "residual" => Ok(Self::Residual),
            _ => Err(PyValueError::new_err(
                "balance_on must be 'raw' or 'residual'",
            )),
        }
    }
}

struct AugmentedBalancingFit {
    att: f64,
    unit_weights: Array2<f64>,
    time_weights: Array2<f64>,
    counterfactual: Array2<f64>,
    treatment_effect: Array2<f64>,
    outcome_model: Array2<f64>,
    pre_rmse: f64,
    fitted_zeta_omega: Array1<f64>,
    fitted_zeta_lambda: Array1<f64>,
    target_units: Vec<i64>,
    target_cohorts: Vec<usize>,
    time_target_periods: Vec<i64>,
    time_target_cohorts: Vec<usize>,
    treatment_info: PanelTreatmentInfo,
}

fn mean_rows(matrix: &Array2<f64>, rows: &[usize], end_column: usize) -> PyResult<Array1<f64>> {
    matrix
        .select(Axis(0), rows)
        .slice(s![.., 0..end_column])
        .mean_axis(Axis(0))
        .ok_or_else(|| PyValueError::new_err("failed to average target pre-period outcomes"))
}

fn panel_att(treatment: &Array2<f64>, effects: &Array2<f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..treatment.nrows() {
        for t in 0..treatment.ncols() {
            if treatment[[i, t]] > 0.5 && effects[[i, t]].is_finite() {
                total += effects[[i, t]];
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        total / count as f64
    }
}

fn checked_penalty(value: Option<f64>, name: &str, default: f64) -> PyResult<f64> {
    match value {
        Some(penalty) if penalty.is_finite() && penalty >= 0.0 => Ok(penalty),
        Some(_) => Err(PyValueError::new_err(format!(
            "{name} must be finite and nonnegative"
        ))),
        None => Ok(default),
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_augmented_balancing_panel(
    y: &Array2<f64>,
    w: &Array2<f64>,
    outcome_model: &Array2<f64>,
    balance: BalanceDimension,
    unit_target: UnitTarget,
    time_target: TimeTarget,
    balance_on: BalanceData,
    zeta_omega_opt: Option<f64>,
    zeta_lambda_opt: Option<f64>,
    max_iterations: u64,
) -> PyResult<AugmentedBalancingFit> {
    if max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
    let treatment_info = infer_panel_treatment(y, w)?;
    ensure_panel_has_never_treated(&treatment_info)?;
    if treatment_info.cohorts.iter().any(|cohort| *cohort == 0) {
        return Err(PyValueError::new_err(
            "AugmentedBalancing needs at least one pre-treatment period for every treated cohort",
        ));
    }
    if outcome_model.raw_dim() != y.raw_dim() {
        return Err(PyValueError::new_err(
            "outcome_model must have the same shape as Y",
        ));
    }
    if !outcome_model.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(
            "outcome_model must contain only finite values",
        ));
    }

    let residual = y - outcome_model;
    let balancing_data = match balance_on {
        BalanceData::Raw => y,
        BalanceData::Residual => &residual,
    };
    let n_unit_targets = match balance {
        BalanceDimension::None => 0,
        _ => match unit_target {
            UnitTarget::Cohort => treatment_info.cohorts.len(),
            UnitTarget::Individual => treatment_info.ever_treated.len(),
        },
    };
    let n_time_targets = match balance {
        BalanceDimension::None => 0,
        _ => treatment_info
            .cohorts
            .iter()
            .map(|cohort| match time_target {
                TimeTarget::All => 1,
                TimeTarget::Period => y.ncols() - cohort,
            })
            .sum(),
    };
    let mut unit_weights = Array2::<f64>::zeros((n_unit_targets, y.nrows()));
    let mut time_weights = Array2::<f64>::zeros((n_time_targets, y.ncols()));
    let mut counterfactual = Array2::<f64>::from_elem(y.raw_dim(), f64::NAN);
    let mut treatment_effect = Array2::<f64>::from_elem(y.raw_dim(), f64::NAN);
    let mut fitted_zeta_omega = Array1::<f64>::zeros(n_unit_targets);
    let mut fitted_zeta_lambda = Array1::<f64>::zeros(n_time_targets);
    let mut target_units = Vec::with_capacity(n_unit_targets);
    let mut target_cohorts = Vec::with_capacity(n_unit_targets);
    let mut time_target_periods = Vec::with_capacity(n_time_targets);
    let mut time_target_cohorts = Vec::with_capacity(n_time_targets);
    let mut unit_target_row = 0usize;
    let mut time_target_row = 0usize;

    if balance == BalanceDimension::None {
        for unit in &treatment_info.ever_treated {
            for period in 0..y.ncols() {
                counterfactual[[*unit, period]] = outcome_model[[*unit, period]];
                treatment_effect[[*unit, period]] =
                    y[[*unit, period]] - outcome_model[[*unit, period]];
            }
        }
        let att = panel_att(w, &treatment_effect);
        let pre_rmse = panel_group_pre_rmse(&treatment_effect, &treatment_info);
        return Ok(AugmentedBalancingFit {
            att,
            unit_weights,
            time_weights,
            counterfactual,
            treatment_effect,
            outcome_model: outcome_model.clone(),
            pre_rmse,
            fitted_zeta_omega,
            fitted_zeta_lambda,
            target_units,
            target_cohorts,
            time_target_periods,
            time_target_cohorts,
            treatment_info,
        });
    }

    for cohort in &treatment_info.cohorts {
        let cohort_treated = cohort_units(&treatment_info, *cohort);
        let controls = &treatment_info.never_treated;
        let n_control = controls.len();
        let n_treated = cohort_treated.len();
        let t_pre = *cohort;
        let t_post = y.ncols() - t_pre;
        let cohort_data = balancing_data.select(Axis(0), controls);
        let cohort_residual = residual.select(Axis(0), controls);

        let sigma = sdid_sigma_estimator(&cohort_data, n_control, t_pre);
        let zeta_lambda = checked_penalty(zeta_lambda_opt, "zeta_lambda", 1e-6 * sigma)?;

        let control_pre = cohort_data.slice(s![.., 0..t_pre]).to_owned();
        let lambda_targets: Vec<(i64, Array1<f64>)> = match time_target {
            TimeTarget::All => vec![(
                -1,
                cohort_data
                    .slice(s![.., t_pre..])
                    .mean_axis(Axis(1))
                    .ok_or_else(|| {
                        PyValueError::new_err("failed to average control post outcomes")
                    })?,
            )],
            TimeTarget::Period => (t_pre..y.ncols())
                .map(|period| (period as i64, cohort_data.column(period).to_owned()))
                .collect(),
        };

        let mut lambdas = Vec::with_capacity(lambda_targets.len());
        for (target_period, lambda_target) in lambda_targets {
            let lambda = if balance.uses_time_weights() {
                fit_simplex_least_squares_weights(
                    &control_pre,
                    &lambda_target,
                    zeta_lambda,
                    true,
                    max_iterations,
                )?
            } else {
                Array1::<f64>::from_elem(t_pre, 1.0 / t_pre as f64)
            };
            time_weights
                .slice_mut(s![time_target_row, 0..t_pre])
                .assign(&lambda);
            fitted_zeta_lambda[time_target_row] = if balance.uses_time_weights() {
                zeta_lambda
            } else {
                0.0
            };
            time_target_periods.push(target_period);
            time_target_cohorts.push(*cohort);
            lambdas.push(lambda);
            time_target_row += 1;
        }

        let target_groups: Vec<Vec<usize>> = match unit_target {
            UnitTarget::Cohort => vec![cohort_treated.clone()],
            UnitTarget::Individual => cohort_treated.iter().map(|unit| vec![*unit]).collect(),
        };

        for target_group in target_groups {
            let default_target_size = match unit_target {
                UnitTarget::Cohort => n_treated,
                UnitTarget::Individual => 1,
            };
            let zeta_omega = checked_penalty(
                zeta_omega_opt,
                "zeta_omega",
                ((default_target_size * t_post) as f64).powf(0.25) * sigma,
            )?;
            let omega = if balance.uses_unit_weights() {
                let target_pre = mean_rows(balancing_data, &target_group, t_pre)?;
                let design = control_pre.t().to_owned();
                fit_simplex_least_squares_weights(
                    &design,
                    &target_pre,
                    zeta_omega,
                    true,
                    max_iterations,
                )?
            } else {
                Array1::<f64>::from_elem(n_control, 1.0 / n_control as f64)
            };

            for (column, unit) in controls.iter().enumerate() {
                unit_weights[[unit_target_row, *unit]] = omega[column];
            }
            fitted_zeta_omega[unit_target_row] = if balance.uses_unit_weights() {
                zeta_omega
            } else {
                0.0
            };
            target_units.push(if unit_target == UnitTarget::Individual {
                target_group[0] as i64
            } else {
                -1
            });
            target_cohorts.push(*cohort);
            unit_target_row += 1;

            let donor_pre = cohort_residual.slice(s![.., 0..t_pre]);

            for unit in target_group {
                for period in 0..t_pre {
                    let predicted =
                        outcome_model[[unit, period]] + omega.dot(&cohort_residual.column(period));
                    counterfactual[[unit, period]] = predicted;
                    treatment_effect[[unit, period]] = y[[unit, period]] - predicted;
                }
                for period in t_pre..y.ncols() {
                    let lambda_index = match time_target {
                        TimeTarget::All => 0,
                        TimeTarget::Period => period - t_pre,
                    };
                    let lambda = &lambdas[lambda_index];
                    let unit_correction = omega.dot(&cohort_residual.column(period));
                    let treated_history = residual.slice(s![unit, 0..t_pre]).dot(lambda);
                    let donor_history = donor_pre.dot(lambda);
                    let interaction_correction = omega.dot(&donor_history);
                    let predicted =
                        outcome_model[[unit, period]] + unit_correction + treated_history
                            - interaction_correction;
                    counterfactual[[unit, period]] = predicted;
                    treatment_effect[[unit, period]] = y[[unit, period]] - predicted;
                }
            }
        }
    }

    let att = panel_att(w, &treatment_effect);
    let pre_rmse = panel_group_pre_rmse(&treatment_effect, &treatment_info);
    Ok(AugmentedBalancingFit {
        att,
        unit_weights,
        time_weights,
        counterfactual,
        treatment_effect,
        outcome_model: outcome_model.clone(),
        pre_rmse,
        fitted_zeta_omega,
        fitted_zeta_lambda,
        target_units,
        target_cohorts,
        time_target_periods,
        time_target_cohorts,
        treatment_info,
    })
}

/// Outcome-model augmentation with optional unit and time balancing for panel ATT.
#[pyclass]
pub struct AugmentedBalancing {
    balance: String,
    unit_target: String,
    time_target: String,
    balance_on: String,
    zeta_omega: Option<f64>,
    zeta_lambda: Option<f64>,
    max_iterations: u64,
    att: Option<f64>,
    unit_weights: Option<Array2<f64>>,
    time_weights: Option<Array2<f64>>,
    counterfactual: Option<Array2<f64>>,
    treatment_effect_values: Option<Array2<f64>>,
    outcome_model: Option<Array2<f64>>,
    pre_rmse: Option<f64>,
    fitted_zeta_omega: Option<Array1<f64>>,
    fitted_zeta_lambda: Option<Array1<f64>>,
    target_units: Option<Vec<i64>>,
    target_cohorts: Option<Vec<usize>>,
    time_target_periods: Option<Vec<i64>>,
    time_target_cohorts: Option<Vec<usize>>,
    treatment_info: Option<PanelTreatmentInfo>,
    y: Option<Array2<f64>>,
}

#[pymethods]
impl AugmentedBalancing {
    #[new]
    #[pyo3(signature = (balance="double", unit_target="cohort", time_target="all", balance_on="raw", zeta_omega=None, zeta_lambda=None, max_iterations=1000))]
    fn new(
        balance: &str,
        unit_target: &str,
        time_target: &str,
        balance_on: &str,
        zeta_omega: Option<f64>,
        zeta_lambda: Option<f64>,
        max_iterations: u64,
    ) -> PyResult<Self> {
        BalanceDimension::parse(balance)?;
        UnitTarget::parse(unit_target)?;
        TimeTarget::parse(time_target)?;
        BalanceData::parse(balance_on)?;
        checked_penalty(zeta_omega, "zeta_omega", 0.0)?;
        checked_penalty(zeta_lambda, "zeta_lambda", 0.0)?;
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        Ok(Self {
            balance: balance.to_string(),
            unit_target: unit_target.to_string(),
            time_target: time_target.to_string(),
            balance_on: balance_on.to_string(),
            zeta_omega,
            zeta_lambda,
            max_iterations,
            att: None,
            unit_weights: None,
            time_weights: None,
            counterfactual: None,
            treatment_effect_values: None,
            outcome_model: None,
            pre_rmse: None,
            fitted_zeta_omega: None,
            fitted_zeta_lambda: None,
            target_units: None,
            target_cohorts: None,
            time_target_periods: None,
            time_target_cohorts: None,
            treatment_info: None,
            y: None,
        })
    }

    #[pyo3(signature = (y, w, outcome_model=None))]
    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        w: PyReadonlyArray2<f64>,
        outcome_model: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let y = to_array2(&y);
        let w = to_array2(&w);
        let outcome_model = outcome_model
            .map(|matrix| to_array2(&matrix))
            .unwrap_or_else(|| Array2::<f64>::zeros(y.raw_dim()));
        let fit = fit_augmented_balancing_panel(
            &y,
            &w,
            &outcome_model,
            BalanceDimension::parse(&self.balance)?,
            UnitTarget::parse(&self.unit_target)?,
            TimeTarget::parse(&self.time_target)?,
            BalanceData::parse(&self.balance_on)?,
            self.zeta_omega,
            self.zeta_lambda,
            self.max_iterations,
        )?;

        self.att = Some(fit.att);
        self.unit_weights = Some(fit.unit_weights);
        self.time_weights = Some(fit.time_weights);
        self.counterfactual = Some(fit.counterfactual);
        self.treatment_effect_values = Some(fit.treatment_effect);
        self.outcome_model = Some(fit.outcome_model);
        self.pre_rmse = Some(fit.pre_rmse);
        self.fitted_zeta_omega = Some(fit.fitted_zeta_omega);
        self.fitted_zeta_lambda = Some(fit.fitted_zeta_lambda);
        self.target_units = Some(fit.target_units);
        self.target_cohorts = Some(fit.target_cohorts);
        self.time_target_periods = Some(fit.time_target_periods);
        self.time_target_cohorts = Some(fit.time_target_cohorts);
        self.treatment_info = Some(fit.treatment_info);
        self.y = Some(y);
        Ok(())
    }

    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let counterfactual = self
            .counterfactual
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("AugmentedBalancing model is not fitted"))?;
        Ok(pyarray2_from_f64(py, counterfactual))
    }

    fn treatment_effect<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let effects = self
            .treatment_effect_values
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("AugmentedBalancing model is not fitted"))?;
        Ok(pyarray2_from_f64(py, effects))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let att = self
            .att
            .ok_or_else(|| PyValueError::new_err("AugmentedBalancing model is not fitted"))?;
        let counterfactual = self.counterfactual.as_ref().unwrap();
        let effects = self.treatment_effect_values.as_ref().unwrap();
        let outcome_model = self.outcome_model.as_ref().unwrap();
        let treatment_info = self.treatment_info.as_ref().unwrap();
        let y = self.y.as_ref().unwrap();
        let (event_study, group_means) = panel_effect_dicts(py, y, counterfactual, treatment_info)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("att", att)?;
        dict.set_item(
            "unit_weights",
            pyarray2_from_f64(py, self.unit_weights.as_ref().unwrap()),
        )?;
        dict.set_item(
            "time_weights",
            pyarray2_from_f64(py, self.time_weights.as_ref().unwrap()),
        )?;
        dict.set_item("counterfactual", pyarray2_from_f64(py, counterfactual))?;
        dict.set_item("treatment_effect", pyarray2_from_f64(py, effects))?;
        dict.set_item("outcome_model", pyarray2_from_f64(py, outcome_model))?;
        dict.set_item("event_study", event_study)?;
        dict.set_item("group_means", group_means)?;
        dict.set_item("pre_rmse", self.pre_rmse)?;
        dict.set_item(
            "zeta_omega",
            pyarray1_from_f64(py, self.fitted_zeta_omega.as_ref().unwrap()),
        )?;
        dict.set_item(
            "zeta_lambda",
            pyarray1_from_f64(py, self.fitted_zeta_lambda.as_ref().unwrap()),
        )?;
        dict.set_item("target_units", self.target_units.clone())?;
        dict.set_item("target_cohorts", self.target_cohorts.clone())?;
        dict.set_item("time_target_periods", self.time_target_periods.clone())?;
        dict.set_item("time_target_cohorts", self.time_target_cohorts.clone())?;
        dict.set_item("control_units", treatment_info.never_treated.clone())?;
        dict.set_item("treated_units", treatment_info.ever_treated.clone())?;
        dict.set_item("cohorts", treatment_info.cohorts.clone())?;
        dict.set_item("balance", self.balance.clone())?;
        dict.set_item("unit_target", self.unit_target.clone())?;
        dict.set_item("time_target", self.time_target.clone())?;
        dict.set_item("balance_on", self.balance_on.clone())?;
        dict.set_item("converged", true)?;
        Ok(dict.into())
    }
}
