use super::panel::{
    cohort_units, ensure_panel_has_never_treated, infer_observed_panel_treatment,
    panel_effect_dicts, panel_group_pre_rmse, PanelTreatmentInfo,
};
use super::panel_missing::{
    available_average, finite_mean, fit_available_weights, observation_summary, observed_outcomes,
};
use crate::utils::{
    bootstrap_indices, pyarray1_from_f64, pyarray2_from_f64, take_rows, take_rows_vec, to_array1,
    to_array2,
};
use nalgebra::{DMatrix, DVector};
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn synthetic_control_rmse(
    donors: &Array2<f64>,
    treated: &Array1<f64>,
    weights: &Array1<f64>,
) -> f64 {
    let residual = donors.dot(weights) - treated;
    (residual.mapv(|value| value * value).mean().unwrap_or(0.0)).sqrt()
}

#[derive(Clone)]
struct SimplexFit {
    weights: Array1<f64>,
    iterations: u64,
    kkt_residual: f64,
}

fn fit_synthetic_control_weights(
    donors: &Array2<f64>,
    treated: &Array1<f64>,
    max_iterations: u64,
) -> PyResult<SimplexFit> {
    crate::validation::validate_finite("donors", donors).map_err(PyValueError::new_err)?;
    crate::validation::validate_finite("treated", treated).map_err(PyValueError::new_err)?;
    if max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
    if donors.nrows() == 0 || donors.ncols() == 0 || donors.nrows() != treated.len() {
        return Err(PyValueError::new_err(
            "need nonempty aligned donors/treated and positive max_iterations",
        ));
    }
    // Under sum(w)=1, ||D w-y|| = ||(D-y 1')w||. This avoids
    // cancellation from the large common outcome level in Gram - linear.
    let mut centered = donors.clone();
    for mut column in centered.columns_mut() {
        column -= treated;
    }
    let scale = centered.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
    if scale > 0.0 {
        centered /= scale;
    }
    let hessian = centered.t().dot(&centered) / donors.nrows() as f64;
    solve_simplex_quadratic_diagnostics(&hessian, &Array1::zeros(donors.ncols()), max_iterations)
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

pub(crate) fn fit_penalized_synthetic_control_weights(
    design: &Array2<f64>,
    target: &Array1<f64>,
    penalty: f64,
    intercept: bool,
    max_iterations: u64,
) -> PyResult<Array1<f64>> {
    if max_iterations == 0 {
        return Err(PyValueError::new_err("max_iterations must be positive"));
    }
    if design.nrows() != target.len() || design.nrows() == 0 || design.ncols() == 0 {
        return Err(PyValueError::new_err(
            "penalized synthetic control needs a non-empty design aligned with target",
        ));
    }
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(PyValueError::new_err(
            "unit_penalty must be finite and nonnegative",
        ));
    }
    if design.ncols() == 1 {
        return Ok(Array1::from_vec(vec![1.0]));
    }

    let count = design.len() as f64;
    let mean = design.sum() / count;
    let variance = design
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (count - 1.0);
    let scale = variance.sqrt();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(PyValueError::new_err(
            "penalized synthetic control donor outcomes have zero scale",
        ));
    }
    let mut scaled_design = design / scale;
    let mut scaled_target = target / scale;
    let mut distances = Array1::<f64>::zeros(design.ncols());
    for column in 0..design.ncols() {
        let difference = &scaled_target - &scaled_design.column(column);
        distances[column] = difference.dot(&difference);
    }
    if intercept {
        for column in 0..scaled_design.ncols() {
            let column_mean = scaled_design.column(column).mean().unwrap_or(0.0);
            scaled_design
                .column_mut(column)
                .mapv_inplace(|value| value - column_mean);
        }
        let target_mean = scaled_target.mean().unwrap_or(0.0);
        scaled_target.mapv_inplace(|value| value - target_mean);
    }
    let mut hessian = scaled_design.t().dot(&scaled_design);
    let numerical_ridge = 1e-12 * hessian.diag().iter().copied().fold(1.0_f64, f64::max);
    for index in 0..hessian.nrows() {
        hessian[[index, index]] += numerical_ridge;
    }
    let linear = scaled_design.t().dot(&scaled_target) - distances * (0.5 * penalty);
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
    Ok(solve_simplex_quadratic_diagnostics(hessian, linear, max_iterations)?.weights)
}

fn simplex_kkt(hessian: &Array2<f64>, linear: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let gradient = hessian.dot(weights) - linear;
    let multiplier = gradient.dot(weights);
    gradient
        .iter()
        .zip(weights)
        .fold(0.0_f64, |error, (&g, &w)| {
            error.max(if w > 0.0 {
                (g - multiplier).abs()
            } else {
                (multiplier - g).max(0.0)
            })
        })
}

fn solve_simplex_quadratic_diagnostics(
    hessian: &Array2<f64>,
    linear: &Array1<f64>,
    max_iterations: u64,
) -> PyResult<SimplexFit> {
    let n_weights = hessian.nrows();
    let mut weights = Array1::<f64>::from_elem(n_weights, 1.0 / n_weights as f64);
    let mut is_free = vec![true; n_weights];
    let tolerance = 1e-10;

    for iteration in 0..max_iterations {
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
                    let kkt_residual = simplex_kkt(hessian, linear, &weights);
                    if !kkt_residual.is_finite()
                        || kkt_residual
                            > 1e-8 * (1.0 + linear.iter().fold(0.0_f64, |a, &b| a.max(b.abs())))
                    {
                        return Err(PyValueError::new_err(
                            "simplex solver failed its KKT optimality check",
                        ));
                    }
                    return Ok(SimplexFit {
                        weights,
                        iterations: iteration + 1,
                        kkt_residual,
                    });
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
    diagnostics: Option<SimplexFit>,
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
    missing_diagnostics: Vec<MaskedSdidDiagnostics>,
}

#[pymethods]
impl SyntheticControl {
    #[new]
    #[pyo3(signature = (max_iterations=500))]
    fn new(max_iterations: u64) -> Self {
        Self {
            max_iterations,
            weights: None,
            diagnostics: None,
            donors: None,
            treated: None,
        }
    }

    fn fit(
        &mut self,
        donors: PyReadonlyArray2<f64>,
        treated: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        self.weights = None;
        self.diagnostics = None;
        self.donors = None;
        self.treated = None;
        let donors = to_array2(&donors);
        let treated = to_array1(&treated);
        let weights = fit_synthetic_control_weights(&donors, &treated, self.max_iterations)?;

        self.weights = Some(weights.weights.clone());
        self.diagnostics = Some(weights);
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
        crate::validation::validate_finite("donors", &donors).map_err(PyValueError::new_err)?;
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
        let diagnostics = self
            .diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No solver diagnostics stored"))?;
        dict.set_item("converged", diagnostics.kkt_residual <= 1e-8)?;
        dict.set_item("iterations", diagnostics.iterations)?;
        dict.set_item("stop_reason", "kkt_optimality")?;
        dict.set_item("solver", "active_set_simplex_qp")?;
        dict.set_item(
            "objective",
            0.5 * synthetic_control_rmse(donors, treated, weights).powi(2),
        )?;
        dict.set_item("scaled_kkt_residual", diagnostics.kkt_residual)?;
        dict.set_item(
            "feasibility_error",
            (weights.sum() - 1.0)
                .abs()
                .max(weights.iter().fold(0.0_f64, |a, &w| a.max(-w))),
        )?;
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
        for (i, idx) in idxs.enumerate() {
            let idx = &idx;
            let donors_b = take_rows(donors, idx);
            let treated_b = take_rows_vec(treated, idx);
            let weights_b =
                fit_synthetic_control_weights(&donors_b, &treated_b, self.max_iterations)?;
            out.row_mut(i).assign(&weights_b.weights);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}

struct MaskedSdidDiagnostics {
    cohort: usize,
    unit_iterations: u64,
    time_iterations: u64,
    unit_objective: f64,
    time_objective: f64,
    unit_gradient: f64,
    time_gradient: f64,
}

fn fit_synthetic_did_missing(
    y: &Array2<f64>,
    w: &Array2<f64>,
    zeta_omega_opt: Option<f64>,
    zeta_lambda_opt: Option<f64>,
    max_iterations: u64,
) -> PyResult<SyntheticDidFitResult> {
    let info = infer_observed_panel_treatment(y, w)?;
    ensure_panel_has_never_treated(&info)?;
    if info.cohorts.contains(&0) {
        return Err(PyValueError::new_err(
            "SyntheticDID needs at least one pre-treatment period for every treated cohort",
        ));
    }
    for zeta in [zeta_omega_opt, zeta_lambda_opt].into_iter().flatten() {
        if !zeta.is_finite() || zeta < 0.0 {
            return Err(PyValueError::new_err("zeta must be finite and nonnegative"));
        }
    }
    let (n, periods) = y.dim();
    let k = info.cohorts.len();
    let mut cf = Array2::from_elem((n, periods), f64::NAN);
    let mut effects = cf.clone();
    let mut omega = Array2::zeros((k, n));
    let mut lambda = Array2::zeros((k, periods));
    let mut unit_intercepts = Array1::zeros(k);
    let mut time_intercepts = Array1::zeros(k);
    let mut zeta_omegas = Array1::zeros(k);
    let mut zeta_lambdas = Array1::zeros(k);
    let mut diagnostics = Vec::new();
    let mut total = 0.0;
    let mut total_count = 0;
    for (c, &g) in info.cohorts.iter().enumerate() {
        let treated = cohort_units(&info, g);
        let means = Array1::from_iter(
            (0..periods).map(|t| finite_mean(treated.iter().map(|i| y[[*i, t]]))),
        );
        let counts: Vec<usize> = (0..periods)
            .map(|t| treated.iter().filter(|i| y[[**i, t]].is_finite()).count())
            .collect();
        let post_count: usize = counts[g..].iter().sum();
        if post_count == 0 {
            return Err(PyValueError::new_err(format!(
                "SDID cohort {g} has no observed treated post outcomes"
            )));
        }
        let pre: Vec<usize> = (0..g)
            .filter(|t| {
                means[*t].is_finite() && info.never_treated.iter().any(|i| y[[*i, *t]].is_finite())
            })
            .collect();
        let donors: Vec<usize> = info
            .never_treated
            .iter()
            .copied()
            .filter(|i| pre.iter().any(|t| y[[*i, *t]].is_finite()))
            .collect();
        if pre.is_empty() || donors.is_empty() {
            return Err(PyValueError::new_err(format!(
                "SDID cohort {g} has no observed donor/treated pre-period overlap"
            )));
        }
        let mut differences = Vec::new();
        for &i in &donors {
            for t in 1..g {
                if y[[i, t]].is_finite() && y[[i, t - 1]].is_finite() {
                    differences.push(y[[i, t]] - y[[i, t - 1]]);
                }
            }
        }
        let sigma = if differences.len() > 1 {
            let mean = finite_mean(differences.iter().copied());
            (differences.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (differences.len() - 1) as f64)
                .sqrt()
        } else {
            0.0
        };
        let zw = zeta_omega_opt.unwrap_or((post_count as f64).powf(0.25) * sigma);
        let zl = zeta_lambda_opt.unwrap_or(1e-6 * sigma);
        let unit_design =
            Array2::from_shape_fn((pre.len(), donors.len()), |(t, j)| y[[donors[j], pre[t]]]);
        let unit_fit = fit_available_weights(
            &unit_design,
            &means.select(Axis(0), &pre),
            zw,
            max_iterations,
        )?;
        let donor_post: Vec<(usize, f64)> = donors
            .iter()
            .copied()
            .filter_map(|i| {
                let available: Vec<usize> = (g..periods)
                    .filter(|t| counts[*t] > 0 && y[[i, *t]].is_finite())
                    .collect();
                let mass: usize = available.iter().map(|t| counts[*t]).sum();
                if mass == 0 {
                    None
                } else {
                    Some((
                        i,
                        available
                            .iter()
                            .map(|t| counts[*t] as f64 * y[[i, *t]])
                            .sum::<f64>()
                            / mass as f64,
                    ))
                }
            })
            .collect();
        // A pre-period absent for every time-weight training donor gets zero weight.
        let time_pre: Vec<usize> = pre
            .iter()
            .copied()
            .filter(|t| donor_post.iter().any(|(i, _)| y[[*i, *t]].is_finite()))
            .collect();
        let time_design = Array2::from_shape_fn((donor_post.len(), time_pre.len()), |(i, t)| {
            y[[donor_post[i].0, time_pre[t]]]
        });
        let time_target = Array1::from_iter(donor_post.iter().map(|(_, v)| *v));
        let time_fit = fit_available_weights(&time_design, &time_target, zl, max_iterations)?;
        for (j, i) in donors.iter().enumerate() {
            omega[[c, *i]] = unit_fit.weights[j];
        }
        for (j, t) in time_pre.iter().enumerate() {
            lambda[[c, *t]] = time_fit.weights[j];
        }
        let synthetic = Array1::from_iter((0..periods).map(|t| {
            available_average(
                &Array1::from_iter(donors.iter().map(|i| y[[*i, t]])),
                &unit_fit.weights,
            )
        }));
        let pre_gaps = Array1::from_iter(time_pre.iter().map(|t| means[*t] - synthetic[*t]));
        if pre_gaps.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "SDID cohort {g} has no positive donor weight for a pre-period contrast"
            )));
        }
        let correction = time_fit.weights.dot(&pre_gaps);
        for t in g..periods {
            if counts[t] > 0 {
                if !synthetic[t].is_finite() {
                    return Err(PyValueError::new_err(format!(
                        "SDID cohort {g}, period {t}: no observed positive-weight donors"
                    )));
                }
                total += counts[t] as f64 * (means[t] - synthetic[t] - correction);
                total_count += counts[t];
            }
        }
        for i in treated {
            for t in 0..periods {
                cf[[i, t]] = synthetic[t] + unit_fit.intercept;
                effects[[i, t]] = y[[i, t]] - cf[[i, t]];
            }
        }
        unit_intercepts[c] = unit_fit.intercept;
        time_intercepts[c] = time_fit.intercept;
        zeta_omegas[c] = zw;
        zeta_lambdas[c] = zl;
        diagnostics.push(MaskedSdidDiagnostics {
            cohort: g,
            unit_iterations: unit_fit.iterations,
            time_iterations: time_fit.iterations,
            unit_objective: unit_fit.objective,
            time_objective: time_fit.objective,
            unit_gradient: unit_fit.projected_gradient,
            time_gradient: time_fit.projected_gradient,
        });
    }
    let pre_rmse = panel_group_pre_rmse(&effects, &info);
    Ok(SyntheticDidFitResult {
        att: total / total_count as f64,
        unit_weights: omega,
        time_weights: lambda,
        counterfactual: cf,
        treatment_effect: effects,
        pre_rmse,
        unit_intercept: unit_intercepts,
        time_intercept: time_intercepts,
        fitted_zeta_omega: zeta_omegas,
        fitted_zeta_lambda: zeta_lambdas,
        control_units: info.never_treated.clone(),
        treated_units: info.ever_treated.clone(),
        cohorts: info.cohorts.clone(),
        treatment_info: info,
        missing_diagnostics: diagnostics,
    })
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
    missing_diagnostics: Vec<MaskedSdidDiagnostics>,
}

fn fit_synthetic_did_panel(
    y: &Array2<f64>,
    w: &Array2<f64>,
    zeta_omega_opt: Option<f64>,
    zeta_lambda_opt: Option<f64>,
    max_iterations: u64,
) -> PyResult<SyntheticDidFitResult> {
    if y.iter().any(|v| !v.is_finite()) {
        return fit_synthetic_did_missing(y, w, zeta_omega_opt, zeta_lambda_opt, max_iterations);
    }
    let treatment_info = infer_observed_panel_treatment(y, w)?;
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
        missing_diagnostics: Vec::new(),
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
    let info = infer_observed_panel_treatment(y, w)?;
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
    let info = infer_observed_panel_treatment(y, w)?;
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
    let info = infer_observed_panel_treatment(y, w)?;
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
            missing_diagnostics: Vec::new(),
        }
    }

    #[pyo3(signature = (y, w, *, observed=None))]
    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        w: PyReadonlyArray2<f64>,
        observed: Option<PyReadonlyArray2<bool>>,
    ) -> PyResult<()> {
        self.att = None;
        self.missing_diagnostics.clear();
        self.unit_weights = None;
        self.time_weights = None;
        self.counterfactual = None;
        self.treatment_effect = None;
        self.pre_rmse = None;
        self.unit_intercept = None;
        self.time_intercept = None;
        self.fitted_zeta_omega = None;
        self.fitted_zeta_lambda = None;
        self.control_units = None;
        self.treated_units = None;
        self.cohorts = None;
        self.treatment_info = None;
        self.y = None;
        self.w = None;
        let y = observed_outcomes(
            to_array2(&y),
            observed.map(|m| {
                let a = m.as_array();
                Array2::from_shape_fn((a.nrows(), a.ncols()), |ij| a[ij])
            }),
        )?;
        let w = to_array2(&w);
        let fit = fit_synthetic_did_panel(
            &y,
            &w,
            self.zeta_omega,
            self.zeta_lambda,
            self.max_iterations,
        )?;

        self.missing_diagnostics = fit.missing_diagnostics;
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
        observation_summary(&dict, y, self.w.as_ref().unwrap())?;
        let diagnostics = pyo3::types::PyList::empty(py);
        for d in &self.missing_diagnostics {
            let entry = pyo3::types::PyDict::new(py);
            entry.set_item("cohort", d.cohort)?;
            entry.set_item("unit_iterations", d.unit_iterations)?;
            entry.set_item("time_iterations", d.time_iterations)?;
            entry.set_item("unit_objective", d.unit_objective)?;
            entry.set_item("time_objective", d.time_objective)?;
            entry.set_item("unit_projected_gradient", d.unit_gradient)?;
            entry.set_item("time_projected_gradient", d.time_gradient)?;
            diagnostics.append(entry)?;
        }
        dict.set_item("missing_diagnostics", diagnostics)?;
        dict.set_item(
            "missing_method",
            if self.missing_diagnostics.is_empty() {
                "complete panel"
            } else {
                "available-case normalized weights"
            },
        )?;
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
