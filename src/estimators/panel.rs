use crate::fit::FitDiagnostics;
use crate::rla::randomized_svd_impl;
use crate::utils::{add_intercept, invert_matrix, pyarray1_from_f64, pyarray2_from_f64, to_array2};
use crate::validation::validate_finite;
use nalgebra::DMatrix;
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    let mut u = svd
        .u
        .ok_or_else(|| PyValueError::new_err("SVD failed to return left singular vectors"))?;
    let vt = svd
        .v_t
        .ok_or_else(|| PyValueError::new_err("SVD failed to return right singular vectors"))?;
    let k = svd.singular_values.len();
    let mut shrunk = Array1::<f64>::zeros(k);
    for j in 0..k {
        let value = (svd.singular_values[j] - threshold).max(0.0);
        shrunk[j] = value;
        u.column_mut(j).scale_mut(value);
    }
    let reconstructed = u * vt;
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
    validate_finite("y", y).map_err(PyValueError::new_err)?;
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

pub(crate) struct PanelTreatmentInfo {
    pub(crate) first_treat: Vec<Option<usize>>,
    pub(crate) ever_treated: Vec<usize>,
    pub(crate) never_treated: Vec<usize>,
    pub(crate) cohorts: Vec<usize>,
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

pub(crate) fn infer_panel_treatment(
    y: &Array2<f64>,
    w: &Array2<f64>,
) -> PyResult<PanelTreatmentInfo> {
    if y.nrows() == 0 || y.ncols() == 0 {
        return Err(PyValueError::new_err("Y must be a non-empty 2D matrix"));
    }
    if w.raw_dim() != y.raw_dim() {
        return Err(PyValueError::new_err("W must have the same shape as Y"));
    }
    validate_finite("Y", y).map_err(PyValueError::new_err)?;
    validate_finite("W", w).map_err(PyValueError::new_err)?;

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

pub(crate) fn ensure_panel_has_never_treated(info: &PanelTreatmentInfo) -> PyResult<()> {
    if info.never_treated.is_empty() {
        return Err(PyValueError::new_err(
            "this estimator currently requires at least one never-treated donor unit",
        ));
    }
    Ok(())
}

pub(crate) fn cohort_units(info: &PanelTreatmentInfo, cohort: usize) -> Vec<usize> {
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

pub(crate) fn panel_effect_dicts<'py>(
    py: Python<'py>,
    y: &Array2<f64>,
    counterfactual: &Array2<f64>,
    treatment_info: &PanelTreatmentInfo,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let summaries = summarize_panel_effects(y, counterfactual, treatment_info);
    panel_summaries_to_dict(py, &summaries)
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

pub(crate) fn panel_group_pre_rmse(effects: &Array2<f64>, info: &PanelTreatmentInfo) -> f64 {
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
        self.fit = None;
        self.residuals = None;
        self.mu = None;
        self.alpha = None;
        self.xi = None;
        self.factor = None;
        self.loading = None;
        self.vnt = None;
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
    validate_finite("e", &e).map_err(PyValueError::new_err)?;
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
    validate_finite("e", &e).map_err(PyValueError::new_err)?;
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
    diagnostics: Option<FitDiagnostics>,
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
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be positive and finite",
            ));
        }
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if effect_iterations == 0 {
            return Err(PyValueError::new_err("effect_iterations must be positive"));
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
            diagnostics: None,
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
        self.completed = None;
        self.low_rank = None;
        self.unit_effects = None;
        self.time_effects = None;
        self.singular_values = None;
        self.fitted_lambda_l = None;
        self.diagnostics = None;
        self.y = None;
        self.w = None;
        self.treatment_info = None;
        self.treatment_effect = None;
        self.att = None;
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
        let mut converged = false;

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
            for i in 0..y_work.nrows() {
                for t in 0..y_work.ncols() {
                    if mask_arr[[i, t]] {
                        let fitted = low_rank[[i, t]] + row_effects[i] + col_effects[t];
                        let residual = y_work[[i, t]] - fitted;
                        projected[[i, t]] += residual;
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
            let mut rss = 0.0;
            for i in 0..y_work.nrows() {
                for t in 0..y_work.ncols() {
                    if mask_arr[[i, t]] {
                        let residual =
                            y_work[[i, t]] - low_rank[[i, t]] - row_effects[i] - col_effects[t];
                        rss += residual * residual;
                    }
                }
            }
            self.history_rmse.push((rss / n_obs).sqrt());
            final_iteration = iteration + 1;
            if let Some(prev) = previous_obj {
                let rel = (prev - obj).abs() / (prev.abs() + 1e-12);
                if obj <= prev && rel < self.tolerance {
                    converged = true;
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
        self.diagnostics = Some(FitDiagnostics::new(
            converged,
            final_iteration as u64,
            if converged {
                "Relative objective tolerance reached"
            } else {
                "Maximum number of iterations reached"
            },
            self.history_objective.last().copied(),
        ));
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

    #[pyo3(signature = (*, include_matrices=true))]
    fn summary<'py>(&self, py: Python<'py>, include_matrices: bool) -> PyResult<Py<PyAny>> {
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
        if include_matrices {
            dict.set_item("completed", pyarray2_from_f64(py, completed))?;
            dict.set_item("low_rank", pyarray2_from_f64(py, low_rank))?;
            dict.set_item("counterfactual", pyarray2_from_f64(py, completed))?;
            dict.set_item("treatment_effect", pyarray2_from_f64(py, treatment_effect))?;
        }
        dict.set_item("include_matrices", include_matrices)?;
        dict.set_item("unit_effects", pyarray1_from_f64(py, unit_effects))?;
        dict.set_item("time_effects", pyarray1_from_f64(py, time_effects))?;
        dict.set_item("singular_values", pyarray1_from_f64(py, singular_values))?;
        dict.set_item("lambda_l", self.fitted_lambda_l)?;
        self.diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?
            .write_summary(&dict)?;
        dict.set_item("history_objective", self.history_objective.clone())?;
        dict.set_item("history_rmse", self.history_rmse.clone())?;
        dict.set_item("svd_method", self.svd_method.clone())?;
        dict.set_item("svd_rank", self.svd_rank)?;
        dict.set_item("svd_oversamples", self.svd_oversamples)?;
        dict.set_item("svd_power_iter", self.svd_power_iter)?;
        dict.set_item("att", self.att)?;
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
        self.cohort_intercepts = None;
        self.cohort_coef = None;
        self.counterfactual = None;
        self.treatment_effect = None;
        self.att = None;
        self.pre_rmse = None;
        self.control_units = None;
        self.treated_units = None;
        self.cohorts = None;
        self.treatment_info = None;
        self.y = None;
        self.w = None;
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
