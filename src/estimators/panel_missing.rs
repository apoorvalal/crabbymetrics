//! Observation masks and available-case synthetic averages. No outcome filling.
use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(super) fn observed_outcomes(
    mut y: Array2<f64>,
    observed: Option<Array2<bool>>,
) -> PyResult<Array2<f64>> {
    if let Some(ref mask) = observed {
        if mask.dim() != y.dim() {
            return Err(PyValueError::new_err(
                "observed must have the same shape as Y",
            ));
        }
    }
    for ((i, t), value) in y.indexed_iter_mut() {
        if observed.as_ref().is_some_and(|m| !m[[i, t]]) {
            *value = f64::NAN;
        } else if value.is_infinite() || (observed.is_some() && !value.is_finite()) {
            return Err(PyValueError::new_err(
                "observed Y entries must be finite; use NaN or observed=False for missing outcomes",
            ));
        }
    }
    Ok(y)
}

pub(super) fn observation_summary(
    dict: &Bound<'_, PyDict>,
    y: &Array2<f64>,
    w: &Array2<f64>,
) -> PyResult<()> {
    let n = y.iter().filter(|v| v.is_finite()).count();
    let treated = y
        .indexed_iter()
        .filter(|(ij, v)| v.is_finite() && w[*ij] > 0.5)
        .count();
    dict.set_item("n_observed", n)?;
    dict.set_item("n_missing", y.len() - n)?;
    dict.set_item("n_untreated_observed", n - treated)?;
    dict.set_item("n_treated_observed", treated)?;
    dict.set_item("att_target", "observed treated cells")?;
    Ok(())
}

pub(super) fn finite_mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values
        .filter(|v| v.is_finite())
        .fold((0.0, 0), |(s, n), v| (s + v, n + 1));
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

pub(super) fn available_average(values: &Array1<f64>, weights: &Array1<f64>) -> f64 {
    let mut numerator = 0.0;
    let mut mass = 0.0;
    for (&value, &weight) in values.iter().zip(weights) {
        if value.is_finite() {
            numerator += weight * value;
            mass += weight;
        }
    }
    if mass > 1e-12 {
        numerator / mass
    } else {
        f64::NAN
    }
}

fn project_simplex(v: &Array1<f64>) -> Array1<f64> {
    let mut ordered = v.to_vec();
    ordered.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.0;
    let mut threshold = 0.0;
    for (j, value) in ordered.iter().enumerate() {
        sum += value;
        let candidate = (sum - 1.0) / (j + 1) as f64;
        if *value > candidate {
            threshold = candidate;
        }
    }
    v.mapv(|x| (x - threshold).max(0.0))
}

/// Profile the intercept in mean squared error of self-normalized averages.
/// Returns None on a zero available-weight denominator (never a zero outcome).
fn objective_gradient(
    design: &Array2<f64>,
    target: &Array1<f64>,
    weights: &Array1<f64>,
    zeta: f64,
) -> Option<(f64, Array1<f64>, f64)> {
    let n = target.len() as f64;
    let mut predictions = Array1::<f64>::zeros(target.len());
    let mut masses = Array1::<f64>::zeros(target.len());
    for i in 0..design.nrows() {
        for j in 0..design.ncols() {
            if design[[i, j]].is_finite() {
                masses[i] += weights[j];
                predictions[i] += weights[j] * design[[i, j]];
            }
        }
        if masses[i] <= 1e-12 {
            return None;
        }
        predictions[i] /= masses[i];
    }
    let intercept = (target - &predictions).sum() / n;
    let residual = &predictions + intercept - target;
    let objective = residual.dot(&residual) / n + zeta.powi(2) * weights.dot(weights);
    let mut gradient = weights * (2.0 * zeta.powi(2));
    for i in 0..design.nrows() {
        for j in 0..design.ncols() {
            if design[[i, j]].is_finite() {
                gradient[j] +=
                    2.0 * residual[i] * (design[[i, j]] - predictions[i]) / (n * masses[i]);
            }
        }
    }
    Some((objective, gradient, intercept))
}

pub(super) struct AvailableFit {
    pub weights: Array1<f64>,
    pub intercept: f64,
    pub objective: f64,
    pub iterations: u64,
    pub projected_gradient: f64,
}

/// A deterministic local solver for the masked (non-convex) simplex objective.
/// Columns with no training observations are excluded before calling this.
pub(super) fn fit_available_weights(
    design: &Array2<f64>,
    target: &Array1<f64>,
    zeta: f64,
    max_iterations: u64,
) -> PyResult<AvailableFit> {
    if design.nrows() == 0
        || design.ncols() == 0
        || target.len() != design.nrows()
        || target.iter().any(|v| !v.is_finite())
        || !zeta.is_finite()
        || zeta < 0.0
        || max_iterations == 0
    {
        return Err(PyValueError::new_err(
            "masked SDID weight problem has no valid overlap or tuning",
        ));
    }
    let mean = finite_mean(design.iter().copied());
    let scale = design
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| (v - mean).abs())
        .chain(target.iter().map(|v| (v - mean).abs()))
        .fold(0.0_f64, f64::max)
        .max(1e-12);
    let x = design.mapv(|v| (v - mean) / scale);
    let y = target.mapv(|v| (v - mean) / scale);
    let penalty = zeta / scale;
    let mut weights = Array1::from_elem(x.ncols(), 1.0 / x.ncols() as f64);
    let mut step: f64 = 1.0;
    for iteration in 0..=max_iterations {
        let (value, gradient, intercept) = objective_gradient(&x, &y, &weights, penalty)
            .ok_or_else(|| {
                PyValueError::new_err(
                    "masked SDID needs positive donor/time overlap in every fitting row",
                )
            })?;
        let projected = project_simplex(&(&weights - &gradient));
        let error = (&weights - &projected)
            .iter()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        if error < 1e-8 {
            return Ok(AvailableFit {
                weights,
                intercept: intercept * scale,
                objective: value * scale.powi(2),
                iterations: iteration,
                projected_gradient: error,
            });
        }
        if iteration == max_iterations {
            break;
        }
        let mut accepted = false;
        for _ in 0..60 {
            let candidate = project_simplex(&(&weights - &(&gradient * step)));
            let direction = &candidate - &weights;
            if let Some((next, _, _)) = objective_gradient(&x, &y, &candidate, penalty) {
                if next <= value + 1e-4 * gradient.dot(&direction) {
                    weights = candidate;
                    step = (step * 1.5).min(1e6);
                    accepted = true;
                    break;
                }
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    Err(PyValueError::new_err(
        "masked SDID weight optimizer did not converge; increase max_iterations or check overlap",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn masked_gradient_matches_finite_differences() {
        let x = array![[1., f64::NAN, 3.], [2., 3., 4.], [f64::NAN, 2., 1.]];
        let y = array![2., 2.7, 2.1];
        let w = array![0.2, 0.3, 0.5];
        let (_, gradient, _) = objective_gradient(&x, &y, &w, 0.3).unwrap();
        for j in 0..3 {
            let mut plus = w.clone();
            plus[j] += 1e-6;
            let mut minus = w.clone();
            minus[j] -= 1e-6;
            let numeric = (objective_gradient(&x, &y, &plus, 0.3).unwrap().0
                - objective_gradient(&x, &y, &minus, 0.3).unwrap().0)
                / 2e-6;
            assert!((numeric - gradient[j]).abs() < 1e-8);
        }
    }
}
