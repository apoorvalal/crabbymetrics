use crate::fit::FitDiagnostics;
use crate::utils::{invert_matrix, pyarray1_from_f64, pyarray2_from_f64, to_array1, to_array2};
use crate::validation::{validate_binary_f64, validate_finite, validate_positive};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn validate_binary_event(event: &Array1<f64>, n: usize) -> PyResult<()> {
    if event.len() != n {
        return Err(PyValueError::new_err("event length must match time length"));
    }
    validate_binary_f64("event indicators", event).map_err(PyValueError::new_err)
}

fn validate_x(x: &Array2<f64>, n: usize) -> PyResult<()> {
    if x.nrows() != n {
        return Err(PyValueError::new_err("x rows must match time length"));
    }
    validate_finite("x", x).map_err(PyValueError::new_err)
}

fn max_abs(v: &Array1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

fn solve_newton(hess: &Array2<f64>, grad: &Array1<f64>, ridge: f64) -> Result<Array1<f64>, String> {
    let mut h = hess.clone();
    for i in 0..h.nrows() {
        h[[i, i]] -= ridge;
    }
    let inv = invert_matrix(&h)?;
    Ok(inv.dot(grad))
}

fn logistic_survival_p_value(stat: f64) -> f64 {
    // For documentation diagnostics only: chi-square(1) survival = erfc(sqrt(x/2)).
    // Abramowitz-Stegun approximation to erf.
    if stat <= 0.0 {
        return 1.0;
    }
    let x = (stat / 2.0).sqrt();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let erf = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    (1.0 - erf).clamp(0.0, 1.0)
}

fn parametric_ll_grad_hess(
    theta: &Array1<f64>,
    x: &Array2<f64>,
    time: &Array1<f64>,
    event: &Array1<f64>,
    weibull: bool,
) -> (f64, Array1<f64>, Array2<f64>) {
    let n = time.len();
    let p = x.ncols();
    let k = 1 + p + if weibull { 1 } else { 0 };
    let alpha = theta[0];
    let gamma = if weibull { theta[k - 1] } else { 0.0 };
    let rho = gamma.exp();
    let mut ll = 0.0;
    let mut grad = Array1::<f64>::zeros(k);
    let mut hess = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let t = time[i];
        let d = event[i];
        let log_t = t.ln();
        let xb = x.row(i).dot(&theta.slice(ndarray::s![1..1 + p]));
        let z = (alpha + xb + rho * log_t).exp(); // cumulative hazard
        ll += d * (alpha + gamma + xb + (rho - 1.0) * log_t) - z;

        let common = d - z;
        grad[0] += common;
        for j in 0..p {
            grad[1 + j] += common * x[[i, j]];
        }
        if weibull {
            grad[k - 1] += d * (1.0 + rho * log_t) - z * rho * log_t;
        }

        // alpha/beta block: -z * w w'
        let mut w = Vec::with_capacity(1 + p);
        w.push(1.0);
        for j in 0..p {
            w.push(x[[i, j]]);
        }
        for a in 0..(1 + p) {
            for b in 0..(1 + p) {
                hess[[a, b]] -= z * w[a] * w[b];
            }
        }
        if weibull {
            let gidx = k - 1;
            let dz_dg_factor = rho * log_t;
            for a in 0..(1 + p) {
                hess[[a, gidx]] -= z * w[a] * dz_dg_factor;
                hess[[gidx, a]] = hess[[a, gidx]];
            }
            hess[[gidx, gidx]] +=
                d * rho * log_t - z * (dz_dg_factor * dz_dg_factor + dz_dg_factor);
        }
    }
    (ll, grad, hess)
}

struct SurvivalFit {
    params: Array1<f64>,
    vcov: Array2<f64>,
    log_likelihood: f64,
    diagnostics: FitDiagnostics,
}

fn fit_parametric(
    x: &Array2<f64>,
    time: &Array1<f64>,
    event: &Array1<f64>,
    weibull: bool,
    max_iter: usize,
    tol: f64,
) -> Result<SurvivalFit, String> {
    if max_iter == 0 {
        return Err("max_iterations must be positive".to_string());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tolerance must be positive and finite".to_string());
    }
    let p = x.ncols();
    let k = 1 + p + if weibull { 1 } else { 0 };
    let mut theta = Array1::<f64>::zeros(k);
    let events = event.sum().max(1.0);
    let total_time = time.sum().max(1e-12);
    theta[0] = (events / total_time).ln();
    if weibull {
        theta[k - 1] = 0.0;
    }
    let mut termination_reason = None;
    let mut used = 0_u64;
    for iter in 0..max_iter {
        used = (iter + 1) as u64;
        let (ll, grad, hess) = parametric_ll_grad_hess(&theta, x, time, event, weibull);
        if max_abs(&grad) < tol {
            termination_reason = Some("Gradient tolerance reached".to_string());
            break;
        }
        let mut step = solve_newton(&hess, &grad, 1e-8)?;
        step.mapv_inplace(|v| v.clamp(-2.0, 2.0));
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let cand = &theta - &(step.mapv(|v| scale * v));
            let (cand_ll, _, _) = parametric_ll_grad_hess(&cand, x, time, event, weibull);
            if cand_ll.is_finite() && cand_ll >= ll - 1e-10 {
                theta = cand;
                accepted = true;
                if (cand_ll - ll).abs() < tol * (1.0 + ll.abs()) {
                    termination_reason = Some("Relative objective tolerance reached".to_string());
                }
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return Err("Newton line search failed".to_string());
        }
        if termination_reason.is_some() {
            break;
        }
    }
    let (log_likelihood, _, hess) = parametric_ll_grad_hess(&theta, x, time, event, weibull);
    let diagnostics = FitDiagnostics::new(
        termination_reason.is_some(),
        used,
        termination_reason.unwrap_or_else(|| "Maximum number of iterations reached".to_string()),
        Some(-log_likelihood),
    );
    diagnostics.require_converged(if weibull {
        "WeibullPH"
    } else {
        "ExponentialPH"
    })?;
    let info = hess.mapv(|v| -v);
    let vcov = invert_matrix(&info)?;
    Ok(SurvivalFit {
        params: theta,
        vcov,
        log_likelihood,
        diagnostics,
    })
}

#[pyclass]
pub struct ExponentialPH {
    coef: Option<Array1<f64>>,
    log_baseline_hazard: f64,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    diagnostics: Option<FitDiagnostics>,
}

#[pymethods]
impl ExponentialPH {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            log_baseline_hazard: 0.0,
            vcov: None,
            log_likelihood: f64::NAN,
            diagnostics: None,
        }
    }

    #[pyo3(signature = (x, time, event, max_iterations=100, tolerance=1e-8))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        self.coef = None;
        self.vcov = None;
        self.diagnostics = None;
        let x = to_array2(&x);
        let time = to_array1(&time);
        let event = to_array1(&event);
        validate_x(&x, time.len())?;
        validate_binary_event(&event, time.len())?;
        validate_positive("time", &time).map_err(PyValueError::new_err)?;
        let fit = fit_parametric(&x, &time, &event, false, max_iterations, tolerance)
            .map_err(PyValueError::new_err)?;
        self.log_baseline_hazard = fit.params[0];
        self.coef = Some(fit.params.slice(ndarray::s![1..]).to_owned());
        self.vcov = Some(fit.vcov);
        self.log_likelihood = fit.log_likelihood;
        self.diagnostics = Some(fit.diagnostics);
        Ok(())
    }

    fn predict_lin<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let out = x.dot(coef) + self.log_baseline_hazard;
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict_hazard<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let out = Array1::from_iter(
            (0..x.nrows()).map(|i| (self.log_baseline_hazard + x.row(i).dot(coef)).exp()),
        );
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict_cumulative_hazard<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        let time = to_array1(&time);
        if x.nrows() != time.len() || x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x/time dimensions do not match fitted model",
            ));
        }
        let lambda = self.log_baseline_hazard.exp();
        let out =
            Array1::from_iter((0..time.len()).map(|i| lambda * time[i] * x.row(i).dot(coef).exp()));
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict_survival<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        let time = to_array1(&time);
        if x.nrows() != time.len() || x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x/time dimensions do not match fitted model",
            ));
        }
        let lambda = self.log_baseline_hazard.exp();
        let out = Array1::from_iter(
            (0..time.len()).map(|i| (-(lambda * time[i] * x.row(i).dot(coef).exp())).exp()),
        );
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_survival(py, x, time)
    }

    fn predict_log_hazard<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_lin(py, x)
    }

    fn survival<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_survival(py, x, time)
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let vcov = self.vcov.as_ref().unwrap();
        let dict = PyDict::new(py);
        dict.set_item("log_baseline_hazard", self.log_baseline_hazard)?;
        dict.set_item("baseline_hazard", self.log_baseline_hazard.exp())?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("hazard_ratio", pyarray1_from_f64(py, &coef.mapv(f64::exp)))?;
        dict.set_item("vcov", pyarray2_from_f64(py, vcov))?;
        dict.set_item("log_likelihood", self.log_likelihood)?;
        self.diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?
            .write_summary(&dict)?;
        Ok(dict.into())
    }
}

#[pyclass]
pub struct WeibullPH {
    coef: Option<Array1<f64>>,
    log_scale_hazard: f64,
    log_shape: f64,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    diagnostics: Option<FitDiagnostics>,
}

#[pymethods]
impl WeibullPH {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            log_scale_hazard: 0.0,
            log_shape: 0.0,
            vcov: None,
            log_likelihood: f64::NAN,
            diagnostics: None,
        }
    }

    #[pyo3(signature = (x, time, event, max_iterations=100, tolerance=1e-8))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        self.coef = None;
        self.vcov = None;
        self.diagnostics = None;
        let x = to_array2(&x);
        let time = to_array1(&time);
        let event = to_array1(&event);
        validate_x(&x, time.len())?;
        validate_binary_event(&event, time.len())?;
        validate_positive("time", &time).map_err(PyValueError::new_err)?;
        let fit = fit_parametric(&x, &time, &event, true, max_iterations, tolerance)
            .map_err(PyValueError::new_err)?;
        let p = x.ncols();
        self.log_scale_hazard = fit.params[0];
        self.coef = Some(fit.params.slice(ndarray::s![1..1 + p]).to_owned());
        self.log_shape = fit.params[1 + p];
        self.vcov = Some(fit.vcov);
        self.log_likelihood = fit.log_likelihood;
        self.diagnostics = Some(fit.diagnostics);
        Ok(())
    }

    fn predict_survival<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        let time = to_array1(&time);
        if x.nrows() != time.len() || x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x/time dimensions do not match fitted model",
            ));
        }
        let lambda = self.log_scale_hazard.exp();
        let rho = self.log_shape.exp();
        let out = Array1::from_iter(
            (0..time.len())
                .map(|i| (-(lambda * time[i].powf(rho) * x.row(i).dot(coef).exp())).exp()),
        );
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_survival(py, x, time)
    }

    fn predict_lin<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        let time = to_array1(&time);
        if x.nrows() != time.len() || x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x/time dimensions do not match fitted model",
            ));
        }
        let rho = self.log_shape.exp();
        let out = Array1::from_iter((0..time.len()).map(|i| {
            self.log_scale_hazard + self.log_shape + (rho - 1.0) * time[i].ln() + x.row(i).dot(coef)
        }));
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict_hazard<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        let time = to_array1(&time);
        if x.nrows() != time.len() || x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x/time dimensions do not match fitted model",
            ));
        }
        let rho = self.log_shape.exp();
        let out = Array1::from_iter((0..time.len()).map(|i| {
            (self.log_scale_hazard
                + self.log_shape
                + (rho - 1.0) * time[i].ln()
                + x.row(i).dot(coef))
            .exp()
        }));
        Ok(pyarray1_from_f64(py, &out))
    }

    fn predict_cumulative_hazard<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        let time = to_array1(&time);
        if x.nrows() != time.len() || x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x/time dimensions do not match fitted model",
            ));
        }
        let lambda = self.log_scale_hazard.exp();
        let rho = self.log_shape.exp();
        let out = Array1::from_iter(
            (0..time.len()).map(|i| lambda * time[i].powf(rho) * x.row(i).dot(coef).exp()),
        );
        Ok(pyarray1_from_f64(py, &out))
    }

    fn survival<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_survival(py, x, time)
    }

    fn predict_log_hazard<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_lin(py, x, time)
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let vcov = self.vcov.as_ref().unwrap();
        let dict = PyDict::new(py);
        dict.set_item("log_scale_hazard", self.log_scale_hazard)?;
        dict.set_item("shape", self.log_shape.exp())?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("hazard_ratio", pyarray1_from_f64(py, &coef.mapv(f64::exp)))?;
        dict.set_item("vcov", pyarray2_from_f64(py, vcov))?;
        dict.set_item("log_likelihood", self.log_likelihood)?;
        self.diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?
            .write_summary(&dict)?;
        Ok(dict.into())
    }
}

struct RiskMoments {
    log_scale: f64,
    mass: f64,
    first: Array1<f64>,
    second: Array2<f64>,
}

impl RiskMoments {
    fn new(p: usize) -> Self {
        Self {
            log_scale: f64::NEG_INFINITY,
            mass: 0.0,
            first: Array1::zeros(p),
            second: Array2::zeros((p, p)),
        }
    }

    fn add(&mut self, eta: f64, x: ndarray::ArrayView1<'_, f64>, derivatives: bool) {
        if eta > self.log_scale {
            let factor = (self.log_scale - eta).exp();
            self.mass *= factor;
            self.first *= factor;
            self.second *= factor;
            self.log_scale = eta;
        }
        let weight = (eta - self.log_scale).exp();
        self.mass += weight;
        if derivatives {
            for a in 0..x.len() {
                self.first[a] += weight * x[a];
                for b in 0..x.len() {
                    self.second[[a, b]] += weight * x[a] * x[b];
                }
            }
        }
    }
}

fn cox_ll_grad_hess(
    beta: &Array1<f64>,
    x: &Array2<f64>,
    start: Option<&Array1<f64>>,
    stop: &Array1<f64>,
    event: &Array1<f64>,
    order: &[usize],
    derivatives: bool,
) -> (f64, Array1<f64>, Array2<f64>) {
    let n = stop.len();
    let p = x.ncols();
    let eta = x.dot(beta);
    let mut ll = 0.0;
    let mut grad = Array1::<f64>::zeros(p);
    let mut hess = Array2::<f64>::zeros((p, p));
    if eta.iter().any(|v| !v.is_finite()) {
        return (f64::NAN, grad, hess);
    }
    let mut risk = RiskMoments::new(p);
    let mut first = 0;
    while first < n {
        let t = stop[order[first]];
        let mut end = first + 1;
        while end < n && stop[order[end]] == t {
            end += 1;
        }
        if let Some(entry) = start {
            risk = RiskMoments::new(p);
            for r in 0..n {
                if entry[r] < t && stop[r] >= t {
                    risk.add(eta[r], x.row(r), derivatives);
                }
            }
        } else {
            // Descending stop times make right-censored risk sets cumulative.
            for &r in &order[first..end] {
                risk.add(eta[r], x.row(r), derivatives);
            }
        }
        for &i in &order[first..end] {
            if event[i] != 1.0 {
                continue;
            }
            if risk.mass <= 0.0 {
                return (f64::NAN, grad, hess);
            }
            ll += (eta[i] - risk.log_scale) - risk.mass.ln();
            if derivatives {
                for a in 0..p {
                    let mean_a = risk.first[a] / risk.mass;
                    grad[a] += x[[i, a]] - mean_a;
                    for b in 0..p {
                        hess[[a, b]] -=
                            risk.second[[a, b]] / risk.mass - mean_a * risk.first[b] / risk.mass;
                    }
                }
            }
        }
        first = end;
    }
    (ll, grad, hess)
}

#[cfg(test)]
mod cox_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn risk_sums_match_breslow_and_finite_differences() {
        let x = array![
            [0.2, -1.0],
            [1.0, 0.3],
            [-0.7, 0.8],
            [0.1, 0.6],
            [0.8, -0.2]
        ];
        let stop = array![2.0, 3.0, 3.0, 5.0, 7.0];
        let entry = array![0.0, 1.0, 0.0, 2.0, 3.0];
        let event = array![1.0, 1.0, 0.0, 1.0, 1.0];
        let beta = array![0.4, -0.2];
        let order = vec![4, 3, 1, 2, 0];
        for start in [None, Some(&entry)] {
            let (ll, grad, hess) = cox_ll_grad_hess(&beta, &x, start, &stop, &event, &order, true);
            let eta = x.dot(&beta);
            let mut reference = 0.0;
            for i in 0..stop.len() {
                if event[i] == 1.0 {
                    let sum: f64 = (0..stop.len())
                        .filter(|&j| stop[j] >= stop[i] && start.is_none_or(|s| s[j] < stop[i]))
                        .map(|j| eta[j].exp())
                        .sum();
                    reference += eta[i] - sum.ln();
                }
            }
            assert!((ll - reference).abs() < 1e-12);
            assert!(
                (ll - cox_ll_grad_hess(&beta, &x, start, &stop, &event, &order, false).0).abs()
                    < 1e-12
            );
            for j in 0..2 {
                let mut hi = beta.clone();
                let mut lo = beta.clone();
                hi[j] += 1e-5;
                lo[j] -= 1e-5;
                let upper = cox_ll_grad_hess(&hi, &x, start, &stop, &event, &order, true);
                let lower = cox_ll_grad_hess(&lo, &x, start, &stop, &event, &order, true);
                assert!((grad[j] - (upper.0 - lower.0) / 2e-5).abs() < 1e-8);
                for k in 0..2 {
                    assert!((hess[[k, j]] - (upper.1[k] - lower.1[k]) / 2e-5).abs() < 1e-8);
                }
            }
        }
    }
}

fn fit_cox_core(
    x: &Array2<f64>,
    start: Option<&Array1<f64>>,
    stop: &Array1<f64>,
    event: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<SurvivalFit, String> {
    if max_iter == 0 {
        return Err("max_iterations must be positive".to_string());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tolerance must be positive and finite".to_string());
    }
    if event.iter().all(|&v| v == 0.0) {
        return Err("Cox inference requires at least one event".to_string());
    }
    let center = x
        .mean_axis(ndarray::Axis(0))
        .ok_or("x must have observations")?;
    let centered = x - &center;
    let x = &centered;
    let mut order: Vec<usize> = (0..stop.len()).collect();
    order.sort_by(|&a, &b| stop[b].total_cmp(&stop[a]));
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    let mut termination_reason = None;
    let mut used = 0_u64;
    for iter in 0..max_iter {
        used = (iter + 1) as u64;
        let (ll, grad, hess) = cox_ll_grad_hess(&beta, x, start, stop, event, &order, true);
        if max_abs(&grad) < tol {
            termination_reason = Some("Gradient tolerance reached".to_string());
            break;
        }
        let mut step = solve_newton(&hess, &grad, 1e-8)?;
        step.mapv_inplace(|v| v.clamp(-1.0, 1.0));
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let cand = &beta - &(step.mapv(|v| scale * v));
            let (cand_ll, _, _) = cox_ll_grad_hess(&cand, x, start, stop, event, &order, false);
            if cand_ll.is_finite() && cand_ll >= ll - 1e-10 {
                beta = cand;
                accepted = true;
                if (cand_ll - ll).abs() < tol * (1.0 + ll.abs()) {
                    termination_reason = Some("Relative objective tolerance reached".to_string());
                }
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return Err("Cox Newton line search failed".to_string());
        }
        if termination_reason.is_some() {
            break;
        }
    }
    let (log_likelihood, _, hess) = cox_ll_grad_hess(&beta, x, start, stop, event, &order, true);
    let diagnostics = FitDiagnostics::new(
        termination_reason.is_some(),
        used,
        termination_reason.unwrap_or_else(|| "Maximum number of iterations reached".to_string()),
        Some(-log_likelihood),
    );
    diagnostics.require_converged(if start.is_some() {
        "AndersenGill"
    } else {
        "CoxPH"
    })?;
    let info = hess.mapv(|v| -v);
    let vcov = invert_matrix(&info)?;
    Ok(SurvivalFit {
        params: beta,
        vcov,
        log_likelihood,
        diagnostics,
    })
}

#[pyclass]
pub struct CoxPH {
    coef: Option<Array1<f64>>,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    diagnostics: Option<FitDiagnostics>,
}

#[pymethods]
impl CoxPH {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            vcov: None,
            log_likelihood: f64::NAN,
            diagnostics: None,
        }
    }

    #[pyo3(signature = (x, time, event, max_iterations=50, tolerance=1e-8))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        self.coef = None;
        self.vcov = None;
        self.diagnostics = None;
        let x = to_array2(&x);
        let time = to_array1(&time);
        let event = to_array1(&event);
        validate_x(&x, time.len())?;
        validate_binary_event(&event, time.len())?;
        validate_positive("time", &time).map_err(PyValueError::new_err)?;
        let fit = py
            .detach(|| fit_cox_core(&x, None, &time, &event, max_iterations, tolerance))
            .map_err(PyValueError::new_err)?;
        self.coef = Some(fit.params);
        self.vcov = Some(fit.vcov);
        self.log_likelihood = fit.log_likelihood;
        self.diagnostics = Some(fit.diagnostics);
        Ok(())
    }

    fn predict_lin<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        Ok(pyarray1_from_f64(py, &x.dot(coef)))
    }

    fn predict_relative_risk<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        Ok(pyarray1_from_f64(py, &x.dot(coef).mapv(f64::exp)))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_relative_risk(py, x)
    }

    fn predict_log_hazard_ratio<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_lin(py, x)
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        survival_summary(
            py,
            "cox_ph",
            self.coef.as_ref(),
            self.vcov.as_ref(),
            self.log_likelihood,
            self.diagnostics.as_ref(),
        )
    }
}

#[pyclass]
pub struct AndersenGill {
    coef: Option<Array1<f64>>,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    diagnostics: Option<FitDiagnostics>,
}

#[pymethods]
impl AndersenGill {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            vcov: None,
            log_likelihood: f64::NAN,
            diagnostics: None,
        }
    }

    #[pyo3(signature = (x, start, stop, event, max_iterations=50, tolerance=1e-8))]
    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        start: PyReadonlyArray1<f64>,
        stop: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        self.coef = None;
        self.vcov = None;
        self.diagnostics = None;
        let x = to_array2(&x);
        let start = to_array1(&start);
        let stop = to_array1(&stop);
        let event = to_array1(&event);
        validate_x(&x, stop.len())?;
        validate_binary_event(&event, stop.len())?;
        if start.len() != stop.len() {
            return Err(PyValueError::new_err("start and stop lengths must match"));
        }
        for i in 0..stop.len() {
            if !start[i].is_finite()
                || !stop[i].is_finite()
                || start[i] < 0.0
                || stop[i] <= start[i]
            {
                return Err(PyValueError::new_err(
                    "intervals must satisfy 0 <= start < stop",
                ));
            }
        }
        let fit = py
            .detach(|| fit_cox_core(&x, Some(&start), &stop, &event, max_iterations, tolerance))
            .map_err(PyValueError::new_err)?;
        self.coef = Some(fit.params);
        self.vcov = Some(fit.vcov);
        self.log_likelihood = fit.log_likelihood;
        self.diagnostics = Some(fit.diagnostics);
        Ok(())
    }

    fn predict_lin<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        Ok(pyarray1_from_f64(py, &x.dot(coef)))
    }

    fn predict_relative_risk<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        Ok(pyarray1_from_f64(py, &x.dot(coef).mapv(f64::exp)))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_relative_risk(py, x)
    }

    fn predict_log_hazard_ratio<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.predict_lin(py, x)
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        survival_summary(
            py,
            "andersen_gill",
            self.coef.as_ref(),
            self.vcov.as_ref(),
            self.log_likelihood,
            self.diagnostics.as_ref(),
        )
    }
}

fn survival_summary<'py>(
    py: Python<'py>,
    model: &str,
    coef: Option<&Array1<f64>>,
    vcov: Option<&Array2<f64>>,
    ll: f64,
    diagnostics: Option<&FitDiagnostics>,
) -> PyResult<Py<PyAny>> {
    let coef = coef.ok_or_else(|| PyValueError::new_err("model is not fitted"))?;
    let vcov = vcov.unwrap();
    let se = Array1::from_iter((0..coef.len()).map(|i| vcov[[i, i]].max(0.0).sqrt()));
    let z = Array1::from_iter((0..coef.len()).map(|i| coef[i] / se[i].max(1e-12)));
    let p = Array1::from_iter(z.iter().map(|v| logistic_survival_p_value(v * v)));
    let dict = PyDict::new(py);
    dict.set_item("model", model)?;
    dict.set_item("coef", pyarray1_from_f64(py, coef))?;
    dict.set_item("hazard_ratio", pyarray1_from_f64(py, &coef.mapv(f64::exp)))?;
    dict.set_item("se", pyarray1_from_f64(py, &se))?;
    dict.set_item("z", pyarray1_from_f64(py, &z))?;
    dict.set_item("p_value", pyarray1_from_f64(py, &p))?;
    dict.set_item("vcov", pyarray2_from_f64(py, vcov))?;
    dict.set_item("log_likelihood", ll)?;
    diagnostics
        .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?
        .write_summary(&dict)?;
    Ok(dict.into())
}
