use crate::utils::{invert_matrix, pyarray1_from_f64, pyarray2_from_f64, to_array1, to_array2};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn validate_binary_event(event: &Array1<f64>, n: usize) -> PyResult<()> {
    if event.len() != n {
        return Err(PyValueError::new_err("event length must match time length"));
    }
    if event
        .iter()
        .any(|v| !v.is_finite() || (*v != 0.0 && *v != 1.0))
    {
        return Err(PyValueError::new_err("event indicators must be 0/1"));
    }
    Ok(())
}

fn validate_x(x: &Array2<f64>, n: usize) -> PyResult<()> {
    if x.nrows() != n {
        return Err(PyValueError::new_err("x rows must match time length"));
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err("x must contain only finite values"));
    }
    Ok(())
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

fn fit_parametric(
    x: &Array2<f64>,
    time: &Array1<f64>,
    event: &Array1<f64>,
    weibull: bool,
    max_iter: usize,
    tol: f64,
) -> Result<(Array1<f64>, Array2<f64>, f64, usize), String> {
    let p = x.ncols();
    let k = 1 + p + if weibull { 1 } else { 0 };
    let mut theta = Array1::<f64>::zeros(k);
    let events = event.sum().max(1.0);
    let total_time = time.sum().max(1e-12);
    theta[0] = (events / total_time).ln();
    if weibull {
        theta[k - 1] = 0.0;
    }
    let mut last_ll = f64::NEG_INFINITY;
    let mut used = 0;
    for iter in 0..max_iter {
        used = iter + 1;
        let (ll, grad, hess) = parametric_ll_grad_hess(&theta, x, time, event, weibull);
        if max_abs(&grad) < tol {
            last_ll = ll;
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
                last_ll = cand_ll;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return Err("Newton line search failed".to_string());
        }
        if (last_ll - ll).abs() < tol * (1.0 + ll.abs()) {
            break;
        }
    }
    let (_, _, hess) = parametric_ll_grad_hess(&theta, x, time, event, weibull);
    let info = hess.mapv(|v| -v);
    let vcov = invert_matrix(&info)?;
    Ok((theta, vcov, last_ll, used))
}

#[pyclass]
pub struct ExponentialPH {
    coef: Option<Array1<f64>>,
    log_baseline_hazard: f64,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    iterations: usize,
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
            iterations: 0,
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
        let x = to_array2(&x);
        let time = to_array1(&time);
        let event = to_array1(&event);
        validate_x(&x, time.len())?;
        validate_binary_event(&event, time.len())?;
        if time.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(PyValueError::new_err("time must be positive and finite"));
        }
        let (theta, vcov, ll, it) =
            fit_parametric(&x, &time, &event, false, max_iterations, tolerance)
                .map_err(PyValueError::new_err)?;
        self.log_baseline_hazard = theta[0];
        self.coef = Some(theta.slice(ndarray::s![1..]).to_owned());
        self.vcov = Some(vcov);
        self.log_likelihood = ll;
        self.iterations = it;
        Ok(())
    }

    fn predict_log_hazard<'py>(
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

    fn survival<'py>(
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
        dict.set_item("iterations", self.iterations)?;
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
    iterations: usize,
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
            iterations: 0,
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
        let x = to_array2(&x);
        let time = to_array1(&time);
        let event = to_array1(&event);
        validate_x(&x, time.len())?;
        validate_binary_event(&event, time.len())?;
        if time.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(PyValueError::new_err("time must be positive and finite"));
        }
        let (theta, vcov, ll, it) =
            fit_parametric(&x, &time, &event, true, max_iterations, tolerance)
                .map_err(PyValueError::new_err)?;
        let p = x.ncols();
        self.log_scale_hazard = theta[0];
        self.coef = Some(theta.slice(ndarray::s![1..1 + p]).to_owned());
        self.log_shape = theta[1 + p];
        self.vcov = Some(vcov);
        self.log_likelihood = ll;
        self.iterations = it;
        Ok(())
    }

    fn survival<'py>(
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

    fn predict_log_hazard<'py>(
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
        dict.set_item("iterations", self.iterations)?;
        Ok(dict.into())
    }
}

fn cox_ll_grad_hess(
    beta: &Array1<f64>,
    x: &Array2<f64>,
    start: Option<&Array1<f64>>,
    stop: &Array1<f64>,
    event: &Array1<f64>,
) -> (f64, Array1<f64>, Array2<f64>) {
    let n = stop.len();
    let p = x.ncols();
    let eta = x.dot(beta);
    let risk_score = eta.mapv(|v| v.clamp(-40.0, 40.0).exp());
    let mut ll = 0.0;
    let mut grad = Array1::<f64>::zeros(p);
    let mut hess = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        if event[i] != 1.0 {
            continue;
        }
        let t = stop[i];
        let mut denom = 0.0;
        let mut xbar_num = Array1::<f64>::zeros(p);
        let mut xx_num = Array2::<f64>::zeros((p, p));
        for r in 0..n {
            let enters = start.map(|s| s[r] < t).unwrap_or(true);
            if enters && stop[r] >= t {
                let w = risk_score[r];
                denom += w;
                for a in 0..p {
                    xbar_num[a] += w * x[[r, a]];
                }
                for a in 0..p {
                    for b in 0..p {
                        xx_num[[a, b]] += w * x[[r, a]] * x[[r, b]];
                    }
                }
            }
        }
        if denom <= 0.0 {
            continue;
        }
        let xbar = xbar_num.mapv(|v| v / denom);
        ll += eta[i] - denom.ln();
        for a in 0..p {
            grad[a] += x[[i, a]] - xbar[a];
        }
        for a in 0..p {
            for b in 0..p {
                hess[[a, b]] -= xx_num[[a, b]] / denom - xbar[a] * xbar[b];
            }
        }
    }
    (ll, grad, hess)
}

fn fit_cox_core(
    x: &Array2<f64>,
    start: Option<&Array1<f64>>,
    stop: &Array1<f64>,
    event: &Array1<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<(Array1<f64>, Array2<f64>, f64, usize), String> {
    let p = x.ncols();
    let mut beta = Array1::<f64>::zeros(p);
    let mut last_ll = f64::NEG_INFINITY;
    let mut used = 0;
    for iter in 0..max_iter {
        used = iter + 1;
        let (ll, grad, hess) = cox_ll_grad_hess(&beta, x, start, stop, event);
        if max_abs(&grad) < tol {
            last_ll = ll;
            break;
        }
        let mut step = solve_newton(&hess, &grad, 1e-8)?;
        step.mapv_inplace(|v| v.clamp(-1.0, 1.0));
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..30 {
            let cand = &beta - &(step.mapv(|v| scale * v));
            let (cand_ll, _, _) = cox_ll_grad_hess(&cand, x, start, stop, event);
            if cand_ll.is_finite() && cand_ll >= ll - 1e-10 {
                beta = cand;
                last_ll = cand_ll;
                accepted = true;
                break;
            }
            scale *= 0.5;
        }
        if !accepted {
            return Err("Cox Newton line search failed".to_string());
        }
        if (last_ll - ll).abs() < tol * (1.0 + ll.abs()) {
            break;
        }
    }
    let (_, _, hess) = cox_ll_grad_hess(&beta, x, start, stop, event);
    let info = hess.mapv(|v| -v);
    let vcov = invert_matrix(&info)?;
    Ok((beta, vcov, last_ll, used))
}

#[pyclass]
pub struct CoxPH {
    coef: Option<Array1<f64>>,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    iterations: usize,
}

#[pymethods]
impl CoxPH {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            vcov: None,
            log_likelihood: f64::NAN,
            iterations: 0,
        }
    }

    #[pyo3(signature = (x, time, event, max_iterations=50, tolerance=1e-8))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        time: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
        let x = to_array2(&x);
        let time = to_array1(&time);
        let event = to_array1(&event);
        validate_x(&x, time.len())?;
        validate_binary_event(&event, time.len())?;
        if time.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(PyValueError::new_err("time must be positive and finite"));
        }
        let (coef, vcov, ll, it) = fit_cox_core(&x, None, &time, &event, max_iterations, tolerance)
            .map_err(PyValueError::new_err)?;
        self.coef = Some(coef);
        self.vcov = Some(vcov);
        self.log_likelihood = ll;
        self.iterations = it;
        Ok(())
    }

    fn predict_log_hazard_ratio<'py>(
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

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        survival_summary(
            py,
            "cox_ph",
            self.coef.as_ref(),
            self.vcov.as_ref(),
            self.log_likelihood,
            self.iterations,
        )
    }
}

#[pyclass]
pub struct AndersenGill {
    coef: Option<Array1<f64>>,
    vcov: Option<Array2<f64>>,
    log_likelihood: f64,
    iterations: usize,
}

#[pymethods]
impl AndersenGill {
    #[new]
    fn new() -> Self {
        Self {
            coef: None,
            vcov: None,
            log_likelihood: f64::NAN,
            iterations: 0,
        }
    }

    #[pyo3(signature = (x, start, stop, event, max_iterations=50, tolerance=1e-8))]
    fn fit(
        &mut self,
        x: PyReadonlyArray2<f64>,
        start: PyReadonlyArray1<f64>,
        stop: PyReadonlyArray1<f64>,
        event: PyReadonlyArray1<f64>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<()> {
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
        let (coef, vcov, ll, it) =
            fit_cox_core(&x, Some(&start), &stop, &event, max_iterations, tolerance)
                .map_err(PyValueError::new_err)?;
        self.coef = Some(coef);
        self.vcov = Some(vcov);
        self.log_likelihood = ll;
        self.iterations = it;
        Ok(())
    }

    fn predict_log_hazard_ratio<'py>(
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

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        survival_summary(
            py,
            "andersen_gill",
            self.coef.as_ref(),
            self.vcov.as_ref(),
            self.log_likelihood,
            self.iterations,
        )
    }
}

fn survival_summary<'py>(
    py: Python<'py>,
    model: &str,
    coef: Option<&Array1<f64>>,
    vcov: Option<&Array2<f64>>,
    ll: f64,
    iterations: usize,
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
    dict.set_item("iterations", iterations)?;
    Ok(dict.into())
}
