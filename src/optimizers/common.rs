//! Shared callback validation, scaling, diagnostics and safeguarded line search.
use crate::utils::{pyarray1_from_f64, to_array1, to_array2};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(super) fn error(message: &str) -> PyErr {
    PyValueError::new_err(message.to_string())
}

pub(super) fn norm_inf(x: &Array1<f64>) -> f64 {
    x.iter().fold(0.0_f64, |a, b| a.max(b.abs()))
}

pub(super) fn norm(x: &Array1<f64>) -> f64 {
    x.iter().fold(0.0_f64, |a, b| a.hypot(*b))
}

pub(super) fn vector(value: &Bound<'_, PyAny>, p: usize) -> PyResult<Array1<f64>> {
    let array = value
        .cast::<PyArray1<f64>>()
        .map_err(|_| error("Derivative must return a 1D float64 NumPy array"))?;
    let out = to_array1(&array.readonly());
    if out.len() != p {
        return Err(error("Derivative length must match x0"));
    }
    Ok(out)
}

pub(super) fn matrix(value: &Bound<'_, PyAny>, p: usize) -> PyResult<Array2<f64>> {
    let array = value
        .cast::<PyArray2<f64>>()
        .map_err(|_| error("Curvature must return a 2D float64 NumPy array"))?;
    let out = to_array2(&array.readonly());
    if out.dim() != (p, p) {
        return Err(error(
            "Curvature shape must be (n_parameters, n_parameters)",
        ));
    }
    if out.iter().any(|x| !x.is_finite()) {
        return Err(error("Curvature must be finite"));
    }
    let size = out.iter().fold(1.0_f64, |a, x| a.max(x.abs()));
    if (&out - &out.t()).iter().any(|x| x.abs() > 1e-10 * size) {
        return Err(error("Curvature must be symmetric"));
    }
    Ok((&out + &out.t()) * 0.5)
}

#[derive(Default)]
pub(super) struct Counts {
    pub nfev: usize,
    pub njev: usize,
    pub nhev: usize,
    pub nscorev: usize,
    pub rejected: usize,
    pub resets: usize,
    pub skipped: usize,
}

pub(super) struct Controls {
    pub max_iterations: usize,
    pub gtol: f64,
    pub ftol: f64,
    pub c1: f64,
    pub c2: f64,
    pub max_linesearch: usize,
}

impl Controls {
    pub fn validate(&self) -> PyResult<()> {
        if self.max_iterations == 0 || self.max_linesearch == 0 {
            return Err(error("Iteration and line-search budgets must be positive"));
        }
        if !self.gtol.is_finite() || self.gtol <= 0.0 {
            return Err(error("Gradient tolerance must be positive and finite"));
        }
        if !self.ftol.is_finite() || self.ftol < 0.0 {
            return Err(error("ftol must be nonnegative and finite"));
        }
        if !(0.0 < self.c1 && self.c1 < self.c2 && self.c2 < 1.0) {
            return Err(error("Wolfe constants must satisfy 0 < c1 < c2 < 1"));
        }
        Ok(())
    }
}

pub(super) fn coordinates(
    x0: Array1<f64>,
    scale: Option<Array1<f64>>,
) -> PyResult<(Array1<f64>, Array1<f64>)> {
    if x0.is_empty() || x0.iter().any(|x| !x.is_finite()) {
        return Err(error("x0 must be nonempty and finite"));
    }
    let scale = scale.unwrap_or_else(|| Array1::ones(x0.len()));
    if scale.len() != x0.len() || scale.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(error(
            "x_scale must have x0's length and positive finite entries",
        ));
    }
    let z = &x0 / &scale;
    if z.iter().any(|x| !x.is_finite()) {
        return Err(error("x0 / x_scale must be finite"));
    }
    Ok((z, scale))
}

pub(super) trait Objective {
    fn value(&mut self, z: &Array1<f64>) -> PyResult<f64>;
    fn gradient(&mut self, z: &Array1<f64>) -> PyResult<Array1<f64>>;
    fn scale(&self) -> &Array1<f64>;
    fn counts(&mut self) -> &mut Counts;
    fn invalidate(&mut self) {}
}

pub(super) struct PythonObjective {
    pub fun: Py<PyAny>,
    pub grad: Py<PyAny>,
    pub hess: Option<Py<PyAny>>,
    pub scale: Array1<f64>,
    pub counts: Counts,
}

impl Objective for PythonObjective {
    fn value(&mut self, z: &Array1<f64>) -> PyResult<f64> {
        let x = z * &self.scale;
        if x.iter().any(|v| !v.is_finite()) {
            return Ok(f64::INFINITY);
        }
        self.counts.nfev += 1;
        Python::attach(|py| {
            self.fun
                .call1(py, (pyarray1_from_f64(py, &x),))?
                .extract(py)
        })
    }

    fn gradient(&mut self, z: &Array1<f64>) -> PyResult<Array1<f64>> {
        self.counts.njev += 1;
        Python::attach(|py| {
            let result = self
                .grad
                .call1(py, (pyarray1_from_f64(py, &(z * &self.scale)),))?;
            Ok(vector(result.bind(py), z.len())? * &self.scale)
        })
    }

    fn scale(&self) -> &Array1<f64> {
        &self.scale
    }
    fn counts(&mut self) -> &mut Counts {
        &mut self.counts
    }
}

impl PythonObjective {
    pub fn hessian(&mut self, z: &Array1<f64>) -> PyResult<Array2<f64>> {
        self.counts.nhev += 1;
        Python::attach(|py| {
            let result = self
                .hess
                .as_ref()
                .expect("Hessian configured")
                .call1(py, (pyarray1_from_f64(py, &(z * &self.scale)),))?;
            let mut h = matrix(result.bind(py), z.len())?;
            for i in 0..z.len() {
                for j in 0..z.len() {
                    h[[i, j]] *= self.scale[i] * self.scale[j];
                }
            }
            if h.iter().any(|x| !x.is_finite()) {
                return Err(error("Scaled curvature must be finite"));
            }
            Ok(h)
        })
    }
}

#[derive(Clone)]
pub(super) struct Evaluation {
    pub f: f64,
    pub g: Array1<f64>,
}

pub(super) fn evaluate<P: Objective>(p: &mut P, z: &Array1<f64>) -> PyResult<Option<Evaluation>> {
    let f = p.value(z)?;
    if !f.is_finite() {
        return Ok(None);
    }
    let g = p.gradient(z)?;
    Ok(g.iter()
        .all(|v| v.is_finite())
        .then_some(Evaluation { f, g }))
}

pub(super) fn initial<P: Objective>(p: &mut P, z: &Array1<f64>) -> PyResult<Evaluation> {
    evaluate(p, z)?.ok_or_else(|| error("Objective and gradient must be finite at x0"))
}

/// Bisection zoom is slower than interpolating but never extrapolates into an
/// invalid bracket. Nonfinite *values* are rejected; Python exceptions propagate.
pub(super) fn line_search<P: Objective>(
    problem: &mut P,
    z: &Array1<f64>,
    current: &Evaluation,
    direction: &Array1<f64>,
    controls: &Controls,
) -> PyResult<Option<(Array1<f64>, Evaluation)>> {
    let slope = current.g.dot(direction);
    if !slope.is_finite() || slope >= 0.0 {
        return Ok(None);
    }
    let mut alpha = 1.0;
    let mut previous_alpha = 0.0;
    let mut previous_f = current.f;
    let mut bracket = None;
    let mut used = 0;
    while used < controls.max_linesearch {
        used += 1;
        let candidate = z + &(direction * alpha);
        let value = evaluate(problem, &candidate)?;
        match value {
            Some(ref next)
                if next.f <= current.f + controls.c1 * alpha * slope
                    && (used == 1 || next.f <= previous_f) =>
            {
                let next_slope = next.g.dot(direction);
                if next_slope.is_finite() && next_slope.abs() <= -controls.c2 * slope {
                    return Ok(Some((candidate, next.clone())));
                }
                if !next_slope.is_finite() || next_slope >= 0.0 {
                    problem.counts().rejected += 1;
                    bracket = Some((alpha, previous_alpha, next.f));
                    break;
                }
                previous_alpha = alpha;
                previous_f = next.f;
                alpha *= 2.0;
            }
            _ => {
                problem.counts().rejected += 1;
                bracket = Some((previous_alpha, alpha, previous_f));
                break;
            }
        }
        problem.counts().rejected += 1;
    }
    if let Some((mut lo, mut hi, mut flo)) = bracket {
        while used < controls.max_linesearch {
            used += 1;
            alpha = lo + 0.5 * (hi - lo);
            if alpha == lo || alpha == hi {
                break;
            }
            let candidate = z + &(direction * alpha);
            if candidate == *z {
                break;
            }
            match evaluate(problem, &candidate)? {
                Some(next)
                    if next.f <= current.f + controls.c1 * alpha * slope && next.f <= flo =>
                {
                    let next_slope = next.g.dot(direction);
                    if next_slope.is_finite() && next_slope.abs() <= -controls.c2 * slope {
                        return Ok(Some((candidate, next)));
                    }
                    if !next_slope.is_finite() || next_slope * (hi - lo) >= 0.0 {
                        hi = lo;
                    }
                    lo = alpha;
                    flo = next.f;
                }
                _ => hi = alpha,
            }
            problem.counts().rejected += 1;
        }
    }
    Ok(None)
}

pub(super) fn finish<P: Objective>(
    py: Python<'_>,
    problem: &mut P,
    z: &Array1<f64>,
    iterations: usize,
    status: &str,
    method: &str,
    controls: &Controls,
) -> PyResult<Py<PyAny>> {
    // Re-score the actual returned point rather than trusting a cost-change or
    // budget termination flag. These callback evaluations are included in counts.
    problem.invalidate();
    let final_eval = initial(problem, z)?;
    let scale = problem.scale();
    let raw_gradient = &final_eval.g / scale;
    let optimality = norm_inf(&final_eval.g);
    let success = optimality <= controls.gtol;
    let dict = PyDict::new(py);
    dict.set_item("x", pyarray1_from_f64(py, &(z * scale)))?;
    dict.set_item("fun", final_eval.f)?;
    dict.set_item("jac", pyarray1_from_f64(py, &raw_gradient))?;
    dict.set_item("optimality", optimality)?;
    dict.set_item("grad_norm", norm_inf(&raw_gradient))?;
    dict.set_item("nit", iterations)?;
    dict.set_item("success", success)?;
    dict.set_item(
        "status",
        if success {
            "gradient_tolerance"
        } else {
            status
        },
    )?;
    dict.set_item(
        "message",
        if success {
            "Final gradient satisfies gtol"
        } else {
            status
        },
    )?;
    dict.set_item("method", method)?;
    let counts = problem.counts();
    for (key, value) in [
        ("nfev", counts.nfev),
        ("njev", counts.njev),
        ("nhev", counts.nhev),
        ("nscorev", counts.nscorev),
        ("rejected_steps", counts.rejected),
        ("resets", counts.resets),
        ("skipped_updates", counts.skipped),
    ] {
        dict.set_item(key, value)?;
    }
    Ok(dict.into())
}
