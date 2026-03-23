use crate::utils::{
    diag_sqrt, invert_matrix, pyarray1_from_f64, pyarray2_from_f64, to_array1, to_array1_i64,
    to_array2,
};
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

fn identity_matrix(n: usize) -> Array2<f64> {
    let mut eye = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        eye[[i, i]] = 1.0;
    }
    eye
}

fn with_diagonal_ridge(a: &Array2<f64>, ridge: f64) -> Array2<f64> {
    let mut out = a.clone();
    let dim = out.nrows().min(out.ncols());
    for i in 0..dim {
        out[[i, i]] += ridge;
    }
    out
}

fn invert_with_ridge(a: &Array2<f64>, ridge: f64) -> Result<Array2<f64>, String> {
    invert_matrix(&with_diagonal_ridge(a, ridge))
}

fn sample_mean_moments(moments: &Array2<f64>) -> Result<Array1<f64>, String> {
    moments
        .mean_axis(Axis(0))
        .ok_or_else(|| "moment function must return a non-empty 2D array".to_string())
}

fn default_newey_west_lags(n: usize) -> usize {
    ((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize).max(1)
}

fn call_moments(
    py: Python,
    moment_fn: &Py<PyAny>,
    theta: &Array1<f64>,
    data: &Py<PyAny>,
) -> PyResult<Array2<f64>> {
    let theta_py = pyarray1_from_f64(py, theta);
    let result = moment_fn
        .call1(py, (theta_py, data.clone_ref(py)))
        .map_err(|e| PyValueError::new_err(format!("moment_fn error: {}", e)))?;

    let moments_py = result
        .cast_bound::<PyArray2<f64>>(py)
        .map_err(|_| PyValueError::new_err("moment_fn must return a 2D numpy array"))?;
    let moments = to_array2(&moments_py.readonly());

    if moments.nrows() == 0 || moments.ncols() == 0 {
        return Err(PyValueError::new_err(
            "moment_fn must return a non-empty (n_obs, n_moments) array",
        ));
    }

    Ok(moments)
}

fn call_jacobian_callback(
    py: Python,
    jacobian_fn: &Py<PyAny>,
    theta: &Array1<f64>,
    data: &Py<PyAny>,
) -> PyResult<Array2<f64>> {
    let theta_py = pyarray1_from_f64(py, theta);
    let result = jacobian_fn
        .call1(py, (theta_py, data.clone_ref(py)))
        .map_err(|e| PyValueError::new_err(format!("jacobian_fn error: {}", e)))?;

    let jacobian_py = result
        .cast_bound::<PyArray2<f64>>(py)
        .map_err(|_| PyValueError::new_err("jacobian_fn must return a 2D numpy array"))?;
    Ok(to_array2(&jacobian_py.readonly()))
}

fn numerical_jacobian(
    py: Python,
    moment_fn: &Py<PyAny>,
    theta: &Array1<f64>,
    data: &Py<PyAny>,
    fd_eps: f64,
) -> PyResult<Array2<f64>> {
    let base_moments = call_moments(py, moment_fn, theta, data)?;
    let base_mean = sample_mean_moments(&base_moments).map_err(PyValueError::new_err)?;
    let m = base_mean.len();
    let p = theta.len();
    let mut jacobian = Array2::<f64>::zeros((m, p));

    for j in 0..p {
        let h = fd_eps * theta[j].abs().max(1.0);
        let mut theta_hi = theta.clone();
        let mut theta_lo = theta.clone();
        theta_hi[j] += h;
        theta_lo[j] -= h;

        let g_hi = sample_mean_moments(&call_moments(py, moment_fn, &theta_hi, data)?)
            .map_err(PyValueError::new_err)?;
        let g_lo = sample_mean_moments(&call_moments(py, moment_fn, &theta_lo, data)?)
            .map_err(PyValueError::new_err)?;
        let diff = (&g_hi - &g_lo) / (2.0 * h);
        jacobian.column_mut(j).assign(&diff);
    }

    Ok(jacobian)
}

fn sample_jacobian(
    py: Python,
    moment_fn: &Py<PyAny>,
    jacobian_fn: Option<&Py<PyAny>>,
    theta: &Array1<f64>,
    data: &Py<PyAny>,
    fd_eps: f64,
) -> PyResult<Array2<f64>> {
    match jacobian_fn {
        Some(jacobian_fn) => call_jacobian_callback(py, jacobian_fn, theta, data),
        None => numerical_jacobian(py, moment_fn, theta, data, fd_eps),
    }
}

fn omega_iid(moments: &Array2<f64>) -> Array2<f64> {
    moments.t().dot(moments) / (moments.nrows() as f64)
}

fn omega_newey_west(moments: &Array2<f64>, lags: usize) -> Array2<f64> {
    let n = moments.nrows();
    let mut omega = omega_iid(moments);
    if n <= 1 || lags == 0 {
        return omega;
    }

    let max_lag = lags.min(n - 1);
    for lag in 1..=max_lag {
        let weight = 1.0 - lag as f64 / (max_lag as f64 + 1.0);
        let lead = moments.slice(s![lag.., ..]).to_owned();
        let lagged = moments.slice(s![..(n - lag), ..]).to_owned();
        let gamma = lead.t().dot(&lagged) / (n as f64);
        omega = omega + weight * (&gamma + &gamma.t().to_owned());
    }

    omega
}

fn omega_cluster(moments: &Array2<f64>, clusters: &Array1<i64>) -> Result<Array2<f64>, String> {
    let n = moments.nrows();
    let m = moments.ncols();
    if clusters.len() != n {
        return Err("clusters length must match the number of observations".to_string());
    }

    let mut grouped: BTreeMap<i64, Array1<f64>> = BTreeMap::new();
    for i in 0..n {
        let entry = grouped
            .entry(clusters[i])
            .or_insert_with(|| Array1::<f64>::zeros(m));
        *entry = &*entry + &moments.row(i).to_owned();
    }

    let mut omega = Array2::<f64>::zeros((m, m));
    for summed in grouped.values() {
        let col = summed.clone().insert_axis(Axis(1));
        let row = summed.clone().insert_axis(Axis(0));
        omega = omega + col.dot(&row);
    }

    Ok(omega / (n as f64))
}

fn criterion_value(gbar: &Array1<f64>, weight: &Array2<f64>) -> f64 {
    0.5 * gbar.dot(&weight.dot(gbar))
}

struct FitResult {
    theta: Array1<f64>,
    criterion: f64,
    nit: usize,
}

fn solve_gauss_newton(
    py: Python,
    moment_fn: &Py<PyAny>,
    jacobian_fn: Option<&Py<PyAny>>,
    data: &Py<PyAny>,
    theta0: &Array1<f64>,
    weight: &Array2<f64>,
    max_iterations: usize,
    tolerance: f64,
    ridge: f64,
    fd_eps: f64,
) -> PyResult<FitResult> {
    let mut theta = theta0.clone();
    let mut iter = 0usize;

    loop {
        let moments = call_moments(py, moment_fn, &theta, data)?;
        let gbar = sample_mean_moments(&moments).map_err(PyValueError::new_err)?;
        let jacobian = sample_jacobian(py, moment_fn, jacobian_fn, &theta, data, fd_eps)?;

        if jacobian.nrows() != gbar.len() || jacobian.ncols() != theta.len() {
            return Err(PyValueError::new_err(format!(
                "jacobian_fn returned shape ({}, {}), expected ({}, {})",
                jacobian.nrows(),
                jacobian.ncols(),
                gbar.len(),
                theta.len()
            )));
        }

        let current_criterion = criterion_value(&gbar, weight);
        let wg = weight.dot(&gbar);
        let normal = jacobian.t().dot(weight).dot(&jacobian);
        let rhs = jacobian.t().dot(&wg);
        let step = invert_with_ridge(&normal, ridge)
            .map_err(PyValueError::new_err)?
            .dot(&rhs);

        if step.dot(&step).sqrt() < tolerance {
            return Ok(FitResult {
                theta,
                criterion: current_criterion,
                nit: iter,
            });
        }

        let mut alpha = 1.0;
        let mut accepted_theta = None;
        let mut accepted_criterion = current_criterion;

        while alpha >= 1e-8 {
            let candidate = &theta - &(step.mapv(|v| alpha * v));
            let candidate_moments = call_moments(py, moment_fn, &candidate, data)?;
            let candidate_gbar =
                sample_mean_moments(&candidate_moments).map_err(PyValueError::new_err)?;
            let candidate_criterion = criterion_value(&candidate_gbar, weight);

            if candidate_criterion < current_criterion {
                accepted_theta = Some(candidate);
                accepted_criterion = candidate_criterion;
                break;
            }

            alpha *= 0.5;
        }

        let candidate = accepted_theta.ok_or_else(|| {
            PyValueError::new_err("Gauss-Newton line search failed to find a descent step")
        })?;

        iter += 1;
        theta = candidate;

        if (current_criterion - accepted_criterion).abs() < tolerance || iter >= max_iterations {
            return Ok(FitResult {
                theta,
                criterion: accepted_criterion,
                nit: iter,
            });
        }
    }
}

#[pyclass]
pub struct GMM {
    max_iterations: usize,
    tolerance: f64,
    ridge: f64,
    fd_eps: f64,
    moment_fn: Py<PyAny>,
    jacobian_fn: Option<Py<PyAny>>,
    theta: Option<Array1<f64>>,
    data: Option<Py<PyAny>>,
    weight_matrix: Option<Array2<f64>>,
    first_step_theta: Option<Array1<f64>>,
    criterion: Option<f64>,
    nit: Option<usize>,
    weighting: Option<String>,
    n_obs: Option<usize>,
    n_moments: Option<usize>,
}

#[pymethods]
impl GMM {
    #[new]
    #[pyo3(signature = (moment_fn, jacobian_fn=None, max_iterations=100, tolerance=1e-6, ridge=1e-8, fd_eps=1e-6))]
    fn new(
        moment_fn: Py<PyAny>,
        jacobian_fn: Option<Py<PyAny>>,
        max_iterations: usize,
        tolerance: f64,
        ridge: f64,
        fd_eps: f64,
    ) -> PyResult<Self> {
        if max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if tolerance <= 0.0 {
            return Err(PyValueError::new_err("tolerance must be positive"));
        }
        if ridge < 0.0 {
            return Err(PyValueError::new_err("ridge must be nonnegative"));
        }
        if fd_eps <= 0.0 {
            return Err(PyValueError::new_err("fd_eps must be positive"));
        }

        Ok(Self {
            max_iterations,
            tolerance,
            ridge,
            fd_eps,
            moment_fn,
            jacobian_fn,
            theta: None,
            data: None,
            weight_matrix: None,
            first_step_theta: None,
            criterion: None,
            nit: None,
            weighting: None,
            n_obs: None,
            n_moments: None,
        })
    }

    #[pyo3(signature = (data, theta0, weighting="auto"))]
    fn fit(
        &mut self,
        py: Python,
        data: Py<PyAny>,
        theta0: PyReadonlyArray1<f64>,
        weighting: &str,
    ) -> PyResult<()> {
        let theta0 = to_array1(&theta0);
        let initial_moments = call_moments(py, &self.moment_fn, &theta0, &data)?;
        let n = initial_moments.nrows();
        let m = initial_moments.ncols();
        let p = theta0.len();

        if m < p {
            return Err(PyValueError::new_err(format!(
                "model is underidentified: {} moments for {} parameters",
                m, p
            )));
        }

        let chosen_weighting = match weighting {
            "auto" => {
                if m == p {
                    "identity".to_string()
                } else {
                    "two_step".to_string()
                }
            }
            "identity" => "identity".to_string(),
            "two_step" => {
                if m == p {
                    "identity".to_string()
                } else {
                    "two_step".to_string()
                }
            }
            _ => {
                return Err(PyValueError::new_err(
                    "weighting must be one of {'auto', 'identity', 'two_step'}",
                ));
            }
        };

        let identity = identity_matrix(m);
        let first_step = solve_gauss_newton(
            py,
            &self.moment_fn,
            self.jacobian_fn.as_ref(),
            &data,
            &theta0,
            &identity,
            self.max_iterations,
            self.tolerance,
            self.ridge,
            self.fd_eps,
        )?;

        let (theta, criterion, nit, weight_matrix, first_step_theta) = if chosen_weighting == "two_step"
        {
            let first_moments = call_moments(py, &self.moment_fn, &first_step.theta, &data)?;
            let omega = omega_iid(&first_moments);
            let weight_matrix = invert_with_ridge(&omega, self.ridge).map_err(PyValueError::new_err)?;
            let second_step = solve_gauss_newton(
                py,
                &self.moment_fn,
                self.jacobian_fn.as_ref(),
                &data,
                &first_step.theta,
                &weight_matrix,
                self.max_iterations,
                self.tolerance,
                self.ridge,
                self.fd_eps,
            )?;
            (
                second_step.theta,
                second_step.criterion,
                first_step.nit + second_step.nit,
                weight_matrix,
                Some(first_step.theta),
            )
        } else {
            (
                first_step.theta,
                first_step.criterion,
                first_step.nit,
                identity,
                None,
            )
        };

        self.theta = Some(theta);
        self.data = Some(data);
        self.weight_matrix = Some(weight_matrix);
        self.first_step_theta = first_step_theta;
        self.criterion = Some(criterion);
        self.nit = Some(nit);
        self.weighting = Some(chosen_weighting);
        self.n_obs = Some(n);
        self.n_moments = Some(m);
        Ok(())
    }

    #[pyo3(signature = (vcov="sandwich", omega="iid", lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: &str,
        omega: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("GMM model is not fitted"))?;
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let weight_matrix = self
            .weight_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted weight matrix stored"))?;

        let moments = call_moments(py, &self.moment_fn, theta, data)?;
        let gbar = sample_mean_moments(&moments).map_err(PyValueError::new_err)?;
        let jacobian =
            sample_jacobian(py, &self.moment_fn, self.jacobian_fn.as_ref(), theta, data, self.fd_eps)?;

        if jacobian.nrows() != moments.ncols() || jacobian.ncols() != theta.len() {
            return Err(PyValueError::new_err(format!(
                "jacobian_fn returned shape ({}, {}), expected ({}, {})",
                jacobian.nrows(),
                jacobian.ncols(),
                moments.ncols(),
                theta.len()
            )));
        }

        let a_matrix = jacobian.t().dot(weight_matrix).dot(&jacobian);
        let a_inv = invert_with_ridge(&a_matrix, self.ridge).map_err(PyValueError::new_err)?;
        let n = moments.nrows();

        let covariance = match vcov {
            "vanilla" => a_inv.mapv(|v| v / (n as f64)),
            "sandwich" => {
                let omega_hat = match omega {
                    "iid" => omega_iid(&moments),
                    "newey_west" => omega_newey_west(&moments, lags.unwrap_or_else(|| default_newey_west_lags(n))),
                    "cluster" => {
                        let clusters = clusters
                            .ok_or_else(|| PyValueError::new_err("clusters must be provided for omega='cluster'"))?;
                        let cluster_ids = to_array1_i64(&clusters);
                        omega_cluster(&moments, &cluster_ids).map_err(PyValueError::new_err)?
                    }
                    _ => {
                        return Err(PyValueError::new_err(
                            "omega must be one of {'iid', 'newey_west', 'cluster'}",
                        ));
                    }
                };
                let middle = jacobian
                    .t()
                    .dot(weight_matrix)
                    .dot(&omega_hat)
                    .dot(weight_matrix)
                    .dot(&jacobian);
                a_inv.dot(&middle).dot(&a_inv).mapv(|v| v / (n as f64))
            }
            _ => {
                return Err(PyValueError::new_err(
                    "vcov must be one of {'vanilla', 'sandwich'}",
                ));
            }
        };

        let se = diag_sqrt(&covariance);
        let j_df = if moments.ncols() > theta.len() {
            Some(moments.ncols() - theta.len())
        } else {
            None
        };
        let j_stat = j_df.map(|_| (n as f64) * 2.0 * criterion_value(&gbar, weight_matrix));

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, theta))?;
        dict.set_item("se", pyarray1_from_f64(py, &se))?;
        dict.set_item("vcov", pyarray2_from_f64(py, &covariance))?;
        dict.set_item("criterion", self.criterion)?;
        dict.set_item("nit", self.nit)?;
        dict.set_item("weighting", self.weighting.clone())?;
        dict.set_item("vcov_type", vcov)?;
        dict.set_item("omega_type", if vcov == "sandwich" { Some(omega) } else { None::<&str> })?;
        dict.set_item("weight_matrix", pyarray2_from_f64(py, weight_matrix))?;
        dict.set_item("nobs", n)?;
        dict.set_item("n_moments", moments.ncols())?;
        dict.set_item("j_stat", j_stat)?;
        dict.set_item("j_df", j_df)?;
        Ok(dict.into())
    }
}
