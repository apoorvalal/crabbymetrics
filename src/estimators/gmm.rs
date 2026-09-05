use crate::hyptests::wald_test_arrays;
use crate::utils::{
    default_newey_west_lags, diag_sqrt, invert_matrix, pyarray1_from_f64, pyarray2_from_f64,
    score_cov_cluster, score_cov_iid, score_cov_newey_west, to_array1, to_array1_i64, to_array2,
};
use crate::validation::validate_finite;
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    validate_finite("information matrix", a)?;
    let scale = a.diag().mapv(f64::sqrt);
    if scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("information matrix is rank deficient".to_string());
    }
    let normalized = Array2::from_shape_fn(a.raw_dim(), |(i, j)| a[[i, j]] / scale[i] / scale[j]);
    let dim = a.nrows();
    let eigen = nalgebra::DMatrix::from_row_iterator(dim, dim, normalized.iter().copied())
        .symmetric_eigen();
    if eigen
        .eigenvalues
        .iter()
        .any(|v| *v <= f64::EPSILON * dim as f64 * 10.0)
    {
        return Err("information matrix is rank deficient".to_string());
    }
    let inverse = invert_matrix(&with_diagonal_ridge(&normalized, ridge))?;
    Ok(Array2::from_shape_fn(a.raw_dim(), |(i, j)| {
        inverse[[i, j]] / scale[i] / scale[j]
    }))
}

fn sample_mean_moments(moments: &Array2<f64>) -> Result<Array1<f64>, String> {
    moments
        .mean_axis(Axis(0))
        .ok_or_else(|| "moment function must return a non-empty 2D array".to_string())
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
    validate_finite("moments", &moments).map_err(PyValueError::new_err)?;

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
    let jacobian = to_array2(&jacobian_py.readonly());
    validate_finite("jacobian", &jacobian).map_err(PyValueError::new_err)?;
    Ok(jacobian)
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

        let moments_hi = call_moments(py, moment_fn, &theta_hi, data)?;
        let moments_lo = call_moments(py, moment_fn, &theta_lo, data)?;
        if moments_hi.dim() != base_moments.dim() || moments_lo.dim() != base_moments.dim() {
            return Err(PyValueError::new_err(
                "moment shape changed during differentiation",
            ));
        }
        let g_hi = sample_mean_moments(&moments_hi).map_err(PyValueError::new_err)?;
        let g_lo = sample_mean_moments(&moments_lo).map_err(PyValueError::new_err)?;
        let diff = (&g_hi - &g_lo) / (2.0 * h);
        jacobian.column_mut(j).assign(&diff);
    }

    validate_finite("numerical jacobian", &jacobian).map_err(PyValueError::new_err)?;
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
    score_cov_iid(moments) / (moments.nrows() as f64)
}

fn omega_newey_west(moments: &Array2<f64>, lags: usize) -> Array2<f64> {
    score_cov_newey_west(moments, lags) / (moments.nrows() as f64)
}

fn omega_cluster(moments: &Array2<f64>, clusters: &Array1<i64>) -> Result<Array2<f64>, String> {
    let (omega, _) = score_cov_cluster(moments, clusters)?;
    Ok(omega / (moments.nrows() as f64))
}

fn criterion_value(gbar: &Array1<f64>, weight: &Array2<f64>) -> f64 {
    0.5 * gbar.dot(&weight.dot(gbar))
}

fn rademacher_moment_projection(
    n_moments: usize,
    sketch_size: usize,
    seed: Option<u64>,
) -> PyResult<Array2<f64>> {
    if sketch_size == 0 {
        return Err(PyValueError::new_err("sketch_size must be positive"));
    }
    if sketch_size > n_moments {
        return Err(PyValueError::new_err(
            "sketch_size must be <= the number of moments",
        ));
    }
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0x6A44_5EED));
    let scale = 1.0 / (sketch_size as f64).sqrt();
    Ok(Array2::from_shape_fn((n_moments, sketch_size), |_| {
        if rng.gen_bool(0.5) {
            scale
        } else {
            -scale
        }
    }))
}

fn project_moments(
    moments: Array2<f64>,
    projection: Option<&Array2<f64>>,
) -> PyResult<Array2<f64>> {
    match projection {
        Some(projection) => {
            if moments.ncols() != projection.nrows() {
                return Err(PyValueError::new_err(
                    "moment projection row count must match moment count",
                ));
            }
            Ok(moments.dot(projection))
        }
        None => Ok(moments),
    }
}

fn project_jacobian(
    jacobian: Array2<f64>,
    projection: Option<&Array2<f64>>,
) -> PyResult<Array2<f64>> {
    match projection {
        Some(projection) => {
            if jacobian.nrows() != projection.nrows() {
                return Err(PyValueError::new_err(
                    "moment projection row count must match jacobian row count",
                ));
            }
            Ok(projection.t().dot(&jacobian))
        }
        None => Ok(jacobian),
    }
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
    projection: Option<&Array2<f64>>,
    max_iterations: usize,
    tolerance: f64,
    ridge: f64,
    fd_eps: f64,
) -> PyResult<FitResult> {
    let mut theta = theta0.clone();
    let mut iter = 0usize;
    let mut expected_shape = None;

    loop {
        let moments = project_moments(call_moments(py, moment_fn, &theta, data)?, projection)?;
        if expected_shape.is_some_and(|shape| shape != moments.dim()) {
            return Err(PyValueError::new_err("moment shape changed during fitting"));
        }
        expected_shape = Some(moments.dim());
        let gbar = sample_mean_moments(&moments).map_err(PyValueError::new_err)?;
        if gbar.len() != weight.nrows() {
            return Err(PyValueError::new_err("moment count changed during fitting"));
        }
        let jacobian = project_jacobian(
            sample_jacobian(py, moment_fn, jacobian_fn, &theta, data, fd_eps)?,
            projection,
        )?;

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
        // Check undamped estimating equations; damping must not certify convergence.
        let newton_step = invert_with_ridge(&normal, 0.0)
            .map_err(PyValueError::new_err)?
            .dot(&rhs);
        let moment_scale = (moments.dot(weight) * &moments).sum() / moments.nrows() as f64;
        let score = rhs
            .iter()
            .enumerate()
            .map(|(j, &v)| v * v / normal[[j, j]])
            .sum::<f64>()
            .sqrt()
            / moment_scale.max(f64::MIN_POSITIVE).sqrt();
        let relative_step = newton_step
            .iter()
            .zip(theta.iter())
            .map(|(s, t)| s.abs() / (1.0 + t.abs()))
            .fold(0.0_f64, f64::max);
        let step = invert_with_ridge(&normal, ridge)
            .map_err(PyValueError::new_err)?
            .dot(&rhs);

        if score <= tolerance && relative_step <= tolerance {
            return Ok(FitResult {
                theta,
                criterion: current_criterion,
                nit: iter,
            });
        }
        if iter >= max_iterations {
            return Err(PyValueError::new_err(format!("GMM optimization did not converge within {max_iterations} iterations (scaled score {score:.3e})")));
        }

        let mut alpha = 1.0;
        let mut accepted_theta = None;

        while alpha >= 1e-8 {
            let candidate = &theta - &(step.mapv(|v| alpha * v));
            let candidate_moments =
                project_moments(call_moments(py, moment_fn, &candidate, data)?, projection)?;
            let candidate_gbar =
                sample_mean_moments(&candidate_moments).map_err(PyValueError::new_err)?;
            if candidate_moments.raw_dim() != moments.raw_dim() {
                return Err(PyValueError::new_err(
                    "moment shape changed during line search",
                ));
            }
            let candidate_criterion = criterion_value(&candidate_gbar, weight);

            if candidate_criterion < current_criterion {
                accepted_theta = Some(candidate);
                break;
            }

            alpha *= 0.5;
        }

        let candidate = accepted_theta.ok_or_else(|| {
            PyValueError::new_err("Gauss-Newton line search failed to find a descent step")
        })?;

        iter += 1;
        theta = candidate;
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
    fitted_moments: Option<Array2<f64>>,
    fitted_jacobian: Option<Array2<f64>>,
    weight_matrix: Option<Array2<f64>>,
    first_step_theta: Option<Array1<f64>>,
    criterion: Option<f64>,
    nit: Option<usize>,
    weighting: Option<String>,
    n_obs: Option<usize>,
    n_moments: Option<usize>,
    original_n_moments: Option<usize>,
    moment_projection: Option<Array2<f64>>,
}

impl GMM {
    fn clear_fit(&mut self) {
        self.theta = None;
        self.fitted_moments = None;
        self.fitted_jacobian = None;
        self.weight_matrix = None;
        self.first_step_theta = None;
        self.criterion = None;
        self.nit = None;
        self.weighting = None;
        self.n_obs = None;
        self.n_moments = None;
        self.original_n_moments = None;
        self.moment_projection = None;
    }

    fn snapshot(
        &self,
        py: Python,
        theta: &Array1<f64>,
        data: &Py<PyAny>,
        projection: Option<&Array2<f64>>,
    ) -> PyResult<(Array2<f64>, Array2<f64>)> {
        let moments = project_moments(call_moments(py, &self.moment_fn, theta, data)?, projection)?;
        let jacobian = project_jacobian(
            sample_jacobian(
                py,
                &self.moment_fn,
                self.jacobian_fn.as_ref(),
                theta,
                data,
                self.fd_eps,
            )?,
            projection,
        )?;
        if jacobian.dim() != (moments.ncols(), theta.len()) {
            return Err(PyValueError::new_err(
                "jacobian shape does not match fitted moments and parameters",
            ));
        }
        Ok((moments, jacobian))
    }

    fn validate_vanilla(
        &self,
        vcov: &str,
        omega: &str,
        assume_optimal_weighting: bool,
    ) -> PyResult<()> {
        if vcov == "vanilla"
            && !assume_optimal_weighting
            && !(self.weighting.as_deref() == Some("two_step") && omega == "iid")
        {
            return Err(PyValueError::new_err("vanilla covariance requires optimal iid weighting; use sandwich or explicitly set assume_optimal_weighting=True"));
        }
        Ok(())
    }
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
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(PyValueError::new_err("tolerance must be positive"));
        }
        if !ridge.is_finite() || ridge < 0.0 {
            return Err(PyValueError::new_err("ridge must be nonnegative"));
        }
        if !fd_eps.is_finite() || fd_eps <= 0.0 {
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
            fitted_moments: None,
            fitted_jacobian: None,
            weight_matrix: None,
            first_step_theta: None,
            criterion: None,
            nit: None,
            weighting: None,
            n_obs: None,
            n_moments: None,
            original_n_moments: None,
            moment_projection: None,
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
        self.clear_fit();
        let theta0 = to_array1(&theta0);
        validate_finite("theta0", &theta0).map_err(PyValueError::new_err)?;
        if theta0.is_empty() {
            return Err(PyValueError::new_err("theta0 must not be empty"));
        }
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
            None,
            self.max_iterations,
            self.tolerance,
            self.ridge,
            self.fd_eps,
        )?;

        let (theta, criterion, nit, weight_matrix, first_step_theta) = if chosen_weighting
            == "two_step"
        {
            let first_moments = call_moments(py, &self.moment_fn, &first_step.theta, &data)?;
            let omega = omega_iid(&first_moments);
            let weight_matrix = invert_with_ridge(&omega, 0.0).map_err(PyValueError::new_err)?;
            let second_step = solve_gauss_newton(
                py,
                &self.moment_fn,
                self.jacobian_fn.as_ref(),
                &data,
                &first_step.theta,
                &weight_matrix,
                None,
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

        let (moments, jacobian) = self.snapshot(py, &theta, &data, None)?;
        if moments.dim() != (n, m) {
            return Err(PyValueError::new_err("moment shape changed during fitting"));
        }
        self.theta = Some(theta);
        self.fitted_moments = Some(moments);
        self.fitted_jacobian = Some(jacobian);
        self.weight_matrix = Some(weight_matrix);
        self.first_step_theta = first_step_theta;
        self.criterion = Some(criterion);
        self.nit = Some(nit);
        self.weighting = Some(chosen_weighting);
        self.n_obs = Some(n);
        self.n_moments = Some(m);
        self.original_n_moments = Some(m);
        self.moment_projection = None;
        Ok(())
    }

    #[pyo3(signature = (data, theta0, sketch_size, weighting="auto", seed=None))]
    fn fit_sketch(
        &mut self,
        py: Python,
        data: Py<PyAny>,
        theta0: PyReadonlyArray1<f64>,
        sketch_size: usize,
        weighting: &str,
        seed: Option<u64>,
    ) -> PyResult<()> {
        self.clear_fit();
        let theta0 = to_array1(&theta0);
        validate_finite("theta0", &theta0).map_err(PyValueError::new_err)?;
        if theta0.is_empty() {
            return Err(PyValueError::new_err("theta0 must not be empty"));
        }
        let initial_moments_full = call_moments(py, &self.moment_fn, &theta0, &data)?;
        let n = initial_moments_full.nrows();
        let m_full = initial_moments_full.ncols();
        let p = theta0.len();
        if sketch_size < p {
            return Err(PyValueError::new_err(
                "sketch_size must be at least the number of parameters",
            ));
        }
        let projection = rademacher_moment_projection(m_full, sketch_size, seed)?;
        let m = sketch_size;
        if m < p {
            return Err(PyValueError::new_err(format!(
                "model is underidentified after sketching: {} moments for {} parameters",
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
            Some(&projection),
            self.max_iterations,
            self.tolerance,
            self.ridge,
            self.fd_eps,
        )?;
        let (theta, criterion, nit, weight_matrix, first_step_theta) = if chosen_weighting
            == "two_step"
        {
            let first_moments = project_moments(
                call_moments(py, &self.moment_fn, &first_step.theta, &data)?,
                Some(&projection),
            )?;
            let omega = omega_iid(&first_moments);
            let weight_matrix = invert_with_ridge(&omega, 0.0).map_err(PyValueError::new_err)?;
            let second_step = solve_gauss_newton(
                py,
                &self.moment_fn,
                self.jacobian_fn.as_ref(),
                &data,
                &first_step.theta,
                &weight_matrix,
                Some(&projection),
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

        let (moments, jacobian) = self.snapshot(py, &theta, &data, Some(&projection))?;
        if moments.dim() != (n, sketch_size) {
            return Err(PyValueError::new_err("moment shape changed during fitting"));
        }
        self.theta = Some(theta);
        self.fitted_moments = Some(moments);
        self.fitted_jacobian = Some(jacobian);
        self.weight_matrix = Some(weight_matrix);
        self.first_step_theta = first_step_theta;
        self.criterion = Some(criterion);
        self.nit = Some(nit);
        self.weighting = Some(chosen_weighting);
        self.n_obs = Some(n);
        self.n_moments = Some(m);
        self.original_n_moments = Some(m_full);
        self.moment_projection = Some(projection);
        Ok(())
    }

    #[pyo3(signature = (vcov="sandwich", omega="iid", lags=None, clusters=None, *, assume_optimal_weighting=false))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: &str,
        omega: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
        assume_optimal_weighting: bool,
    ) -> PyResult<Py<PyAny>> {
        self.validate_vanilla(vcov, omega, assume_optimal_weighting)?;
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("GMM model is not fitted"))?;
        let moments = self
            .fitted_moments
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted moments stored"))?;
        let jacobian = self
            .fitted_jacobian
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted jacobian stored"))?;
        let weight_matrix = self
            .weight_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted weight matrix stored"))?;

        let gbar = sample_mean_moments(moments).map_err(PyValueError::new_err)?;

        if jacobian.nrows() != moments.ncols() || jacobian.ncols() != theta.len() {
            return Err(PyValueError::new_err(format!(
                "jacobian_fn returned shape ({}, {}), expected ({}, {})",
                jacobian.nrows(),
                jacobian.ncols(),
                moments.ncols(),
                theta.len()
            )));
        }

        let a_matrix = jacobian.t().dot(weight_matrix).dot(jacobian);
        let a_inv = invert_with_ridge(&a_matrix, 0.0).map_err(PyValueError::new_err)?;
        let n = moments.nrows();

        let covariance = match vcov {
            "vanilla" => a_inv.mapv(|v| v / (n as f64)),
            "sandwich" => {
                let omega_hat = match omega {
                    "iid" => omega_iid(moments),
                    "newey_west" => omega_newey_west(
                        moments,
                        lags.unwrap_or_else(|| default_newey_west_lags(n)),
                    ),
                    "cluster" => {
                        let clusters = clusters.ok_or_else(|| {
                            PyValueError::new_err("clusters must be provided for omega='cluster'")
                        })?;
                        let cluster_ids = to_array1_i64(&clusters);
                        omega_cluster(moments, &cluster_ids).map_err(PyValueError::new_err)?
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
                    .dot(jacobian);
                a_inv.dot(&middle).dot(&a_inv).mapv(|v| v / (n as f64))
            }
            _ => {
                return Err(PyValueError::new_err(
                    "vcov must be one of {'vanilla', 'sandwich'}",
                ));
            }
        };

        let se = diag_sqrt(&covariance).map_err(PyValueError::new_err)?;
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
        dict.set_item("converged", true)?;
        dict.set_item(
            "termination_reason",
            "Scaled first-order and Newton-step tolerances reached",
        )?;
        dict.set_item("inference_data", "fit_time_snapshot")?;
        dict.set_item("assume_optimal_weighting", assume_optimal_weighting)?;
        dict.set_item(
            "j_test_valid",
            self.weighting.as_deref() == Some("two_step") && omega == "iid",
        )?;
        dict.set_item(
            "j_test_assumptions",
            "iid observations, valid moments, identification, and optimal weighting",
        )?;
        dict.set_item("weighting", self.weighting.clone())?;
        dict.set_item("vcov_type", vcov)?;
        dict.set_item(
            "omega_type",
            if vcov == "sandwich" {
                Some(omega)
            } else {
                None::<&str>
            },
        )?;
        dict.set_item("weight_matrix", pyarray2_from_f64(py, weight_matrix))?;
        dict.set_item("nobs", n)?;
        dict.set_item("n_moments", moments.ncols())?;
        dict.set_item("original_n_moments", self.original_n_moments)?;
        dict.set_item(
            "sketch_size",
            self.moment_projection.as_ref().map(|p| p.ncols()),
        )?;
        dict.set_item("j_stat", j_stat)?;
        dict.set_item("j_df", j_df)?;
        Ok(dict.into())
    }

    #[pyo3(signature = (r, q=None, vcov="sandwich", omega="iid", lags=None, clusters=None, *, assume_optimal_weighting=false))]
    fn wald_test<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<f64>,
        q: Option<PyReadonlyArray1<f64>>,
        vcov: &str,
        omega: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
        assume_optimal_weighting: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.validate_vanilla(vcov, omega, assume_optimal_weighting)?;
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("GMM model is not fitted"))?;
        let moments = self
            .fitted_moments
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted moments stored"))?;
        let jacobian = self
            .fitted_jacobian
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted jacobian stored"))?;
        let weight_matrix = self
            .weight_matrix
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No fitted weight matrix stored"))?;

        if jacobian.nrows() != moments.ncols() || jacobian.ncols() != theta.len() {
            return Err(PyValueError::new_err(format!(
                "jacobian_fn returned shape ({}, {}), expected ({}, {})",
                jacobian.nrows(),
                jacobian.ncols(),
                moments.ncols(),
                theta.len()
            )));
        }

        let a_matrix = jacobian.t().dot(weight_matrix).dot(jacobian);
        let a_inv = invert_with_ridge(&a_matrix, 0.0).map_err(PyValueError::new_err)?;
        let n = moments.nrows();
        let covariance = match vcov {
            "vanilla" => a_inv.mapv(|v| v / (n as f64)),
            "sandwich" => {
                let omega_hat = match omega {
                    "iid" => omega_iid(moments),
                    "newey_west" => omega_newey_west(
                        moments,
                        lags.unwrap_or_else(|| default_newey_west_lags(n)),
                    ),
                    "cluster" => {
                        let clusters = clusters.ok_or_else(|| {
                            PyValueError::new_err("clusters must be provided for omega='cluster'")
                        })?;
                        let cluster_ids = to_array1_i64(&clusters);
                        omega_cluster(moments, &cluster_ids).map_err(PyValueError::new_err)?
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
                    .dot(jacobian);
                a_inv.dot(&middle).dot(&a_inv).mapv(|v| v / (n as f64))
            }
            _ => {
                return Err(PyValueError::new_err(
                    "vcov must be one of {'vanilla', 'sandwich'}",
                ));
            }
        };
        let rmat = to_array2(&r);
        let qvec = q.as_ref().map(to_array1);
        wald_test_arrays(py, theta, &covariance, &rmat, qvec.as_ref())
    }
}
