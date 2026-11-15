mod bhhh;
mod common;
mod quasi_newton;
mod trust_region;

use crate::fit::optimization_success;
use crate::utils::{pyarray1_from_f64, to_array1, to_array2};
use argmin::core::{
    CostFunction, Error as ArgminError, Executor, Gradient, Jacobian, Operator, State,
    TerminationReason, TerminationStatus,
};
use argmin::solver::{
    conjugategradient::{beta::PolakRibierePlus, NonlinearConjugateGradient},
    linesearch::{condition::ArmijoCondition, BacktrackingLineSearch},
    simulatedannealing::{Anneal, SATempFunc, SimulatedAnnealing},
};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand09::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::{Arc, Mutex};

fn extract_array1_from_pyany(
    value: Bound<'_, PyAny>,
    err_msg: &str,
) -> Result<Array1<f64>, ArgminError> {
    let array = value
        .cast::<PyArray1<f64>>()
        .map_err(|_| ArgminError::msg(err_msg.to_string()))?;
    Ok(to_array1(&array.readonly()))
}

fn extract_array2_from_pyany(
    value: Bound<'_, PyAny>,
    err_msg: &str,
) -> Result<Array2<f64>, ArgminError> {
    let array = value
        .cast::<PyArray2<f64>>()
        .map_err(|_| ArgminError::msg(err_msg.to_string()))?;
    Ok(to_array2(&array.readonly()))
}

fn vec_from_array1(x: &Array1<f64>) -> Vec<f64> {
    x.to_vec()
}

fn vecvec_from_array2(x: &Array2<f64>) -> Vec<Vec<f64>> {
    x.rows().into_iter().map(|row| row.to_vec()).collect()
}

fn array1_from_vec(x: &[f64]) -> Array1<f64> {
    Array1::from_vec(x.to_vec())
}

fn call_objective_array1(objective_fn: &Py<PyAny>, theta: &Array1<f64>) -> PyResult<f64> {
    Python::attach(|py| {
        let theta_py = pyarray1_from_f64(py, theta);
        objective_fn
            .call1(py, (theta_py,))
            .map_err(|e| PyValueError::new_err(format!("Python callback error: {}", e)))?
            .extract::<f64>(py)
            .map_err(|e| PyValueError::new_err(format!("Failed to extract objective: {}", e)))
    })
}

fn call_gradient_array1(gradient_fn: &Py<PyAny>, theta: &Array1<f64>) -> PyResult<Array1<f64>> {
    Python::attach(|py| {
        let theta_py = pyarray1_from_f64(py, theta);
        let result = gradient_fn
            .call1(py, (theta_py,))
            .map_err(|e| PyValueError::new_err(format!("Python callback error: {}", e)))?;
        extract_array1_from_pyany(
            result.bind(py).clone(),
            "Gradient must return a 1D numpy array",
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))
    })
}

fn optimize_result_dict_explicit<'py>(
    py: Python<'py>,
    x: &Array1<f64>,
    fun: f64,
    nit: u64,
    success: bool,
    message: &str,
    method: &str,
) -> PyResult<Py<PyAny>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("x", pyarray1_from_f64(py, x))?;
    dict.set_item("fun", fun)?;
    dict.set_item("nit", nit)?;
    dict.set_item("success", success)?;
    dict.set_item("message", message)?;
    dict.set_item("method", method)?;
    Ok(dict.into())
}

fn optimize_result_dict<'py>(
    py: Python<'py>,
    x: &Array1<f64>,
    fun: f64,
    nit: u64,
    status: &TerminationStatus,
    method: &str,
) -> PyResult<Py<PyAny>> {
    let message = status.to_string();
    optimize_result_dict_explicit(
        py,
        x,
        fun,
        nit,
        optimization_success(status),
        &message,
        method,
    )
}

struct ScalarObjectiveProblem {
    objective_fn: Py<PyAny>,
    gradient_fn: Py<PyAny>,
}

impl CostFunction for ScalarObjectiveProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, ArgminError> {
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            self.objective_fn
                .call1(py, (theta_py,))
                .map_err(|e| ArgminError::msg(format!("Python callback error: {}", e)))?
                .extract::<f64>(py)
                .map_err(|e| ArgminError::msg(format!("Failed to extract objective: {}", e)))
        })
    }
}

impl Gradient for ScalarObjectiveProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, theta: &Self::Param) -> Result<Self::Gradient, ArgminError> {
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            let result = self
                .gradient_fn
                .call1(py, (theta_py,))
                .map_err(|e| ArgminError::msg(format!("Python callback error: {}", e)))?;
            extract_array1_from_pyany(
                result.bind(py).clone(),
                "Gradient must return a 1D numpy array",
            )
        })
    }
}

struct ResidualProblem {
    residual_fn: Py<PyAny>,
    jacobian_fn: Py<PyAny>,
}

impl Operator for ResidualProblem {
    type Param = Vec<f64>;
    type Output = Vec<f64>;

    fn apply(&self, theta: &Self::Param) -> Result<Self::Output, ArgminError> {
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, &array1_from_vec(theta));
            let result = self
                .residual_fn
                .call1(py, (theta_py,))
                .map_err(|e| ArgminError::msg(format!("Python callback error: {}", e)))?;
            extract_array1_from_pyany(
                result.bind(py).clone(),
                "Residual function must return a 1D numpy array",
            )
            .map(|arr| vec_from_array1(&arr))
        })
    }
}

impl Jacobian for ResidualProblem {
    type Param = Vec<f64>;
    type Jacobian = Vec<Vec<f64>>;

    fn jacobian(&self, theta: &Self::Param) -> Result<Self::Jacobian, ArgminError> {
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, &array1_from_vec(theta));
            let result = self
                .jacobian_fn
                .call1(py, (theta_py,))
                .map_err(|e| ArgminError::msg(format!("Python callback error: {}", e)))?;
            extract_array2_from_pyany(
                result.bind(py).clone(),
                "Jacobian function must return a 2D numpy array",
            )
            .map(|arr| vecvec_from_array2(&arr))
        })
    }
}

struct AnnealingProblem {
    objective_fn: Py<PyAny>,
    lower_bound: Array1<f64>,
    upper_bound: Array1<f64>,
    step_size: f64,
    rng: Arc<Mutex<Xoshiro256PlusPlus>>,
}

impl CostFunction for AnnealingProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> Result<Self::Output, ArgminError> {
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            self.objective_fn
                .call1(py, (theta_py,))
                .map_err(|e| ArgminError::msg(format!("Python callback error: {}", e)))?
                .extract::<f64>(py)
                .map_err(|e| ArgminError::msg(format!("Failed to extract objective: {}", e)))
        })
    }
}

impl Anneal for AnnealingProblem {
    type Param = Array1<f64>;
    type Output = Array1<f64>;
    type Float = f64;

    fn anneal(
        &self,
        param: &Self::Param,
        extent: Self::Float,
    ) -> Result<Self::Output, ArgminError> {
        let mut candidate = param.clone();
        let mut rng = self
            .rng
            .lock()
            .map_err(|_| ArgminError::msg("Failed to lock simulated annealing RNG"))?;
        let n_modifications = extent.floor().max(1.0) as usize;
        let index_dist = rand09::distr::Uniform::try_from(0..param.len())
            .map_err(|e| ArgminError::msg(e.to_string()))?;
        let perturb_dist = rand09::distr::Uniform::new_inclusive(-self.step_size, self.step_size)
            .map_err(|e| ArgminError::msg(e.to_string()))?;

        for _ in 0..n_modifications {
            let idx = rand09::Rng::sample(&mut *rng, index_dist);
            let perturbation = rand09::Rng::sample(&mut *rng, perturb_dist);
            candidate[idx] =
                (candidate[idx] + perturbation).clamp(self.lower_bound[idx], self.upper_bound[idx]);
        }

        Ok(candidate)
    }
}

fn optional_bounds(
    x0: &Array1<f64>,
    lower: Option<&PyReadonlyArray1<f64>>,
    upper: Option<&PyReadonlyArray1<f64>>,
) -> PyResult<(Array1<f64>, Array1<f64>)> {
    let lower_bound = match lower {
        Some(lower) => {
            let bound = to_array1(lower);
            if bound.len() != x0.len() {
                return Err(PyValueError::new_err(
                    "lower bound length must match x0 length",
                ));
            }
            bound
        }
        None => Array1::from_elem(x0.len(), f64::NEG_INFINITY),
    };

    let upper_bound = match upper {
        Some(upper) => {
            let bound = to_array1(upper);
            if bound.len() != x0.len() {
                return Err(PyValueError::new_err(
                    "upper bound length must match x0 length",
                ));
            }
            bound
        }
        None => Array1::from_elem(x0.len(), f64::INFINITY),
    };

    for i in 0..x0.len() {
        if lower_bound[i] > upper_bound[i] {
            return Err(PyValueError::new_err(
                "lower bound cannot exceed upper bound",
            ));
        }
    }

    Ok((lower_bound, upper_bound))
}

#[pyclass]
pub struct Optimizers;

#[pymethods]
impl Optimizers {
    /// Safeguarded BFGS with independent gradient/cost tolerances and a final
    /// gradient certificate. x_scale defines x = diag(x_scale) z.
    #[staticmethod]
    #[pyo3(signature = (fun, x0, grad, max_iterations=100, tolerance=1e-6, *, gtol=None, ftol=0.0, c1=1e-4, c2=0.9, max_linesearch=80, x_scale=None))]
    #[allow(clippy::too_many_arguments)]
    fn minimize_bfgs(
        py: Python<'_>,
        fun: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        grad: Py<PyAny>,
        max_iterations: usize,
        tolerance: f64,
        gtol: Option<f64>,
        ftol: f64,
        c1: f64,
        c2: f64,
        max_linesearch: usize,
        x_scale: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        run_quasi(
            py,
            fun,
            x0,
            grad,
            max_iterations,
            tolerance,
            gtol,
            ftol,
            c1,
            c2,
            max_linesearch,
            x_scale,
            false,
        )
    }

    /// Limited-memory BFGS with seven safeguarded correction pairs.
    #[staticmethod]
    #[pyo3(signature = (fun, x0, grad, max_iterations=100, tolerance=1e-6, *, gtol=None, ftol=0.0, c1=1e-4, c2=0.9, max_linesearch=80, x_scale=None))]
    #[allow(clippy::too_many_arguments)]
    fn minimize_lbfgs(
        py: Python<'_>,
        fun: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        grad: Py<PyAny>,
        max_iterations: usize,
        tolerance: f64,
        gtol: Option<f64>,
        ftol: f64,
        c1: f64,
        c2: f64,
        max_linesearch: usize,
        x_scale: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        run_quasi(
            py,
            fun,
            x0,
            grad,
            max_iterations,
            tolerance,
            gtol,
            ftol,
            c1,
            c2,
            max_linesearch,
            x_scale,
            true,
        )
    }

    /// Exact-Hessian dense trust regions, including indefinite and singular
    /// subproblems. This is a native Rust solver, not a GALAHAD binding.
    #[staticmethod]
    #[pyo3(signature = (fun, x0, grad, hess, max_iterations=100, tolerance=1e-6, *, gtol=None, ftol=0.0, initial_radius=1.0, max_radius=1e6, eta=0.1, x_scale=None))]
    #[allow(clippy::too_many_arguments)]
    fn minimize_trust_region(
        py: Python<'_>,
        fun: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        grad: Py<PyAny>,
        hess: Py<PyAny>,
        max_iterations: usize,
        tolerance: f64,
        gtol: Option<f64>,
        ftol: f64,
        initial_radius: f64,
        max_radius: f64,
        eta: f64,
        x_scale: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        let (z, scale) = common::coordinates(to_array1(&x0), x_scale.as_ref().map(to_array1))?;
        let problem = common::PythonObjective {
            fun,
            grad,
            hess: Some(hess),
            scale,
            counts: common::Counts::default(),
        };
        let controls = common::Controls {
            max_iterations,
            gtol: gtol.unwrap_or(tolerance),
            ftol,
            c1: 1e-4,
            c2: 0.9,
            max_linesearch: 80,
        };
        trust_region::solve(py, problem, z, controls, initial_radius, max_radius, eta)
    }

    /// BHHH for negative-log-likelihood scores. score_stats is an optional
    /// memory-bounded alternative returning (score_sum, outer_sum, n_observations).
    #[staticmethod]
    #[pyo3(signature = (fun, x0, scores=None, max_iterations=100, tolerance=1e-6, *, score_stats=None, reduction="mean", penalty_grad=None, penalty_hess=None, gtol=None, ftol=0.0, damping=1e-8, c1=1e-4, c2=0.9, max_linesearch=80, x_scale=None))]
    #[allow(clippy::too_many_arguments)]
    fn minimize_bhhh(
        py: Python<'_>,
        fun: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        scores: Option<Py<PyAny>>,
        max_iterations: usize,
        tolerance: f64,
        score_stats: Option<Py<PyAny>>,
        reduction: &str,
        penalty_grad: Option<Py<PyAny>>,
        penalty_hess: Option<Py<PyAny>>,
        gtol: Option<f64>,
        ftol: f64,
        damping: f64,
        c1: f64,
        c2: f64,
        max_linesearch: usize,
        x_scale: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Py<PyAny>> {
        let (z, scale) = common::coordinates(to_array1(&x0), x_scale.as_ref().map(to_array1))?;
        let problem = bhhh::ScoreObjective::new(
            fun,
            scores,
            score_stats,
            penalty_grad,
            penalty_hess,
            reduction,
            scale,
        )?;
        let controls = common::Controls {
            max_iterations,
            gtol: gtol.unwrap_or(tolerance),
            ftol,
            c1,
            c2,
            max_linesearch,
        };
        bhhh::solve(py, problem, z, controls, damping)
    }

    #[staticmethod]
    #[pyo3(signature = (fun, x0, grad, max_iterations=100, restart_iters=10, restart_orthogonality=0.1, tolerance=1e-6))]
    fn minimize_nonlinear_cg<'py>(
        py: Python<'py>,
        fun: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        grad: Py<PyAny>,
        max_iterations: usize,
        restart_iters: u64,
        restart_orthogonality: f64,
        tolerance: f64,
    ) -> PyResult<Py<PyAny>> {
        minimize_nonlinear_cg(
            py,
            fun,
            x0,
            grad,
            max_iterations,
            restart_iters,
            restart_orthogonality,
            tolerance,
        )
    }

    #[staticmethod]
    #[pyo3(signature = (residual_fn, x0, jacobian_fn, max_iterations=100, tolerance=1e-6))]
    fn minimize_gauss_newton_ls<'py>(
        py: Python<'py>,
        residual_fn: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        jacobian_fn: Py<PyAny>,
        max_iterations: usize,
        tolerance: f64,
    ) -> PyResult<Py<PyAny>> {
        minimize_gauss_newton_ls(py, residual_fn, x0, jacobian_fn, max_iterations, tolerance)
    }

    #[staticmethod]
    #[pyo3(signature = (fun, x0, lower=None, upper=None, temp=15.0, step_size=0.1, max_iterations=5000, seed=None))]
    fn minimize_simulated_annealing<'py>(
        py: Python<'py>,
        fun: Py<PyAny>,
        x0: PyReadonlyArray1<f64>,
        lower: Option<PyReadonlyArray1<f64>>,
        upper: Option<PyReadonlyArray1<f64>>,
        temp: f64,
        step_size: f64,
        max_iterations: usize,
        seed: Option<u64>,
    ) -> PyResult<Py<PyAny>> {
        minimize_simulated_annealing(
            py,
            fun,
            x0,
            lower,
            upper,
            temp,
            step_size,
            max_iterations,
            seed,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn run_quasi(
    py: Python<'_>,
    fun: Py<PyAny>,
    x0: PyReadonlyArray1<f64>,
    grad: Py<PyAny>,
    max_iterations: usize,
    tolerance: f64,
    gtol: Option<f64>,
    ftol: f64,
    c1: f64,
    c2: f64,
    max_linesearch: usize,
    x_scale: Option<PyReadonlyArray1<f64>>,
    limited_memory: bool,
) -> PyResult<Py<PyAny>> {
    let (z, scale) = common::coordinates(to_array1(&x0), x_scale.as_ref().map(to_array1))?;
    let problem = common::PythonObjective {
        fun,
        grad,
        hess: None,
        scale,
        counts: common::Counts::default(),
    };
    let controls = common::Controls {
        max_iterations,
        gtol: gtol.unwrap_or(tolerance),
        ftol,
        c1,
        c2,
        max_linesearch,
    };
    quasi_newton::solve(py, problem, z, controls, limited_memory)
}

#[pyfunction]
#[pyo3(signature = (fun, x0, grad, max_iterations=100, restart_iters=10, restart_orthogonality=0.1, tolerance=1e-6))]
pub fn minimize_nonlinear_cg<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    x0: PyReadonlyArray1<f64>,
    grad: Py<PyAny>,
    max_iterations: usize,
    restart_iters: u64,
    restart_orthogonality: f64,
    tolerance: f64,
) -> PyResult<Py<PyAny>> {
    let x0 = to_array1(&x0);
    let fun_eval = Python::attach(|py| fun.clone_ref(py));
    let grad_eval = Python::attach(|py| grad.clone_ref(py));
    let problem = ScalarObjectiveProblem {
        objective_fn: fun,
        gradient_fn: grad,
    };
    let linesearch: BacktrackingLineSearch<Array1<f64>, Array1<f64>, ArmijoCondition<f64>, f64> =
        BacktrackingLineSearch::new(
            ArmijoCondition::new(0.2).map_err(|err| PyValueError::new_err(err.to_string()))?,
        );
    let beta_method = PolakRibierePlus::new();
    let solver = NonlinearConjugateGradient::new(linesearch, beta_method)
        .restart_iters(restart_iters)
        .restart_orthogonality(restart_orthogonality);
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(x0).max_iters(max_iterations as u64))
        .run()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let x =
        result.state.get_best_param().cloned().ok_or_else(|| {
            PyValueError::new_err("Nonlinear CG did not produce a parameter vector")
        })?;
    let fun_value = call_objective_array1(&fun_eval, &x)?;
    let final_gradient = call_gradient_array1(&grad_eval, &x)?;
    let grad_norm = final_gradient.dot(&final_gradient).sqrt();
    let status = result.state.get_termination_status();
    let success = optimization_success(status) || grad_norm <= tolerance;
    let message = if success && !optimization_success(status) {
        format!(
            "Gradient norm below tolerance ({:.3e} <= {:.3e})",
            grad_norm, tolerance
        )
    } else {
        status.to_string()
    };
    optimize_result_dict_explicit(
        py,
        &x,
        fun_value,
        result.state.get_iter(),
        success,
        &message,
        "nonlinear_cg",
    )
}

#[pyfunction]
#[pyo3(signature = (residual_fn, x0, jacobian_fn, max_iterations=100, tolerance=1e-6))]
pub fn minimize_gauss_newton_ls<'py>(
    py: Python<'py>,
    residual_fn: Py<PyAny>,
    x0: PyReadonlyArray1<f64>,
    jacobian_fn: Py<PyAny>,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<Py<PyAny>> {
    let x0 = vec_from_array1(&to_array1(&x0));
    let residual_eval = Python::attach(|py| residual_fn.clone_ref(py));
    let problem = ResidualProblem {
        residual_fn,
        jacobian_fn,
    };
    let mut x = x0;
    let mut iter = 0u64;
    let status;

    loop {
        let residual = problem
            .apply(&x)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let jacobian = problem
            .jacobian(&x)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        if jacobian.len() != residual.len() || jacobian.iter().any(|row| row.len() != x.len()) {
            return Err(PyValueError::new_err(
                "Jacobian shape must be (n_residuals, n_parameters)",
            ));
        }
        let residual_arr = array1_from_vec(&residual);
        let jacobian_arr = Array2::from_shape_vec(
            (residual.len(), x.len()),
            jacobian.into_iter().flatten().collect(),
        )
        .map_err(|_| PyValueError::new_err("Invalid Jacobian shape"))?;
        let current_cost = 0.5 * residual_arr.dot(&residual_arr);

        let jt = jacobian_arr.t().to_owned();
        let gradient = jt.dot(&residual_arr);
        if gradient.dot(&gradient).sqrt() <= tolerance {
            status = TerminationStatus::Terminated(TerminationReason::SolverConverged);
            break;
        }
        let normal = jt.dot(&jacobian_arr);
        let inv = crate::utils::invert_matrix(&normal).map_err(PyValueError::new_err)?;
        let step = inv.dot(&gradient);
        if step.dot(&step).sqrt() <= tolerance {
            status = TerminationStatus::Terminated(TerminationReason::SolverConverged);
            break;
        }

        let mut alpha = 1.0;
        let mut accepted = false;
        let mut candidate = x.clone();
        let mut candidate_cost = current_cost;

        while alpha >= 1e-8 {
            candidate = x
                .iter()
                .zip(step.iter())
                .map(|(xi, si)| xi - alpha * si)
                .collect::<Vec<_>>();
            let candidate_residual = problem
                .apply(&candidate)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let candidate_residual_arr = array1_from_vec(&candidate_residual);
            candidate_cost = 0.5 * candidate_residual_arr.dot(&candidate_residual_arr);
            if candidate_cost < current_cost {
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }

        if !accepted {
            status = TerminationStatus::Terminated(TerminationReason::SolverExit(
                "Gauss-Newton line search failed to find a descent step".to_string(),
            ));
            break;
        }

        iter += 1;
        if (current_cost - candidate_cost).abs() < tolerance {
            x = candidate;
            status = TerminationStatus::Terminated(TerminationReason::SolverConverged);
            break;
        }

        x = candidate;
        if iter >= max_iterations as u64 {
            status = TerminationStatus::Terminated(TerminationReason::MaxItersReached);
            break;
        }
    }

    let residual = Python::attach(|py| {
        let theta_py = pyarray1_from_f64(py, &array1_from_vec(&x));
        let result = residual_eval
            .call1(py, (theta_py,))
            .map_err(|e| PyValueError::new_err(format!("Python callback error: {}", e)))?;
        extract_array1_from_pyany(
            result.bind(py).clone(),
            "Residual function must return a 1D numpy array",
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))
    })?;
    let fun_value = residual.dot(&residual) * 0.5;
    let x_array = array1_from_vec(&x);
    optimize_result_dict(py, &x_array, fun_value, iter, &status, "gauss_newton_ls")
}

#[pyfunction]
#[pyo3(signature = (fun, x0, lower=None, upper=None, temp=15.0, step_size=0.1, max_iterations=5000, seed=None))]
pub fn minimize_simulated_annealing<'py>(
    py: Python<'py>,
    fun: Py<PyAny>,
    x0: PyReadonlyArray1<f64>,
    lower: Option<PyReadonlyArray1<f64>>,
    upper: Option<PyReadonlyArray1<f64>>,
    temp: f64,
    step_size: f64,
    max_iterations: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    if temp <= 0.0 {
        return Err(PyValueError::new_err("temp must be positive"));
    }
    if step_size <= 0.0 {
        return Err(PyValueError::new_err("step_size must be positive"));
    }

    let x0 = to_array1(&x0);
    let fun_eval = Python::attach(|py| fun.clone_ref(py));
    let (lower_bound, upper_bound) = optional_bounds(&x0, lower.as_ref(), upper.as_ref())?;
    let rng = match seed {
        Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
        None => Xoshiro256PlusPlus::from_os_rng(),
    };
    let problem = AnnealingProblem {
        objective_fn: fun,
        lower_bound,
        upper_bound,
        step_size,
        rng: Arc::new(Mutex::new(rng)),
    };
    let solver_rng = match seed {
        Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(1)),
        None => Xoshiro256PlusPlus::from_os_rng(),
    };
    let solver: SimulatedAnnealing<f64, Xoshiro256PlusPlus> =
        SimulatedAnnealing::new_with_rng(temp, solver_rng)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .with_temp_func(SATempFunc::Boltzmann)
            .with_stall_best(1000)
            .with_stall_accepted(1000);
    let result = Executor::new(problem, solver)
        .configure(|state| state.param(x0).max_iters(max_iterations as u64))
        .run()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let x = result.state.get_best_param().cloned().ok_or_else(|| {
        PyValueError::new_err("Simulated annealing did not produce a parameter vector")
    })?;
    let fun_value = call_objective_array1(&fun_eval, &x)?;
    optimize_result_dict(
        py,
        &x,
        fun_value,
        result.state.get_iter(),
        result.state.get_termination_status(),
        "simulated_annealing",
    )
}
