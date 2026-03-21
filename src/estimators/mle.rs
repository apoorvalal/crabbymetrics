use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, fisher_cov_binary, fisher_cov_multinomial,
    fisher_cov_poisson, invert_matrix, pyarray1_from_f64, pyarray1_from_i32, pyarray2_from_f64,
    take_rows, take_rows_i32, take_rows_vec, to_array1, to_array1_i32, to_array2,
};
use argmin::core::{CostFunction, Executor, Gradient, Hessian};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::newton::NewtonCG;
use argmin::solver::quasinewton::LBFGS;
use linfa::prelude::{Fit, Predict};
use linfa::Dataset;
use linfa_logistic::{LogisticRegression as LinfaLogisticRegression, MultiLogisticRegression};
use ndarray::{array, concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::{Rng, SeedableRng};

#[pyclass]
pub struct Logit {
    alpha: f64,
    fit_intercept: bool,
    max_iterations: u64,
    gradient_tolerance: f64,
    model: Option<linfa_logistic::FittedLogisticRegression<f64, i32>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<i32>>,
}

#[pymethods]
impl Logit {
    #[new]
    #[pyo3(signature = (alpha=1.0, max_iterations=100, gradient_tolerance=1e-4))]
    fn new(alpha: f64, max_iterations: u64, gradient_tolerance: f64) -> Self {
        Self {
            alpha,
            fit_intercept: true,
            max_iterations,
            gradient_tolerance,
            model: None,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i32>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1_i32(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let dataset = Dataset::new(x.clone(), y.clone());
        let params = LinfaLogisticRegression::new()
            .alpha(self.alpha)
            .with_intercept(self.fit_intercept)
            .max_iterations(self.max_iterations)
            .gradient_tolerance(self.gradient_tolerance);
        let model = params
            .fit(&dataset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.model = Some(model);
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Logit model is not fitted"))?;
        let x = to_array2(&x);
        let pred = model.predict(&x);
        Ok(pyarray1_from_i32(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Logit model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let probs = model.predict_probabilities(x);
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = fisher_cov_binary(&design, &probs).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept, coef, intercept_se, coef_se) = if self.fit_intercept {
            (
                model.intercept(),
                model.params().to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, model.params().to_owned(), None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, &coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((
            n_bootstrap,
            x.ncols() + if self.fit_intercept { 1 } else { 0 },
        ));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_i32(y, idx);
            let dataset = Dataset::new(xb, yb);
            let params = LinfaLogisticRegression::new()
                .alpha(self.alpha)
                .with_intercept(self.fit_intercept)
                .max_iterations(self.max_iterations)
                .gradient_tolerance(self.gradient_tolerance);
            let model = params
                .fit(&dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            if self.fit_intercept {
                out[[i, 0]] = model.intercept();
                out.row_mut(i).slice_mut(s![1..]).assign(model.params());
            } else {
                out.row_mut(i).assign(model.params());
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct MultinomialLogit {
    alpha: f64,
    fit_intercept: bool,
    max_iterations: u64,
    gradient_tolerance: f64,
    model: Option<linfa_logistic::MultiFittedLogisticRegression<f64, i32>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<i32>>,
}

#[pymethods]
impl MultinomialLogit {
    #[new]
    #[pyo3(signature = (alpha=1.0, max_iterations=100, gradient_tolerance=1e-4))]
    fn new(alpha: f64, max_iterations: u64, gradient_tolerance: f64) -> Self {
        Self {
            alpha,
            fit_intercept: true,
            max_iterations,
            gradient_tolerance,
            model: None,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i32>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1_i32(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let dataset = Dataset::new(x.clone(), y.clone());
        let params = MultiLogisticRegression::new()
            .alpha(self.alpha)
            .with_intercept(self.fit_intercept)
            .max_iterations(self.max_iterations)
            .gradient_tolerance(self.gradient_tolerance);
        let model = params
            .fit(&dataset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        self.model = Some(model);
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MultinomialLogit model is not fitted"))?;
        let x = to_array2(&x);
        let pred = model.predict(&x);
        Ok(pyarray1_from_i32(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MultinomialLogit model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let probs = model.predict_probabilities(x);
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };

        let cov = fisher_cov_multinomial(&design, &probs).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);
        let k = design.ncols();
        let c = probs.ncols();
        let mut coef = Array2::<f64>::zeros((c, k));
        let mut se = Array2::<f64>::zeros((c, k));
        let params = model.params();
        let intercept = model.intercept();

        for class in 0..c {
            for j in 0..k {
                let idx = class * k + j;
                se[[class, j]] = se_all[idx];
                if self.fit_intercept {
                    if j == 0 {
                        coef[[class, j]] = intercept[class];
                    } else {
                        coef[[class, j]] = params[[j - 1, class]];
                    }
                } else {
                    coef[[class, j]] = params[[j, class]];
                }
            }
        }

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray2_from_f64(py, &coef))?;
        dict.set_item("se", pyarray2_from_f64(py, &se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let k = x.ncols() + if self.fit_intercept { 1 } else { 0 };
        let c = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MultinomialLogit model is not fitted"))?
            .classes()
            .len();
        let mut out = Array2::<f64>::zeros((n_bootstrap, k * c));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_i32(y, idx);
            let dataset = Dataset::new(xb, yb);
            let params = MultiLogisticRegression::new()
                .alpha(self.alpha)
                .with_intercept(self.fit_intercept)
                .max_iterations(self.max_iterations)
                .gradient_tolerance(self.gradient_tolerance);
            let model = params
                .fit(&dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let mut offset = 0;
            for class in 0..c {
                if self.fit_intercept {
                    out[[i, offset]] = model.intercept()[class];
                    out.row_mut(i)
                        .slice_mut(s![offset + 1..offset + k])
                        .assign(&model.params().column(class));
                } else {
                    out.row_mut(i)
                        .slice_mut(s![offset..offset + k])
                        .assign(&model.params().column(class));
                }
                offset += k;
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct Poisson {
    alpha: f64,
    fit_intercept: bool,
    max_iterations: usize,
    tolerance: f64,
    coef: Option<Array1<f64>>,
    intercept: f64,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

struct PoissonProblem<'a> {
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, f64>,
    fit_intercept: bool,
    alpha: f64,
}

impl CostFunction for PoissonProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        let (intercept, beta) = if self.fit_intercept {
            (p[0], p.slice(s![1..]))
        } else {
            (0.0, p.view())
        };
        let eta = self.x.dot(&beta).mapv(|v| v + intercept);
        let exp_eta = eta.mapv(|v| v.exp());
        let ll = exp_eta.sum() - self.y.dot(&eta);
        let l2 = 0.5 * self.alpha * beta.dot(&beta);
        Ok(ll + l2)
    }
}

impl Gradient for PoissonProblem<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, p: &Self::Param) -> std::result::Result<Self::Param, argmin::core::Error> {
        let (intercept, beta) = if self.fit_intercept {
            (p[0], p.slice(s![1..]))
        } else {
            (0.0, p.view())
        };
        let eta = self.x.dot(&beta).mapv(|v| v + intercept);
        let exp_eta = eta.mapv(|v| v.exp());
        let residual = &exp_eta - &self.y;
        let mut grad_beta = self.x.t().dot(&residual);
        if self.alpha > 0.0 {
            grad_beta = grad_beta + beta.to_owned() * self.alpha;
        }
        if self.fit_intercept {
            let mut grad = Array1::<f64>::zeros(beta.len() + 1);
            grad[0] = residual.sum();
            grad.slice_mut(s![1..]).assign(&grad_beta);
            Ok(grad)
        } else {
            Ok(grad_beta)
        }
    }
}

impl Hessian for PoissonProblem<'_> {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>;

    fn hessian(&self, p: &Self::Param) -> std::result::Result<Self::Hessian, argmin::core::Error> {
        let (intercept, beta) = if self.fit_intercept {
            (p[0], p.slice(s![1..]))
        } else {
            (0.0, p.view())
        };
        let eta = self.x.dot(&beta).mapv(|v| v + intercept);
        let w = eta.mapv(|v| v.exp());
        let n = self.x.nrows();
        let k = self.x.ncols();

        let mut xw = self.x.to_owned();
        for i in 0..n {
            let wi = w[i];
            xw.row_mut(i).mapv_inplace(|v| v * wi);
        }
        let mut h_beta = self.x.t().dot(&xw);
        for j in 0..k {
            h_beta[[j, j]] += self.alpha.max(0.0) + 1e-8;
        }

        if self.fit_intercept {
            let mut h = Array2::<f64>::zeros((k + 1, k + 1));
            h[[0, 0]] = w.sum() + 1e-8;
            for j in 0..k {
                let val = self.x.column(j).dot(&w);
                h[[0, j + 1]] = val;
                h[[j + 1, 0]] = val;
            }
            h.slice_mut(s![1.., 1..]).assign(&h_beta);
            Ok(h)
        } else {
            Ok(h_beta)
        }
    }
}

#[pymethods]
impl Poisson {
    #[new]
    #[pyo3(signature = (alpha=0.0, max_iterations=100, tolerance=1e-4))]
    fn new(alpha: f64, max_iterations: usize, tolerance: f64) -> Self {
        Self {
            alpha,
            fit_intercept: true,
            max_iterations,
            tolerance,
            coef: None,
            intercept: 0.0,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let mut coef = Array1::<f64>::zeros(x.ncols());
        if self.fit_intercept {
            let mean_y = y.mean().unwrap_or(0.0).max(1e-12);
            coef = concatenate(Axis(0), &[array![mean_y.ln()].view(), coef.view()])
                .map_err(|_| PyValueError::new_err("failed to init coefficients"))?;
        }

        let problem = PoissonProblem {
            x: x.view(),
            y: y.view(),
            fit_intercept: self.fit_intercept,
            alpha: self.alpha,
        };
        let linesearch = MoreThuenteLineSearch::new();
        let solver = NewtonCG::new(linesearch)
            .with_tolerance(self.tolerance)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mut result = Executor::new(problem, solver)
            .configure(|state| state.param(coef).max_iters(self.max_iterations as u64))
            .run()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let params = result
            .state
            .take_best_param()
            .ok_or_else(|| PyValueError::new_err("solver failed to converge"))?;

        if self.fit_intercept {
            self.intercept = params[0];
            self.coef = Some(params.slice(s![1..]).to_owned());
        } else {
            self.intercept = 0.0;
            self.coef = Some(params);
        }
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Poisson model is not fitted"))?;
        let x = to_array2(&x);
        let eta = x.dot(coef) + self.intercept;
        let mu = eta.mapv(|v| v.exp());
        Ok(pyarray1_from_f64(py, &mu))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Poisson model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let mu = (x.dot(coef) + self.intercept).mapv(|v| v.exp());
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = fisher_cov_poisson(&design, &mu).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept, coef, intercept_se, coef_se) = if self.fit_intercept {
            (
                self.intercept,
                coef.to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, coef.to_owned(), None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, &coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((
            n_bootstrap,
            x.ncols() + if self.fit_intercept { 1 } else { 0 },
        ));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let mut coef = Array1::<f64>::zeros(xb.ncols());
            if self.fit_intercept {
                let mean_y = yb.mean().unwrap_or(0.0).max(1e-12);
                coef = concatenate(Axis(0), &[array![mean_y.ln()].view(), coef.view()])
                    .map_err(|_| PyValueError::new_err("failed to init coefficients"))?;
            }
            let problem = PoissonProblem {
                x: xb.view(),
                y: yb.view(),
                fit_intercept: self.fit_intercept,
                alpha: self.alpha,
            };
            let linesearch = MoreThuenteLineSearch::new();
            let solver = NewtonCG::new(linesearch)
                .with_tolerance(self.tolerance)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let mut result = Executor::new(problem, solver)
                .configure(|state| state.param(coef).max_iters(self.max_iterations as u64))
                .run()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let params = result
                .state
                .take_best_param()
                .ok_or_else(|| PyValueError::new_err("solver failed to converge"))?;
            if self.fit_intercept {
                out[[i, 0]] = params[0];
                out.row_mut(i)
                    .slice_mut(s![1..])
                    .assign(&params.slice(s![1..]));
            } else {
                out.row_mut(i).assign(&params);
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct MEstimator {
    max_iterations: usize,
    tolerance: f64,
    objective_fn: Option<Py<PyAny>>,
    score_fn: Option<Py<PyAny>>,
    theta: Option<Array1<f64>>,
    data: Option<Py<PyAny>>,
    vcov: Option<Array2<f64>>,
}

struct MEstimatorProblem {
    objective_fn: Py<PyAny>,
    data: Py<PyAny>,
}

impl CostFunction for MEstimatorProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        Python::with_gil(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            let result = self
                .objective_fn
                .call1(py, (theta_py, self.data.clone_ref(py)))
                .map_err(|e| argmin::core::Error::msg(format!("Python callback error: {}", e)))?;

            let tuple = result
                .downcast_bound::<pyo3::types::PyTuple>(py)
                .map_err(|_| {
                    argmin::core::Error::msg("Objective function must return (obj, grad)")
                })?;

            if tuple.len() != 2 {
                return Err(argmin::core::Error::msg(
                    "Objective function must return (obj, grad)",
                ));
            }

            let obj_value: f64 = tuple.get_item(0)?.extract().map_err(|e| {
                argmin::core::Error::msg(format!("Failed to extract objective: {}", e))
            })?;

            Ok(obj_value)
        })
    }
}

impl Gradient for MEstimatorProblem {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(
        &self,
        theta: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        Python::with_gil(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            let result = self
                .objective_fn
                .call1(py, (theta_py, self.data.clone_ref(py)))
                .map_err(|e| argmin::core::Error::msg(format!("Python callback error: {}", e)))?;

            let tuple = result
                .downcast_bound::<pyo3::types::PyTuple>(py)
                .map_err(|_| {
                    argmin::core::Error::msg("Objective function must return (obj, grad)")
                })?;

            if tuple.len() != 2 {
                return Err(argmin::core::Error::msg(
                    "Objective function must return (obj, grad)",
                ));
            }

            let grad_item = tuple.get_item(1)?;
            let grad_py = grad_item
                .downcast::<PyArray1<f64>>()
                .map_err(|_| argmin::core::Error::msg("Gradient must be a numpy array"))?;

            let grad = to_array1(&grad_py.readonly());
            Ok(grad)
        })
    }
}

#[pymethods]
impl MEstimator {
    #[new]
    #[pyo3(signature = (objective_fn, score_fn, max_iterations=100, tolerance=1e-6))]
    fn new(
        objective_fn: Py<PyAny>,
        score_fn: Py<PyAny>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            max_iterations,
            tolerance,
            objective_fn: Some(objective_fn),
            score_fn: Some(score_fn),
            theta: None,
            data: None,
            vcov: None,
        }
    }

    fn fit(&mut self, py: Python, data: Py<PyAny>, theta0: PyReadonlyArray1<f64>) -> PyResult<()> {
        let theta_init = to_array1(&theta0);
        let objective_fn = self
            .objective_fn
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("objective_fn not set"))?
            .clone_ref(py);

        let problem = MEstimatorProblem {
            objective_fn,
            data: data.clone_ref(py),
        };

        let linesearch = MoreThuenteLineSearch::new();
        let solver = LBFGS::new(linesearch, 7);

        let mut result = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(theta_init)
                    .max_iters(self.max_iterations as u64)
            })
            .run()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let theta = result
            .state
            .take_best_param()
            .ok_or_else(|| PyValueError::new_err("optimization failed to converge"))?;

        self.theta = Some(theta);
        self.data = Some(data);
        self.vcov = None;

        Ok(())
    }

    fn compute_vcov(&mut self, py: Python) -> PyResult<()> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No data stored"))?;
        let score_fn = self
            .score_fn
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("score_fn not set"))?;

        let theta_py = pyarray1_from_f64(py, theta);
        let scores_result = score_fn
            .call1(py, (theta_py, data.clone_ref(py)))
            .map_err(|e| PyValueError::new_err(format!("score_fn error: {}", e)))?;

        let scores_py = scores_result
            .downcast_bound::<PyArray2<f64>>(py)
            .map_err(|_| PyValueError::new_err("score_fn must return 2D numpy array"))?;

        let scores = to_array2(&scores_py.readonly());
        let n = scores.nrows();
        let k = scores.ncols();

        if k != theta.len() {
            return Err(PyValueError::new_err(format!(
                "score dimension {} does not match theta dimension {}",
                k,
                theta.len()
            )));
        }

        let b_matrix = scores.t().dot(&scores) / (n as f64);
        let a_matrix = b_matrix.clone();
        let a_inv = invert_matrix(&a_matrix).map_err(PyValueError::new_err)?;
        let vcov = a_inv.dot(&b_matrix).dot(&a_inv) / (n as f64);

        self.vcov = Some(vcov);
        Ok(())
    }

    fn summary<'py>(&mut self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        if self.theta.is_none() {
            return Err(PyValueError::new_err("Model not fitted"));
        }

        if self.vcov.is_none() {
            self.compute_vcov(py)?;
        }

        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;

        let vcov = self
            .vcov
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Failed to compute vcov"))?;

        let se = diag_sqrt(vcov);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, theta))?;
        dict.set_item("se", pyarray1_from_f64(py, &se))?;
        Ok(dict.into())
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;
        let data = self
            .data
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No data stored"))?;
        let objective_fn = self
            .objective_fn
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("objective_fn not set"))?;

        let data_dict = data
            .downcast_bound::<pyo3::types::PyDict>(py)
            .map_err(|_| {
                PyValueError::new_err("data must be a dict with 'indices' key for bootstrap")
            })?;

        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        let n_key = pyo3::intern!(py, "n");
        let n: usize = data_dict
            .get_item(n_key)?
            .ok_or_else(|| PyValueError::new_err("data dict must have 'n' key"))?
            .extract()?;

        let mut out = Array2::<f64>::zeros((n_bootstrap, theta.len()));

        for i in 0..n_bootstrap {
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let indices_py = PyArray1::from_vec(py, indices);

            let boot_data = pyo3::types::PyDict::new(py);
            for (key, value) in data_dict.iter() {
                boot_data.set_item(key, value)?;
            }
            boot_data.set_item(pyo3::intern!(py, "indices"), indices_py)?;

            let problem = MEstimatorProblem {
                objective_fn: objective_fn.clone_ref(py),
                data: boot_data.into(),
            };

            let linesearch = MoreThuenteLineSearch::new();
            let solver = LBFGS::new(linesearch, 7);

            let mut result = Executor::new(problem, solver)
                .configure(|state| {
                    state
                        .param(theta.clone())
                        .max_iters(self.max_iterations as u64)
                })
                .run()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

            let theta_boot = result
                .state
                .take_best_param()
                .ok_or_else(|| PyValueError::new_err("bootstrap optimization failed"))?;

            out.row_mut(i).assign(&theta_boot);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}
