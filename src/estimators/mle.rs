use crate::fit::FitDiagnostics;
use crate::hyptests::wald_test_arrays;
use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, fisher_cov_binary, fisher_cov_multinomial,
    fisher_cov_poisson, invert_matrix, pyarray1_from_f64, pyarray1_from_i32, pyarray2_from_f64,
    qmle_cov_poisson, take_rows, take_rows_i32, take_rows_vec, to_array1, to_array1_i32, to_array2,
};
use crate::validation::{validate_finite, validate_nonnegative};
use argmin::core::{CostFunction, Executor, Gradient, Hessian, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::newton::NewtonCG;
use argmin::solver::quasinewton::LBFGS;
use ndarray::{array, concatenate, s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};
use std::collections::BTreeSet;

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn softmax_rows(logits: &Array2<f64>) -> Array2<f64> {
    let mut out = logits.clone();
    for mut row in out.outer_iter_mut() {
        let max = row.iter().fold(f64::NEG_INFINITY, |m, v| m.max(*v));
        row.mapv_inplace(|v| (v - max).exp());
        let denom = row.sum().max(1e-12);
        row.mapv_inplace(|v| v / denom);
    }
    out
}

fn validate_logistic_configuration(
    alpha: f64,
    max_iterations: u64,
    gradient_tolerance: f64,
) -> Result<(), String> {
    if !alpha.is_finite() || alpha < 0.0 {
        return Err("alpha must be finite and nonnegative".to_string());
    }
    if max_iterations == 0 {
        return Err("max_iterations must be positive".to_string());
    }
    if !gradient_tolerance.is_finite() || gradient_tolerance <= 0.0 {
        return Err("gradient_tolerance must be positive and finite".to_string());
    }
    Ok(())
}

#[derive(Clone)]
struct BinaryLogitFit {
    coef: Array1<f64>,
    intercept: f64,
    diagnostics: FitDiagnostics,
}

struct BinaryLogitProblem<'a> {
    x: ArrayView2<'a, f64>,
    y: ArrayView1<'a, i32>,
    fit_intercept: bool,
    alpha: f64,
}

impl BinaryLogitProblem<'_> {
    fn split_params<'a>(&self, params: &'a Array1<f64>) -> (ArrayView1<'a, f64>, f64) {
        let n_features = self.x.ncols();
        let coef = params.slice(s![..n_features]);
        let intercept = if self.fit_intercept {
            params[n_features]
        } else {
            0.0
        };
        (coef, intercept)
    }
}

impl CostFunction for BinaryLogitProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let (coef, intercept) = self.split_params(params);
        let mut objective = 0.5 * self.alpha * coef.dot(&coef);
        for i in 0..self.x.nrows() {
            let eta = self.x.row(i).dot(&coef) + intercept;
            objective += softplus(eta) - f64::from(self.y[i]) * eta;
        }
        Ok(objective)
    }
}

impl Gradient for BinaryLogitProblem<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let n_features = self.x.ncols();
        let (coef, intercept) = self.split_params(params);
        let mut gradient = Array1::<f64>::zeros(params.len());
        for i in 0..self.x.nrows() {
            let eta = self.x.row(i).dot(&coef) + intercept;
            let residual = sigmoid(eta) - f64::from(self.y[i]);
            for j in 0..n_features {
                gradient[j] += self.x[[i, j]] * residual;
            }
            if self.fit_intercept {
                gradient[n_features] += residual;
            }
        }
        for j in 0..n_features {
            gradient[j] += self.alpha * coef[j];
        }
        Ok(gradient)
    }
}

fn fit_binary_logit(
    x: &Array2<f64>,
    y: &Array1<i32>,
    fit_intercept: bool,
    alpha: f64,
    max_iterations: u64,
    gradient_tolerance: f64,
) -> Result<BinaryLogitFit, String> {
    validate_logistic_configuration(alpha, max_iterations, gradient_tolerance)?;
    validate_finite("x", x)?;
    if x.nrows() != y.len() {
        return Err("x rows must match y length".to_string());
    }
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("x must contain at least one row and one column".to_string());
    }
    if y.iter().any(|value| !matches!(value, 0 | 1)) {
        return Err("Logit labels must contain only 0 and 1".to_string());
    }
    if !y.iter().any(|value| *value == 0) || !y.iter().any(|value| *value == 1) {
        return Err("Logit requires observations from both outcome classes".to_string());
    }

    let problem = BinaryLogitProblem {
        x: x.view(),
        y: y.view(),
        fit_intercept,
        alpha,
    };
    let solver = LBFGS::new(MoreThuenteLineSearch::new(), 10)
        .with_tolerance_grad(gradient_tolerance)
        .map_err(|err| err.to_string())?;
    let initial = Array1::<f64>::zeros(x.ncols() + usize::from(fit_intercept));
    let mut result = Executor::new(problem, solver)
        .configure(|state| state.param(initial).max_iters(max_iterations))
        .run()
        .map_err(|err| err.to_string())?;
    let diagnostics = FitDiagnostics::from_argmin(
        result.state.get_termination_status(),
        result.state.get_iter(),
        Some(result.state.get_best_cost()),
    );
    diagnostics.require_converged("Logit")?;
    let params = result
        .state
        .take_best_param()
        .ok_or_else(|| "Logit solver returned no parameters".to_string())?;
    let n_features = x.ncols();
    Ok(BinaryLogitFit {
        coef: params.slice(s![..n_features]).to_owned(),
        intercept: if fit_intercept {
            params[n_features]
        } else {
            0.0
        },
        diagnostics,
    })
}

#[derive(Clone)]
struct MultinomialLogitFit {
    coef: Array2<f64>,
    intercept: Array1<f64>,
    classes: Vec<i32>,
    diagnostics: FitDiagnostics,
}

struct MultinomialLogitProblem<'a> {
    x: ArrayView2<'a, f64>,
    class_index: &'a [usize],
    n_classes: usize,
    fit_intercept: bool,
    alpha: f64,
}

impl MultinomialLogitProblem<'_> {
    fn parameters_per_class(&self) -> usize {
        self.x.ncols() + usize::from(self.fit_intercept)
    }

    fn class_intercept(&self, params: &Array1<f64>, class: usize) -> f64 {
        if self.fit_intercept {
            params[class * self.parameters_per_class()]
        } else {
            0.0
        }
    }

    fn class_coef<'a>(&self, params: &'a Array1<f64>, class: usize) -> ArrayView1<'a, f64> {
        let offset = class * self.parameters_per_class() + usize::from(self.fit_intercept);
        params.slice(s![offset..offset + self.x.ncols()])
    }

    fn row_logits(&self, params: &Array1<f64>, row: usize) -> Array1<f64> {
        let mut logits = Array1::<f64>::zeros(self.n_classes);
        for class in 0..self.n_classes {
            logits[class] = self.x.row(row).dot(&self.class_coef(params, class))
                + self.class_intercept(params, class);
        }
        logits
    }
}

impl CostFunction for MultinomialLogitProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut objective = 0.0;
        for row in 0..self.x.nrows() {
            let logits = self.row_logits(params, row);
            let max_logit = logits
                .iter()
                .fold(f64::NEG_INFINITY, |current, value| current.max(*value));
            let log_denom = max_logit
                + logits
                    .iter()
                    .map(|value| (*value - max_logit).exp())
                    .sum::<f64>()
                    .ln();
            objective += log_denom - logits[self.class_index[row]];
        }
        for class in 0..self.n_classes {
            let coef = self.class_coef(params, class);
            objective += 0.5 * self.alpha * coef.dot(&coef);
        }
        Ok(objective)
    }
}

impl Gradient for MultinomialLogitProblem<'_> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>;

    fn gradient(&self, params: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        let block_size = self.parameters_per_class();
        let mut gradient = Array1::<f64>::zeros(params.len());
        for row in 0..self.x.nrows() {
            let logits = self.row_logits(params, row);
            let max_logit = logits
                .iter()
                .fold(f64::NEG_INFINITY, |current, value| current.max(*value));
            let mut probabilities = logits.mapv(|value| (value - max_logit).exp());
            probabilities /= probabilities.sum();
            for class in 0..self.n_classes {
                let residual = probabilities[class]
                    - if self.class_index[row] == class {
                        1.0
                    } else {
                        0.0
                    };
                let block_offset = class * block_size;
                if self.fit_intercept {
                    gradient[block_offset] += residual;
                }
                let coef_offset = block_offset + usize::from(self.fit_intercept);
                for feature in 0..self.x.ncols() {
                    gradient[coef_offset + feature] += self.x[[row, feature]] * residual;
                }
            }
        }
        for class in 0..self.n_classes {
            let coef_offset = class * block_size + usize::from(self.fit_intercept);
            for feature in 0..self.x.ncols() {
                gradient[coef_offset + feature] += self.alpha * params[coef_offset + feature];
            }
        }
        Ok(gradient)
    }
}

fn fit_multinomial_logit(
    x: &Array2<f64>,
    y: &Array1<i32>,
    fit_intercept: bool,
    alpha: f64,
    max_iterations: u64,
    gradient_tolerance: f64,
) -> Result<MultinomialLogitFit, String> {
    validate_logistic_configuration(alpha, max_iterations, gradient_tolerance)?;
    validate_finite("x", x)?;
    if x.nrows() != y.len() {
        return Err("x rows must match y length".to_string());
    }
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("x must contain at least one row and one column".to_string());
    }
    let classes: Vec<i32> = y
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if classes.len() < 2 {
        return Err("MultinomialLogit requires at least two outcome classes".to_string());
    }
    let class_index: Vec<usize> = y
        .iter()
        .map(|label| {
            classes
                .binary_search(label)
                .expect("classes were constructed from labels")
        })
        .collect();
    let n_classes = classes.len();
    let block_size = x.ncols() + usize::from(fit_intercept);
    let initial = Array1::<f64>::zeros(block_size * n_classes);
    let problem = MultinomialLogitProblem {
        x: x.view(),
        class_index: &class_index,
        n_classes,
        fit_intercept,
        alpha,
    };
    let solver = LBFGS::new(MoreThuenteLineSearch::new(), 10)
        .with_tolerance_grad(gradient_tolerance)
        .map_err(|err| err.to_string())?;
    let mut result = Executor::new(problem, solver)
        .configure(|state| state.param(initial).max_iters(max_iterations))
        .run()
        .map_err(|err| err.to_string())?;
    let diagnostics = FitDiagnostics::from_argmin(
        result.state.get_termination_status(),
        result.state.get_iter(),
        Some(result.state.get_best_cost()),
    );
    diagnostics.require_converged("MultinomialLogit")?;
    let params = result
        .state
        .take_best_param()
        .ok_or_else(|| "MultinomialLogit solver returned no parameters".to_string())?;

    let mut coef = Array2::<f64>::zeros((x.ncols(), n_classes));
    let mut intercept = Array1::<f64>::zeros(n_classes);
    for class in 0..n_classes {
        let block_offset = class * block_size;
        if fit_intercept {
            intercept[class] = params[block_offset];
        }
        let coef_offset = block_offset + usize::from(fit_intercept);
        coef.column_mut(class)
            .assign(&params.slice(s![coef_offset..coef_offset + x.ncols()]));
    }
    Ok(MultinomialLogitFit {
        coef,
        intercept,
        classes,
        diagnostics,
    })
}

fn multinomial_logits(x: &Array2<f64>, model: &MultinomialLogitFit) -> Array2<f64> {
    let mut logits = x.dot(&model.coef);
    for class in 0..logits.ncols() {
        logits
            .column_mut(class)
            .mapv_inplace(|value| value + model.intercept[class]);
    }
    logits
}

#[pyclass]
pub struct Logit {
    alpha: f64,
    fit_intercept: bool,
    max_iterations: u64,
    gradient_tolerance: f64,
    model: Option<BinaryLogitFit>,
    x: Option<Array2<f64>>,
    y: Option<Array1<i32>>,
}

#[pymethods]
impl Logit {
    #[new]
    #[pyo3(signature = (alpha=0.0, max_iterations=100, gradient_tolerance=1e-4))]
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
        self.model = None;
        self.x = None;
        self.y = None;
        let x = to_array2(&x);
        let y = to_array1_i32(&y);
        let model = fit_binary_logit(
            &x,
            &y,
            self.fit_intercept,
            self.alpha,
            self.max_iterations,
            self.gradient_tolerance,
        )
        .map_err(PyValueError::new_err)?;
        self.model = Some(model);
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn predict_lin<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Logit model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != model.coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let eta = x.dot(&model.coef) + model.intercept;
        Ok(pyarray1_from_f64(py, &eta))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Logit model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != model.coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let eta = x.dot(&model.coef) + model.intercept;
        let probs = eta.mapv(sigmoid);
        Ok(pyarray1_from_f64(py, &probs))
    }

    #[pyo3(signature = (x, cutoff=0.5))]
    fn predict_label<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        cutoff: f64,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Logit model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != model.coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let eta = x.dot(&model.coef) + model.intercept;
        let pred = eta.mapv(|v| if sigmoid(v) >= cutoff { 1 } else { 0 });
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

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", model.intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, &model.coef))?;
        dict.set_item("penalty", self.alpha)?;
        dict.set_item("inference_available", self.alpha == 0.0)?;
        model.diagnostics.write_summary(&dict)?;

        if self.alpha == 0.0 {
            let probs = (x.dot(&model.coef) + model.intercept).mapv(sigmoid);
            let design = if self.fit_intercept {
                add_intercept(x)
            } else {
                x.clone()
            };
            let cov = fisher_cov_binary(&design, &probs).map_err(PyValueError::new_err)?;
            let se_all = diag_sqrt(&cov).map_err(PyValueError::new_err)?;
            if self.fit_intercept {
                dict.set_item("intercept_se", se_all[0])?;
                dict.set_item(
                    "coef_se",
                    pyarray1_from_f64(py, &se_all.slice(s![1..]).to_owned()),
                )?;
            } else {
                dict.set_item("intercept_se", py.None())?;
                dict.set_item("coef_se", pyarray1_from_f64(py, &se_all))?;
            }
            dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        } else {
            dict.set_item("intercept_se", py.None())?;
            dict.set_item("coef_se", py.None())?;
            dict.set_item("vcov", py.None())?;
        }
        Ok(dict.into())
    }

    #[pyo3(signature = (r, q=None))]
    fn wald_test<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<f64>,
        q: Option<PyReadonlyArray1<f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.alpha != 0.0 {
            return Err(PyValueError::new_err(
                "Wald inference is only available for unpenalized Logit fits (alpha=0)",
            ));
        }
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Logit model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let probs = (x.dot(&model.coef) + model.intercept).mapv(sigmoid);
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = fisher_cov_binary(&design, &probs).map_err(PyValueError::new_err)?;
        let mut params =
            Array1::<f64>::zeros(model.coef.len() + if self.fit_intercept { 1 } else { 0 });
        if self.fit_intercept {
            params[0] = model.intercept;
            params.slice_mut(s![1..]).assign(&model.coef);
        } else {
            params.assign(&model.coef);
        }
        let rmat = to_array2(&r);
        let qvec = q.as_ref().map(to_array1);
        wald_test_arrays(py, &params, &cov, &rmat, qvec.as_ref())
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
            let model = fit_binary_logit(
                &xb,
                &yb,
                self.fit_intercept,
                self.alpha,
                self.max_iterations,
                self.gradient_tolerance,
            )
            .map_err(|err| {
                PyValueError::new_err(format!("Logit bootstrap replicate {i} failed: {err}"))
            })?;
            if self.fit_intercept {
                out[[i, 0]] = model.intercept;
                out.row_mut(i).slice_mut(s![1..]).assign(&model.coef);
            } else {
                out.row_mut(i).assign(&model.coef);
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
    model: Option<MultinomialLogitFit>,
    x: Option<Array2<f64>>,
    y: Option<Array1<i32>>,
}

#[pymethods]
impl MultinomialLogit {
    #[new]
    #[pyo3(signature = (alpha=0.0, max_iterations=100, gradient_tolerance=1e-4))]
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
        self.model = None;
        self.x = None;
        self.y = None;
        let x = to_array2(&x);
        let y = to_array1_i32(&y);
        let model = fit_multinomial_logit(
            &x,
            &y,
            self.fit_intercept,
            self.alpha,
            self.max_iterations,
            self.gradient_tolerance,
        )
        .map_err(PyValueError::new_err)?;
        self.model = Some(model);
        self.x = Some(x);
        self.y = Some(y);
        Ok(())
    }

    fn predict_lin<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MultinomialLogit model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != model.coef.nrows() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let logits = multinomial_logits(&x, model);
        Ok(pyarray2_from_f64(py, &logits))
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MultinomialLogit model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != model.coef.nrows() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let logits = multinomial_logits(&x, model);
        let probs = softmax_rows(&logits);
        Ok(pyarray2_from_f64(py, &probs))
    }

    fn predict_label<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<i32>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("MultinomialLogit model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != model.coef.nrows() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let logits = multinomial_logits(&x, model);
        let pred = Array1::from_iter(logits.outer_iter().map(|row| {
            let mut best_index = 0;
            let mut best_value = row[0];
            for class in 1..row.len() {
                if row[class] > best_value {
                    best_index = class;
                    best_value = row[class];
                }
            }
            model.classes[best_index]
        }));
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

        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let k = design.ncols();
        let c = model.classes.len();
        let reference_index = c - 1;
        let mut raw_coef = Array2::<f64>::zeros((c, k));
        let params = &model.coef;
        let intercept = &model.intercept;

        for class in 0..c {
            for j in 0..k {
                if self.fit_intercept {
                    if j == 0 {
                        raw_coef[[class, j]] = intercept[class];
                    } else {
                        raw_coef[[class, j]] = params[[j - 1, class]];
                    }
                } else {
                    raw_coef[[class, j]] = params[[j, class]];
                }
            }
        }

        let mut coef = Array2::<f64>::zeros((c - 1, k));
        for class in 0..(c - 1) {
            let contrast = &raw_coef.row(class) - &raw_coef.row(reference_index);
            coef.row_mut(class).assign(&contrast);
        }

        let classes = Array1::from_vec(model.classes.clone());
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray2_from_f64(py, &coef))?;
        dict.set_item(
            "class_labels",
            pyarray1_from_i32(py, &classes.slice(s![..reference_index]).to_owned()),
        )?;
        dict.set_item("reference_class", classes[reference_index])?;
        dict.set_item("penalty", self.alpha)?;
        dict.set_item("inference_available", self.alpha == 0.0)?;
        model.diagnostics.write_summary(&dict)?;

        if self.alpha == 0.0 {
            let probs = softmax_rows(&multinomial_logits(x, model));
            let cov = fisher_cov_multinomial(&design, &probs, reference_index)
                .map_err(PyValueError::new_err)?;
            let se_all = diag_sqrt(&cov).map_err(PyValueError::new_err)?;
            let se = Array2::from_shape_vec((c - 1, k), se_all.to_vec())
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            dict.set_item("se", pyarray2_from_f64(py, &se))?;
            dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        } else {
            dict.set_item("se", py.None())?;
            dict.set_item("vcov", py.None())?;
        }
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
            .classes
            .len();
        let mut out = Array2::<f64>::zeros((n_bootstrap, k * (c - 1)));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_i32(y, idx);
            let model = fit_multinomial_logit(
                &xb,
                &yb,
                self.fit_intercept,
                self.alpha,
                self.max_iterations,
                self.gradient_tolerance,
            )
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "MultinomialLogit bootstrap replicate {i} failed: {err}"
                ))
            })?;
            if model.classes.len() != c {
                return Err(PyValueError::new_err(
                    "bootstrap sample omitted at least one outcome class",
                ));
            }
            let reference_index = c - 1;
            for class in 0..reference_index {
                let offset = class * k;
                if self.fit_intercept {
                    out[[i, offset]] = model.intercept[class] - model.intercept[reference_index];
                    let contrast = &model.coef.column(class) - &model.coef.column(reference_index);
                    out.row_mut(i)
                        .slice_mut(s![offset + 1..offset + k])
                        .assign(&contrast);
                } else {
                    let contrast = &model.coef.column(class) - &model.coef.column(reference_index);
                    out.row_mut(i)
                        .slice_mut(s![offset..offset + k])
                        .assign(&contrast);
                }
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
    diagnostics: Option<FitDiagnostics>,
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
            diagnostics: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.coef = None;
        self.x = None;
        self.y = None;
        self.diagnostics = None;
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        if !self.alpha.is_finite() || self.alpha < 0.0 {
            return Err(PyValueError::new_err(
                "alpha must be finite and nonnegative",
            ));
        }
        if self.max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be finite and positive",
            ));
        }
        validate_finite("x", &x).map_err(PyValueError::new_err)?;
        validate_nonnegative("y", &y).map_err(PyValueError::new_err)?;
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
        let diagnostics = FitDiagnostics::from_argmin(
            result.state.get_termination_status(),
            result.state.get_iter(),
            Some(result.state.get_best_cost()),
        );
        diagnostics
            .require_converged("Poisson")
            .map_err(PyValueError::new_err)?;
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
        self.diagnostics = Some(diagnostics);
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
            .ok_or_else(|| PyValueError::new_err("Poisson model is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let eta = x.dot(coef) + self.intercept;
        Ok(pyarray1_from_f64(py, &eta))
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
        if x.ncols() != coef.len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let eta = x.dot(coef) + self.intercept;
        let mu = eta.mapv(|v| v.exp());
        Ok(pyarray1_from_f64(py, &mu))
    }

    #[pyo3(signature = (vcov="vanilla"))]
    fn summary<'py>(&self, py: Python<'py>, vcov: &str) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Poisson model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("penalty", self.alpha)?;
        self.diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?
            .write_summary(&dict)?;
        dict.set_item("inference_available", self.alpha == 0.0)?;

        if !matches!(vcov, "vanilla" | "sandwich" | "qmle") {
            return Err(PyValueError::new_err(
                "vcov must be one of {'vanilla', 'sandwich', 'qmle'}",
            ));
        }
        if self.alpha == 0.0 {
            let mu = (x.dot(coef) + self.intercept).mapv(|v| v.exp());
            let design = if self.fit_intercept {
                add_intercept(x)
            } else {
                x.clone()
            };
            let cov = match vcov {
                "vanilla" => fisher_cov_poisson(&design, &mu).map_err(PyValueError::new_err)?,
                "sandwich" | "qmle" => {
                    qmle_cov_poisson(&design, y, &mu).map_err(PyValueError::new_err)?
                }
                _ => unreachable!(),
            };
            let se_all = diag_sqrt(&cov).map_err(PyValueError::new_err)?;
            if self.fit_intercept {
                dict.set_item("intercept_se", se_all[0])?;
                dict.set_item(
                    "coef_se",
                    pyarray1_from_f64(py, &se_all.slice(s![1..]).to_owned()),
                )?;
            } else {
                dict.set_item("intercept_se", py.None())?;
                dict.set_item("coef_se", pyarray1_from_f64(py, &se_all))?;
            }
            dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
            dict.set_item("vcov_type", if vcov == "qmle" { "sandwich" } else { vcov })?;
        } else {
            dict.set_item("intercept_se", py.None())?;
            dict.set_item("coef_se", py.None())?;
            dict.set_item("vcov", py.None())?;
            dict.set_item("vcov_type", py.None())?;
        }
        Ok(dict.into())
    }

    #[pyo3(signature = (r, q=None, vcov="vanilla"))]
    fn wald_test<'py>(
        &self,
        py: Python<'py>,
        r: PyReadonlyArray2<f64>,
        q: Option<PyReadonlyArray1<f64>>,
        vcov: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        if self.alpha != 0.0 {
            return Err(PyValueError::new_err(
                "Wald inference is only available for unpenalized Poisson fits (alpha=0)",
            ));
        }
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Poisson model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No response stored"))?;
        let eta = x.dot(coef) + self.intercept;
        let mu = eta.mapv(|v| v.exp());
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = match vcov {
            "vanilla" => fisher_cov_poisson(&design, &mu).map_err(PyValueError::new_err)?,
            "sandwich" | "qmle" => {
                qmle_cov_poisson(&design, y, &mu).map_err(PyValueError::new_err)?
            }
            _ => {
                return Err(PyValueError::new_err(
                    "vcov must be one of {'vanilla', 'sandwich', 'qmle'}",
                ));
            }
        };
        let mut params = Array1::<f64>::zeros(coef.len() + if self.fit_intercept { 1 } else { 0 });
        if self.fit_intercept {
            params[0] = self.intercept;
            params.slice_mut(s![1..]).assign(coef);
        } else {
            params.assign(coef);
        }
        let rmat = to_array2(&r);
        let qvec = q.as_ref().map(to_array1);
        wald_test_arrays(py, &params, &cov, &rmat, qvec.as_ref())
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
            let diagnostics = FitDiagnostics::from_argmin(
                result.state.get_termination_status(),
                result.state.get_iter(),
                Some(result.state.get_best_cost()),
            );
            diagnostics
                .require_converged(&format!("Poisson bootstrap replicate {i}"))
                .map_err(PyValueError::new_err)?;
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
    derivative_step: f64,
    score_fn: Option<Py<PyAny>>,
    theta: Option<Array1<f64>>,
    data: Option<Py<PyAny>>,
    vcov: Option<Array2<f64>>,
    diagnostics: Option<FitDiagnostics>,
}

struct MEstimatorProblem {
    objective_fn: Py<PyAny>,
    data: Py<PyAny>,
}

impl CostFunction for MEstimatorProblem {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, theta: &Self::Param) -> std::result::Result<Self::Output, argmin::core::Error> {
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            let result = self
                .objective_fn
                .call1(py, (theta_py, self.data.clone_ref(py)))
                .map_err(|e| argmin::core::Error::msg(format!("Python callback error: {}", e)))?;

            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py).map_err(|_| {
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
        Python::attach(|py| {
            let theta_py = pyarray1_from_f64(py, theta);
            let result = self
                .objective_fn
                .call1(py, (theta_py, self.data.clone_ref(py)))
                .map_err(|e| argmin::core::Error::msg(format!("Python callback error: {}", e)))?;

            let tuple = result.cast_bound::<pyo3::types::PyTuple>(py).map_err(|_| {
                argmin::core::Error::msg("Objective function must return (obj, grad)")
            })?;

            if tuple.len() != 2 {
                return Err(argmin::core::Error::msg(
                    "Objective function must return (obj, grad)",
                ));
            }

            let grad_item = tuple.get_item(1)?;
            let grad_py = grad_item
                .cast::<PyArray1<f64>>()
                .map_err(|_| argmin::core::Error::msg("Gradient must be a numpy array"))?;

            let grad = to_array1(&grad_py.readonly());
            Ok(grad)
        })
    }
}

fn call_mestimator_scores(
    py: Python,
    score_fn: &Py<PyAny>,
    data: &Py<PyAny>,
    theta: &Array1<f64>,
) -> PyResult<Array2<f64>> {
    let theta_py = pyarray1_from_f64(py, theta);
    let result = score_fn
        .call1(py, (theta_py, data.clone_ref(py)))
        .map_err(|err| PyValueError::new_err(format!("score_fn error: {}", err)))?;
    let scores_py = result
        .cast_bound::<PyArray2<f64>>(py)
        .map_err(|_| PyValueError::new_err("score_fn must return a 2D numpy array"))?;
    let scores = to_array2(&scores_py.readonly());
    if scores.nrows() == 0 {
        return Err(PyValueError::new_err("score_fn returned no observations"));
    }
    Ok(scores)
}

#[pymethods]
impl MEstimator {
    #[new]
    #[pyo3(signature = (objective_fn, score_fn, max_iterations=100, tolerance=1e-6, derivative_step=1e-6))]
    fn new(
        objective_fn: Py<PyAny>,
        score_fn: Py<PyAny>,
        max_iterations: usize,
        tolerance: f64,
        derivative_step: f64,
    ) -> Self {
        Self {
            max_iterations,
            tolerance,
            objective_fn: Some(objective_fn),
            derivative_step,
            score_fn: Some(score_fn),
            theta: None,
            data: None,
            vcov: None,
            diagnostics: None,
        }
    }

    fn fit(&mut self, py: Python, data: Py<PyAny>, theta0: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.theta = None;
        self.data = None;
        self.vcov = None;
        self.diagnostics = None;
        let theta_init = to_array1(&theta0);
        validate_finite("theta0", &theta_init).map_err(PyValueError::new_err)?;
        if self.max_iterations == 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be positive and finite",
            ));
        }
        if !self.derivative_step.is_finite() || self.derivative_step <= 0.0 {
            return Err(PyValueError::new_err(
                "derivative_step must be finite and positive",
            ));
        }
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
        let solver = LBFGS::new(linesearch, 7)
            .with_tolerance_grad(self.tolerance)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .with_tolerance_cost(self.tolerance)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let mut result = Executor::new(problem, solver)
            .configure(|state| {
                state
                    .param(theta_init)
                    .max_iters(self.max_iterations as u64)
            })
            .run()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let diagnostics = FitDiagnostics::from_argmin(
            result.state.get_termination_status(),
            result.state.get_iter(),
            Some(result.state.get_best_cost()),
        );
        diagnostics
            .require_converged("MEstimator")
            .map_err(PyValueError::new_err)?;
        let theta = result
            .state
            .take_best_param()
            .ok_or_else(|| PyValueError::new_err("optimization failed to converge"))?;

        self.theta = Some(theta);
        self.data = Some(data);
        self.diagnostics = Some(diagnostics);
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

        let scores = call_mestimator_scores(py, score_fn, data, theta)?;
        let n = scores.nrows();
        let k = scores.ncols();
        if k != theta.len() {
            return Err(PyValueError::new_err(format!(
                "score dimension {} does not match theta dimension {}",
                k,
                theta.len()
            )));
        }

        let mut a_matrix = Array2::<f64>::zeros((k, k));
        for column in 0..k {
            let step = self.derivative_step * theta[column].abs().max(1.0);
            let mut theta_plus = theta.clone();
            let mut theta_minus = theta.clone();
            theta_plus[column] += step;
            theta_minus[column] -= step;
            let scores_plus = call_mestimator_scores(py, score_fn, data, &theta_plus)?;
            let scores_minus = call_mestimator_scores(py, score_fn, data, &theta_minus)?;
            if scores_plus.raw_dim() != scores.raw_dim()
                || scores_minus.raw_dim() != scores.raw_dim()
            {
                return Err(PyValueError::new_err(
                    "score_fn output shape changed during numerical differentiation",
                ));
            }
            let mean_plus = scores_plus
                .mean_axis(Axis(0))
                .ok_or_else(|| PyValueError::new_err("score_fn returned no observations"))?;
            let mean_minus = scores_minus
                .mean_axis(Axis(0))
                .ok_or_else(|| PyValueError::new_err("score_fn returned no observations"))?;
            let derivative = (&mean_plus - &mean_minus) / (2.0 * step);
            a_matrix.column_mut(column).assign(&derivative);
        }

        let b_matrix = scores.t().dot(&scores) / (n as f64);
        let a_inv = invert_matrix(&a_matrix).map_err(PyValueError::new_err)?;
        let vcov_raw = a_inv.dot(&b_matrix).dot(&a_inv.t()) / (n as f64);
        let vcov = (&vcov_raw + &vcov_raw.t()) * 0.5;

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

        let se = diag_sqrt(vcov).map_err(PyValueError::new_err)?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, theta))?;
        dict.set_item("se", pyarray1_from_f64(py, &se))?;
        dict.set_item("vcov", pyarray2_from_f64(py, vcov))?;
        self.diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("fit diagnostics are unavailable"))?
            .write_summary(&dict)?;
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

        let data_dict = data.cast_bound::<pyo3::types::PyDict>(py).map_err(|_| {
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
            let solver = LBFGS::new(linesearch, 7)
                .with_tolerance_grad(self.tolerance)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
                .with_tolerance_cost(self.tolerance)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

            let mut result = Executor::new(problem, solver)
                .configure(|state| {
                    state
                        .param(theta.clone())
                        .max_iters(self.max_iterations as u64)
                })
                .run()
                .map_err(|err| PyValueError::new_err(err.to_string()))?;

            let diagnostics = FitDiagnostics::from_argmin(
                result.state.get_termination_status(),
                result.state.get_iter(),
                Some(result.state.get_best_cost()),
            );
            diagnostics
                .require_converged(&format!("MEstimator bootstrap replicate {i}"))
                .map_err(PyValueError::new_err)?;
            let theta_boot = result
                .state
                .take_best_param()
                .ok_or_else(|| PyValueError::new_err("bootstrap optimization failed"))?;

            out.row_mut(i).assign(&theta_boot);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}
