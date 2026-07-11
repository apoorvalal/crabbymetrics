use crate::hyptests::wald_test_arrays;
use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, fisher_cov_binary, fisher_cov_multinomial,
    fisher_cov_poisson, invert_matrix, pyarray1_from_f64, pyarray1_from_i32, pyarray2_from_f64,
    qmle_cov_poisson, take_rows, take_rows_i32, take_rows_vec, to_array1, to_array1_i32, to_array2,
};
use argmin::core::{
    CostFunction, Executor, Gradient, Hessian, State, TerminationReason, TerminationStatus,
};
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
use pyo3::types::PyDict;
use rand::{Rng, SeedableRng};

fn optimization_success(status: &TerminationStatus) -> bool {
    matches!(
        status,
        TerminationStatus::Terminated(TerminationReason::SolverConverged)
            | TerminationStatus::Terminated(TerminationReason::TargetCostReached)
    )
}

fn binary_logit_orientation(model: &linfa_logistic::FittedLogisticRegression<f64, i32>) -> f64 {
    if model.labels().pos.class == 1 {
        1.0
    } else {
        -1.0
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
        let x = to_array2(&x);
        let y = to_array1_i32(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        if y.iter().any(|value| !matches!(value, 0 | 1)) {
            return Err(PyValueError::new_err(
                "Logit labels must contain only 0 and 1",
            ));
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
        if x.ncols() != model.params().len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let orientation = binary_logit_orientation(model);
        let eta = (x.dot(model.params()) + model.intercept()) * orientation;
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
        if x.ncols() != model.params().len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let orientation = binary_logit_orientation(model);
        let eta = (x.dot(model.params()) + model.intercept()) * orientation;
        let probs = eta.mapv(|v| 1.0 / (1.0 + (-v).exp()));
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
        if x.ncols() != model.params().len() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let orientation = binary_logit_orientation(model);
        let eta = (x.dot(model.params()) + model.intercept()) * orientation;
        let pred = eta.mapv(|v| {
            if 1.0 / (1.0 + (-v).exp()) >= cutoff {
                1
            } else {
                0
            }
        });
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
        let orientation = binary_logit_orientation(model);
        let public_coef = model.params().mapv(|value| value * orientation);
        dict.set_item("intercept", model.intercept() * orientation)?;
        dict.set_item("coef", pyarray1_from_f64(py, &public_coef))?;
        dict.set_item("penalty", self.alpha)?;
        dict.set_item("inference_available", self.alpha == 0.0)?;

        if self.alpha == 0.0 {
            let probs = model.predict_probabilities(x);
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
        let probs = model.predict_probabilities(x);
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = fisher_cov_binary(&design, &probs).map_err(PyValueError::new_err)?;
        let orientation = binary_logit_orientation(model);
        let mut params =
            Array1::<f64>::zeros(model.params().len() + if self.fit_intercept { 1 } else { 0 });
        if self.fit_intercept {
            params[0] = model.intercept() * orientation;
            params
                .slice_mut(s![1..])
                .assign(&model.params().mapv(|value| value * orientation));
        } else {
            params.assign(&model.params().mapv(|value| value * orientation));
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
            let dataset = Dataset::new(xb, yb);
            let params = LinfaLogisticRegression::new()
                .alpha(self.alpha)
                .with_intercept(self.fit_intercept)
                .max_iterations(self.max_iterations)
                .gradient_tolerance(self.gradient_tolerance);
            let model = params
                .fit(&dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let orientation = binary_logit_orientation(&model);
            if self.fit_intercept {
                out[[i, 0]] = model.intercept() * orientation;
                out.row_mut(i)
                    .slice_mut(s![1..])
                    .assign(&model.params().mapv(|value| value * orientation));
            } else {
                out.row_mut(i)
                    .assign(&model.params().mapv(|value| value * orientation));
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
        if x.ncols() != model.params().nrows() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let mut logits = x.dot(model.params());
        for class in 0..logits.ncols() {
            logits
                .column_mut(class)
                .mapv_inplace(|v| v + model.intercept()[class]);
        }
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
        if x.ncols() != model.params().nrows() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
        let mut logits = x.dot(model.params());
        for class in 0..logits.ncols() {
            logits
                .column_mut(class)
                .mapv_inplace(|v| v + model.intercept()[class]);
        }
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
        if x.ncols() != model.params().nrows() {
            return Err(PyValueError::new_err(
                "x column count does not match fitted model",
            ));
        }
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

        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let k = design.ncols();
        let c = model.classes().len();
        let reference_index = c - 1;
        let mut raw_coef = Array2::<f64>::zeros((c, k));
        let params = model.params();
        let intercept = model.intercept();

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

        let classes = Array1::from_vec(model.classes().to_vec());
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray2_from_f64(py, &coef))?;
        dict.set_item(
            "class_labels",
            pyarray1_from_i32(py, &classes.slice(s![..reference_index]).to_owned()),
        )?;
        dict.set_item("reference_class", classes[reference_index])?;
        dict.set_item("penalty", self.alpha)?;
        dict.set_item("inference_available", self.alpha == 0.0)?;

        if self.alpha == 0.0 {
            let probs = model.predict_probabilities(x);
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
            .classes()
            .len();
        let mut out = Array2::<f64>::zeros((n_bootstrap, k * (c - 1)));
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
            if model.classes().len() != c {
                return Err(PyValueError::new_err(
                    "bootstrap sample omitted at least one outcome class",
                ));
            }
            let reference_index = c - 1;
            for class in 0..reference_index {
                let offset = class * k;
                if self.fit_intercept {
                    out[[i, offset]] =
                        model.intercept()[class] - model.intercept()[reference_index];
                    let contrast =
                        &model.params().column(class) - &model.params().column(reference_index);
                    out.row_mut(i)
                        .slice_mut(s![offset + 1..offset + k])
                        .assign(&contrast);
                } else {
                    let contrast =
                        &model.params().column(class) - &model.params().column(reference_index);
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
        if !optimization_success(result.state.get_termination_status()) {
            return Err(PyValueError::new_err(format!(
                "Poisson optimization did not converge: {}",
                result.state.get_termination_status()
            )));
        }
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
        dict.set_item("converged", true)?;
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
            if !optimization_success(result.state.get_termination_status()) {
                return Err(PyValueError::new_err(format!(
                    "Poisson bootstrap optimization did not converge: {}",
                    result.state.get_termination_status()
                )));
            }
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
    iterations: Option<usize>,
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
        .downcast_bound::<PyArray2<f64>>(py)
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
            iterations: None,
        }
    }

    fn fit(&mut self, py: Python, data: Py<PyAny>, theta0: PyReadonlyArray1<f64>) -> PyResult<()> {
        let theta_init = to_array1(&theta0);
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

        if !optimization_success(result.state.get_termination_status()) {
            return Err(PyValueError::new_err(format!(
                "M-estimator optimization did not converge: {}",
                result.state.get_termination_status()
            )));
        }
        let iterations = result.state.get_iter() as usize;
        let theta = result
            .state
            .take_best_param()
            .ok_or_else(|| PyValueError::new_err("optimization failed to converge"))?;

        self.theta = Some(theta);
        self.data = Some(data);
        self.vcov = None;

        self.iterations = Some(iterations);
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
        dict.set_item("converged", true)?;
        dict.set_item("iterations", self.iterations)?;
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

            if !optimization_success(result.state.get_termination_status()) {
                return Err(PyValueError::new_err(format!(
                    "M-estimator bootstrap optimization did not converge: {}",
                    result.state.get_termination_status()
                )));
            }
            let theta_boot = result
                .state
                .take_best_param()
                .ok_or_else(|| PyValueError::new_err("bootstrap optimization failed"))?;

            out.row_mut(i).assign(&theta_boot);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}
