use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, fisher_cov_binary, hc1_cov, invert_matrix,
    pyarray1_from_f64, pyarray2_from_f64, solve_least_squares_vec, take_rows, take_rows_vec,
    to_array1, to_array1_i32, to_array2,
};
use linfa::prelude::{Fit, FitWith, Predict};
use linfa::Dataset;
use linfa_elasticnet::ElasticNet as LinfaElasticNet;
use linfa_ftrl::Ftrl as LinfaFtrl;
use ndarray::{s, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_penalties(value: &Bound<'_, PyAny>) -> PyResult<Array1<f64>> {
    let penalties = if let Ok(scalar) = value.extract::<f64>() {
        Array1::from_vec(vec![scalar])
    } else if let Ok(array) = value.extract::<PyReadonlyArray1<f64>>() {
        to_array1(&array)
    } else if let Ok(values) = value.extract::<Vec<f64>>() {
        Array1::from_vec(values)
    } else {
        return Err(PyValueError::new_err(
            "penalty must be a nonnegative float or a 1D array-like of nonnegative floats",
        ));
    };

    if penalties.is_empty() {
        return Err(PyValueError::new_err(
            "penalty grid must contain at least one value",
        ));
    }
    if penalties
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(PyValueError::new_err(
            "penalty values must be finite and nonnegative",
        ));
    }

    Ok(penalties)
}

fn ridge_penalty_matrix(n_params: usize, penalty: f64, fit_intercept: bool) -> Array2<f64> {
    let mut penalty_matrix = Array2::<f64>::zeros((n_params, n_params));
    let start = if fit_intercept { 1 } else { 0 };
    for j in start..n_params {
        penalty_matrix[[j, j]] = penalty;
    }
    penalty_matrix
}

fn trace(a: &Array2<f64>) -> f64 {
    let dim = a.nrows().min(a.ncols());
    let mut out = 0.0;
    for i in 0..dim {
        out += a[[i, i]];
    }
    out
}

fn fit_ridge_params(
    design: &Array2<f64>,
    y: &Array1<f64>,
    penalty: f64,
    fit_intercept: bool,
) -> Result<Array1<f64>, String> {
    if penalty == 0.0 {
        return solve_least_squares_vec(design, y);
    }

    let n = design.nrows();
    let p = design.ncols();
    let start = if fit_intercept { 1 } else { 0 };
    let penalty_rows = p.saturating_sub(start);

    if penalty_rows == 0 {
        return solve_least_squares_vec(design, y);
    }

    let mut aug_design = Array2::<f64>::zeros((n + penalty_rows, p));
    aug_design.slice_mut(s![..n, ..]).assign(design);

    let sqrt_penalty = penalty.sqrt();
    for j in 0..penalty_rows {
        aug_design[[n + j, start + j]] = sqrt_penalty;
    }

    let mut aug_y = Array1::<f64>::zeros(n + penalty_rows);
    aug_y.slice_mut(s![..n]).assign(y);

    solve_least_squares_vec(&aug_design, &aug_y)
}

fn ridge_fit_path(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalties: &Array1<f64>,
    fit_intercept: bool,
) -> Result<(Array1<f64>, Array2<f64>), String> {
    let design = if fit_intercept {
        add_intercept(x)
    } else {
        x.clone()
    };
    let n_penalties = penalties.len();
    let n_features = x.ncols();
    let mut intercept_path = Array1::<f64>::zeros(n_penalties);
    let mut coef_path = Array2::<f64>::zeros((n_features, n_penalties));

    for (j, penalty) in penalties.iter().enumerate() {
        let params = fit_ridge_params(&design, y, *penalty, fit_intercept)?;
        if fit_intercept {
            intercept_path[j] = params[0];
            coef_path.column_mut(j).assign(&params.slice(s![1..]));
        } else {
            coef_path.column_mut(j).assign(&params);
        }
    }

    Ok((intercept_path, coef_path))
}

fn ridge_cv_mse(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalties: &Array1<f64>,
    fit_intercept: bool,
    cv: usize,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    if n != y.len() {
        return Err("x rows must match y length".to_string());
    }
    let n_folds = cv.min(n);
    if n_folds < 2 {
        return Err("cv must be at least 2 and no larger than the number of observations".to_string());
    }

    let fold_id: Vec<usize> = (0..n).map(|i| i % n_folds).collect();
    let mut scores = Array1::<f64>::zeros(penalties.len());

    for (j, penalty) in penalties.iter().enumerate() {
        let mut fold_mse = 0.0;
        for fold in 0..n_folds {
            let train_idx: Vec<usize> = (0..n).filter(|i| fold_id[*i] != fold).collect();
            let test_idx: Vec<usize> = (0..n).filter(|i| fold_id[*i] == fold).collect();

            let x_train = take_rows(x, &train_idx);
            let y_train = take_rows_vec(y, &train_idx);
            let x_test = take_rows(x, &test_idx);
            let y_test = take_rows_vec(y, &test_idx);

            let design_train = if fit_intercept {
                add_intercept(&x_train)
            } else {
                x_train
            };
            let design_test = if fit_intercept {
                add_intercept(&x_test)
            } else {
                x_test
            };

            let params = fit_ridge_params(&design_train, &y_train, *penalty, fit_intercept)?;
            let pred = design_test.dot(&params);
            let residuals = &y_test - &pred;
            fold_mse += residuals.dot(&residuals) / (residuals.len() as f64);
        }
        scores[j] = fold_mse / (n_folds as f64);
    }

    Ok(scores)
}

fn ridge_covariance(
    design: &Array2<f64>,
    residuals: &Array1<f64>,
    penalty: f64,
    fit_intercept: bool,
    vcov: &str,
) -> Result<Array2<f64>, String> {
    let n = design.nrows();
    let p = design.ncols();
    if residuals.len() != n {
        return Err("residual length mismatch".to_string());
    }

    let xtx = design.t().dot(design);
    let penalty_matrix = ridge_penalty_matrix(p, penalty, fit_intercept);
    let bread = &xtx + &penalty_matrix;
    let bread_inv = invert_matrix(&bread)?;
    let df_eff = trace(&xtx.dot(&bread_inv));
    let denom = n as f64 - df_eff;
    if denom <= 0.0 {
        return Err("need more observations than effective ridge degrees of freedom".to_string());
    }

    match vcov {
        "vanilla" => {
            let sigma2 = residuals.dot(residuals) / denom;
            Ok(bread_inv.dot(&xtx).dot(&bread_inv) * sigma2)
        }
        "hc1" => {
            let mut meat = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                let xi = design.row(i);
                let u = residuals[i];
                let outer = xi
                    .to_owned()
                    .insert_axis(Axis(1))
                    .dot(&xi.to_owned().insert_axis(Axis(0)));
                meat = meat + outer * (u * u);
            }
            let scale = n as f64 / denom;
            Ok(bread_inv.dot(&meat).dot(&bread_inv) * scale)
        }
        _ => Err("vcov must be one of {'hc1', 'vanilla'}".to_string()),
    }
}

#[pyclass]
pub struct Ridge {
    penalties: Array1<f64>,
    fit_intercept: bool,
    cv: usize,
    penalty_is_grid: bool,
    intercept: f64,
    coef: Option<Array1<f64>>,
    intercept_path: Option<Array1<f64>>,
    coef_path: Option<Array2<f64>>,
    selected_penalty: Option<f64>,
    best_penalty_index: Option<usize>,
    cv_mse: Option<Array1<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

#[pymethods]
impl Ridge {
    #[new]
    #[pyo3(signature = (penalty=None, cv=5))]
    fn new(py: Python<'_>, penalty: Option<Py<PyAny>>, cv: usize) -> PyResult<Self> {
        let penalty_is_grid = penalty.is_some();
        let penalties = match penalty {
            Some(value) => parse_penalties(value.bind(py))?,
            None => Array1::from_vec(vec![1.0]),
        };

        Ok(Self {
            penalty_is_grid: penalty_is_grid && penalties.len() > 1,
            penalties,
            fit_intercept: true,
            cv,
            intercept: 0.0,
            coef: None,
            intercept_path: None,
            coef_path: None,
            selected_penalty: None,
            best_penalty_index: None,
            cv_mse: None,
            x: None,
            y: None,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }

        let penalties = self.penalties.clone();
        let (cv_mse, best_penalty_index) = if penalties.len() > 1 {
            let cv_mse =
                ridge_cv_mse(&x, &y, &penalties, self.fit_intercept, self.cv).map_err(PyValueError::new_err)?;
            let mut best_idx = 0usize;
            let mut best_score = cv_mse[0];
            for (idx, score) in cv_mse.iter().enumerate().skip(1) {
                if *score < best_score {
                    best_score = *score;
                    best_idx = idx;
                }
            }
            (Some(cv_mse), Some(best_idx))
        } else {
            (None, None)
        };

        let (intercept_path, coef_path) =
            ridge_fit_path(&x, &y, &penalties, self.fit_intercept).map_err(PyValueError::new_err)?;
        let selected_index = best_penalty_index.unwrap_or(0);

        self.intercept = intercept_path[selected_index];
        self.coef = Some(coef_path.column(selected_index).to_owned());
        self.intercept_path = Some(intercept_path);
        self.coef_path = Some(coef_path);
        self.selected_penalty = Some(penalties[selected_index]);
        self.best_penalty_index = best_penalty_index;
        self.cv_mse = cv_mse;
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
            .ok_or_else(|| PyValueError::new_err("Ridge model is not fitted"))?;
        let x = to_array2(&x);
        let pred: Array1<f64> = x.dot(coef) + self.intercept;
        Ok(pyarray1_from_f64(py, &pred))
    }

    #[pyo3(signature = (vcov="hc1"))]
    fn summary<'py>(&self, py: Python<'py>, vcov: &str) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Ridge model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let penalty = self
            .selected_penalty
            .ok_or_else(|| PyValueError::new_err("Ridge model is not fitted"))?;

        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let mut params = Array1::<f64>::zeros(design.ncols());
        if self.fit_intercept {
            params[0] = self.intercept;
            params.slice_mut(s![1..]).assign(coef);
        } else {
            params.assign(coef);
        }
        let fitted = design.dot(&params);
        let residuals = y - &fitted;
        let cov =
            ridge_covariance(&design, &residuals, penalty, self.fit_intercept, vcov).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);
        let (intercept_se, coef_se) = if self.fit_intercept {
            (Some(se_all[0]), se_all.slice(s![1..]).to_owned())
        } else {
            (None, se_all)
        };

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", self.intercept)?;
        dict.set_item("coef", pyarray1_from_f64(py, coef))?;
        dict.set_item("intercept_se", intercept_se)?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &coef_se))?;
        dict.set_item("penalty", penalty)?;
        dict.set_item("penalties", pyarray1_from_f64(py, &self.penalties))?;
        dict.set_item("vcov_type", vcov)?;
        if let Some(best_idx) = self.best_penalty_index {
            dict.set_item("best_penalty_index", best_idx)?;
        } else {
            dict.set_item("best_penalty_index", py.None())?;
        }
        if let Some(cv_mse) = &self.cv_mse {
            dict.set_item("cv_mse", pyarray1_from_f64(py, cv_mse))?;
        } else {
            dict.set_item("cv_mse", py.None())?;
        }
        if let Some(intercept_path) = &self.intercept_path {
            dict.set_item("intercept_path", pyarray1_from_f64(py, intercept_path))?;
        }
        if let Some(coef_path) = &self.coef_path {
            dict.set_item("coef_path", pyarray2_from_f64(py, coef_path))?;
        }
        Ok(dict.into())
    }

    #[getter]
    fn best_penalty_index(&self) -> Option<usize> {
        self.best_penalty_index
    }

    #[getter]
    fn best_penalty(&self) -> Option<f64> {
        self.selected_penalty
    }

    #[getter]
    fn penalty_is_grid(&self) -> bool {
        self.penalty_is_grid
    }

    #[pyo3(signature = (n_bootstrap, seed=None))]
    fn bootstrap<'py>(
        &self,
        py: Python<'py>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.coef
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Ridge model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let penalty = self
            .selected_penalty
            .ok_or_else(|| PyValueError::new_err("Ridge model is not fitted"))?;

        let design_cols = x.ncols() + if self.fit_intercept { 1 } else { 0 };
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((n_bootstrap, design_cols));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let design_b = if self.fit_intercept {
                add_intercept(&xb)
            } else {
                xb
            };
            let params =
                fit_ridge_params(&design_b, &yb, penalty, self.fit_intercept).map_err(PyValueError::new_err)?;
            out.row_mut(i).assign(&params);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct ElasticNet {
    penalty: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: u32,
    model: Option<linfa_elasticnet::ElasticNet<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<f64>>,
}

#[pymethods]
impl ElasticNet {
    #[new]
    #[pyo3(signature = (penalty=1.0, l1_ratio=0.5, tolerance=1e-4, max_iterations=1000))]
    fn new(penalty: f64, l1_ratio: f64, tolerance: f64, max_iterations: u32) -> Self {
        Self {
            penalty,
            l1_ratio,
            fit_intercept: true,
            tolerance,
            max_iterations,
            model: None,
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
        let dataset = Dataset::new(x.clone(), y.clone());
        let params = LinfaElasticNet::params()
            .penalty(self.penalty)
            .l1_ratio(self.l1_ratio)
            .with_intercept(self.fit_intercept)
            .tolerance(self.tolerance)
            .max_iterations(self.max_iterations);
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ElasticNet model is not fitted"))?;
        let x = to_array2(&x);
        let pred = model.predict(&x);
        Ok(pyarray1_from_f64(py, &pred))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ElasticNet model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let y_hat = model.predict(x);
        let residuals = y - &y_hat;
        let design = if self.fit_intercept {
            add_intercept(x)
        } else {
            x.clone()
        };
        let cov = hc1_cov(&design, &residuals).map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov);

        let (intercept, coef, intercept_se, coef_se) = if self.fit_intercept {
            (
                model.intercept(),
                model.hyperplane().to_owned(),
                Some(se_all[0]),
                se_all.slice(s![1..]).to_owned(),
            )
        } else {
            (0.0, model.hyperplane().to_owned(), None, se_all)
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
            let dataset = Dataset::new(xb, yb);
            let params = LinfaElasticNet::params()
                .penalty(self.penalty)
                .l1_ratio(self.l1_ratio)
                .with_intercept(self.fit_intercept)
                .tolerance(self.tolerance)
                .max_iterations(self.max_iterations);
            let model = params
                .fit(&dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            if self.fit_intercept {
                out[[i, 0]] = model.intercept();
                out.row_mut(i)
                    .slice_mut(s![1..])
                    .assign(&model.hyperplane());
            } else {
                out.row_mut(i).assign(&model.hyperplane());
            }
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct FTRL {
    alpha: f64,
    beta: f64,
    l1_ratio: f64,
    l2_ratio: f64,
    model: Option<LinfaFtrl<f64>>,
    x: Option<Array2<f64>>,
    y: Option<Array1<bool>>,
}

#[pymethods]
impl FTRL {
    #[new]
    #[pyo3(signature = (alpha=0.1, beta=1.0, l1_ratio=1.0, l2_ratio=1.0))]
    fn new(alpha: f64, beta: f64, l1_ratio: f64, l2_ratio: f64) -> Self {
        Self {
            alpha,
            beta,
            l1_ratio,
            l2_ratio,
            model: None,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<i32>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1_i32(&y).mapv(|v| v != 0);
        let dataset = Dataset::new(x.clone(), y.clone());
        let params = linfa_ftrl::Ftrl::params()
            .alpha(self.alpha)
            .beta(self.beta)
            .l1_ratio(self.l1_ratio)
            .l2_ratio(self.l2_ratio);
        let model = params
            .fit_with(None, &dataset)
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
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FTRL model is not fitted"))?;
        let x = to_array2(&x);
        let probs = model.predict(&x).mapv(|v| f64::from(*v));
        Ok(pyarray1_from_f64(py, &probs))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FTRL model is not fitted"))?;
        let x = self
            .x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;

        let weights = model.get_weights();
        let probs = model.predict(x).mapv(|v| f64::from(*v));
        let cov = fisher_cov_binary(x, &probs).map_err(PyValueError::new_err)?;
        let se = diag_sqrt(&cov);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", pyarray1_from_f64(py, &weights))?;
        dict.set_item("coef_se", pyarray1_from_f64(py, &se))?;
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
        let mut out = Array2::<f64>::zeros((n_bootstrap, x.ncols()));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let mut yb = Array1::from_elem(idx.len(), false);
            for (j, &row) in idx.iter().enumerate() {
                yb[j] = y[row];
            }
            let dataset = Dataset::new(xb, yb);
            let params = linfa_ftrl::Ftrl::params()
                .alpha(self.alpha)
                .beta(self.beta)
                .l1_ratio(self.l1_ratio)
                .l2_ratio(self.l2_ratio);
            let model = params
                .fit_with(None, &dataset)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            out.row_mut(i).assign(&model.get_weights());
        }
        Ok(pyarray2_from_f64(py, &out))
    }
}
