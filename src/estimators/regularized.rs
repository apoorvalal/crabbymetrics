use crate::fit::FitDiagnostics;
use crate::utils::{
    add_intercept, bootstrap_indices, diag_sqrt, invert_matrix, pyarray1_from_f64,
    pyarray2_from_f64, sandwich_cov_from_parameter_scores, scale_rows, scale_vec,
    solve_least_squares_vec, sqrt_sample_weight, take_rows, take_rows_vec, to_array1,
    to_array1_i64, to_array2,
};
use crate::validation::validate_finite;
use linfa::prelude::{Fit, Predict};
use linfa::Dataset;
use linfa_elasticnet::ElasticNet as LinfaElasticNet;
use ndarray::{s, Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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

fn apply_sqrt_weights(
    design: &Array2<f64>,
    values: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
) -> Result<(Array2<f64>, Array1<f64>), String> {
    let sqrt_weight = sqrt_sample_weight(sample_weight, design.nrows())?;
    match sqrt_weight.as_ref() {
        Some(scale) => Ok((scale_rows(design, scale)?, scale_vec(values, scale)?)),
        None => Ok((design.clone(), values.clone())),
    }
}

fn weighted_mean_squared_error(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
    sample_weight: Option<&Array1<f64>>,
) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("prediction length mismatch".to_string());
    }
    match sample_weight {
        Some(weights) => {
            if weights.len() != y_true.len() {
                return Err(
                    "sample_weight length must match the number of observations".to_string()
                );
            }
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;
            for i in 0..y_true.len() {
                let weight = weights[i];
                let resid = y_true[i] - y_pred[i];
                weighted_sum += weight * resid * resid;
                total_weight += weight;
            }
            if total_weight <= 0.0 {
                return Err("sample_weight must contain at least one positive value".to_string());
            }
            Ok(weighted_sum / total_weight)
        }
        None => {
            let residuals = y_true - y_pred;
            Ok(residuals.dot(&residuals) / (residuals.len() as f64))
        }
    }
}

fn fit_ridge_params(
    design: &Array2<f64>,
    y: &Array1<f64>,
    penalty: f64,
    fit_intercept: bool,
    sample_weight: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let (design_work, y_work) = apply_sqrt_weights(design, y, sample_weight)?;
    if penalty == 0.0 {
        return solve_least_squares_vec(&design_work, &y_work);
    }

    let n = design_work.nrows();
    let p = design_work.ncols();
    let start = if fit_intercept { 1 } else { 0 };
    let penalty_rows = p.saturating_sub(start);

    if penalty_rows == 0 {
        return solve_least_squares_vec(&design_work, &y_work);
    }

    let mut aug_design = Array2::<f64>::zeros((n + penalty_rows, p));
    aug_design.slice_mut(s![..n, ..]).assign(&design_work);

    let sqrt_penalty = penalty.sqrt();
    for j in 0..penalty_rows {
        aug_design[[n + j, start + j]] = sqrt_penalty;
    }

    let mut aug_y = Array1::<f64>::zeros(n + penalty_rows);
    aug_y.slice_mut(s![..n]).assign(&y_work);

    solve_least_squares_vec(&aug_design, &aug_y)
}

fn ridge_fit_path(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalties: &Array1<f64>,
    fit_intercept: bool,
    sample_weight: Option<&Array1<f64>>,
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
        let params = fit_ridge_params(&design, y, *penalty, fit_intercept, sample_weight)?;
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
    sample_weight: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    if n != y.len() {
        return Err("x rows must match y length".to_string());
    }
    let n_folds = cv.min(n);
    if n_folds < 2 {
        return Err(
            "cv must be at least 2 and no larger than the number of observations".to_string(),
        );
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
            let w_train = sample_weight.map(|weights| take_rows_vec(weights, &train_idx));
            let w_test = sample_weight.map(|weights| take_rows_vec(weights, &test_idx));

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

            let params = fit_ridge_params(
                &design_train,
                &y_train,
                *penalty,
                fit_intercept,
                w_train.as_ref(),
            )?;
            let pred = design_test.dot(&params);
            fold_mse += weighted_mean_squared_error(&y_test, &pred, w_test.as_ref())?;
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
    lags: Option<usize>,
    clusters: Option<&Array1<i64>>,
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
        "hc1" | "newey_west" | "cluster" => {
            let mut raw_scores = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                let scale = residuals[i];
                raw_scores
                    .row_mut(i)
                    .assign(&design.row(i).mapv(|value| value * scale));
            }
            let param_scores = raw_scores.dot(&bread_inv);
            sandwich_cov_from_parameter_scores(&param_scores, vcov, denom, lags, clusters)
        }
        _ => Err("vcov must be one of {'hc1', 'vanilla', 'newey_west', 'cluster'}".to_string()),
    }
}

const MAX_POLYNOMIAL_TERMS: usize = 100_000;
const MAX_POLYNOMIAL_DESIGN_CELLS: usize = 50_000_000;

fn polynomial_term_count(n_features: usize, degree: usize) -> Result<usize, String> {
    if n_features == 0 {
        return Err("x must have at least one feature".to_string());
    }
    if degree == 0 {
        return Err("degree must be at least 1".to_string());
    }

    let mut combinations = 1_u128;
    for current_degree in 1..=degree {
        combinations = combinations
            .checked_mul((n_features + current_degree) as u128)
            .ok_or_else(|| "polynomial term count overflowed".to_string())?
            / current_degree as u128;
    }
    let count = combinations
        .checked_sub(1)
        .ok_or_else(|| "polynomial term count underflowed".to_string())?;
    if count > MAX_POLYNOMIAL_TERMS as u128 {
        return Err(format!(
            "polynomial design has {} terms; maximum supported is {}",
            count, MAX_POLYNOMIAL_TERMS
        ));
    }
    Ok(count as usize)
}

fn validate_polynomial_design_size(n_rows: usize, n_terms: usize) -> Result<(), String> {
    let cells = n_rows
        .checked_mul(n_terms + 1)
        .ok_or_else(|| "polynomial design size overflowed".to_string())?;
    if cells > MAX_POLYNOMIAL_DESIGN_CELLS {
        return Err(format!(
            "polynomial design needs {} cells; maximum supported is {}",
            cells, MAX_POLYNOMIAL_DESIGN_CELLS
        ));
    }
    Ok(())
}

fn append_polynomial_terms(
    n_features: usize,
    start: usize,
    remaining_degree: usize,
    current: &mut Vec<usize>,
    terms: &mut Vec<Vec<usize>>,
) {
    if remaining_degree == 0 {
        terms.push(current.clone());
        return;
    }
    for feature in start..n_features {
        current.push(feature);
        append_polynomial_terms(n_features, feature, remaining_degree - 1, current, terms);
        current.pop();
    }
}

fn polynomial_terms(n_features: usize, degree: usize) -> Result<Vec<Vec<usize>>, String> {
    let expected = polynomial_term_count(n_features, degree)?;
    let mut terms = Vec::with_capacity(expected);
    let mut current = Vec::new();
    for term_degree in 1..=degree {
        append_polynomial_terms(n_features, 0, term_degree, &mut current, &mut terms);
    }
    Ok(terms)
}

fn select_columns(x: &Array2<f64>, columns: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((x.nrows(), columns.len()));
    for (j, col) in columns.iter().enumerate() {
        out.column_mut(j).assign(&x.column(*col));
    }
    out
}

fn polynomial_design(x: &Array2<f64>, terms: &[Vec<usize>]) -> Result<Array2<f64>, String> {
    validate_polynomial_design_size(x.nrows(), terms.len())?;
    let mut out = Array2::<f64>::ones((x.nrows(), terms.len()));
    for (j, term) in terms.iter().enumerate() {
        for feature in term {
            for i in 0..x.nrows() {
                out[[i, j]] *= x[[i, *feature]];
            }
        }
    }
    Ok(out)
}

fn standardize_polynomial_design(poly: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut means = Array1::<f64>::zeros(poly.ncols());
    let mut scales = Array1::<f64>::ones(poly.ncols());
    for j in 0..poly.ncols() {
        let mean = poly.column(j).mean().unwrap_or(0.0);
        let variance = poly
            .column(j)
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / poly.nrows() as f64;
        let scale = variance.sqrt();
        means[j] = mean;
        if scale > 1e-12 {
            scales[j] = scale;
        }
        for i in 0..poly.nrows() {
            poly[[i, j]] = (poly[[i, j]] - means[j]) / scales[j];
        }
    }
    (means, scales)
}

struct PolynomialBaseLearner {
    features: Vec<usize>,
    terms: std::sync::Arc<Vec<Vec<usize>>>,
    term_means: Array1<f64>,
    term_scales: Array1<f64>,
    params: Array1<f64>,
    train_mse: f64,
}

impl PolynomialBaseLearner {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, String> {
        if self.features.iter().any(|feature| *feature >= x.ncols()) {
            return Err("x has fewer columns than the fitted model expects".to_string());
        }
        let x_sub = select_columns(x, &self.features);
        let mut poly = polynomial_design(&x_sub, self.terms.as_ref())?;
        for j in 0..poly.ncols() {
            for i in 0..poly.nrows() {
                poly[[i, j]] = (poly[[i, j]] - self.term_means[j]) / self.term_scales[j];
            }
        }
        let design = add_intercept(&poly);
        if design.ncols() != self.params.len() {
            return Err("base learner parameter length mismatch".to_string());
        }
        Ok(design.dot(&self.params))
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
    sample_weight: Option<Array1<f64>>,
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
            sample_weight: None,
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
            let cv_mse = ridge_cv_mse(&x, &y, &penalties, self.fit_intercept, self.cv, None)
                .map_err(PyValueError::new_err)?;
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
            ridge_fit_path(&x, &y, &penalties, self.fit_intercept, None)
                .map_err(PyValueError::new_err)?;
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
        self.sample_weight = None;
        Ok(())
    }

    fn fit_weighted(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray1<f64>,
        sample_weight: Vec<f64>,
    ) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        let sample_weight = Array1::from_vec(sample_weight);

        let penalties = self.penalties.clone();
        let (cv_mse, best_penalty_index) = if penalties.len() > 1 {
            let cv_mse = ridge_cv_mse(
                &x,
                &y,
                &penalties,
                self.fit_intercept,
                self.cv,
                Some(&sample_weight),
            )
            .map_err(PyValueError::new_err)?;
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
            ridge_fit_path(&x, &y, &penalties, self.fit_intercept, Some(&sample_weight))
                .map_err(PyValueError::new_err)?;
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
        self.sample_weight = Some(sample_weight);
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

    #[pyo3(signature = (vcov="hc1", lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: &str,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
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
        let sample_weight = self.sample_weight.as_ref();

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
        let (design_work, residuals_work) = apply_sqrt_weights(&design, &residuals, sample_weight)
            .map_err(PyValueError::new_err)?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let cov = ridge_covariance(
            &design_work,
            &residuals_work,
            penalty,
            self.fit_intercept,
            vcov,
            lags,
            cluster_ids.as_ref(),
        )
        .map_err(PyValueError::new_err)?;
        let se_all = diag_sqrt(&cov).map_err(PyValueError::new_err)?;
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
        let sample_weight = self.sample_weight.as_ref();

        let design_cols = x.ncols() + if self.fit_intercept { 1 } else { 0 };
        let idxs = bootstrap_indices(x.nrows(), n_bootstrap, seed);
        let mut out = Array2::<f64>::zeros((n_bootstrap, design_cols));
        for (i, idx) in idxs.iter().enumerate() {
            let xb = take_rows(x, idx);
            let yb = take_rows_vec(y, idx);
            let wb = sample_weight.map(|weights| take_rows_vec(weights, idx));
            let design_b = if self.fit_intercept {
                add_intercept(&xb)
            } else {
                xb
            };
            let params = fit_ridge_params(&design_b, &yb, penalty, self.fit_intercept, wb.as_ref())
                .map_err(PyValueError::new_err)?;
            out.row_mut(i).assign(&params);
        }

        Ok(pyarray2_from_f64(py, &out))
    }
}

#[pyclass]
pub struct BaggedPolynomialRegressor {
    n_estimators: usize,
    degree: usize,
    max_features: Option<usize>,
    max_samples: Option<usize>,
    bootstrap: bool,
    penalty: f64,
    seed: u64,
    learners: Option<Vec<PolynomialBaseLearner>>,
    n_features_in: Option<usize>,
    max_features_fitted: Option<usize>,
    max_samples_fitted: Option<usize>,
    n_terms: Option<usize>,
    oob_mse: Option<f64>,
    oob_coverage: f64,
}

#[pymethods]
impl BaggedPolynomialRegressor {
    #[new]
    #[pyo3(signature = (n_estimators=50, degree=2, max_features=None, max_samples=None, bootstrap=true, penalty=1.0, seed=42))]
    fn new(
        n_estimators: usize,
        degree: usize,
        max_features: Option<usize>,
        max_samples: Option<usize>,
        bootstrap: bool,
        penalty: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if n_estimators == 0 {
            return Err(PyValueError::new_err("n_estimators must be at least 1"));
        }
        if degree == 0 {
            return Err(PyValueError::new_err("degree must be at least 1"));
        }
        if !penalty.is_finite() || penalty < 0.0 {
            return Err(PyValueError::new_err(
                "penalty must be finite and nonnegative",
            ));
        }
        if matches!(max_features, Some(0)) {
            return Err(PyValueError::new_err("max_features must be at least 1"));
        }
        if matches!(max_samples, Some(0)) {
            return Err(PyValueError::new_err("max_samples must be at least 1"));
        }
        Ok(Self {
            n_estimators,
            degree,
            max_features,
            max_samples,
            bootstrap,
            penalty,
            seed,
            learners: None,
            n_features_in: None,
            max_features_fitted: None,
            max_samples_fitted: None,
            n_terms: None,
            oob_mse: None,
            oob_coverage: 0.0,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        let y = to_array1(&y);
        if x.nrows() != y.len() {
            return Err(PyValueError::new_err("x rows must match y length"));
        }
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if x.ncols() == 0 {
            return Err(PyValueError::new_err("x must have at least one feature"));
        }
        validate_finite("x", &x).map_err(PyValueError::new_err)?;
        validate_finite("y", &y).map_err(PyValueError::new_err)?;

        let n = x.nrows();
        let p = x.ncols();
        let max_features = self.max_features.unwrap_or(p);
        if max_features > p {
            return Err(PyValueError::new_err(
                "max_features cannot exceed the number of x columns",
            ));
        }
        let max_samples = self.max_samples.unwrap_or(n);
        if max_samples > n {
            return Err(PyValueError::new_err(
                "max_samples cannot exceed the number of x rows",
            ));
        }

        let terms = std::sync::Arc::new(
            polynomial_terms(max_features, self.degree).map_err(PyValueError::new_err)?,
        );
        validate_polynomial_design_size(max_samples, terms.len()).map_err(PyValueError::new_err)?;

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut learners = Vec::with_capacity(self.n_estimators);
        let all_features: Vec<usize> = (0..p).collect();
        let mut oob_sum = Array1::<f64>::zeros(n);
        let mut oob_count = vec![0_usize; n];

        for _ in 0..self.n_estimators {
            let row_idx: Vec<usize> = if self.bootstrap {
                (0..max_samples).map(|_| rng.gen_range(0..n)).collect()
            } else {
                let mut rows: Vec<usize> = (0..n).collect();
                rows.shuffle(&mut rng);
                rows.truncate(max_samples);
                rows.sort_unstable();
                rows
            };
            let mut in_bag = vec![false; n];
            for row in &row_idx {
                in_bag[*row] = true;
            }

            let mut feature_idx = all_features.clone();
            feature_idx.shuffle(&mut rng);
            feature_idx.truncate(max_features);
            feature_idx.sort_unstable();

            let x_rows = take_rows(&x, &row_idx);
            let y_rows = take_rows_vec(&y, &row_idx);
            let x_sub = select_columns(&x_rows, &feature_idx);
            let mut poly =
                polynomial_design(&x_sub, terms.as_ref()).map_err(PyValueError::new_err)?;
            let (term_means, term_scales) = standardize_polynomial_design(&mut poly);
            let design = add_intercept(&poly);
            let params = fit_ridge_params(&design, &y_rows, self.penalty, true, None)
                .map_err(PyValueError::new_err)?;
            let residuals = &y_rows - &design.dot(&params);
            let train_mse = residuals.dot(&residuals) / residuals.len() as f64;

            let learner = PolynomialBaseLearner {
                features: feature_idx,
                terms: terms.clone(),
                term_means,
                term_scales,
                params,
                train_mse,
            };
            if self.bootstrap {
                let prediction = learner.predict(&x).map_err(PyValueError::new_err)?;
                for i in 0..n {
                    if !in_bag[i] {
                        oob_sum[i] += prediction[i];
                        oob_count[i] += 1;
                    }
                }
            }
            learners.push(learner);
        }

        let covered: Vec<usize> = (0..n).filter(|i| oob_count[*i] > 0).collect();
        self.oob_coverage = covered.len() as f64 / n as f64;
        self.oob_mse = if covered.is_empty() {
            None
        } else {
            let squared_error = covered
                .iter()
                .map(|i| {
                    let prediction = oob_sum[*i] / oob_count[*i] as f64;
                    (y[*i] - prediction).powi(2)
                })
                .sum::<f64>();
            Some(squared_error / covered.len() as f64)
        };
        self.learners = Some(learners);
        self.n_features_in = Some(p);
        self.max_features_fitted = Some(max_features);
        self.max_samples_fitted = Some(max_samples);
        self.n_terms = Some(terms.len());
        Ok(())
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let learners = self.learners.as_ref().ok_or_else(|| {
            PyValueError::new_err("BaggedPolynomialRegressor model is not fitted")
        })?;
        let expected_features = self.n_features_in.ok_or_else(|| {
            PyValueError::new_err("BaggedPolynomialRegressor model is not fitted")
        })?;
        let x = to_array2(&x);
        if x.ncols() != expected_features {
            return Err(PyValueError::new_err(format!(
                "x has {} columns; fitted model expects {}",
                x.ncols(),
                expected_features
            )));
        }
        validate_finite("x", &x).map_err(PyValueError::new_err)?;
        validate_polynomial_design_size(x.nrows(), self.n_terms.unwrap_or(0))
            .map_err(PyValueError::new_err)?;

        let mut prediction = Array1::<f64>::zeros(x.nrows());
        for learner in learners {
            prediction = prediction + learner.predict(&x).map_err(PyValueError::new_err)?;
        }
        prediction /= learners.len() as f64;
        Ok(pyarray1_from_f64(py, &prediction))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let learners = self.learners.as_ref().ok_or_else(|| {
            PyValueError::new_err("BaggedPolynomialRegressor model is not fitted")
        })?;
        let feature_indices: Vec<Vec<usize>> = learners
            .iter()
            .map(|learner| learner.features.clone())
            .collect();
        let term_counts = vec![self.n_terms.unwrap_or(0); learners.len()];
        let train_mse =
            Array1::from_vec(learners.iter().map(|learner| learner.train_mse).collect());

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("n_estimators", self.n_estimators)?;
        dict.set_item("degree", self.degree)?;
        dict.set_item("max_features", self.max_features_fitted)?;
        dict.set_item("max_samples", self.max_samples_fitted)?;
        dict.set_item("bootstrap", self.bootstrap)?;
        dict.set_item("penalty", self.penalty)?;
        dict.set_item("seed", self.seed)?;
        dict.set_item("n_features_in", self.n_features_in)?;
        dict.set_item("n_terms", self.n_terms)?;
        dict.set_item("feature_indices", feature_indices)?;
        dict.set_item("term_counts", term_counts)?;
        dict.set_item("train_mse", pyarray1_from_f64(py, &train_mse))?;
        dict.set_item("oob_mse", self.oob_mse)?;
        dict.set_item("oob_coverage", self.oob_coverage)?;
        dict.set_item("inference_available", false)?;
        Ok(dict.into())
    }
}

fn elastic_net_duality_gap(
    x: &Array2<f64>,
    y_centered: &Array1<f64>,
    coef: &Array1<f64>,
    l1_ratio: f64,
    penalty: f64,
) -> f64 {
    let residual = y_centered - &x.dot(coef);
    let n = x.nrows() as f64;
    let l1_reg = l1_ratio * penalty * n;
    let l2_reg = (1.0 - l1_ratio) * penalty * n;
    let dual_vector = x.t().dot(&residual) - &(coef * l2_reg);
    let dual_norm = dual_vector
        .iter()
        .fold(0.0_f64, |current, value| current.max(value.abs()));
    let residual_norm_sq = residual.dot(&residual);
    let coef_norm_sq = coef.dot(coef);
    let (scale, mut gap) = if dual_norm > l1_reg {
        let scale = l1_reg / dual_norm;
        (scale, 0.5 * residual_norm_sq * (1.0 + scale * scale))
    } else {
        (1.0, residual_norm_sq)
    };
    gap += l1_reg * coef.iter().map(|value| value.abs()).sum::<f64>()
        - scale * residual.dot(y_centered)
        + 0.5 * l2_reg * (1.0 + scale * scale) * coef_norm_sq;
    gap
}

fn elastic_net_objective(
    x: &Array2<f64>,
    y: &Array1<f64>,
    model: &linfa_elasticnet::ElasticNet<f64>,
    penalty: f64,
    l1_ratio: f64,
) -> f64 {
    let residual = y - &model.predict(x);
    let coef = model.hyperplane();
    0.5 * residual.dot(&residual) / x.nrows() as f64
        + penalty * l1_ratio * coef.iter().map(|value| value.abs()).sum::<f64>()
        + 0.5 * penalty * (1.0 - l1_ratio) * coef.dot(coef)
}

fn fit_elastic_net(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalty: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: u32,
) -> Result<(linfa_elasticnet::ElasticNet<f64>, FitDiagnostics, f64, f64), String> {
    if x.nrows() != y.len() {
        return Err("x rows must match y length".to_string());
    }
    if x.nrows() == 0 || x.ncols() == 0 {
        return Err("x must contain at least one row and one column".to_string());
    }
    validate_finite("x", x)?;
    validate_finite("y", y)?;
    if !penalty.is_finite() || penalty < 0.0 {
        return Err("penalty must be finite and nonnegative".to_string());
    }
    if !l1_ratio.is_finite() || !(0.0..=1.0).contains(&l1_ratio) {
        return Err("l1_ratio must be finite and between 0 and 1".to_string());
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err("tolerance must be positive and finite".to_string());
    }
    if max_iterations == 0 {
        return Err("max_iterations must be positive".to_string());
    }

    let dataset = Dataset::new(x.clone(), y.clone());
    let params = LinfaElasticNet::params()
        .penalty(penalty)
        .l1_ratio(l1_ratio)
        .with_intercept(fit_intercept)
        .tolerance(tolerance)
        .max_iterations(max_iterations);
    let model = params.fit(&dataset).map_err(|err| err.to_string())?;
    let y_centered = if fit_intercept {
        y - y.mean().unwrap_or(0.0)
    } else {
        y.clone()
    };
    let duality_gap =
        elastic_net_duality_gap(x, &y_centered, model.hyperplane(), l1_ratio, penalty);
    let duality_gap_tolerance = tolerance * y_centered.dot(&y_centered);
    let converged = duality_gap.is_finite() && duality_gap <= duality_gap_tolerance;
    let termination_reason = if converged {
        "Duality gap tolerance reached".to_string()
    } else {
        format!(
            "Maximum number of iterations reached (duality gap {duality_gap:.6e} exceeds {duality_gap_tolerance:.6e})"
        )
    };
    let diagnostics = FitDiagnostics::new(
        converged,
        u64::from(model.n_steps()),
        termination_reason,
        Some(elastic_net_objective(x, y, &model, penalty, l1_ratio)),
    );
    diagnostics.require_converged("ElasticNet")?;
    Ok((model, diagnostics, duality_gap_tolerance, duality_gap))
}

#[pyclass]
pub struct ElasticNet {
    penalty: f64,
    l1_ratio: f64,
    fit_intercept: bool,
    tolerance: f64,
    max_iterations: u32,
    model: Option<linfa_elasticnet::ElasticNet<f64>>,
    diagnostics: Option<FitDiagnostics>,
    duality_gap_tolerance: Option<f64>,
    duality_gap: Option<f64>,
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
            diagnostics: None,
            duality_gap_tolerance: None,
            duality_gap: None,
            x: None,
            y: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> PyResult<()> {
        self.model = None;
        self.diagnostics = None;
        self.duality_gap_tolerance = None;
        self.duality_gap = None;
        self.x = None;
        self.y = None;
        let x = to_array2(&x);
        let y = to_array1(&y);
        let (model, diagnostics, duality_gap_tolerance, duality_gap) = fit_elastic_net(
            &x,
            &y,
            self.penalty,
            self.l1_ratio,
            self.fit_intercept,
            self.tolerance,
            self.max_iterations,
        )
        .map_err(PyValueError::new_err)?;
        self.model = Some(model);
        self.diagnostics = Some(diagnostics);
        self.duality_gap_tolerance = Some(duality_gap_tolerance);
        self.duality_gap = Some(duality_gap);
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
        let diagnostics = self
            .diagnostics
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("ElasticNet fit diagnostics are unavailable"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("intercept", model.intercept())?;
        dict.set_item("coef", pyarray1_from_f64(py, model.hyperplane()))?;
        dict.set_item("penalty", self.penalty)?;
        dict.set_item("l1_ratio", self.l1_ratio)?;
        dict.set_item("duality_gap", self.duality_gap)?;
        dict.set_item("duality_gap_tolerance", self.duality_gap_tolerance)?;
        diagnostics.write_summary(&dict)?;
        dict.set_item("inference_available", false)?;
        dict.set_item("intercept_se", py.None())?;
        dict.set_item("coef_se", py.None())?;
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
            let (model, _, _, _) = fit_elastic_net(
                &xb,
                &yb,
                self.penalty,
                self.l1_ratio,
                self.fit_intercept,
                self.tolerance,
                self.max_iterations,
            )
            .map_err(|err| {
                PyValueError::new_err(format!("ElasticNet bootstrap replicate {i} failed: {err}"))
            })?;
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
