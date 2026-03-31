use crate::utils::{
    add_intercept, pyarray1_from_f64, pyarray2_from_f64, sandwich_cov_from_parameter_scores,
    scale_rows, score_cov_iid, solve_least_squares_mat, solve_least_squares_vec, take_rows,
    take_rows_vec, to_array1, to_array1_i64, to_array2,
};
use ndarray::{concatenate, Array1, Array2, Axis};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_finite_1d(name: &str, values: &Array1<f64>) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{} must contain only finite values", name));
    }
    Ok(())
}

fn validate_finite_2d(name: &str, values: &Array2<f64>) -> Result<(), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{} must contain only finite values", name));
    }
    Ok(())
}

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

fn ridge_fit_params(x: &Array2<f64>, y: &Array1<f64>, penalty: f64) -> Result<Array1<f64>, String> {
    if x.nrows() != y.len() {
        return Err("x rows must match y length".to_string());
    }
    let design = add_intercept(x);
    if penalty == 0.0 {
        return solve_least_squares_vec(&design, y);
    }

    let n = design.nrows();
    let p = design.ncols();
    let penalty_rows = p - 1;

    let mut aug_design = Array2::<f64>::zeros((n + penalty_rows, p));
    aug_design.slice_mut(ndarray::s![..n, ..]).assign(&design);
    let sqrt_penalty = penalty.sqrt();
    for j in 0..penalty_rows {
        aug_design[[n + j, j + 1]] = sqrt_penalty;
    }

    let mut aug_y = Array1::<f64>::zeros(n + penalty_rows);
    aug_y.slice_mut(ndarray::s![..n]).assign(y);

    solve_least_squares_vec(&aug_design, &aug_y)
}

fn ridge_predict(x: &Array2<f64>, params: &Array1<f64>) -> Result<Array1<f64>, String> {
    if params.len() != x.ncols() + 1 {
        return Err("ridge parameter length mismatch".to_string());
    }
    Ok(add_intercept(x).dot(params))
}

fn mean_squared_error(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<f64, String> {
    if y_true.len() != y_pred.len() {
        return Err("prediction length mismatch".to_string());
    }
    let residual = y_true - y_pred;
    Ok(residual.dot(&residual) / (residual.len() as f64))
}

fn ridge_cv_mse(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalties: &Array1<f64>,
    cv: usize,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    if n != y.len() {
        return Err("x rows must match y length".to_string());
    }
    let n_folds = cv.min(n);
    if n_folds < 2 {
        return Err("cv must be at least 2".to_string());
    }

    let fold_id: Vec<usize> = (0..n).map(|i| i % n_folds).collect();
    let mut mse = Array1::<f64>::zeros(penalties.len());

    for (j, penalty) in penalties.iter().enumerate() {
        let mut total = 0.0;
        for fold in 0..n_folds {
            let train_idx: Vec<usize> = (0..n).filter(|i| fold_id[*i] != fold).collect();
            let test_idx: Vec<usize> = (0..n).filter(|i| fold_id[*i] == fold).collect();
            let x_train = take_rows(x, &train_idx);
            let y_train = take_rows_vec(y, &train_idx);
            let x_test = take_rows(x, &test_idx);
            let y_test = take_rows_vec(y, &test_idx);

            let params = ridge_fit_params(&x_train, &y_train, *penalty)?;
            let pred = ridge_predict(&x_test, &params)?;
            total += mean_squared_error(&y_test, &pred)?;
        }
        mse[j] = total / (n_folds as f64);
    }

    Ok(mse)
}

fn select_ridge_penalty(
    x: &Array2<f64>,
    y: &Array1<f64>,
    penalties: &Array1<f64>,
    cv: usize,
) -> Result<(Array1<f64>, f64, Option<usize>), String> {
    if penalties.len() == 1 || x.nrows() < 4 {
        let penalty = penalties[0];
        let params = ridge_fit_params(x, y, penalty)?;
        return Ok((params, penalty, None));
    }

    let cv_mse = ridge_cv_mse(x, y, penalties, cv)?;
    let mut best_idx = 0usize;
    let mut best_score = cv_mse[0];
    for (idx, score) in cv_mse.iter().enumerate().skip(1) {
        if *score < best_score {
            best_score = *score;
            best_idx = idx;
        }
    }
    let penalty = penalties[best_idx];
    let params = ridge_fit_params(x, y, penalty)?;
    Ok((params, penalty, Some(best_idx)))
}

fn make_kfold_splits(
    n: usize,
    n_folds: usize,
    seed: u64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>, String> {
    if n_folds < 2 {
        return Err("n_folds must be at least 2".to_string());
    }
    if n == 0 {
        return Err("need at least one observation".to_string());
    }
    let k = n_folds.min(n);
    if k < 2 {
        return Err("n_folds must be at least 2".to_string());
    }

    let mut fold_id = vec![0usize; n];
    let offset = (seed as usize) % k;
    for (idx, slot) in fold_id.iter_mut().enumerate() {
        *slot = (idx + offset) % k;
    }

    let mut out = Vec::with_capacity(k);
    for fold in 0..k {
        let test_idx: Vec<usize> = (0..n).filter(|i| fold_id[*i] == fold).collect();
        let train_idx: Vec<usize> = (0..n).filter(|i| fold_id[*i] != fold).collect();
        out.push((train_idx, test_idx));
    }
    Ok(out)
}

fn mean_moments(moments: &Array2<f64>) -> Result<Array1<f64>, String> {
    moments
        .mean_axis(Axis(0))
        .ok_or_else(|| "moment function must return a non-empty array".to_string())
}

fn numerical_mean_jacobian<F>(
    theta: &Array1<f64>,
    fd_eps: f64,
    moment_fn: F,
) -> Result<Array2<f64>, String>
where
    F: Fn(&Array1<f64>) -> Result<Array2<f64>, String>,
{
    let base_mean = mean_moments(&moment_fn(theta)?)?;
    let m = base_mean.len();
    let p = theta.len();
    let mut jacobian = Array2::<f64>::zeros((m, p));

    for j in 0..p {
        let h = fd_eps * theta[j].abs().max(1.0);
        let mut theta_hi = theta.clone();
        let mut theta_lo = theta.clone();
        theta_hi[j] += h;
        theta_lo[j] -= h;

        let g_hi = mean_moments(&moment_fn(&theta_hi)?)?;
        let g_lo = mean_moments(&moment_fn(&theta_lo)?)?;
        let diff = (&g_hi - &g_lo) / (2.0 * h);
        jacobian.column_mut(j).assign(&diff);
    }

    Ok(jacobian)
}

fn exact_identified_covariance(
    moments: &Array2<f64>,
    mean_jacobian: &Array2<f64>,
    vcov: &str,
    lags: Option<usize>,
    clusters: Option<&Array1<i64>>,
) -> Result<Array2<f64>, String> {
    let n = moments.nrows();
    let p = mean_jacobian.ncols();
    if mean_jacobian.nrows() != p {
        return Err("moment Jacobian must be square for exact-identified covariance".to_string());
    }
    if moments.ncols() != p {
        return Err("moment dimension must match parameter dimension".to_string());
    }

    let inv_j = crate::utils::invert_matrix(mean_jacobian)?;
    let param_scores = moments.dot(&inv_j.t()) / (n as f64);

    match vcov {
        "vanilla" => Ok(score_cov_iid(&param_scores)),
        "hc1" | "newey_west" | "cluster" => sandwich_cov_from_parameter_scores(
            &param_scores,
            vcov,
            n as f64 - p as f64,
            lags,
            clusters,
        ),
        _ => Err("vcov must be one of {'vanilla', 'hc1', 'newey_west', 'cluster'}".to_string()),
    }
}

fn column_array(values: &Array1<f64>) -> Array2<f64> {
    values.clone().insert_axis(Axis(1))
}

fn center_controls(w: &Array2<f64>, mu: &Array1<f64>) -> Result<Array2<f64>, String> {
    if w.ncols() != mu.len() {
        return Err("control mean length mismatch".to_string());
    }
    let mut centered = w.clone();
    for i in 0..w.nrows() {
        for j in 0..w.ncols() {
            centered[[i, j]] -= mu[j];
        }
    }
    Ok(centered)
}

fn mean_controls(w: &Array2<f64>) -> Result<Array1<f64>, String> {
    w.mean_axis(Axis(0))
        .ok_or_else(|| "controls must be non-empty".to_string())
}

fn build_ob_design(wc: &Array2<f64>, d: &Array1<f64>) -> Result<Array2<f64>, String> {
    if wc.nrows() != d.len() {
        return Err("row count mismatch".to_string());
    }
    let d_col = column_array(d);
    let wc_by_d = scale_rows(wc, d)?;
    let ones = Array2::ones((wc.nrows(), 1));
    concatenate(
        Axis(1),
        &[ones.view(), wc.view(), wc_by_d.view(), d_col.view()],
    )
    .map_err(|_| "failed to build average-derivative design".to_string())
}

fn build_dr_instruments(
    wc: &Array2<f64>,
    d: &Array1<f64>,
    gw: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    if wc.nrows() != d.len() || d.len() != gw.len() {
        return Err("row count mismatch".to_string());
    }
    let gw_col = column_array(gw);
    let wc_by_d = scale_rows(wc, d)?;
    let ones = Array2::ones((wc.nrows(), 1));
    concatenate(
        Axis(1),
        &[ones.view(), wc.view(), wc_by_d.view(), gw_col.view()],
    )
    .map_err(|_| "failed to build average-derivative instruments".to_string())
}

fn fit_linear_iv(
    design: &Array2<f64>,
    instruments: &Array2<f64>,
    y: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    if design.nrows() != y.len() || instruments.nrows() != y.len() {
        return Err("row count mismatch".to_string());
    }
    let pi_hat = solve_least_squares_mat(instruments, design)?;
    let x_hat = instruments.dot(&pi_hat);
    solve_least_squares_vec(&x_hat, y)
}

fn validate_binary(d: &Array1<f64>) -> Result<(), String> {
    if d.iter().any(|value| !(*value == 0.0 || *value == 1.0)) {
        return Err("treatment must contain only 0/1 values".to_string());
    }
    let treated = d.iter().filter(|value| **value == 1.0).count();
    let control = d.len() - treated;
    if treated == 0 || control == 0 {
        return Err("need both treated and control observations".to_string());
    }
    Ok(())
}

fn eplm_theta(y: &Array1<f64>, d: &Array1<f64>, w: &Array2<f64>) -> Result<Array1<f64>, String> {
    let w_design = add_intercept(w);
    let pi = solve_least_squares_vec(&w_design, d)?;
    let ehat = w_design.dot(&pi);
    let z = d - &ehat;
    let denom = z.dot(d);
    if denom.abs() < 1e-12 {
        return Err("EPLM residualized treatment has near-zero variation".to_string());
    }
    let beta = z.dot(y) / denom;
    let mut theta = Array1::<f64>::zeros(pi.len() + 1);
    theta.slice_mut(ndarray::s![..pi.len()]).assign(&pi);
    theta[pi.len()] = beta;
    Ok(theta)
}

fn eplm_moments(
    y: &Array1<f64>,
    d: &Array1<f64>,
    w: &Array2<f64>,
    theta: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let p = w.ncols() + 1;
    if theta.len() != p + 1 {
        return Err("EPLM parameter length mismatch".to_string());
    }
    let pi = theta.slice(ndarray::s![..p]).to_owned();
    let beta = theta[p];
    let w_design = add_intercept(w);
    let ehat = w_design.dot(&pi);
    let z = d - &ehat;
    let outcome_resid = y - &(d * beta);
    let m1 = scale_rows(&w_design, &z)?;
    let m2 = column_array(&(z * outcome_resid));
    concatenate(Axis(1), &[m1.view(), m2.view()])
        .map_err(|_| "failed to build EPLM moments".to_string())
}

fn ob_theta(y: &Array1<f64>, d: &Array1<f64>, w: &Array2<f64>) -> Result<Array1<f64>, String> {
    let mu = mean_controls(w)?;
    let wc = center_controls(w, &mu)?;
    let rx = build_ob_design(&wc, d)?;
    let lambda = solve_least_squares_vec(&rx, y)?;
    let mut theta = Array1::<f64>::zeros(mu.len() + lambda.len());
    theta.slice_mut(ndarray::s![..mu.len()]).assign(&mu);
    theta.slice_mut(ndarray::s![mu.len()..]).assign(&lambda);
    Ok(theta)
}

fn ob_moments(
    y: &Array1<f64>,
    d: &Array1<f64>,
    w: &Array2<f64>,
    theta: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let p = w.ncols();
    let q = 2 * p + 2;
    if theta.len() != p + q {
        return Err("AverageDerivative(ob) parameter length mismatch".to_string());
    }
    let mu = theta.slice(ndarray::s![..p]).to_owned();
    let lambda = theta.slice(ndarray::s![p..]).to_owned();
    let wc = center_controls(w, &mu)?;
    let rx = build_ob_design(&wc, d)?;
    let resid = y - &rx.dot(&lambda);
    let m1 = wc;
    let m2 = scale_rows(&rx, &resid)?;
    concatenate(Axis(1), &[m1.view(), m2.view()])
        .map_err(|_| "failed to build AverageDerivative(ob) moments".to_string())
}

fn ipw_theta(y: &Array1<f64>, d: &Array1<f64>, w: &Array2<f64>) -> Result<Array1<f64>, String> {
    let w_design = add_intercept(w);
    let pi = solve_least_squares_vec(&w_design, d)?;
    let ehat = w_design.dot(&pi);
    let resid = d - &ehat;
    let sigma2 = resid.dot(&resid) / (d.len() as f64);
    if sigma2 <= 1e-12 {
        return Err("AverageDerivative(ipw) residual variance is too small".to_string());
    }
    let gw = &resid / sigma2;
    let denom = gw.dot(d);
    if denom.abs() < 1e-12 {
        return Err("AverageDerivative(ipw) denominator is too small".to_string());
    }
    let beta = gw.dot(y) / denom;
    let mut theta = Array1::<f64>::zeros(pi.len() + 2);
    theta.slice_mut(ndarray::s![..pi.len()]).assign(&pi);
    theta[pi.len()] = sigma2.ln();
    theta[pi.len() + 1] = beta;
    Ok(theta)
}

fn ipw_moments(
    y: &Array1<f64>,
    d: &Array1<f64>,
    w: &Array2<f64>,
    theta: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let p = w.ncols() + 1;
    if theta.len() != p + 2 {
        return Err("AverageDerivative(ipw) parameter length mismatch".to_string());
    }
    let pi = theta.slice(ndarray::s![..p]).to_owned();
    let log_sigma2 = theta[p];
    let beta = theta[p + 1];
    let sigma2 = log_sigma2.exp();
    let w_design = add_intercept(w);
    let ehat = w_design.dot(&pi);
    let resid = d - &ehat;
    let gw = &resid / sigma2;
    let m1 = scale_rows(&w_design, &resid)?;
    let m2 = column_array(&(resid.mapv(|value| value * value - sigma2)));
    let m3 = column_array(&(gw * &(y - &(d * beta))));
    concatenate(Axis(1), &[m1.view(), m2.view(), m3.view()])
        .map_err(|_| "failed to build AverageDerivative(ipw) moments".to_string())
}

fn dr_theta(y: &Array1<f64>, d: &Array1<f64>, w: &Array2<f64>) -> Result<Array1<f64>, String> {
    let w_design = add_intercept(w);
    let pi = solve_least_squares_vec(&w_design, d)?;
    let ehat = w_design.dot(&pi);
    let resid = d - &ehat;
    let sigma2 = resid.dot(&resid) / (d.len() as f64);
    if sigma2 <= 1e-12 {
        return Err("AverageDerivative(dr) residual variance is too small".to_string());
    }
    let mu = mean_controls(w)?;
    let wc = center_controls(w, &mu)?;
    let rx = build_ob_design(&wc, d)?;
    let z = build_dr_instruments(&wc, d, &(&resid / sigma2))?;
    let lambda = fit_linear_iv(&rx, &z, y)?;
    let mut theta = Array1::<f64>::zeros(pi.len() + 1 + mu.len() + lambda.len());
    let mut offset = 0usize;
    theta
        .slice_mut(ndarray::s![offset..offset + pi.len()])
        .assign(&pi);
    offset += pi.len();
    theta[offset] = sigma2.ln();
    offset += 1;
    theta
        .slice_mut(ndarray::s![offset..offset + mu.len()])
        .assign(&mu);
    offset += mu.len();
    theta.slice_mut(ndarray::s![offset..]).assign(&lambda);
    Ok(theta)
}

fn dr_moments(
    y: &Array1<f64>,
    d: &Array1<f64>,
    w: &Array2<f64>,
    theta: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let p = w.ncols();
    let pi_dim = p + 1;
    let lambda_dim = 2 * p + 2;
    let expected = pi_dim + 1 + p + lambda_dim;
    if theta.len() != expected {
        return Err("AverageDerivative(dr) parameter length mismatch".to_string());
    }

    let mut offset = 0usize;
    let pi = theta.slice(ndarray::s![offset..offset + pi_dim]).to_owned();
    offset += pi_dim;
    let log_sigma2 = theta[offset];
    offset += 1;
    let mu = theta.slice(ndarray::s![offset..offset + p]).to_owned();
    offset += p;
    let lambda = theta.slice(ndarray::s![offset..]).to_owned();

    let sigma2 = log_sigma2.exp();
    let w_design = add_intercept(w);
    let ehat = w_design.dot(&pi);
    let resid = d - &ehat;
    let wc = center_controls(w, &mu)?;
    let rx = build_ob_design(&wc, d)?;
    let z = build_dr_instruments(&wc, d, &(&resid / sigma2))?;
    let outcome_resid = y - &rx.dot(&lambda);

    let m1 = scale_rows(&w_design, &resid)?;
    let m2 = column_array(&(resid.mapv(|value| value * value - sigma2)));
    let m3 = wc;
    let m4 = scale_rows(&z, &outcome_resid)?;
    concatenate(Axis(1), &[m1.view(), m2.view(), m3.view(), m4.view()])
        .map_err(|_| "failed to build AverageDerivative(dr) moments".to_string())
}

#[pyclass]
pub struct EPLM {
    fd_eps: f64,
    theta: Option<Array1<f64>>,
    y: Option<Array1<f64>>,
    d: Option<Array1<f64>>,
    w: Option<Array2<f64>>,
}

#[pymethods]
impl EPLM {
    #[new]
    #[pyo3(signature = (fd_eps=1e-6))]
    fn new(fd_eps: f64) -> PyResult<Self> {
        if !fd_eps.is_finite() || fd_eps <= 0.0 {
            return Err(PyValueError::new_err(
                "fd_eps must be a positive finite float",
            ));
        }
        Ok(Self {
            fd_eps,
            theta: None,
            y: None,
            d: None,
            w: None,
        })
    }

    fn fit(
        &mut self,
        y: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        w: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let y = to_array1(&y);
        let d = to_array1(&d);
        let w = to_array2(&w);
        if y.len() != d.len() || w.nrows() != y.len() {
            return Err(PyValueError::new_err("row count mismatch"));
        }
        validate_finite_1d("y", &y).map_err(PyValueError::new_err)?;
        validate_finite_1d("d", &d).map_err(PyValueError::new_err)?;
        validate_finite_2d("w", &w).map_err(PyValueError::new_err)?;
        let theta = eplm_theta(&y, &d, &w).map_err(PyValueError::new_err)?;
        self.theta = Some(theta);
        self.y = Some(y);
        self.d = Some(d);
        self.w = Some(w);
        Ok(())
    }

    #[pyo3(signature = (vcov=None, lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: Option<&str>,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("EPLM model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let d = self
            .d
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let w = self
            .w
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let vcov = vcov.unwrap_or("hc1");
        let cluster_ids = clusters.as_ref().map(to_array1_i64);

        let moments = eplm_moments(y, d, w, theta).map_err(PyValueError::new_err)?;
        let jac = numerical_mean_jacobian(theta, self.fd_eps, |candidate| {
            eplm_moments(y, d, w, candidate)
        })
        .map_err(PyValueError::new_err)?;
        let cov = exact_identified_covariance(&moments, &jac, vcov, lags, cluster_ids.as_ref())
            .map_err(PyValueError::new_err)?;

        let p = w.ncols() + 1;
        let nuisance = theta.slice(ndarray::s![..p]).to_owned();
        let coef = theta[p];
        let se = cov[[p, p]].abs().sqrt();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", coef)?;
        dict.set_item("se", se)?;
        dict.set_item(
            "vcov",
            pyarray2_from_f64(py, &cov.slice(ndarray::s![p..=p, p..=p]).to_owned()),
        )?;
        dict.set_item("nuisance_coef", pyarray1_from_f64(py, &nuisance))?;
        Ok(dict.into())
    }
}

#[pyclass]
pub struct AverageDerivative {
    method: String,
    fd_eps: f64,
    theta: Option<Array1<f64>>,
    y: Option<Array1<f64>>,
    d: Option<Array1<f64>>,
    w: Option<Array2<f64>>,
}

#[pymethods]
impl AverageDerivative {
    #[new]
    #[pyo3(signature = (method="dr", fd_eps=1e-6))]
    fn new(method: &str, fd_eps: f64) -> PyResult<Self> {
        if !fd_eps.is_finite() || fd_eps <= 0.0 {
            return Err(PyValueError::new_err(
                "fd_eps must be a positive finite float",
            ));
        }
        match method {
            "ob" | "ipw" | "dr" => Ok(Self {
                method: method.to_string(),
                fd_eps,
                theta: None,
                y: None,
                d: None,
                w: None,
            }),
            _ => Err(PyValueError::new_err(
                "method must be one of {'ob', 'ipw', 'dr'}",
            )),
        }
    }

    fn fit(
        &mut self,
        y: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        w: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let y = to_array1(&y);
        let d = to_array1(&d);
        let w = to_array2(&w);
        if y.len() != d.len() || w.nrows() != y.len() {
            return Err(PyValueError::new_err("row count mismatch"));
        }
        validate_finite_1d("y", &y).map_err(PyValueError::new_err)?;
        validate_finite_1d("d", &d).map_err(PyValueError::new_err)?;
        validate_finite_2d("w", &w).map_err(PyValueError::new_err)?;

        let theta = match self.method.as_str() {
            "ob" => ob_theta(&y, &d, &w),
            "ipw" => ipw_theta(&y, &d, &w),
            "dr" => dr_theta(&y, &d, &w),
            _ => unreachable!(),
        }
        .map_err(PyValueError::new_err)?;

        self.theta = Some(theta);
        self.y = Some(y);
        self.d = Some(d);
        self.w = Some(w);
        Ok(())
    }

    #[pyo3(signature = (vcov=None, lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: Option<&str>,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("AverageDerivative model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let d = self
            .d
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let w = self
            .w
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let vcov = vcov.unwrap_or("hc1");
        let cluster_ids = clusters.as_ref().map(to_array1_i64);

        let (moments, jac, coef_index) = match self.method.as_str() {
            "ob" => {
                let moments = ob_moments(y, d, w, theta).map_err(PyValueError::new_err)?;
                let jac = numerical_mean_jacobian(theta, self.fd_eps, |candidate| {
                    ob_moments(y, d, w, candidate)
                })
                .map_err(PyValueError::new_err)?;
                (moments, jac, theta.len() - 1)
            }
            "ipw" => {
                let moments = ipw_moments(y, d, w, theta).map_err(PyValueError::new_err)?;
                let jac = numerical_mean_jacobian(theta, self.fd_eps, |candidate| {
                    ipw_moments(y, d, w, candidate)
                })
                .map_err(PyValueError::new_err)?;
                (moments, jac, theta.len() - 1)
            }
            "dr" => {
                let moments = dr_moments(y, d, w, theta).map_err(PyValueError::new_err)?;
                let jac = numerical_mean_jacobian(theta, self.fd_eps, |candidate| {
                    dr_moments(y, d, w, candidate)
                })
                .map_err(PyValueError::new_err)?;
                (moments, jac, theta.len() - 1)
            }
            _ => unreachable!(),
        };

        let cov = exact_identified_covariance(&moments, &jac, vcov, lags, cluster_ids.as_ref())
            .map_err(PyValueError::new_err)?;
        let coef = theta[coef_index];
        let se = cov[[coef_index, coef_index]].abs().sqrt();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("method", self.method.clone())?;
        dict.set_item("coef", coef)?;
        dict.set_item("se", se)?;
        dict.set_item(
            "vcov",
            pyarray2_from_f64(
                py,
                &cov.slice(ndarray::s![
                    coef_index..=coef_index,
                    coef_index..=coef_index
                ])
                .to_owned(),
            ),
        )?;
        Ok(dict.into())
    }
}

#[pyclass]
pub struct PartiallyLinearDML {
    penalties: Array1<f64>,
    cv: usize,
    n_folds: usize,
    seed: u64,
    coef: Option<f64>,
    y: Option<Array1<f64>>,
    d: Option<Array1<f64>>,
    x: Option<Array2<f64>>,
    l_hat: Option<Array1<f64>>,
    m_hat: Option<Array1<f64>>,
    outcome_penalties: Option<Array1<f64>>,
    treatment_penalties: Option<Array1<f64>>,
}

#[pymethods]
impl PartiallyLinearDML {
    #[new]
    #[pyo3(signature = (penalty=None, cv=5, n_folds=5, seed=42))]
    fn new(
        py: Python<'_>,
        penalty: Option<Py<PyAny>>,
        cv: usize,
        n_folds: usize,
        seed: u64,
    ) -> PyResult<Self> {
        let penalties = match penalty {
            Some(value) => parse_penalties(value.bind(py))?,
            None => Array1::from_vec(vec![1.0]),
        };
        if cv < 2 {
            return Err(PyValueError::new_err("cv must be at least 2"));
        }
        if n_folds < 2 {
            return Err(PyValueError::new_err("n_folds must be at least 2"));
        }
        Ok(Self {
            penalties,
            cv,
            n_folds,
            seed,
            coef: None,
            y: None,
            d: None,
            x: None,
            l_hat: None,
            m_hat: None,
            outcome_penalties: None,
            treatment_penalties: None,
        })
    }

    fn fit(
        &mut self,
        y: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let y = to_array1(&y);
        let d = to_array1(&d);
        let x = to_array2(&x);
        if y.len() != d.len() || x.nrows() != y.len() {
            return Err(PyValueError::new_err("row count mismatch"));
        }
        validate_finite_1d("y", &y).map_err(PyValueError::new_err)?;
        validate_finite_1d("d", &d).map_err(PyValueError::new_err)?;
        validate_finite_2d("x", &x).map_err(PyValueError::new_err)?;

        let splits =
            make_kfold_splits(y.len(), self.n_folds, self.seed).map_err(PyValueError::new_err)?;
        let mut l_hat = Array1::<f64>::zeros(y.len());
        let mut m_hat = Array1::<f64>::zeros(y.len());
        let mut outcome_penalties = Array1::<f64>::zeros(splits.len());
        let mut treatment_penalties = Array1::<f64>::zeros(splits.len());

        for (fold, (train_idx, test_idx)) in splits.iter().enumerate() {
            let x_train = take_rows(&x, train_idx);
            let y_train = take_rows_vec(&y, train_idx);
            let d_train = take_rows_vec(&d, train_idx);
            let x_test = take_rows(&x, test_idx);

            let (outcome_params, outcome_penalty, _) =
                select_ridge_penalty(&x_train, &y_train, &self.penalties, self.cv)
                    .map_err(PyValueError::new_err)?;
            let (treat_params, treat_penalty, _) =
                select_ridge_penalty(&x_train, &d_train, &self.penalties, self.cv)
                    .map_err(PyValueError::new_err)?;

            let y_pred = ridge_predict(&x_test, &outcome_params).map_err(PyValueError::new_err)?;
            let d_pred = ridge_predict(&x_test, &treat_params).map_err(PyValueError::new_err)?;
            for (local, idx) in test_idx.iter().enumerate() {
                l_hat[*idx] = y_pred[local];
                m_hat[*idx] = d_pred[local];
            }
            outcome_penalties[fold] = outcome_penalty;
            treatment_penalties[fold] = treat_penalty;
        }

        let d_resid = &d - &m_hat;
        let y_resid = &y - &l_hat;
        let denom = d_resid.dot(&d_resid);
        if denom.abs() < 1e-12 {
            return Err(PyValueError::new_err(
                "PartiallyLinearDML residualized treatment has near-zero variation",
            ));
        }
        let coef = d_resid.dot(&y_resid) / denom;

        self.coef = Some(coef);
        self.y = Some(y);
        self.d = Some(d);
        self.x = Some(x);
        self.l_hat = Some(l_hat);
        self.m_hat = Some(m_hat);
        self.outcome_penalties = Some(outcome_penalties);
        self.treatment_penalties = Some(treatment_penalties);
        Ok(())
    }

    #[pyo3(signature = (vcov=None, lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: Option<&str>,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let coef = self
            .coef
            .ok_or_else(|| PyValueError::new_err("PartiallyLinearDML model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let d = self
            .d
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let l_hat = self
            .l_hat
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No nuisance predictions stored"))?;
        let m_hat = self
            .m_hat
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No nuisance predictions stored"))?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let vcov = vcov.unwrap_or("hc1");

        let d_resid = d - m_hat;
        let y_resid = y - l_hat;
        let score = column_array(&(d_resid.clone() * (y_resid - &(d_resid.clone() * coef))));
        let jac = Array2::from_elem((1, 1), -d_resid.dot(&d_resid) / (d_resid.len() as f64));
        let cov = exact_identified_covariance(&score, &jac, vcov, lags, cluster_ids.as_ref())
            .map_err(PyValueError::new_err)?;
        let se = cov[[0, 0]].abs().sqrt();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("coef", coef)?;
        dict.set_item("se", se)?;
        dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        if let Some(penalties) = &self.outcome_penalties {
            dict.set_item("outcome_penalties", pyarray1_from_f64(py, penalties))?;
        }
        if let Some(penalties) = &self.treatment_penalties {
            dict.set_item("treatment_penalties", pyarray1_from_f64(py, penalties))?;
        }
        Ok(dict.into())
    }
}

#[pyclass]
pub struct AIPW {
    penalties: Array1<f64>,
    cv: usize,
    n_folds: usize,
    seed: u64,
    propensity_clip: f64,
    ate: Option<f64>,
    y: Option<Array1<f64>>,
    d: Option<Array1<f64>>,
    x: Option<Array2<f64>>,
    mu0_hat: Option<Array1<f64>>,
    mu1_hat: Option<Array1<f64>>,
    pi_hat: Option<Array1<f64>>,
    outcome0_penalties: Option<Array1<f64>>,
    outcome1_penalties: Option<Array1<f64>>,
    propensity_penalties: Option<Array1<f64>>,
}

#[pymethods]
impl AIPW {
    #[new]
    #[pyo3(signature = (penalty=None, cv=5, n_folds=5, propensity_clip=0.02, seed=42))]
    fn new(
        py: Python<'_>,
        penalty: Option<Py<PyAny>>,
        cv: usize,
        n_folds: usize,
        propensity_clip: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let penalties = match penalty {
            Some(value) => parse_penalties(value.bind(py))?,
            None => Array1::from_vec(vec![1.0]),
        };
        if cv < 2 {
            return Err(PyValueError::new_err("cv must be at least 2"));
        }
        if n_folds < 2 {
            return Err(PyValueError::new_err("n_folds must be at least 2"));
        }
        if !(0.0..0.5).contains(&propensity_clip) {
            return Err(PyValueError::new_err(
                "propensity_clip must lie in [0, 0.5)",
            ));
        }
        Ok(Self {
            penalties,
            cv,
            n_folds,
            seed,
            propensity_clip,
            ate: None,
            y: None,
            d: None,
            x: None,
            mu0_hat: None,
            mu1_hat: None,
            pi_hat: None,
            outcome0_penalties: None,
            outcome1_penalties: None,
            propensity_penalties: None,
        })
    }

    fn fit(
        &mut self,
        y: PyReadonlyArray1<f64>,
        d: PyReadonlyArray1<f64>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let y = to_array1(&y);
        let d = to_array1(&d);
        let x = to_array2(&x);
        if y.len() != d.len() || x.nrows() != y.len() {
            return Err(PyValueError::new_err("row count mismatch"));
        }
        validate_finite_1d("y", &y).map_err(PyValueError::new_err)?;
        validate_finite_1d("d", &d).map_err(PyValueError::new_err)?;
        validate_finite_2d("x", &x).map_err(PyValueError::new_err)?;
        validate_binary(&d).map_err(PyValueError::new_err)?;

        let splits =
            make_kfold_splits(y.len(), self.n_folds, self.seed).map_err(PyValueError::new_err)?;
        let mut mu0_hat = Array1::<f64>::zeros(y.len());
        let mut mu1_hat = Array1::<f64>::zeros(y.len());
        let mut pi_hat = Array1::<f64>::zeros(y.len());
        let mut outcome0_penalties = Array1::<f64>::zeros(splits.len());
        let mut outcome1_penalties = Array1::<f64>::zeros(splits.len());
        let mut propensity_penalties = Array1::<f64>::zeros(splits.len());

        for (fold, (train_idx, test_idx)) in splits.iter().enumerate() {
            let x_train = take_rows(&x, train_idx);
            let y_train = take_rows_vec(&y, train_idx);
            let d_train = take_rows_vec(&d, train_idx);
            let x_test = take_rows(&x, test_idx);

            let treated_train: Vec<usize> =
                (0..d_train.len()).filter(|i| d_train[*i] == 1.0).collect();
            let control_train: Vec<usize> =
                (0..d_train.len()).filter(|i| d_train[*i] == 0.0).collect();
            if treated_train.is_empty() || control_train.is_empty() {
                return Err(PyValueError::new_err(
                    "each training fold must contain both treated and control observations",
                ));
            }

            let x_treat = take_rows(&x_train, &treated_train);
            let y_treat = take_rows_vec(&y_train, &treated_train);
            let x_control = take_rows(&x_train, &control_train);
            let y_control = take_rows_vec(&y_train, &control_train);

            let (mu1_params, mu1_penalty, _) =
                select_ridge_penalty(&x_treat, &y_treat, &self.penalties, self.cv)
                    .map_err(PyValueError::new_err)?;
            let (mu0_params, mu0_penalty, _) =
                select_ridge_penalty(&x_control, &y_control, &self.penalties, self.cv)
                    .map_err(PyValueError::new_err)?;
            let (pi_params, pi_penalty, _) =
                select_ridge_penalty(&x_train, &d_train, &self.penalties, self.cv)
                    .map_err(PyValueError::new_err)?;

            let mu1_pred = ridge_predict(&x_test, &mu1_params).map_err(PyValueError::new_err)?;
            let mu0_pred = ridge_predict(&x_test, &mu0_params).map_err(PyValueError::new_err)?;
            let pi_pred = ridge_predict(&x_test, &pi_params).map_err(PyValueError::new_err)?;

            for (local, idx) in test_idx.iter().enumerate() {
                mu1_hat[*idx] = mu1_pred[local];
                mu0_hat[*idx] = mu0_pred[local];
                pi_hat[*idx] =
                    pi_pred[local].clamp(self.propensity_clip, 1.0 - self.propensity_clip);
            }

            outcome0_penalties[fold] = mu0_penalty;
            outcome1_penalties[fold] = mu1_penalty;
            propensity_penalties[fold] = pi_penalty;
        }

        let pseudo = &mu1_hat - &mu0_hat + &(&d * &((y.clone() - &mu1_hat) / &pi_hat))
            - &((&(1.0 - &d)) * &((y.clone() - &mu0_hat) / &(1.0 - &pi_hat)));
        let ate = pseudo
            .mean()
            .ok_or_else(|| PyValueError::new_err("empty pseudo-outcome"))?;

        self.ate = Some(ate);
        self.y = Some(y);
        self.d = Some(d);
        self.x = Some(x);
        self.mu0_hat = Some(mu0_hat);
        self.mu1_hat = Some(mu1_hat);
        self.pi_hat = Some(pi_hat);
        self.outcome0_penalties = Some(outcome0_penalties);
        self.outcome1_penalties = Some(outcome1_penalties);
        self.propensity_penalties = Some(propensity_penalties);
        Ok(())
    }

    #[pyo3(signature = (vcov=None, lags=None, clusters=None))]
    fn summary<'py>(
        &self,
        py: Python<'py>,
        vcov: Option<&str>,
        lags: Option<usize>,
        clusters: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<Py<PyAny>> {
        let ate = self
            .ate
            .ok_or_else(|| PyValueError::new_err("AIPW model is not fitted"))?;
        let y = self
            .y
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let d = self
            .d
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No training data stored"))?;
        let mu0_hat = self
            .mu0_hat
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No nuisance predictions stored"))?;
        let mu1_hat = self
            .mu1_hat
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No nuisance predictions stored"))?;
        let pi_hat = self
            .pi_hat
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("No nuisance predictions stored"))?;
        let cluster_ids = clusters.as_ref().map(to_array1_i64);
        let vcov = vcov.unwrap_or("hc1");

        let pseudo = mu1_hat - mu0_hat + &(d * &((y - mu1_hat) / pi_hat))
            - &((1.0 - d) * &((y - mu0_hat) / &(1.0 - pi_hat)));
        let score = column_array(&(pseudo - ate));
        let jac = Array2::from_elem((1, 1), -1.0);
        let cov = exact_identified_covariance(&score, &jac, vcov, lags, cluster_ids.as_ref())
            .map_err(PyValueError::new_err)?;
        let se = cov[[0, 0]].abs().sqrt();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("ate", ate)?;
        dict.set_item("se", se)?;
        dict.set_item("vcov", pyarray2_from_f64(py, &cov))?;
        if let Some(penalties) = &self.outcome0_penalties {
            dict.set_item("outcome0_penalties", pyarray1_from_f64(py, penalties))?;
        }
        if let Some(penalties) = &self.outcome1_penalties {
            dict.set_item("outcome1_penalties", pyarray1_from_f64(py, penalties))?;
        }
        if let Some(penalties) = &self.propensity_penalties {
            dict.set_item("propensity_penalties", pyarray1_from_f64(py, penalties))?;
        }
        Ok(dict.into())
    }
}
