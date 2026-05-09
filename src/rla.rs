use crate::utils::{add_intercept, pyarray1_from_f64, pyarray2_from_f64, solve_least_squares_vec};
use nalgebra::DMatrix;
use ndarray::{s, Array1, Array2};
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct RandomizedSvdResult {
    pub u: Array2<f64>,
    pub singular_values: Array1<f64>,
    pub vt: Array2<f64>,
}

fn validate_finite_matrix(name: &str, a: &Array2<f64>) -> PyResult<()> {
    if a.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn validate_rank_params(
    rows: usize,
    cols: usize,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
) -> PyResult<usize> {
    let min_dim = rows.min(cols);
    if min_dim == 0 {
        return Err(PyValueError::new_err("matrix must have nonzero dimensions"));
    }
    if rank == 0 || rank > min_dim {
        return Err(PyValueError::new_err(
            "rank must be between 1 and min(matrix.shape)",
        ));
    }
    if power_iter > 10 {
        return Err(PyValueError::new_err("power_iter must be <= 10"));
    }
    Ok((rank + oversamples).min(min_dim))
}

fn array2_to_dmatrix(a: &Array2<f64>) -> DMatrix<f64> {
    let data: Vec<f64> = a.iter().copied().collect();
    DMatrix::from_row_slice(a.nrows(), a.ncols(), &data)
}

fn dmatrix_to_array2(m: &DMatrix<f64>, rows: usize, cols: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            out[[i, j]] = m[(i, j)];
        }
    }
    out
}

fn orthonormal_basis(y: &Array2<f64>, cols: usize) -> PyResult<Array2<f64>> {
    let dm = array2_to_dmatrix(y);
    let qr = dm.qr();
    let q = qr.q();
    let q_cols = cols.min(q.ncols());
    Ok(dmatrix_to_array2(
        &q.columns(0, q_cols).into_owned(),
        q.nrows(),
        q_cols,
    ))
}

fn rademacher_matrix(rows: usize, cols: usize, rng: &mut StdRng) -> Array2<f64> {
    let scale = 1.0 / (cols as f64).sqrt();
    Array2::from_shape_fn(
        (rows, cols),
        |_| {
            if rng.gen_bool(0.5) {
                scale
            } else {
                -scale
            }
        },
    )
}

pub fn randomized_range_finder(
    a: &Array2<f64>,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<Array2<f64>> {
    validate_finite_matrix("a", a)?;
    let sketch_cols = validate_rank_params(a.nrows(), a.ncols(), rank, oversamples, power_iter)?;
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xC0FFEE));
    let omega = rademacher_matrix(a.ncols(), sketch_cols, &mut rng);
    let mut y = a.dot(&omega);
    let mut q = orthonormal_basis(&y, sketch_cols)?;

    for _ in 0..power_iter {
        let z = a.t().dot(&q);
        let zq = orthonormal_basis(&z, sketch_cols)?;
        y = a.dot(&zq);
        q = orthonormal_basis(&y, sketch_cols)?;
    }

    Ok(q)
}

pub fn randomized_svd_impl(
    a: &Array2<f64>,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<RandomizedSvdResult> {
    validate_finite_matrix("a", a)?;
    let target_rank = rank;
    validate_rank_params(a.nrows(), a.ncols(), rank, oversamples, power_iter)?;
    let q = randomized_range_finder(a, rank, oversamples, power_iter, seed)?;
    let b = q.t().dot(a);
    let b_dm = array2_to_dmatrix(&b);
    let svd = b_dm.svd(true, true);
    let u_hat = svd
        .u
        .ok_or_else(|| PyValueError::new_err("SVD failed to return left singular vectors"))?;
    let vt_dm = svd
        .v_t
        .ok_or_else(|| PyValueError::new_err("SVD failed to return right singular vectors"))?;
    let k = target_rank.min(svd.singular_values.len());
    let u_hat_arr = dmatrix_to_array2(&u_hat.columns(0, k).into_owned(), u_hat.nrows(), k);
    let u = q.dot(&u_hat_arr);
    let mut singular_values = Array1::<f64>::zeros(k);
    for j in 0..k {
        singular_values[j] = svd.singular_values[j];
    }
    let vt = dmatrix_to_array2(&vt_dm.rows(0, k).into_owned(), k, vt_dm.ncols());

    Ok(RandomizedSvdResult {
        u,
        singular_values,
        vt,
    })
}

fn sketch_design_response(
    design: &Array2<f64>,
    y: &Array1<f64>,
    sketch_size: usize,
    seed: Option<u64>,
) -> PyResult<(Array2<f64>, Array1<f64>)> {
    let n = design.nrows();
    let p = design.ncols();
    if y.len() != n {
        return Err(PyValueError::new_err("x rows must match y length"));
    }
    if sketch_size < p {
        return Err(PyValueError::new_err(
            "sketch_size must be at least the number of design columns",
        ));
    }
    if sketch_size == 0 {
        return Err(PyValueError::new_err("sketch_size must be positive"));
    }

    // CountSketch row embedding: every original observation contributes once to a
    // signed bucket. This keeps the work O(n p), unlike a dense Gaussian/Rademacher
    // sketch which costs O(sketch_size * n * p), and is the right primitive for
    // tall econometric designs.
    let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0xBAD5EED));
    let mut sx = Array2::<f64>::zeros((sketch_size, p));
    let mut sy = Array1::<f64>::zeros(sketch_size);
    for i in 0..n {
        let bucket = rng.gen_range(0..sketch_size);
        let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
        for j in 0..p {
            sx[[bucket, j]] += sign * design[[i, j]];
        }
        sy[bucket] += sign * y[i];
    }
    Ok((sx, sy))
}

pub fn sketch_ols_params(
    x: &Array2<f64>,
    y: &Array1<f64>,
    fit_intercept: bool,
    sketch_size: usize,
    seed: Option<u64>,
) -> PyResult<Array1<f64>> {
    validate_finite_matrix("x", x)?;
    if y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("y must contain only finite values"));
    }
    let design = if fit_intercept {
        add_intercept(x)
    } else {
        x.clone()
    };
    let (sx, sy) = sketch_design_response(&design, y, sketch_size, seed)?;
    solve_least_squares_vec(&sx, &sy).map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (a, rank, oversamples=10, power_iter=1, seed=None))]
pub fn randomized_svd<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let a = crate::utils::to_array2(&a);
    let result = randomized_svd_impl(&a, rank, oversamples, power_iter, seed)?;
    let dict = PyDict::new(py);
    dict.set_item("u", pyarray2_from_f64(py, &result.u))?;
    dict.set_item(
        "singular_values",
        pyarray1_from_f64(py, &result.singular_values),
    )?;
    dict.set_item("vt", pyarray2_from_f64(py, &result.vt))?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (a, rank, oversamples=10, power_iter=1, seed=None))]
pub fn randomized_range<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f64>,
    rank: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = crate::utils::to_array2(&a);
    let q = randomized_range_finder(&a, rank, oversamples, power_iter, seed)?;
    Ok(pyarray2_from_f64(py, &q))
}

#[pyfunction]
#[pyo3(signature = (x, y, sketch_size, fit_intercept=true, seed=None))]
pub fn sketch_ols<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    sketch_size: usize,
    fit_intercept: bool,
    seed: Option<u64>,
) -> PyResult<Py<PyAny>> {
    let x = crate::utils::to_array2(&x);
    let y = crate::utils::to_array1(&y);
    let params = sketch_ols_params(&x, &y, fit_intercept, sketch_size, seed)?;
    let (intercept, coef) = if fit_intercept {
        (params[0], params.slice(s![1..]).to_owned())
    } else {
        (0.0, params)
    };
    let dict = PyDict::new(py);
    dict.set_item("intercept", intercept)?;
    dict.set_item("coef", pyarray1_from_f64(py, &coef))?;
    dict.set_item("sketch_size", sketch_size)?;
    dict.set_item("fit_intercept", fit_intercept)?;
    Ok(dict.into())
}
