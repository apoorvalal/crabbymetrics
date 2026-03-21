use nalgebra::DMatrix;
use ndarray::{concatenate, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub fn add_intercept(x: &Array2<f64>) -> Array2<f64> {
    let ones = Array2::ones((x.nrows(), 1));
    concatenate(Axis(1), &[ones.view(), x.view()]).expect("failed to add intercept")
}

pub fn invert_matrix(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("matrix is not square".to_string());
    }
    let data: Vec<f64> = a.iter().copied().collect();
    let mat = DMatrix::from_row_slice(n, n, &data);
    let inv = mat
        .try_inverse()
        .ok_or_else(|| "matrix is singular".to_string())?;
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            out[[i, j]] = inv[(i, j)];
        }
    }
    Ok(out)
}

pub fn hc1_cov(x: &Array2<f64>, residuals: &Array1<f64>) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let k = x.ncols();
    if residuals.len() != n {
        return Err("residual length mismatch".to_string());
    }

    let xtx = x.t().dot(x);
    let xtx_inv = invert_matrix(&xtx)?;

    let mut meat = Array2::<f64>::zeros((k, k));
    for i in 0..n {
        let xi = x.row(i);
        let u = residuals[i];
        let outer = xi
            .to_owned()
            .insert_axis(Axis(1))
            .dot(&xi.to_owned().insert_axis(Axis(0)));
        meat = meat + outer * (u * u);
    }

    let mut cov = xtx_inv.dot(&meat).dot(&xtx_inv);
    let scale = n as f64 / (n as f64 - k as f64);
    cov.mapv_inplace(|v| v * scale);
    Ok(cov)
}

pub fn diag_sqrt(a: &Array2<f64>) -> Array1<f64> {
    let n = a.nrows().min(a.ncols());
    let mut out = Array1::zeros(n);
    for i in 0..n {
        out[i] = a[[i, i]].abs().sqrt();
    }
    out
}

pub fn fisher_cov_binary(x: &Array2<f64>, probs: &Array1<f64>) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let k = x.ncols();
    if probs.len() != n {
        return Err("prob length mismatch".to_string());
    }

    let mut weighted = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let w = probs[i] * (1.0 - probs[i]);
        for j in 0..k {
            weighted[[i, j]] = x[[i, j]] * w.sqrt();
        }
    }

    let info = weighted.t().dot(&weighted);
    let mut info_reg = info.clone();
    for i in 0..k {
        info_reg[[i, i]] += 1e-8;
    }
    invert_matrix(&info_reg)
}

pub fn fisher_cov_poisson(x: &Array2<f64>, mu: &Array1<f64>) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let k = x.ncols();
    if mu.len() != n {
        return Err("mu length mismatch".to_string());
    }

    let mut weighted = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let w = mu[i].max(1e-12);
        for j in 0..k {
            weighted[[i, j]] = x[[i, j]] * w.sqrt();
        }
    }

    let info = weighted.t().dot(&weighted);
    let mut info_reg = info.clone();
    for i in 0..k {
        info_reg[[i, i]] += 1e-8;
    }
    invert_matrix(&info_reg)
}

pub fn fisher_cov_multinomial(x: &Array2<f64>, probs: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let k = x.ncols();
    let c = probs.ncols();
    if probs.nrows() != n {
        return Err("prob length mismatch".to_string());
    }

    let dim = k * c;
    let mut h = Array2::<f64>::zeros((dim, dim));

    for i in 0..n {
        let xi = x.row(i);
        let mut outer_x = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for s in 0..k {
                outer_x[[r, s]] = xi[r] * xi[s];
            }
        }

        for a in 0..c {
            for b in 0..c {
                let w = if a == b {
                    probs[[i, a]] * (1.0 - probs[[i, a]])
                } else {
                    -probs[[i, a]] * probs[[i, b]]
                };
                let row_offset = a * k;
                let col_offset = b * k;
                for r in 0..k {
                    for s in 0..k {
                        h[[row_offset + r, col_offset + s]] += w * outer_x[[r, s]];
                    }
                }
            }
        }
    }

    for i in 0..dim {
        h[[i, i]] += 1e-8;
    }

    invert_matrix(&h)
}

pub fn bootstrap_indices(n: usize, n_bootstrap: usize, seed: Option<u64>) -> Vec<Vec<usize>> {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };
    (0..n_bootstrap)
        .map(|_| (0..n).map(|_| rng.gen_range(0..n)).collect())
        .collect()
}

pub fn take_rows(x: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((idx.len(), x.ncols()));
    for (i, &row) in idx.iter().enumerate() {
        out.row_mut(i).assign(&x.row(row));
    }
    out
}

pub fn take_rows_vec(y: &Array1<f64>, idx: &[usize]) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(idx.len());
    for (i, &row) in idx.iter().enumerate() {
        out[i] = y[row];
    }
    out
}

pub fn take_rows_u32(x: &Array2<u32>, idx: &[usize]) -> Array2<u32> {
    let mut out = Array2::<u32>::zeros((idx.len(), x.ncols()));
    for (i, &row) in idx.iter().enumerate() {
        out.row_mut(i).assign(&x.row(row));
    }
    out
}

pub fn take_rows_i32(y: &Array1<i32>, idx: &[usize]) -> Array1<i32> {
    let mut out = Array1::<i32>::zeros(idx.len());
    for (i, &row) in idx.iter().enumerate() {
        out[i] = y[row];
    }
    out
}

pub fn to_array1(x: &PyReadonlyArray1<f64>) -> Array1<f64> {
    Array1::from_iter(x.as_array().iter().copied())
}

pub fn to_array1_i32(x: &PyReadonlyArray1<i32>) -> Array1<i32> {
    Array1::from_iter(x.as_array().iter().copied())
}

pub fn to_array2(x: &PyReadonlyArray2<f64>) -> Array2<f64> {
    let shape = x.shape();
    let mut data = Vec::with_capacity(shape[0] * shape[1]);
    for v in x.as_array().iter() {
        data.push(*v);
    }
    Array2::from_shape_vec((shape[0], shape[1]), data).expect("invalid shape")
}

pub fn to_array2_u32(x: &PyReadonlyArray2<u32>) -> Array2<u32> {
    let shape = x.shape();
    let mut data = Vec::with_capacity(shape[0] * shape[1]);
    for v in x.as_array().iter() {
        data.push(*v);
    }
    Array2::from_shape_vec((shape[0], shape[1]), data).expect("invalid shape")
}

pub fn pyarray1_from_f64<'py>(py: Python<'py>, data: &Array1<f64>) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_vec(py, data.to_vec())
}

pub fn pyarray1_from_i32<'py>(py: Python<'py>, data: &Array1<i32>) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_vec(py, data.to_vec())
}

pub fn pyarray2_from_f64<'py>(py: Python<'py>, data: &Array2<f64>) -> Bound<'py, PyArray2<f64>> {
    let vec2: Vec<Vec<f64>> = data.rows().into_iter().map(|row| row.to_vec()).collect();
    PyArray2::from_vec2(py, &vec2).expect("failed to build array")
}
