use linfa_linalg::qr::{LeastSquaresQrInto, QRInto};
use ndarray::{concatenate, Array1, Array2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::BTreeMap;

pub(crate) use crate::validation::validate_sample_weight;

pub fn add_intercept(x: &Array2<f64>) -> Array2<f64> {
    let ones = Array2::ones((x.nrows(), 1));
    concatenate(Axis(1), &[ones.view(), x.view()]).expect("failed to add intercept")
}

pub fn sqrt_sample_weight(
    sample_weight: Option<&Array1<f64>>,
    n: usize,
) -> Result<Option<Array1<f64>>, String> {
    match sample_weight {
        Some(weights) => {
            validate_sample_weight(weights, n)?;
            Ok(Some(weights.mapv(|value| value.sqrt())))
        }
        None => Ok(None),
    }
}

pub fn scale_rows(x: &Array2<f64>, scale: &Array1<f64>) -> Result<Array2<f64>, String> {
    if x.nrows() != scale.len() {
        return Err("row scale length must match the number of rows".to_string());
    }
    let mut out = x.clone();
    for i in 0..x.nrows() {
        out.row_mut(i).mapv_inplace(|value| value * scale[i]);
    }
    Ok(out)
}

pub fn scale_vec(y: &Array1<f64>, scale: &Array1<f64>) -> Result<Array1<f64>, String> {
    if y.len() != scale.len() {
        return Err("vector scale length must match the number of observations".to_string());
    }
    let mut out = y.clone();
    for i in 0..y.len() {
        out[i] *= scale[i];
    }
    Ok(out)
}

pub fn invert_matrix(a: &Array2<f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err("matrix is not square".to_string());
    }
    a.to_owned()
        .qr_into()
        .and_then(|decomp| decomp.inverse())
        .map_err(|err| err.to_string())
}

pub fn solve_least_squares_vec(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let solution = a
        .to_owned()
        .least_squares_into(b.to_owned().insert_axis(Axis(1)))
        .map_err(|err| err.to_string())?;
    Ok(solution.remove_axis(Axis(1)))
}

pub fn solve_least_squares_mat(a: &Array2<f64>, b: &Array2<f64>) -> Result<Array2<f64>, String> {
    a.to_owned()
        .least_squares_into(b.to_owned())
        .map_err(|err| err.to_string())
}

pub fn default_newey_west_lags(n: usize) -> usize {
    ((4.0 * (n as f64 / 100.0).powf(2.0 / 9.0)).floor() as usize).max(1)
}

pub fn score_cov_iid(scores: &Array2<f64>) -> Array2<f64> {
    scores.t().dot(scores)
}

pub fn score_cov_newey_west(scores: &Array2<f64>, lags: usize) -> Array2<f64> {
    let n = scores.nrows();
    let mut cov = score_cov_iid(scores);
    if n <= 1 || lags == 0 {
        return cov;
    }

    let max_lag = lags.min(n - 1);
    for lag in 1..=max_lag {
        let weight = 1.0 - lag as f64 / (max_lag as f64 + 1.0);
        let lead = scores.slice(ndarray::s![lag.., ..]).to_owned();
        let lagged = scores.slice(ndarray::s![..(n - lag), ..]).to_owned();
        let gamma = lead.t().dot(&lagged);
        cov = cov + weight * (&gamma + &gamma.t().to_owned());
    }

    cov
}

pub fn score_cov_cluster(
    scores: &Array2<f64>,
    clusters: &Array1<i64>,
) -> Result<(Array2<f64>, usize), String> {
    let n = scores.nrows();
    let p = scores.ncols();
    if clusters.len() != n {
        return Err("clusters length must match the number of observations".to_string());
    }

    let mut grouped: BTreeMap<i64, Array1<f64>> = BTreeMap::new();
    for i in 0..n {
        let entry = grouped
            .entry(clusters[i])
            .or_insert_with(|| Array1::<f64>::zeros(p));
        *entry = &*entry + &scores.row(i).to_owned();
    }

    let n_clusters = grouped.len();
    let mut cov = Array2::<f64>::zeros((p, p));
    for summed in grouped.values() {
        let col = summed.clone().insert_axis(Axis(1));
        let row = summed.clone().insert_axis(Axis(0));
        cov = cov + col.dot(&row);
    }

    Ok((cov, n_clusters))
}

pub fn sandwich_cov_from_parameter_scores(
    scores: &Array2<f64>,
    vcov: &str,
    df_resid: f64,
    lags: Option<usize>,
    clusters: Option<&Array1<i64>>,
) -> Result<Array2<f64>, String> {
    let n = scores.nrows();
    if df_resid <= 0.0 {
        return Err("need positive residual degrees of freedom".to_string());
    }

    match vcov {
        "hc1" => Ok(score_cov_iid(scores) * (n as f64 / df_resid)),
        "newey_west" => Ok(score_cov_newey_west(
            scores,
            lags.unwrap_or_else(|| default_newey_west_lags(n)),
        ) * (n as f64 / df_resid)),
        "cluster" => {
            let cluster_ids = clusters
                .ok_or_else(|| "clusters must be provided for vcov='cluster'".to_string())?;
            let (cov, n_clusters) = score_cov_cluster(scores, cluster_ids)?;
            if n_clusters < 2 {
                return Err("cluster covariance requires at least two clusters".to_string());
            }
            let n_f64 = n as f64;
            let g_f64 = n_clusters as f64;
            let scale = (g_f64 / (g_f64 - 1.0)) * ((n_f64 - 1.0) / df_resid);
            Ok(cov * scale)
        }
        _ => Err("vcov must be one of {'hc1', 'vanilla', 'newey_west', 'cluster'}".to_string()),
    }
}

pub fn diag_sqrt(a: &Array2<f64>) -> Result<Array1<f64>, String> {
    if a.nrows() != a.ncols() {
        return Err("covariance matrix must be square".to_string());
    }
    let scale = a
        .iter()
        .filter(|value| value.is_finite())
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    if a.iter().any(|value| !value.is_finite()) {
        return Err("covariance matrix must contain only finite values".to_string());
    }
    let tolerance = 1e-12 * scale;
    let mut out = Array1::zeros(a.nrows());
    for i in 0..a.nrows() {
        let value = a[[i, i]];
        if value < -tolerance {
            return Err(format!(
                "covariance diagonal at index {} is negative ({})",
                i, value
            ));
        }
        out[i] = value.max(0.0).sqrt();
    }
    Ok(out)
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

pub fn qmle_cov_poisson(
    x: &Array2<f64>,
    y: &Array1<f64>,
    mu: &Array1<f64>,
) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let k = x.ncols();
    if y.len() != n {
        return Err("y length mismatch".to_string());
    }
    if mu.len() != n {
        return Err("mu length mismatch".to_string());
    }

    let bread = fisher_cov_poisson(x, mu)?;
    let mut scores = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        let resid = y[i] - mu[i];
        for j in 0..k {
            scores[[i, j]] = x[[i, j]] * resid;
        }
    }
    let meat = scores.t().dot(&scores);
    Ok(bread.dot(&meat).dot(&bread))
}

pub fn fisher_cov_multinomial(
    x: &Array2<f64>,
    probs: &Array2<f64>,
    reference_class: usize,
) -> Result<Array2<f64>, String> {
    let n = x.nrows();
    let k = x.ncols();
    let c = probs.ncols();
    if probs.nrows() != n {
        return Err("prob length mismatch".to_string());
    }

    if c < 2 {
        return Err("multinomial covariance requires at least two classes".to_string());
    }
    if reference_class >= c {
        return Err("reference class index is out of bounds".to_string());
    }

    let modeled_classes: Vec<usize> = (0..c).filter(|class| *class != reference_class).collect();
    let dim = k * (c - 1);
    let mut h = Array2::<f64>::zeros((dim, dim));

    for i in 0..n {
        let xi = x.row(i);
        let mut outer_x = Array2::<f64>::zeros((k, k));
        for r in 0..k {
            for s in 0..k {
                outer_x[[r, s]] = xi[r] * xi[s];
            }
        }

        for (a_idx, &a) in modeled_classes.iter().enumerate() {
            for (b_idx, &b) in modeled_classes.iter().enumerate() {
                let w = if a == b {
                    probs[[i, a]] * (1.0 - probs[[i, a]])
                } else {
                    -probs[[i, a]] * probs[[i, b]]
                };
                let row_offset = a_idx * k;
                let col_offset = b_idx * k;
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

pub fn to_array1_i64(x: &PyReadonlyArray1<i64>) -> Array1<i64> {
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
    if data.nrows() == 0 || data.ncols() == 0 {
        return PyArray2::zeros(py, [data.nrows(), data.ncols()], false);
    }
    let vec2: Vec<Vec<f64>> = data.rows().into_iter().map(|row| row.to_vec()).collect();
    PyArray2::from_vec2(py, &vec2).expect("failed to build array")
}

#[cfg(test)]
mod tests {
    use super::diag_sqrt;
    use ndarray::array;

    #[test]
    fn diag_sqrt_accepts_valid_covariance() {
        let se = diag_sqrt(&array![[4.0, 0.5], [0.5, 9.0]]).unwrap();
        assert_eq!(se.to_vec(), vec![2.0, 3.0]);
    }

    #[test]
    fn diag_sqrt_clips_only_roundoff_negative_values() {
        let se = diag_sqrt(&array![[-1e-14, 0.0], [0.0, 1.0]]).unwrap();
        assert_eq!(se.to_vec(), vec![0.0, 1.0]);
        assert!(diag_sqrt(&array![[-1e-4, 0.0], [0.0, 1.0]]).is_err());
    }

    #[test]
    fn diag_sqrt_rejects_nonfinite_or_nonsquare_covariance() {
        assert!(diag_sqrt(&array![[f64::NAN]]).is_err());
        assert!(diag_sqrt(&array![[1.0, 0.0]]).is_err());
    }
}
