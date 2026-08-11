use crate::utils::{pyarray1_from_f64, pyarray2_from_f64, to_array2};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2, Axis};
use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DEFAULT_VARIMAX_MAX_ITER: usize = 100;
const DEFAULT_VARIMAX_TOL: f64 = 1e-5;
const DEFAULT_L1_TOL: f64 = 1e-7;
const EPSILON_ROT: f64 = 0.05;

fn validate_loadings(loadings: &Array2<f64>, name: &str) -> PyResult<()> {
    if loadings.nrows() == 0 || loadings.ncols() == 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must have nonzero dimensions"
        )));
    }
    if loadings.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn eye(n: usize) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        out[[i, i]] = 1.0;
    }
    out
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

fn min_symmetric_eigenvalue(a: &Array2<f64>) -> f64 {
    let dm = array2_to_dmatrix(a);
    let eig = dm.symmetric_eigen();
    eig.eigenvalues
        .iter()
        .fold(f64::INFINITY, |acc, value| acc.min(*value))
}

fn varimax_objective(loadings: &Array2<f64>) -> f64 {
    let n = loadings.nrows() as f64;
    let mut objective = 0.0;
    for col in loadings.columns() {
        let sum_sq = col.iter().map(|value| value * value).sum::<f64>();
        let sum_fourth = col.iter().map(|value| value.powi(4)).sum::<f64>();
        objective += sum_fourth - sum_sq.powi(2) / n;
    }
    objective
}

fn normalize_rows(loadings: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut normalized = loadings.clone();
    let mut scale = Array1::<f64>::ones(loadings.nrows());
    for i in 0..loadings.nrows() {
        let norm = loadings
            .row(i)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if norm > 1e-12 {
            scale[i] = norm;
            for j in 0..loadings.ncols() {
                normalized[[i, j]] /= norm;
            }
        }
    }
    (normalized, scale)
}

fn apply_row_scale(loadings: &mut Array2<f64>, scale: &Array1<f64>) {
    for i in 0..loadings.nrows() {
        for j in 0..loadings.ncols() {
            loadings[[i, j]] *= scale[i];
        }
    }
}

fn varimax_impl(
    loadings: &Array2<f64>,
    normalize: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Array2<f64>, Array2<f64>, f64, usize, bool)> {
    if loadings.ncols() == 1 {
        let rotation = eye(1);
        return Ok((
            loadings.clone(),
            rotation,
            varimax_objective(loadings),
            0,
            true,
        ));
    }

    let (x, row_scale) = if normalize {
        normalize_rows(loadings)
    } else {
        (loadings.clone(), Array1::<f64>::ones(loadings.nrows()))
    };
    let p = x.nrows() as f64;
    let k = x.ncols();
    let mut rotation = eye(k);
    let mut old_sum = 0.0;
    let mut n_iter = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let z = x.dot(&rotation);
        let mut adjusted = z.mapv(|value| value.powi(3));
        let col_sums = z.map_axis(Axis(0), |col| col.iter().map(|v| v * v).sum::<f64>());
        for i in 0..z.nrows() {
            for j in 0..z.ncols() {
                adjusted[[i, j]] -= z[[i, j]] * col_sums[j] / p;
            }
        }
        let b = x.t().dot(&adjusted);
        let svd = array2_to_dmatrix(&b).svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| PyValueError::new_err("SVD failed in varimax update"))?;
        let vt = svd
            .v_t
            .ok_or_else(|| PyValueError::new_err("SVD failed in varimax update"))?;
        rotation = dmatrix_to_array2(&(u * vt), k, k);
        let singular_sum = svd.singular_values.iter().sum::<f64>();
        if old_sum != 0.0 && singular_sum < old_sum * (1.0 + tol) {
            converged = true;
            break;
        }
        old_sum = singular_sum;
    }

    let mut rotated = x.dot(&rotation);
    if normalize {
        apply_row_scale(&mut rotated, &row_scale);
    }
    Ok((
        rotated.clone(),
        rotation,
        varimax_objective(&rotated),
        n_iter,
        converged,
    ))
}

fn l1_grid_size(factors: usize) -> usize {
    match factors {
        0 | 1 => 0,
        2 => 500,
        3 => 1000,
        4 => 2000,
        5 => 4000,
        6..=8 => 6000,
        _ => 10000,
    }
}

fn normalize_vector(mut x: Vec<f64>) -> Vec<f64> {
    let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm == 0.0 {
        x[0] = 1.0;
        return x;
    }
    for value in &mut x {
        *value /= norm;
    }
    x
}

fn random_unit_vector(dim: usize, rng: &mut StdRng) -> Vec<f64> {
    loop {
        let x: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-12 {
            return x.into_iter().map(|v| v / norm).collect();
        }
    }
}

fn spherical_to_cartesian(theta: &[f64]) -> Vec<f64> {
    let r = theta.len() + 1;
    let mut out = vec![0.0; r];
    out[0] = theta[0].cos();
    for k in 1..(r - 1) {
        let sin_prod = theta[..k].iter().map(|angle| angle.sin()).product::<f64>();
        out[k] = sin_prod * theta[k].cos();
    }
    out[r - 1] = theta.iter().map(|angle| angle.sin()).product::<f64>();
    out
}

fn cartesian_to_spherical(direction: &[f64]) -> Vec<f64> {
    let r = direction.len();
    let mut theta = vec![0.0; r - 1];
    for i in 0..(r - 1) {
        let tail_norm = direction[(i + 1)..]
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        theta[i] = tail_norm.atan2(direction[i]);
    }
    theta
}

fn l1_direction_objective(loadings: &Array2<f64>, theta: &[f64]) -> f64 {
    let direction = spherical_to_cartesian(theta);
    let mut total = 0.0;
    for i in 0..loadings.nrows() {
        let mut value = 0.0;
        for j in 0..loadings.ncols() {
            value += loadings[[i, j]] * direction[j];
        }
        total += value.abs();
    }
    total
}

fn nelder_mead_direction(
    loadings: &Array2<f64>,
    start: &[f64],
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, f64, usize, bool) {
    let dim = start.len();
    let step = 0.05_f64.max(tol.sqrt());
    let mut simplex = Vec::with_capacity(dim + 1);
    simplex.push(start.to_vec());
    for d in 0..dim {
        let mut point = start.to_vec();
        point[d] += step;
        simplex.push(point);
    }
    let mut values: Vec<f64> = simplex
        .iter()
        .map(|point| l1_direction_objective(loadings, point))
        .collect();

    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;
    let mut converged = false;
    let mut iters = 0;

    for iter in 0..max_iter {
        iters = iter + 1;
        let mut order: Vec<usize> = (0..simplex.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        simplex = order.iter().map(|&idx| simplex[idx].clone()).collect();
        values = order.iter().map(|&idx| values[idx]).collect();

        let spread = values[dim] - values[0];
        let param_spread = simplex[1..]
            .iter()
            .map(|point| {
                point
                    .iter()
                    .zip(simplex[0].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0, f64::max);
        if spread.abs() <= tol && param_spread <= tol.sqrt() {
            converged = true;
            break;
        }

        let mut centroid = vec![0.0; dim];
        for point in simplex.iter().take(dim) {
            for d in 0..dim {
                centroid[d] += point[d] / dim as f64;
            }
        }

        let reflect: Vec<f64> = (0..dim)
            .map(|d| centroid[d] + alpha * (centroid[d] - simplex[dim][d]))
            .collect();
        let reflect_value = l1_direction_objective(loadings, &reflect);

        if reflect_value < values[0] {
            let expand: Vec<f64> = (0..dim)
                .map(|d| centroid[d] + gamma * (reflect[d] - centroid[d]))
                .collect();
            let expand_value = l1_direction_objective(loadings, &expand);
            if expand_value < reflect_value {
                simplex[dim] = expand;
                values[dim] = expand_value;
            } else {
                simplex[dim] = reflect;
                values[dim] = reflect_value;
            }
        } else if reflect_value < values[dim - 1] {
            simplex[dim] = reflect;
            values[dim] = reflect_value;
        } else {
            let contract = if reflect_value < values[dim] {
                (0..dim)
                    .map(|d| centroid[d] + rho * (reflect[d] - centroid[d]))
                    .collect::<Vec<_>>()
            } else {
                (0..dim)
                    .map(|d| centroid[d] + rho * (simplex[dim][d] - centroid[d]))
                    .collect::<Vec<_>>()
            };
            let contract_value = l1_direction_objective(loadings, &contract);
            if contract_value < values[dim] {
                simplex[dim] = contract;
                values[dim] = contract_value;
            } else {
                for i in 1..=dim {
                    for d in 0..dim {
                        simplex[i][d] = simplex[0][d] + sigma * (simplex[i][d] - simplex[0][d]);
                    }
                    values[i] = l1_direction_objective(loadings, &simplex[i]);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..simplex.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let best = simplex[order[0]].clone();
    (
        spherical_to_cartesian(&best),
        values[order[0]],
        iters,
        converged,
    )
}

#[derive(Clone)]
struct Candidate {
    direction: Vec<f64>,
    l1_norm: f64,
    count: usize,
    l0_norm: usize,
}

fn l1_norm_for_direction(loadings: &Array2<f64>, direction: &[f64]) -> f64 {
    let mut total = 0.0;
    for i in 0..loadings.nrows() {
        let mut value = 0.0;
        for j in 0..loadings.ncols() {
            value += loadings[[i, j]] * direction[j];
        }
        total += value.abs();
    }
    total
}

fn rotated_column(loadings: &Array2<f64>, direction: &[f64]) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(loadings.nrows());
    for i in 0..loadings.nrows() {
        let mut value = 0.0;
        for j in 0..loadings.ncols() {
            value += loadings[[i, j]] * direction[j];
        }
        out[i] = value;
    }
    out
}

fn cluster_directions(loadings: &Array2<f64>, directions: &[Vec<f64>]) -> Vec<Candidate> {
    let factors = loadings.ncols();
    let mut sorted: Vec<Candidate> = directions
        .iter()
        .map(|direction| {
            let mut signed = direction.clone();
            let sign = if signed[0] < 0.0 { -1.0 } else { 1.0 };
            for value in &mut signed {
                *value *= sign;
            }
            Candidate {
                l1_norm: l1_norm_for_direction(loadings, &signed),
                direction: signed,
                count: 1,
                l0_norm: 0,
            }
        })
        .collect();
    sorted.sort_by(|a, b| a.l1_norm.total_cmp(&b.l1_norm));

    let mut clustered: Vec<Candidate> = Vec::new();
    for candidate in sorted {
        let mut matched = false;
        for existing in &mut clustered {
            let distance = existing
                .direction
                .iter()
                .zip(candidate.direction.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                / (factors as f64).sqrt();
            if distance < EPSILON_ROT {
                existing.count += 1;
                matched = true;
                break;
            }
        }
        if !matched {
            clustered.push(candidate);
        }
    }

    let threshold = 1.0 / (loadings.nrows() as f64).ln();
    for candidate in &mut clustered {
        let col = rotated_column(loadings, &candidate.direction);
        candidate.l0_norm = col.iter().filter(|value| value.abs() < threshold).count();
    }
    clustered
}

fn consolidate_candidates(loadings: &Array2<f64>, candidates: &[Candidate]) -> Array2<f64> {
    let factors = loadings.ncols();
    let mut chosen: Vec<Vec<f64>> = Vec::new();
    for candidate in candidates {
        let mut temp = Array2::<f64>::zeros((loadings.nrows(), chosen.len() + 1));
        for (j, direction) in chosen.iter().enumerate() {
            temp.column_mut(j)
                .assign(&rotated_column(loadings, direction));
        }
        temp.column_mut(chosen.len())
            .assign(&rotated_column(loadings, &candidate.direction));

        let gram = temp.t().dot(&temp);
        let min_temp = min_symmetric_eigenvalue(&gram) / loadings.nrows() as f64;
        let non_singular = if chosen.is_empty() {
            true
        } else {
            let mut old = Array2::<f64>::zeros((loadings.nrows(), chosen.len()));
            for (j, direction) in chosen.iter().enumerate() {
                old.column_mut(j)
                    .assign(&rotated_column(loadings, direction));
            }
            let old_min = min_symmetric_eigenvalue(&old.t().dot(&old)) / loadings.nrows() as f64;
            min_temp > (1.0 / factors as f64).sqrt() / 3.0 && min_temp > old_min / 4.0
        };
        if non_singular {
            chosen.push(candidate.direction.clone());
        }
        if chosen.len() == factors {
            break;
        }
    }

    for j in 0..factors {
        if chosen.len() == factors {
            break;
        }
        let mut unit = vec![0.0; factors];
        unit[j] = 1.0;
        if !chosen.iter().any(|direction| {
            direction
                .iter()
                .zip(unit.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
                < 1e-10
        }) {
            chosen.push(unit);
        }
    }

    let mut rotation = Array2::<f64>::zeros((factors, factors));
    for (j, direction) in chosen.iter().take(factors).enumerate() {
        for i in 0..factors {
            rotation[[i, j]] = direction[i];
        }
    }
    rotation
}

fn l1_sparse_impl(
    loadings: &Array2<f64>,
    n_starts: Option<usize>,
    seed: Option<u64>,
    max_iter: Option<usize>,
    tol: f64,
    initial_directions: Option<Array2<f64>>,
) -> PyResult<(
    Array2<f64>,
    Array2<f64>,
    Vec<Candidate>,
    Array2<f64>,
    Array1<f64>,
    usize,
)> {
    let factors = loadings.ncols();
    if factors < 2 {
        return Err(PyValueError::new_err(
            "l1_sparse_rotation requires at least two columns",
        ));
    }

    let mut candidate_directions: Vec<Vec<f64>> = Vec::new();
    let mut candidate_objectives = Vec::new();
    let mut total_iters = 0;

    if let Some(directions) = initial_directions {
        if directions.nrows() != factors {
            return Err(PyValueError::new_err(
                "initial_directions must have loadings.shape[1] rows",
            ));
        }
        for col in directions.columns() {
            let direction = normalize_vector(col.iter().copied().collect());
            candidate_objectives.push(l1_norm_for_direction(loadings, &direction));
            candidate_directions.push(direction);
        }
    } else {
        let draws = n_starts.unwrap_or_else(|| l1_grid_size(factors));
        if draws == 0 {
            return Err(PyValueError::new_err("n_starts must be positive"));
        }
        let mut rng = StdRng::seed_from_u64(seed.unwrap_or(0x51A7E5));
        let iter_limit = max_iter.unwrap_or(200 * (factors - 1));
        for _ in 0..draws {
            let start = random_unit_vector(factors, &mut rng);
            let theta = cartesian_to_spherical(&start);
            let (direction, value, iters, _) =
                nelder_mead_direction(loadings, &theta, iter_limit, tol);
            candidate_objectives.push(value);
            candidate_directions.push(normalize_vector(direction));
            total_iters += iters;
        }
    }

    let mut candidates = cluster_directions(loadings, &candidate_directions);
    let draws = candidate_directions.len().max(1);
    let mut non_outliers: Vec<Candidate> = candidates
        .iter()
        .filter(|candidate| candidate.count as f64 / draws as f64 >= 0.005)
        .cloned()
        .collect();
    if non_outliers.len() < factors {
        non_outliers = candidates.clone();
    }
    non_outliers.sort_by(|a, b| {
        b.l0_norm
            .cmp(&a.l0_norm)
            .then_with(|| a.l1_norm.total_cmp(&b.l1_norm))
    });
    let rotation = consolidate_candidates(loadings, &non_outliers);
    let rotated = loadings.dot(&rotation);

    candidates.sort_by(|a, b| a.l1_norm.total_cmp(&b.l1_norm));
    let mut directions_out = Array2::<f64>::zeros((factors, candidate_directions.len()));
    for (j, direction) in candidate_directions.iter().enumerate() {
        for i in 0..factors {
            directions_out[[i, j]] = direction[i];
        }
    }
    let objectives_out = Array1::from_vec(candidate_objectives);

    Ok((
        rotated,
        rotation,
        non_outliers,
        directions_out,
        objectives_out,
        total_iters,
    ))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn inv_norm_cdf(p: f64) -> f64 {
    // Peter J. Acklam's rational approximation.
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let plow = 0.02425;
    let phigh = 1.0 - plow;
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p > phigh {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

#[pyfunction]
#[pyo3(signature = (loadings, *, normalize=false, max_iter=DEFAULT_VARIMAX_MAX_ITER, tol=DEFAULT_VARIMAX_TOL))]
pub fn varimax_rotation<'py>(
    py: Python<'py>,
    loadings: PyReadonlyArray2<f64>,
    normalize: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyAny>> {
    let loadings = to_array2(&loadings);
    validate_loadings(&loadings, "loadings")?;
    if max_iter == 0 {
        return Err(PyValueError::new_err("max_iter must be positive"));
    }
    if tol <= 0.0 || !tol.is_finite() {
        return Err(PyValueError::new_err("tol must be positive and finite"));
    }
    let (rotated, rotation, objective, n_iter, converged) =
        varimax_impl(&loadings, normalize, max_iter, tol)?;
    let dict = PyDict::new(py);
    dict.set_item("rotated", pyarray2_from_f64(py, &rotated))?;
    dict.set_item("rotation", pyarray2_from_f64(py, &rotation))?;
    dict.set_item("objective", objective)?;
    dict.set_item("n_iter", n_iter)?;
    dict.set_item("converged", converged)?;
    dict.set_item("normalize", normalize)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (loadings, *, n_starts=None, seed=None, max_iter=None, tol=DEFAULT_L1_TOL, initial_directions=None))]
pub fn l1_sparse_rotation<'py>(
    py: Python<'py>,
    loadings: PyReadonlyArray2<f64>,
    n_starts: Option<usize>,
    seed: Option<u64>,
    max_iter: Option<usize>,
    tol: f64,
    initial_directions: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Py<PyAny>> {
    let loadings = to_array2(&loadings);
    validate_loadings(&loadings, "loadings")?;
    if let Some(draws) = n_starts {
        if draws == 0 {
            return Err(PyValueError::new_err("n_starts must be positive"));
        }
    }
    if let Some(iter) = max_iter {
        if iter == 0 {
            return Err(PyValueError::new_err("max_iter must be positive"));
        }
    }
    if tol <= 0.0 || !tol.is_finite() {
        return Err(PyValueError::new_err("tol must be positive and finite"));
    }
    let initial_directions = initial_directions.map(|directions| to_array2(&directions));
    let (rotated, rotation, candidates, candidate_directions, candidate_objective, total_iters) =
        l1_sparse_impl(&loadings, n_starts, seed, max_iter, tol, initial_directions)?;
    let fval = Array1::from_vec(
        candidates
            .iter()
            .map(|candidate| candidate.l1_norm)
            .collect(),
    );
    let sol_frequency = Array1::from_vec(
        candidates
            .iter()
            .map(|candidate| candidate.count as f64)
            .collect(),
    );
    let l0_norm = Array1::from_vec(
        candidates
            .iter()
            .map(|candidate| candidate.l0_norm as f64)
            .collect(),
    );

    let dict = PyDict::new(py);
    dict.set_item("rotated", pyarray2_from_f64(py, &rotated))?;
    dict.set_item("rotation", pyarray2_from_f64(py, &rotation))?;
    dict.set_item("fval", pyarray1_from_f64(py, &fval))?;
    dict.set_item("sol_frequency", pyarray1_from_f64(py, &sol_frequency))?;
    dict.set_item("l0_norm", pyarray1_from_f64(py, &l0_norm))?;
    dict.set_item(
        "candidate_directions",
        pyarray2_from_f64(py, &candidate_directions),
    )?;
    dict.set_item(
        "candidate_objective",
        pyarray1_from_f64(py, &candidate_objective),
    )?;
    dict.set_item("total_iterations", total_iters)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (loadings, threshold=None))]
pub fn count_small_loadings<'py>(
    py: Python<'py>,
    loadings: PyReadonlyArray2<f64>,
    threshold: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let loadings = to_array2(&loadings);
    validate_loadings(&loadings, "loadings")?;
    let threshold = threshold.unwrap_or_else(|| 1.0 / (loadings.nrows() as f64).ln());
    if threshold <= 0.0 || !threshold.is_finite() {
        return Err(PyValueError::new_err(
            "threshold must be positive and finite",
        ));
    }
    let mut out = Array1::<f64>::zeros(loadings.ncols());
    for j in 0..loadings.ncols() {
        out[j] = loadings
            .column(j)
            .iter()
            .filter(|value| value.abs() < threshold)
            .count() as f64;
    }
    Ok(pyarray1_from_f64(py, &out))
}

#[pyfunction]
#[pyo3(signature = (loadings, threshold=None, alpha=0.05, gamma0=0.03))]
pub fn local_factor_diagnostic<'py>(
    py: Python<'py>,
    loadings: PyReadonlyArray2<f64>,
    threshold: Option<f64>,
    alpha: f64,
    gamma0: f64,
) -> PyResult<Py<PyAny>> {
    let loadings = to_array2(&loadings);
    validate_loadings(&loadings, "loadings")?;
    if !(0.0..1.0).contains(&alpha) {
        return Err(PyValueError::new_err("alpha must be in (0, 1)"));
    }
    if gamma0 < 0.0 || !gamma0.is_finite() {
        return Err(PyValueError::new_err(
            "gamma0 must be nonnegative and finite",
        ));
    }
    let n = loadings.nrows();
    let h_n = threshold.unwrap_or_else(|| 1.0 / (n as f64).ln());
    if h_n <= 0.0 || !h_n.is_finite() {
        return Err(PyValueError::new_err(
            "threshold must be positive and finite",
        ));
    }
    let expected_small = erf_approx(h_n / 2.0_f64.sqrt());
    let c_gamma = inv_norm_cdf(1.0 - alpha / 2.0);
    let gamma = gamma0
        + expected_small
        + c_gamma * ((expected_small * (1.0 - expected_small)) / n as f64).sqrt();
    let gamma_n = (gamma * n as f64).floor();
    let mut n_small = Array1::<f64>::zeros(loadings.ncols());
    for j in 0..loadings.ncols() {
        n_small[j] = loadings
            .column(j)
            .iter()
            .filter(|value| value.abs() < h_n)
            .count() as f64;
    }
    let most_small = n_small.iter().fold(0.0_f64, |acc, value| acc.max(*value));
    let dict = PyDict::new(py);
    dict.set_item("has_local_factors", most_small > gamma_n)?;
    dict.set_item("n_small", pyarray1_from_f64(py, &n_small))?;
    dict.set_item("gamma_n", gamma_n)?;
    dict.set_item("h_n", h_n)?;
    dict.set_item("expected_small", expected_small)?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (loadings, axis=0))]
pub fn inverse_participation_ratio<'py>(
    py: Python<'py>,
    loadings: PyReadonlyArray2<f64>,
    axis: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let loadings = to_array2(&loadings);
    validate_loadings(&loadings, "loadings")?;
    match axis {
        0 => {
            let mut out = Array1::<f64>::zeros(loadings.ncols());
            for j in 0..loadings.ncols() {
                out[j] = loadings.column(j).iter().map(|value| value.powi(4)).sum();
            }
            Ok(pyarray1_from_f64(py, &out))
        }
        1 => {
            let mut out = Array1::<f64>::zeros(loadings.nrows());
            for i in 0..loadings.nrows() {
                out[i] = loadings.row(i).iter().map(|value| value.powi(4)).sum();
            }
            Ok(pyarray1_from_f64(py, &out))
        }
        _ => Err(PyValueError::new_err("axis must be 0 or 1")),
    }
}

#[pyfunction]
pub fn cumulative_participation(loadings: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let loadings = to_array2(&loadings);
    validate_loadings(&loadings, "loadings")?;
    Ok(loadings
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|value| value * value).sum::<f64>().powi(2))
        .sum())
}
