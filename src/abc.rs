use crate::utils::{
    diag_sqrt, invert_matrix, solve_least_squares_vec, to_array1, to_array2, to_array2_u32,
};
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Clone)]
enum ColumnKind {
    Intercept,
    Continuous,
    Categorical {
        k: usize,
        level: u32,
    },
    ContinuousCategorical {
        j: usize,
        k: usize,
        level: u32,
    },
    CategoricalCategorical {
        a: usize,
        b: usize,
        level_a: u32,
        level_b: u32,
    },
}

#[pyclass]
pub struct ABCOLS {
    theta: Option<Array1<f64>>,
    se: Option<Array1<f64>>,
    fitted: Option<Array1<f64>>,
    residuals: Option<Array1<f64>>,
    x_full: Option<Array2<f64>>,
    constraints: Option<Array2<f64>>,
    q: Option<Array2<f64>>,
    z: Option<Array2<f64>>,
    column_names: Vec<String>,
    column_kinds: Vec<ColumnKind>,
    constraint_names: Vec<String>,
    continuous_means: Vec<f64>,
    n_levels: Vec<usize>,
    cont_cat_interactions: Vec<(usize, usize)>,
    cat_cat_interactions: Vec<(usize, usize)>,
    center_continuous: bool,
    sigma2: f64,
    df_resid: usize,
    rank: usize,
    max_constraint_violation: f64,
}

#[pymethods]
impl ABCOLS {
    #[new]
    pub fn new() -> Self {
        Self {
            theta: None,
            se: None,
            fitted: None,
            residuals: None,
            x_full: None,
            constraints: None,
            q: None,
            z: None,
            column_names: Vec::new(),
            column_kinds: Vec::new(),
            constraint_names: Vec::new(),
            continuous_means: Vec::new(),
            n_levels: Vec::new(),
            cont_cat_interactions: Vec::new(),
            cat_cat_interactions: Vec::new(),
            center_continuous: true,
            sigma2: f64::NAN,
            df_resid: 0,
            rank: 0,
            max_constraint_violation: f64::NAN,
        }
    }

    #[pyo3(signature = (y, x, categories, cont_cat_interactions=None, cat_cat_interactions=None, center_continuous=true))]
    pub fn fit(
        &mut self,
        y: PyReadonlyArray1<f64>,
        x: PyReadonlyArray2<f64>,
        categories: PyReadonlyArray2<u32>,
        cont_cat_interactions: Option<Vec<(usize, usize)>>,
        cat_cat_interactions: Option<Vec<(usize, usize)>>,
        center_continuous: bool,
    ) -> PyResult<()> {
        *self = Self::new();
        let y = to_array1(&y);
        let x_raw = to_array2(&x);
        let cats = to_array2_u32(&categories);
        let n = y.len();
        if x_raw.nrows() != n || cats.nrows() != n {
            return Err(PyValueError::new_err(
                "x, categories, and y must have the same number of rows",
            ));
        }
        if y.iter().any(|v| !v.is_finite()) || x_raw.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err("y and x must contain finite values"));
        }
        let p_cont = x_raw.ncols();
        let p_cat = cats.ncols();
        if p_cat == 0 {
            return Err(PyValueError::new_err(
                "ABCOLS requires at least one categorical column; use OLS otherwise",
            ));
        }
        let cont_cat = cont_cat_interactions.unwrap_or_default();
        let cat_cat = cat_cat_interactions.unwrap_or_default();
        for (j, k) in &cont_cat {
            if *j >= p_cont || *k >= p_cat {
                return Err(PyValueError::new_err(
                    "continuous:categorical interaction index out of bounds",
                ));
            }
        }
        for (a, b) in &cat_cat {
            if *a >= p_cat || *b >= p_cat || a == b {
                return Err(PyValueError::new_err(
                    "categorical:categorical interaction indices must be distinct and in bounds",
                ));
            }
        }

        let (x_centered, means) = center_x(&x_raw, center_continuous);
        let n_levels = infer_levels(&cats)?;
        let level_weights = compute_level_weights(&cats, &n_levels);
        let cell_weights = compute_cell_weights(&cats, &n_levels, &cat_cat)?;
        let (x_full, column_kinds, column_names) =
            build_design(&x_centered, &cats, &n_levels, &cont_cat, &cat_cat)?;
        let (constraints, constraint_names) = build_constraints(
            &column_kinds,
            &n_levels,
            &level_weights,
            &cell_weights,
            &cont_cat,
            &cat_cat,
        );
        let q = null_space(&constraints)?;
        let z = x_full.dot(&q);
        if z.ncols() == 0 || n <= z.ncols() {
            return Err(PyValueError::new_err(
                "design has no residual degrees of freedom after ABC constraints",
            ));
        }
        let phi = solve_least_squares_vec(&z, &y).map_err(PyRuntimeError::new_err)?;
        let theta = q.dot(&phi);
        let fitted = x_full.dot(&theta);
        let residuals = &y - &fitted;
        let df_resid = n - z.ncols();
        let sigma2 = residuals.dot(&residuals) / df_resid as f64;
        let ztz_inv = invert_matrix(&z.t().dot(&z)).map_err(PyRuntimeError::new_err)?;
        let v_phi = ztz_inv.mapv(|v| v * sigma2);
        let v_theta = q.dot(&v_phi).dot(&q.t());
        let se = diag_sqrt(&v_theta).map_err(PyValueError::new_err)?;
        let violation = if constraints.nrows() == 0 {
            0.0
        } else {
            constraints
                .dot(&theta)
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()))
        };

        self.theta = Some(theta);
        self.se = Some(se);
        self.fitted = Some(fitted);
        self.residuals = Some(residuals);
        self.x_full = Some(x_full);
        self.constraints = Some(constraints);
        self.q = Some(q);
        self.z = Some(z);
        self.column_names = column_names;
        self.column_kinds = column_kinds;
        self.constraint_names = constraint_names;
        self.continuous_means = means;
        self.n_levels = n_levels;
        self.cont_cat_interactions = cont_cat;
        self.cat_cat_interactions = cat_cat;
        self.center_continuous = center_continuous;
        self.sigma2 = sigma2;
        self.df_resid = df_resid;
        self.rank = self.z.as_ref().unwrap().ncols();
        self.max_constraint_violation = violation;
        Ok(())
    }

    pub fn predict(
        &self,
        x: PyReadonlyArray2<f64>,
        categories: PyReadonlyArray2<u32>,
    ) -> PyResult<Vec<f64>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fit"))?;
        let x = to_array2(&x);
        let cats = to_array2_u32(&categories);
        if x.nrows() != cats.nrows() {
            return Err(PyValueError::new_err(
                "x and categories must have the same number of rows",
            ));
        }
        if x.ncols() != self.continuous_means.len() || cats.ncols() != self.n_levels.len() {
            return Err(PyValueError::new_err(
                "new data column count does not match fitted model",
            ));
        }
        let mut xc = x.clone();
        if self.center_continuous {
            for j in 0..xc.ncols() {
                for i in 0..xc.nrows() {
                    xc[[i, j]] -= self.continuous_means[j];
                }
            }
        }
        validate_category_bounds(&cats, &self.n_levels)?;
        let (x_full, _, _) = build_design(
            &xc,
            &cats,
            &self.n_levels,
            &self.cont_cat_interactions,
            &self.cat_cat_interactions,
        )?;
        Ok(x_full.dot(theta).to_vec())
    }

    pub fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let theta = self
            .theta
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fit"))?;
        let se = self.se.as_ref().unwrap();
        let dict = PyDict::new(py);
        dict.set_item("coef", theta.to_vec())?;
        dict.set_item("se", se.to_vec())?;
        dict.set_item("column_names", self.column_names.clone())?;
        dict.set_item("constraint_names", self.constraint_names.clone())?;
        dict.set_item("sigma2", self.sigma2)?;
        dict.set_item("df_resid", self.df_resid)?;
        dict.set_item("rank", self.rank)?;
        dict.set_item("max_constraint_violation", self.max_constraint_violation)?;
        dict.set_item("continuous_means", self.continuous_means.clone())?;
        dict.set_item("n_levels", self.n_levels.clone())?;
        Ok(dict)
    }

    pub fn fitted_values(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fit"))?
            .to_vec())
    }

    pub fn residuals(&self) -> PyResult<Vec<f64>> {
        Ok(self
            .residuals
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fit"))?
            .to_vec())
    }

    pub fn design_matrix(&self) -> PyResult<Vec<Vec<f64>>> {
        let x = self
            .x_full
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fit"))?;
        Ok(x.outer_iter().map(|row| row.to_vec()).collect())
    }

    pub fn constraint_matrix(&self) -> PyResult<Vec<Vec<f64>>> {
        let a = self
            .constraints
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("model is not fit"))?;
        Ok(a.outer_iter().map(|row| row.to_vec()).collect())
    }

    pub fn column_names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, &self.column_names)
    }
}

fn center_x(x: &Array2<f64>, center: bool) -> (Array2<f64>, Vec<f64>) {
    let mut out = x.clone();
    let mut means = vec![0.0; x.ncols()];
    if center {
        for j in 0..x.ncols() {
            let mean = x.column(j).sum() / x.nrows() as f64;
            means[j] = mean;
            for i in 0..x.nrows() {
                out[[i, j]] -= mean;
            }
        }
    }
    (out, means)
}

fn infer_levels(cats: &Array2<u32>) -> PyResult<Vec<usize>> {
    let mut levels = Vec::with_capacity(cats.ncols());
    for k in 0..cats.ncols() {
        let max_level = cats.column(k).iter().copied().max().unwrap_or(0) as usize;
        if max_level >= cats.nrows() {
            return Err(PyValueError::new_err(format!(
                "categorical column {k} must use contiguous observed codes starting at 0"
            )));
        }
        let mut counts = vec![0usize; max_level + 1];
        for &value in cats.column(k) {
            counts[value as usize] += 1;
        }
        if let Some(empty) = counts.iter().position(|c| *c == 0) {
            return Err(PyValueError::new_err(format!(
                "empty level {empty} detected for categorical column {k}; levels must be contiguous observed codes starting at 0"
            )));
        }
        levels.push(max_level + 1);
    }
    Ok(levels)
}

fn validate_category_bounds(cats: &Array2<u32>, n_levels: &[usize]) -> PyResult<()> {
    for k in 0..cats.ncols() {
        for &value in cats.column(k) {
            if value as usize >= n_levels[k] {
                return Err(PyValueError::new_err(format!(
                    "unseen level {value} for categorical column {k}"
                )));
            }
        }
    }
    Ok(())
}

fn compute_level_weights(cats: &Array2<u32>, n_levels: &[usize]) -> Vec<Vec<f64>> {
    let n = cats.nrows() as f64;
    let mut weights = Vec::with_capacity(cats.ncols());
    for k in 0..cats.ncols() {
        let mut counts = vec![0.0; n_levels[k]];
        for &value in cats.column(k) {
            counts[value as usize] += 1.0 / n;
        }
        weights.push(counts);
    }
    weights
}

fn compute_cell_weights(
    cats: &Array2<u32>,
    n_levels: &[usize],
    interactions: &[(usize, usize)],
) -> PyResult<Vec<((usize, usize), Array2<f64>)>> {
    let n = cats.nrows() as f64;
    let mut out = Vec::new();
    for &(a, b) in interactions {
        let mut w = Array2::<f64>::zeros((n_levels[a], n_levels[b]));
        for i in 0..cats.nrows() {
            w[[cats[[i, a]] as usize, cats[[i, b]] as usize]] += 1.0 / n;
        }
        for la in 0..n_levels[a] {
            for lb in 0..n_levels[b] {
                if w[[la, lb]] == 0.0 {
                    return Err(PyValueError::new_err(format!(
                        "empty cell detected for categorical interaction {a}:{b} at levels {la}:{lb}"
                    )));
                }
            }
        }
        out.push(((a, b), w));
    }
    Ok(out)
}

fn build_design(
    x: &Array2<f64>,
    cats: &Array2<u32>,
    n_levels: &[usize],
    cont_cat: &[(usize, usize)],
    cat_cat: &[(usize, usize)],
) -> PyResult<(Array2<f64>, Vec<ColumnKind>, Vec<String>)> {
    validate_category_bounds(cats, n_levels)?;
    let n = x.nrows();
    let mut width = 1usize
        .checked_add(x.ncols())
        .ok_or_else(|| PyValueError::new_err("design width overflow"))?;
    for extra in n_levels
        .iter()
        .copied()
        .chain(cont_cat.iter().map(|&(_, k)| n_levels[k]))
        .chain(
            cat_cat
                .iter()
                .map(|&(a, b)| n_levels[a].saturating_mul(n_levels[b])),
        )
    {
        width = width
            .checked_add(extra)
            .ok_or_else(|| PyValueError::new_err("design width overflow"))?;
    }
    crate::validation::validate_dense_capacity("ABCOLS design", n, width)
        .map_err(PyValueError::new_err)?;
    crate::validation::validate_dense_capacity("ABCOLS constraint workspace", width, width)
        .map_err(PyValueError::new_err)?;
    let mut cols: Vec<Array1<f64>> = Vec::new();
    let mut kinds = Vec::new();
    let mut names = Vec::new();

    cols.push(Array1::ones(n));
    kinds.push(ColumnKind::Intercept);
    names.push("Intercept".to_string());

    for j in 0..x.ncols() {
        cols.push(x.column(j).to_owned());
        kinds.push(ColumnKind::Continuous);
        names.push(format!("x{j}"));
    }
    for k in 0..cats.ncols() {
        for level in 0..n_levels[k] {
            let col = cats
                .column(k)
                .mapv(|v| if v as usize == level { 1.0 } else { 0.0 });
            cols.push(col);
            kinds.push(ColumnKind::Categorical {
                k,
                level: level as u32,
            });
            names.push(format!("c{k}[{level}]"));
        }
    }
    for &(j, k) in cont_cat {
        for level in 0..n_levels[k] {
            let mut col = Array1::<f64>::zeros(n);
            for i in 0..n {
                if cats[[i, k]] as usize == level {
                    col[i] = x[[i, j]];
                }
            }
            cols.push(col);
            kinds.push(ColumnKind::ContinuousCategorical {
                j,
                k,
                level: level as u32,
            });
            names.push(format!("x{j}:c{k}[{level}]"));
        }
    }
    for &(a, b) in cat_cat {
        for la in 0..n_levels[a] {
            for lb in 0..n_levels[b] {
                let mut col = Array1::<f64>::zeros(n);
                for i in 0..n {
                    if cats[[i, a]] as usize == la && cats[[i, b]] as usize == lb {
                        col[i] = 1.0;
                    }
                }
                cols.push(col);
                kinds.push(ColumnKind::CategoricalCategorical {
                    a,
                    b,
                    level_a: la as u32,
                    level_b: lb as u32,
                });
                names.push(format!("c{a}[{la}]:c{b}[{lb}]"));
            }
        }
    }

    let mut mat = Array2::<f64>::zeros((n, cols.len()));
    for (j, col) in cols.iter().enumerate() {
        mat.column_mut(j).assign(col);
    }
    Ok((mat, kinds, names))
}

fn build_constraints(
    kinds: &[ColumnKind],
    n_levels: &[usize],
    level_weights: &[Vec<f64>],
    cell_weights: &[((usize, usize), Array2<f64>)],
    cont_cat: &[(usize, usize)],
    cat_cat: &[(usize, usize)],
) -> (Array2<f64>, Vec<String>) {
    let p = kinds.len();
    let mut rows: Vec<Array1<f64>> = Vec::new();
    let mut names = Vec::new();

    for k in 0..n_levels.len() {
        let mut row = Array1::<f64>::zeros(p);
        for (idx, kind) in kinds.iter().enumerate() {
            if let ColumnKind::Categorical { k: kk, level } = kind {
                if *kk == k {
                    row[idx] = level_weights[k][*level as usize];
                }
            }
        }
        rows.push(row);
        names.push(format!("ABC: c{k} main effect"));
    }

    for &(j, k) in cont_cat {
        let mut row = Array1::<f64>::zeros(p);
        for (idx, kind) in kinds.iter().enumerate() {
            if let ColumnKind::ContinuousCategorical {
                j: jj,
                k: kk,
                level,
            } = kind
            {
                if *jj == j && *kk == k {
                    row[idx] = level_weights[k][*level as usize];
                }
            }
        }
        rows.push(row);
        names.push(format!("ABC: x{j}:c{k} interaction"));
    }

    for &(a, b) in cat_cat {
        let weights = cell_weights
            .iter()
            .find(|((aa, bb), _)| *aa == a && *bb == b)
            .map(|(_, w)| w)
            .expect("cell weights missing");
        for la in 0..n_levels[a] {
            let mut row = Array1::<f64>::zeros(p);
            for (idx, kind) in kinds.iter().enumerate() {
                if let ColumnKind::CategoricalCategorical {
                    a: aa,
                    b: bb,
                    level_a,
                    level_b,
                } = kind
                {
                    if *aa == a && *bb == b && *level_a as usize == la {
                        row[idx] = weights[[la, *level_b as usize]];
                    }
                }
            }
            rows.push(row);
            names.push(format!("ABC: c{a}:c{b} margin c{a}={la}"));
        }
        for lb in 1..n_levels[b] {
            let mut row = Array1::<f64>::zeros(p);
            for (idx, kind) in kinds.iter().enumerate() {
                if let ColumnKind::CategoricalCategorical {
                    a: aa,
                    b: bb,
                    level_a,
                    level_b,
                } = kind
                {
                    if *aa == a && *bb == b && *level_b as usize == lb {
                        row[idx] = weights[[*level_a as usize, lb]];
                    }
                }
            }
            rows.push(row);
            names.push(format!("ABC: c{a}:c{b} margin c{b}={lb}"));
        }
    }

    let mut mat = Array2::<f64>::zeros((rows.len(), p));
    for (i, row) in rows.iter().enumerate() {
        mat.row_mut(i).assign(row);
    }
    (mat, names)
}

fn null_space(a: &Array2<f64>) -> PyResult<Array2<f64>> {
    let m = a.nrows();
    let p = a.ncols();
    if m == 0 {
        let mut q = Array2::<f64>::zeros((p, p));
        for i in 0..p {
            q[[i, i]] = 1.0;
        }
        return Ok(q);
    }
    let data: Vec<f64> = a.iter().copied().collect();
    let dm = DMatrix::from_row_slice(m, p, &data);
    let gram = dm.transpose() * dm;
    let eig = SymmetricEigen::new(gram);
    let max_eval = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let tol = (m.max(p) as f64) * f64::EPSILON * max_eval.max(1.0) * 100.0;
    let null_indices: Vec<usize> = eig
        .eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| if value.abs() <= tol { Some(idx) } else { None })
        .collect();
    if null_indices.is_empty() {
        return Err(PyValueError::new_err(
            "ABC constraints leave no free coefficients",
        ));
    }
    let mut q = Array2::<f64>::zeros((p, null_indices.len()));
    for (col, &src_col) in null_indices.iter().enumerate() {
        for row in 0..p {
            q[[row, col]] = eig.eigenvectors[(row, src_col)];
        }
    }
    Ok(q)
}
