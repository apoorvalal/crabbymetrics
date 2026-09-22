//! Untreated-only additive fixed-effect counterfactuals on an incomplete panel.
use super::linear::fit_fixed_effects_ols;
use crate::utils::{pyarray1_from_f64, pyarray2_from_f64, to_array2};
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

struct ImputationFit {
    counterfactual: Array2<f64>,
    effects: Array2<f64>,
    training: Array2<bool>,
    evaluation: Array2<bool>,
    coef: Array1<f64>,
    att: f64,
    att_by_unit: Array1<f64>,
    att_by_time: Array1<f64>,
    treated_weight: f64,
    untreated_rmse: f64,
    n_observed: usize,
    n_zero_weight: usize,
    n_reversing: usize,
    iterations: Vec<usize>,
    residual_norms: Vec<f64>,
}

/// Additive unit/time FE fitted exclusively on observed untreated cells.
/// Matrices use units x periods. No absorbing-adoption restriction is imposed.
#[pyclass]
pub struct FEImputation {
    tolerance: f64,
    max_iterations: usize,
    fitted: Option<ImputationFit>,
}

#[pymethods]
impl FEImputation {
    #[new]
    #[pyo3(signature = (tolerance=1e-10, max_iterations=1000))]
    fn new(tolerance: f64, max_iterations: usize) -> PyResult<Self> {
        if !tolerance.is_finite() || tolerance <= 0.0 || max_iterations == 0 {
            return Err(PyValueError::new_err(
                "tolerance and max_iterations must be positive",
            ));
        }
        Ok(Self {
            tolerance,
            max_iterations,
            fitted: None,
        })
    }

    #[pyo3(signature = (y, treatment, x=None, *, observed=None, sample_weight=None))]
    fn fit(
        &mut self,
        y: PyReadonlyArray2<f64>,
        treatment: PyReadonlyArray2<f64>,
        x: Option<PyReadonlyArray3<f64>>,
        observed: Option<PyReadonlyArray2<bool>>,
        sample_weight: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        self.fitted = None;
        let y = to_array2(&y);
        let d = to_array2(&treatment);
        let (n, t) = y.dim();
        if n == 0 || t == 0 || d.dim() != y.dim() || n > u32::MAX as usize || t > u32::MAX as usize
        {
            return Err(PyValueError::new_err(
                "y and treatment must be aligned nonempty units x periods matrices",
            ));
        }
        let obs = match observed {
            Some(a) => {
                let a = a.as_array();
                Array2::from_shape_vec((a.nrows(), a.ncols()), a.iter().copied().collect()).unwrap()
            }
            None => y.mapv(f64::is_finite),
        };
        let w = sample_weight
            .map(|a| to_array2(&a))
            .unwrap_or_else(|| Array2::ones((n, t)));
        if obs.dim() != y.dim() || w.dim() != y.dim() {
            return Err(PyValueError::new_err(
                "observed and sample_weight must match y.shape",
            ));
        }
        if w.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(PyValueError::new_err(
                "sample_weight must be finite and nonnegative",
            ));
        }
        let x = x.map(|a| a.as_array().to_owned());
        let k = x.as_ref().map_or(0, |a| a.shape()[2]);
        if x.as_ref()
            .is_some_and(|a| a.shape()[0] != n || a.shape()[1] != t)
        {
            return Err(PyValueError::new_err(
                "x must have shape (units, periods, covariates)",
            ));
        }
        let mut training = Array2::from_elem((n, t), false);
        let mut evaluation = training.clone();
        let mut cells = Vec::new();
        let mut train = Vec::new();
        let mut n_observed = 0;
        let mut n_zero_weight = 0;
        let mut n_reversing = 0;
        for i in 0..n {
            let mut previous = None;
            let mut reverses = false;
            for j in 0..t {
                if !obs[[i, j]] {
                    continue;
                }
                n_observed += 1;
                if !y[[i, j]].is_finite() || ![0.0, 1.0].contains(&d[[i, j]]) {
                    return Err(PyValueError::new_err(
                        "observed cells require finite y and binary treatment",
                    ));
                }
                if let Some(old) = previous {
                    if old == 1.0 && d[[i, j]] == 0.0 {
                        reverses = true;
                    }
                }
                previous = Some(d[[i, j]]);
                if w[[i, j]] == 0.0 {
                    n_zero_weight += 1;
                    continue;
                }
                if x.as_ref()
                    .is_some_and(|a| (0..k).any(|c| !a[[i, j, c]].is_finite()))
                {
                    return Err(PyValueError::new_err(
                        "positive-weight observed covariates must be finite",
                    ));
                }
                cells.push((i, j));
                if d[[i, j]] == 0.0 {
                    training[[i, j]] = true;
                    train.push((i, j));
                } else {
                    evaluation[[i, j]] = true;
                }
            }
            if reverses {
                n_reversing += 1;
            }
        }
        if train.is_empty() || !evaluation.iter().any(|v| *v) {
            return Err(PyValueError::new_err(
                "need observed positive-weight untreated and treated cells",
            ));
        }
        let design = |cells: &[(usize, usize)]| {
            Array2::from_shape_fn((cells.len(), k), |(r, c)| {
                let (i, j) = cells[r];
                x.as_ref().unwrap()[[i, j, c]]
            })
        };
        let categories = |cells: &[(usize, usize)]| {
            Array2::from_shape_fn((cells.len(), 2), |(r, c)| {
                if c == 0 {
                    cells[r].0 as u32
                } else {
                    cells[r].1 as u32
                }
            })
        };
        let train_y = Array1::from_iter(train.iter().map(|&(i, j)| y[[i, j]]));
        let train_w = Array1::from_iter(train.iter().map(|&(i, j)| w[[i, j]]));
        let fitted = fit_fixed_effects_ols(
            &design(&train),
            &train_y,
            &categories(&train),
            Some(&train_w),
            self.tolerance,
            self.max_iterations,
        )?;
        super::linear::validate_fe_prediction_design(
            &design(&train),
            &fitted.x_resid,
            Some(&train_w),
        )?;
        // Reject unseen or disconnected FE combinations; never extrapolate their gauge.
        let predicted =
            design(&cells).dot(&fitted.coef) + fitted.levels.predict(&categories(&cells))?;
        let mut counterfactual = Array2::from_elem((n, t), f64::NAN);
        let mut effects = counterfactual.clone();
        let mut unit_sum = Array1::<f64>::zeros(n);
        let mut unit_weight = Array1::<f64>::zeros(n);
        let mut time_sum = Array1::<f64>::zeros(t);
        let mut time_weight = Array1::<f64>::zeros(t);
        let mut square_error = 0.0;
        for (r, &(i, j)) in cells.iter().enumerate() {
            counterfactual[[i, j]] = predicted[r];
            let gap = y[[i, j]] - predicted[r];
            if evaluation[[i, j]] {
                effects[[i, j]] = gap;
                unit_sum[i] += w[[i, j]] * gap;
                unit_weight[i] += w[[i, j]];
                time_sum[j] += w[[i, j]] * gap;
                time_weight[j] += w[[i, j]];
            } else {
                square_error += w[[i, j]] * gap * gap;
            }
        }
        let treated_weight = unit_weight.sum();
        let att = unit_sum.sum() / treated_weight;
        let divide = |a: Array1<f64>, b: &Array1<f64>| {
            Array1::from_iter(
                a.iter()
                    .zip(b)
                    .map(|(&a, &b)| if b > 0.0 { a / b } else { f64::NAN }),
            )
        };
        self.fitted = Some(ImputationFit {
            counterfactual,
            effects,
            training,
            evaluation,
            coef: fitted.coef,
            att,
            att_by_unit: divide(unit_sum, &unit_weight),
            att_by_time: divide(time_sum, &time_weight),
            treated_weight,
            untreated_rmse: (square_error / train_w.sum()).sqrt(),
            n_observed,
            n_zero_weight,
            n_reversing,
            iterations: fitted.iterations,
            residual_norms: fitted.residual_norms,
        });
        Ok(())
    }

    /// Fitted counterfactual levels; excluded/missing cells are NaN.
    fn predict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FEImputation model is not fitted"))?;
        Ok(pyarray2_from_f64(py, &f.counterfactual))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let f = self
            .fitted
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("FEImputation model is not fitted"))?;
        let out = PyDict::new(py);
        out.set_item("att", f.att)?;
        out.set_item("coef", pyarray1_from_f64(py, &f.coef))?;
        out.set_item("att_by_unit", pyarray1_from_f64(py, &f.att_by_unit))?;
        out.set_item("att_by_time", pyarray1_from_f64(py, &f.att_by_time))?;
        out.set_item("counterfactual", pyarray2_from_f64(py, &f.counterfactual))?;
        out.set_item("treatment_effect", pyarray2_from_f64(py, &f.effects))?;
        out.set_item(
            "training_mask",
            PyArray1::from_vec(py, f.training.iter().copied().collect())
                .reshape([f.training.nrows(), f.training.ncols()])?,
        )?;
        out.set_item(
            "evaluation_mask",
            PyArray1::from_vec(py, f.evaluation.iter().copied().collect())
                .reshape([f.evaluation.nrows(), f.evaluation.ncols()])?,
        )?;
        out.set_item("n_train", f.training.iter().filter(|v| **v).count())?;
        out.set_item("n_treated", f.evaluation.iter().filter(|v| **v).count())?;
        out.set_item("n_observed", f.n_observed)?;
        out.set_item("n_zero_weight", f.n_zero_weight)?;
        out.set_item("n_reversing_units", f.n_reversing)?;
        out.set_item("treated_weight", f.treated_weight)?;
        out.set_item("untreated_rmse", f.untreated_rmse)?;
        out.set_item("absorption_iterations", f.iterations.clone())?;
        out.set_item("absorption_residual_norms", f.residual_norms.clone())?;
        out.set_item("converged", true)?;
        out.set_item("estimand", "weighted_observed_treated_cells")?;
        out.set_item("inference", "not_implemented")?;
        Ok(out.into())
    }
}
