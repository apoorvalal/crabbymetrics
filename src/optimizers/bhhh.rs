//! Score outer-product curvature for a mean or sum negative log likelihood.
use super::common::*;
use super::trust_region::spectrum;
use crate::utils::{pyarray1_from_f64, to_array2};
use nalgebra::DVector;
use ndarray::{Array1, Array2};
use numpy::{PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub(super) struct ScoreObjective {
    pub fun: Py<PyAny>,
    pub scores: Option<Py<PyAny>>,
    pub score_stats: Option<Py<PyAny>>,
    pub penalty_grad: Option<Py<PyAny>>,
    pub penalty_hess: Option<Py<PyAny>>,
    pub mean: bool,
    pub scale: Array1<f64>,
    pub counts: Counts,
    cache: Option<(Array1<f64>, Array1<f64>, Array2<f64>)>,
}

impl ScoreObjective {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        fun: Py<PyAny>,
        scores: Option<Py<PyAny>>,
        score_stats: Option<Py<PyAny>>,
        penalty_grad: Option<Py<PyAny>>,
        penalty_hess: Option<Py<PyAny>>,
        reduction: &str,
        scale: Array1<f64>,
    ) -> PyResult<Self> {
        if scores.is_some() == score_stats.is_some() {
            return Err(error("Provide exactly one of scores or score_stats"));
        }
        if penalty_grad.is_some() != penalty_hess.is_some() {
            return Err(error("Provide penalty_grad and penalty_hess together"));
        }
        let mean = match reduction {
            "mean" => true,
            "sum" => false,
            _ => return Err(error("reduction must be 'mean' or 'sum'")),
        };
        Ok(Self {
            fun,
            scores,
            score_stats,
            penalty_grad,
            penalty_hess,
            mean,
            scale,
            counts: Counts::default(),
            cache: None,
        })
    }

    fn derivatives(&mut self, z: &Array1<f64>) -> PyResult<()> {
        if self.cache.as_ref().is_some_and(|(point, _, _)| point == z) {
            return Ok(());
        }
        self.counts.nscorev += 1;
        let p = z.len();
        let (mut g, mut c) = Python::attach(|py| -> PyResult<_> {
            let x = z * &self.scale;
            let (mut g, mut c, n) = if let Some(callback) = &self.scores {
                let value = callback.call1(py, (pyarray1_from_f64(py, &x),))?;
                let bound = value.bind(py);
                let array = bound
                    .cast::<PyArray2<f64>>()
                    .map_err(|_| error("scores must return a 2D float64 NumPy array"))?;
                let rows = to_array2(&array.readonly());
                if rows.nrows() == 0 || rows.ncols() != p {
                    return Err(error(
                        "scores shape must be (n_observations > 0, n_parameters)",
                    ));
                }
                (
                    rows.sum_axis(ndarray::Axis(0)),
                    rows.t().dot(&rows),
                    rows.nrows(),
                )
            } else {
                let value = self
                    .score_stats
                    .as_ref()
                    .unwrap()
                    .call1(py, (pyarray1_from_f64(py, &x),))?;
                let tuple = value.bind(py).cast::<PyTuple>().map_err(|_| {
                    error("score_stats must return (score_sum, outer_sum, n_observations)")
                })?;
                if tuple.len() != 3 {
                    return Err(error(
                        "score_stats must return (score_sum, outer_sum, n_observations)",
                    ));
                }
                let n: usize = tuple.get_item(2)?.extract()?;
                if n == 0 {
                    return Err(error("score_stats n_observations must be positive"));
                }
                (
                    vector(&tuple.get_item(0)?, p)?,
                    matrix(&tuple.get_item(1)?, p)?,
                    n,
                )
            };
            if self.mean {
                g /= n as f64;
                c /= n as f64;
            }
            if let (Some(pg), Some(ph)) = (&self.penalty_grad, &self.penalty_hess) {
                let vg = pg.call1(py, (pyarray1_from_f64(py, &x),))?;
                let vh = ph.call1(py, (pyarray1_from_f64(py, &x),))?;
                g += &vector(vg.bind(py), p)?;
                c += &matrix(vh.bind(py), p)?;
            }
            Ok((g, c))
        })?;
        g *= &self.scale;
        for i in 0..p {
            for j in 0..p {
                c[[i, j]] *= self.scale[i] * self.scale[j];
            }
        }
        if c.iter().any(|x| !x.is_finite()) {
            return Err(error("Score cross-product must be finite"));
        }
        self.cache = Some((z.clone(), g, c));
        Ok(())
    }

    fn curvature(&mut self, z: &Array1<f64>) -> PyResult<Array2<f64>> {
        self.derivatives(z)?;
        Ok(self.cache.as_ref().unwrap().2.clone())
    }
}

impl Objective for ScoreObjective {
    fn value(&mut self, z: &Array1<f64>) -> PyResult<f64> {
        let x = z * &self.scale;
        if x.iter().any(|v| !v.is_finite()) {
            return Ok(f64::INFINITY);
        }
        self.counts.nfev += 1;
        Python::attach(|py| {
            self.fun
                .call1(py, (pyarray1_from_f64(py, &x),))?
                .extract(py)
        })
    }
    fn gradient(&mut self, z: &Array1<f64>) -> PyResult<Array1<f64>> {
        self.counts.njev += 1;
        self.derivatives(z)?;
        Ok(self.cache.as_ref().unwrap().1.clone())
    }
    fn scale(&self) -> &Array1<f64> {
        &self.scale
    }
    fn counts(&mut self) -> &mut Counts {
        &mut self.counts
    }
    fn invalidate(&mut self) {
        self.cache = None;
    }
}

pub(super) fn solve(
    py: Python<'_>,
    mut problem: ScoreObjective,
    mut z: Array1<f64>,
    controls: Controls,
    damping: f64,
) -> PyResult<Py<PyAny>> {
    controls.validate()?;
    if !damping.is_finite() || damping <= 0.0 {
        return Err(error("damping must be positive and finite"));
    }
    let mut current = initial(&mut problem, &z)?;
    let mut iterations = 0;
    let mut status = "max_iterations";
    while iterations < controls.max_iterations && norm_inf(&current.g) > controls.gtol {
        let c = problem.curvature(&z)?;
        let eig = spectrum(&c)?;
        let size = eig.eigenvalues.iter().fold(1.0_f64, |a, x| a.max(x.abs()));
        if eig.eigenvalues.min() < -1e-10 * size {
            return Err(error("BHHH curvature must be positive semidefinite"));
        }
        let projected = eig.eigenvectors.transpose()
            * DVector::from_iterator(z.len(), current.g.iter().copied());
        let coefficients = DVector::from_iterator(
            z.len(),
            (0..z.len()).map(|i| -projected[i] / eig.eigenvalues[i].max(damping * size)),
        );
        let direction = Array1::from_iter((&eig.eigenvectors * coefficients).iter().copied());
        let Some((new_z, next)) = line_search(&mut problem, &z, &current, &direction, &controls)?
        else {
            status = "line_search_failed";
            break;
        };
        let change = (current.f - next.f).abs();
        z = new_z;
        current = next;
        iterations += 1;
        if controls.ftol > 0.0 && change <= controls.ftol * (1.0 + current.f.abs()) {
            status = "cost_stagnation";
            break;
        }
    }
    finish(py, &mut problem, &z, iterations, status, "bhhh", &controls)
}
