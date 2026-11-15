use super::common::*;
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use std::collections::VecDeque;

pub(super) fn solve(
    py: Python<'_>,
    mut problem: PythonObjective,
    mut z: Array1<f64>,
    controls: Controls,
    limited_memory: bool,
) -> PyResult<Py<PyAny>> {
    controls.validate()?;
    let mut current = initial(&mut problem, &z)?;
    let p = z.len();
    // L-BFGS must never allocate a dense p-by-p inverse, even during a reset.
    let mut inverse = (!limited_memory).then(|| Array2::<f64>::eye(p));
    let mut history: VecDeque<(Array1<f64>, Array1<f64>, f64)> = VecDeque::new();
    let mut iterations = 0;
    let mut status = "max_iterations";
    while iterations < controls.max_iterations && norm_inf(&current.g) > controls.gtol {
        let mut direction = if limited_memory {
            let mut q = current.g.clone();
            let mut alphas = Vec::with_capacity(history.len());
            for (s, y, rho) in history.iter().rev() {
                let alpha = rho * s.dot(&q);
                q -= &(y * alpha);
                alphas.push(alpha);
            }
            let gamma = history.back().map_or(1.0, |(s, y, _)| s.dot(y) / y.dot(y));
            let mut r = q * gamma;
            for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
                let beta = rho * y.dot(&r);
                r += &(s * (alpha - beta));
            }
            -r
        } else {
            -inverse.as_ref().unwrap().dot(&current.g)
        };
        if direction.iter().any(|x| !x.is_finite()) || current.g.dot(&direction) >= 0.0 {
            if let Some(m) = &mut inverse {
                *m = Array2::eye(p);
            }
            history.clear();
            problem.counts.resets += 1;
            direction = -&current.g;
        }
        let mut next = line_search(&mut problem, &z, &current, &direction, &controls)?;
        if next.is_none()
            && (!history.is_empty()
                || inverse
                    .as_ref()
                    .is_some_and(|m| *m != Array2::<f64>::eye(p)))
        {
            // A bad curvature model must not make a finite point unusable.
            if let Some(m) = &mut inverse {
                *m = Array2::eye(p);
            }
            history.clear();
            problem.counts.resets += 1;
            next = line_search(&mut problem, &z, &current, &(-&current.g), &controls)?;
        }
        let Some((new_z, new_eval)) = next else {
            status = "line_search_failed";
            break;
        };
        let s = &new_z - &z;
        let y = &new_eval.g - &current.g;
        let sy = s.dot(&y);
        if sy.is_finite() && sy > 1e-12 * norm(&s) * norm(&y) && sy > 0.0 {
            if limited_memory {
                if history.len() == 7 {
                    history.pop_front();
                }
                history.push_back((s, y, 1.0 / sy));
            } else {
                // Expanded inverse BFGS update: O(p^2), with symmetry restored.
                let inverse = inverse.as_mut().unwrap();
                let my = inverse.dot(&y);
                let factor = (sy + y.dot(&my)) / (sy * sy);
                for i in 0..p {
                    for j in 0..p {
                        inverse[[i, j]] +=
                            factor * s[i] * s[j] - (my[i] * s[j] + s[i] * my[j]) / sy;
                    }
                }
                *inverse = (&*inverse + &inverse.t()) * 0.5;
            }
        } else {
            problem.counts.skipped += 1;
        }
        let change = (current.f - new_eval.f).abs();
        z = new_z;
        current = new_eval;
        iterations += 1;
        if controls.ftol > 0.0 && change <= controls.ftol * (1.0 + current.f.abs()) {
            status = "cost_stagnation";
            break;
        }
    }
    finish(
        py,
        &mut problem,
        &z,
        iterations,
        status,
        if limited_memory { "lbfgs" } else { "bfgs" },
        &controls,
    )
}
