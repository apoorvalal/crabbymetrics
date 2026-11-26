//! Dense Euclidean trust regions; the spectral subproblem includes the hard case.
use super::common::*;
use nalgebra::{DMatrix, DVector, SymmetricEigen};
use ndarray::{Array1, Array2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(super) fn spectrum(h: &Array2<f64>) -> PyResult<SymmetricEigen<f64, nalgebra::Dyn>> {
    let eig = SymmetricEigen::try_new(
        DMatrix::from_row_iterator(h.nrows(), h.ncols(), h.iter().copied()),
        f64::EPSILON,
        10000,
    )
    .ok_or_else(|| error("Symmetric eigendecomposition did not converge"))?;
    if eig
        .eigenvalues
        .iter()
        .chain(eig.eigenvectors.iter())
        .any(|x| !x.is_finite())
    {
        return Err(error(
            "Symmetric eigendecomposition produced nonfinite values",
        ));
    }
    Ok(eig)
}

fn step(eig: &SymmetricEigen<f64, nalgebra::Dyn>, g: &Array1<f64>, radius: f64) -> Array1<f64> {
    let values = &eig.eigenvalues;
    let projected =
        eig.eigenvectors.transpose() * DVector::from_iterator(g.len(), g.iter().copied());
    let least = (0..g.len())
        .min_by(|&i, &j| values[i].total_cmp(&values[j]))
        .unwrap();
    let lower = (-values[least]).max(0.0);
    let scale = values.iter().fold(1.0_f64, |a, x| a.max(x.abs()));
    let small = 32.0 * f64::EPSILON * scale;
    let mut base = DVector::zeros(g.len());
    let mut singular_gradient = false;
    for i in 0..g.len() {
        let denominator = values[i] + lower;
        if denominator > small {
            base[i] = -projected[i] / denominator;
        } else if projected[i].abs() > 32.0 * f64::EPSILON * norm(g) {
            singular_gradient = true;
        }
    }
    if !singular_gradient && base.norm() <= radius {
        if lower > 0.0 {
            // At lambda=-lambda_min, add a minimum-eigenvalue component to reach
            // the boundary. Omitting this loses negative curvature orthogonal to g.
            let remaining = ((radius - base.norm()) * (radius + base.norm()))
                .max(0.0)
                .sqrt();
            base[least] += if projected[least] > 0.0 {
                -remaining
            } else {
                remaining
            };
        }
        let out = &eig.eigenvectors * base;
        return Array1::from_iter(out.iter().copied());
    }
    let mut lo = lower;
    let mut hi = lower + (norm(g) / radius).max(scale * f64::EPSILON);
    let at = |lambda: f64| -> DVector<f64> {
        DVector::from_iterator(
            g.len(),
            (0..g.len()).map(|i| -projected[i] / (values[i] + lambda)),
        )
    };
    while at(hi).norm() > radius {
        hi = lower + 2.0 * (hi - lower);
    }
    let mut solution = at(hi);
    for _ in 0..120 {
        let mid = lo + 0.5 * (hi - lo);
        if mid == lo || mid == hi {
            break;
        }
        let candidate = at(mid);
        if candidate.norm() > radius {
            lo = mid;
        } else {
            hi = mid;
            solution = candidate;
        }
        if (solution.norm() - radius).abs() <= 1e-12 * radius {
            break;
        }
    }
    Array1::from_iter((&eig.eigenvectors * solution).iter().copied())
}

pub(super) fn solve(
    py: Python<'_>,
    mut problem: PythonObjective,
    mut z: Array1<f64>,
    controls: Controls,
    initial_radius: f64,
    max_radius: f64,
    eta: f64,
) -> PyResult<Py<PyAny>> {
    controls.validate()?;
    if !initial_radius.is_finite()
        || initial_radius <= 0.0
        || !max_radius.is_finite()
        || max_radius < initial_radius
    {
        return Err(error(
            "Radii must satisfy 0 < initial_radius <= max_radius < infinity",
        ));
    }
    if !eta.is_finite() || eta <= 0.0 || eta >= 0.25 {
        return Err(error("eta must be between zero and 0.25"));
    }
    let mut radius = initial_radius;
    let mut current = initial(&mut problem, &z)?;
    let mut h = problem.hessian(&z)?;
    let mut eig = spectrum(&h)?;
    let mut iterations = 0;
    let mut status = "max_iterations";
    while iterations < controls.max_iterations {
        let min_eigenvalue = eig.eigenvalues.min();
        let curvature_tol =
            64.0 * f64::EPSILON * eig.eigenvalues.iter().fold(1.0_f64, |a, x| a.max(x.abs()));
        if norm_inf(&current.g) <= controls.gtol && min_eigenvalue >= -curvature_tol {
            status = "gradient_tolerance";
            break;
        }
        let s = step(&eig, &current.g, radius);
        let predicted = -current.g.dot(&s) - 0.5 * s.dot(&h.dot(&s));
        if !predicted.is_finite() || predicted <= 0.0 || z == &z + &s {
            status = "model_stagnation";
            break;
        }
        let candidate = &z + &s;
        let trial_f = problem.value(&candidate)?;
        let ratio = (current.f - trial_f) / predicted;
        if !ratio.is_finite() || ratio < 0.25 {
            radius *= 0.25;
        } else if ratio > 0.75 && norm(&s) >= 0.9 * radius {
            radius = (2.0 * radius).min(max_radius);
        }
        iterations += 1;
        if ratio.is_finite() && ratio > eta {
            let g = problem.gradient(&candidate)?;
            if g.iter().all(|x| x.is_finite()) {
                let change = (current.f - trial_f).abs();
                z = candidate;
                current = Evaluation { f: trial_f, g };
                h = problem.hessian(&z)?;
                eig = spectrum(&h)?;
                if controls.ftol > 0.0 && change <= controls.ftol * (1.0 + current.f.abs()) {
                    status = "cost_stagnation";
                    break;
                }
            } else {
                problem.counts.rejected += 1;
                radius *= 0.25;
            }
        } else {
            problem.counts.rejected += 1;
        }
        if radius < f64::EPSILON * (1.0 + norm(&z)) {
            status = "radius_too_small";
            break;
        }
    }
    let min_eigenvalue = eig.eigenvalues.min();
    let curvature_tol =
        64.0 * f64::EPSILON * eig.eigenvalues.iter().fold(1.0_f64, |a, x| a.max(x.abs()));
    let result = finish(
        py,
        &mut problem,
        &z,
        iterations,
        status,
        "trust_region",
        &controls,
    )?;
    let dict = result.bind(py).cast::<PyDict>()?;
    dict.set_item("radius", radius)?;
    dict.set_item("min_eigenvalue", min_eigenvalue)?;
    if min_eigenvalue < -curvature_tol {
        dict.set_item("success", false)?;
        dict.set_item("status", "negative_curvature_remaining")?;
        dict.set_item("message", "Final Hessian retains negative curvature")?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn positive_quadratic_interior_newton_step() {
        let h = array![[2.0, 0.0], [0.0, 4.0]];
        let s = step(&spectrum(&h).unwrap(), &array![2.0, -8.0], 10.0);
        assert!(norm(&(s - array![-1.0, 2.0])) < 1e-12);
    }

    #[test]
    fn boundary_step_satisfies_shifted_kkt() {
        let h = array![[2.0, 0.0], [0.0, 4.0]];
        let g = array![2.0, -8.0];
        let s = step(&spectrum(&h).unwrap(), &g, 0.5);
        assert!((norm(&s) - 0.5).abs() < 1e-11);
        let residual = h.dot(&s) + &g;
        let lambda = -residual.dot(&s) / s.dot(&s);
        assert!(lambda > 0.0);
        assert!(norm(&(residual + s * lambda)) < 1e-10);
    }

    #[test]
    fn hard_case_uses_negative_curvature_orthogonal_to_gradient() {
        let h = array![[-2.0, 0.0], [0.0, 1.0]];
        let g = array![0.0, 1.0];
        let s = step(&spectrum(&h).unwrap(), &g, 1.0);
        assert!((norm(&s) - 1.0).abs() < 1e-12);
        assert!((s[1] + 1.0 / 3.0).abs() < 1e-12);
        assert!(s[0].abs() > 0.9);
        assert!(norm(&(h.dot(&s) + &g + 2.0 * &s)) < 1e-12);
    }

    #[test]
    fn tiny_positive_eigenvalue_is_not_floored() {
        let h = array![[1e-12]];
        let s = step(&spectrum(&h).unwrap(), &array![1e-12], 10.0);
        assert!((s[0] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn singular_psd_uses_minimum_norm_interior_solution() {
        let h = array![[0.0, 0.0], [0.0, 2.0]];
        let s = step(&spectrum(&h).unwrap(), &array![0.0, 2.0], 10.0);
        assert!(norm(&(s - array![0.0, -1.0])) < 1e-12);
    }
}
