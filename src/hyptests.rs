use crate::utils::{invert_matrix, to_array1, to_array2};
use crate::validation::validate_finite;
use ndarray::{Array1, Array2, Axis};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const ITMAX: usize = 10_000;
const EPS: f64 = 3.0e-14;
const FPMIN: f64 = 1.0e-300;

fn ln_gamma(z: f64) -> f64 {
    // Lanczos approximation, coefficients from Numerical Recipes.
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    let x = z;
    let mut y = z;
    let tmp = x + 5.5;
    let tmp = (x + 0.5) * tmp.ln() - tmp;
    let mut ser = 1.000000000190015;
    for c in coeffs {
        y += 1.0;
        ser += c / y;
    }
    tmp + (2.5066282746310005 * ser / x).ln()
}

fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let gln = ln_gamma(a);
    let mut ap = a;
    let mut del = 1.0 / a;
    let mut sum = del;
    for _ in 0..ITMAX {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * EPS {
            return (sum * (-x + a * x.ln() - gln).exp()).clamp(0.0, 1.0);
        }
    }
    (sum * (-x + a * x.ln() - gln).exp()).clamp(0.0, 1.0)
}

fn regularized_gamma_q(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let gln = ln_gamma(a);
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / FPMIN;
    let mut d = 1.0 / b.max(FPMIN);
    let mut h = d;
    for i in 1..=ITMAX {
        let i_f = i as f64;
        let an = -i_f * (i_f - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = b + an / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            return ((-x + a * x.ln() - gln).exp() * h).clamp(0.0, 1.0);
        }
    }
    ((-x + a * x.ln() - gln).exp() * h).clamp(0.0, 1.0)
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=ITMAX {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;

        let mut aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;

        aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

fn regularized_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        (bt * beta_cf(a, b, x) / a).clamp(0.0, 1.0)
    } else {
        (1.0 - bt * beta_cf(b, a, 1.0 - x) / b).clamp(0.0, 1.0)
    }
}

pub(crate) fn f_sf(statistic: f64, df1: usize, df2: usize) -> PyResult<f64> {
    if df1 == 0 || df2 == 0 {
        return Err(PyValueError::new_err(
            "F degrees of freedom must be positive",
        ));
    }
    if !statistic.is_finite() || statistic < 0.0 {
        return Err(PyValueError::new_err(
            "F statistic must be finite and nonnegative",
        ));
    }
    let d1 = df1 as f64;
    let d2 = df2 as f64;
    let x = (d1 * statistic) / (d1 * statistic + d2);
    Ok((1.0 - regularized_beta(0.5 * d1, 0.5 * d2, x)).clamp(0.0, 1.0))
}

fn chi_square_sf(statistic: f64, df: usize) -> PyResult<f64> {
    if df == 0 {
        return Err(PyValueError::new_err("degrees of freedom must be positive"));
    }
    if !statistic.is_finite() || statistic < 0.0 {
        return Err(PyValueError::new_err(
            "test statistic must be finite and nonnegative",
        ));
    }
    let a = 0.5 * df as f64;
    let x = 0.5 * statistic;
    let p = if x < a + 1.0 {
        1.0 - regularized_gamma_p(a, x)
    } else {
        regularized_gamma_q(a, x)
    };
    Ok(p.clamp(0.0, 1.0))
}

pub(crate) fn wald_test_arrays<'py>(
    py: Python<'py>,
    beta: &Array1<f64>,
    cov: &Array2<f64>,
    rmat: &Array2<f64>,
    qvec: Option<&Array1<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let zero_q;
    let qvec = match qvec {
        Some(q) => q,
        None => {
            zero_q = Array1::<f64>::zeros(rmat.nrows());
            &zero_q
        }
    };

    validate_finite("coef", beta).map_err(PyValueError::new_err)?;
    validate_finite("vcov", cov).map_err(PyValueError::new_err)?;
    validate_finite("r", rmat).map_err(PyValueError::new_err)?;
    validate_finite("q", qvec).map_err(PyValueError::new_err)?;

    let k = beta.len();
    let df = rmat.nrows();
    if df == 0 {
        return Err(PyValueError::new_err(
            "r must contain at least one restriction row",
        ));
    }
    if cov.nrows() != k || cov.ncols() != k {
        return Err(PyValueError::new_err(
            "vcov must be square with dimension len(coef)",
        ));
    }
    if rmat.ncols() != k {
        return Err(PyValueError::new_err("r must have len(coef) columns"));
    }
    if qvec.len() != df {
        return Err(PyValueError::new_err(
            "q must have one entry per restriction row",
        ));
    }

    let diff = rmat.dot(beta) - qvec;
    let rcov = rmat.dot(cov).dot(&rmat.t());
    let rcov_inv = invert_matrix(&rcov).map_err(PyValueError::new_err)?;
    let tmp = rcov_inv.dot(&diff.clone().insert_axis(Axis(1)));
    let statistic = diff.dot(&tmp.column(0)).max(0.0);
    let p_value = chi_square_sf(statistic, df)?;

    let out = PyDict::new(py);
    out.set_item("statistic", statistic)?;
    out.set_item("df", df)?;
    out.set_item("p_value", p_value)?;
    out.set_item("test", "wald")?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (coef, vcov, r, q=None))]
pub fn wald_test<'py>(
    py: Python<'py>,
    coef: PyReadonlyArray1<f64>,
    vcov: PyReadonlyArray2<f64>,
    r: PyReadonlyArray2<f64>,
    q: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let beta = to_array1(&coef);
    let cov = to_array2(&vcov);
    let rmat = to_array2(&r);
    let qvec = q.as_ref().map(to_array1);
    wald_test_arrays(py, &beta, &cov, &rmat, qvec.as_ref())
}

fn likelihood_ratio_test_impl<'py>(
    py: Python<'py>,
    unrestricted_loglik: f64,
    restricted_loglik: f64,
    df: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if !unrestricted_loglik.is_finite() || !restricted_loglik.is_finite() {
        return Err(PyValueError::new_err("log likelihoods must be finite"));
    }
    if df == 0 {
        return Err(PyValueError::new_err("degrees of freedom must be positive"));
    }
    let statistic = 2.0 * (unrestricted_loglik - restricted_loglik);
    if statistic < -1e-10 {
        return Err(PyValueError::new_err(
            "unrestricted_loglik must be at least restricted_loglik",
        ));
    }
    let statistic = statistic.max(0.0);
    let p_value = chi_square_sf(statistic, df)?;
    let out = PyDict::new(py);
    out.set_item("statistic", statistic)?;
    out.set_item("df", df)?;
    out.set_item("p_value", p_value)?;
    out.set_item("test", "likelihood_ratio")?;
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (unrestricted_loglik, restricted_loglik, df))]
pub fn likelihood_ratio_test<'py>(
    py: Python<'py>,
    unrestricted_loglik: f64,
    restricted_loglik: f64,
    df: usize,
) -> PyResult<Bound<'py, PyDict>> {
    likelihood_ratio_test_impl(py, unrestricted_loglik, restricted_loglik, df)
}

#[pyfunction]
#[pyo3(signature = (unrestricted_loglik, restricted_loglik, df))]
pub fn lr_test<'py>(
    py: Python<'py>,
    unrestricted_loglik: f64,
    restricted_loglik: f64,
    df: usize,
) -> PyResult<Bound<'py, PyDict>> {
    likelihood_ratio_test_impl(py, unrestricted_loglik, restricted_loglik, df)
}
