use crate::utils::{pyarray1_from_f64, pyarray2_from_f64, to_array2};
use linfa::prelude::{Fit, Predict, Transformer};
use linfa::DatasetBase;
use linfa_kernel::{Kernel, KernelMethod};
use linfa_reduction::Pca;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn dense_kernel_matrix(kernel: &Kernel<f64>) -> Array2<f64> {
    let n = kernel.size();
    let mut out = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let column = kernel.column(j);
        for (i, value) in column.iter().enumerate() {
            out[[i, j]] = *value;
        }
    }
    out
}

fn parse_kernel_method(
    kernel: &str,
    bandwidth: f64,
    coef0: f64,
    degree: f64,
) -> PyResult<KernelMethod<f64>> {
    match kernel.to_ascii_lowercase().as_str() {
        "gaussian" | "rbf" => {
            if bandwidth <= 0.0 {
                return Err(PyValueError::new_err(
                    "bandwidth must be positive for a gaussian kernel",
                ));
            }
            Ok(KernelMethod::Gaussian(bandwidth))
        }
        "linear" => Ok(KernelMethod::Linear),
        "polynomial" | "poly" => {
            if degree <= 0.0 {
                return Err(PyValueError::new_err(
                    "degree must be positive for a polynomial kernel",
                ));
            }
            Ok(KernelMethod::Polynomial(coef0, degree))
        }
        _ => Err(PyValueError::new_err(
            "kernel must be one of 'gaussian', 'rbf', 'linear', or 'polynomial'",
        )),
    }
}

fn canonical_kernel_name(kernel: &str) -> PyResult<&'static str> {
    match kernel.to_ascii_lowercase().as_str() {
        "gaussian" | "rbf" => Ok("gaussian"),
        "linear" => Ok("linear"),
        "polynomial" | "poly" => Ok("polynomial"),
        _ => Err(PyValueError::new_err(
            "kernel must be one of 'gaussian', 'rbf', 'linear', or 'polynomial'",
        )),
    }
}

fn cross_kernel_matrix(
    x_new: &Array2<f64>,
    x_train: &Array2<f64>,
    method: &KernelMethod<f64>,
) -> PyResult<Array2<f64>> {
    if x_new.ncols() != x_train.ncols() {
        return Err(PyValueError::new_err(
            "x columns must match the fitted training design",
        ));
    }

    let mut out = Array2::<f64>::zeros((x_new.nrows(), x_train.nrows()));
    for i in 0..x_new.nrows() {
        let row = x_new.row(i);
        for j in 0..x_train.nrows() {
            out[[i, j]] = method.distance(row, x_train.row(j));
        }
    }
    Ok(out)
}

#[pyclass(name = "PCA")]
pub struct PcaTransformer {
    n_components: usize,
    whiten: bool,
    model: Option<Pca<f64>>,
    n_features: Option<usize>,
}

#[pymethods]
impl PcaTransformer {
    #[new]
    #[pyo3(signature = (n_components, whiten=false))]
    fn new(n_components: usize, whiten: bool) -> Self {
        Self {
            n_components,
            whiten,
            model: None,
            n_features: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if self.n_components == 0 || self.n_components > x.ncols() {
            return Err(PyValueError::new_err(
                "n_components must be between 1 and the number of columns in x",
            ));
        }

        let dataset = DatasetBase::from(x.clone());
        let model = Pca::params(self.n_components)
            .whiten(self.whiten)
            .fit(&dataset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        self.n_features = Some(x.ncols());
        self.model = Some(model);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
        let x = to_array2(&x);
        let scores = model.predict(&x);
        Ok(pyarray2_from_f64(py, &scores))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if self.n_components == 0 || self.n_components > x.ncols() {
            return Err(PyValueError::new_err(
                "n_components must be between 1 and the number of columns in x",
            ));
        }

        let dataset = DatasetBase::from(x.clone());
        let model = Pca::params(self.n_components)
            .whiten(self.whiten)
            .fit(&dataset)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let scores = model.predict(&x);

        self.n_features = Some(x.ncols());
        self.model = Some(model);
        Ok(pyarray2_from_f64(py, &scores))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        scores: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
        let scores = to_array2(&scores);
        if scores.ncols() != self.n_components {
            return Err(PyValueError::new_err(
                "score columns must match n_components",
            ));
        }
        let reconstructed = model.inverse_transform(scores);
        Ok(pyarray2_from_f64(py, &reconstructed))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("n_components", self.n_components)?;
        dict.set_item("n_features", self.n_features.unwrap_or(0))?;
        dict.set_item("whiten", self.whiten)?;
        dict.set_item("components", pyarray2_from_f64(py, model.components()))?;
        dict.set_item("mean", pyarray1_from_f64(py, model.mean()))?;
        dict.set_item(
            "explained_variance",
            pyarray1_from_f64(py, &model.explained_variance()),
        )?;
        dict.set_item(
            "explained_variance_ratio",
            pyarray1_from_f64(py, &model.explained_variance_ratio()),
        )?;
        dict.set_item(
            "singular_values",
            pyarray1_from_f64(py, model.singular_values()),
        )?;
        Ok(dict.into())
    }
}

#[pyclass]
pub struct KernelBasis {
    kernel: String,
    bandwidth: f64,
    coef0: f64,
    degree: f64,
    train_x: Option<Array2<f64>>,
    train_basis: Option<Array2<f64>>,
}

#[pymethods]
impl KernelBasis {
    #[new]
    #[pyo3(signature = (kernel="gaussian", bandwidth=0.5, coef0=1.0, degree=2.0))]
    fn new(kernel: &str, bandwidth: f64, coef0: f64, degree: f64) -> PyResult<Self> {
        let kernel = canonical_kernel_name(kernel)?.to_string();
        let _ = parse_kernel_method(&kernel, bandwidth, coef0, degree)?;
        Ok(Self {
            kernel,
            bandwidth,
            coef0,
            degree,
            train_x: None,
            train_basis: None,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }

        let method = parse_kernel_method(&self.kernel, self.bandwidth, self.coef0, self.degree)?;
        let params = Kernel::<f64>::params().method(method);
        let fitted = params.transform(&x);

        self.train_basis = Some(dense_kernel_matrix(&fitted));
        self.train_x = Some(x);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let train_x = self
            .train_x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("KernelBasis is not fitted"))?;
        let x = to_array2(&x);
        let method = parse_kernel_method(&self.kernel, self.bandwidth, self.coef0, self.degree)?;
        let basis = cross_kernel_matrix(&x, train_x, &method)?;
        Ok(pyarray2_from_f64(py, &basis))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }

        let method = parse_kernel_method(&self.kernel, self.bandwidth, self.coef0, self.degree)?;
        let params = Kernel::<f64>::params().method(method);
        let fitted = params.transform(&x);
        let basis = dense_kernel_matrix(&fitted);

        self.train_x = Some(x);
        self.train_basis = Some(basis.clone());
        Ok(pyarray2_from_f64(py, &basis))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let train_x = self
            .train_x
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("KernelBasis is not fitted"))?;
        let train_basis = self
            .train_basis
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("KernelBasis is not fitted"))?;

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("kernel", &self.kernel)?;
        dict.set_item("n_train", train_x.nrows())?;
        dict.set_item("n_features", train_x.ncols())?;
        dict.set_item("bandwidth", self.bandwidth)?;
        dict.set_item("coef0", self.coef0)?;
        dict.set_item("degree", self.degree)?;
        dict.set_item(
            "diagonal",
            pyarray1_from_f64(py, &train_basis.diag().to_owned()),
        )?;
        Ok(dict.into())
    }
}
