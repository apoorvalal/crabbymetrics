use crate::rla::randomized_svd_impl;
use crate::utils::{pyarray1_from_f64, pyarray2_from_f64, to_array2};
use linfa::prelude::{Fit, Predict, Transformer};
use linfa::DatasetBase;
use linfa_kernel::{Kernel, KernelMethod};
use linfa_reduction::Pca;
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

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

fn array2_to_dmatrix(matrix: &Array2<f64>) -> DMatrix<f64> {
    let data: Vec<f64> = matrix.iter().copied().collect();
    DMatrix::from_row_slice(matrix.nrows(), matrix.ncols(), &data)
}

fn dmatrix_to_array2(matrix: &DMatrix<f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((matrix.nrows(), matrix.ncols()));
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            out[[i, j]] = matrix[(i, j)];
        }
    }
    out
}

fn symmetric_inverse_sqrt(matrix: &Array2<f64>, ridge: f64) -> PyResult<Array2<f64>> {
    if matrix.nrows() != matrix.ncols() {
        return Err(PyValueError::new_err("matrix must be square"));
    }
    if !ridge.is_finite() || ridge < 0.0 {
        return Err(PyValueError::new_err(
            "ridge must be finite and nonnegative",
        ));
    }
    let mut dm = array2_to_dmatrix(matrix);
    for j in 0..dm.ncols() {
        dm[(j, j)] += ridge;
    }
    let eig = SymmetricEigen::new(dm);
    let k = eig.eigenvalues.len();
    let mut diag = DMatrix::<f64>::zeros(k, k);
    for j in 0..k {
        let value = eig.eigenvalues[j].max(1e-14);
        diag[(j, j)] = 1.0 / value.sqrt();
    }
    let inv_sqrt = &eig.eigenvectors * diag * eig.eigenvectors.transpose();
    Ok(dmatrix_to_array2(&inv_sqrt))
}

fn parse_kernel_method(
    kernel: &str,
    bandwidth: f64,
    coef0: f64,
    degree: f64,
) -> PyResult<KernelMethod<f64>> {
    if !bandwidth.is_finite() || !coef0.is_finite() || !degree.is_finite() {
        return Err(PyValueError::new_err("kernel parameters must be finite"));
    }
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
    crate::validation::validate_finite("x", x_new).map_err(PyValueError::new_err)?;
    crate::validation::validate_dense_capacity("cross-kernel", x_new.nrows(), x_train.nrows())
        .map_err(PyValueError::new_err)?;
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

fn total_sample_variance(x: &Array2<f64>) -> PyResult<f64> {
    if x.nrows() < 2 {
        return Err(PyValueError::new_err(
            "PCA requires at least two observations",
        ));
    }
    let mean = x
        .mean_axis(ndarray::Axis(0))
        .ok_or_else(|| PyValueError::new_err("failed to compute mean"))?;
    let sum_squares = x
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(mean.iter())
                .map(|(value, center)| (value - center).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    Ok(sum_squares / (x.nrows() - 1) as f64)
}

#[pyclass(name = "PCA")]
pub struct PcaTransformer {
    n_components: usize,
    whiten: bool,
    model: Option<Pca<f64>>,
    n_features: Option<usize>,
    n_samples: Option<usize>,
    total_variance: Option<f64>,
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
            n_samples: None,
            total_variance: None,
        }
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        let total_variance = total_sample_variance(&x)?;
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
        self.n_samples = Some(x.nrows());
        self.total_variance = Some(total_variance);
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
        let total_variance = total_sample_variance(&x)?;
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
        self.n_samples = Some(x.nrows());
        self.total_variance = Some(total_variance);
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
        let reconstructed = if self.whiten {
            let n_samples = self
                .n_samples
                .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
            let mut unscaled = scores;
            for (j, singular_value) in model.singular_values().iter().enumerate() {
                let inverse_whitening = singular_value * singular_value / (n_samples - 1) as f64;
                unscaled
                    .column_mut(j)
                    .mapv_inplace(|value| value * inverse_whitening);
            }
            unscaled.dot(model.components()) + model.mean()
        } else {
            model.inverse_transform(scores)
        };
        Ok(pyarray2_from_f64(py, &reconstructed))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
        let n_samples = self
            .n_samples
            .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
        let total_variance = self
            .total_variance
            .ok_or_else(|| PyValueError::new_err("PCA model is not fitted"))?;
        let explained_variance = model
            .singular_values()
            .mapv(|value| value * value / (n_samples - 1) as f64);
        let explained_variance_ratio = if total_variance > 0.0 {
            explained_variance.mapv(|value| value / total_variance)
        } else {
            Array1::<f64>::zeros(explained_variance.len())
        };
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("n_components", self.n_components)?;
        dict.set_item("n_features", self.n_features.unwrap_or(0))?;
        dict.set_item("n_samples", n_samples)?;
        dict.set_item("whiten", self.whiten)?;
        dict.set_item("components", pyarray2_from_f64(py, model.components()))?;
        dict.set_item("mean", pyarray1_from_f64(py, model.mean()))?;
        dict.set_item(
            "explained_variance",
            pyarray1_from_f64(py, &explained_variance),
        )?;
        dict.set_item(
            "explained_variance_ratio",
            pyarray1_from_f64(py, &explained_variance_ratio),
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
        self.train_x = None;
        self.train_basis = None;
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }

        crate::validation::validate_finite("x", &x).map_err(PyValueError::new_err)?;
        crate::validation::validate_dense_capacity("kernel", x.nrows(), x.nrows())
            .map_err(PyValueError::new_err)?;
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

#[pyclass]
pub struct NystromBasis {
    kernel: String,
    n_components: usize,
    bandwidth: f64,
    coef0: f64,
    degree: f64,
    ridge: f64,
    seed: Option<u64>,
    landmarks: Option<Array2<f64>>,
    landmark_indices: Option<Vec<usize>>,
    kmm_inv_sqrt: Option<Array2<f64>>,
}

#[pymethods]
impl NystromBasis {
    #[new]
    #[pyo3(signature = (n_components, kernel="gaussian", bandwidth=0.5, coef0=1.0, degree=2.0, ridge=1e-10, seed=None))]
    fn new(
        n_components: usize,
        kernel: &str,
        bandwidth: f64,
        coef0: f64,
        degree: f64,
        ridge: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_components == 0 {
            return Err(PyValueError::new_err("n_components must be positive"));
        }
        if !ridge.is_finite() || ridge < 0.0 {
            return Err(PyValueError::new_err(
                "ridge must be finite and nonnegative",
            ));
        }
        let kernel = canonical_kernel_name(kernel)?.to_string();
        let _ = parse_kernel_method(&kernel, bandwidth, coef0, degree)?;
        Ok(Self {
            kernel,
            n_components,
            bandwidth,
            coef0,
            degree,
            ridge,
            seed,
            landmarks: None,
            landmark_indices: None,
            kmm_inv_sqrt: None,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if self.n_components > x.nrows() {
            return Err(PyValueError::new_err(
                "n_components must be <= the number of rows in x",
            ));
        }
        let mut indices: Vec<usize> = (0..x.nrows()).collect();
        if let Some(seed) = self.seed {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        indices.truncate(self.n_components);
        indices.sort_unstable();

        let mut landmarks = Array2::<f64>::zeros((self.n_components, x.ncols()));
        for (out_i, &src_i) in indices.iter().enumerate() {
            landmarks.row_mut(out_i).assign(&x.row(src_i));
        }
        let method = parse_kernel_method(&self.kernel, self.bandwidth, self.coef0, self.degree)?;
        let kmm = cross_kernel_matrix(&landmarks, &landmarks, &method)?;
        let kmm_inv_sqrt = symmetric_inverse_sqrt(&kmm, self.ridge)?;

        self.landmarks = Some(landmarks);
        self.landmark_indices = Some(indices);
        self.kmm_inv_sqrt = Some(kmm_inv_sqrt);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let landmarks = self
            .landmarks
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("NystromBasis is not fitted"))?;
        let kmm_inv_sqrt = self
            .kmm_inv_sqrt
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("NystromBasis is not fitted"))?;
        let x = to_array2(&x);
        let method = parse_kernel_method(&self.kernel, self.bandwidth, self.coef0, self.degree)?;
        let knm = cross_kernel_matrix(&x, landmarks, &method)?;
        let features = knm.dot(kmm_inv_sqrt);
        Ok(pyarray2_from_f64(py, &features))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_array2(&x);
        if x_arr.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if self.n_components > x_arr.nrows() {
            return Err(PyValueError::new_err(
                "n_components must be <= the number of rows in x",
            ));
        }
        let mut indices: Vec<usize> = (0..x_arr.nrows()).collect();
        if let Some(seed) = self.seed {
            let mut rng = StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        indices.truncate(self.n_components);
        indices.sort_unstable();

        let mut landmarks = Array2::<f64>::zeros((self.n_components, x_arr.ncols()));
        for (out_i, &src_i) in indices.iter().enumerate() {
            landmarks.row_mut(out_i).assign(&x_arr.row(src_i));
        }
        let method = parse_kernel_method(&self.kernel, self.bandwidth, self.coef0, self.degree)?;
        let kmm = cross_kernel_matrix(&landmarks, &landmarks, &method)?;
        let kmm_inv_sqrt = symmetric_inverse_sqrt(&kmm, self.ridge)?;
        let knm = cross_kernel_matrix(&x_arr, &landmarks, &method)?;
        let features = knm.dot(&kmm_inv_sqrt);

        self.landmarks = Some(landmarks);
        self.landmark_indices = Some(indices);
        self.kmm_inv_sqrt = Some(kmm_inv_sqrt);
        Ok(pyarray2_from_f64(py, &features))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let landmarks = self
            .landmarks
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("NystromBasis is not fitted"))?;
        let indices = self
            .landmark_indices
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("NystromBasis is not fitted"))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("kernel", &self.kernel)?;
        dict.set_item("n_components", self.n_components)?;
        dict.set_item("n_features", landmarks.ncols())?;
        dict.set_item("bandwidth", self.bandwidth)?;
        dict.set_item("coef0", self.coef0)?;
        dict.set_item("degree", self.degree)?;
        dict.set_item("ridge", self.ridge)?;
        dict.set_item("landmark_indices", indices.clone())?;
        Ok(dict.into())
    }
}

#[pyclass]
pub struct RandomFourierFeatures {
    n_components: usize,
    bandwidth: f64,
    seed: Option<u64>,
    weights: Option<Array2<f64>>,
    bias: Option<Array1<f64>>,
    n_features: Option<usize>,
}

fn standard_normal_pair(rng: &mut StdRng) -> (f64, f64) {
    let u1 = rng.gen::<f64>().clamp(f64::MIN_POSITIVE, 1.0);
    let u2 = rng.gen::<f64>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    (radius * theta.cos(), radius * theta.sin())
}

#[pymethods]
impl RandomFourierFeatures {
    #[new]
    #[pyo3(signature = (n_components, bandwidth=0.5, seed=None))]
    fn new(n_components: usize, bandwidth: f64, seed: Option<u64>) -> PyResult<Self> {
        if n_components == 0 {
            return Err(PyValueError::new_err("n_components must be positive"));
        }
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            return Err(PyValueError::new_err("bandwidth must be positive"));
        }
        Ok(Self {
            n_components,
            bandwidth,
            seed,
            weights: None,
            bias: None,
            n_features: None,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        let mut rng = StdRng::seed_from_u64(self.seed.unwrap_or(0xF0471E5));
        let scale = (2.0 / self.bandwidth).sqrt();
        let mut weights = Array2::<f64>::zeros((x.ncols(), self.n_components));
        let mut flat_index = 0usize;
        while flat_index < x.ncols() * self.n_components {
            let (z1, z2) = standard_normal_pair(&mut rng);
            let row = flat_index / self.n_components;
            let col = flat_index % self.n_components;
            weights[[row, col]] = scale * z1;
            flat_index += 1;
            if flat_index < x.ncols() * self.n_components {
                let row = flat_index / self.n_components;
                let col = flat_index % self.n_components;
                weights[[row, col]] = scale * z2;
                flat_index += 1;
            }
        }
        let bias = Array1::from_shape_fn(self.n_components, |_| {
            rng.gen::<f64>() * 2.0 * std::f64::consts::PI
        });
        self.weights = Some(weights);
        self.bias = Some(bias);
        self.n_features = Some(x.ncols());
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomFourierFeatures is not fitted"))?;
        let bias = self
            .bias
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomFourierFeatures is not fitted"))?;
        let x = to_array2(&x);
        if x.ncols() != weights.nrows() {
            return Err(PyValueError::new_err(
                "x columns must match the fitted training design",
            ));
        }
        let mut features = x.dot(weights);
        let scale = (2.0 / self.n_components as f64).sqrt();
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                features[[i, j]] = (features[[i, j]] + bias[j]).cos() * scale;
            }
        }
        Ok(pyarray2_from_f64(py, &features))
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
        let mut rng = StdRng::seed_from_u64(self.seed.unwrap_or(0xF0471E5));
        let weight_scale = (2.0 / self.bandwidth).sqrt();
        let mut weights = Array2::<f64>::zeros((x.ncols(), self.n_components));
        let mut flat_index = 0usize;
        while flat_index < x.ncols() * self.n_components {
            let (z1, z2) = standard_normal_pair(&mut rng);
            let row = flat_index / self.n_components;
            let col = flat_index % self.n_components;
            weights[[row, col]] = weight_scale * z1;
            flat_index += 1;
            if flat_index < x.ncols() * self.n_components {
                let row = flat_index / self.n_components;
                let col = flat_index % self.n_components;
                weights[[row, col]] = weight_scale * z2;
                flat_index += 1;
            }
        }
        let bias = Array1::from_shape_fn(self.n_components, |_| {
            rng.gen::<f64>() * 2.0 * std::f64::consts::PI
        });
        let mut features = x.dot(&weights);
        let feature_scale = (2.0 / self.n_components as f64).sqrt();
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                features[[i, j]] = (features[[i, j]] + bias[j]).cos() * feature_scale;
            }
        }
        self.weights = Some(weights);
        self.bias = Some(bias);
        self.n_features = Some(x.ncols());
        Ok(pyarray2_from_f64(py, &features))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomFourierFeatures is not fitted"))?;
        let bias = self
            .bias
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomFourierFeatures is not fitted"))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("kernel", "gaussian")?;
        dict.set_item("n_components", self.n_components)?;
        dict.set_item("n_features", self.n_features.unwrap_or(0))?;
        dict.set_item("bandwidth", self.bandwidth)?;
        dict.set_item("weights", pyarray2_from_f64(py, weights))?;
        dict.set_item("bias", pyarray1_from_f64(py, bias))?;
        Ok(dict.into())
    }
}

#[pyclass(name = "RandomizedPCA")]
pub struct RandomizedPcaTransformer {
    n_components: usize,
    oversamples: usize,
    power_iter: usize,
    seed: Option<u64>,
    mean: Option<Array1<f64>>,
    components: Option<Array2<f64>>,
    singular_values: Option<Array1<f64>>,
}

#[pymethods]
impl RandomizedPcaTransformer {
    #[new]
    #[pyo3(signature = (n_components, oversamples=10, power_iter=1, seed=None))]
    fn new(
        n_components: usize,
        oversamples: usize,
        power_iter: usize,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        if n_components == 0 {
            return Err(PyValueError::new_err("n_components must be positive"));
        }
        if power_iter > 10 {
            return Err(PyValueError::new_err("power_iter must be <= 10"));
        }
        Ok(Self {
            n_components,
            oversamples,
            power_iter,
            seed,
            mean: None,
            components: None,
            singular_values: None,
        })
    }

    fn fit(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = to_array2(&x);
        if x.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if self.n_components > x.ncols().min(x.nrows()) {
            return Err(PyValueError::new_err(
                "n_components must be <= min(x.shape)",
            ));
        }
        let mean = x
            .mean_axis(ndarray::Axis(0))
            .ok_or_else(|| PyValueError::new_err("failed to compute mean"))?;
        let mut centered = x.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }
        let svd = randomized_svd_impl(
            &centered,
            self.n_components,
            self.oversamples,
            self.power_iter,
            self.seed,
        )?;
        self.mean = Some(mean);
        self.components = Some(svd.vt);
        self.singular_values = Some(svd.singular_values);
        Ok(())
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let mut x = to_array2(&x);
        if x.ncols() != mean.len() {
            return Err(PyValueError::new_err(
                "x columns must match the fitted training design",
            ));
        }
        for mut row in x.rows_mut() {
            row -= mean;
        }
        let scores = x.dot(&components.t());
        Ok(pyarray2_from_f64(py, &scores))
    }

    fn fit_transform<'py>(
        &mut self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_arr = to_array2(&x);
        if x_arr.nrows() == 0 {
            return Err(PyValueError::new_err("x must have at least one row"));
        }
        if self.n_components > x_arr.ncols().min(x_arr.nrows()) {
            return Err(PyValueError::new_err(
                "n_components must be <= min(x.shape)",
            ));
        }
        let mean = x_arr
            .mean_axis(ndarray::Axis(0))
            .ok_or_else(|| PyValueError::new_err("failed to compute mean"))?;
        let mut centered = x_arr.clone();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }
        let svd = randomized_svd_impl(
            &centered,
            self.n_components,
            self.oversamples,
            self.power_iter,
            self.seed,
        )?;
        let scores = centered.dot(&svd.vt.t());
        self.mean = Some(mean);
        self.components = Some(svd.vt);
        self.singular_values = Some(svd.singular_values);
        Ok(pyarray2_from_f64(py, &scores))
    }

    fn inverse_transform<'py>(
        &self,
        py: Python<'py>,
        scores: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let scores = to_array2(&scores);
        if scores.ncols() != self.n_components {
            return Err(PyValueError::new_err(
                "score columns must match n_components",
            ));
        }
        let mut reconstructed = scores.dot(components);
        for mut row in reconstructed.rows_mut() {
            row += mean;
        }
        Ok(pyarray2_from_f64(py, &reconstructed))
    }

    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let components = self
            .components
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let singular_values = self
            .singular_values
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("RandomizedPCA is not fitted"))?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("n_components", self.n_components)?;
        dict.set_item("n_features", mean.len())?;
        dict.set_item("oversamples", self.oversamples)?;
        dict.set_item("power_iter", self.power_iter)?;
        dict.set_item("mean", pyarray1_from_f64(py, mean))?;
        dict.set_item("components", pyarray2_from_f64(py, components))?;
        dict.set_item("singular_values", pyarray1_from_f64(py, singular_values))?;
        Ok(dict.into())
    }
}
