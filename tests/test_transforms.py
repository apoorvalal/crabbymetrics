import numpy as np

import crabbymetrics as cm


def test_pca_recovers_low_rank_structure_and_feeds_regression():
    rng = np.random.default_rng(4242)
    n = 500
    n_features = 6

    factors = rng.normal(size=(n, 2))
    loadings = np.array(
        [
            [1.4, -0.2, 0.9, 0.0, 0.7, -0.1],
            [0.1, 1.2, -0.5, 1.1, 0.0, 0.8],
        ]
    )
    x = factors @ loadings + 0.03 * rng.normal(size=(n, n_features))
    y = 0.4 + factors @ np.array([1.3, -0.9]) + 0.05 * rng.normal(size=n)

    pca = cm.PCA(2)
    scores = pca.fit_transform(x)
    scores_again = pca.transform(x[:37])
    reconstructed = pca.inverse_transform(scores)
    summary = pca.summary()

    assert scores.shape == (n, 2)
    assert scores_again.shape == (37, 2)
    assert reconstructed.shape == x.shape
    assert summary["components"].shape == (2, n_features)
    assert summary["mean"].shape == (n_features,)
    assert summary["explained_variance"].shape == (2,)
    assert summary["explained_variance_ratio"].shape == (2,)
    np.testing.assert_allclose(
        np.sum(summary["explained_variance_ratio"]),
        1.0,
        atol=1e-10,
        rtol=0.0,
    )

    baseline_mse = np.mean((x - x.mean(axis=0, keepdims=True)) ** 2)
    reconstructed_mse = np.mean((x - reconstructed) ** 2)
    assert reconstructed_mse < 0.05 * baseline_mse

    model = cm.OLS()
    model.fit(scores, y)
    preds = model.predict(scores)
    r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2 > 0.97


def test_kernel_basis_matches_linear_kernel_exactly():
    rng = np.random.default_rng(707)
    x_train = rng.normal(size=(12, 4))
    x_new = rng.normal(size=(5, 4))

    basis = cm.KernelBasis("linear")
    gram = basis.fit_transform(x_train)
    cross = basis.transform(x_new)
    summary = basis.summary()

    np.testing.assert_allclose(gram, x_train @ x_train.T, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cross, x_new @ x_train.T, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(summary["diagonal"], np.sum(x_train**2, axis=1))
    assert summary["kernel"] == "linear"
    assert summary["n_train"] == x_train.shape[0]
    assert summary["n_features"] == x_train.shape[1]


def test_kernel_basis_matches_gaussian_cross_kernel():
    rng = np.random.default_rng(808)
    x_train = rng.normal(size=(9, 3))
    x_new = rng.normal(size=(4, 3))
    bandwidth = 1.7

    basis = cm.KernelBasis("gaussian", bandwidth=bandwidth)
    gram = basis.fit_transform(x_train)
    cross = basis.transform(x_new)

    expected_gram = np.exp(
        -np.sum((x_train[:, None, :] - x_train[None, :, :]) ** 2, axis=2) / bandwidth
    )
    expected_cross = np.exp(
        -np.sum((x_new[:, None, :] - x_train[None, :, :]) ** 2, axis=2) / bandwidth
    )

    np.testing.assert_allclose(gram, expected_gram, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(cross, expected_cross, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.diag(gram), np.ones(x_train.shape[0]))
