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


def test_nystrom_basis_full_landmarks_reconstructs_gaussian_kernel():
    rng = np.random.default_rng(909)
    x = rng.normal(size=(14, 3))
    bandwidth = 2.3

    basis = cm.NystromBasis(
        n_components=x.shape[0],
        kernel="gaussian",
        bandwidth=bandwidth,
        ridge=1e-12,
        seed=1,
    )
    features = basis.fit_transform(x)
    approx = features @ features.T
    expected = np.exp(-np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2) / bandwidth)
    summary = basis.summary()

    assert features.shape == (x.shape[0], x.shape[0])
    assert summary["kernel"] == "gaussian"
    assert summary["n_components"] == x.shape[0]
    np.testing.assert_allclose(approx, expected, atol=1e-5, rtol=1e-5)


def test_nystrom_basis_transform_shape_and_reproducibility():
    rng = np.random.default_rng(1001)
    x = rng.normal(size=(30, 4))
    x_new = rng.normal(size=(7, 4))

    first = cm.NystromBasis(8, kernel="linear", seed=42)
    second = cm.NystromBasis(8, kernel="linear", seed=42)
    first_train = first.fit_transform(x)
    second_train = second.fit_transform(x)
    first_new = first.transform(x_new)
    second_new = second.transform(x_new)

    assert first_train.shape == (30, 8)
    assert first_new.shape == (7, 8)
    np.testing.assert_allclose(first_train, second_train)
    np.testing.assert_allclose(first_new, second_new)
    assert first.summary()["landmark_indices"] == second.summary()["landmark_indices"]


def test_nystrom_basis_rejects_too_many_components():
    rng = np.random.default_rng(1112)
    x = rng.normal(size=(5, 2))
    basis = cm.NystromBasis(6)
    with np.testing.assert_raises(ValueError):
        basis.fit(x)


def test_random_fourier_features_approximate_gaussian_kernel():
    rng = np.random.default_rng(1213)
    x = rng.normal(size=(35, 3))
    bandwidth = 1.9

    rff = cm.RandomFourierFeatures(2500, bandwidth=bandwidth, seed=5)
    features = rff.fit_transform(x)
    approx = features @ features.T
    expected = np.exp(-np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2) / bandwidth)

    assert features.shape == (35, 2500)
    assert np.mean(np.abs(approx - expected)) < 0.04
    summary = rff.summary()
    assert summary["kernel"] == "gaussian"
    assert summary["weights"].shape == (3, 2500)
    assert summary["bias"].shape == (2500,)


def test_random_fourier_features_transform_is_reproducible():
    rng = np.random.default_rng(1415)
    x = rng.normal(size=(20, 2))
    x_new = rng.normal(size=(6, 2))

    first = cm.RandomFourierFeatures(64, bandwidth=0.8, seed=99)
    second = cm.RandomFourierFeatures(64, bandwidth=0.8, seed=99)
    first_train = first.fit_transform(x)
    second_train = second.fit_transform(x)
    first_new = first.transform(x_new)
    second_new = second.transform(x_new)

    np.testing.assert_allclose(first_train, second_train)
    np.testing.assert_allclose(first_new, second_new)
    assert first_new.shape == (6, 64)


def test_random_fourier_features_rejects_bad_bandwidth():
    with np.testing.assert_raises(ValueError):
        cm.RandomFourierFeatures(10, bandwidth=0.0)


def test_randomized_pca_compresses_low_rank_features_for_balancing():
    rng = np.random.default_rng(1617)
    n_source = 80
    n_target = 35
    latent_source = rng.normal(size=(n_source, 3))
    latent_target = rng.normal(loc=0.15, size=(n_target, 3))
    loadings = rng.normal(size=(3, 24))
    source = latent_source @ loadings + 0.02 * rng.normal(size=(n_source, 24))
    target = latent_target @ loadings + 0.02 * rng.normal(size=(n_target, 24))
    combined = np.vstack([source, target])

    rpca = cm.RandomizedPCA(3, oversamples=8, power_iter=2, seed=10)
    combined_scores = rpca.fit_transform(combined)
    source_scores = combined_scores[:n_source]
    target_scores = combined_scores[n_source:]

    bal = cm.BalancingWeights(objective="quadratic", max_weight=0.08, l2_norm=0.05)
    bal.fit(source_scores, target_scores)
    summary = bal.summary()

    assert combined_scores.shape == (n_source + n_target, 3)
    assert summary["success"]
    assert summary["l2_diff"] <= 0.055
    rpca_summary = rpca.summary()
    assert rpca_summary["components"].shape == (3, 24)
    reconstructed = rpca.inverse_transform(combined_scores)
    assert np.linalg.norm(reconstructed - combined) / np.linalg.norm(combined) < 0.04


def test_randomized_pca_reproducible_transform():
    rng = np.random.default_rng(1819)
    x = rng.normal(size=(25, 7))
    x_new = rng.normal(size=(6, 7))

    first = cm.RandomizedPCA(4, oversamples=4, power_iter=1, seed=77)
    second = cm.RandomizedPCA(4, oversamples=4, power_iter=1, seed=77)
    first_scores = first.fit_transform(x)
    second_scores = second.fit_transform(x)

    np.testing.assert_allclose(np.abs(first_scores), np.abs(second_scores))
    np.testing.assert_allclose(np.abs(first.transform(x_new)), np.abs(second.transform(x_new)))
