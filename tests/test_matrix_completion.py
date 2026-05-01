import numpy as np

import crabbymetrics as cm


def make_low_rank_panel(seed=0, n=36, t=28, rank=2, noise=0.03):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, rank))
    v = rng.normal(size=(t, rank))
    unit_fe = np.linspace(-1.0, 1.0, n)
    time_fe = 0.5 * np.sin(np.linspace(-2.0, 2.0, t))
    mean = u @ v.T + unit_fe[:, None] + time_fe[None, :]
    y = mean + noise * rng.normal(size=(n, t))
    return y, mean


def test_matrix_completion_improves_holdout_prediction():
    y, mean = make_low_rank_panel(seed=1)
    rng = np.random.default_rng(2)
    mask = rng.random(y.shape) < 0.65
    assert np.any(~mask)

    row_mean = np.divide(
        np.where(mask, y, 0.0).sum(axis=1),
        np.maximum(mask.sum(axis=1), 1),
    )
    col_mean = np.divide(
        np.where(mask, y, 0.0).sum(axis=0),
        np.maximum(mask.sum(axis=0), 1),
    )
    baseline = row_mean[:, None] + col_mean[None, :] - y[mask].mean()

    model = cm.MatrixCompletion(lambda_fraction=0.08, max_iterations=300, tolerance=1e-7)
    model.fit(y, mask=mask)
    pred = model.predict()
    summary = model.summary()

    holdout = ~mask
    mc_mse = np.mean((pred[holdout] - mean[holdout]) ** 2)
    baseline_mse = np.mean((baseline[holdout] - mean[holdout]) ** 2)

    assert summary["iterations"] <= 300
    assert np.sum(summary["singular_values"] > 1e-8) <= 8
    assert mc_mse < 0.55 * baseline_mse


def test_matrix_completion_large_lambda_zeroes_low_rank_component():
    y, _ = make_low_rank_panel(seed=3, n=18, t=14)
    probe = cm.MatrixCompletion(lambda_fraction=1.05, max_iterations=80, tolerance=1e-8)
    probe.fit(y)

    assert np.linalg.norm(probe.summary()["low_rank"]) < 1e-6
