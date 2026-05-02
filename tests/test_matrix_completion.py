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


def make_treatment(n, t, n_treated=10, t0=18):
    w = np.zeros((n, t))
    w[:n_treated, t0:] = 1.0
    return w


def test_matrix_completion_improves_treated_cell_prediction():
    y, mean = make_low_rank_panel(seed=1)
    w = make_treatment(*y.shape, n_treated=12, t0=18)
    mask = w < 0.5

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
    model.fit(y, w)
    pred = model.predict()
    summary = model.summary()

    treated_cells = w > 0.5
    mc_mse = np.mean((pred[treated_cells] - mean[treated_cells]) ** 2)
    baseline_mse = np.mean((baseline[treated_cells] - mean[treated_cells]) ** 2)

    assert summary["iterations"] <= 300
    assert np.sum(summary["singular_values"] > 1e-8) <= 8
    assert mc_mse < 0.65 * baseline_mse
    assert "event_study" in summary
    assert "group_means" in summary
    assert "weighted" in summary["group_means"]
    assert "unweighted" in summary["group_means"]


def test_matrix_completion_large_lambda_zeroes_low_rank_component():
    y, _ = make_low_rank_panel(seed=3, n=18, t=14)
    w = make_treatment(*y.shape, n_treated=5, t0=9)
    probe = cm.MatrixCompletion(lambda_fraction=1.05, max_iterations=80, tolerance=1e-8)
    probe.fit(y, w)

    assert np.linalg.norm(probe.summary()["low_rank"]) < 1e-6
