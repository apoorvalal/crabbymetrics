import numpy as np

import crabbymetrics as cm


def test_panel_factor_reconstructs_rank_two_surface():
    rng = np.random.default_rng(123)
    factor = rng.normal(size=(24, 2))
    loading = rng.normal(size=(16, 2))
    y = factor @ loading.T

    out = cm.panel_factor(y, 2)

    assert np.linalg.norm(out["fe"] - y) / np.linalg.norm(y) < 1e-10
    assert out["factor"].shape == (24, 2)
    assert out["loading"].shape == (16, 2)


def test_panel_fe_matches_fect_soft_threshold_scaling():
    rng = np.random.default_rng(456)
    y = rng.normal(size=(12, 9))
    lam = 0.0005

    out = cm.panel_fe(y, lam, hard=False)
    u, s, vt = np.linalg.svd(y / y.size, full_matrices=False)
    shrunk = np.where(s > lam, s - lam, 0.0)
    expected = (u * shrunk) @ vt * y.size

    assert np.allclose(out["fe"], expected)
    assert np.allclose(out["singular_values"], shrunk)


def test_interactive_fixed_effects_recovers_low_rank_plus_additive_signal():
    rng = np.random.default_rng(321)
    factor = rng.normal(size=(30, 2))
    loading = rng.normal(size=(20, 2))
    alpha = np.linspace(-1, 1, 20)
    xi = np.sin(np.linspace(0, 2, 30))
    y = 0.7 + alpha[None, :] + xi[:, None] + factor @ loading.T

    model = cm.InteractiveFixedEffects(rank=2, force=3)
    model.fit(y)
    pred = model.predict()
    summary = model.summary()

    assert np.linalg.norm(pred - y) / np.linalg.norm(y) < 1e-10
    assert summary["factor"].shape == (30, 2)
    assert summary["loading"].shape == (20, 2)
    assert summary["rank"] == 2
    assert summary["force"] == 3
