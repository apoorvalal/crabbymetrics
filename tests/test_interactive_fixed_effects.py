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


def test_panel_factor_randomized_reconstructs_low_rank_surface():
    rng = np.random.default_rng(987)
    factor = rng.normal(size=(40, 3))
    loading = rng.normal(size=(22, 3))
    y = factor @ loading.T

    out = cm.panel_factor(
        y,
        3,
        factor_method="randomized",
        oversamples=6,
        power_iter=2,
        seed=77,
    )

    assert np.linalg.norm(out["fe"] - y) / np.linalg.norm(y) < 1e-8
    assert out["factor"].shape == (40, 3)
    assert out["loading"].shape == (22, 3)


def test_interactive_fixed_effects_randomized_tracks_exact_fit():
    rng = np.random.default_rng(654)
    factor = rng.normal(size=(42, 2))
    loading = rng.normal(size=(24, 2))
    alpha = np.linspace(-0.8, 0.8, 24)
    xi = np.cos(np.linspace(0, 2, 42))
    y = 0.4 + alpha[None, :] + xi[:, None] + factor @ loading.T

    exact = cm.InteractiveFixedEffects(rank=2, force=3)
    exact.fit(y)
    randomized = cm.InteractiveFixedEffects(
        rank=2,
        force=3,
        factor_method="randomized",
        factor_oversamples=6,
        factor_power_iter=2,
        factor_seed=88,
    )
    randomized.fit(y)

    summary = randomized.summary()
    assert summary["factor_method"] == "randomized"
    np.testing.assert_allclose(randomized.predict(), exact.predict(), rtol=1e-8, atol=1e-8)


def test_interactive_fixed_effects_rejects_bad_factor_method():
    with np.testing.assert_raises(ValueError):
        cm.InteractiveFixedEffects(factor_method="nearly")
