"""No-covariate parity tests adapted from xuyiqing/fect testthat simulations.

The original fect tests exercise formula calls such as `Y ~ D` with no X's:
- test-simulation-ife.R: factor-model IFE DGP and r=0 FE equivalence
- test-book-claims.R: no-covariate MC smoke tests

crabbymetrics currently exposes the Rust-backed lower-level panel surfaces rather
than the full R formula/treatment wrapper, so these tests validate the same
core no-covariate pieces: additive FE, panel factors, and MC-style treatment-cell
imputation.
"""

import numpy as np

import crabbymetrics as cm


def make_fect_ife_dgp(seed=3001, n=30, tt=15, t0=10, ntr=10, tau=2.0, noise=0.25):
    rng = np.random.default_rng(seed)
    alpha_i = rng.normal(0.0, 1.0, size=n)
    xi_t = rng.normal(0.0, 0.5, size=tt)
    lambda_i = rng.normal(0.0, 1.0, size=n)
    f_t = rng.normal(0.0, 1.0, size=tt)

    y0 = xi_t[:, None] + alpha_i[None, :] + f_t[:, None] * lambda_i[None, :]
    y0 = y0 + noise * rng.normal(size=(tt, n))
    d = np.zeros((tt, n), dtype=bool)
    d[t0:, :ntr] = True
    y = y0 + tau * d
    return y, y0, d


def two_way_additive_projection(y):
    """Closed-form balanced two-way additive FE projection matching force=3,r=0."""
    return y.mean() + (y.mean(axis=0) - y.mean())[None, :] + (y.mean(axis=1) - y.mean())[:, None]


def test_no_covariate_ife_factor_dgp_recovers_untreated_surface():
    # Adapted from fect/tests/testthat/test-simulation-ife.R's no-covariate
    # one-factor DGP, but run on Y(0) because this is the lower-level IFE fit.
    _, y0, _ = make_fect_ife_dgp(noise=0.0)

    model = cm.InteractiveFixedEffects(rank=1, force=3)
    model.fit(y0)
    pred = model.predict()

    assert np.linalg.norm(pred - y0) / np.linalg.norm(y0) < 1e-10


def test_no_covariate_ife_r0_matches_two_way_fe_projection():
    rng = np.random.default_rng(3002)
    n, tt = 30, 15
    alpha_i = rng.normal(0.0, 1.0, size=n)
    xi_t = rng.normal(0.0, 0.5, size=tt)
    y = xi_t[:, None] + alpha_i[None, :] + 0.5 * rng.normal(size=(tt, n))

    model = cm.InteractiveFixedEffects(rank=0, force=3)
    model.fit(y)

    assert np.allclose(model.predict(), two_way_additive_projection(y))


def test_no_covariate_mc_imputes_treated_cells_close_to_known_tau():
    # Adapted from fect's method='mc' no-covariate smoke tests.  The full R
    # wrapper estimates ATT from Y ~ D; here we mask treated cells and check the
    # Rust MatrixCompletion counterfactual recovers the known ATT reasonably.
    tau = 2.0
    y, y0, d = make_fect_ife_dgp(seed=8070, n=40, tt=20, t0=12, ntr=12, tau=tau, noise=0.05)
    # MatrixCompletion now uses the common n_units x n_periods Y/W panel API.
    y_nt = y.T
    y0_nt = y0.T
    w_nt = d.T.astype(float)

    model = cm.MatrixCompletion(lambda_fraction=0.04, max_iterations=400, tolerance=1e-7)
    model.fit(y_nt, w_nt)
    yhat0 = model.predict()
    treated = w_nt > 0.5
    att = np.mean(y_nt[treated] - yhat0[treated])

    assert abs(att - tau) < 0.5
    assert np.mean((yhat0[treated] - y0_nt[treated]) ** 2) < 0.5
