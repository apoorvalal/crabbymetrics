"""Independent numerical references and failure contracts for smooth solvers."""
import numpy as np
import pytest
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess
from scipy.special import expit

import crabbymetrics as cm


QUASI = (cm.Optimizers.minimize_bfgs, cm.Optimizers.minimize_lbfgs)


def test_lbfgs_handles_many_parameters_with_vector_history():
    result = cm.Optimizers.minimize_lbfgs(
        lambda x: float(0.5 * x @ x), np.ones(20000), lambda x: x.copy(),
        max_iterations=10, gtol=1e-10,
    )
    assert result["success"]
    np.testing.assert_allclose(result["x"], np.zeros(20000), atol=1e-10)


@pytest.mark.parametrize("solver", QUASI)
def test_rosenbrock_agrees_with_independent_minimum(solver):
    x0 = np.array([-1.2, 1.0])
    result = solver(rosen, x0, rosen_der, max_iterations=1000, gtol=1e-8)
    assert result["success"], result
    np.testing.assert_allclose(result["x"], np.ones(2), atol=1e-6)
    np.testing.assert_array_equal(x0, [-1.2, 1.0])
    np.testing.assert_allclose(result["jac"], rosen_der(result["x"]))
    assert result["optimality"] <= 1e-8


@pytest.mark.parametrize("solver", QUASI)
def test_overflowing_trial_is_rejected_without_losing_finite_iterate(solver):
    with np.errstate(over="ignore", invalid="ignore"):
        result = solver(
            lambda x: float(np.exp(x[0]) - 1e5 * x[0]), np.zeros(1),
            lambda x: np.array([np.exp(x[0]) - 1e5]),
            max_iterations=200, gtol=1e-6,
        )
    assert result["success"], result
    np.testing.assert_allclose(result["x"], [np.log(1e5)], atol=1e-10)
    assert result["rejected_steps"] > 0


@pytest.mark.parametrize("solver", QUASI)
def test_cost_stagnation_does_not_claim_gradient_convergence(solver):
    result = solver(
        rosen, np.array([-1.2, 1.0]), rosen_der,
        max_iterations=1000, gtol=1e-12, ftol=1.0,
    )
    assert not result["success"]
    assert result["status"] == "cost_stagnation"
    assert result["optimality"] > 1e-12


@pytest.mark.parametrize("solver", QUASI)
def test_budget_exhaustion_retains_honest_diagnostics(solver):
    result = solver(rosen, np.array([-1.2, 1.0]), rosen_der, max_iterations=1, gtol=1e-12)
    assert not result["success"]
    assert result["status"] == "max_iterations"
    assert result["nit"] == 1
    assert np.isfinite(result["fun"])


@pytest.mark.parametrize("solver", QUASI)
def test_evaluation_counters_include_final_certificate(solver):
    count = {"f": 0, "g": 0}

    def fun(x):
        count["f"] += 1
        return float((x[0] - 2.0) ** 2)

    def grad(x):
        count["g"] += 1
        return 2.0 * (x - 2.0)

    result = solver(fun, np.array([5.0]), grad)
    assert result["success"]
    assert result["nfev"] == count["f"]
    assert result["njev"] == count["g"]


def test_trust_region_rosenbrock():
    result = cm.Optimizers.minimize_trust_region(
        rosen, np.array([-1.2, 1.0]), rosen_der, rosen_hess,
        max_iterations=200, gtol=1e-9,
    )
    assert result["success"], result
    np.testing.assert_allclose(result["x"], np.ones(2), atol=1e-7)
    assert result["nhev"] > 1
    assert result["min_eigenvalue"] > 0.0


def test_trust_region_escapes_zero_gradient_strict_saddle():
    result = cm.Optimizers.minimize_trust_region(
        lambda x: float((x[0] ** 2 - 1.0) ** 2 + x[1] ** 2), np.zeros(2),
        lambda x: np.array([4 * x[0] * (x[0] ** 2 - 1.0), 2 * x[1]]),
        lambda x: np.diag([12 * x[0] ** 2 - 4, 2.0]), gtol=1e-10,
    )
    assert result["success"], result
    assert abs(abs(result["x"][0]) - 1) < 1e-8
    assert result["fun"] < 1e-16
    assert result["nit"] > 0


def test_trust_region_negative_curvature_at_budget_is_not_success():
    result = cm.Optimizers.minimize_trust_region(
        lambda x: float(-x[0] ** 2), np.zeros(1), lambda x: -2 * x,
        lambda x: np.array([[-2.0]]), max_iterations=1,
    )
    assert not result["success"]
    assert result["status"] == "negative_curvature_remaining"


def test_public_trust_region_solves_indefinite_hard_case():
    h = np.diag([-2.0, 1.0])
    g = np.array([0.0, 1.0])
    result = cm.Optimizers.minimize_trust_region(
        lambda x: float(g @ x + 0.5 * x @ h @ x), np.zeros(2),
        lambda x: g + h @ x, lambda x: h, max_iterations=1,
    )
    # Analytic global solution of this radius-one quadratic subproblem.
    np.testing.assert_allclose(abs(result["x"][0]), np.sqrt(8 / 9), atol=1e-10)
    np.testing.assert_allclose(result["x"][1], -1 / 3, atol=1e-10)
    assert not result["success"]  # The original unconstrained quadratic is unbounded.


@pytest.mark.parametrize("kind", ("bfgs", "lbfgs", "trust_region"))
def test_coordinate_scaling_recovers_ill_scaled_quadratic(kind):
    scale = np.array([1e6, 1e-6])
    target = np.array([2e6, -3e-6])
    fun = lambda x: float(0.5 * np.sum(((x - target) / scale) ** 2))
    grad = lambda x: (x - target) / scale ** 2
    args = (fun, np.zeros(2), grad)
    if kind == "trust_region":
        args += (lambda x: np.diag(1 / scale ** 2),)
    result = getattr(cm.Optimizers, "minimize_" + kind)(*args, x_scale=scale, gtol=1e-10)
    assert result["success"], result
    np.testing.assert_allclose(result["x"] / scale, target / scale, atol=1e-9)
    np.testing.assert_allclose(result["jac"], grad(result["x"]))
    assert result["optimality"] == pytest.approx(np.max(np.abs(result["jac"] * scale)))


def logit_problem(seed=721):
    rng = np.random.default_rng(seed)
    X = np.column_stack((np.ones(800), rng.normal(size=(800, 3))))
    y = rng.binomial(1, expit(X @ np.array([0.3, -0.7, 0.2, 0.6])))
    fun = lambda b: float(np.mean(np.logaddexp(0, X @ b) - y * (X @ b)))
    scores = lambda b: X * (expit(X @ b) - y)[:, None]
    return X, y, fun, scores


@pytest.mark.parametrize("reduction", ("mean", "sum"))
def test_bhhh_row_scores_and_batched_statistics_match_independent_logit(reduction):
    X, y, mean_fun, scores = logit_problem()
    factor = len(y) if reduction == "sum" else 1
    fun = lambda b: factor * mean_fun(b)
    reference = minimize(mean_fun, np.zeros(4), jac=lambda b: scores(b).mean(axis=0),
                         method="BFGS", options={"gtol": 1e-11})

    def stats(b):
        total, outer = np.zeros(4), np.zeros((4, 4))
        for ix in np.array_split(np.arange(len(y)), 7):
            block = X[ix] * (expit(X[ix] @ b) - y[ix])[:, None]
            total += block.sum(axis=0)
            outer += block.T @ block
        return total, outer, len(y)

    results = [
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4), scores, reduction=reduction,
                                  max_iterations=200, gtol=1e-7),
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4), score_stats=stats, reduction=reduction,
                                  max_iterations=200, gtol=1e-7),
    ]
    for result in results:
        assert result["success"], result
        np.testing.assert_allclose(result["x"], reference.x, atol=2e-6)
        expected_gradient = factor * scores(result["x"]).mean(axis=0)
        np.testing.assert_allclose(result["jac"], expected_gradient, atol=1e-11)
        assert result["nscorev"] > 0
    np.testing.assert_allclose(results[0]["x"], results[1]["x"], atol=1e-8)


def test_bhhh_keeps_penalty_out_of_individual_scores():
    X, y, fun, scores = logit_problem()
    penalty = np.diag([0.0, 0.2, 0.2, 0.2])
    penalized = lambda b: fun(b) + 0.5 * b @ penalty @ b
    gradient = lambda b: scores(b).mean(axis=0) + penalty @ b
    reference = minimize(penalized, np.zeros(4), jac=gradient, method="BFGS",
                         options={"gtol": 1e-10})
    result = cm.Optimizers.minimize_bhhh(
        penalized, np.zeros(4), scores, penalty_grad=lambda b: penalty @ b,
        penalty_hess=lambda b: penalty.copy(), gtol=1e-8, max_iterations=200,
    )
    assert result["success"], result
    np.testing.assert_allclose(result["x"], reference.x, atol=1e-6)
    np.testing.assert_allclose(result["jac"], gradient(result["x"]), atol=1e-12)


def test_bhhh_poisson_with_offset_matches_exact_hessian_reference():
    rng = np.random.default_rng(181)
    X = np.column_stack([np.ones(1000), rng.normal(size=(1000, 2))])
    offset = rng.uniform(-0.5, 0.5, len(X))
    y = rng.poisson(np.exp(offset + X @ np.array([0.2, -0.4, 0.3])))
    fun = lambda b: float(np.mean(np.exp(offset + X @ b) - y * (offset + X @ b)))
    scores = lambda b: X * (np.exp(offset + X @ b) - y)[:, None]
    reference = minimize(fun, np.zeros(3), jac=lambda b: scores(b).mean(axis=0),
        hess=lambda b: X.T @ (np.exp(offset + X @ b)[:, None] * X) / len(X),
        method="trust-exact", options={"gtol": 1e-11})
    result = cm.Optimizers.minimize_bhhh(fun, np.zeros(3), scores,
                                       max_iterations=200, gtol=1e-8)
    assert result["success"], result
    np.testing.assert_allclose(result["x"], reference.x, atol=1e-7)


@pytest.mark.parametrize("kind", ("bfgs", "lbfgs", "trust_region", "bhhh"))
def test_exact_start_has_zero_iterations(kind):
    fun = lambda x: float(0.5 * x @ x)
    if kind == "bhhh":
        result = cm.Optimizers.minimize_bhhh(fun, np.zeros(2), lambda x: x[None, :])
    else:
        args = (fun, np.zeros(2), lambda x: x.copy())
        if kind == "trust_region":
            args += (lambda x: np.eye(2),)
        result = getattr(cm.Optimizers, "minimize_" + kind)(*args)
    assert result["success"]
    assert result["nit"] == 0


@pytest.mark.parametrize("kwargs, message", [
    ({"max_iterations": 0}, "budgets"),
    ({"gtol": 0.0}, "Gradient tolerance"),
    ({"ftol": -1.0}, "ftol"),
    ({"c1": 0.9, "c2": 0.1}, "Wolfe"),
    ({"x_scale": np.array([0.0, 1.0])}, "x_scale"),
])
def test_invalid_quasi_controls_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        cm.Optimizers.minimize_bfgs(rosen, np.array([-1.2, 1.0]), rosen_der, **kwargs)


def test_malformed_callbacks_raise_and_user_exceptions_propagate():
    with pytest.raises(ValueError, match="length"):
        cm.Optimizers.minimize_bfgs(rosen, np.ones(2), lambda x: np.ones(3))
    with pytest.raises(ValueError, match="finite at x0"):
        cm.Optimizers.minimize_bfgs(lambda x: np.nan, np.ones(2), rosen_der)
    with pytest.raises(ValueError, match="symmetric"):
        cm.Optimizers.minimize_trust_region(
            rosen, np.ones(2), rosen_der, lambda x: np.array([[1.0, 2.0], [0.0, 1.0]]))

    def broken(x):
        raise RuntimeError("intentional callback error")

    with pytest.raises(RuntimeError, match="intentional callback error"):
        cm.Optimizers.minimize_bfgs(broken, np.ones(2), rosen_der)


def test_bhhh_summed_likelihood_accepts_roundoff_ties():
    rng = np.random.default_rng(7)
    X = np.column_stack([np.ones(1200), rng.normal(size=(1200, 3))])
    y = rng.binomial(1, expit(X @ np.array([0.35, -0.95, 0.9, -0.4])))
    fun = lambda b: float(np.sum(np.logaddexp(0, X @ b) - y * (X @ b)))
    scores = lambda b: X * (expit(X @ b) - y)[:, None]
    result = cm.Optimizers.minimize_bhhh(fun, np.zeros(4), scores,
        reduction="sum", max_iterations=300, gtol=1e-6)
    assert result["success"], result
    assert np.max(np.abs(scores(result["x"]).sum(axis=0))) <= 1e-6


def test_bhhh_validates_score_contract():
    X, y, fun, scores = logit_problem()
    with pytest.raises(ValueError, match="exactly one"):
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4))
    with pytest.raises(ValueError, match="exactly one"):
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4), scores, score_stats=lambda x: None)
    with pytest.raises(ValueError, match="together"):
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4), scores, penalty_grad=lambda x: x)
    with pytest.raises(ValueError, match="scores shape"):
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4), lambda x: np.ones((8, 3)))
    with pytest.raises(ValueError, match="positive semidefinite"):
        cm.Optimizers.minimize_bhhh(fun, np.zeros(4),
            score_stats=lambda x: (np.ones(4), -np.eye(4), 1))


@pytest.mark.parametrize("kind", ("bfgs", "trust_region"))
@pytest.mark.parametrize("family", ("Poisson", "NLS"))
def test_scaled_poisson_and_distant_nls_regressions(kind, family):
    rng = np.random.default_rng(20261114)
    design = np.column_stack([np.ones(300), rng.normal(size=(300, 3))])
    truth = np.array([0.2, 0.1, -0.2, 0.3])
    if family == "Poisson":
        X = design * np.array([1.0, 1.0, 100.0, 10000.0])
        y = rng.poisson(np.exp(design @ truth))
        start = np.zeros(4)
        fun = lambda b: float(np.mean(np.exp(X @ b) - y * (X @ b)))
        grad = lambda b: X.T @ (np.exp(X @ b) - y) / len(X)
        hess = lambda b: X.T @ (np.exp(X @ b)[:, None] * X) / len(X)
    else:
        X = design
        y = np.exp(X @ truth) + 0.1 * rng.normal(size=len(X))
        start = -6 * truth
        fun = lambda b: float(0.5 * np.mean((np.exp(X @ b) - y) ** 2))
        grad = lambda b: X.T @ (np.exp(X @ b) * (np.exp(X @ b) - y)) / len(X)
        hess = lambda b: X.T @ ((np.exp(X @ b) * (2*np.exp(X @ b) - y))[:, None] * X) / len(X)
    scale = np.sqrt(np.mean(X**2, axis=0))
    reference = minimize(lambda z: fun(z / scale), np.zeros(4),
        jac=lambda z: grad(z / scale) / scale,
        hess=lambda z: hess(z / scale) / np.outer(scale, scale),
        method="trust-exact", options={"gtol": 1e-10, "maxiter": 1000})
    args = (fun, start, grad) + ((hess,) if kind == "trust_region" else ())
    with np.errstate(over="ignore", invalid="ignore"):
        result = getattr(cm.Optimizers, "minimize_" + kind)(*args, max_iterations=1000, gtol=1e-8)
    assert result["success"], result
    assert fun(result["x"]) - reference.fun < 1e-9
    assert np.max(np.abs(grad(result["x"]) / scale)) < 1e-6
