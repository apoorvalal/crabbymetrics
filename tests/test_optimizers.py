import numpy as np

import crabbymetrics as cm


def quadratic(x):
    return float((x[0] - 1.0) ** 2 + 2.0 * (x[1] + 2.0) ** 2)


def quadratic_grad(x):
    return np.array([2.0 * (x[0] - 1.0), 4.0 * (x[1] + 2.0)])


def test_gradient_based_optimizers_match_quadratic_minimum():
    x0 = np.array([4.5, 3.0])
    target = np.array([1.0, -2.0])

    for method in (
        cm.Optimizers.minimize_lbfgs,
        cm.Optimizers.minimize_bfgs,
        cm.Optimizers.minimize_nonlinear_cg,
    ):
        result = method(quadratic, x0, quadratic_grad, max_iterations=200)
        np.testing.assert_allclose(result["x"], target, atol=1e-6, rtol=0.0)
        assert result["fun"] < 1e-10
        assert result["nit"] > 0
        assert result["success"]
        assert result["method"] in {"lbfgs", "bfgs", "nonlinear_cg"}


def residual(theta):
    return np.array([theta[0] - 1.0, np.sqrt(2.0) * (theta[1] + 2.0)])


def jacobian(theta):
    del theta
    return np.array([[1.0, 0.0], [0.0, np.sqrt(2.0)]])


def test_gauss_newton_ls_matches_least_squares_solution():
    x0 = np.array([5.0, 5.0])
    result = cm.Optimizers.minimize_gauss_newton_ls(
        residual, x0, jacobian, max_iterations=25
    )

    np.testing.assert_allclose(result["x"], np.array([1.0, -2.0]), atol=1e-8, rtol=0.0)
    assert result["fun"] < 1e-12
    assert result["success"]
    assert result["method"] == "gauss_newton_ls"


def test_simulated_annealing_finds_low_cost_solution():
    def objective(x):
        return float((x[0] - 1.5) ** 2)

    x0 = np.array([6.0])
    result = cm.Optimizers.minimize_simulated_annealing(
        objective,
        x0,
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
        temp=6.0,
        step_size=0.5,
        max_iterations=3000,
        seed=123,
    )

    assert result["fun"] < 0.1
    assert abs(result["x"][0] - 1.5) < 0.35
    assert not result["success"]
    assert result["method"] == "simulated_annealing"



def test_gauss_newton_accepts_an_exact_initial_solution():
    target = np.array([1.0, -2.0])
    result = cm.Optimizers.minimize_gauss_newton_ls(
        residual,
        target,
        jacobian,
        max_iterations=5,
        tolerance=1e-10,
    )

    np.testing.assert_allclose(result["x"], target, atol=0.0, rtol=0.0)
    assert result["fun"] == 0.0
    assert result["nit"] == 0
    assert result["success"]


def test_gauss_newton_rejects_incompatible_jacobian_shape():
    def bad_jacobian(theta):
        del theta
        return np.ones((1, 2))

    with np.testing.assert_raises_regex(ValueError, "Jacobian shape"):
        cm.Optimizers.minimize_gauss_newton_ls(
            residual,
            np.array([5.0, 5.0]),
            bad_jacobian,
        )
