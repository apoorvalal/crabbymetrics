import numpy as np
from crabbymetrics import MEstimator, Poisson


call_counter = {'n': 0}

def poisson_objective(theta, data):
    """
    Poisson negative log-likelihood and gradient.

    With safeguards against numerical overflow.

    Returns:
        (objective, gradient) tuple
    """
    call_counter['n'] += 1

    X = data['X']
    y = data['y']
    indices = data.get('indices', np.arange(len(y)))

    X_sample = X[indices]
    y_sample = y[indices]

    eta = X_sample @ theta

    # Clip eta to prevent overflow in exp()
    eta = np.clip(eta, -20, 20)

    mu = np.exp(eta)

    obj = np.sum(mu - y_sample * eta)
    grad = X_sample.T @ (mu - y_sample)

    if call_counter['n'] <= 5 or call_counter['n'] % 20 == 0:
        print(f"    Call {call_counter['n']}: theta={theta}, obj={obj:.2f}, |grad|={np.linalg.norm(grad):.2f}")

    return obj, grad


def poisson_scores(theta, data):
    """
    Compute per-observation score derivatives for sandwich variance.

    Returns:
        Array of shape (n_obs, n_params) with dq_i/dtheta for each observation
    """
    X = data['X']
    y = data['y']

    eta = X @ theta
    # Clip to match objective function
    eta = np.clip(eta, -20, 20)
    mu = np.exp(eta)

    scores = X * (mu - y)[:, np.newaxis]

    return scores


def main():
    rng = np.random.default_rng(42)
    n = 700
    k = 3
    beta_true = np.array([0.2, 0.4, -0.6])

    X = rng.normal(size=(n, k))
    eta = X @ beta_true
    mu = np.exp(eta)
    y = rng.poisson(mu).astype(float)

    print("=" * 60)
    print("Poisson Regression Comparison")
    print("=" * 60)
    print(f"True coefficients: {beta_true}")
    print()

    print("1. Using MEstimator (general M-estimation framework)")
    print("-" * 60)

    data = {'X': X, 'y': y, 'n': n}
    # Better initialization: use log(mean(y)) as a rough starting point
    theta0 = np.ones(k) * 0.1

    # Test objective function at initial point
    obj_init, grad_init = poisson_objective(theta0, data)
    print(f"  Initial objective: {obj_init:.4f}")
    print(f"  Initial gradient norm: {np.linalg.norm(grad_init):.4f}")

    mestim = MEstimator(
        objective_fn=poisson_objective,
        score_fn=poisson_scores,
        max_iterations=200,
        tolerance=1e-6
    )

    mestim.fit(data, theta0)
    summary_m = mestim.summary()

    print(f"Estimated coefficients: {summary_m['coef']}")
    print(f"Standard errors:        {summary_m['se']}")
    print()

    print("2. Using built-in Poisson estimator")
    print("-" * 60)

    poisson_model = Poisson(alpha=0.0, fit_intercept=False, max_iterations=200)
    poisson_model.fit(X, y)
    summary_p = poisson_model.summary()

    print(f"Estimated coefficients: {summary_p['coef']}")
    print(f"Standard errors:        {summary_p['coef_se']}")
    print()

    print("3. Comparison")
    print("-" * 60)
    coef_diff = np.abs(summary_m['coef'] - summary_p['coef'])
    se_diff = np.abs(summary_m['se'] - summary_p['coef_se'])

    print(f"Max coefficient difference: {np.max(coef_diff):.2e}")
    print(f"Max SE difference:          {np.max(se_diff):.2e}")
    print(f"Max relative coef error:    {np.max(coef_diff / np.abs(summary_p['coef'])):.2%}")
    print(f"Max relative SE error:      {np.max(se_diff / summary_p['coef_se']):.2%}")

    if np.max(coef_diff) < 1e-3 and np.max(se_diff / summary_p['coef_se']) < 0.05:
        print("\nSUCCESS: MEstimator matches built-in Poisson!")
    else:
        print("\nWARNING: Results differ more than expected")

    print()
    print("4. Testing bootstrap (5 iterations)")
    print("-" * 60)
    boot_draws = mestim.bootstrap(5, seed=123)
    print(f"Bootstrap draws shape: {boot_draws.shape}")
    print(f"Bootstrap mean: {boot_draws.mean(axis=0)}")
    print(f"Bootstrap std:  {boot_draws.std(axis=0)}")


if __name__ == "__main__":
    main()
