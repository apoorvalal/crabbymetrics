import numpy as np
from crabbymetrics import Poisson


def main() -> None:
    rng = np.random.default_rng(4)
    n = 700
    k = 2
    beta = np.array([0.4, -0.6])
    intercept = 0.2

    x = rng.normal(size=(n, k))
    logits = intercept + x @ beta
    mu = np.exp(logits)
    y = rng.poisson(mu).astype(float)

    model = Poisson(alpha=0.0, fit_intercept=True, max_iterations=200)
    model.fit(x, y)

    print("true intercept:", intercept)
    print("true coef:", beta)
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
