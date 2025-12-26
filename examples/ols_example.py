import numpy as np
from crabbymetrics import OLS


def main() -> None:
    rng = np.random.default_rng(0)
    n = 500
    k = 3
    beta = np.array([1.5, -2.0, 0.5])
    intercept = 0.7

    x = rng.normal(size=(n, k))
    y = intercept + x @ beta + rng.normal(scale=0.5, size=n)

    model = OLS(fit_intercept=True)
    model.fit(x, y)

    print("true intercept:", intercept)
    print("true coef:", beta)
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
