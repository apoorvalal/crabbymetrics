import numpy as np
from crabbymetrics import ElasticNet


def main() -> None:
    rng = np.random.default_rng(1)
    n = 600
    k = 6
    beta = np.array([2.0, -1.5, 0.0, 0.0, 0.8, -0.3])
    intercept = -0.4

    x = rng.normal(size=(n, k))
    y = intercept + x @ beta + rng.normal(scale=0.7, size=n)

    model = ElasticNet(penalty=0.1, l1_ratio=0.5, fit_intercept=True)
    model.fit(x, y)

    print("true intercept:", intercept)
    print("true coef:", beta)
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
