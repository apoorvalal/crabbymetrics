import numpy as np
from crabbymetrics import Logit


def main() -> None:
    rng = np.random.default_rng(2)
    n = 800
    k = 4
    beta = np.array([1.2, -0.8, 0.4, -1.1])
    intercept = 0.3

    x = rng.normal(size=(n, k))
    logits = intercept + x @ beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.int32)

    model = Logit(alpha=1.0, fit_intercept=True, max_iterations=200)
    model.fit(x, y)

    print("true intercept:", intercept)
    print("true coef:", beta)
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
