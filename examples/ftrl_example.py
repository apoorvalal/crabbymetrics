import numpy as np
from crabbymetrics import FTRL


def main() -> None:
    rng = np.random.default_rng(6)
    n = 1000
    k = 5
    beta = np.array([0.8, -1.2, 0.4, 0.0, 0.6])

    x = rng.normal(size=(n, k))
    logits = x @ beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.int32)

    model = FTRL(alpha=0.1, beta=1.0, l1_ratio=1.0, l2_ratio=1.0)
    model.fit(x, y)

    print("true coef:", beta)
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
