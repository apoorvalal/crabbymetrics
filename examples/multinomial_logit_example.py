import numpy as np
from crabbymetrics import MultinomialLogit


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exps = np.exp(x)
    return exps / exps.sum(axis=1, keepdims=True)


def main() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    k = 3
    c = 3
    coef = np.array(
        [
            [1.0, -0.5, 0.2],
            [-0.7, 0.9, -0.4],
            [0.2, -0.3, 0.8],
        ]
    )
    intercept = np.array([0.3, -0.2, 0.0])

    x = rng.normal(size=(n, k))
    logits = x @ coef.T + intercept
    probs = softmax(logits)
    y = np.array([rng.choice(c, p=probs[i]) for i in range(n)], dtype=np.int32)

    model = MultinomialLogit(alpha=1.0, fit_intercept=True, max_iterations=200)
    model.fit(x, y)

    print("true intercept:", intercept)
    print("true coef:", coef)
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
