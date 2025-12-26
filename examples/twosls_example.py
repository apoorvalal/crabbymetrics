import numpy as np
from crabbymetrics import TwoSLS


def main() -> None:
    rng = np.random.default_rng(5)
    n = 800

    intercept = 0.5
    beta_endog = 1.4
    beta_exog = -0.8

    z = rng.normal(size=(n, 1))
    x_exog = rng.normal(size=(n, 1))

    u = rng.normal(size=n)
    x_endog = 0.9 * z[:, 0] + 0.4 * u + rng.normal(scale=0.1, size=n)

    y = intercept + beta_endog * x_endog + beta_exog * x_exog[:, 0] + u + rng.normal(
        scale=0.1, size=n
    )

    x_endog = x_endog.reshape(-1, 1)

    model = TwoSLS(fit_intercept=True)
    model.fit(x_endog, x_exog, z, y)

    print("true intercept:", intercept)
    print("true coef:", np.array([beta_endog, beta_exog]))
    print("summary:", model.summary())


if __name__ == "__main__":
    main()
