"""Ablation for randomized sketching OLS.

Run after building the extension, e.g.

    uv run maturin develop
    uv run python benchmarks/sketching_ols_ablation.py

The script compares full OLS against dense Rademacher sketch sizes expressed as
multiples of the design dimension. It is intentionally small enough to run on a
laptop while still making the bias/variance tradeoff visible.
"""

import json
import time

import numpy as np

import crabbymetrics as cm


def run_once(n, p, noise, multiple, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    intercept = 0.4
    y = intercept + x @ beta + noise * rng.normal(size=n)
    sketch_size = int(multiple * (p + 1))

    full = cm.OLS()
    t0 = time.perf_counter()
    full.fit(x, y)
    full_time = time.perf_counter() - t0
    full_summary = full.summary(vcov="vanilla")
    full_param = np.concatenate([[full_summary["intercept"]], full_summary["coef"]])

    sketch = cm.OLS()
    t0 = time.perf_counter()
    sketch.fit_sketch(x, y, sketch_size=sketch_size, seed=seed + 10_000)
    sketch_time = time.perf_counter() - t0
    sketch_summary = sketch.summary(vcov="vanilla")
    sketch_param = np.concatenate([[sketch_summary["intercept"]], sketch_summary["coef"]])

    x_test = rng.normal(size=(2000, p))
    full_pred = full.predict(x_test)
    sketch_pred = sketch.predict(x_test)

    return {
        "n": n,
        "p": p,
        "noise": noise,
        "multiple": multiple,
        "sketch_size": sketch_size,
        "coef_rel_error": float(
            np.linalg.norm(sketch_param - full_param) / np.linalg.norm(full_param)
        ),
        "prediction_rmse_vs_full": float(np.sqrt(np.mean((sketch_pred - full_pred) ** 2))),
        "full_fit_seconds": full_time,
        "sketch_fit_seconds": sketch_time,
        "speedup": full_time / sketch_time if sketch_time > 0 else np.inf,
    }


def main():
    n = 20_000
    p = 40
    noise = 0.2
    multiples = [2, 4, 8, 12, 16]
    seeds = range(5)
    rows = [run_once(n, p, noise, multiple, seed) for multiple in multiples for seed in seeds]

    summary = []
    for multiple in multiples:
        block = [row for row in rows if row["multiple"] == multiple]
        summary.append(
            {
                "multiple": multiple,
                "sketch_size": block[0]["sketch_size"],
                "median_coef_rel_error": float(np.median([r["coef_rel_error"] for r in block])),
                "median_prediction_rmse_vs_full": float(
                    np.median([r["prediction_rmse_vs_full"] for r in block])
                ),
                "median_speedup": float(np.median([r["speedup"] for r in block])),
            }
        )

    print(json.dumps({"rows": rows, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
