"""Read-only estimator audit probes; no estimator implementation is modified.

Run from the repository root with .venv/bin/python reviews/estimator-hardening/probes.py.
Every probe runs in a separate, time- and RSS-limited process. Inputs are synthetic.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import crabbymetrics as cm
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sample(n=120, p=3, seed=20260905):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, p))
    y = 0.5 + x @ np.linspace(0.3, 0.9, p) + rng.normal(scale=0.5, size=n)
    return rng, x, y


def outcome(function):
    try:
        result = function()
        return {"status": "returned", "value": result}
    except BaseException as exc:  # noqa: BLE001 - isolated probe captures PyO3 panics.
        return {"status": type(exc).__name__, "message": str(exc)[:350]}


def prediction_width():
    _, x, y = sample()
    results = {}
    for cls in (cm.OLS, cm.Ridge, cm.ElasticNet):
        model = cls()
        model.fit(x, y)
        results[cls.__name__] = outcome(
            lambda model=model: model.predict(x[:, :2]).shape
        )
    return results


def ridge_bad_weights():
    _, x, y = sample()
    return {
        label: outcome(
            lambda penalty=penalty: cm.Ridge(penalty=penalty).fit_weighted(
                x, y, [1.0, 1.0]
            )
        )
        for label, penalty in [("scalar", 1.0), ("grid", [0.1, 1.0])]
    }


def nonfinite_inputs():
    _, x, y = sample()
    x[0, 0] = np.nan
    results = {}
    for cls in (cm.OLS, cm.Ridge):

        def fit(cls=cls):
            model = cls()
            model.fit(x, y)
            return {
                "finite_predictions": bool(
                    np.isfinite(model.predict(np.zeros((1, 3)))).all()
                )
            }

        results[cls.__name__] = outcome(fit)

    def sc():
        model = cm.SyntheticControl()
        model.fit(x[:, :1], y)
        return model.summary()

    results["SyntheticControl_one_donor"] = outcome(sc)

    def kernel():
        model = cm.KernelBasis(bandwidth=float("nan"))
        return {
            "finite_basis": bool(
                np.isfinite(model.fit_transform(np.ones((4, 2)))).all()
            )
        }

    results["KernelBasis_nan_bandwidth"] = outcome(kernel)
    return results


def failed_refit():
    _, x, y = sample()
    results = {}
    for cls in (cm.OLS, cm.Ridge, cm.ElasticNet, cm.Logit):
        model = cls()
        target = (y > y.mean()).astype(np.int32) if cls is cm.Logit else y
        model.fit(x, target)
        before = model.predict(x)
        failure = outcome(lambda model=model, target=target: model.fit(x[:-1], target))
        after = outcome(
            lambda model=model, before=before: bool(
                np.array_equal(model.predict(x), before)
            )
        )
        results[cls.__name__] = {"refit": failure, "old_predictions_remain": after}
    return results


def elasticnet_translation():
    from sklearn.linear_model import ElasticNet

    _, x, y = sample(n=600)
    x -= x.mean(axis=0)
    shifted = x + np.array([8.0, -5.0, 12.0])
    results = {}
    for name, cls, kwargs in [
        (
            "native",
            cm.ElasticNet,
            {
                "penalty": 0.03,
                "l1_ratio": 0.35,
                "tolerance": 1e-9,
                "max_iterations": 20000,
            },
        ),
        (
            "sklearn",
            ElasticNet,
            {"alpha": 0.03, "l1_ratio": 0.35, "tol": 1e-9, "max_iter": 20000},
        ),
    ]:
        a, b = cls(**kwargs), cls(**kwargs)
        a.fit(x, y)
        b.fit(shifted, y)
        results[name] = {
            "max_prediction_change": float(
                np.max(np.abs(a.predict(x) - b.predict(shifted)))
            ),
            "centered_mse": float(np.mean((y - a.predict(x)) ** 2)),
            "shifted_mse": float(np.mean((y - b.predict(shifted)) ** 2)),
        }
        if name == "native":
            results[name]["shifted_converged"] = b.summary()["converged"]
    return results


def zero_weight_rows():
    rng, x, y = sample(n=80)
    a, b = cm.OLS(), cm.OLS()
    a.fit_weighted(x, y, np.ones(80))
    b.fit_weighted(
        np.vstack([x, rng.normal(size=x.shape)]),
        np.r_[y, rng.normal(size=80)],
        np.r_[np.ones(80), np.zeros(80)],
    )
    return {
        "max_prediction_change": float(np.max(np.abs(a.predict(x) - b.predict(x)))),
        "vanilla_se_ratio": (
            b.summary(vcov="vanilla")["coef_se"] / a.summary(vcov="vanilla")["coef_se"]
        ).tolist(),
        "expected_df_ratio": float(np.sqrt((80 - 4) / (160 - 4))),
    }


def gmm_moment_scale():
    data = np.linspace(2.0, 6.0, 100)
    results = {}
    for scale in (1.0, 1e-6, 1e-10):
        model = cm.GMM(
            lambda t, y, scale=scale: scale * (y - t[0])[:, None],
            lambda t, y, scale=scale: np.array([[-scale]]),
        )
        model.fit(data, np.array([0.0]), weighting="identity")
        summary = model.summary()
        results[str(scale)] = {
            key: summary[key] for key in ("coef", "converged", "nit", "criterion")
        }
    return {"expected_mean": float(data.mean()), "fits": results}


def gmm_vanilla():
    rng = np.random.default_rng(37)
    data = rng.normal(4, 3, 200)
    model = cm.GMM(lambda t, y: (y - t[0])[:, None], lambda t, y: np.array([[-1.0]]))
    model.fit(data, np.zeros(1))
    return {
        "vanilla": outcome(lambda: model.summary(vcov="vanilla")["se"]),
        "sandwich_se": model.summary()["se"],
        "expected_se": float(data.std(ddof=0) / np.sqrt(data.size)),
    }


def callback_data_mutation():
    original = np.linspace(-2.0, 2.0, 100)
    results = {}

    def objective(t, data):
        residual = data - t[0]
        return float(np.mean(residual**2) / 2), np.array([-residual.mean()])

    def make():
        return cm.MEstimator(objective, lambda t, data: (data - t[0])[:, None])

    for label, summarize_first in [
        ("before_first_summary", False),
        ("after_first_summary", True),
    ]:
        data = original.copy()
        model = make()
        model.fit(data, np.zeros(1))
        if summarize_first:
            model.summary()
        data *= 4
        results[label] = model.summary()["se"]
    results["original_se"] = float(original.std(ddof=0) / np.sqrt(original.size))
    return results


def cox_translation():
    rng = np.random.default_rng(38)
    x = rng.normal(size=(180, 1))
    duration = rng.exponential(np.exp(-0.7 * x[:, 0])) + 0.01
    event = rng.binomial(1, 0.8, len(x)).astype(float)
    results = {}
    for shift in (0.0, 100.0, 1000.0):

        def fit(shift=shift):
            model = cm.CoxPH()
            model.fit(x + shift, duration, event)
            summary = model.summary()
            return {
                key: summary[key]
                for key in ("coef", "log_likelihood", "converged", "iterations")
            }

        results[str(shift)] = outcome(fit)
    return results


def cox_scaling():
    results = []
    for n in (100, 200, 400, 800):
        rng = np.random.default_rng(39)
        x = rng.normal(size=(n, 3))
        duration = rng.exponential(np.exp(-0.4 * x[:, 0])) + 0.01
        event = rng.binomial(1, 0.8, n).astype(float)
        times = []
        for _ in range(3):
            model = cm.CoxPH()
            start = time.perf_counter()
            model.fit(x, duration, event)
            times.append(time.perf_counter() - start)
        results.append(
            {
                "n": n,
                "median_seconds": float(np.median(times)),
                "iterations": model.summary()["iterations"],
            }
        )
    slope = np.polyfit(
        np.log([r["n"] for r in results]),
        np.log([r["median_seconds"] for r in results]),
        1,
    )[0]
    return {"measurements": results, "log_log_slope": float(slope)}


def ill_conditioned_ols():
    rng = np.random.default_rng(40)
    z = rng.normal(size=300)
    x = np.column_stack([z, z + 1e-7 * rng.normal(size=300)])
    y = z + rng.normal(scale=0.1, size=300)
    model = cm.OLS()
    model.fit(x, y)
    design = np.column_stack([np.ones(300), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    _, r = np.linalg.qr(design, mode="reduced")
    r_inv = np.linalg.solve(r, np.eye(3))
    expected = np.sqrt(
        np.diag(r_inv @ r_inv.T) * np.sum((y - design @ beta) ** 2) / (300 - 3)
    )[1:]
    actual = model.summary(vcov="vanilla")["coef_se"]
    return {
        "condition_number": float(np.linalg.cond(design)),
        "native_se": actual,
        "qr_reference_se": expected,
        "max_relative_se_error": float(np.max(np.abs(actual / expected - 1))),
    }


def matrix_completion_diagnostics():
    rng = np.random.default_rng(41)
    y = rng.normal(size=(10, 16))
    w = np.zeros_like(y)
    w[7:, 10:] = 1
    model = cm.MatrixCompletion(
        lambda_l=0.05, max_iterations=1, fit_unit_effects=False, fit_time_effects=False
    )
    model.fit(y, w)
    summary = model.summary()
    fitted = model.predict()
    mask = w == 0
    return {
        "recorded_last_rmse": float(summary["history_rmse"][-1]),
        "returned_fit_rmse": float(np.sqrt(np.mean((y[mask] - fitted[mask]) ** 2))),
        "starting_zero_fit_rmse": float(np.sqrt(np.mean(y[mask] ** 2))),
        "converged": summary["converged"],
        "summary_completed_counterfactual_share_memory": bool(
            np.shares_memory(summary["completed"], summary["counterfactual"])
        ),
    }


def balancing_nan():
    rng = np.random.default_rng(42)
    source = rng.normal(size=(80, 2))
    target = source[:40].copy()
    source[0, 0] = np.nan
    model = cm.BalancingWeights(autoscale=True)
    model.fit(source, target)
    return model.summary()


def logit_rank_deficiency():
    x = np.ones((100, 1))
    y = (np.arange(100) % 2).astype(np.int32)
    model = cm.Logit()
    model.fit(x, y)
    summary = model.summary()
    return {
        "design_rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(100), x]))),
        "parameter_count": 2,
        **{
            key: summary[key]
            for key in (
                "coef",
                "intercept",
                "coef_se",
                "intercept_se",
                "inference_available",
                "converged",
            )
        },
    }


def gil_responsiveness(n=2600):
    rng = np.random.default_rng(45)
    x = rng.normal(size=(n, 3))
    duration = rng.exponential(np.exp(-0.4 * x[:, 0])) + 0.01
    event = rng.binomial(1, 0.8, n).astype(float)
    elapsed = []

    def timed_call(function):
        observed = []
        started = time.perf_counter()
        timer = threading.Timer(
            0.01, lambda: observed.append(time.perf_counter() - started)
        )
        timer.start()
        function()
        wall = time.perf_counter() - started
        timer.join()
        return {"call_seconds": wall, "timer_delay_seconds": observed[0]}

    baseline = timed_call(lambda: time.sleep(0.15))
    for _ in range(2):
        model = cm.CoxPH()
        elapsed.append(timed_call(lambda model=model: model.fit(x, duration, event)))
    return {
        "n": n,
        "requested_timer_delay": 0.01,
        "sleep_control": baseline,
        "native_fits": elapsed,
    }


def gil_responsiveness_large():
    return gil_responsiveness(n=200000)


def ridge_grid_reuse():
    _, x, y = sample(n=1200, p=24)
    penalties = np.logspace(-2, 2, 8)
    records = {}
    for label in ("grid", "repeated_scalar_fits"):
        times = []
        for _ in range(3):
            started = time.perf_counter()
            if label == "grid":
                model = cm.Ridge(penalty=penalties, cv=5)
                model.fit(x, y)
            else:
                for penalty in penalties:
                    for fold in range(5):
                        keep = np.arange(len(y)) % 5 != fold
                        model = cm.Ridge(penalty=float(penalty))
                        model.fit(x[keep], y[keep])
                        model.predict(x[~keep])
                    model = cm.Ridge(penalty=float(penalty))
                    model.fit(x, y)
            times.append(time.perf_counter() - started)
        records[label] = {"median_seconds": float(np.median(times)), "times": times}
    return {
        "n": len(y),
        "p": x.shape[1],
        "penalties": len(penalties),
        "folds": 5,
        "timings": records,
        "note": "Repeated scalar fits are a work-equivalent QR proxy, not the old native grid implementation.",
    }


CASES = {
    fn.__name__: fn
    for fn in [
        prediction_width,
        ridge_bad_weights,
        nonfinite_inputs,
        failed_refit,
        elasticnet_translation,
        zero_weight_rows,
        gmm_moment_scale,
        gmm_vanilla,
        callback_data_mutation,
        cox_translation,
        cox_scaling,
        ill_conditioned_ols,
        matrix_completion_diagnostics,
        balancing_nan,
        logit_rank_deficiency,
        gil_responsiveness,
        gil_responsiveness_large,
        ridge_grid_reuse,
    ]
}


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("evidence-after.json")
    )
    args = parser.parse_args()
    if args.case:
        print(json.dumps(json_safe(outcome(CASES[args.case])), allow_nan=False))
        return
    sys.path.insert(0, str(ROOT))
    from benchmarks.scaling.process_runner import run_process
    from benchmarks.scaling.run_grid import THREAD_ENV

    records = {}
    for name in CASES:
        result = run_process(
            [sys.executable, str(Path(__file__).resolve()), "--case", name],
            os.environ | THREAD_ENV,
            20,
            1024**3,
        )
        lines = result.stdout.strip().splitlines()
        try:
            payload = (
                json.loads(lines[-1])
                if result.status is None
                else {"status": result.status}
            )
        except (IndexError, json.JSONDecodeError):
            payload = {"status": "invalid_result", "stdout": result.stdout[-1000:]}
        records[name] = {
            "result": payload,
            "wall_seconds": result.wall_seconds,
            "peak_rss_bytes": result.peak_rss_bytes,
            "returncode": result.returncode,
        }
        print(name, json.dumps(json_safe(payload), allow_nan=False), flush=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    evidence = {
        "commit": commit,
        "source_dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "src",
                    "Cargo.toml",
                    "Cargo.lock",
                ],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("crabbymetrics", "numpy", "scipy", "scikit-learn")
        },
        "thread_environment": THREAD_ENV,
        "cases": records,
    }
    args.output.write_text(
        json.dumps(json_safe(evidence), indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
