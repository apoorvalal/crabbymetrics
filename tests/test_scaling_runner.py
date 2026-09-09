"""Regression coverage for guarded execution and persisted benchmark results."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import psutil
import pytest


@pytest.fixture
def scaling(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    return SimpleNamespace(
        grid=importlib.import_module("benchmarks.scaling.run_grid"),
        process=importlib.import_module("benchmarks.scaling.process_runner"),
        cell=importlib.import_module("benchmarks.scaling.benchmark_cell"),
    )


@pytest.mark.parametrize("raw", ["", "0", "-1", "1,,2", "1,", "nan", "1.5"])
def test_grid_rejects_invalid_dimensions(scaling, raw):
    with pytest.raises(argparse.ArgumentTypeError):
        scaling.grid.parse_ints(raw)


def test_grid_sorts_and_deduplicates_for_monotone_pruning(scaling):
    assert scaling.grid.parse_ints("10_000,100,1000,100") == (100, 1000, 10000)


@pytest.mark.parametrize(
    "argv",
    [
        ["--timeout", "0"],
        ["--timeout", "nan"],
        ["--memory-gib", "-1"],
        ["--reserve-gib", "inf"],
        ["--reserve-gib", "-1"],
        ["--seed", "-1"],
    ],
)
def test_grid_rejects_invalid_budgets(scaling, argv):
    with pytest.raises(SystemExit):
        scaling.grid.parse_args(argv)


def test_memory_cap_never_exceeds_reserve_or_explicit_limit(scaling, monkeypatch):
    memory = SimpleNamespace(total=16 * 2**30, available=3 * 2**30)
    monkeypatch.setattr(scaling.grid.psutil, "virtual_memory", lambda: memory)
    args = scaling.grid.parse_args([])
    assert scaling.grid.choose_memory_cap(args) == 0
    args.reserve_gib = 0
    args.memory_gib = 0.125
    assert scaling.grid.choose_memory_cap(args) == 128 * 2**20
    args.memory_gib = 100
    assert scaling.grid.choose_memory_cap(args) == memory.available


def test_selection_rejects_typos_and_provenance_only_references(scaling):
    with pytest.raises(SystemExit):
        scaling.grid.selected_estimators(" , ")
    for name in ("", "sklearn-rigde", "r-synth"):
        with pytest.raises(SystemExit):
            scaling.grid.selected_implementations("SyntheticControl", name)
    assert scaling.grid.selected_implementations("OLS", "sklearn-ridge") == ()
    assert scaling.grid.selected_implementations(
        "Ridge", "sklearn-ridge,sklearn-ridge"
    ) == ("sklearn-ridge",)


def test_csv_append_preserves_column_meanings_and_adds_new_fields(scaling, tmp_path):
    path = tmp_path / "grid.csv"
    path.write_text("status,n,legacy\nok,100,keep\n", encoding="utf-8")
    scaling.grid.write_rows(path, [{"n": 200, "status": "timeout"}], append=True)
    scaling.grid.write_rows(
        path, [{"n": 300, "status": "ok", "fit_seconds": 0.5}], append=True
    )
    scaling.grid.write_rows(path, [], append=True)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"status": "ok", "n": "100", "legacy": "keep", "fit_seconds": ""},
        {"status": "timeout", "n": "200", "legacy": "", "fit_seconds": ""},
        {"status": "ok", "n": "300", "legacy": "", "fit_seconds": "0.5"},
    ]
    scaling.grid.write_rows(path, [{"n": 7, "status": "dry_run"}], append=False)
    with path.open(newline="") as handle:
        assert list(csv.DictReader(handle)) == [{"n": "7", "status": "dry_run"}]


def test_appending_retains_previous_host_configurations(scaling, tmp_path):
    args = scaling.grid.parse_args(["--output", str(tmp_path / "grid.csv")])
    path = tmp_path / "grid.host.json"
    scaling.grid.write_host_metadata(path, args, 123)
    args.append = True
    scaling.grid.write_host_metadata(path, args, 456)
    scaling.grid.write_host_metadata(path, args, 789)
    payload = json.loads(path.read_text())
    assert payload["memory_cap_bytes"] == 789
    assert [run["memory_cap_bytes"] for run in payload["previous_runs"]] == [123, 456]


@pytest.mark.parametrize(
    "stdout,returncode",
    [
        ("not JSON", 0),
        ('{"status":"unknown"}', 0),
        ('{"status":"ok"}', 0),
        ('{"status":"ok","fit_seconds":1,"checksum":0}', 1),
        ('{"status":"ok","fit_seconds":-1,"checksum":0}', 0),
        ('{"status":"ok","fit_seconds":NaN,"checksum":0}', 0),
        ('{"status":"ok","fit_seconds":1,"checksum":Infinity}', 0),
    ],
)
def test_invalid_child_results_are_errors(scaling, stdout, returncode):
    assert scaling.grid.parse_payload(stdout, returncode)["status"] == "error"


def test_result_parser_ignores_diagnostic_lines(scaling):
    payload = {"status": "ok", "fit_seconds": 0.2, "checksum": 1}
    stdout = f"library log\n{json.dumps(payload)}\n{{not json}}\nnull\n"
    assert scaling.grid.parse_payload(stdout, 0) == payload
    assert (
        scaling.grid.parse_payload('{"status":"missing_dependency"}', 1)["status"]
        == "missing_dependency"
    )


def test_missing_executable_is_a_recorded_cell_failure(scaling, monkeypatch):
    monkeypatch.setattr(
        scaling.grid,
        "child_command",
        lambda *args: ["/nonexistent/crabbymetrics-runner"],
    )
    result = scaling.grid.run_cell(
        "OLS", "crabbymetrics", 100, 2, scaling.grid.parse_args([]), 2**30
    )
    assert result["status"] == "missing_dependency"


def test_verbose_child_does_not_block_on_full_output_pipes(scaling):
    code = "import sys; sys.stdout.write('x' * 2_000_000); sys.stderr.write('y' * 2_000_000); print('\\nfinished')"
    result = scaling.process.run_process(
        [sys.executable, "-c", code], os.environ.copy(), 5, 2**30
    )
    assert result.status is None
    assert result.returncode == 0
    assert result.stdout.endswith("finished\n")
    assert len(result.stdout) <= 1024 * 1024
    assert result.stderr == "y" * 2000


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cleanup")
@pytest.mark.parametrize("parent_exits", [False, True])
def test_cleanup_terminates_grandchildren(scaling, parent_exits):
    code = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print(child.pid, flush=True); "
        f"time.sleep({0 if parent_exits else 30})"
    )
    result = scaling.process.run_process(
        [sys.executable, "-c", code], os.environ.copy(), 1, 2**30
    )
    assert result.status == (None if parent_exits else "timeout")
    child_pid = int(result.stdout.strip())
    try:
        child = psutil.Process(child_pid)
        child.wait(timeout=3)
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired:
        # Minimal Linux containers may not reap reparented children promptly.
        assert child.status() == psutil.STATUS_ZOMBIE
        return
    assert not psutil.pid_exists(child_pid)


def test_rss_guard_terminates_the_cell(scaling, monkeypatch):
    monkeypatch.setattr(scaling.process, "descendants_rss", lambda process: 1024)
    result = scaling.process.run_process(
        [sys.executable, "-c", "import time; time.sleep(30)"], os.environ.copy(), 5, 512
    )
    assert result.status == "killed_rss_guard"
    assert result.peak_rss_bytes == 1024


def test_disappearing_process_is_not_a_monitor_failure(scaling):
    class ExitedProcess:
        def children(self, recursive):
            raise psutil.NoSuchProcess(123)

        def memory_info(self):
            raise psutil.NoSuchProcess(123)

    assert scaling.process.descendants_rss(ExitedProcess()) == 0


def test_interrupt_reaps_the_running_cell(scaling, monkeypatch):
    processes = []
    real_popen = scaling.process.subprocess.Popen

    def capture_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process

    def interrupt(process):
        raise KeyboardInterrupt

    monkeypatch.setattr(scaling.process.subprocess, "Popen", capture_popen)
    monkeypatch.setattr(scaling.process, "descendants_rss", interrupt)
    with pytest.raises(KeyboardInterrupt):
        scaling.process.run_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            os.environ.copy(),
            5,
            2**30,
        )
    assert processes[0].poll() is not None


def test_completed_cells_survive_later_interruptions(scaling, monkeypatch, tmp_path):
    output = tmp_path / "grid.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_grid",
            "--estimators",
            "OLS",
            "--implementations",
            "native",
            "--n",
            "100,200",
            "--k",
            "2",
            "--output",
            str(output),
        ],
    )

    def run_cell(estimator, implementation, n, k, args, cap):
        if n == 200:
            raise KeyboardInterrupt
        return {"n": n, "status": "ok", "fit_seconds": 0.1}

    monkeypatch.setattr(scaling.grid, "run_cell", run_cell)
    with pytest.raises(KeyboardInterrupt):
        scaling.grid.main()
    with output.open(newline="") as handle:
        assert next(csv.DictReader(handle))["n"] == "100"


def test_failure_prunes_larger_dimensions(scaling, monkeypatch, tmp_path):
    output = tmp_path / "grid.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_grid",
            "--estimators",
            "OLS",
            "--implementations",
            "native",
            "--n",
            "200,100",
            "--k",
            "2",
            "--output",
            str(output),
        ],
    )
    calls = []

    def run_cell(estimator, implementation, n, k, args, cap):
        calls.append(n)
        return {"n": n, "status": "missing_dependency"}

    monkeypatch.setattr(scaling.grid, "run_cell", run_cell)
    assert scaling.grid.main() == 0
    assert calls == [100]
    with output.open(newline="") as handle:
        assert [row["status"] for row in csv.DictReader(handle)] == [
            "missing_dependency",
            "pruned_after_failure",
        ]


def test_fit_timer_does_not_reuse_previous_results(scaling):
    timer = scaling.cell.FitTimer()
    assert timer(lambda: 7) == 7
    assert timer.seconds is not None
    with pytest.raises(ValueError):
        timer(int, "invalid")
    assert timer.seconds is None


def test_checksum_does_not_mask_nonfinite_coefficients(scaling):
    assert np.isnan(
        scaling.cell.result_checksum(SimpleNamespace(coef_=np.array([np.nan])))
    )


class CaptureFit:
    def __call__(self, function, *args, **kwargs):
        self.model = getattr(function, "__self__", None)
        self.args = args
        self.kwargs = kwargs
        return function(*args, **kwargs)


@pytest.mark.parametrize(
    "estimator,implementation",
    [
        ("Logit", "sklearn-logistic-regression"),
        ("MultinomialLogit", "sklearn-multinomial-logit"),
        ("Poisson", "sklearn-poisson-regressor"),
    ],
)
def test_glm_adapters_fit_the_same_unpenalized_problem(
    scaling, estimator, implementation
):
    native, reference = CaptureFit(), CaptureFit()
    scaling.cell.run_crabbymetrics(
        estimator, 600, 3, np.random.default_rng(17), measured_fit=native
    )
    scaling.cell.run_sklearn(
        implementation, 600, 3, np.random.default_rng(17), measured_fit=reference
    )
    x, y = native.args
    np.testing.assert_array_equal(x, reference.args[0])
    np.testing.assert_array_equal(y, reference.args[1])
    expected = (
        reference.model.predict(x)
        if estimator == "Poisson"
        else reference.model.predict_proba(x)
    )
    if estimator == "Logit":
        expected = expected[:, 1]
    np.testing.assert_allclose(native.model.predict(x), expected, atol=2e-3, rtol=2e-3)
    if estimator == "Poisson":
        assert reference.model.alpha == 0
    else:
        assert reference.model.C == np.inf


def test_horizontal_ridge_adapters_use_the_same_panel(scaling):
    native, reference = CaptureFit(), CaptureFit()
    scaling.cell.run_crabbymetrics(
        "HorizontalPanelRidge", 120, 8, np.random.default_rng(17), measured_fit=native
    )
    scaling.cell.run_horizontal_ridge(
        120, 8, np.random.default_rng(17), measured_fit=reference
    )
    y, w = native.args
    np.testing.assert_array_equal(reference.args[0], y[:4, :80].T)
    np.testing.assert_array_equal(reference.args[1], y[4:, :80].mean(axis=0))
    expected = reference.model.predict(y[:4].T)
    np.testing.assert_allclose(
        native.model.predict()[4:], np.tile(expected, (4, 1)), atol=1e-10
    )
    assert w[4:, 80:].all()


def test_elastic_net_adapters_center_the_same_design(scaling):
    native, reference = CaptureFit(), CaptureFit()
    scaling.cell.run_crabbymetrics(
        "ElasticNet", 600, 3, np.random.default_rng(17), measured_fit=native
    )
    scaling.cell.run_sklearn(
        "sklearn-elastic-net", 600, 3, np.random.default_rng(17), measured_fit=reference
    )
    x, y = native.args
    np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(y.mean(), 0, atol=1e-15)
    np.testing.assert_array_equal(x, reference.args[0])
    np.testing.assert_array_equal(y, reference.args[1])
    np.testing.assert_allclose(
        native.model.predict(x), reference.model.predict(x), atol=1e-4
    )


def test_iv_adapters_use_the_same_outcome_and_instruments(scaling):
    native, reference = CaptureFit(), CaptureFit()
    scaling.cell.run_crabbymetrics(
        "TwoSLS", 120, 3, np.random.default_rng(17), measured_fit=native
    )
    scaling.cell.run_pyfixest(
        "pyfixest-iv", 120, 3, np.random.default_rng(17), measured_fit=reference
    )
    endog, exog, z, y = native.args
    frame = reference.kwargs["data"]
    np.testing.assert_array_equal(frame["y"], y)
    np.testing.assert_array_equal(frame["d"], endog[:, 0])
    np.testing.assert_array_equal(frame[["x0", "x1", "x2"]], exog)
    np.testing.assert_array_equal(frame[["z0", "z1", "z2"]], z)


@pytest.mark.parametrize("n,k", [(2, 4), (10, 3), (10, 1)])
def test_panel_data_rejects_impossible_dimensions(scaling, n, k):
    with pytest.raises(ValueError, match="panel benchmarks require"):
        scaling.cell.panel_data(n, k, np.random.default_rng(17))
