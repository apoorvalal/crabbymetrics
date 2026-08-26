#!/usr/bin/env python3
"""Run the estimator scaling grid with subprocess, memory, and timeout guards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil
from registry import (
    ESTIMATORS,
    REFERENCE_URLS,
    implementations,
    runnable_implementations,
)

HERE = Path(__file__).resolve().parent
CELL_RUNNER = HERE / "benchmark_cell.py"
DEFAULT_NS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
DEFAULT_KS = (5, 10, 20, 50, 100)
FAILURE_STATUSES = {"killed_rss_guard", "preflight_oom", "timeout", "error"}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RCPP_PARALLEL_NUM_THREADS": "1",
}


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.replace("_", "")) for part in raw.split(",") if part.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", default="all", help="comma list or 'all'")
    parser.add_argument(
        "--implementations", default="all", help="comma list, 'native', or 'all'"
    )
    parser.add_argument("--n", type=parse_ints, default=DEFAULT_NS)
    parser.add_argument("--k", type=parse_ints, default=DEFAULT_KS)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--memory-gib", type=float, default=None)
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent.parent
        / "docs"
        / "ablations"
        / "data"
        / "estimator-scaling.csv",
    )
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def choose_memory_cap(args: argparse.Namespace) -> int:
    total = psutil.virtual_memory().total
    available = psutil.virtual_memory().available
    reserve = int(args.reserve_gib * 2**30)
    auto = min(int(total * 0.40), max(512 * 2**20, available - reserve))
    requested = int(args.memory_gib * 2**30) if args.memory_gib is not None else auto
    return max(256 * 2**20, min(requested, max(256 * 2**20, available - reserve)))


def estimated_peak_bytes(estimator: str, n: int, k: int) -> int:
    spec = ESTIMATORS[estimator]
    base = n * k * 8
    if spec["family"] == "dynamic":
        base = max(50, n // 4) * 4 * k * 8
    # Account for Python generation + foreign-library copies/workspaces; deliberately conservative.
    return int(base * spec["memory_factor"] + 192 * 2**20)


def descendants_rss(process: psutil.Process) -> int:
    total = 0
    for proc in [process, *process.children(recursive=True)]:
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def terminate_tree(process: psutil.Process) -> None:
    children = process.children(recursive=True)
    for proc in reversed(children):
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        return
    _, alive = psutil.wait_procs([*children, process], timeout=2)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def child_command(
    estimator: str, implementation: str, n: int, k: int, seed: int
) -> list[str]:
    if implementation.startswith("r-"):
        return [
            "Rscript",
            str(HERE / "reference_runner.R"),
            estimator,
            implementation,
            str(n),
            str(k),
            str(seed),
        ]
    return [
        sys.executable,
        str(CELL_RUNNER),
        "--estimator",
        estimator,
        "--implementation",
        implementation,
        "--n",
        str(n),
        "--k",
        str(k),
        "--seed",
        str(seed),
    ]


def run_cell(
    estimator: str,
    implementation: str,
    n: int,
    k: int,
    args: argparse.Namespace,
    cap_bytes: int,
) -> dict[str, Any]:
    predicted = estimated_peak_bytes(estimator, n, k)
    common: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "estimator": estimator,
        "implementation": implementation,
        "n": n,
        "k": k,
        "predicted_peak_bytes": predicted,
        "memory_cap_bytes": cap_bytes,
        "timeout_seconds": args.timeout,
    }
    if predicted > cap_bytes:
        return {**common, "status": "preflight_oom", "peak_rss_bytes": 0}

    command = child_command(estimator, implementation, n, k, args.seed)
    if args.dry_run:
        return {
            **common,
            "status": "dry_run",
            "command": shlex.join(command),
            "peak_rss_bytes": 0,
        }

    env = os.environ.copy()
    env.update(THREAD_ENV)
    started = time.perf_counter()
    popen = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    process = psutil.Process(popen.pid)
    peak_rss = 0
    status = "error"
    while popen.poll() is None:
        peak_rss = max(peak_rss, descendants_rss(process))
        elapsed = time.perf_counter() - started
        if peak_rss > cap_bytes:
            status = "killed_rss_guard"
            terminate_tree(process)
            break
        if elapsed > args.timeout:
            status = "timeout"
            terminate_tree(process)
            break
        time.sleep(0.05)
    stdout, stderr = popen.communicate()
    wall = time.perf_counter() - started
    if status not in {"killed_rss_guard", "timeout"}:
        lines = [line for line in stdout.splitlines() if line.strip().startswith("{")]
        try:
            payload = json.loads(lines[-1])
            status = payload.pop("status", "error")
        except (IndexError, json.JSONDecodeError):
            payload = {"error": "child emitted no valid JSON"}
    else:
        payload = {}
    return {
        **common,
        **payload,
        "status": status,
        "wall_seconds": wall,
        "peak_rss_bytes": peak_rss,
        "stderr_tail": (
            stderr[-2000:].replace(str(HERE.parent.parent), "<repo>")
            if status != "ok"
            else ""
        ),
    }


def selected_estimators(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return tuple(ESTIMATORS)
    selected = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(selected) - set(ESTIMATORS))
    if unknown:
        raise SystemExit(f"unknown estimators: {', '.join(unknown)}")
    return selected


def selected_implementations(estimator: str, raw: str) -> tuple[str, ...]:
    available = implementations(estimator)
    if raw == "all":
        return runnable_implementations(estimator)
    if raw == "native":
        return ("crabbymetrics",)
    requested = tuple(item.strip() for item in raw.split(",") if item.strip())
    return tuple(item for item in requested if item in available)


def write_rows(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def write_host_metadata(path: Path, args: argparse.Namespace, cap_bytes: int) -> None:
    try:
        output_label = str(args.output.resolve().relative_to(HERE.parent.parent))
    except ValueError:
        output_label = str(args.output)
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "memory_cap_bytes": cap_bytes,
        "arguments": vars(args) | {"output": output_label},
        "reference_urls": REFERENCE_URLS,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    cap_bytes = choose_memory_cap(args)
    rows: list[dict[str, Any]] = []
    pruned: set[tuple[str, str, int]] = set()
    for estimator in selected_estimators(args.estimators):
        for implementation in selected_implementations(estimator, args.implementations):
            for k in args.k:
                for n in args.n:
                    key = (estimator, implementation, k)
                    if key in pruned:
                        rows.append(
                            {
                                "estimator": estimator,
                                "implementation": implementation,
                                "n": n,
                                "k": k,
                                "status": "pruned_after_failure",
                            }
                        )
                        continue
                    row = run_cell(estimator, implementation, n, k, args, cap_bytes)
                    rows.append(row)
                    print(
                        f"{estimator:28} {implementation:34} n={n:>9} k={k:>3} "
                        f"{row['status']}",
                        flush=True,
                    )
                    if row["status"] in FAILURE_STATUSES:
                        pruned.add(key)
    write_rows(args.output, rows, args.append)
    write_host_metadata(args.output.with_suffix(".host.json"), args, cap_bytes)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
