#!/usr/bin/env python3
"""Run the estimator scaling grid with subprocess, memory, and timeout guards."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

if __package__:
    from .process_runner import run_process
    from .registry import (
        ADAPTER_REVISION,
        ESTIMATORS,
        REFERENCE_URLS,
        runnable_implementations,
    )
else:
    from process_runner import run_process
    from registry import (
        ADAPTER_REVISION,
        ESTIMATORS,
        REFERENCE_URLS,
        runnable_implementations,
    )

HERE = Path(__file__).resolve().parent
CELL_RUNNER = HERE / "benchmark_cell.py"
DEFAULT_NS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
DEFAULT_KS = (5, 10, 20, 50, 100)
FAILURE_STATUSES = {
    "killed_rss_guard",
    "preflight_oom",
    "timeout",
    "error",
    "missing_dependency",
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RCPP_PARALLEL_NUM_THREADS": "1",
}


def parse_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.replace("_", "")) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        ) from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("grid dimensions must be positive integers")
    return tuple(sorted(set(values)))


def positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimators", default="all", help="comma list or 'all'")
    parser.add_argument(
        "--implementations", default="all", help="comma list, 'native', or 'all'"
    )
    parser.add_argument("--n", type=parse_ints, default=DEFAULT_NS)
    parser.add_argument("--k", type=parse_ints, default=DEFAULT_KS)
    parser.add_argument("--timeout", type=positive_float, default=60.0)
    parser.add_argument("--memory-gib", type=positive_float, default=None)
    parser.add_argument("--reserve-gib", type=nonnegative_float, default=4.0)
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
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def choose_memory_cap(args: argparse.Namespace) -> int:
    memory = psutil.virtual_memory()
    reserve = int(args.reserve_gib * 2**30)
    headroom = max(0, memory.available - reserve)
    auto = min(int(memory.total * 0.40), headroom)
    requested = int(args.memory_gib * 2**30) if args.memory_gib is not None else auto
    return min(requested, headroom)


def estimated_peak_bytes(estimator: str, n: int, k: int) -> int:
    spec = ESTIMATORS[estimator]
    base = n * k * 8
    if spec["family"] == "dynamic":
        base = max(50, n // 4) * 4 * k * 8
    # Account for Python generation + foreign-library copies/workspaces; deliberately conservative.
    return int(base * spec["memory_factor"] + 192 * 2**20)


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
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": estimator,
        "implementation": implementation,
        "n": n,
        "k": k,
        "seed": args.seed,
        "adapter_revision": ADAPTER_REVISION,
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
    try:
        result = run_process(command, env, args.timeout, cap_bytes)
    except OSError as exc:
        status = "missing_dependency" if isinstance(exc, FileNotFoundError) else "error"
        return {**common, "status": status, "error": str(exc), "peak_rss_bytes": 0}
    payload = (
        parse_payload(result.stdout, result.returncode) if result.status is None else {}
    )
    status = result.status or payload.pop("status")
    return {
        **payload,
        **common,
        "status": status,
        "returncode": result.returncode,
        "wall_seconds": result.wall_seconds,
        "peak_rss_bytes": result.peak_rss_bytes,
        "stderr_tail": (
            result.stderr.replace(str(HERE.parent.parent), "<repo>")
            if status != "ok"
            else ""
        ),
    }


def parse_payload(stdout: str, returncode: int) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "status" not in payload:
            continue
        if payload["status"] not in ("ok", "error", "missing_dependency"):
            return {"status": "error", "error": "child emitted an unknown status"}
        if payload["status"] == "ok":
            if returncode != 0:
                return {
                    "status": "error",
                    "error": f"child exited with code {returncode} after reporting success",
                }
            for name in ("fit_seconds", "checksum"):
                value = payload.get(name)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    return {"status": "error", "error": f"child emitted invalid {name}"}
            if payload["fit_seconds"] < 0:
                return {
                    "status": "error",
                    "error": "child emitted negative fit_seconds",
                }
        return payload
    return {"status": "error", "error": "child emitted no valid result JSON"}


def selected_estimators(raw: str) -> tuple[str, ...]:
    if raw == "all":
        return tuple(ESTIMATORS)
    selected = tuple(
        dict.fromkeys(item.strip() for item in raw.split(",") if item.strip())
    )
    if not selected:
        raise SystemExit("select at least one estimator")
    unknown = sorted(set(selected) - set(ESTIMATORS))
    if unknown:
        raise SystemExit(f"unknown estimators: {', '.join(unknown)}")
    return selected


def selected_implementations(estimator: str, raw: str) -> tuple[str, ...]:
    available = runnable_implementations(estimator)
    if raw == "all":
        return runnable_implementations(estimator)
    if raw == "native":
        return ("crabbymetrics",)
    requested = tuple(
        dict.fromkeys(item.strip() for item in raw.split(",") if item.strip())
    )
    known = {item for name in ESTIMATORS for item in runnable_implementations(name)}
    unknown = sorted(set(requested) - known)
    if unknown or not requested:
        raise SystemExit(
            f"unknown or non-runnable implementations: {', '.join(unknown) or raw}"
        )
    return tuple(item for item in requested if item in available)


def write_rows(path: Path, rows: list[dict[str, Any]], append: bool) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = {key for row in rows for key in row}
    existing_fields: list[str] = []
    if append and path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle))
        if fields <= set(existing_fields):
            with path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(
                    handle, fieldnames=existing_fields, lineterminator="\n"
                ).writerows(rows)
            return
    fieldnames = [*existing_fields, *sorted(fields - set(existing_fields))]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, newline="", encoding="utf-8", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            if existing_fields:
                with path.open(newline="", encoding="utf-8") as old:
                    writer.writerows(csv.DictReader(old))
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_host_metadata(path: Path, args: argparse.Namespace, cap_bytes: int) -> None:
    try:
        output_label = str(args.output.resolve().relative_to(HERE.parent.parent))
    except ValueError:
        output_label = str(args.output)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "adapter_revision": ADAPTER_REVISION,
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
    if args.append and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        payload["previous_runs"] = [*previous.pop("previous_runs", []), previous]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    selected = {
        name: selected_implementations(name, args.implementations)
        for name in selected_estimators(args.estimators)
    }
    if not any(selected.values()):
        raise SystemExit("no runnable implementations match the selected estimators")
    cap_bytes = choose_memory_cap(args)
    counts: dict[str, int] = {}
    pruned: set[tuple[str, str, int]] = set()
    write_host_metadata(args.output.with_suffix(".host.json"), args, cap_bytes)
    for estimator, implementations in selected.items():
        for implementation in implementations:
            for k in args.k:
                for n in args.n:
                    key = (estimator, implementation, k)
                    if key in pruned:
                        row = {
                            "estimator": estimator,
                            "implementation": implementation,
                            "n": n,
                            "k": k,
                            "status": "pruned_after_failure",
                            "seed": args.seed,
                            "adapter_revision": ADAPTER_REVISION,
                        }
                    else:
                        row = run_cell(estimator, implementation, n, k, args, cap_bytes)
                        print(
                            f"{estimator:28} {implementation:34} n={n:>9} k={k:>3} "
                            f"{row['status']}",
                            flush=True,
                        )
                    write_rows(args.output, [row], args.append or bool(counts))
                    counts[row["status"]] = counts.get(row["status"], 0) + 1
                    if row["status"] in FAILURE_STATUSES:
                        pruned.add(key)
    print(json.dumps({"output": str(args.output), "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
