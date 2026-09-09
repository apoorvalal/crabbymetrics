"""Bounded subprocess execution shared by Python and R benchmark cells."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import BinaryIO

import psutil


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    status: str | None
    wall_seconds: float
    peak_rss_bytes: int
    stdout: str
    stderr: str


def descendants_rss(process: psutil.Process) -> int:
    try:
        children = process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        children = []
    total = 0
    for proc in [process, *children]:
        try:
            total += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def terminate_tree(popen: subprocess.Popen, process: psutil.Process | None) -> None:
    # Each cell owns a new session. The group remains addressable even when the
    # direct child exits before its workers, unlike a parent-based psutil walk.
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(popen.pid, sig)
            except ProcessLookupError:
                break
            if sig == signal.SIGTERM:
                try:
                    popen.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
    else:
        try:
            processes = [process, *process.children(recursive=True)] if process else []
        except psutil.NoSuchProcess:
            processes = []
        for proc in reversed(processes):
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                pass
        if popen.poll() is None:
            popen.kill()
    popen.wait()


def read_tail(handle: BinaryIO, limit: int) -> str:
    handle.seek(0, os.SEEK_END)
    handle.seek(max(0, handle.tell() - limit))
    return handle.read(limit).decode("utf-8", errors="replace")


def run_process(
    command: list[str], env: dict[str, str], timeout: float, cap_bytes: int
) -> ProcessResult:
    started = time.perf_counter()
    # Pipes can fill while the parent monitors RSS. Spool output to files and
    # read bounded tails so verbose solvers cannot deadlock or exhaust RAM.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        popen = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, env=env, start_new_session=True
        )
        process = None
        peak_rss = 0
        status = None
        try:
            try:
                process = psutil.Process(popen.pid)
            except psutil.NoSuchProcess:
                pass
            while popen.poll() is None:
                if process is not None:
                    peak_rss = max(peak_rss, descendants_rss(process))
                if peak_rss > cap_bytes:
                    status = "killed_rss_guard"
                    break
                if time.perf_counter() - started > timeout:
                    status = "timeout"
                    break
                time.sleep(0.05)
        finally:
            terminate_tree(popen, process)
        return ProcessResult(
            returncode=popen.returncode,
            status=status,
            wall_seconds=time.perf_counter() - started,
            peak_rss_bytes=peak_rss,
            stdout=read_tail(stdout, 1024 * 1024),
            stderr=read_tail(stderr, 2000),
        )
