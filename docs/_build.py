"""Execute a documentation release in an isolated, resumable staging tree."""

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import crabbymetrics
import yaml


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--pages", nargs="*", help="Docs-relative pages; default: whole site")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    stage = args.stage.resolve()
    if stage == repo or repo in stage.parents:
        parser.error("--stage must be outside the source checkout")
    if stage.exists() and any(stage.iterdir()) and not (stage / "execution.json").is_file():
        parser.error("--stage must be empty or contain this builder's execution.json")
    stage.mkdir(parents=True, exist_ok=True)
    source = stage / "source"
    docs = source / "docs"
    log_dir = stage / "logs"
    log_dir.mkdir(exist_ok=True)
    ignored = shutil.ignore_patterns(
        "_site", "_freeze", ".quarto", ".jupyter_cache", "__pycache__", "*.pyc",
        "*.html", "*_files", "*.quarto_ipynb", "site_libs", "search.json",
        "build-info.json",
    )
    shutil.copytree(repo / "docs", docs, dirs_exist_ok=True, ignore=ignored)
    shutil.copytree(repo / "benchmarks", source / "benchmarks", dirs_exist_ok=True, ignore=ignored)
    config = yaml.safe_load((repo / "docs/_quarto.yml").read_text())
    site = docs / config["project"]["output-dir"]
    pages = sorted({
        path.relative_to(repo / "docs").as_posix()
        for pattern in config["project"]["render"]
        for path in (repo / "docs").glob(pattern)
    })
    if args.pages and set(args.pages) - set(pages):
        parser.error("every requested page must belong to the Quarto render inventory")
    selected = args.pages or pages
    manifest_path = stage / "execution.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"pages": {}}
    extension = next(
        path for path in Path(crabbymetrics.__file__).parent.iterdir()
        if any(path.name.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)
    )
    environment = {
        "python": sys.version.split()[0],
        "quarto": subprocess.check_output(["quarto", "--version"], text=True).strip(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("crabbymetrics", "numpy", "pandas", "matplotlib", "pyfixest",
                      "scipy", "polars", "IPython", "jupyter-cache", "PyYAML")},
        "extension_sha256": sha256(extension),
    }
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    source_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    shared = [repo / "docs/_quarto.yml", repo / "docs/styles.css",
              *sorted((repo / "docs/reference").glob("*.py"))]
    shared += sorted((repo / "docs/data").rglob("*"))
    shared += sorted((repo / "docs/ablations/data").glob("*"))
    shared += [repo / "benchmarks/scaling" / name
               for name in ("registry.py", "report_metadata.py")]
    shared_hash = hashlib.sha256(json.dumps(environment, sort_keys=True).encode())
    for path in shared:
        if path.is_file():
            shared_hash.update(path.relative_to(repo).as_posix().encode())
            shared_hash.update(path.read_bytes())
    env = os.environ.copy()
    env.update(QUARTO_PYTHON=sys.executable, MPLBACKEND="Agg")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    failures = []
    for number, page in enumerate(selected, 1):
        key = hashlib.sha256(
            (sha256(docs / page) + shared_hash.hexdigest()).encode()
        ).hexdigest()
        previous = manifest["pages"].get(page, {})
        html = (site / page).with_suffix(".html")
        if (previous.get("status") == "ok" and previous.get("input_sha256") == key
                and html.exists() and previous.get("html_sha256") == sha256(html)):
            log = log_dir / (page.replace("/", "__") + ".log")
            if "executed_cells" not in previous and log.exists():
                previous["executed_cells"] = len(re.findall(r"Cell \d+/\d+:.*Done", log.read_text()))
            print(f"[{number}/{len(selected)}] verified existing execution: {page}", flush=True)
            continue
        log = log_dir / (page.replace("/", "__") + ".log")
        command = [
            "quarto", "render", str(docs / page), "--execute", "--cache-refresh",
            "-M", "freeze:false", "--execute-daemon", "0", "--no-clean",
        ]
        print(f"[{number}/{len(selected)}] executing {page}", flush=True)
        start = time.monotonic()
        with log.open("w") as output:
            try:
                result = subprocess.run(command, env=env, stdout=output,
                                        stderr=subprocess.STDOUT, timeout=3600, check=False)
                ok = result.returncode == 0 and html.exists()
            except subprocess.TimeoutExpired:
                ok = False
        record = {
            "status": "ok" if ok else "failed",
            "input_sha256": key,
            "executed_at_utc": datetime.now(timezone.utc).isoformat(),
            "seconds": round(time.monotonic() - start, 2),
            "executed_cells": len(re.findall(r"Cell \d+/\d+:.*Done", log.read_text())),
        }
        if ok:
            record["html_sha256"] = sha256(html)
        else:
            failures.append(page)
            print(log.read_text()[-4000:], flush=True)
        manifest["pages"][page] = record
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest.update(environment=environment, source_revision=source_revision,
                    source_dirty=source_dirty, execution_mode="fresh",
                    render_flags=["--execute", "--cache-refresh", "-M", "freeze:false"])
    complete = True
    for page in pages:
        key = hashlib.sha256(
            (sha256(docs / page) + shared_hash.hexdigest()).encode()
        ).hexdigest()
        result = manifest["pages"].get(page, {})
        html = (site / page).with_suffix(".html")
        complete &= (result.get("status") == "ok" and result.get("input_sha256") == key
                     and html.is_file() and result.get("html_sha256") == sha256(html))
    manifest["complete"] = complete and not failures
    manifest["resources"] = {
        name: sha256(docs / name) for name in ("styles.css", "llms.txt")
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["complete"]:
        (site / "build-info.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for page in pages:
            target = site / page
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(docs / page, target)
        shutil.copy2(docs / "reference/_api_doc_utils.py", site / "reference/_api_doc_utils.py")
        shutil.copy2(docs / "llms.txt", site / "llms.txt")
        shutil.copy2(docs / "styles.css", site / "styles.css")
    print(json.dumps({"complete": manifest["complete"], "failures": failures,
                      "site": str(site)}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
