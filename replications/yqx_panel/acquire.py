"""Fetch checksum-pinned public inputs; do not vendor third-party datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
LECTURE_COMMIT = "891412f174fc0283677de84d612622ff602dd1d3"
SOURCES = {
    "fatalities.csv": ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/Fatalities.csv", "ae265c1b14b4a023179b3af5db33b6a659ccf804987f26e8349826d4083ee149"),
    "divorce.rda": ("https://raw.githubusercontent.com/evanjflack/bacondecomp/master/data/divorce.rda", "59ab3530a8229905c9f737c6b9bfbf1d78f56afc6c1a8123d74a39ac490f98b1"),
    "smoking.rda": ("https://raw.githubusercontent.com/edunford/tidysynth/master/data/smoking.rda", "a946d153e8faee62e6deba7a16db6f4c1251d9e56d5b41de7e79a593eb8110d9"),
    "fdid.RData": ("https://raw.githubusercontent.com/xuyiqing/fdid/main/data/fdid.RData", "ae07e2d1ddeedb96e9aca1914800805bfdb18970e91da549a59f8d3f256db0b3"),
    "gs2020.rda": ("https://raw.githubusercontent.com/xuyiqing/fect/master/data/gs2020.rda", "fac564dcc755a6d9d9ebd26f4114c8da0cd58f14e0be0160b21f20e29162358f"),
    "turnout.rda": ("https://raw.githubusercontent.com/xuyiqing/fect/master/data/turnout.rda", "f4a73077a45d0e1b81994a0b39a1770b9199930dba375ebe64e99a8ecbd48790"),
    "hh2019.rda": ("https://raw.githubusercontent.com/xuyiqing/fect/master/data/hh2019.rda", "796e958d24f0584b90d62e160bb54ddf168ee032e34e98f76f2ddd2b2240137b"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lectures", required=True, type=Path)
    args = p.parse_args()
    import pandas as pd

    commit = subprocess.check_output(["git", "-C", str(args.lectures), "rev-parse", "HEAD"], text=True).strip()
    if commit != LECTURE_COMMIT:
        raise ValueError(f"Expected lectures {LECTURE_COMMIT}, found {commit}; review changes before updating the pin")
    CACHE.mkdir(exist_ok=True)
    provenance = []
    for name, (url, sha) in SOURCES.items():
        dst = CACHE / name
        if not dst.exists():
            dst.write_bytes(urllib.request.urlopen(url, timeout=45).read())
        actual = hashlib.sha256(dst.read_bytes()).hexdigest()
        if actual != sha:
            raise ValueError(f"Checksum mismatch for {name}: {actual}")
        provenance.append(dict(file=name, url=url, sha256=sha, bytes=dst.stat().st_size))
    for name, rel in {
        "swiss": "01-parametric/figs/Swiss_Panel_long.dta",
        "swiss_full": "01-parametric/figs/swissnat.dta",
        "ck": "02-did/figs/CK1994_longformat.dta",
    }.items():
        src = args.lectures / rel
        # These teaching files store literal year values with Stata date formats.
        # Keep numeric years, as haven/foreign do in the lecture R scripts.
        frame = pd.read_stata(src, convert_categoricals=False, convert_dates=False)
        for column in frame.select_dtypes(include="floating"):
            frame[column] = frame[column].astype(float)
        frame.to_csv(CACHE / f"{name}.csv", index=False, float_format="%.17g")
        provenance.append(dict(file=f"{name}.csv", source=rel, lecture_commit=commit, sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
    subprocess.run(["Rscript", str(HERE / "export_data.R"), str(CACHE)], check=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "data-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified {len(provenance)} data sources; private local cache: {CACHE}")


if __name__ == "__main__":
    main()
