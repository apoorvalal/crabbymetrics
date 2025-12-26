#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 x.y.z" >&2
  exit 1
fi

version="$1"
tag="v${version}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Error: not inside a git repository" >&2
  exit 1
fi

git add .

git commit -m "release ${tag}"

git push

git tag -a "${tag}" -m "${tag}"

git push origin "${tag}"

echo "Tagged and pushed ${tag}"
