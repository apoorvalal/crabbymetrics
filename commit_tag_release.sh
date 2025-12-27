#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 x.y.z" >&2
  exit 1
fi

version="$1"
tag="v${version}"

git tag -a "${tag}" -m "${tag}"

git push origin "${tag}"

echo "Tagged and pushed ${tag}"
