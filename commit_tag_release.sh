#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "Usage: $0" >&2
  exit 1
fi

version="$(
  awk -F '"' '
    /^\[package\]/ { in_package = 1; next }
    /^\[/ && $0 !~ /^\[package\]/ { in_package = 0 }
    in_package && /^version = / { print $2; exit }
  ' Cargo.toml
)"

if [[ -z "${version}" ]]; then
  echo "Failed to read package version from Cargo.toml" >&2
  exit 1
fi

tag="v${version}"

git tag -a "${tag}" -m "${tag}"

git push origin "${tag}"

echo "Tagged and pushed ${tag}"
