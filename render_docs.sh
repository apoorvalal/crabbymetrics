#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docs_dir="${1:-docs}"

cd "$repo_root"

if [[ ! -d "$docs_dir" ]]; then
    echo "Docs directory not found: $docs_dir" >&2
    exit 1
fi

if ! command -v quarto >/dev/null 2>&1; then
    echo "quarto is not installed or not on PATH" >&2
    exit 1
fi

if [[ -x "$repo_root/.venv/bin/python" ]]; then
    export QUARTO_PYTHON="$repo_root/.venv/bin/python"
fi

if [[ -f "$docs_dir/_quarto.yml" ]]; then
    echo "Rendering Quarto project in $docs_dir"
    quarto render "$docs_dir"
    exit 0
fi

mapfile -t qmd_files < <(find "$docs_dir" -type f -name '*.qmd' | sort)

if (( ${#qmd_files[@]} == 0 )); then
    echo "No Quarto documents found under $docs_dir"
    exit 0
fi

for qmd_file in "${qmd_files[@]}"; do
    echo "Rendering $qmd_file"
    quarto render "$qmd_file"
done
