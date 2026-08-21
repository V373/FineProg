#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

exec conda run --no-capture-output -n fineprog python \
  "${PROJECT_ROOT}/tests/sweep_fruit_gaussian_num_bins.py" "$@"
