#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${1:-$PROJECT_ROOT/svs_process/preprocess_config.yaml}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" "$PROJECT_ROOT/svs_process/preprocess_wsi.py" --config "$CONFIG_PATH"
