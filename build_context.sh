#!/usr/bin/env bash
set -euo pipefail

# Directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# Destination file for the merged context (default: ./context_bundle.txt)
OUTPUT_FILE="${1:-${REPO_ROOT}/context_bundle.txt}"

# Files to include in the bundle
FILES=(
  "pipeline/stap/temporal.py"
  "sim/kwave/common.py"
  "tests/test_temporal_synthetic_guardrails.py"
  "tests/test_ka_effect_on_scores.py"
  "tests/test_ridge_split_directional.py"
  "tests/test_tile_temporal.py"
)

OUTPUT_ABS="$(cd "${REPO_ROOT}" && realpath -m "${OUTPUT_FILE}")"
mkdir -p "$(dirname "${OUTPUT_ABS}")"
printf '' > "${OUTPUT_ABS}"

for rel_path in "${FILES[@]}"; do
  src="${REPO_ROOT}/${rel_path}"
  if [[ ! -f "${src}" ]]; then
    echo "Warning: ${rel_path} not found, skipping." >&2
    continue
  fi
  {
    printf '===== FILE: %s =====\n' "${rel_path}"
    cat "${src}"
    printf '\n\n'
  } >> "${OUTPUT_ABS}"
done

echo "Context bundle written to: ${OUTPUT_ABS}"
printf 'Included files:\n'
printf '  %s\n' "${FILES[@]}"
