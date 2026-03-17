#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[repro] Regenerating Table 7 (same-residual SIMUS detector-family table)"
conda run -n fus-detectors python scripts/simus_detector_family_ablation_table.py

echo "[repro] Regenerating Figure 8 (headline same-residual SIMUS figure)"
conda run -n fus-detectors python scripts/fig_simus_detector_family_headline.py

echo "[repro] Done"
echo "  - reports/simus_detector_family_ablation_table.tex"
echo "  - figs/paper/simus_detector_family_headline.pdf"
