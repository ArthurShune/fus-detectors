#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper paper/preprint.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper paper/methods_companion.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=paper paper/supplement.tex
