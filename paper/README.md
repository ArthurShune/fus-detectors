# Paper Files

This directory contains the paper-facing artifacts for `fus-detectors`.

## Read First

- `preprint.pdf`: main preprint
- `methods_companion.pdf`: extended methods and companion analyses
- `supplement.pdf`: supplement-only build

## Source Files

- `preprint.tex`: wrapper that builds the main preprint
- `methods_companion.tex`: wrapper that builds the companion PDF
- `manuscript.tex`: full manuscript source shared by the preprint and companion builds
- `supplement.tex`: wrapper that builds the supplement-only PDF

## Canonical Build

Use the repo-root helper so outputs land in `paper/` instead of leaving duplicate PDFs in the
repo root:

- `./scripts/build_paper_artifacts.sh`
