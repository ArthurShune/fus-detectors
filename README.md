# STAP-for-fUS

Knowledge-aided space-time adaptive processing (STAP) for functional ultrasound (fUS), with reproducible simulation/phantom evaluation and latency profiling workflows.

This repository contains:
- STAP core implementations and baselines (MC-SVD, RPCA, HOSVD, adaptive/local SVD variants).
- Reproduction scripts for manuscript tables/figures.
- Refactor verification gates (`quick`, `phase`, `full`) used to control regressions.

## Quick Start

```bash
conda env create -f environment.yml
conda activate stap-fus
pre-commit install
python scripts/verify_gpu.py
```

## Reproducibility Entry Points

- **Smoke gate (data-safe):**
  - `make refactor-quick-ci`
- **Local quick regression gate:**
  - `make refactor-quick`
- **Phase boundary gate:**
  - `make refactor-phase`
- **Release boundary gate:**
  - `make refactor-full`
- **Example manuscript reproduction command:**
  - `bash scripts/reproduce_table5_brain_kwave.sh`

## Repository Map

- `pipeline/` — STAP and covariance/detector core code.
- `sim/` — simulation and STAP integration runtime.
- `scripts/` — experiment/reproduction/orchestration scripts.
- `tests/` — focused unit/regression tests for core paths.
- `docs/refactor/` — phased refactor plans, checklists, and phase reports.
- `configs/` — reproducibility and sweep configurations.

## Public-Repo Policies

- Contribution guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`

## Notes

- `k-Wave-python` downloads required binaries on first use.
- `environment.yml` targets CUDA-capable PyTorch; ensure NVIDIA driver compatibility.
- Large datasets and run artifacts are intentionally ignored by git (`runs/`, `reports/`, `data/`).
