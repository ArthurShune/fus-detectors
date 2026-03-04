# STAP-for-fUS (skeleton)

CUDA-enabled, MATLAB-free scaffold for the public demo.
- **Propagation/heterogeneity**: via `k-Wave-python` (wraps k-Wave C++/CUDA binaries).
- **GPU math**: PyTorch + CuPy.
- **Pipeline**: SVD baseline and STAP tiles; EVT + conformal; Confirm-2.

## Quickstart
```bash
conda env create -f environment.yml
conda activate stap-fus
pre-commit install
python scripts/verify_gpu.py  # checks Torch/CuPy/k-Wave imports
```

## Reproduce (one-command example)
```bash
bash scripts/reproduce_table5_brain_kwave.sh
```

## Refactor Verification Gates
```bash
# quick
make refactor-quick

# phase boundary
make refactor-phase

# release/milestone
make refactor-full
```
CI uses a data-safe quick gate (`make refactor-quick-ci`) in `.github/workflows/refactor_quick_gate.yml`.

### Notes

* `k-Wave-python` auto-downloads the needed CPU/GPU binaries on first use.
* The env installs `pytorch-cuda=12.1`; ensure a compatible NVIDIA driver.
* If no GPU is available, the pipeline will still run on CPU for dev.
