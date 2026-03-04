# Latency Optimization Engineering Spec (STAP + Baselines)

This doc defines a **phased, regression-safe** plan to reduce end-to-end latency of the deployed pipeline in this repo, while preventing accuracy regressions against the manuscript baselines.

Key constraints:
- **No-regression**: Paper figures/tables must reproduce (same inputs → same outputs), unless a change is explicitly declared as an algorithm change.
- **Fair latency**: If we make latency claims or show timing tables, apply the same class of “obvious” engineering optimizations to baselines (same device/dtype, remove Python loops, no CPU↔GPU ping-pong).
- **Deployment focus**: The primary “deployable” latency target in Brain-* is **MC–SVD + STAP** (STAP is evaluated on top of MC–SVD residual in the paper).

---

## 0) Baseline Contract (Paper Results)

### Frozen operating profiles (must not change)
Source: `stap_fus_methodology.tex` Table `tab:fixed_profiles`.

- **Brain-* (k-Wave; PRF 1500; 64-frame windows)**:
  - Baselines: MC–SVD energy fraction `e=0.90`; adaptive/global SVD similarity-cutoff; block-wise local SVD; RPCA; HOSVD (baselines scored by PD on residual)
  - STAP: tiles `8×8`, stride `3`, `Lt=8`, robust covariance (Tyler-type) + diagonal loading `λ=0.07`
  - Bands (Hz): `Pf=[30,250]`, guard `[250,400]`, `Pa=[400,750]`

- **Gammex flow phantom (linear17; PRF 2500; N=17 shots; pooled over cine frames)**:
  - Baseline: SVD band-pass keep `[2,17]` + conventional comparators (PD/log-PD/Kasai)
  - STAP: tiles `8×8`, `Lt=16`, Tyler-type covariance + `λ=0.07`, stride `6` (along) / `4` (across)
  - Bands (Hz): `Pf=[150,450]`, guard `[450,700]`, `Pa=[700,1200]`

- **ULM Zenodo 7883227 (PRF 1000; 128-frame blocks)**:
  - Baseline: MC–SVD energy fraction `e=0.975`
  - STAP: tiles `8×8`, stride `3`, `Lt=64`, SCM covariance + `λ=0.07`
  - Bands (Hz): `Pf=[10,150]`, guard `[150,200]`, `Pa=[200,500]`

### Acceptance numbers (paper anchors)
Source: `stap_fus_methodology.tex`.

- Brain-* STAP TPR@`1e-3` medians (Table `tab:brain_kwave_vnext_baselines`; matched-subspace pre-KA on MC--SVD residual):
  - Brain-OpenSkull: `0.7751`
  - Brain-AliasContract: `0.1619`
  - Brain-SkullOR: `0.8055`

- Gammex STAP TPR@`1e-3` (Table `tab:twinkling_gammex_structural_roc`):
  - Along: `0.946 [0.943,0.948]`
  - Across: `0.938 [0.936,0.940]`

### Repro commands (canonical)
Source: `appendix_repro_manifest.tex` (included by `stap_fus_methodology.tex`).

Refresh via `PYTHONPATH=. conda run -n stap-fus python scripts/generate_repro_manifest.py` when needed.

**Brain-* ROC curve figure (reads precomputed runs):**
```bash
PYTHONPATH=. python scripts/fig_brain_kwave_roc_curves.py \
  --runs-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \
  --out-pdf figs/paper/brain_kwave_roc_curves.pdf
```

**Brain-* low-FPR baseline matrix report (writes the baseline table inputs):**
```bash
PYTHONPATH=. conda run -n stap-fus python scripts/fair_filter_comparison.py \
  --mode matrix --eval-score vnext \
  --matrix-regimes open,aliascontract,skullor \
  --matrix-seeds-open 1 --matrix-seeds-aliascontract 2 --matrix-seeds-skullor 2 \
  --window-length 64 --window-offsets 0,64,128,192,256 \
  --matrix-use-profile \
  --matrix-mcsvd-energy-frac 0.90 --matrix-mcsvd-baseline-support window \
  --methods mcsvd,svd_similarity,local_svd,rpca,hosvd,stap_full \
  --generated-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \
  --autogen-missing --stap-device cuda \
  --out-csv reports/fair_matrix_vnext_r3_localbaselines.csv \
  --out-json reports/fair_matrix_vnext_r3_localbaselines.json
```

**Brain-* cross-window threshold-transfer audit:**
```bash
PYTHONPATH=. python scripts/brain_crosswindow_calibration.py \
  --runs-root runs/pilot/fair_filter_matrix_pd_r3_localbaselines \
  --alphas 1e-4,3e-4,1e-3 \
  --out-csv reports/brain_crosswindow_calibration.csv \
  --out-json reports/brain_crosswindow_calibration_summary.json
```

**Gammex structural ROC (along view; analogous across-view command in manifest):**
```bash
SEQ_DIR="data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (along - linear probe)"
OUT_ROOT="runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka"

PYTHONPATH=. python scripts/twinkling_make_bundles.py \
  --seq-dir "$SEQ_DIR" \
  --out-root "$OUT_ROOT" \
  --frames 0:85 \
  --prf-hz 2500 --Lt 16 \
  --tile-hw 8 8 --tile-stride 6 \
  --cov-estimator tyler_pca --diag-load 0.07 \
  --baseline-type svd_bandpass --svd-keep-min 2 --svd-keep-max 17 \
  --score-mode msd --score-ka-v2-enable --score-ka-v2-mode auto

PYTHONPATH=. python scripts/twinkling_eval_structural.py \
  --root "$OUT_ROOT" \
  --out-csv reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines.csv \
  --out-summary-json reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ka_with_baselines_summary.json
```

---

## 1) Latency Reporting Policy (Reviewer-proof)

If we produce latency tables/claims:
- Same **device** (CPU-only table or GPU-only table; do not mix).
- Same **dtype** (complex64/float32).
- Same **batching strategy** class (no Python-per-tile loops for one method while another is vectorized).
- No **CPU↔GPU ping-pong** mid-pipeline.
- Warmup + timed runs; GPU timing via CUDA events or explicit synchronization at boundaries.

### “Real-time factor” (RTF)
Define for a slow-time ensemble length `T` acquired at PRF:
```
t_data = T / PRF
RTF = t_proc / t_data
```
Targets (recommended):
- real-time: `RTF ≤ 1`
- near real-time: `1 < RTF ≤ 2`

Example (Brain-*): `T=64`, `PRF=1500 Hz` → `t_data≈42.7 ms`.

---

## 1.5) What Makes GPU STAP Fast Here (Shipped Implementation)

This section is the “how”: the concrete engineering strategy that takes STAP from “thousands of tiny per-tile ops” to a small number of high-throughput GPU kernels with low control-plane overhead.

### A) Vectorize tiling (remove Python/NumPy raster loops)

Problem: STAP is applied per tile (e.g., `8×8` tiles over a ~`240×240` plane), so a naive implementation spends most time in:
- Python loops over tiles
- `np.stack(...)` per batch
- CPU→GPU copies per tile/batch

Shipped fix (opt-in):
- `sim/kwave/common.py:_stap_pd` has an **unfold-tiling** path gated by `STAP_TILING_UNFOLD=1`.
- Tiles are extracted on-device via `torch.Tensor.unfold(...)` views, processed in chunks, and overlap-added back into full maps without returning to NumPy on the hot path.

Key outputs for sanity:
- `meta.json` telemetry fields: `stap_unfold_tiling_enabled/eligible/used/error`.

### B) Batch the temporal core (turn “many small tiles” into “one batched solve”)

Shipped core:
- `pipeline/stap/temporal.py:stap_temporal_core_batched(...)` (matched-subspace / MSD score mode)
- `pipeline/stap/temporal.py:pd_temporal_core_batched(...)` (PD-oriented “band-fraction only” mode)

Used by:
- `sim/kwave/common.py:_stap_pd_tile_lcmv_batch(...)` (batched wrapper; forced-on for CUDA when KA/debug are off)

Microbatching knob:
- `STAP_TILE_BATCH` (or `--tile-batch` in the latency scripts) controls the chunk size `B` processed per batched core call.

### C) Reduce “wide TRSM” overhead with `inv_gemm` strategies (Tyler + band-energy)

Hotspot: batched triangular solves (`solve_triangular`) with a **very wide RHS** (snapshot count `P=N*h*w`) can be slower than GEMM on CUDA.

Shipped strategy:
- In Tyler whitening, `pipeline/stap/temporal.py:_tyler_covariance_batched(...)` chooses between:
  - `direct`: TRSM on the full RHS, or
  - `inv_gemm`: compute `L^{-1}` once (small RHS) then use `bmm` (GEMM) on the wide RHS.
- In band-energy whitening/projection, `pipeline/stap/temporal.py:_band_energy_whitened_batched(...)` applies the same idea and additionally prefers “small solve + GEMM” forms for Gram/projection work.

Control knobs:
- `STAP_TYLER_SOLVE_MODE=auto|direct|inv_gemm`
- `STAP_BAND_SOLVE_MODE=auto|direct|inv_gemm`
- `STAP_BAND_PROJECT_MODE` / `STAP_BAND_DUAL_QUAD_MODE` (projection algebra choices; defaults aim to preserve baseline behavior)

### D) Cut kernel-launch overhead with optional Triton fusion (Tyler iteration)

Shipped optional kernels:
- `pipeline/stap/triton_ops.py` provides fused kernels for Tyler weights and covariance updates.
- These are enabled on CUDA when Triton is available and the toggles allow them.

Key toggles (all optional):
- `STAP_TYLER_TRITON_WEIGHTS`
- `STAP_TYLER_TRITON_UPDATE`
- `STAP_TYLER_TRITON_WEIGHTS_UPDATE`
- `STAP_TYLER_TRITON_SOLVE_UPDATE`

### E) CUDA graphs for fixed-shape steady-state replay (cut Python+launch overhead further)

Shipped mechanism:
- `sim/kwave/common.py:_stap_fast_cuda_graph_run(...)` captures the **fixed-shape batched core** into a `torch.cuda.CUDAGraph` and replays it.
- Cache key includes shapes + key env knobs so captures are safe and repeatable.

Enable with:
- `STAP_FAST_CUDA_GRAPH=1`

Important capture constraints (by design):
- During capture, **Tyler early-stop is disabled** because convergence checks require host sync; capture uses a fixed iteration count:
  - `pipeline/stap/temporal.py:_tyler_covariance_batched(...)` forces `early_stop=False` when capturing.
  - Latency experiments can additionally set `STAP_TYLER_MAX_ITER` and `STAP_TYLER_EARLY_STOP=0`.
- Optional Triton paths are **disabled by default during capture** for robustness; opt-in with:
  - `STAP_TYLER_TRITON_CAPTURE=1`

### F) Report “steady-state” latency (exclude capture/JIT/cold window)

Why: GPU runs include one-time overheads (CUDA init, Triton JIT, CUDA-graph capture). We publish steady-state means that exclude the first window/frame.

Shipped harnesses:
- Brain-* pilot replay: `scripts/latency_rerun_check.py` prints `cold(win1)` and `steady(avg win2..N)`.
- Real-data Shin/Gammex replay: `scripts/latency_realdata_rerun_check.py` prints the same and emits a JSON summary.

Publication rule of thumb:
- Use **steady-state mean windows/frames 2..N** for tables/claims.
- Treat window/frame 1 as “cold/capture” overhead, report separately only if needed.

---

## 2) Current STAP Implementation (Core Path + Shapes)

This is the “core” to optimize: **tile extraction → covariance → whitening → scoring → stitch-back**.

### Call chain (Brain-* / k-Wave path)
- Tiling + overlap-add lives in `sim/kwave/common.py`:
  - `_stap_pd(...)` is the tile raster + batcher + stitcher.
  - `_stap_pd_tile_lcmv_batch(...)` is a batch wrapper with an optional fast path.
  - `_stap_pd_tile_lcmv(...)` is the slow per-tile kernel (debug/KA/gating compatible).

- Batched temporal STAP core lives in `pipeline/stap/temporal.py`:
  - `stap_temporal_core_batched(...)` (full STAP score)
  - `pd_temporal_core_batched(...)` (PD-oriented “fast PD-only” variant)

- Hankel + pooled SCM covariance helper: `pipeline/stap/temporal_shared.py`
  - `build_temporal_hankels_batch(...)`

### Shapes in the batched core
Let input batch be `cube_batch_T_hw`:
- Input: `(B, T, h, w)` complex tensor
- Hankelize (time unfold, step=1), centered:
  - `S`: `(B, Lt, N, h, w)` where `N = T - Lt + 1`
- Flatten snapshots+space for covariance:
  - `S_flat`: `(B, Lt, K)` where `K = N*h*w`
- Covariance:
  - `R_hat`: `(B, Lt, Lt)`
- Whitening + projection (bandpass constraint matrix `Ct`, subspace basis `C_exp`):
  - energy tensors often shaped like `(B, N, h, w)` before snapshot aggregation
- Final per-tile maps returned to tiler:
  - `band_frac`: `(B, h, w)` float32
  - `score`: `(B, h, w)` float32

### Known copy/sync hazards (baseline for optimization)
These are the current high-impact latency issues we’ll address in early sprints:

**CPU↔GPU ping-pong (catastrophic):**
- In `_stap_pd_tile_lcmv_batch` when input is already a CUDA tensor:
  - it converts to NumPy via `.detach().cpu().numpy()` and then back to CUDA via `torch.as_tensor(..., device=cuda)`.

**Per-tile CPU extraction/stacking (Sprint 2 target):**
- `_stap_pd` uses per-tile `np.ascontiguousarray(...)` and then `np.stack(...)` before moving to GPU.

**Python loops inside batched scoring (Sprint 3 target):**
- `stap_temporal_core_batched` aggregates over snapshots with `for b in range(B)` and `aggregate_over_snapshots(...)`.

---

## 3) Phased Plan (Each Phase Has Close Conditions)

### Phase 0 — Baseline lock + benchmark harness
Scope:
- Freeze “contract” configs (above) and establish a repeatable regression+latency workflow.
- Decide and document the latency reporting policy (RTF, timing methodology, warmup, device/dtype rules).

Close conditions:
1) Paper reproduction checks (no code changes required):
   - Brain ROC figure: `scripts/fig_brain_kwave_roc_curves.py` (manifest)
   - Brain cross-window audit: `scripts/brain_crosswindow_calibration.py` (manifest)
   - Gammex structural ROC: `scripts/twinkling_make_bundles.py` + `scripts/twinkling_eval_structural.py` (manifest)
2) Latency baseline captured in a checked-in note (not code):
   - Record machine, GPU model, torch version, and the exact commands used for timing.
   - Capture **end-to-end** times for “MC–SVD” and “MC–SVD + STAP” on the same device.

---

### Sprint 1 — GPU-pure STAP execution (remove ping-pong)
Scope:
- Eliminate unintended `.cpu().numpy()` round-trips in the **fast/batched STAP** path.
- Ensure only one dtype/device normalization is applied (complex64, chosen device).

Primary touchpoints:
- `sim/kwave/common.py:_stap_pd_tile_lcmv_batch`

Close conditions:
1) Paper regression checks:
   - Run the Phase-0 reproduction commands; outputs unchanged.
2) Latency improvement check (GPU):
   - Re-run a fixed replay (same pilot window) with `--stap-device cuda` and `STAP_FAST_PATH=1`.
   - Confirm `stap_fallback_telemetry` no longer includes a GPU→CPU→GPU copy on the hot path (via code inspection / profiler).
   - Record updated end-to-end time per 64-frame window (baseline vs baseline+STAP).

---

### Sprint 2 — Torch tiling via unfold/fold (remove raster loops + np.stack)
Scope:
- Replace per-tile CPU slicing + `np.stack(...)` with GPU tiling:
  - extract tiles via `unfold` views
  - stitch via `torch.nn.functional.fold` + counts normalization

Primary touchpoints:
- `sim/kwave/common.py:_stap_pd` (tile extraction + overlap-add)

Close conditions:
1) Numerical equivalence:
   - For a fixed seed and a representative cube, match current `score_stap_preka.npy` within tolerance.
2) Paper regression checks:
   - Phase-0 reproduction commands unchanged.
3) Latency improvement:
   - Reduced wall time and reduced Python time in profiler; updated timing record.

---

### Sprint 3 — Remove Python loops in batched core (vectorize)
Scope:
- Replace Python loops in snapshot aggregation and telemetry collection with tensor ops.
- Prefer batched linear algebra (GEMM, batched solves) over per-tile loops.

Primary touchpoints:
- `pipeline/stap/temporal.py:stap_temporal_core_batched`
- `pipeline/stap/temporal_shared.py:robust_temporal_cov_batch` (if/when robust covariance is retained)

Close conditions:
1) Paper regression checks unchanged.
2) Latency improvement:
   - Lower kernel launch overhead; improved throughput for large tile batches.

---

### Sprint 4 — Optimize MC–SVD baseline on the same device
Scope:
- Implement an optimized, GPU-first MC–SVD baseline (since it is part of the deployed pipeline in Brain-*).
- Keep accuracy identical to the frozen baseline profile (`e=0.90` for Brain-*; `e=0.975` for ULM 7883227).

Implementation toggles (regression-safe; default off):
- `MC_SVD_TORCH=1`: run MC–SVD baseline using torch ops (CUDA when `device=cuda`) for the projector + PD computation.
- `MC_SVD_REG_TORCH=1`: additionally run the phase-correlation registration stage in torch/CUDA (optional; may be slower on some GPUs).
- `MC_SVD_TORCH_RETURN_CUBE=1`: keep the MC–SVD residual cube as a torch tensor on-device when `return_filtered_cube=True`
  (avoids GPU→CPU→GPU copies before STAP).

Quick latency+parity check (pilot harness):
```bash
PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \
  --src runs/latency_pilot_open \
  --out-root runs/latency_s4_check \
  --profile Brain-OpenSkull \
  --window-length 64 --window-offset 0 \
  --stap-device cuda --stap-debug-samples 0
```

Steady-state GPU latency (includes torch/CUDA registration profiling; no parity check for that extra run):
```bash
PYTHONPATH=. conda run -n stap-fus python scripts/latency_rerun_check.py \
  --src runs/latency_pilot_open \
  --out-root runs/latency_s4_check \
  --profile Brain-OpenSkull \
  --window-length 64 --window-offset 0 \
  --stap-device cuda --stap-debug-samples 0 \
  --cuda-warmup-heavy \
  --profile-baseline-reg-torch
```

Close conditions:
1) Paper regression checks unchanged.
2) End-to-end deployable latency:
   - Report **MC–SVD + STAP** RTF for Brain-* (64 @ PRF 1500).

---

### Sprint 5 — Paper-ready latency reporting (fair baselines)
Scope:
- Produce a GPU latency table and breakdown mirroring `tab:brain_latency_results` (CPU-only in the paper).
- If comparing to RPCA/HOSVD timings: either optimize them best-effort on the same device or clearly label “offline baseline”.

Close conditions:
1) Paper regression checks unchanged.
2) Updated latency table(s) + RTF definition included in manuscript assets.
