# Phase 4 Chunk 1: Hotspot Baseline Profile

## Scope
- Establish a reproducible performance baseline before any Phase 4 kernel-level changes.
- Identify highest-return optimization targets after existing GPU/runtime improvements.

## Profiling Runs
- Environment: `conda run -n fus-detectors`, `PYTHONPATH=.`
- Script: `scripts/profile_stap_hotspots.py`
- Commands:
  - `--mode core_pd --device cuda --event-timing --iters 3 --warmup 2`
  - `--mode stap_pd --device cuda --event-timing --iters 2 --warmup 1 --tile-batch 192`
  - `--mode stap_pd --device cuda --event-timing --iters 2 --warmup 1 --tile-batch 512`
  - `--mode stap_pd --device cuda --event-timing --iters 2 --warmup 1 --tile-batch 768`
- Raw outputs (local artifacts):
  - `reports/refactor/phase4_chunk1/profile_core_pd.txt`
  - `reports/refactor/phase4_chunk1/profile_stap_pd_tb192.txt`
  - `reports/refactor/phase4_chunk1/profile_stap_pd_tb512.txt`
  - `reports/refactor/phase4_chunk1/profile_stap_pd_tb768.txt`
  - parsed summary: `reports/refactor/phase4_chunk1/hotspot_summary.json`

## Key Findings
- **Core-only (`core_pd`)**:
  - `stap:TOTAL` avg: **40.10 ms/iter**.
  - `stap:covariance:tyler` avg: **24.18 ms/iter** (~60% of core total).
  - `stap:band_energy` avg: **7.09 ms/iter** (~18% of core total).
- **End-to-end (`stap_pd`) tile batch sensitivity**:
  - `tile_batch=192`: `stap:TOTAL` avg **488.91 ms/iter**.
  - `tile_batch=512`: `stap:TOTAL` avg **255.07 ms/iter** (**~1.92x faster** vs 192).
  - `tile_batch=768`: `stap:TOTAL` avg **266.62 ms/iter** (regresses vs 512).
- **Dominant stage family at E2E (tb512)**:
  - `stap:covariance:tyler` avg **91.99 ms/iter** (largest block).
  - `stap:band_energy` avg **45.94 ms/iter**.
  - `stap:covariance:tyler:update` avg **39.47 ms/iter** (largest Tyler sub-stage).
- **Launch/sync overhead remains material** (`torch.profiler`, tb512):
  - `cudaLaunchKernel` self CPU: **~23%**.
  - `cudaStreamSynchronize` self CPU: **~11%**.
  - `cudaMemcpyAsync` self CPU: **~8%**.
- **Tiling overhead is not primary**:
  - `stap:tiling:*` stages are each sub-ms to low-ms; not first optimization target.

## Improvement Opportunities (Prioritized)
1. **Tyler inner-loop kernels (`update/solve/chol/trace`)**
   - Highest absolute runtime block after prior optimizations.
   - Priority actions:
     - reduce kernel count per iteration (fusion/combined passes),
     - prefer fixed-iteration fast path for captured workloads,
     - avoid redundant synchronization between Tyler sub-stages.
2. **Launch-overhead reduction on fixed shapes**
   - Current traces still show substantial launch/sync tax.
   - Priority actions:
     - CUDA graph capture for fixed-shape core (`B=192/512`, `Lt=8`, fixed grid),
     - keep fallback path for dynamic shapes/debug.
3. **Band-energy projection path**
   - `stap:band_energy:project` + `dual_invgemm` remains non-trivial.
   - Priority actions:
     - fuse projection micro-kernels where safe,
     - reduce intermediate allocations in projection path.
4. **Telemetry runtime gating**
   - `stap:telemetry` remains measurable in hot path.
   - Priority actions:
     - enforce lightweight production telemetry mode for latency-critical runs.

## Chunk 1 Conclusion
- Phase 4 should start with **Tyler + launch-overhead** work; those two areas still dominate post-optimization runtime.
- `tile_batch=512` is currently the best tested operating point for end-to-end `_stap_pd` profiling on this workstation.
