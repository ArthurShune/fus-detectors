# Phase 4 Chunk 2: Launch-Overhead Path (CUDA-Graph Gating)

## Scope
- Implement safer fixed-shape CUDA-graph gating for fast STAP batches.
- Profile graph vs non-graph runtime on the current optimized stack.

## Code Changes
- `sim/kwave/common.py`
  - Added CUDA-graph mode parsing:
    - `STAP_FAST_CUDA_GRAPH=off|on|auto`
  - Added minimum-batch gating:
    - `STAP_FAST_CUDA_GRAPH_MIN_BATCH`
    - `auto` default min batch is `192`.
  - Applied batch-size gating in both fast paths:
    - `_stap_pd_tile_lcmv_batch` graph dispatch
    - unfold-tiling graph dispatch
- `scripts/profile_stap_hotspots.py`
  - Added profiling CLI controls:
    - `--cuda-graph {off,on,auto}`
    - `--cuda-graph-min-batch`
  - Added run-header prints for graph mode/min-batch.

## Profiling Commands
- Off baseline:
  - `PYTHONPATH=. conda run -n stap-fus python scripts/profile_stap_hotspots.py --mode stap_pd --device cuda --event-timing --iters 2 --warmup 1 --tile-batch 512 --cuda-graph off --row-limit 50`
- Auto graph (min batch 512):
  - `PYTHONPATH=. conda run -n stap-fus python scripts/profile_stap_hotspots.py --mode stap_pd --device cuda --event-timing --iters 2 --warmup 1 --tile-batch 512 --cuda-graph auto --cuda-graph-min-batch 512 --row-limit 50`

Artifacts:
- `reports/refactor/phase4_chunk2/profile_stap_pd_tb512_graph_off_args.txt`
- `reports/refactor/phase4_chunk2/profile_stap_pd_tb512_graph_auto512_args.txt`
- `reports/refactor/phase4_chunk2/profile_compare_tb512_args.json`

## Results (This Workstation / This Workload)
- `tile_batch=512`, graph **off**:
  - `stap:TOTAL` = `729.335 ms` (2 iters aggregate)
  - `stap:core` = `530.180 ms`
  - profiler self-CPU mix:
    - `cudaLaunchKernel`: `23.54%`
    - `cudaMemcpyAsync`: `7.58%`
    - `cudaStreamSynchronize`: `8.40%`
- `tile_batch=512`, graph **auto(min=512)**:
  - `stap:TOTAL` = `1025.486 ms` (2 iters aggregate)
  - `stap:core` = `888.801 ms`
  - profiler self-CPU mix:
    - `cudaLaunchKernel`: `4.70%`
    - `cudaMemcpyAsync`: `80.49%`
    - `cudaStreamSynchronize`: `0.49%`

Interpretation:
- Graph replay reduced launch API overhead, but shifted cost heavily into copy traffic for this dynamic unfold/gather workload.
- Net result in this setting is regression; graph should remain opt-in and workload-tuned.

## Opportunities Identified
1. **Tyler kernel path remains primary target**
   - `tyler:update/solve/chol/trace` remains the dominant compute block in non-graph runs.
2. **Unfold-path graph replay needs data-movement redesign**
   - Current replay path likely pays too much for input/output movement around static graph buffers.
3. **Next chunk recommendation**
   - Prioritize kernel-level Tyler path work (fusion/reduced launches) before broader graph rollout.
   - Revisit graph only after reducing movement overhead or introducing more static chunking.

## Verification
- `make refactor-quick` passed after this chunk.
