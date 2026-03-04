#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

try:
    import torch
    from torch.profiler import ProfilerActivity, profile
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    ProfilerActivity = None  # type: ignore
    profile = None  # type: ignore


@dataclass(frozen=True)
class StapProfileConfig:
    device: str
    dtype: str
    mode: str
    B: int
    T: int
    H: int
    W: int
    tile_h: int
    tile_w: int
    stride: int
    Lt: int
    prf_hz: float
    diag_load: float
    cov_estimator: str
    huber_c: float
    fd_span_mode: str
    fd_span_rel: Tuple[float, float]
    grid_step_rel: float
    fd_min_pts: int
    fd_max_pts: int
    fd_min_abs_hz: float
    msd_ridge: float
    msd_agg_mode: str
    msd_ratio_rho: float
    motion_half_span_rel: float | None
    warmup: int
    iters: int
    row_limit: int
    sort_by: str
    chrome_trace: str | None
    seed: int
    mask_flow_frac: float
    event_timing: bool
    tile_statistic: bool
    tile_batch: int | None
    cuda_graph: str
    cuda_graph_min_batch: int | None


def _parse_pair_floats(text: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected 'a,b' for a float pair.")
    return float(parts[0]), float(parts[1])


def _bool_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _torch_device(device: str) -> str:
    dev = str(device or "cuda")
    if dev.lower().startswith("cuda") and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _torch_complex_dtype(dtype: str) -> "torch.dtype":
    dt = str(dtype or "complex64").strip().lower()
    if dt in {"complex128", "c128"}:
        return torch.complex128
    return torch.complex64


def _warmup_cuda_kernels(device: str) -> None:
    if torch is None:
        return
    if not str(device).lower().startswith("cuda"):
        return
    if not torch.cuda.is_available():
        return
    dev = device
    _ = torch.empty((1,), device=dev)
    a = torch.randn((32, 32), device=dev, dtype=torch.float32)
    b = torch.randn((32, 32), device=dev, dtype=torch.float32)
    ac = (a + 1j * b).to(dtype=torch.complex64)
    bc = (b + 1j * a).to(dtype=torch.complex64)
    _ = ac @ bc
    C = ac @ ac.conj().T
    C_pd = C + (1e-2 * torch.eye(C.shape[0], device=dev, dtype=C.dtype))
    L = torch.linalg.cholesky(C_pd)
    rr = torch.randn((C.shape[0], 64), device=dev, dtype=torch.float32)
    ri = torch.randn((C.shape[0], 64), device=dev, dtype=torch.float32)
    rhs = (rr + 1j * ri).to(dtype=C.dtype)
    _ = torch.linalg.solve_triangular(L, rhs, upper=False)
    _ = torch.cholesky_solve(rhs, L)
    x = torch.randn((32, 32), device=dev, dtype=torch.float32)
    xc = (x + 1j * x).to(dtype=torch.complex64)
    _ = torch.fft.fft2(xc)
    v = torch.rand((1024,), device=dev, dtype=torch.float32)
    _ = torch.sort(v)
    _ = torch.topk(v, k=32)
    torch.cuda.synchronize()


def _run_core_profile(cfg: StapProfileConfig) -> None:
    if torch is None or profile is None:
        raise RuntimeError("torch + torch.profiler are required for profiling.")

    from pipeline.stap.temporal import (
        pd_temporal_core_batched,
        stap_cuda_event_timing,
        stap_stage_ctx,
        stap_temporal_core_batched,
    )

    dev = _torch_device(cfg.device)
    dt = _torch_complex_dtype(cfg.dtype)

    os.environ.setdefault("STAP_PROFILE_MARKERS", "1")
    os.environ.setdefault("STAP_SNAPSHOT_STRIDE", "4")
    os.environ.setdefault("STAP_MAX_SNAPSHOTS", "64")
    if cfg.tile_statistic:
        os.environ["STAP_FAST_TILE_STATISTIC"] = "1"

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    _warmup_cuda_kernels(dev)

    B, T, h, w = int(cfg.B), int(cfg.T), int(cfg.tile_h), int(cfg.tile_w)
    rr = torch.randn((B, T, h, w), device=dev, dtype=torch.float32)
    ri = torch.randn((B, T, h, w), device=dev, dtype=torch.float32)
    cube = (rr + 1j * ri).to(dtype=dt)

    fn = pd_temporal_core_batched if cfg.mode == "core_pd" else stap_temporal_core_batched
    fn_name = fn.__name__

    kwargs = dict(
        prf_hz=float(cfg.prf_hz),
        Lt=int(cfg.Lt),
        diag_load=float(cfg.diag_load),
        kappa_shrink=200.0,
        kappa_msd=200.0,
        cov_estimator=str(cfg.cov_estimator),
        huber_c=float(cfg.huber_c),
        grid_step_rel=float(cfg.grid_step_rel),
        fd_span_rel=tuple(float(x) for x in cfg.fd_span_rel),
        min_pts=int(cfg.fd_min_pts),
        max_pts=int(cfg.fd_max_pts),
        fd_min_abs_hz=float(cfg.fd_min_abs_hz),
        motion_half_span_rel=cfg.motion_half_span_rel,
        msd_ridge=float(cfg.msd_ridge),
        msd_agg_mode=str(cfg.msd_agg_mode),
        msd_ratio_rho=float(cfg.msd_ratio_rho),
        msd_contrast_alpha=0.6,
        msd_lambda=0.05,
        device=dev,
        use_ref_cov=False,
        fd_span_mode=str(cfg.fd_span_mode),
        flow_band_hz=None,
        return_torch=True,
    )

    with torch.no_grad():
        for _ in range(max(0, int(cfg.warmup))):
            _ = fn(cube, **kwargs)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    print(f"[profile] mode={cfg.mode} fn={fn_name} dev={dev} dtype={dt} B={B} T={T} h={h} w={w} Lt={cfg.Lt}")
    print(
        f"[profile] cov_estimator={cfg.cov_estimator} snapshot_stride={os.getenv('STAP_SNAPSHOT_STRIDE')} "
        f"max_snaps={os.getenv('STAP_MAX_SNAPSHOTS')}"
    )
    if cfg.tile_statistic:
        print("[profile] STAP_FAST_TILE_STATISTIC=1")

    if cfg.event_timing and dev.startswith("cuda"):
        with stap_cuda_event_timing(enabled=True) as timer:
            with torch.no_grad():
                for _ in range(max(1, int(cfg.iters))):
                    with stap_stage_ctx("stap:TOTAL"):
                        _ = fn(cube, **kwargs)
        stage_ms = timer.summary_ms()
        if stage_ms:
            iters = max(1, int(cfg.iters))
            total_ms = float(stage_ms.get("stap:TOTAL", sum(stage_ms.values())))
            print("\n[cuda_event] stage times (ms)")
            print(f"{'total_ms':>10} {'avg_ms':>10}  key")
            for name, ms in sorted(stage_ms.items(), key=lambda kv: float(kv[1]), reverse=True):
                print(f"{ms:10.3f} {ms/iters:10.3f}  {name}")
            print(f"[cuda_event] TOTAL={total_ms:.3f} ms over iters={iters}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if dev.startswith("cuda") else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(max(1, int(cfg.iters))):
                _ = fn(cube, **kwargs)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    if cfg.chrome_trace:
        prof.export_chrome_trace(str(cfg.chrome_trace))
        print(f"[profile] wrote chrome trace: {cfg.chrome_trace}")

    # Full table (top ops by CUDA time).
    print("\n[torch.profiler] key_averages (sorted)")
    print(prof.key_averages().table(sort_by=str(cfg.sort_by), row_limit=int(cfg.row_limit)))

    # Marker-only view (helps stage attribution).
    markers = [e for e in prof.key_averages() if str(e.key).startswith("stap:")]
    if markers:
        print("\n[torch.profiler] stage markers (stap:*)")
        sort_key = str(cfg.sort_by)
        if "cuda" in sort_key:
            markers.sort(key=lambda e: float(getattr(e, "cuda_time_total", 0.0)), reverse=True)
        elif "cpu" in sort_key:
            markers.sort(key=lambda e: float(getattr(e, "cpu_time_total", 0.0)), reverse=True)

        # torch.profiler timings are reported in microseconds; present stage attribution in ms.
        header = f"{'cuda_ms':>10} {'cpu_ms':>10} {'count':>7}  key"
        print(header)
        for e in markers[: int(cfg.row_limit)]:
            cuda_us = float(getattr(e, "cuda_time_total", 0.0))
            cpu_us = float(getattr(e, "cpu_time_total", 0.0))
            count = int(getattr(e, "count", 0))
            print(f"{cuda_us/1000.0:10.3f} {cpu_us/1000.0:10.3f} {count:7d}  {e.key}")


def _run_stap_pd_profile(cfg: StapProfileConfig) -> None:
    if torch is None or profile is None:
        raise RuntimeError("torch + torch.profiler are required for profiling.")

    # Local import: this module is large and optional for core-only profiling.
    from sim.kwave.common import _stap_pd
    from pipeline.stap.temporal import stap_cuda_event_timing, stap_stage_ctx

    dev = _torch_device(cfg.device)
    dt = _torch_complex_dtype(cfg.dtype)

    os.environ.setdefault("STAP_PROFILE_MARKERS", "1")
    os.environ.setdefault("STAP_FAST_PATH", "1")
    os.environ.setdefault("STAP_FAST_PD_ONLY", "1")
    os.environ.setdefault("STAP_TILING_UNFOLD", "1")
    os.environ.setdefault("STAP_SNAPSHOT_STRIDE", "4")
    os.environ.setdefault("STAP_MAX_SNAPSHOTS", "64")
    graph_mode = str(cfg.cuda_graph or "off").strip().lower()
    if graph_mode == "on":
        os.environ["STAP_FAST_CUDA_GRAPH"] = "1"
    elif graph_mode == "auto":
        os.environ["STAP_FAST_CUDA_GRAPH"] = "auto"
    else:
        os.environ["STAP_FAST_CUDA_GRAPH"] = "0"
    if cfg.cuda_graph_min_batch is not None and int(cfg.cuda_graph_min_batch) > 0:
        os.environ["STAP_FAST_CUDA_GRAPH_MIN_BATCH"] = str(int(cfg.cuda_graph_min_batch))
    if cfg.tile_statistic:
        os.environ["STAP_FAST_TILE_STATISTIC"] = "1"

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    _warmup_cuda_kernels(dev)

    T, H, W = int(cfg.T), int(cfg.H), int(cfg.W)
    rr = torch.randn((T, H, W), device=dev, dtype=torch.float32)
    ri = torch.randn((T, H, W), device=dev, dtype=torch.float32)
    cube = (rr + 1j * ri).to(dtype=dt)

    # Random sparse-ish masks: approximate the pilot coverage (~3% pixels => ~10-15% tiles active).
    mask_flow = (np.random.rand(H, W) < float(cfg.mask_flow_frac)).astype(bool, copy=False)
    mask_bg = (~mask_flow).astype(bool, copy=False)

    kwargs = dict(
        tile_hw=(int(cfg.tile_h), int(cfg.tile_w)),
        stride=int(cfg.stride),
        Lt=int(cfg.Lt),
        prf_hz=float(cfg.prf_hz),
        diag_load=float(cfg.diag_load),
        estimator=str(cfg.cov_estimator),
        huber_c=float(cfg.huber_c),
        grid_step_rel=float(cfg.grid_step_rel),
        fd_span_rel=tuple(float(x) for x in cfg.fd_span_rel),
        min_pts=int(cfg.fd_min_pts),
        max_pts=int(cfg.fd_max_pts),
        fd_min_abs_hz=float(cfg.fd_min_abs_hz),
        motion_half_span_rel=cfg.motion_half_span_rel,
        msd_ridge=float(cfg.msd_ridge),
        msd_agg_mode=str(cfg.msd_agg_mode),
        msd_ratio_rho=float(cfg.msd_ratio_rho),
        msd_contrast_alpha=0.6,
        msd_lambda=0.05,
        stap_device=dev,
        tile_batch=cfg.tile_batch,
        pd_base_full=None,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        conditional_enable=True,
        ka_mode="none",
        ka_prior_library=None,
        ka_opts=None,
        psd_telemetry=False,
        feasibility_mode="legacy",
    )

    with torch.no_grad():
        for _ in range(max(0, int(cfg.warmup))):
            _ = _stap_pd(cube, **kwargs)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    print(
        f"[profile] mode=stap_pd dev={dev} dtype={dt} T={T} H={H} W={W} tile={cfg.tile_h}x{cfg.tile_w} "
        f"stride={cfg.stride} Lt={cfg.Lt} mask_flow_frac={cfg.mask_flow_frac} "
        f"cuda_graph={graph_mode}"
    )
    if cfg.tile_batch is not None:
        print(f"[profile] tile_batch={int(cfg.tile_batch)}")
    if cfg.cuda_graph_min_batch is not None and int(cfg.cuda_graph_min_batch) > 0:
        print(f"[profile] cuda_graph_min_batch={int(cfg.cuda_graph_min_batch)}")
    if cfg.tile_statistic:
        print("[profile] STAP_FAST_TILE_STATISTIC=1")

    if cfg.event_timing and dev.startswith("cuda"):
        with stap_cuda_event_timing(enabled=True) as timer:
            with torch.no_grad():
                for _ in range(max(1, int(cfg.iters))):
                    with stap_stage_ctx("stap:TOTAL"):
                        _ = _stap_pd(cube, **kwargs)
        stage_ms = timer.summary_ms()
        if stage_ms:
            iters = max(1, int(cfg.iters))
            total_ms = float(stage_ms.get("stap:TOTAL", sum(stage_ms.values())))
            print("\n[cuda_event] stage times (ms)")
            print(f"{'total_ms':>10} {'avg_ms':>10}  key")
            for name, ms in sorted(stage_ms.items(), key=lambda kv: float(kv[1]), reverse=True):
                print(f"{ms:10.3f} {ms/iters:10.3f}  {name}")
            print(f"[cuda_event] TOTAL={total_ms:.3f} ms over iters={iters}")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if dev.startswith("cuda") else [ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(max(1, int(cfg.iters))):
                _ = _stap_pd(cube, **kwargs)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()

    if cfg.chrome_trace:
        prof.export_chrome_trace(str(cfg.chrome_trace))
        print(f"[profile] wrote chrome trace: {cfg.chrome_trace}")

    print("\n[torch.profiler] key_averages (sorted)")
    print(prof.key_averages().table(sort_by=str(cfg.sort_by), row_limit=int(cfg.row_limit)))


def parse_args() -> StapProfileConfig:
    ap = argparse.ArgumentParser(description="Profile STAP hotspots (fast path) with torch.profiler.")
    ap.add_argument(
        "--mode",
        type=str,
        default="core_pd",
        choices=["core_pd", "core_full", "stap_pd"],
        help="Profile the temporal core or end-to-end _stap_pd unfold path.",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="complex64", choices=["complex64", "complex128"])

    # Shapes: defaults match Brain-* pilot geometry.
    ap.add_argument("--B", type=int, default=192, help="Tile batch size for core profiling.")
    ap.add_argument("--T", type=int, default=64, help="Slow-time window length.")
    ap.add_argument("--H", type=int, default=160, help="Full image height (stap_pd mode).")
    ap.add_argument("--W", type=int, default=160, help="Full image width (stap_pd mode).")
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--Lt", type=int, default=8)

    # Algorithm knobs: defaults match clinical Brain-* preset.
    ap.add_argument("--prf-hz", type=float, default=1500.0)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--huber-c", type=float, default=5.0)
    ap.add_argument("--fd-span-mode", type=str, default="fixed")
    ap.add_argument("--fd-span-rel", type=str, default="0.30,1.10")
    ap.add_argument("--grid-step-rel", type=float, default=0.20)
    ap.add_argument("--fd-min-pts", type=int, default=9)
    ap.add_argument("--fd-max-pts", type=int, default=15)
    ap.add_argument("--fd-min-abs-hz", type=float, default=0.0)
    ap.add_argument("--msd-ridge", type=float, default=0.10)
    ap.add_argument("--msd-agg", type=str, default="median")
    ap.add_argument("--msd-ratio-rho", type=float, default=0.05)
    ap.add_argument("--motion-half-span-rel", type=float, default=None)

    # Profiler controls.
    ap.add_argument("--warmup", type=int, default=2, help="Warmup iters before profiling.")
    ap.add_argument("--iters", type=int, default=1, help="Profiled iterations.")
    ap.add_argument("--row-limit", type=int, default=30)
    ap.add_argument("--sort-by", type=str, default="cuda_time_total")
    ap.add_argument("--chrome-trace", type=str, default=None, help="Optional chrome trace output path.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--mask-flow-frac",
        type=float,
        default=0.03,
        help="Pixel fraction for random flow mask in stap_pd mode.",
    )
    ap.add_argument(
        "--event-timing",
        action="store_true",
        help="Print per-stage CUDA times using CUDA events (independent of torch.profiler CUDA time).",
    )
    ap.add_argument(
        "--tile-statistic",
        action="store_true",
        help=(
            "EXPERIMENTAL / NOT FOR PAPER RESULTS: enable covariance-only (tile-statistic) STAP scoring "
            "(sets STAP_FAST_TILE_STATISTIC=1). Known to catastrophically regress strict-FPR ROC on "
            "Twinkling/Gammex because it replaces per-snapshot nonlinear MSD aggregation with a "
            "ratio-of-means approximation (mean(f) != f(mean))."
        ),
    )
    ap.add_argument(
        "--tile-batch",
        type=int,
        default=0,
        help=(
            "Tile batch size passed to _stap_pd in stap_pd mode (0 = internal default). "
            "For latency experiments, values like 384/512 often improve throughput."
        ),
    )
    ap.add_argument(
        "--cuda-graph",
        type=str,
        default="off",
        choices=["off", "on", "auto"],
        help=(
            "CUDA-graph mode for stap_pd profiling: off/on/auto. "
            "'auto' uses STAP_FAST_CUDA_GRAPH auto policy."
        ),
    )
    ap.add_argument(
        "--cuda-graph-min-batch",
        type=int,
        default=0,
        help=(
            "Minimum batch size for CUDA-graph use in stap_pd mode. "
            "0 keeps environment/default policy."
        ),
    )

    args = ap.parse_args()
    return StapProfileConfig(
        device=str(args.device),
        dtype=str(args.dtype),
        mode=str(args.mode),
        B=int(args.B),
        T=int(args.T),
        H=int(args.H),
        W=int(args.W),
        tile_h=int(args.tile_h),
        tile_w=int(args.tile_w),
        stride=int(args.stride),
        Lt=int(args.Lt),
        prf_hz=float(args.prf_hz),
        diag_load=float(args.diag_load),
        cov_estimator=str(args.cov_estimator),
        huber_c=float(args.huber_c),
        fd_span_mode=str(args.fd_span_mode),
        fd_span_rel=_parse_pair_floats(str(args.fd_span_rel)),
        grid_step_rel=float(args.grid_step_rel),
        fd_min_pts=int(args.fd_min_pts),
        fd_max_pts=int(args.fd_max_pts),
        fd_min_abs_hz=float(args.fd_min_abs_hz),
        msd_ridge=float(args.msd_ridge),
        msd_agg_mode=str(args.msd_agg),
        msd_ratio_rho=float(args.msd_ratio_rho),
        motion_half_span_rel=args.motion_half_span_rel if args.motion_half_span_rel is not None else None,
        warmup=int(args.warmup),
        iters=int(args.iters),
        row_limit=int(args.row_limit),
        sort_by=str(args.sort_by),
        chrome_trace=str(args.chrome_trace) if args.chrome_trace else None,
        seed=int(args.seed),
        mask_flow_frac=float(args.mask_flow_frac),
        event_timing=bool(args.event_timing),
        tile_statistic=bool(args.tile_statistic),
        tile_batch=int(args.tile_batch) if int(args.tile_batch) > 0 else None,
        cuda_graph=str(args.cuda_graph),
        cuda_graph_min_batch=(
            int(args.cuda_graph_min_batch) if int(args.cuda_graph_min_batch) > 0 else None
        ),
    )


def main() -> None:
    cfg = parse_args()
    if torch is None:
        raise SystemExit("torch is required to run this profiler.")
    if cfg.mode in {"core_pd", "core_full"}:
        _run_core_profile(cfg)
    elif cfg.mode == "stap_pd":
        _run_stap_pd_profile(cfg)
    else:
        raise SystemExit(f"Unknown mode {cfg.mode}")


if __name__ == "__main__":
    main()
