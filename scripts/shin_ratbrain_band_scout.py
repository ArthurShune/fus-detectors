from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from sim.kwave import common as kw


def _parse_slice(spec: str) -> list[int] | None:
    spec = (spec or "").strip()
    if spec in {"", "all", ":", "0:"}:
        return None
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]' or 'all'.")
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    if stop is None:
        raise ValueError("Slice spec must include stop (e.g. 0:128).")
    return list(range(start, stop, step))


def _default_masks_generic(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    cy = min(int(0.6 * H), H - 1)
    cx = W // 2
    rr = max(4, int(0.12 * min(H, W)))
    yy, xx = np.ogrid[:H, :W]
    mask_flow = (yy - cy) ** 2 + (xx - cx) ** 2 <= rr**2
    mask_bg = np.ones((H, W), dtype=bool)
    mask_bg[: max(1, H // 8), :] = False
    mask_bg &= ~mask_flow
    if mask_bg.sum() < 64:
        mask_bg = ~mask_flow
    return mask_flow.astype(bool), mask_bg.astype(bool)


def _compute_band_templates(
    cube: np.ndarray,
    *,
    prf_hz: float,
    tile_hw: tuple[int, int],
    stride: int,
    mask_flow: np.ndarray,
    pd_base: np.ndarray | None,
    max_tiles: int,
    seed: int,
    c_bg: float,
    c_flow: float,
    q_tail: float,
    q_mid_lo: float,
    q_mid_hi: float,
    tapers: int,
    bandwidth: float,
    normalize_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return (freqs, median_psd_flow, median_psd_bg, median_psd_bg_tail, median_psd_bg_mid)."""
    T, H, W = cube.shape
    cov_flow, coords = kw._tile_coverages(mask_flow.astype(bool), tile_hw, stride)
    idx_bg = np.nonzero(cov_flow <= float(c_bg))[0]
    idx_flow = np.nonzero(cov_flow >= float(c_flow))[0]
    rng = np.random.default_rng(int(seed))
    if idx_bg.size > max_tiles:
        idx_bg = rng.choice(idx_bg, size=max_tiles, replace=False)
    if idx_flow.size > max_tiles:
        idx_flow = rng.choice(idx_flow, size=max_tiles, replace=False)

    th, tw = tile_hw
    freqs_ref: np.ndarray | None = None

    def _maybe_normalize(psd: np.ndarray) -> np.ndarray:
        mode = (normalize_mode or "none").strip().lower()
        if mode in {"", "none"}:
            return psd
        if mode not in {"total", "total_no_dc"}:
            raise ValueError(f"Unsupported normalize_mode={normalize_mode!r}; use none,total,total_no_dc.")
        eps = 1e-12
        if mode == "total_no_dc" and psd.size > 1:
            denom = float(np.sum(psd[1:])) + eps
        else:
            denom = float(np.sum(psd)) + eps
        return psd / denom

    def _median_psd(tile_idx: np.ndarray) -> np.ndarray:
        psds: list[np.ndarray] = []
        nonlocal freqs_ref
        for ti in tile_idx.tolist():
            y0, x0 = coords[int(ti)]
            tile = cube[:, y0 : y0 + th, x0 : x0 + tw]
            series = np.mean(tile.reshape(T, -1), axis=1)
            freqs, psd = kw._multi_taper_psd(series, prf_hz, tapers=tapers, bandwidth=bandwidth)
            psd = _maybe_normalize(psd.astype(np.float32, copy=False))
            if freqs_ref is None:
                freqs_ref = freqs
            elif freqs.shape != freqs_ref.shape or not np.allclose(freqs, freqs_ref, atol=1e-6):
                raise RuntimeError("Frequency grid mismatch across PSD evaluations.")
            psds.append(psd.astype(np.float32, copy=False))
        if not psds:
            raise RuntimeError("No PSD observations for requested tile set.")
        return np.median(np.stack(psds, axis=0), axis=0).astype(np.float32, copy=False)

    if idx_bg.size == 0 or idx_flow.size == 0:
        raise RuntimeError(f"Need both bg and flow proxy tiles; got bg={idx_bg.size}, flow={idx_flow.size}.")

    psd_flow_med = _median_psd(idx_flow)
    psd_bg_med = _median_psd(idx_bg)

    # Background-tail vs background-mid templates (label-free actionability proxy).
    if pd_base is not None:
        tile_pd: list[float] = []
        idx_bg_all = np.nonzero(cov_flow <= float(c_bg))[0]
        for ti in idx_bg_all.tolist():
            y0, x0 = coords[int(ti)]
            patch = pd_base[y0 : y0 + th, x0 : x0 + tw]
            tile_pd.append(float(np.mean(patch)))
        tile_pd_arr = np.asarray(tile_pd, dtype=np.float64)
        if tile_pd_arr.size == idx_bg_all.size and tile_pd_arr.size >= 10:
            q_hi = float(np.quantile(tile_pd_arr, q_tail))
            q_lo = float(np.quantile(tile_pd_arr, q_mid_lo))
            q_hi_mid = float(np.quantile(tile_pd_arr, q_mid_hi))
            idx_tail = idx_bg_all[tile_pd_arr >= q_hi]
            idx_mid = idx_bg_all[(tile_pd_arr >= q_lo) & (tile_pd_arr <= q_hi_mid)]
            if idx_tail.size > max_tiles:
                idx_tail = rng.choice(idx_tail, size=max_tiles, replace=False)
            if idx_mid.size > max_tiles:
                idx_mid = rng.choice(idx_mid, size=max_tiles, replace=False)
            if idx_tail.size and idx_mid.size:
                psd_bg_tail = _median_psd(idx_tail)
                psd_bg_mid = _median_psd(idx_mid)
            else:
                psd_bg_tail = None
                psd_bg_mid = None
        else:
            psd_bg_tail = None
            psd_bg_mid = None
    else:
        psd_bg_tail = None
        psd_bg_mid = None

    # Package: callers can compute ratios; we return only flow/bg medians here.
    # Tail/mid templates are stored as globals via closure variables in main.
    return freqs_ref, psd_flow_med, psd_bg_med, psd_bg_tail, psd_bg_mid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Band-scout for Shin RatBrain Fig3 IQ: compute robust tile-level PSD templates on\n"
            "flow/bg proxy sets (derived from baseline PD) and write a ratio curve for\n"
            "profile calibration (Pf/Pa/guard selection + Lt sizing)."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"),
        help="Directory containing SizeInfo.dat and IQData*.dat (default: %(default)s).",
    )
    parser.add_argument("--iq-file", type=str, default="IQData001.dat")
    parser.add_argument("--frames", type=str, default="0:128", help="Frame slice start:stop[:step].")
    parser.add_argument("--prf-hz", type=float, default=1000.0)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="mc_svd",
        help="Baseline type: mc_svd or svd_bandpass (default: %(default)s).",
    )
    parser.add_argument("--svd-rank", type=int, default=None, help="For mc_svd: fixed rank removed.")
    parser.add_argument("--svd-energy-frac", type=float, default=0.95, help="For mc_svd: energy frac.")
    parser.add_argument("--svd-keep-min", type=int, default=None, help="For svd_bandpass: 1-based keep min.")
    parser.add_argument("--svd-keep-max", type=int, default=None, help="For svd_bandpass: 1-based keep max.")
    parser.add_argument("--flow-mask-pd-quantile", type=float, default=0.99)
    parser.add_argument("--flow-mask-min-pixels", type=int, default=64)
    parser.add_argument(
        "--flow-mask-union-default",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--flow-mask-depth-min-frac", type=float, default=0.0)
    parser.add_argument("--flow-mask-depth-max-frac", type=float, default=1.0)
    parser.add_argument("--flow-mask-erode-iters", type=int, default=0)
    parser.add_argument("--flow-mask-dilate-iters", type=int, default=1)
    parser.add_argument("--c-bg", type=float, default=0.05, help="Tile bg proxy: c_flow <= c-bg.")
    parser.add_argument("--c-flow", type=float, default=0.20, help="Tile flow proxy: c_flow >= c-flow.")
    parser.add_argument("--max-tiles", type=int, default=256, help="Max tiles per proxy set.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--psd-tapers", type=int, default=3)
    parser.add_argument("--psd-bandwidth", type=float, default=2.0)
    parser.add_argument(
        "--psd-normalize",
        type=str,
        default="total_no_dc",
        help="PSD normalization: none,total,total_no_dc (default: %(default)s).",
    )
    parser.add_argument(
        "--bg-tail-quantile",
        type=float,
        default=0.99,
        help="Background-tail quantile on tile PD means (default: %(default)s).",
    )
    parser.add_argument(
        "--bg-mid-quantiles",
        type=str,
        default="0.40,0.60",
        help="Two comma-separated mid quantiles on bg tile PD means (default: %(default)s).",
    )
    parser.add_argument(
        "--out-npz",
        type=Path,
        default=Path("reports/shin_band_scout.npz"),
        help="Output .npz path (default: %(default)s).",
    )
    args = parser.parse_args()

    data_root = args.data_root
    iq_path = data_root / args.iq_file
    if not iq_path.is_file():
        raise FileNotFoundError(f"IQ file not found: {iq_path}")

    info = load_shin_metadata(data_root)
    frames = _parse_slice(args.frames)
    Icube = load_shin_iq(iq_path, info, frames=frames)
    Icube = np.asarray(Icube, dtype=np.complex64)

    baseline_device = "cuda" if (kw._resolve_stap_device(None).startswith("cuda")) else "cpu"
    baseline_type = (args.baseline_type or "mc_svd").strip().lower()
    if baseline_type == "mc_svd":
        pd_base, baseline_tele, filtered = kw._baseline_pd_mcsvd(
            Icube,
            reg_enable=False,
            svd_rank=args.svd_rank,
            svd_energy_frac=float(args.svd_energy_frac),
            device=baseline_device,
            return_filtered_cube=True,
        )
    elif baseline_type in {"svd_bandpass", "svd_range", "ulm_svd"}:
        if args.svd_keep_min is None:
            raise ValueError("--svd-keep-min is required for --baseline-type svd_bandpass")
        pd_base, baseline_tele, filtered = kw._baseline_pd_svd_bandpass(
            Icube,
            reg_enable=False,
            svd_keep_min=int(args.svd_keep_min),
            svd_keep_max=int(args.svd_keep_max) if args.svd_keep_max is not None else None,
            device=baseline_device,
            return_filtered_cube=True,
        )
    else:
        raise ValueError(f"Unsupported baseline-type {baseline_type!r}")

    H, W = pd_base.shape
    mask_flow_default, mask_bg_default = _default_masks_generic(H, W)
    mask_flow, mask_bg, flow_mask_stats = kw._resolve_flow_mask(
        pd_base,
        mask_flow_default,
        mask_bg_default,
        mode="pd_auto",
        pd_quantile=float(args.flow_mask_pd_quantile),
        depth_min_frac=float(args.flow_mask_depth_min_frac),
        depth_max_frac=float(args.flow_mask_depth_max_frac),
        erode_iters=int(args.flow_mask_erode_iters),
        dilate_iters=int(args.flow_mask_dilate_iters),
        min_pixels=int(args.flow_mask_min_pixels),
        min_coverage_frac=0.0,
        union_with_default=bool(args.flow_mask_union_default),
    )

    try:
        q_mid_lo, q_mid_hi = (float(x.strip()) for x in str(args.bg_mid_quantiles).split(","))
    except Exception as exc:
        raise ValueError("--bg-mid-quantiles must be 'qlo,qhi' (e.g. 0.40,0.60).") from exc
    freqs, psd_flow, psd_bg, psd_bg_tail, psd_bg_mid = _compute_band_templates(
        filtered,
        prf_hz=float(args.prf_hz),
        tile_hw=(int(args.tile_h), int(args.tile_w)),
        stride=int(args.tile_stride),
        mask_flow=mask_flow,
        pd_base=pd_base,
        max_tiles=int(args.max_tiles),
        seed=int(args.seed),
        c_bg=float(args.c_bg),
        c_flow=float(args.c_flow),
        q_tail=float(args.bg_tail_quantile),
        q_mid_lo=q_mid_lo,
        q_mid_hi=q_mid_hi,
        tapers=int(args.psd_tapers),
        bandwidth=float(args.psd_bandwidth),
        normalize_mode=str(args.psd_normalize),
    )
    eps = 1e-12
    ratio = np.log((psd_flow + eps) / (psd_bg + eps)).astype(np.float32, copy=False)
    if psd_bg_tail is not None and psd_bg_mid is not None:
        ratio_tail_mid = np.log((psd_bg_tail + eps) / (psd_bg_mid + eps)).astype(np.float32, copy=False)
    else:
        ratio_tail_mid = None

    out_path = Path(args.out_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        freqs_hz=freqs.astype(np.float32, copy=False),
        psd_flow_median=psd_flow.astype(np.float32, copy=False),
        psd_bg_median=psd_bg.astype(np.float32, copy=False),
        log_ratio_flow_bg=ratio,
        psd_bg_tail_median=psd_bg_tail.astype(np.float32, copy=False) if psd_bg_tail is not None else None,
        psd_bg_mid_median=psd_bg_mid.astype(np.float32, copy=False) if psd_bg_mid is not None else None,
        log_ratio_bg_tail_mid=ratio_tail_mid.astype(np.float32, copy=False) if ratio_tail_mid is not None else None,
        pd_base=pd_base.astype(np.float32, copy=False),
        mask_flow=mask_flow.astype(np.bool_, copy=False),
        mask_bg=mask_bg.astype(np.bool_, copy=False),
        baseline_telemetry=json.dumps(baseline_tele),
        flow_mask_stats=json.dumps(flow_mask_stats),
        meta=json.dumps(
            {
                "iq_file": args.iq_file,
                "frames": args.frames,
                "prf_hz": float(args.prf_hz),
                "tile_hw": [int(args.tile_h), int(args.tile_w)],
                "tile_stride": int(args.tile_stride),
                "baseline_type": baseline_type,
                "psd_normalize": str(args.psd_normalize),
                "bg_tail_quantile": float(args.bg_tail_quantile),
                "bg_mid_quantiles": [q_mid_lo, q_mid_hi],
            }
        ),
    )

    # Print a compact summary to guide profile selection.
    peak_flow = float(freqs[int(np.argmax(psd_flow))])
    peak_bg = float(freqs[int(np.argmax(psd_bg))])
    pos = ratio > 0
    pos_frac = float(np.mean(pos))
    print(f"[shin-band-scout] wrote: {out_path}")
    print(f"[shin-band-scout] baseline_type={baseline_type} svd={baseline_tele}")
    print(f"[shin-band-scout] flow_mask_pixels={int(mask_flow.sum())} bg_mask_pixels={int(mask_bg.sum())}")
    print(
        "[shin-band-scout]"
        f" psd_normalize={str(args.psd_normalize)}"
        f" peak_hz flow={peak_flow:.1f}"
        f" bg={peak_bg:.1f}"
        f" log_ratio_pos_frac={pos_frac:.3f}"
    )
    print(
        "[shin-band-scout] ratio summary (flow/bg): "
        f"p10={float(np.quantile(ratio,0.10)):.3f} "
        f"p50={float(np.median(ratio)):.3f} "
        f"p90={float(np.quantile(ratio,0.90)):.3f}"
    )
    if ratio_tail_mid is not None:
        print(
            "[shin-band-scout] ratio summary (bg tail/mid): "
            f"p10={float(np.quantile(ratio_tail_mid,0.10)):.3f} "
            f"p50={float(np.median(ratio_tail_mid)):.3f} "
            f"p90={float(np.quantile(ratio_tail_mid,0.90)):.3f}"
        )


if __name__ == "__main__":
    main()
