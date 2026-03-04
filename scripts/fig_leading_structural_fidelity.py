#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from pipeline.realdata.twinkling_artifact import RawBCFPar, decode_rawbcf_cfm_cube, parse_rawbcf_par, read_rawbcf_frame
from pipeline.realdata.twinkling_bmode_mask import build_tube_masks_from_rawbcf
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_indices(spec: str, n_max: int) -> list[int]:
    spec = (spec or "").strip()
    if not spec:
        return list(range(n_max))
    if "," in spec:
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
        return out
    if ":" in spec:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid slice spec: {spec!r}")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1]) if parts[1] else n_max
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, min(stop, n_max), step))
    return [int(spec)]


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float, float]:
    bg = np.asarray(bg_scores, dtype=np.float64).ravel()
    bg = bg[np.isfinite(bg)]
    n = int(bg.size)
    if n <= 0:
        raise ValueError("Empty background score pool.")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        tau = float("inf")
        return tau, 0.0
    if a >= 1.0:
        tau = float(np.min(bg))
        return tau, 1.0
    k = int(np.ceil(a * n))
    k = max(1, min(k, n))
    tau = float(np.partition(bg, n - k)[n - k])
    realized = float(np.mean(bg >= tau))
    return tau, realized


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate a leading qualitative figure: baseline vs STAP score maps, plus a matched-FPR "
            "decision-difference panel, on a structurally labeled Gammex flow phantom acquisition."
        )
    )
    ap.add_argument(
        "--seq-dir",
        type=Path,
        default=Path(
            "data/twinkling_artifact/Flow in Gammex phantom/Flow in Gammex phantom (along - linear probe)"
        ),
        help="Gammex sequence directory containing RawBCFCine.dat / RawBCFCine.par (default: %(default)s).",
    )
    ap.add_argument("--par-path", type=Path, default=None, help="Optional explicit .par path (default: auto).")
    ap.add_argument("--dat-path", type=Path, default=None, help="Optional explicit .dat path (default: auto).")
    ap.add_argument("--frame-idx", type=int, default=0, help="Frame index to render (default: %(default)s).")
    ap.add_argument(
        "--mask-ref-frames",
        type=str,
        default="0:10",
        help="Frame indices used to build the median B-mode reference for tube segmentation (default: %(default)s).",
    )
    ap.add_argument("--prf-hz", type=float, default=2500.0, help="Assumed Doppler PRF in Hz (default: %(default)s).")
    ap.add_argument("--Lt", type=int, default=16, help="STAP temporal aperture Lt (default: %(default)s).")
    ap.add_argument(
        "--tile-hw",
        type=int,
        nargs=2,
        default=(8, 8),
        metavar=("H", "W"),
        help="Tile height/width in pixels (default: %(default)s).",
    )
    ap.add_argument("--tile-stride", type=int, default=6, help="Tile stride (default: %(default)s).")
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca", help="Covariance estimator (default: %(default)s).")
    ap.add_argument("--diag-load", type=float, default=0.07, help="Diagonal loading (default: %(default)s).")
    ap.add_argument(
        "--stap-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="STAP compute device inside bundle writer (default: %(default)s).",
    )
    ap.add_argument(
        "--stap-fast-path",
        action="store_true",
        help="Enable STAP fast path (sets STAP_FAST_PATH=1; default: off).",
    )
    ap.add_argument(
        "--fd-span-mode",
        type=str,
        default="psd",
        choices=["psd", "flow_band", "fixed", "band"],
        help="STAP flow-subspace frequency grid policy (default: %(default)s).",
    )
    ap.add_argument(
        "--feasibility-mode",
        type=str,
        default="legacy",
        choices=["legacy", "updated", "blend"],
        help="STAP feasibility mode (default: %(default)s).",
    )
    ap.add_argument(
        "--svd-keep-min",
        type=int,
        default=2,
        help="Baseline SVD band-pass keep-min index (default: %(default)s).",
    )
    ap.add_argument(
        "--svd-keep-max",
        type=int,
        default=17,
        help="Baseline SVD band-pass keep-max index (default: %(default)s).",
    )
    ap.add_argument(
        "--fpr",
        type=float,
        default=1e-2,
        help="Background FPR target used to threshold each method (default: %(default)s).",
    )
    ap.add_argument(
        "--flow-band-hz",
        type=float,
        nargs=2,
        default=(150.0, 450.0),
        metavar=("F_LO", "F_HI"),
        help=(
            "Flow band in Hz used for band-ratio telemetry + (indirectly) Pf/Pa-aware STAP heuristics "
            "(default: %(default)s; matches scripts/twinkling_make_bundles.py)."
        ),
    )
    ap.add_argument(
        "--alias-band-hz",
        type=float,
        nargs=2,
        default=(700.0, 1200.0),
        metavar=("A_LO", "A_HI"),
        help=(
            "Alias band in Hz used for band-ratio telemetry + (indirectly) Pf/Pa-aware STAP heuristics "
            "(default: %(default)s; matches scripts/twinkling_make_bundles.py)."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figs/paper/leading_structural_fidelity_gammex.pdf"),
        help="Output PDF path (default: %(default)s).",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Output DPI (used mainly for rasterized elements; default: %(default)s).",
    )
    return ap.parse_args()


def _robust_log_image(x: np.ndarray, *, eps: float = 1e-12) -> Tuple[np.ndarray, float, float]:
    x = np.asarray(x, dtype=np.float64)
    xx = np.log10(np.clip(x, 0.0, None) + float(eps))
    finite = xx[np.isfinite(xx)]
    if finite.size == 0:
        return xx.astype(np.float32, copy=False), 0.0, 1.0
    vmin = float(np.quantile(finite, 0.02))
    vmax = float(np.quantile(finite, 0.995))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1.0
    return xx.astype(np.float32, copy=False), vmin, vmax


def _bbox_from_mask(mask: np.ndarray, *, pad: int = 8) -> tuple[int, int, int, int]:
    yy, xx = np.where(mask)
    if yy.size == 0 or xx.size == 0:
        raise ValueError("Empty mask; cannot compute zoom region.")
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0 = max(0, y0 - int(pad))
    x0 = max(0, x0 - int(pad))
    y1 = min(int(mask.shape[0]), y1 + int(pad))
    x1 = min(int(mask.shape[1]), x1 + int(pad))
    return y0, y1, x0, x1


def main() -> int:
    args = parse_args()
    if bool(args.stap_fast_path):
        os.environ["STAP_FAST_PATH"] = "1"

    seq_dir = Path(args.seq_dir)
    if not seq_dir.is_dir():
        raise SystemExit(f"Missing sequence dir: {seq_dir}")

    par_path = Path(args.par_path) if args.par_path is not None else (seq_dir / "RawBCFCine.par")
    dat_path = Path(args.dat_path) if args.dat_path is not None else (seq_dir / "RawBCFCine.dat")
    if not par_path.is_file():
        raise SystemExit(f"Missing .par file: {par_path}")
    if not dat_path.is_file():
        raise SystemExit(f"Missing .dat file: {dat_path}")

    par_dict = parse_rawbcf_par(par_path)
    par = RawBCFPar.from_dict(par_dict)
    par.validate()

    frame_idx = int(args.frame_idx)
    if frame_idx < 0 or frame_idx >= int(par.num_frames):
        raise SystemExit(f"--frame-idx out of range: {frame_idx} (NumOfFrames={par.num_frames})")

    ref_frames = _parse_indices(str(args.mask_ref_frames), int(par.num_frames))
    if not ref_frames:
        raise SystemExit("--mask-ref-frames parsed as empty.")

    tube = build_tube_masks_from_rawbcf(dat_path, par, ref_frame_indices=ref_frames)

    frame = read_rawbcf_frame(dat_path, par, int(frame_idx))
    Icube = decode_rawbcf_cfm_cube(frame, par, order="beam_major")

    flow_lo, flow_hi = float(args.flow_band_hz[0]), float(args.flow_band_hz[1])
    alias_lo, alias_hi = float(args.alias_band_hz[0]), float(args.alias_band_hz[1])
    alias_center = 0.5 * (alias_lo + alias_hi)
    alias_width = 0.5 * (alias_hi - alias_lo)

    out_root = Path("runs/_tmp_leading_structural_fidelity")
    dataset_name = f"gammex_leading/frame{frame_idx:03d}"
    paths = write_acceptance_bundle_from_icube(
        out_root=out_root,
        dataset_name=dataset_name,
        Icube=Icube,
        prf_hz=float(args.prf_hz),
        tile_hw=(int(args.tile_hw[0]), int(args.tile_hw[1])),
        tile_stride=int(args.tile_stride),
        Lt=int(args.Lt),
        diag_load=float(args.diag_load),
        cov_estimator=str(args.cov_estimator),
        baseline_type="svd_bandpass",
        svd_keep_min=int(args.svd_keep_min),
        svd_keep_max=int(args.svd_keep_max),
        score_mode="msd",
        stap_device=str(args.stap_device),
        stap_conditional_enable=False,
        band_ratio_flow_low_hz=flow_lo,
        band_ratio_flow_high_hz=flow_hi,
        band_ratio_alias_center_hz=alias_center,
        band_ratio_alias_width_hz=alias_width,
        fd_span_mode=str(args.fd_span_mode),
        feasibility_mode=str(args.feasibility_mode),
        mask_flow_override=tube.mask_flow,
        mask_bg_override=tube.mask_bg,
        score_ka_v2_enable=False,
    )

    bundle_dir = Path(paths["meta"]).parent
    score_base = np.load(bundle_dir / "score_base.npy", allow_pickle=False).astype(np.float64, copy=False)
    score_stap = np.load(bundle_dir / "score_stap_preka.npy", allow_pickle=False).astype(np.float64, copy=False)
    mask_flow = np.load(bundle_dir / "mask_flow.npy", allow_pickle=False).astype(bool, copy=False)
    mask_bg = np.load(bundle_dir / "mask_bg.npy", allow_pickle=False).astype(bool, copy=False)

    if score_base.shape != score_stap.shape or score_base.shape != mask_flow.shape:
        raise SystemExit(
            f"Shape mismatch: base={score_base.shape} stap={score_stap.shape} flow={mask_flow.shape}"
        )

    alpha = float(args.fpr)
    thr_base, fpr_base = _right_tail_threshold(score_base[mask_bg], alpha=alpha)
    thr_stap, fpr_stap = _right_tail_threshold(score_stap[mask_bg], alpha=alpha)

    det_base = score_base >= float(thr_base)
    det_stap = score_stap >= float(thr_stap)
    base_only = det_base & (~det_stap)
    stap_only = det_stap & (~det_base)

    # --- Plot ---
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    base_log, base_vmin, base_vmax = _robust_log_image(score_base, eps=1e-9)
    stap_log, stap_vmin, stap_vmax = _robust_log_image(score_stap, eps=1e-14)

    y0, y1, x0, x1 = _bbox_from_mask(mask_flow, pad=10)

    dpi = int(args.dpi)
    fig = plt.figure(figsize=(10.0, 6.4), dpi=dpi)
    gs = fig.add_gridspec(2, 3, wspace=0.04, hspace=0.08)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    ax02 = fig.add_subplot(gs[0, 2])
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    ax12 = fig.add_subplot(gs[1, 2])

    for ax in (ax00, ax01, ax02, ax10, ax11, ax12):
        ax.set_axis_off()

    # Okabe-Ito palette.
    c_stap = "#0072B2"  # blue
    c_base = "#D55E00"  # vermillion

    # Full-view maps.
    ax00.imshow(base_log, cmap="viridis", vmin=base_vmin, vmax=base_vmax, origin="upper", interpolation="nearest")
    ax00.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax00.contour(mask_flow.astype(float), levels=[0.5], colors=[c_stap], linewidths=[1.1], origin="upper")
    ax00.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax00.set_title("Baseline score", fontsize=10)

    ax01.imshow(stap_log, cmap="viridis", vmin=stap_vmin, vmax=stap_vmax, origin="upper", interpolation="nearest")
    ax01.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax01.contour(mask_flow.astype(float), levels=[0.5], colors=[c_stap], linewidths=[1.1], origin="upper")
    ax01.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax01.set_title("STAP score (pre-KA)", fontsize=10)

    # Difference panel underlay: B-mode ROI reference (CFM coordinates).
    ax02.imshow(tube.I_ref_norm, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    diff = np.zeros(mask_flow.shape, dtype=np.uint8)
    diff[base_only] = 1
    diff[stap_only] = 2
    cmap = mpl.colors.ListedColormap([(0, 0, 0, 0.0), mpl.colors.to_rgba(c_base, 0.85), mpl.colors.to_rgba(c_stap, 0.85)])
    ax02.imshow(diff, cmap=cmap, vmin=0, vmax=2, origin="upper", interpolation="nearest")
    ax02.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax02.contour(mask_flow.astype(float), levels=[0.5], colors=[c_stap], linewidths=[1.1], origin="upper")
    ax02.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax02.set_title(f"Decision diff @ FPR={alpha:g}", fontsize=10)
    ax02.legend(
        handles=[Patch(facecolor=c_stap, edgecolor="none", label="STAP only"), Patch(facecolor=c_base, edgecolor="none", label="Baseline only")],
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    # Zoomed maps.
    ax10.imshow(base_log[y0:y1, x0:x1], cmap="viridis", vmin=base_vmin, vmax=base_vmax, origin="upper", interpolation="nearest")
    ax10.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax10.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_stap], linewidths=[1.1], origin="upper")
    ax10.set_title("Baseline (zoom)", fontsize=10)

    ax11.imshow(stap_log[y0:y1, x0:x1], cmap="viridis", vmin=stap_vmin, vmax=stap_vmax, origin="upper", interpolation="nearest")
    ax11.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax11.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_stap], linewidths=[1.1], origin="upper")
    ax11.set_title("STAP (zoom)", fontsize=10)

    ax12.imshow(tube.I_ref_norm[y0:y1, x0:x1], cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    ax12.imshow(diff[y0:y1, x0:x1], cmap=cmap, vmin=0, vmax=2, origin="upper", interpolation="nearest")
    ax12.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax12.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_stap], linewidths=[1.1], origin="upper")
    ax12.set_title("Decision diff (zoom)", fontsize=10)

    fig.suptitle(
        f"Gammex flow phantom (along-linear; frame {frame_idx}) — matched background FPR thresholds",
        fontsize=11,
        y=0.98,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    tpr_base = float(np.mean(score_base[mask_flow] >= float(thr_base)))
    tpr_stap = float(np.mean(score_stap[mask_flow] >= float(thr_stap)))
    print(f"[leading-fig] bundle={bundle_dir} out={out_path}")
    print(f"[leading-fig] base: thr={thr_base:.6g} realized_fpr={fpr_base:.6g}")
    print(f"[leading-fig] stap: thr={thr_stap:.6g} realized_fpr={fpr_stap:.6g}")
    print(f"[leading-fig] base: tpr={tpr_base:.6g}")
    print(f"[leading-fig] stap: tpr={tpr_stap:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
