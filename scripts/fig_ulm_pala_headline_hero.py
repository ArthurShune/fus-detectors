#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from pipeline.realdata.ulm_zenodo_7883227 import load_ulm_block_iq
from scripts.ulm7883227_structural_roc import (
    _compute_reference_map,
    _derive_structural_masks,
    _evaluate_window_score,
    _load_bundle_scores,
)
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


ROOT = Path(__file__).resolve().parents[1]


def _variant_display_name(variant: str) -> str:
    mapping = {
        "msd_ratio": "Whitened matched-subspace specialist",
        "whitened_power": "Whitened-power specialist",
        "adaptive_guard": "Adaptive guard specialist",
        "unwhitened_ratio": "Fixed matched-subspace head",
    }
    return mapping.get(str(variant), str(variant).replace("_", " ").title())


def _normalize(x: np.ndarray, *, qlo: float = 0.02, qhi: float = 0.995) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.quantile(vals, qlo))
    hi = float(np.quantile(vals, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def _crop_bbox(*masks: np.ndarray, pad: int = 8) -> tuple[int, int, int, int]:
    union = np.zeros_like(np.asarray(masks[0], dtype=bool))
    for m in masks:
        union |= np.asarray(m, dtype=bool)
    yy, xx = np.where(union)
    if yy.size == 0 or xx.size == 0:
        h, w = union.shape
        return 0, h, 0, w
    y0 = max(0, int(yy.min()) - int(pad))
    y1 = min(int(union.shape[0]), int(yy.max()) + 1 + int(pad))
    x0 = max(0, int(xx.min()) - int(pad))
    x1 = min(int(union.shape[1]), int(xx.max()) + 1 + int(pad))
    return y0, y1, x0, x1


def _score_panel(
    ax,
    *,
    bg_img: np.ndarray,
    support_mask: np.ndarray,
    core_mask: np.ndarray,
    shell_mask: np.ndarray,
    det_mask: np.ndarray,
    color: str,
    title: str,
    auc: float,
    fpr70: float,
) -> None:
    import matplotlib.pyplot as plt

    bg = np.ma.masked_where(~np.asarray(support_mask, dtype=bool), bg_img)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(0.98, 0.98, 0.98, 1.0))
    ax.imshow(bg, cmap=cmap, interpolation="nearest")
    ax.contour(support_mask.astype(np.uint8), levels=[0.5], colors=["#f8fafc"], linewidths=1.0)
    ax.contour(shell_mask.astype(np.uint8), levels=[0.5], colors=["#ffb000"], linewidths=1.3, linestyles="--")
    ax.contour(core_mask.astype(np.uint8), levels=[0.5], colors=["#00d5ff"], linewidths=1.5)

    overlay = np.zeros((*det_mask.shape, 4), dtype=np.float32)
    rgb = np.array(
        [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)],
        dtype=np.float32,
    ) / 255.0
    overlay[..., :3] = rgb[None, None, :]
    overlay[..., 3] = 0.68 * (det_mask.astype(np.float32) * support_mask.astype(np.float32))
    ax.imshow(overlay, interpolation="nearest")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    if np.isfinite(float(auc)) and np.isfinite(float(fpr70)):
        ax.text(
            0.5,
            -0.10,
            f"AUC {auc:.3f}   |   shell FPR @ 70% core recall {fpr70:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.2,
            color="#0f172a",
        )
    ax.set_xticks([])
    ax.set_yticks([])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate the headline PALA-backed real in-vivo ULM hero figure."
    )
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "ulm_zenodo_7883227")
    ap.add_argument("--pala-example-root", type=Path, default=Path("/tmp/PALA_repo_1073521"))
    ap.add_argument("--cache-root", type=Path, default=ROOT / "tmp" / "ulm7883227_structural_roc_cache")
    ap.add_argument("--block-id", type=int, default=1)
    ap.add_argument("--window-start", type=int, default=128)
    ap.add_argument("--window-frames", type=int, default=64)
    ap.add_argument("--lt", type=int, default=64)
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--svd-energy-frac", type=float, default=0.975)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument("--cov-estimator", type=str, default="scm")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--prf-hz", type=float, default=1000.0)
    ap.add_argument("--flow-low-hz", type=float, default=10.0)
    ap.add_argument("--flow-high-hz", type=float, default=150.0)
    ap.add_argument("--alias-center-hz", type=float, default=350.0)
    ap.add_argument("--alias-width-hz", type=float, default=150.0)
    ap.add_argument("--vessel-quantile", type=float, default=0.92)
    ap.add_argument("--background-quantile", type=float, default=0.50)
    ap.add_argument("--guard-dilate-iters", type=int, default=3)
    ap.add_argument("--edge-margin", type=int, default=4)
    ap.add_argument("--peak-size", type=int, default=3)
    ap.add_argument("--peak-dilate-iters", type=int, default=1)
    ap.add_argument("--shell-inner-dilate-iters", type=int, default=4)
    ap.add_argument("--shell-outer-dilate-iters", type=int, default=10)
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_pala_structural_roc.json",
    )
    ap.add_argument(
        "--latency-json",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_pala_latency_rtx4080.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figs" / "paper" / "ulm7883227_pala_headline_hero.pdf",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import scienceplots  # noqa: F401
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib required: {exc}") from exc

    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.titleweight": "bold",
            "axes.titlesize": 10.5,
            "font.size": 9,
        }
    )

    summary = json.loads(Path(args.summary_json).read_text())
    specialist_variant = str(summary["frozen_profile"]["stap_detector_variant"])
    specialist_name = _variant_display_name(specialist_variant)

    ref_map, _ = _compute_reference_map(
        int(args.block_id),
        root=Path(args.data_root),
        cache_dir=Path(args.cache_root),
        reg_enable=False,
        reg_subpixel=4,
        svd_energy_frac=float(args.svd_energy_frac),
        device=str(args.stap_device),
        mode="pala_example_matout",
        local_density_quantile=0.9995,
        local_density_peak_size=3,
        local_density_sigma=1.0,
        pala_example_root=Path(args.pala_example_root),
        pala_powdop_blocks=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
        pala_svd_cutoff_start=5,
        pala_trim_sr_border=1,
    )
    support_path = _.get("support_mask_path")
    support_mask = (
        np.load(support_path, allow_pickle=False).astype(bool)
        if support_path
        else np.isfinite(ref_map)
    )
    mask_flow, mask_bg, _ = _derive_structural_masks(
        ref_map,
        support_mask=support_mask,
        vessel_quantile=float(args.vessel_quantile),
        bg_quantile=float(args.background_quantile),
        erode_iters=1,
        guard_dilate_iters=int(args.guard_dilate_iters),
        edge_margin=int(args.edge_margin),
        vessel_mask_mode="peaks",
        peak_size=int(args.peak_size),
        peak_dilate_iters=int(args.peak_dilate_iters),
        background_mask_mode="shell",
        shell_inner_dilate_iters=int(args.shell_inner_dilate_iters),
        shell_outer_dilate_iters=int(args.shell_outer_dilate_iters),
    )

    start = int(args.window_start)
    stop = int(args.window_start) + int(args.window_frames)
    cube = load_ulm_block_iq(int(args.block_id), frames=slice(start, stop), root=Path(args.data_root))
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as td:
        paths = write_acceptance_bundle_from_icube(
            out_root=Path(td),
            dataset_name=f"ulm_hero_block{int(args.block_id):03d}_{start:04d}_{stop:04d}",
            Icube=cube,
            prf_hz=float(args.prf_hz),
            tile_hw=(int(args.tile_h), int(args.tile_w)),
            tile_stride=int(args.tile_stride),
            Lt=min(int(args.lt), int(args.window_frames) - 1),
            diag_load=float(args.diag_load),
            cov_estimator=str(args.cov_estimator),
            stap_device=str(args.stap_device),
            baseline_type="mc_svd",
            reg_enable=False,
            reg_subpixel=4,
            svd_energy_frac=float(args.svd_energy_frac),
            run_stap=True,
            stap_detector_variant="whitened_power",
            score_ka_v2_enable=False,
            stap_conditional_enable=False,
            flow_mask_mode="pd_auto",
            mask_flow_override=mask_flow,
            mask_bg_override=mask_bg,
            band_ratio_flow_low_hz=float(args.flow_low_hz),
            band_ratio_flow_high_hz=float(args.flow_high_hz),
            band_ratio_alias_center_hz=float(args.alias_center_hz),
            band_ratio_alias_width_hz=float(args.alias_width_hz),
        )
        scores = _load_bundle_scores(Path(paths["meta"]).parent, stap_variant="whitened_power", include_postka=False)

    pd_metrics = _evaluate_window_score(scores["pd"], mask_flow, mask_bg, fprs=[1e-3, 1e-2, 1e-1], tpr_targets=[0.7])
    wp_metrics = _evaluate_window_score(scores["matched_subspace"], mask_flow, mask_bg, fprs=[1e-3, 1e-2, 1e-1], tpr_targets=[0.7])
    tau_pd = float(pd_metrics["thr_tpr@70"])
    tau_wp = float(wp_metrics["thr_tpr@70"])
    det_pd = np.asarray(scores["pd"] >= tau_pd, dtype=bool)
    det_wp = np.asarray(scores["matched_subspace"] >= tau_wp, dtype=bool)

    y0, y1, x0, x1 = _crop_bbox(mask_flow, mask_bg, support_mask, pad=6)
    ref_crop = _normalize(ref_map[y0:y1, x0:x1])
    support_crop = support_mask[y0:y1, x0:x1]
    flow_crop = mask_flow[y0:y1, x0:x1]
    bg_crop = mask_bg[y0:y1, x0:x1]
    pd_crop = det_pd[y0:y1, x0:x1]
    wp_crop = det_wp[y0:y1, x0:x1]

    fig = plt.figure(figsize=(11.4, 3.95))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        wspace=0.18,
    )

    ax_ref = fig.add_subplot(gs[0, 0])
    ax_pd = fig.add_subplot(gs[0, 1])
    ax_wp = fig.add_subplot(gs[0, 2])

    ref_disp = np.ma.masked_where(~support_crop, ref_crop)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color=(0.98, 0.98, 0.98, 1.0))
    ax_ref.imshow(ref_disp, cmap=cmap, interpolation="nearest")
    ax_ref.contour(support_crop.astype(np.uint8), levels=[0.5], colors=["#f8fafc"], linewidths=1.0)
    ax_ref.contour(bg_crop.astype(np.uint8), levels=[0.5], colors=["#ffb000"], linewidths=1.5, linestyles="--")
    ax_ref.contour(flow_crop.astype(np.uint8), levels=[0.5], colors=["#00d5ff"], linewidths=1.8)
    ax_ref.set_title("External PALA localization reference\nregistered to the IQ grid", pad=8)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    ax_ref.set_anchor("C")

    _score_panel(
        ax_pd,
        bg_img=ref_crop,
        support_mask=support_crop,
        core_mask=flow_crop,
        shell_mask=bg_crop,
        det_mask=pd_crop,
        color="#ff6b57",
        title="Baseline power Doppler\nmatched 70% core recall",
        auc=float("nan"),
        fpr70=float("nan"),
    )
    ax_pd.set_anchor("C")
    _score_panel(
        ax_wp,
        bg_img=ref_crop,
        support_mask=support_crop,
        core_mask=flow_crop,
        shell_mask=bg_crop,
        det_mask=wp_crop,
        color="#4fd1c5",
        title=f"{specialist_name}\nmatched 70% core recall",
        auc=float("nan"),
        fpr70=float("nan"),
    )
    ax_wp.set_anchor("C")

    for ax, label in zip((ax_ref, ax_pd, ax_wp), ("a", "b", "c"), strict=True):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="#0f172a",
        )

    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.09)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.out.suffix.lower() == ".pdf":
        fig.savefig(args.out.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
