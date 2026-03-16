#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_RATIO_BUNDLE = Path(
    "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast/"
    "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine/"
    "frame000"
)
DEFAULT_POWER_BUNDLE = Path(
    "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_whitened_power/"
    "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine/"
    "frame000"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate the leading Gammex structural-fidelity figure from frozen audited bundle outputs. "
            "The figure compares baseline, matched-subspace, and whitened-power score maps on the same "
            "along-view phantom frame."
        )
    )
    ap.add_argument(
        "--ratio-bundle",
        type=Path,
        default=DEFAULT_RATIO_BUNDLE,
        help="Frame-level bundle directory for the matched-subspace run.",
    )
    ap.add_argument(
        "--power-bundle",
        type=Path,
        default=DEFAULT_POWER_BUNDLE,
        help="Frame-level bundle directory for the whitened-power run.",
    )
    ap.add_argument(
        "--underlay",
        choices=["pd_base", "score_base", "none"],
        default="pd_base",
        help="Background image used under the decision-difference panel.",
    )
    ap.add_argument(
        "--fpr",
        type=float,
        default=1e-2,
        help="Background FPR target used to threshold each method.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figs/paper/leading_structural_fidelity_gammex.pdf"),
        help="Output PDF path.",
    )
    ap.add_argument("--dpi", type=int, default=250, help="Figure DPI.")
    return ap.parse_args()


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float, float]:
    bg = np.asarray(bg_scores, dtype=np.float64).ravel()
    bg = bg[np.isfinite(bg)]
    if bg.size == 0:
        raise ValueError("Empty background score pool.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        return float("inf"), 0.0
    if alpha >= 1.0:
        tau = float(np.min(bg))
        return tau, 1.0
    k = int(np.ceil(float(alpha) * int(bg.size)))
    k = max(1, min(k, int(bg.size)))
    tau = float(np.partition(bg, int(bg.size) - k)[int(bg.size) - k])
    return tau, float(np.mean(bg >= tau))


def _robust_log_image(x: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, float, float]:
    xx = np.log10(np.clip(np.asarray(x, dtype=np.float64), 0.0, None) + float(eps))
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
    return (
        max(0, y0 - int(pad)),
        min(int(mask.shape[0]), y1 + int(pad)),
        max(0, x0 - int(pad)),
        min(int(mask.shape[1]), x1 + int(pad)),
    )


def _load_required(bundle_dir: Path, name: str, *, dtype=None) -> np.ndarray:
    path = bundle_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required bundle artifact: {path}")
    arr = np.load(path, allow_pickle=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def main() -> int:
    args = parse_args()

    ratio_bundle = Path(args.ratio_bundle)
    power_bundle = Path(args.power_bundle)
    if not ratio_bundle.is_dir():
        raise SystemExit(f"Missing ratio bundle dir: {ratio_bundle}")
    if not power_bundle.is_dir():
        raise SystemExit(f"Missing whitened-power bundle dir: {power_bundle}")

    score_base = _load_required(ratio_bundle, "score_base.npy", dtype=np.float64)
    score_ratio = _load_required(ratio_bundle, "score_stap_preka.npy", dtype=np.float64)
    score_power = _load_required(power_bundle, "score_stap_preka.npy", dtype=np.float64)
    mask_flow = _load_required(ratio_bundle, "mask_flow.npy", dtype=bool)
    mask_bg = _load_required(ratio_bundle, "mask_bg.npy", dtype=bool)

    if (
        score_base.shape != score_ratio.shape
        or score_base.shape != score_power.shape
        or score_base.shape != mask_flow.shape
        or score_base.shape != mask_bg.shape
    ):
        raise SystemExit(
            "Shape mismatch among loaded Gammex artifacts: "
            f"base={score_base.shape}, ratio={score_ratio.shape}, power={score_power.shape}, "
            f"flow={mask_flow.shape}, bg={mask_bg.shape}"
        )

    underlay_map: np.ndarray | None
    if args.underlay == "pd_base":
        pd_path = ratio_bundle / "pd_base.npy"
        underlay_map = np.load(pd_path, allow_pickle=False).astype(np.float64, copy=False) if pd_path.is_file() else score_base
    elif args.underlay == "score_base":
        underlay_map = score_base
    else:
        underlay_map = None

    alpha = float(args.fpr)
    thr_base, fpr_base = _right_tail_threshold(score_base[mask_bg], alpha)
    thr_ratio, fpr_ratio = _right_tail_threshold(score_ratio[mask_bg], alpha)
    thr_power, fpr_power = _right_tail_threshold(score_power[mask_bg], alpha)

    det_base = score_base >= thr_base
    det_power = score_power >= thr_power
    base_only = det_base & (~det_power)
    power_only = det_power & (~det_base)

    import matplotlib as mpl

    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    base_log, base_vmin, base_vmax = _robust_log_image(score_base, eps=1e-9)
    ratio_log, ratio_vmin, ratio_vmax = _robust_log_image(score_ratio, eps=1e-14)
    power_log, power_vmin, power_vmax = _robust_log_image(score_power, eps=1e-14)
    if underlay_map is None:
        underlay_log = None
        underlay_vmin, underlay_vmax = 0.0, 1.0
    else:
        underlay_log, underlay_vmin, underlay_vmax = _robust_log_image(underlay_map, eps=1e-9)

    y0, y1, x0, x1 = _bbox_from_mask(mask_flow, pad=10)

    dpi = int(args.dpi)
    fig = plt.figure(figsize=(13.0, 6.4), dpi=dpi)
    gs = fig.add_gridspec(2, 4, wspace=0.04, hspace=0.08)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    for ax in axes:
        ax.set_axis_off()
    ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13 = axes

    c_ratio = "#0072B2"
    c_power = "#009E73"
    c_base = "#D55E00"

    ax00.imshow(base_log, cmap="viridis", vmin=base_vmin, vmax=base_vmax, origin="upper", interpolation="nearest")
    ax00.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax00.contour(mask_flow.astype(float), levels=[0.5], colors=[c_ratio], linewidths=[1.1], origin="upper")
    ax00.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax00.set_title("Baseline score", fontsize=10)

    ax01.imshow(ratio_log, cmap="viridis", vmin=ratio_vmin, vmax=ratio_vmax, origin="upper", interpolation="nearest")
    ax01.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax01.contour(mask_flow.astype(float), levels=[0.5], colors=[c_ratio], linewidths=[1.1], origin="upper")
    ax01.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax01.set_title("Matched-subspace score", fontsize=10)

    ax02.imshow(power_log, cmap="viridis", vmin=power_vmin, vmax=power_vmax, origin="upper", interpolation="nearest")
    ax02.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax02.contour(mask_flow.astype(float), levels=[0.5], colors=[c_power], linewidths=[1.1], origin="upper")
    ax02.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax02.set_title("Whitened-power score", fontsize=10)

    if underlay_log is None:
        ax03.imshow(np.zeros_like(score_base), cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    else:
        ax03.imshow(underlay_log, cmap="gray", vmin=underlay_vmin, vmax=underlay_vmax, origin="upper", interpolation="nearest")
    diff = np.zeros(mask_flow.shape, dtype=np.uint8)
    diff[base_only] = 1
    diff[power_only] = 2
    diff_cmap = mpl.colors.ListedColormap(
        [(0, 0, 0, 0.0), mpl.colors.to_rgba(c_base, 0.85), mpl.colors.to_rgba(c_power, 0.85)]
    )
    ax03.imshow(diff, cmap=diff_cmap, vmin=0, vmax=2, origin="upper", interpolation="nearest")
    ax03.contour(mask_flow.astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax03.contour(mask_flow.astype(float), levels=[0.5], colors=[c_power], linewidths=[1.1], origin="upper")
    ax03.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=1.4))
    ax03.set_title(f"Whitened-power vs baseline @ FPR={alpha:g}", fontsize=10)
    ax03.legend(
        handles=[
            Patch(facecolor=c_power, edgecolor="none", label="Whitened-power only"),
            Patch(facecolor=c_base, edgecolor="none", label="Baseline only"),
        ],
        loc="lower right",
        frameon=True,
        fontsize=8,
    )

    ax10.imshow(base_log[y0:y1, x0:x1], cmap="viridis", vmin=base_vmin, vmax=base_vmax, origin="upper", interpolation="nearest")
    ax10.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax10.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_ratio], linewidths=[1.1], origin="upper")
    ax10.set_title("Baseline (zoom)", fontsize=10)

    ax11.imshow(ratio_log[y0:y1, x0:x1], cmap="viridis", vmin=ratio_vmin, vmax=ratio_vmax, origin="upper", interpolation="nearest")
    ax11.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax11.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_ratio], linewidths=[1.1], origin="upper")
    ax11.set_title("Matched-subspace (zoom)", fontsize=10)

    ax12.imshow(power_log[y0:y1, x0:x1], cmap="viridis", vmin=power_vmin, vmax=power_vmax, origin="upper", interpolation="nearest")
    ax12.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax12.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_power], linewidths=[1.1], origin="upper")
    ax12.set_title("Whitened-power (zoom)", fontsize=10)

    if underlay_log is None:
        ax13.imshow(np.zeros_like(score_base[y0:y1, x0:x1]), cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
    else:
        ax13.imshow(
            underlay_log[y0:y1, x0:x1],
            cmap="gray",
            vmin=underlay_vmin,
            vmax=underlay_vmax,
            origin="upper",
            interpolation="nearest",
        )
    ax13.imshow(diff[y0:y1, x0:x1], cmap=diff_cmap, vmin=0, vmax=2, origin="upper", interpolation="nearest")
    ax13.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=["black"], linewidths=[1.6], origin="upper")
    ax13.contour(mask_flow[y0:y1, x0:x1].astype(float), levels=[0.5], colors=[c_power], linewidths=[1.1], origin="upper")
    ax13.set_title("Decision diff (zoom)", fontsize=10)

    fig.subplots_adjust(left=0.015, right=0.995, top=0.975, bottom=0.035)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    tpr_base = float(np.mean(score_base[mask_flow] >= thr_base))
    tpr_ratio = float(np.mean(score_ratio[mask_flow] >= thr_ratio))
    tpr_power = float(np.mean(score_power[mask_flow] >= thr_power))
    print(f"[leading-fig] ratio_bundle={ratio_bundle}")
    print(f"[leading-fig] power_bundle={power_bundle}")
    print(f"[leading-fig] out={args.out}")
    print(f"[leading-fig] base: thr={thr_base:.6g} realized_fpr={fpr_base:.6g} tpr={tpr_base:.6g}")
    print(f"[leading-fig] ratio: thr={thr_ratio:.6g} realized_fpr={fpr_ratio:.6g} tpr={tpr_ratio:.6g}")
    print(f"[leading-fig] power: thr={thr_power:.6g} realized_fpr={fpr_power:.6g} tpr={tpr_power:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
