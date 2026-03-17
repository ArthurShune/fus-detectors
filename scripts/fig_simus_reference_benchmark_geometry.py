#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseSpec:
    title: str
    dataset_dir: Path


DEFAULT_CASES = [
    CaseSpec(
        title="Mobile held-out seed 127",
        dataset_dir=ROOT / "runs" / "sim" / "simus_v2_acceptance_clin_mobile_pf_v2_phase3_seed127" / "dataset",
    ),
    CaseSpec(
        title="Intra-op held-out seed 127",
        dataset_dir=ROOT
        / "runs"
        / "sim"
        / "simus_v2_acceptance_clin_intraop_parenchyma_pf_v3_phase3_seed127"
        / "dataset",
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate the SIMUS held-out benchmark-geometry figure used to make the "
            "positive, background-calibration, and nuisance regions concrete in the main text."
        )
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "figs" / "paper" / "simus_reference_benchmark_geometry.pdf",
        help="Output PDF path.",
    )
    ap.add_argument("--dpi", type=int, default=250)
    return ap.parse_args()


def _power_underlay(dataset_dir: Path) -> tuple[np.ndarray, float, float]:
    icube = np.load(dataset_dir / "icube.npy", mmap_mode="r")
    power = np.sum(np.abs(np.asarray(icube, dtype=np.complex64)) ** 2, axis=0, dtype=np.float64)
    img = np.log10(np.clip(power, 0.0, None) + 1e-6)
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return np.zeros_like(img, dtype=np.float32), 0.0, 1.0
    vmin = float(np.quantile(vals, 0.02))
    vmax = float(np.quantile(vals, 0.995))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return img.astype(np.float32, copy=False), vmin, vmax


def _load_mask(dataset_dir: Path, name: str) -> np.ndarray:
    return np.load(dataset_dir / name, allow_pickle=False).astype(bool, copy=False)


def _overlay_rgba(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    rgb = np.array(
        [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)],
        dtype=np.float32,
    ) / 255.0
    out = np.zeros((*mask.shape, 4), dtype=np.float32)
    out[..., :3] = rgb[None, None, :]
    out[..., 3] = alpha * mask.astype(np.float32)
    return out


def main() -> int:
    args = parse_args()

    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "nature", "no-latex"])
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.titlesize": 10,
        }
    )

    c_pos = "#22c55e"
    c_bg = "#38bdf8"
    c_nuis = "#ef4444"

    fig, axes = plt.subplots(1, 2, figsize=(8.7, 4.2), constrained_layout=True)

    for ax, case in zip(axes, DEFAULT_CASES):
        img, vmin, vmax = _power_underlay(case.dataset_dir)
        mask_h1 = _load_mask(case.dataset_dir, "mask_h1_pf_main.npy")
        mask_h0_bg = _load_mask(case.dataset_dir, "mask_h0_bg.npy")
        mask_h0_nuis = _load_mask(case.dataset_dir, "mask_h0_nuisance_pa.npy")

        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
        ax.imshow(_overlay_rgba(mask_h1, c_pos, 0.10), origin="upper", interpolation="nearest")
        ax.imshow(_overlay_rgba(mask_h0_nuis, c_nuis, 0.10), origin="upper", interpolation="nearest")
        ax.contour(mask_h0_bg.astype(float), levels=[0.5], colors=[c_bg], linewidths=[0.9], linestyles=["--"], origin="upper")
        ax.contour(mask_h0_nuis.astype(float), levels=[0.5], colors=[c_nuis], linewidths=[1.3], origin="upper")
        ax.contour(mask_h1.astype(float), levels=[0.5], colors=[c_pos], linewidths=[1.3], origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            (
                f"{case.title}\n"
                f"H1={int(mask_h1.sum())}, H0 bg={int(mask_h0_bg.sum())}, "
                f"H0 nuis={int(mask_h0_nuis.sum())}"
            ),
            pad=8,
        )

    legend_handles = [
        Line2D([0], [0], color=c_pos, lw=1.6, label="Positive flow mask"),
        Line2D([0], [0], color=c_bg, lw=1.2, ls="--", label="Background calibration mask"),
        Line2D([0], [0], color=c_nuis, lw=1.6, label="Nuisance mask"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[fig-simus-reference-benchmark-geometry] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
