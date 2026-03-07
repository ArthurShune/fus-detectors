#!/usr/bin/env python3
"""
Macé/Urban atlas-alignment sanity overlay figure.

Why this exists
---------------
Several Macé analyses in this repository rely on mapping Allen atlas volumes into
the scan grid using the dataset-provided affine (Transformation.mat). A
skeptical reviewer can reasonably ask for a *visual sanity check* that the
alignment is correct, since misregistration would invalidate any atlas-derived
offline summaries.

This script writes a compact, paper-friendly figure that shows (for 2-3
representative planes):
  - mean log-PD image (scan space),
  - the same mean log-PD with outlines of a few atlas ROIs (scan space),
  - the same mean log-PD with contours of atlas.Vascular (scan space).

By default the ROIs match the offline Macé H1/H0 shorthand used elsewhere:
  - positives: VIS/SC/LGd (green)
  - negatives: CP/CA/DG (red)

Usage
-----
    PYTHONPATH=. python scripts/mace_atlas_overlay_fig.py \
      --scan-name scan_anatomy \
      --plane-indices 5 10 15 \
      --out-png figs/paper/mace_atlas_overlay.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np

from pipeline.realdata import mace_data_root
from pipeline.realdata.mace_wholebrain import (
    build_mace_transform_matrix,
    load_all_mace_scans,
    load_mace_anatomy_scan,
    load_mace_atlas,
    load_mace_region_info,
    load_mace_transform,
    scan_plane_to_atlas_indices,
)


def _default_pos_prefixes() -> List[str]:
    return ["VIS", "SC", "LGd"]


def _default_neg_prefixes() -> List[str]:
    return ["CP", "CA", "DG"]


def _labels_for_prefixes(acronyms: Sequence[str], label_for_acr: dict[str, int], prefixes: Sequence[str]) -> List[int]:
    labels: List[int] = []
    for acr in acronyms:
        for p in prefixes:
            if acr.startswith(p):
                if acr in label_for_acr:
                    labels.append(int(label_for_acr[acr]))
                break
    # Deduplicate, preserve order.
    seen: dict[int, None] = {}
    uniq: List[int] = []
    for l in labels:
        if l not in seen:
            seen[l] = None
            uniq.append(l)
    return uniq


def _quantile_range(img: np.ndarray, mask: np.ndarray | None, qlo: float, qhi: float) -> tuple[float, float]:
    a = np.asarray(img, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape == a.shape and np.any(m):
            a = a[m]
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(a, float(qlo)))
    hi = float(np.quantile(a, float(qhi)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo = float(np.nanmin(a))
        hi = float(np.nanmax(a))
    return lo, hi


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate Macé atlas-alignment sanity overlay figure.")
    ap.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory containing whole-brain-fUS (defaults to data/whole-brain-fUS).",
    )
    ap.add_argument(
        "--scan-name",
        type=str,
        default="scan_anatomy",
        help="Scan name to visualize (e.g., scan_anatomy, scan1, scan3, scan4).",
    )
    ap.add_argument(
        "--plane-indices",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Plane indices (0-based) to include (2-3 recommended).",
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("figs/paper/mace_atlas_overlay.png"),
        help="Output PNG path (paper figure).",
    )
    ap.add_argument("--also-pdf", action="store_true", help="Also write a matching PDF next to the PNG.")
    ap.add_argument(
        "--eps-pd",
        type=float,
        default=1e-8,
        help="Epsilon inside log(meanPD + eps) for display.",
    )
    ap.add_argument(
        "--pos-prefix",
        action="append",
        default=None,
        help="Acronym prefix for positive ROI outlines (can be given multiple times).",
    )
    ap.add_argument(
        "--neg-prefix",
        action="append",
        default=None,
        help="Acronym prefix for negative ROI outlines (can be given multiple times).",
    )
    ap.add_argument(
        "--vascular-levels",
        type=float,
        nargs="+",
        default=[0.5],
        help="atlas.Vascular contour levels to draw.",
    )
    ap.add_argument(
        "--vmin-q",
        type=float,
        default=0.02,
        help="Lower quantile for display scaling (computed within atlas-brain pixels).",
    )
    ap.add_argument(
        "--vmax-q",
        type=float,
        default=0.98,
        help="Upper quantile for display scaling (computed within atlas-brain pixels).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_root = mace_data_root() if args.data_root is None else Path(args.data_root)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Load scan + atlas + transform.
    scan_name = str(args.scan_name)
    if scan_name in {"scan_anatomy", "anatomy"}:
        scan = load_mace_anatomy_scan(data_root)
    else:
        scans = load_all_mace_scans(data_root)
        scan = next((s for s in scans if str(s.name) == scan_name), None)
        if scan is None:
            avail = ", ".join([s.name for s in scans] + ["scan_anatomy"])
            raise SystemExit(f"Scan '{args.scan_name}' not found. Available: {avail}")

    atlas = load_mace_atlas(data_root)
    region_info = load_mace_region_info(data_root)
    transf = load_mace_transform(data_root)
    A, t = build_mace_transform_matrix(transf)

    pos_prefixes = args.pos_prefix or _default_pos_prefixes()
    neg_prefixes = args.neg_prefix or _default_neg_prefixes()
    pos_labels = _labels_for_prefixes(region_info.acronyms, dict(region_info.label_for_acr), pos_prefixes)
    neg_labels = _labels_for_prefixes(region_info.acronyms, dict(region_info.label_for_acr), neg_prefixes)
    if not pos_labels or not neg_labels:
        raise SystemExit("Empty ROI label sets; adjust --pos-prefix/--neg-prefix.")

    plane_indices = [int(p) for p in args.plane_indices]
    if not plane_indices:
        raise SystemExit("No --plane-indices provided.")
    for p in plane_indices:
        if p < 0 or p >= int(scan.n_planes):
            raise SystemExit(f"plane index {p} out of range for scan {scan.name} (Z={scan.n_planes})")

    Ha, Wa, Za = atlas.regions.shape

    # Deferred matplotlib import (keeps CLI snappy).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 9,
            "axes.titlesize": 10,
        }
    )

    nrows = len(plane_indices)
    fig, axes = plt.subplots(nrows, 3, figsize=(11.0, 3.3 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)  # (1, 3)

    col_titles = ["Mean log-PD", "Mean log-PD + atlas ROI outlines", "Mean log-PD + atlas.Vascular contours"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title)

    for r, plane_idx in enumerate(plane_indices):
        pd_T_HW = scan.pd[:, :, :, plane_idx]
        mean_pd = pd_T_HW.mean(axis=0).astype(np.float64, copy=False)
        img = np.log(mean_pd + float(args.eps_pd))
        H, W = img.shape

        # Map scan pixels to atlas indices; restrict to atlas in-bounds + brain (Regions != 0).
        i_dv, i_ap, i_lr, inside = scan_plane_to_atlas_indices(
            H, W, int(plane_idx), A, t, (Ha, Wa, Za)
        )
        inside = inside.astype(bool, copy=False)
        inside_idx = np.nonzero(inside)[0]

        regions_flat = np.zeros((H * W,), dtype=atlas.regions.dtype)
        if inside_idx.size > 0:
            regions_flat[inside_idx] = atlas.regions[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]]
        regions_scan = regions_flat.reshape(H, W)
        brain_mask = regions_scan != 0

        pos_mask = np.isin(regions_scan, np.asarray(pos_labels, dtype=regions_scan.dtype))
        neg_mask = np.isin(regions_scan, np.asarray(neg_labels, dtype=regions_scan.dtype))
        neg_mask = neg_mask & ~pos_mask

        vascular_flat = np.full((H * W,), np.nan, dtype=np.float32)
        if inside_idx.size > 0:
            vascular_flat[inside_idx] = atlas.vascular[i_dv[inside_idx], i_ap[inside_idx], i_lr[inside_idx]].astype(
                np.float32, copy=False
            )
        # Exclude atlas background from the vascular overlay.
        vascular_flat[regions_flat == 0] = np.nan
        vascular_scan = vascular_flat.reshape(H, W)

        vmin, vmax = _quantile_range(img, brain_mask, float(args.vmin_q), float(args.vmax_q))

        # Row label: plane index and, if present, the dataset-reported physical plane location.
        plane_mm = None
        try:
            if scan.planes_mm is not None and int(plane_idx) < int(scan.planes_mm.shape[0]):
                plane_mm = float(scan.planes_mm[int(plane_idx)])
        except Exception:
            plane_mm = None
        row_label = f"plane {plane_idx}" if plane_mm is None else f"plane {plane_idx} ({plane_mm:.2f} mm)"

        # Column 1: mean PD only.
        ax = axes[r, 0]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_ylabel(row_label)
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 2: ROI outlines.
        ax = axes[r, 1]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        try:
            ax.contour(brain_mask.astype(float), levels=[0.5], colors="white", linewidths=0.5, alpha=0.6)
        except Exception:
            pass
        if np.any(pos_mask):
            ax.contour(pos_mask.astype(float), levels=[0.5], colors="#2ca02c", linewidths=1.0, alpha=0.9)
        if np.any(neg_mask):
            ax.contour(neg_mask.astype(float), levels=[0.5], colors="#d62728", linewidths=1.0, alpha=0.9)
        ax.text(
            0.01,
            0.99,
            f"POS: {','.join(pos_prefixes)}\nNEG: {','.join(neg_prefixes)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Column 3: vascular contours.
        ax = axes[r, 2]
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        levels = [float(x) for x in args.vascular_levels]
        levels = [x for x in levels if np.isfinite(x)]
        if levels:
            try:
                ax.contour(vascular_scan, levels=levels, colors="#17becf", linewidths=1.0, alpha=0.95)
            except Exception:
                pass
        ax.text(
            0.01,
            0.99,
            f"atlas.Vascular levels: {','.join([f'{x:g}' for x in levels])}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    if bool(args.also_pdf):
        out_pdf = out_png.with_suffix(".pdf")
        fig.savefig(out_pdf)

    print(f"[mace-atlas-overlay] wrote {out_png}")


if __name__ == "__main__":
    main()
