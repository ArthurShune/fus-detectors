#!/usr/bin/env python3
"""
Generate diagnostic figures for the motion-resilience experiments.
Outputs PNGs under reports/motion_figs/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.ndimage import binary_dilation, binary_erosion
except ImportError:  # pragma: no cover - fallback when SciPy missing
    binary_dilation = None
    binary_erosion = None

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target

BASE_DIR = Path("runs/motion")
OUTPUT_DIR = Path("reports/motion_figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_bundle(bundle_path: Path):
    meta = json.loads((bundle_path / "meta.json").read_text())
    tile_hw = tuple(int(v) for v in meta["tile_hw"])
    stride = int(meta["tile_stride"])
    mask_flow = np.load(bundle_path / "mask_flow.npy").astype(bool)
    mask_bg = np.load(bundle_path / "mask_bg.npy").astype(bool)
    mask_flow_pd_path = bundle_path / "mask_flow_pd.npy"
    mask_flow_pd = np.load(mask_flow_pd_path).astype(bool) if mask_flow_pd_path.exists() else None
    pd_base = np.load(bundle_path / "pd_base.npy")
    pd_stap = np.load(bundle_path / "pd_stap.npy")
    score_map = np.load(bundle_path / "stap_score_map.npy")
    base_score_map = np.load(bundle_path / "base_score_map.npy")
    band_ratio = np.load(bundle_path / "stap_band_ratio_map.npy")
    return {
        "meta": meta,
        "tile_hw": tile_hw,
        "stride": stride,
        "mask_flow": mask_flow,
        "mask_bg": mask_bg,
        "mask_flow_pd": mask_flow_pd,
        "pd_base": pd_base,
        "pd_stap": pd_stap,
        "score_map": score_map,
        "base_score_map": base_score_map,
        "band_ratio": band_ratio,
        "bundle_path": bundle_path,
    }


def _tile_iter(shape: tuple[int, int], tile_hw: tuple[int, int], stride: int):
    H, W = shape
    th, tw = tile_hw
    for y in range(0, H - th + 1, stride):
        for x in range(0, W - tw + 1, stride):
            yield y, x


def compute_tile_coverages(mask_flow: np.ndarray, tile_hw: tuple[int, int], stride: int):
    covs = []
    coords = []
    th, tw = tile_hw
    for y0, x0 in _tile_iter(mask_flow.shape, tile_hw, stride):
        tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
        covs.append(float(tile.mean()))
        coords.append((y0, x0))
    return np.asarray(covs), coords


def coverage_gate(tile_covs, coords, threshold, shape, tile_hw):
    th, tw = tile_hw
    gate = np.zeros(shape, dtype=bool)
    for cov, (y0, x0) in zip(tile_covs, coords):
        if cov >= threshold:
            gate[y0 : y0 + th, x0 : x0 + tw] = True
    return gate


def roc_for_bundle(bundle, thresholds=(0.2, 0.5, 0.8)):
    mask_flow = bundle["mask_flow"]
    mask_bg = bundle["mask_bg"]
    scores = {
        "baseline": bundle["base_score_map"],
        "alias": bundle["score_map"],
    }
    results = {}
    tile_covs, coords = compute_tile_coverages(mask_flow, bundle["tile_hw"], bundle["stride"])
    for thr in thresholds:
        gate = coverage_gate(tile_covs, coords, thr, mask_flow.shape, bundle["tile_hw"])
        flow_mask = mask_flow & gate
        bg_mask = mask_bg & gate
        for name, score_map in scores.items():
            pos = score_map[flow_mask]
            neg = score_map[bg_mask]
            if pos.size == 0 or neg.size == 0:
                continue
            fpr, tpr, _ = roc_curve(pos, neg, num_thresh=4096)
            auc = partial_auc(fpr, tpr, fpr_max=1e-5)
            n_neg = int(neg.size)
            fpr_min = 1.0 / float(n_neg) if n_neg > 0 else 1.0
            fpr_min = float(np.clip(fpr_min, 1e-8, 1.0))
            tpr_empirical = tpr_at_fpr_target(fpr, tpr, target_fpr=fpr_min)
            tpr_at_1e5 = tpr_at_fpr_target(fpr, tpr, target_fpr=1e-5) if fpr_min <= 1e-5 else None
            results.setdefault(thr, {})[name] = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc,
                "tpr_empirical": tpr_empirical,
                "tpr_at_1e5": tpr_at_1e5,
                "fpr_min": fpr_min,
                "n_neg": n_neg,
            }
    return results


def _plot_roc_segment(ax, fpr, tpr, *, label, color, linestyle="-", fpr_max=None):
    """Plot only the low-FPR segment on a log-x axis without pathological points.

    - Drop fpr==0 (undefined on log scale) and any NaNs/inf
    - Clip to a sensible maximum (1e-4) to focus on the operating region
    - Ensure monotonic ordering for matplotlib
    """
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    if fpr_max is None:
        fpr_max = np.nanmax(fpr)
    mask = np.isfinite(fpr) & np.isfinite(tpr) & (fpr > 0.0) & (fpr <= float(fpr_max))
    if not np.any(mask):
        return
    f = fpr[mask]
    t = tpr[mask]
    order = np.argsort(f)
    f = f[order]
    t = t[order]
    ax.plot(f, t, label=label, color=color, linestyle=linestyle)


def _min_empirical_fpr(roc_dict: dict) -> float:
    m = 1.0
    for thr in roc_dict.values():
        for stats in thr.values():
            m = min(m, float(stats.get("fpr_min", 1.0)))
    return m if np.isfinite(m) and m > 0 else 1e-3


def _format_roc_annotation(label: str, stats: dict) -> str:
    fmin = stats.get("fpr_min", 1.0)
    tpr_emp = stats.get("tpr_empirical", 0.0)
    auc = stats.get("auc", 0.0)
    parts = [
        f"{label}: TPR@Fmin({fmin:.2e})={tpr_emp:.3f}",
    ]
    tpr_1e5 = stats.get("tpr_at_1e5")
    if tpr_1e5 is not None:
        parts.append(f"TPR@1e-5={tpr_1e5:.3f}")
    else:
        parts.append("TPR@1e-5=n/a")
    parts.append(f"pAUC={auc:.3e}")
    return ", ".join(parts)


def plot_coverage_roc():
    alias_bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pftrace_bundle = load_bundle(BASE_DIR / "alias_pftrace" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    roc_alias = roc_for_bundle(alias_bundle)
    roc_pftrace = roc_for_bundle(pftrace_bundle)
    thresholds = [0.2, 0.5, 0.8]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fpr_min_resolvable = min(_min_empirical_fpr(roc_alias), _min_empirical_fpr(roc_pftrace))
    for ax, thr in zip(axes, thresholds):
        # Focus on resolvable FPR region; if counts are small, widen xlim
        lo = max(1e-8, fpr_min_resolvable * 0.5)
        hi = max(1e-4, fpr_min_resolvable * 20)
        _plot_roc_segment(
            ax,
            roc_alias[thr]["baseline"]["fpr"],
            roc_alias[thr]["baseline"]["tpr"],
            label="SVD baseline",
            color="gray",
            fpr_max=hi,
        )
        _plot_roc_segment(
            ax,
            roc_alias[thr]["alias"]["fpr"],
            roc_alias[thr]["alias"]["tpr"],
            label="KA alias",
            color="tab:blue",
            fpr_max=hi,
        )
        _plot_roc_segment(
            ax,
            roc_pftrace[thr]["alias"]["fpr"],
            roc_pftrace[thr]["alias"]["tpr"],
            label="KA + Pf-trace",
            color="tab:orange",
            linestyle="--",
            fpr_max=hi,
        )
        for label, stats in [
            ("SVD", roc_alias[thr]["baseline"]),
            ("Alias", roc_alias[thr]["alias"]),
            ("Pf-trace", roc_pftrace[thr]["alias"]),
        ]:
            ax.annotate(
                _format_roc_annotation(label, stats),
                xy=(0.6, 0.15 - 0.05 * ["SVD", "Alias", "Pf-trace"].index(label)),
                xycoords="axes fraction",
                fontsize=8,
            )
        ax.set_title(f"Coverage ≥ {int(thr*100)}%")
        ax.set_xlabel("FPR")
        ax.set_xlim(lo, hi)
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        # Mark the minimum resolvable FPR from sample size
        ax.axvline(fpr_min_resolvable, color="k", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("TPR")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coverage_roc.png", dpi=200)
    plt.close(fig)


def plot_coverage_masks():
    thresholds = [0.2, 0.5, 0.8]
    bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pd_base = bundle["pd_base"]
    pd_log = np.log1p(pd_base)
    vmin, vmax = np.percentile(pd_log, [5, 95])
    tile_covs, coords = compute_tile_coverages(
        bundle["mask_flow"], bundle["tile_hw"], bundle["stride"]
    )
    fig, axes = plt.subplots(1, len(thresholds) + 1, figsize=(14, 4))
    ax0 = axes[0]
    im0 = ax0.imshow(pd_log, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax0.contour(bundle["mask_flow"], levels=[0.5], colors="white", linewidths=0.7)
    ax0.set_title("PD Baseline (log1p)")
    ax0.axis("off")
    plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    for ax, thr in zip(axes[1:], thresholds):
        gate = coverage_gate(tile_covs, coords, thr, bundle["mask_flow"].shape, bundle["tile_hw"])
        cov_frac = gate.mean()
        im = ax.imshow(gate.astype(float), origin="lower", cmap="Greens", vmin=0.0, vmax=1.0)
        ax.contour(bundle["mask_flow"], levels=[0.5], colors="white", linewidths=0.7)
        ax.set_title(f"Coverage ≥ {int(thr*100)}% (frac={cov_frac:.2f})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Coverage gates derived from flow mask overlap")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "coverage_masks.png", dpi=200)
    plt.close(fig)


def plot_pd_maps():
    bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pf_bundle = load_bundle(BASE_DIR / "alias_pftrace" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pd_base = bundle["pd_base"]
    pd_alias = bundle["pd_stap"]
    pd_pf = pf_bundle["pd_stap"]

    # Robust log scaling shared across PD maps
    pd_log = [np.log1p(arr) for arr in (pd_base, pd_alias, pd_pf)]
    pd_vals = np.concatenate([arr.ravel() for arr in pd_log])
    vmin_pd, vmax_pd = np.percentile(pd_vals, [1, 99])

    # Difference scaling (symmetric)
    diff_alias = pd_alias - pd_base
    diff_pf = pd_pf - pd_base
    diff_vals = np.concatenate([diff_alias.ravel(), diff_pf.ravel()])
    diff_lim = np.percentile(np.abs(diff_vals), 99)
    vmin_diff, vmax_diff = -diff_lim, diff_lim

    # Flow overlay: derive from alias PD by thresholding high-energy region
    flow_mask = pd_alias >= np.percentile(pd_alias, 96)

    def show(
        ax, data, title, cmap="viridis", vmin=None, vmax=None, add_flow=False, colorbar=False
    ):
        im = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        if add_flow:
            ax.contour(flow_mask, levels=[0.5], colors="white", linewidths=0.6)
        if colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        return im

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    show(
        axes[0, 0],
        pd_log[0],
        "PD Baseline (log1p)",
        vmin=vmin_pd,
        vmax=vmax_pd,
        add_flow=True,
        colorbar=True,
    )
    show(
        axes[0, 1],
        pd_log[1],
        "PD KA Alias (log1p)",
        vmin=vmin_pd,
        vmax=vmax_pd,
        add_flow=True,
        colorbar=True,
    )
    show(
        axes[0, 2],
        pd_log[2],
        "PD KA + Pf-trace (log1p)",
        vmin=vmin_pd,
        vmax=vmax_pd,
        add_flow=True,
        colorbar=True,
    )
    show(
        axes[1, 0],
        diff_alias,
        "KA Alias − Base",
        cmap="RdBu_r",
        vmin=vmin_diff,
        vmax=vmax_diff,
        add_flow=False,
        colorbar=True,
    )
    show(
        axes[1, 1],
        diff_pf,
        "KA Pf − Base",
        cmap="RdBu_r",
        vmin=vmin_diff,
        vmax=vmax_diff,
        add_flow=False,
        colorbar=True,
    )
    show(
        axes[1, 2],
        1.0 - bundle["band_ratio"],
        "1 − Band Fraction",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        add_flow=False,
        colorbar=True,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pd_maps.png", dpi=200)
    plt.close(fig)


def compute_tile_stats(bundle):
    mask_flow = bundle["mask_flow"]
    mask_bg = bundle["mask_bg"]
    pd_base = bundle["pd_base"]
    pd_stap = bundle["pd_stap"]
    tile_covs, coords = compute_tile_coverages(mask_flow, bundle["tile_hw"], bundle["stride"])
    flow_ratios = []
    bg_ratios = []
    for cov, (y0, x0) in zip(tile_covs, coords):
        th, tw = bundle["tile_hw"]
        flow_tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
        bg_tile = mask_bg[y0 : y0 + th, x0 : x0 + tw]
        if flow_tile.any():
            mu_base = pd_base[y0 : y0 + th, x0 : x0 + tw][flow_tile].mean()
            mu_stap = pd_stap[y0 : y0 + th, x0 : x0 + tw][flow_tile].mean()
            flow_ratios.append(mu_stap / max(mu_base, 1e-8))
        if bg_tile.any():
            var_base = pd_base[y0 : y0 + th, x0 : x0 + tw][bg_tile].var()
            var_stap = pd_stap[y0 : y0 + th, x0 : x0 + tw][bg_tile].var()
            bg_ratios.append((var_stap + 1e-12) / (var_base + 1e-12))
    return np.asarray(tile_covs), np.asarray(flow_ratios), np.asarray(bg_ratios)


def plot_histograms():
    bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    covs, flow_ratios, bg_ratios = compute_tile_stats(bundle)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(covs, bins=20, color="tab:blue", alpha=0.8)
    axes[0].set_title("Tile Flow Coverage")
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Count")
    axes[1].hist(flow_ratios, bins=20, color="tab:orange", alpha=0.8)
    axes[1].set_title("Flow Mean Ratio (KA/Base)")
    axes[1].set_xlabel("Ratio")
    axes[2].hist(bg_ratios, bins=20, color="tab:green", alpha=0.8)
    axes[2].set_title("Background Variance Ratio")
    axes[2].set_xlabel("Ratio")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tile_histograms.png", dpi=200)
    plt.close(fig)


def plot_alias_scatter():
    debug_dir = BASE_DIR / "alias_aliascap" / "pw_7.5MHz_5ang_2ens_128T_seed1" / "stap_debug"
    alias_ratios = []
    scales = []
    for npz_path in debug_dir.glob("*.npz"):
        with np.load(npz_path, allow_pickle=True) as data:
            if "band_fraction_cap_scale" not in data or "psd_flow_alias_ratio" not in data:
                continue
            cap_scale = float(data["band_fraction_cap_scale"])
            alias_ratio = float(data["psd_flow_alias_ratio"])
        if not np.isfinite(cap_scale) or not np.isfinite(alias_ratio):
            continue
        alias_ratios.append(alias_ratio)
        scales.append(cap_scale)
    if not alias_ratios:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(alias_ratios, scales, alpha=0.7)
    ax.set_xlabel("Alias Ratio")
    ax.set_ylabel("Cap Scale")
    ax.set_title("Alias Cap Scale vs Alias Ratio")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "alias_cap_scatter.png", dpi=200)
    plt.close(fig)


def plot_psd_examples():
    debug_dir = BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1" / "stap_debug"
    samples = sorted(debug_dir.glob("*.npz"))[:3]
    if not samples:
        return
    fig, axes = plt.subplots(len(samples), 1, figsize=(6, 3 * len(samples)))
    if len(samples) == 1:
        axes = [axes]
    for ax, npz_path in zip(axes, samples):
        data = np.load(npz_path, allow_pickle=True)
        freqs = data["psd_top_freqs_hz"]
        power = data["psd_top_power"]
        kept = data.get("psd_kept_freqs_hz")
        ax.stem(freqs, power, linefmt="C0-", markerfmt="C0o", basefmt=" ")
        if kept is not None:
            kept = np.atleast_1d(kept)
            kept_power = []
            for f in kept:
                idx = (np.abs(freqs - f)).argmin()
                kept_power.append(power[idx])
            ax.stem(
                kept,
                kept_power,
                linefmt="C1-",
                markerfmt="C1o",
                basefmt=" ",
                label="Kept bins",
            )
        ax.set_title(f"PSD bins ({npz_path.name})")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "psd_bins.png", dpi=200)
    plt.close(fig)


def plot_score_maps():
    bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pf_bundle = load_bundle(BASE_DIR / "alias_pftrace" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    score_base = bundle["base_score_map"]
    score_alias = bundle["score_map"]
    score_pf = pf_bundle["score_map"]
    all_vals = np.concatenate([score_base.ravel(), score_alias.ravel(), score_pf.ravel()])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    diff_alias = score_alias - score_base
    diff_pf = score_pf - score_base
    diff_vals = np.concatenate([diff_alias.ravel(), diff_pf.ravel()])
    diff_lim = np.percentile(np.abs(diff_vals), 99)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, arr, title in zip(
        axes[0],
        [score_base, score_alias, score_pf],
        ["Score Baseline", "Score KA Alias", "Score KA + Pf-trace"],
    ):
        im = ax.imshow(arr, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    for ax, arr, title in zip(
        axes[1],
        [diff_alias, diff_pf, score_pf - score_alias],
        ["Alias − Base (score)", "Pf − Base (score)", "Pf − Alias (score)"],
    ):
        im = ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_maps.png", dpi=200)
    plt.close(fig)


def build_pd_flow_mask(
    pd_map: np.ndarray,
    *,
    quantile: float = 0.97,
    depth_min_frac: float = 0.15,
    depth_max_frac: float = 0.9,
    erode_iters: int = 1,
    dilate_iters: int = 2,
    base_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask = np.zeros_like(pd_map, dtype=bool)
    finite = np.isfinite(pd_map)
    if not finite.any():
        return base_mask.copy() if base_mask is not None else mask
    valid = pd_map[finite]
    q = float(np.clip(quantile, 0.0, 1.0))
    thresh = np.quantile(valid, q)
    mask = (pd_map >= thresh) & finite
    H = pd_map.shape[0]
    y0 = int(np.clip(depth_min_frac * H, 0, H))
    y1 = int(np.clip(depth_max_frac * H, y0 + 1, H))
    depth_gate = np.zeros_like(mask, dtype=bool)
    depth_gate[y0:y1, :] = True
    mask &= depth_gate
    if erode_iters > 0 and binary_erosion is not None:
        mask = binary_erosion(mask, iterations=erode_iters)
    if dilate_iters > 0 and binary_dilation is not None:
        mask = binary_dilation(mask, iterations=dilate_iters)
    if base_mask is not None:
        mask = mask | base_mask
    return mask


def _percent_change_map(
    pd_base: np.ndarray,
    pd_new: np.ndarray,
    mask: np.ndarray,
    *,
    use_log: bool = False,
    denom_pct: float = 5.0,
) -> np.ndarray:
    change = np.zeros_like(pd_base, dtype=np.float32)
    if not mask.any():
        return change
    if use_log:
        base = np.log1p(np.clip(pd_base, 0.0, None))
        new = np.log1p(np.clip(pd_new, 0.0, None))
        change[mask] = new[mask] - base[mask]
        return change
    valid = pd_base[mask]
    floor = np.percentile(valid, denom_pct) if valid.size else 0.0
    denom = np.maximum(pd_base, max(floor, 1e-6))
    change[mask] = (pd_new[mask] - pd_base[mask]) / denom[mask]
    return change


def plot_pd_percent_changes():
    bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pf_bundle = load_bundle(BASE_DIR / "alias_pftrace" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pd_base = bundle["pd_base"]
    pd_alias = bundle["pd_stap"]
    pd_pf = pf_bundle["pd_stap"]
    mask_flow = bundle.get("mask_flow_pd")
    if mask_flow is None:
        mask_flow = build_pd_flow_mask(
            pd_alias,
            quantile=0.97,
            depth_min_frac=0.18,
            depth_max_frac=0.85,
            erode_iters=1,
            dilate_iters=2,
            base_mask=bundle["mask_flow"],
        )
    mask_bg = bundle["mask_bg"]

    alias_flow = _percent_change_map(pd_base, pd_alias, mask_flow, use_log=True)
    pf_flow = _percent_change_map(pd_base, pd_pf, mask_flow, use_log=True)
    alias_bg = _percent_change_map(pd_base, pd_alias, mask_bg, use_log=True)
    pf_bg = _percent_change_map(pd_base, pd_pf, mask_bg, use_log=True)

    flow_vals = np.concatenate([alias_flow[mask_flow], pf_flow[mask_flow]])
    bg_vals = np.concatenate([alias_bg[mask_bg], pf_bg[mask_bg]])
    lim_flow = np.percentile(np.abs(flow_vals), 99) if flow_vals.size else 0.01
    lim_bg = np.percentile(np.abs(bg_vals), 99) if bg_vals.size else 0.01

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, data, title, lim in [
        (axes[0, 0], alias_flow, "Alias flow %Δ", lim_flow),
        (axes[0, 1], pf_flow, "Pf-trace flow %Δ", lim_flow),
        (axes[1, 0], alias_bg, "Alias bg %Δ", lim_bg),
        (axes[1, 1], pf_bg, "Pf-trace bg %Δ", lim_bg),
    ]:
        im = ax.imshow(data, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("log1p Δ vs baseline (PD-derived flow mask)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pd_percent_change.png", dpi=200)
    plt.close(fig)


def plot_score_maps():
    bundle = load_bundle(BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    pf_bundle = load_bundle(BASE_DIR / "alias_pftrace" / "pw_7.5MHz_5ang_2ens_128T_seed1")
    score_base = bundle["base_score_map"]
    score_alias = bundle["score_map"]
    score_pf = pf_bundle["score_map"]

    all_vals = np.concatenate([score_base.ravel(), score_alias.ravel(), score_pf.ravel()])
    vmin, vmax = np.percentile(all_vals, [1, 99])
    diff_alias = score_alias - score_base
    diff_pf = score_pf - score_base
    diff_vals = np.concatenate([diff_alias.ravel(), diff_pf.ravel()])
    diff_lim = np.percentile(np.abs(diff_vals), 99)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, score, title in zip(
        axes[0],
        [score_base, score_alias, score_pf],
        ["Score Baseline", "Score KA Alias", "Score KA + Pf-trace"],
    ):
        im = ax.imshow(score, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    for ax, diff, title in zip(
        axes[1],
        [diff_alias, diff_pf, score_pf - score_alias],
        ["Alias − Base (score)", "Pf − Base (score)", "Pf − Alias (score)"],
    ):
        im = ax.imshow(diff, origin="lower", cmap="RdBu_r", vmin=-diff_lim, vmax=diff_lim)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "score_maps.png", dpi=200)
    plt.close(fig)


def plot_pftrace_bars():
    pf_meta = json.loads(
        (
            BASE_DIR
            / "alias_pftrace_stride3_lt4_shrink"
            / "pw_7.5MHz_5ang_2ens_128T_seed1"
            / "meta.json"
        ).read_text()
    )
    alias_meta = json.loads(
        (
            BASE_DIR / "alias_stride3_lt4_shrink" / "pw_7.5MHz_5ang_2ens_128T_seed1" / "meta.json"
        ).read_text()
    )
    fields = ["ka_median_retain_f_total", "ka_median_shrink_perp_total"]
    labels = ["Retain f (total)", "Shrink perp (total)"]
    alias_vals = [alias_meta["stap_fallback_telemetry"].get(f) for f in fields]
    pf_vals = [pf_meta["stap_fallback_telemetry"].get(f) for f in fields]
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(fields))
    width = 0.35
    ax.bar(x - width / 2, alias_vals, width, label="Alias")
    ax.bar(x + width / 2, pf_vals, width, label="Pf-trace")
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, label="Target retain=0.8")
    ax.axhline(y=0.6, color="gray", linestyle=":", linewidth=1, label="Target shrink=0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Value")
    ax.set_title("Pf-trace directional metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pftrace_directional.png", dpi=200)
    plt.close(fig)


def plot_motion_fraction():
    bundle_dir = BASE_DIR / "alias" / "pw_7.5MHz_5ang_2ens_128T_seed1"
    meta = json.loads((bundle_dir / "meta.json").read_text())
    tele = meta.get("stap_fallback_telemetry", {})
    debug_dir = bundle_dir / "stap_debug"
    tiles = []
    for npz_path in debug_dir.glob("*.npz"):
        with np.load(npz_path, allow_pickle=True) as data:
            tile = data.get("motion_fraction_tile")
        if tile is not None:
            arr = np.asarray(tile, dtype=np.float32)
            if arr.size > 0 and np.isfinite(arr).any():
                tiles.append(arr)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if tiles:
        first = tiles[0]
        im = axes[0].imshow(first, origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        axes[0].set_title("Motion fraction (sample tile)")
        axes[0].axis("off")
        plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        hist_data = np.concatenate([t.ravel() for t in tiles])
        hist_data = hist_data[np.isfinite(hist_data)]
        axes[1].hist(hist_data, bins=30, color="tab:purple", alpha=0.8)
        axes[1].set_xlabel("Motion fraction")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Motion fraction distribution (debug tiles)")
    else:
        axes[0].axis("off")
        axes[0].text(
            0.5,
            0.5,
            "No motion_fraction_tile records were saved\n(motion basis disabled for this run).",
            ha="center",
            va="center",
            fontsize=12,
        )
        axes[1].axis("off")
    stats_text = []
    med = tele.get("median_motion_fraction")
    p90 = tele.get("p90_motion_fraction")
    stats_text.append(
        f"median_motion_fraction: {med:.3f}"
        if isinstance(med, (int, float))
        else "median_motion_fraction: n/a"
    )
    stats_text.append(
        f"p90_motion_fraction: {p90:.3f}"
        if isinstance(p90, (int, float))
        else "p90_motion_fraction: n/a"
    )
    axes[1].text(0.5, 0.2, "\n".join(stats_text), ha="center", va="center", fontsize=11)
    fig.suptitle("Motion fraction diagnostics")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "motion_fraction.png", dpi=200)
    plt.close(fig)


def main():
    plot_coverage_roc()
    plot_coverage_masks()
    plot_pd_maps()
    plot_pd_percent_changes()
    plot_score_maps()
    plot_histograms()
    plot_alias_scatter()
    plot_psd_examples()
    plot_pftrace_bars()
    plot_motion_fraction()
    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
