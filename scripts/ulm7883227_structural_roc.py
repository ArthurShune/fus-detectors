#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.realdata.ulm_zenodo_7883227 import (
    list_ulm_blocks,
    load_ulm_block_iq,
    load_ulm_zenodo_7883227_params,
)
from sim.kwave.common import _baseline_pd_mcsvd
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube



def _parse_blocks(spec: str | None, *, root: Path) -> list[int]:
    raw = (spec or "").strip().lower()
    if raw in {"", "all", ":"}:
        return list_ulm_blocks(root)
    out: list[int] = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            step = 1 if hi >= lo else -1
            out.extend(list(range(lo, hi + step, step)))
        else:
            out.append(int(part))
    vals = sorted({int(v) for v in out})
    if not vals:
        raise ValueError("expected at least one block id")
    return vals


def _window_starts(total_frames: int, window_frames: int, stride: int) -> list[int]:
    if window_frames <= 0 or stride <= 0:
        raise ValueError("window_frames and stride must be positive")
    if window_frames > total_frames:
        return [0]
    starts = list(range(0, total_frames - window_frames + 1, stride))
    tail = total_frames - window_frames
    if starts[-1] != tail:
        starts.append(tail)
    return starts


def _finite(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos_f = _finite(pos)
    neg_f = _finite(neg)
    if pos_f.size == 0 or neg_f.size == 0:
        return None
    neg_sorted = np.sort(neg_f)
    less = np.searchsorted(neg_sorted, pos_f, side="left")
    right = np.searchsorted(neg_sorted, pos_f, side="right")
    equal = right - less
    return float((float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos_f.size * neg_f.size))


def _threshold_from_neg(neg: np.ndarray, fpr: float) -> tuple[float | None, float | None]:
    neg_f = _finite(neg)
    if neg_f.size == 0:
        return None, None
    q = float(np.clip(1.0 - float(fpr), 0.0, 1.0))
    thr = float(np.quantile(neg_f, q))
    fpr_emp = float(np.mean(neg_f >= thr))
    return thr, fpr_emp


def _bootstrap_ci(
    xs: list[float | None],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | None]:
    arr = np.asarray([x for x in xs if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"center": None, "lo": None, "hi": None, "n": 0}
    center = float(np.mean(arr))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.mean(arr[idx], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"center": center, "lo": float(lo), "hi": float(hi), "n": int(arr.size)}


def _fmt_ci(sec: dict[str, Any], *, digits: int = 3) -> str:
    c = sec.get("center")
    lo = sec.get("lo")
    hi = sec.get("hi")
    if c is None:
        return "--"
    return f"{float(c):.{digits}f} [{float(lo):.{digits}f},{float(hi):.{digits}f}]"


def _safe_quantile(x: np.ndarray, q: float) -> float:
    vals = _finite(x)
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, float(np.clip(q, 0.0, 1.0))))


def _connected_components(binary: np.ndarray) -> int:
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    _, n = ndi.label(np.asarray(binary, dtype=bool), structure=structure)
    return int(n)


def _derive_structural_masks(
    reference_pd: np.ndarray,
    *,
    vessel_quantile: float,
    bg_quantile: float,
    erode_iters: int,
    guard_dilate_iters: int,
    edge_margin: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ref = np.asarray(reference_pd, dtype=np.float64)
    finite = np.isfinite(ref)
    if not finite.any():
        raise ValueError("reference map has no finite values")
    vessel_thr = _safe_quantile(ref[finite], vessel_quantile)
    positive_vals = ref[finite & (ref > 0.0)]
    if positive_vals.size and vessel_thr <= 0.0 and (positive_vals.size / max(1, int(finite.sum()))) < 0.5:
        vessel_thr = _safe_quantile(positive_vals, vessel_quantile)
    vessel = (ref >= vessel_thr) & finite
    if erode_iters > 0:
        vessel = ndi.binary_erosion(vessel, iterations=int(erode_iters))
    vessel = ndi.binary_opening(vessel, iterations=1)

    bg_thr = _safe_quantile(ref[finite], bg_quantile)
    bg_candidate = finite.copy()
    if guard_dilate_iters > 0:
        vessel_guard = ndi.binary_dilation(vessel, iterations=int(guard_dilate_iters))
    else:
        vessel_guard = vessel.copy()
    bg_candidate &= ~vessel_guard
    if edge_margin > 0:
        bg_candidate[:edge_margin, :] = False
        bg_candidate[-edge_margin:, :] = False
        bg_candidate[:, :edge_margin] = False
        bg_candidate[:, -edge_margin:] = False
        vessel[:edge_margin, :] = False
        vessel[-edge_margin:, :] = False
        vessel[:, :edge_margin] = False
        vessel[:, -edge_margin:] = False
    bg = bg_candidate & (ref <= bg_thr)
    if int(bg.sum()) < 1000:
        candidate_vals = ref[bg_candidate]
        if candidate_vals.size:
            bg_thr = _safe_quantile(candidate_vals, min(max(float(bg_quantile), 0.5), 0.8))
            bg = bg_candidate & (ref <= bg_thr)
    if int(bg.sum()) < 1000:
        bg = bg_candidate.copy()
    qc = {
        "vessel_threshold": float(vessel_thr),
        "background_threshold": float(bg_thr),
        "n_vessel": int(vessel.sum()),
        "n_background": int(bg.sum()),
        "guard_dilate_iters": int(guard_dilate_iters),
        "edge_margin_px": int(edge_margin),
        "vessel_components": int(_connected_components(vessel)),
        "background_components": int(_connected_components(bg)),
        "fpr_floor_background": float(1.0 / max(1, int(bg.sum()))),
    }
    return vessel.astype(bool), bg.astype(bool), qc


def _compute_reference_pd_map(
    block_id: int,
    *,
    root: Path,
    cache_dir: Path,
    reg_enable: bool,
    reg_subpixel: int,
    svd_energy_frac: float,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    map_path = cache_dir / f"reference_pd_block{int(block_id):03d}.npy"
    tele_path = cache_dir / f"reference_pd_block{int(block_id):03d}.json"
    if map_path.is_file() and tele_path.is_file():
        return (
            np.load(map_path, allow_pickle=False).astype(np.float32, copy=False),
            json.loads(tele_path.read_text()),
        )
    iq = load_ulm_block_iq(int(block_id), frames=None, root=root)
    pd_map, tele = _baseline_pd_mcsvd(
        iq,
        reg_enable=bool(reg_enable),
        reg_subpixel=max(1, int(reg_subpixel)),
        svd_rank=None,
        svd_energy_frac=float(svd_energy_frac),
        device=str(device),
        return_filtered_cube=False,
    )
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(map_path, pd_map.astype(np.float32, copy=False), allow_pickle=False)
    tele_payload = {
        "block_id": int(block_id),
        "shape": list(pd_map.shape),
        "reg_enable": bool(reg_enable),
        "reg_subpixel": int(reg_subpixel),
        "svd_energy_frac": float(svd_energy_frac),
        "device": str(device),
        "telemetry": tele,
    }
    tele_path.write_text(json.dumps(tele_payload, indent=2, sort_keys=True))
    return pd_map.astype(np.float32, copy=False), tele_payload


def _local_density_map(
    filtered_cube: np.ndarray,
    *,
    quantile: float,
    peak_size: int = 3,
    gaussian_sigma: float = 1.0,
) -> np.ndarray:
    power = (np.abs(np.asarray(filtered_cube, dtype=np.complex64)) ** 2).astype(np.float32, copy=False)
    thr = _safe_quantile(power, quantile)
    peaks = power >= float(thr)
    peaks &= power == ndi.maximum_filter(power, size=(1, int(peak_size), int(peak_size)), mode="nearest")
    peaks &= power > ndi.median_filter(power, size=(1, int(peak_size), int(peak_size)), mode="nearest")
    counts = peaks.sum(axis=0).astype(np.float32)
    if counts.max() > 0:
        counts /= float(counts.max())
    if gaussian_sigma > 0:
        counts = ndi.gaussian_filter(counts, sigma=float(gaussian_sigma)).astype(np.float32, copy=False)
    if counts.max() > 0:
        counts /= float(counts.max())
    return counts.astype(np.float32, copy=False)


def _compute_reference_map(
    block_id: int,
    *,
    root: Path,
    cache_dir: Path,
    reg_enable: bool,
    reg_subpixel: int,
    svd_energy_frac: float,
    device: str,
    mode: str,
    local_density_quantile: float,
    local_density_peak_size: int,
    local_density_sigma: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    mode_norm = str(mode).strip().lower()
    if mode_norm == "pd":
        return _compute_reference_pd_map(
            block_id,
            root=root,
            cache_dir=cache_dir / "pd",
            reg_enable=reg_enable,
            reg_subpixel=reg_subpixel,
            svd_energy_frac=svd_energy_frac,
            device=device,
        )
    if mode_norm != "local_density":
        raise ValueError(f"unsupported reference mode {mode!r}")
    cache_dir = cache_dir / "local_density"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = (
        f"q{int(round(10000.0 * float(local_density_quantile))):05d}"
        f"_p{int(local_density_peak_size)}_g{str(float(local_density_sigma)).replace('.', 'p')}"
    )
    map_path = cache_dir / f"reference_local_density_{tag}_block{int(block_id):03d}.npy"
    tele_path = cache_dir / f"reference_local_density_{tag}_block{int(block_id):03d}.json"
    if map_path.is_file() and tele_path.is_file():
        return (
            np.load(map_path, allow_pickle=False).astype(np.float32, copy=False),
            json.loads(tele_path.read_text()),
        )
    iq = load_ulm_block_iq(int(block_id), frames=None, root=root)
    _, tele, cube = _baseline_pd_mcsvd(
        iq,
        reg_enable=bool(reg_enable),
        reg_subpixel=max(1, int(reg_subpixel)),
        svd_rank=None,
        svd_energy_frac=float(svd_energy_frac),
        device=str(device),
        return_filtered_cube=True,
    )
    ref_map = _local_density_map(
        cube,
        quantile=float(local_density_quantile),
        peak_size=max(3, int(local_density_peak_size)),
        gaussian_sigma=float(local_density_sigma),
    )
    np.save(map_path, ref_map.astype(np.float32, copy=False), allow_pickle=False)
    tele_payload = {
        "block_id": int(block_id),
        "shape": list(ref_map.shape),
        "reference_mode": "local_density",
        "local_density_quantile": float(local_density_quantile),
        "local_density_peak_size": int(local_density_peak_size),
        "local_density_sigma": float(local_density_sigma),
        "reg_enable": bool(reg_enable),
        "reg_subpixel": int(reg_subpixel),
        "svd_energy_frac": float(svd_energy_frac),
        "device": str(device),
        "telemetry": tele,
    }
    tele_path.write_text(json.dumps(tele_payload, indent=2, sort_keys=True))
    return ref_map.astype(np.float32, copy=False), tele_payload


def _evaluate_window_score(
    score: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    *,
    fprs: list[float],
) -> dict[str, Any]:
    pos = np.asarray(score, dtype=np.float64)[np.asarray(mask_flow, dtype=bool)]
    neg = np.asarray(score, dtype=np.float64)[np.asarray(mask_bg, dtype=bool)]
    out: dict[str, Any] = {
        "auc": _auc_pos_vs_neg(pos, neg),
        "n_pos": int(np.asarray(mask_flow, dtype=bool).sum()),
        "n_neg": int(np.asarray(mask_bg, dtype=bool).sum()),
    }
    for fpr in fprs:
        tag = f"{float(fpr):.0e}"
        thr, realized = _threshold_from_neg(neg, fpr)
        if thr is None:
            out[f"tpr@{tag}"] = None
            out[f"fpr@{tag}"] = None
            out[f"thr@{tag}"] = None
        else:
            out[f"tpr@{tag}"] = float(np.mean(pos >= thr)) if pos.size else None
            out[f"fpr@{tag}"] = realized
            out[f"thr@{tag}"] = float(thr)
    return out


def _stap_variant_label(variant: str) -> str:
    raw = str(variant).strip().lower()
    if raw == "unwhitened_ratio":
        return "Matched-subspace detector (fixed head)"
    if raw == "adaptive_guard":
        return "Matched-subspace detector (adaptive head)"
    return "Matched-subspace detector (whitened specialist)"


def _score_specs(stap_variant: str = "msd_ratio") -> list[tuple[str, str, str]]:
    return [
        ("pd", "score_base.npy", "Baseline (power Doppler)"),
        ("log_pd", "score_base_pdlog.npy", "Baseline (log-power Doppler)"),
        ("kasai", "score_base_kasai.npy", "Baseline (Kasai lag-1 magnitude)"),
        ("matched_subspace", "score_stap_preka.npy", _stap_variant_label(stap_variant)),
    ]


def _load_bundle_scores(bundle_dir: Path, *, stap_variant: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, filename, _ in _score_specs(stap_variant):
        path = bundle_dir / filename
        if path.is_file():
            out[key] = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    return out


def _make_mask_figure(
    *,
    ref_map: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    out_path: Path,
    title_prefix: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib required for mask figure: {exc}") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax in axes:
        im = ax.imshow(np.asarray(ref_map, dtype=np.float64), cmap="magma")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].contour(mask_flow.astype(np.uint8), levels=[0.5], colors=["#00c2ff"], linewidths=1.4)
    axes[0].set_title(f"{title_prefix}: vessel mask")
    axes[1].contour(mask_bg.astype(np.uint8), levels=[0.5], colors=["#ff9f1c"], linewidths=1.2, linestyles="--")
    axes[1].set_title(f"{title_prefix}: background mask")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _bootstrap_curve(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    n_boot: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    vals = np.asarray(
        [float(r[metric_key]) for r in rows if r.get(metric_key) is not None and np.isfinite(r[metric_key])],
        dtype=np.float64,
    )
    if vals.size == 0:
        return None, None, None
    center = float(np.mean(vals))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boots = np.mean(vals[idx], axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return center, float(lo), float(hi)


def _make_roc_figure(
    *,
    per_window_rows: list[dict[str, Any]],
    fprs_plot: list[float],
    out_path: Path,
    n_boot: int,
    seed: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"matplotlib required for ROC figure: {exc}") from exc

    styles = {
        "pd": dict(color="#666666", linewidth=1.8),
        "log_pd": dict(color="#999999", linewidth=1.2, linestyle="--"),
        "kasai": dict(color="#ff7f0e", linewidth=1.6),
        "matched_subspace": dict(color="#1f77b4", linewidth=2.0),
    }
    labels = {k: lbl for k, _, lbl in _score_specs(str(((per_window_rows[0] if per_window_rows else {}).get("stap_detector_variant")) or "msd_ratio"))}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for score_idx, (score_key, _, _) in enumerate(_score_specs(str(((per_window_rows[0] if per_window_rows else {}).get("stap_detector_variant")) or "msd_ratio"))):
        score_rows = [r for r in per_window_rows if r["score_key"] == score_key]
        xs: list[float] = []
        ys: list[float] = []
        ylo: list[float] = []
        yhi: list[float] = []
        for i, fpr in enumerate(fprs_plot):
            tag = f"{float(fpr):.0e}"
            center, lo, hi = _bootstrap_curve(
                score_rows,
                metric_key=f"tpr@{tag}",
                n_boot=n_boot,
                seed=int(seed) + 17 * i + 97 * (score_idx + 1),
            )
            realized, _, _ = _bootstrap_curve(
                score_rows,
                metric_key=f"fpr@{tag}",
                n_boot=n_boot,
                seed=int(seed) + 313 * i + 131 * (score_idx + 1),
            )
            if center is None or realized is None:
                continue
            xs.append(float(realized))
            ys.append(float(center))
            ylo.append(float(lo))
            yhi.append(float(hi))
        if not xs:
            continue
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        lo = np.asarray(ylo, dtype=np.float64)
        hi = np.asarray(yhi, dtype=np.float64)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        lo = lo[order]
        hi = hi[order]
        st = styles[score_key]
        ax.plot(x, y, label=labels[score_key], **st)
        if score_key in {"matched_subspace", "pd"}:
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.12, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(min(fprs_plot) * 0.8, max(fprs_plot) * 1.2)
    ax.set_ylim(-0.01, 1.0)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    if out_path.suffix.lower() == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), dpi=250, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _write_summary_table(
    *,
    summary: dict[str, Any],
    out_path: Path,
    fprs: list[float],
) -> None:
    stap_variant = str(((summary.get("frozen_profile") or {}).get("stap_detector_variant")) or "msd_ratio")
    score_order = [k for k, _, _ in _score_specs(stap_variant)]
    score_labels = {k: lbl for k, _, lbl in _score_specs(stap_variant)}
    col_tags = [f"{float(fpr):.0e}" for fpr in fprs]
    ref_mode = str(((summary.get("reference_design") or {}).get("reference_mode")) or "local_density")
    if ref_mode == "local_density":
        ref_text = (
            "a leave-one-block-out full-acquisition microbubble-density reference obtained by "
            "accumulating sparse local maxima on MC--SVD-filtered long-horizon residuals from the remaining blocks"
        )
    else:
        ref_text = (
            "a leave-one-block-out full-acquisition vascular reference built from MC--SVD-filtered "
            "long-integration power Doppler on the remaining blocks"
        )
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/ulm7883227_structural_roc.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\hline")
    hdr = "Score & AUC (H1 vs H0)"
    for tag in col_tags:
        exponent = int(round(math.log10(float(tag))))
        power = f"10^{{{exponent}}}"
        hdr += f" & TPR@$\\alpha={power}$"
    hdr += " \\\\"
    lines.append(hdr)
    lines.append("\\hline")
    for score_key in score_order:
        sec = summary["scores"].get(score_key)
        if not sec:
            continue
        row = [score_labels[score_key], _fmt_ci(sec["auc"], digits=3)]
        for tag in col_tags:
            row.append(_fmt_ci(sec[f"tpr@{tag}"], digits=3))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Structural-label ROC on real in vivo rat-brain IQ from ULM Zenodo 7883227. "
        f"For each evaluation block, vessel and background masks are derived once from {ref_text}; "
        "short 128-frame windows from the held-out block are then scored on the "
        "same MC--SVD residual cube using different downstream detector heads. Entries report window-level "
        "means with 95\\% bootstrap CIs over windows. Reproducibility details are provided in \\SuppOrApp{app:repro}.}"
    )
    lines.append("\\label{tab:ulm7883227_structural_roc}")
    lines.append("\\end{table}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Structural-label ROC on ULM Zenodo 7883227 real in vivo brain IQ using a "
            "leave-one-block-out long-integration vascular reference."
        )
    )
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "ulm_zenodo_7883227")
    ap.add_argument("--cache-root", type=Path, default=ROOT / "tmp" / "ulm7883227_structural_roc_cache")
    ap.add_argument("--bundle-tmp-root", type=Path, default=ROOT / "tmp" / "ulm7883227_structural_roc_bundles")
    ap.add_argument("--ref-blocks", type=str, default="all")
    ap.add_argument("--eval-blocks", type=str, default="all")
    ap.add_argument("--window-frames", type=int, default=128)
    ap.add_argument("--window-stride", type=int, default=128)
    ap.add_argument("--max-windows-per-block", type=int, default=None)
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--lt", type=int, default=64)
    ap.add_argument("--prf-hz", type=float, default=None)
    ap.add_argument("--svd-energy-frac", type=float, default=0.975)
    ap.add_argument(
        "--stap-detector-variant",
        type=str,
        default="msd_ratio",
        choices=["msd_ratio", "unwhitened_ratio", "adaptive_guard"],
    )
    ap.add_argument("--reference-mode", type=str, default="local_density", choices=["local_density", "pd"])
    ap.add_argument("--reference-local-density-quantile", type=float, default=0.9995)
    ap.add_argument("--reference-local-density-peak-size", type=int, default=3)
    ap.add_argument("--reference-local-density-sigma", type=float, default=1.0)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument("--cov-estimator", type=str, default="scm")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--reg-enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reg-subpixel", type=int, default=4)
    ap.add_argument("--vessel-quantile", type=float, default=0.92)
    ap.add_argument("--background-quantile", type=float, default=0.50)
    ap.add_argument("--mask-erode-iters", type=int, default=1)
    ap.add_argument("--guard-dilate-iters", type=int, default=3)
    ap.add_argument("--edge-margin", type=int, default=4)
    ap.add_argument("--fprs", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3])
    ap.add_argument(
        "--fprs-plot",
        type=float,
        nargs="+",
        default=[1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1],
    )
    ap.add_argument("--bootstrap-n", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=1337)
    ap.add_argument("--keep-bundles", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--force-reference", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--out-root", type=Path, default=ROOT / "runs" / "real" / "ulm7883227_structural_roc")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_structural_roc.csv",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_structural_roc.json",
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_structural_roc_table.tex",
    )
    ap.add_argument(
        "--out-mask-fig",
        type=Path,
        default=ROOT / "figs" / "paper" / "ulm7883227_structural_masks.pdf",
    )
    ap.add_argument(
        "--out-roc-fig",
        type=Path,
        default=ROOT / "figs" / "paper" / "ulm7883227_structural_roc_curves.pdf",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    params = load_ulm_zenodo_7883227_params(args.data_root)
    prf_hz = float(args.prf_hz) if args.prf_hz is not None else float(params.frame_rate_hz)
    ref_blocks = _parse_blocks(args.ref_blocks, root=args.data_root)
    eval_blocks = _parse_blocks(args.eval_blocks, root=args.data_root)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.bundle_tmp_root.mkdir(parents=True, exist_ok=True)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_mask_fig.parent.mkdir(parents=True, exist_ok=True)
    args.out_roc_fig.parent.mkdir(parents=True, exist_ok=True)

    if bool(args.force_reference):
        shutil.rmtree(args.cache_root, ignore_errors=True)
        args.cache_root.mkdir(parents=True, exist_ok=True)

    ref_maps: dict[int, np.ndarray] = {}
    ref_tele: dict[int, dict[str, Any]] = {}
    for block_id in ref_blocks:
        ref_map, tele = _compute_reference_map(
            int(block_id),
            root=args.data_root,
            cache_dir=args.cache_root,
            reg_enable=bool(args.reg_enable),
            reg_subpixel=int(args.reg_subpixel),
            svd_energy_frac=float(args.svd_energy_frac),
            device=str(args.stap_device),
            mode=str(args.reference_mode),
            local_density_quantile=float(args.reference_local_density_quantile),
            local_density_peak_size=int(args.reference_local_density_peak_size),
            local_density_sigma=float(args.reference_local_density_sigma),
        )
        ref_maps[int(block_id)] = ref_map
        ref_tele[int(block_id)] = tele

    representative_block = int(eval_blocks[0])
    representative_ref = None
    representative_flow = None
    representative_bg = None

    per_window_rows: list[dict[str, Any]] = []
    bundle_cleanup: list[Path] = []
    mask_qc: dict[int, dict[str, Any]] = {}
    reference_manifest: dict[int, dict[str, Any]] = {}

    for block_id in eval_blocks:
        loo_ids = [bid for bid in ref_blocks if int(bid) != int(block_id)]
        if not loo_ids:
            loo_ids = list(ref_blocks)
        loo_stack = np.stack([ref_maps[int(bid)] for bid in loo_ids], axis=0)
        ref_map = np.mean(loo_stack, axis=0).astype(np.float32, copy=False)
        mask_flow, mask_bg, qc = _derive_structural_masks(
            ref_map,
            vessel_quantile=float(args.vessel_quantile),
            bg_quantile=float(args.background_quantile),
            erode_iters=int(args.mask_erode_iters),
            guard_dilate_iters=int(args.guard_dilate_iters),
            edge_margin=int(args.edge_margin),
        )
        if int(mask_bg.sum()) < 1000:
            raise RuntimeError(f"block {block_id}: background mask too small for 1e-3 tail ({int(mask_bg.sum())} pixels)")
        mask_qc[int(block_id)] = qc
        reference_manifest[int(block_id)] = {
            "reference_blocks": [int(b) for b in loo_ids],
            "mask_qc": qc,
        }
        if int(block_id) == representative_block:
            representative_ref = ref_map.copy()
            representative_flow = mask_flow.copy()
            representative_bg = mask_bg.copy()

        iq_full = load_ulm_block_iq(int(block_id), frames=None, root=args.data_root)
        starts = _window_starts(int(iq_full.shape[0]), int(args.window_frames), int(args.window_stride))
        if args.max_windows_per_block is not None:
            starts = starts[: int(args.max_windows_per_block)]
        for win_idx, start in enumerate(starts):
            stop = int(start) + int(args.window_frames)
            cube = iq_full[int(start) : int(stop)]
            dataset_name = f"ulm7883227_struct_block{int(block_id):03d}_{int(start):04d}_{int(stop):04d}"
            bundle_root = args.bundle_tmp_root if bool(args.keep_bundles) else Path(
                tempfile.mkdtemp(prefix="ulm7883227_struct_", dir=args.bundle_tmp_root)
            )
            if not bool(args.keep_bundles):
                bundle_cleanup.append(bundle_root)
            paths = write_acceptance_bundle_from_icube(
                out_root=bundle_root,
                dataset_name=dataset_name,
                Icube=cube,
                prf_hz=float(prf_hz),
                tile_hw=(int(args.tile_h), int(args.tile_w)),
                tile_stride=int(args.tile_stride),
                Lt=int(args.lt),
                diag_load=float(args.diag_load),
                cov_estimator=str(args.cov_estimator),
                stap_device=str(args.stap_device),
                baseline_type="mc_svd",
                reg_enable=bool(args.reg_enable),
                reg_subpixel=int(args.reg_subpixel),
                svd_energy_frac=float(args.svd_energy_frac),
                run_stap=True,
                stap_detector_variant=str(args.stap_detector_variant),
                score_ka_v2_enable=False,
                stap_conditional_enable=False,
                flow_mask_mode="pd_auto",
                mask_flow_override=mask_flow,
                mask_bg_override=mask_bg,
                band_ratio_flow_low_hz=10.0,
                band_ratio_flow_high_hz=150.0,
                band_ratio_alias_center_hz=350.0,
                band_ratio_alias_width_hz=150.0,
                meta_extra={
                    "ulm_structural_roc": {
                        "evaluation_block_id": int(block_id),
                        "window_start": int(start),
                        "window_stop": int(stop),
                        "reference_block_ids": [int(b) for b in loo_ids],
                    }
                },
            )
            bundle_dir = Path(paths["meta"]).parent
            scores = _load_bundle_scores(bundle_dir, stap_variant=str(args.stap_detector_variant))
            for score_key, _, score_label in _score_specs(str(args.stap_detector_variant)):
                score = scores.get(score_key)
                if score is None:
                    continue
                metrics = _evaluate_window_score(
                    score,
                    mask_flow,
                    mask_bg,
                    fprs=[float(f) for f in args.fprs_plot],
                )
                row = {
                    "block_id": int(block_id),
                    "window_index": int(win_idx),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "score_key": str(score_key),
                    "score_label": str(score_label),
                    "stap_detector_variant": str(args.stap_detector_variant),
                    "n_vessel": int(mask_flow.sum()),
                    "n_background": int(mask_bg.sum()),
                    **metrics,
                }
                per_window_rows.append(row)

    # Restrict summary table to requested headline FPRs.
    summary_scores: dict[str, Any] = {}
    for score_key, _, score_label in _score_specs(str(args.stap_detector_variant)):
        rows = [r for r in per_window_rows if r["score_key"] == score_key]
        if not rows:
            continue
        sec: dict[str, Any] = {
            "label": score_label,
            "auc": _bootstrap_ci([r["auc"] for r in rows], n_boot=int(args.bootstrap_n), seed=int(args.bootstrap_seed) + 1),
            "n_windows": len(rows),
            "n_blocks": len({int(r["block_id"]) for r in rows}),
        }
        for i, fpr in enumerate(args.fprs):
            tag = f"{float(fpr):.0e}"
            sec[f"tpr@{tag}"] = _bootstrap_ci(
                [r.get(f"tpr@{tag}") for r in rows],
                n_boot=int(args.bootstrap_n),
                seed=int(args.bootstrap_seed) + 100 * (i + 1),
            )
            sec[f"fpr@{tag}"] = _bootstrap_ci(
                [r.get(f"fpr@{tag}") for r in rows],
                n_boot=int(args.bootstrap_n),
                seed=int(args.bootstrap_seed) + 200 * (i + 1),
            )
        summary_scores[score_key] = sec

    summary = {
        "dataset": "ULM Zenodo 7883227",
        "reference_design": {
            "type": "leave_one_block_out_structural_reference",
            "reference_mode": str(args.reference_mode),
            "reference_local_density_quantile": float(args.reference_local_density_quantile),
            "reference_local_density_peak_size": int(args.reference_local_density_peak_size),
            "reference_local_density_sigma": float(args.reference_local_density_sigma),
            "reference_blocks": [int(b) for b in ref_blocks],
            "evaluation_blocks": [int(b) for b in eval_blocks],
            "window_frames": int(args.window_frames),
            "window_stride": int(args.window_stride),
            "prf_hz": float(prf_hz),
            "fprs_table": [float(f) for f in args.fprs],
            "fprs_plot": [float(f) for f in args.fprs_plot],
        },
        "frozen_profile": {
            "baseline": "MC-SVD",
            "svd_energy_frac": float(args.svd_energy_frac),
            "stap_detector_variant": str(args.stap_detector_variant),
            "tile_h": int(args.tile_h),
            "tile_w": int(args.tile_w),
            "tile_stride": int(args.tile_stride),
            "Lt": int(args.lt),
            "cov_estimator": str(args.cov_estimator),
            "diag_load": float(args.diag_load),
            "bands_hz": {
                "flow": [10.0, 150.0],
                "guard": [150.0, 200.0],
                "alias": [200.0, 500.0],
            },
        },
        "mask_qc": mask_qc,
        "reference_manifest": reference_manifest,
        "reference_cache": ref_tele,
        "scores": summary_scores,
        "representative_block_id": representative_block,
    }

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        cols: list[str] = []
        for row in per_window_rows:
            for key in row:
                if key not in cols:
                    cols.append(key)
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(per_window_rows)

    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_table(summary=summary, out_path=args.out_tex, fprs=[float(f) for f in args.fprs])

    if representative_ref is None or representative_flow is None or representative_bg is None:
        raise RuntimeError("failed to select representative reference map")
    _make_mask_figure(
        ref_map=representative_ref,
        mask_flow=representative_flow,
        mask_bg=representative_bg,
        out_path=args.out_mask_fig,
        title_prefix=f"Representative block {int(representative_block):03d} reference",
    )
    _make_roc_figure(
        per_window_rows=per_window_rows,
        fprs_plot=[float(f) for f in args.fprs_plot],
        out_path=args.out_roc_fig,
        n_boot=int(args.bootstrap_n),
        seed=int(args.bootstrap_seed),
    )

    for path in bundle_cleanup:
        shutil.rmtree(path, ignore_errors=True)

    print(f"[ulm7883227-struct] wrote {args.out_csv}")
    print(f"[ulm7883227-struct] wrote {args.out_json}")
    print(f"[ulm7883227-struct] wrote {args.out_tex}")
    print(f"[ulm7883227-struct] wrote {args.out_mask_fig}")
    print(f"[ulm7883227-struct] wrote {args.out_roc_fig}")


if __name__ == "__main__":
    main()
