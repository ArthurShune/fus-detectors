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

import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import match_template
from skimage.filters import threshold_otsu

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.realdata.ulm_zenodo_7883227 import (
    list_ulm_blocks,
    load_ulm_block_iq,
    load_ulm_zenodo_7883227_params,
)
from scripts.ulm_zenodo_7883227_motion_sweep import (
    _apply_translation_per_frame,
    _apply_warp_per_frame,
    _brainlike_displacement_fields,
    _motion_shifts,
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


def _threshold_from_pos(pos: np.ndarray, tpr: float) -> tuple[float | None, float | None]:
    pos_f = _finite(pos)
    if pos_f.size == 0:
        return None, None
    q = float(np.clip(1.0 - float(tpr), 0.0, 1.0))
    thr = float(np.quantile(pos_f, q))
    tpr_emp = float(np.mean(pos_f >= thr))
    return thr, tpr_emp


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
    support_mask: np.ndarray | None,
    vessel_quantile: float,
    bg_quantile: float,
    erode_iters: int,
    guard_dilate_iters: int,
    edge_margin: int,
    vessel_mask_mode: str,
    peak_size: int,
    peak_dilate_iters: int,
    background_mask_mode: str,
    shell_inner_dilate_iters: int,
    shell_outer_dilate_iters: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    ref = np.asarray(reference_pd, dtype=np.float64)
    finite = np.isfinite(ref)
    if not finite.any():
        raise ValueError("reference map has no finite values")
    if support_mask is None:
        support = finite.copy()
    else:
        support = np.asarray(support_mask, dtype=bool) & finite
    if not support.any():
        raise ValueError("structural support mask has no valid pixels")
    vessel_thr = _safe_quantile(ref[finite], vessel_quantile)
    positive_vals = ref[finite & (ref > 0.0)]
    if positive_vals.size and vessel_thr <= 0.0 and (positive_vals.size / max(1, int(finite.sum()))) < 0.5:
        vessel_thr = _safe_quantile(positive_vals, vessel_quantile)
    vessel_mode = str(vessel_mask_mode or "area").strip().lower()
    if vessel_mode == "peaks":
        peaks = (ref >= vessel_thr) & support
        peaks &= ref == ndi.maximum_filter(ref, size=max(3, int(peak_size)), mode="nearest")
        vessel = peaks
        if peak_dilate_iters > 0:
            vessel = ndi.binary_dilation(vessel, iterations=int(peak_dilate_iters))
    else:
        vessel = (ref >= vessel_thr) & support
        if erode_iters > 0:
            vessel = ndi.binary_erosion(vessel, iterations=int(erode_iters))
        vessel = ndi.binary_opening(vessel, iterations=1)

    bg_thr = _safe_quantile(ref[support], bg_quantile)
    bg_candidate = support.copy()
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
    bg_mode = str(background_mask_mode or "global_low").strip().lower()
    if bg_mode == "shell":
        inner = ndi.binary_dilation(vessel, iterations=max(1, int(shell_inner_dilate_iters)))
        outer = ndi.binary_dilation(vessel, iterations=max(int(shell_outer_dilate_iters), int(shell_inner_dilate_iters) + 1))
        bg = outer & ~inner & support
        if edge_margin > 0:
            bg[:edge_margin, :] = False
            bg[-edge_margin:, :] = False
            bg[:, :edge_margin] = False
            bg[:, -edge_margin:] = False
        bg &= ~vessel
        bg_thr = float("nan")
    else:
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
        "background_threshold": float(bg_thr) if np.isfinite(bg_thr) else None,
        "n_vessel": int(vessel.sum()),
        "n_background": int(bg.sum()),
        "n_support": int(support.sum()),
        "guard_dilate_iters": int(guard_dilate_iters),
        "edge_margin_px": int(edge_margin),
        "vessel_components": int(_connected_components(vessel)),
        "background_components": int(_connected_components(bg)),
        "fpr_floor_background": float(1.0 / max(1, int(bg.sum()))),
        "vessel_mask_mode": vessel_mode,
        "background_mask_mode": bg_mode,
        "peak_size": int(peak_size),
        "peak_dilate_iters": int(peak_dilate_iters),
        "shell_inner_dilate_iters": int(shell_inner_dilate_iters),
        "shell_outer_dilate_iters": int(shell_outer_dilate_iters),
    }
    return vessel.astype(bool), bg.astype(bool), qc


def _transform_structural_masks(
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    *,
    support_mask: np.ndarray,
    mode: str,
    shift_y: int,
    shift_x: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mode_norm = str(mode or "none").strip().lower()
    flow = np.asarray(mask_flow, dtype=bool)
    bg = np.asarray(mask_bg, dtype=bool)
    support = np.asarray(support_mask, dtype=bool)
    if mode_norm in {"", "none", "identity"}:
        return flow, bg, {"mask_transform": "none"}
    if mode_norm in {"translate", "shift", "roll"}:
        flow_t = np.roll(flow, shift=(int(shift_y), int(shift_x)), axis=(0, 1))
        bg_t = np.roll(bg, shift=(int(shift_y), int(shift_x)), axis=(0, 1))
        return flow_t, bg_t, {
            "mask_transform": "translate",
            "shift_y_px": int(shift_y),
            "shift_x_px": int(shift_x),
        }
    if mode_norm in {"permute", "shuffle", "random"}:
        idx = np.flatnonzero(support)
        n_flow = int(flow.sum())
        n_bg = int(bg.sum())
        need = n_flow + n_bg
        if idx.size < need:
            raise ValueError(
                f"support mask too small for permuted control: support={idx.size}, need={need}"
            )
        rng = np.random.default_rng(int(seed))
        choice = rng.choice(idx, size=need, replace=False)
        flow_t = np.zeros_like(flow, dtype=bool)
        bg_t = np.zeros_like(bg, dtype=bool)
        flow_t[np.unravel_index(choice[:n_flow], flow.shape)] = True
        bg_t[np.unravel_index(choice[n_flow:], bg.shape)] = True
        return flow_t, bg_t, {
            "mask_transform": "permute",
            "seed": int(seed),
            "support_pixels": int(idx.size),
        }
    raise ValueError(f"unsupported mask transform {mode!r}")


def _derive_powdop_support_mask(powdop_crop: np.ndarray) -> np.ndarray:
    img = _to_gray_unit(np.asarray(powdop_crop, dtype=np.float32))
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        raise ValueError("powdop crop has no finite values")
    thr = float(min(threshold_otsu(vals), np.quantile(vals, 0.12)))
    thr = max(thr, 0.02)
    support = img >= thr
    support = ndi.binary_closing(support, iterations=3)
    support = ndi.binary_fill_holes(support)
    support = ndi.binary_opening(support, iterations=1)
    if int(support.sum()) == 0:
        support = img >= float(np.quantile(vals, 0.20))
        support = ndi.binary_fill_holes(support)
    return support.astype(bool)


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
    pala_example_root: Path,
    pala_powdop_blocks: list[int],
    pala_svd_cutoff_start: int,
    pala_trim_sr_border: int,
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
    if mode_norm == "pala_example_matout":
        return _compute_reference_pala_example_matout(
            root=root,
            cache_dir=cache_dir / "pala_example_matout",
            pala_example_root=Path(pala_example_root),
            pala_powdop_blocks=[int(b) for b in pala_powdop_blocks],
            svd_cutoff_start=int(pala_svd_cutoff_start),
            trim_sr_border=int(pala_trim_sr_border),
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


def _to_gray_unit(arr: np.ndarray) -> np.ndarray:
    img = np.asarray(arr, dtype=np.float32)
    if img.ndim == 3:
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    img = img.astype(np.float32, copy=False)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape[:2], dtype=np.float32)
    vals = img[finite]
    lo = float(vals.min())
    hi = float(vals.max())
    if hi <= lo:
        out = np.zeros_like(img, dtype=np.float32)
        out[finite] = 1.0
        return out
    out = np.zeros_like(img, dtype=np.float32)
    out[finite] = (img[finite] - lo) / (hi - lo)
    return out


def _pala_svd_filter(cube_t_hw: np.ndarray, *, cutoff_start: int) -> np.ndarray:
    cube = np.asarray(cube_t_hw, dtype=np.complex64)
    if cube.ndim != 3:
        raise ValueError(f"expected (T,H,W) cube, got {cube.shape}")
    T, H, W = cube.shape
    keep0 = max(int(cutoff_start) - 1, 0)
    if keep0 <= 0:
        return cube
    X = np.transpose(cube, (1, 2, 0)).reshape(H * W, T)
    gram = X.conj().T @ X
    evals, U = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    U = U[:, order]
    V = X @ U
    Xf = V[:, keep0:] @ U[:, keep0:].conj().T
    return Xf.reshape(H, W, T).transpose(2, 0, 1).astype(np.complex64, copy=False)


def _compute_pala_powdop_mean(
    *,
    root: Path,
    blocks: list[int],
    cutoff_start: int,
    cache_dir: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mean_path = cache_dir / (
        f"pala_powdop_mean_blocks_{'-'.join(f'{int(b):03d}' for b in blocks)}_cut{int(cutoff_start)}.npy"
    )
    if mean_path.is_file():
        return np.load(mean_path, allow_pickle=False).astype(np.float32, copy=False)
    pds: list[np.ndarray] = []
    for block_id in blocks:
        per_path = cache_dir / f"pala_powdop_block{int(block_id):03d}_cut{int(cutoff_start)}.npy"
        if per_path.is_file():
            pd = np.load(per_path, allow_pickle=False).astype(np.float32, copy=False)
        else:
            cube = load_ulm_block_iq(int(block_id), frames=None, root=root)
            filt = _pala_svd_filter(cube, cutoff_start=int(cutoff_start))
            pd = np.sqrt(np.sum(np.abs(np.transpose(filt, (1, 2, 0))) ** 2, axis=2)).astype(
                np.float32, copy=False
            )
            np.save(per_path, pd, allow_pickle=False)
        pds.append(pd)
    mean_pd = np.mean(np.stack(pds, axis=0), axis=0).astype(np.float32, copy=False)
    np.save(mean_path, mean_pd, allow_pickle=False)
    return mean_pd


def _mask_pala_scalebar(matout_gray: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    img = _to_gray_unit(matout_gray)
    H, W = img.shape
    bright = img >= max(0.75, float(np.quantile(img, 0.995)))
    roi = np.zeros_like(bright, dtype=bool)
    roi[max(0, H - 160) :, : min(W, 240)] = True
    labels, n = ndi.label(bright & roi)
    masked = img.copy()
    payload: dict[str, Any] = {"removed": False}
    if n <= 0:
        return masked, payload
    best_lab = None
    best_area = 0
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        area = int(ys.size)
        if w >= 4 * max(1, h) and area >= best_area:
            best_lab = lab
            best_area = area
    if best_lab is None:
        return masked, payload
    mask = labels == int(best_lab)
    masked[mask] = 0.0
    ys, xs = np.where(mask)
    payload = {
        "removed": True,
        "area_px": int(mask.sum()),
        "bbox_yxhw": [
            int(ys.min()),
            int(xs.min()),
            int(ys.max() - ys.min() + 1),
            int(xs.max() - xs.min() + 1),
        ],
    }
    return masked, payload


def _candidate_oriented_maps(raw_pd: np.ndarray) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_pd, dtype=np.float32)
    return {
        "transpose": raw.T,
        "transpose_fliplr": np.fliplr(raw.T),
        "transpose_flipud": np.flipud(raw.T),
        "transpose_flipud_fliplr": np.flipud(np.fliplr(raw.T)),
        "raw": raw,
        "flipud": np.flipud(raw),
        "fliplr": np.fliplr(raw),
    }


def _register_pala_example_crop(
    *,
    mean_pd_raw: np.ndarray,
    pala_powdop_tif: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / "pala_example_registration.json"
    if out_path.is_file():
        return json.loads(out_path.read_text())
    template = _to_gray_unit(iio.imread(pala_powdop_tif))
    if template.shape != (78, 118):
        raise ValueError(f"unexpected PALA PowDop template shape {template.shape}, expected (78, 118)")
    results: list[dict[str, Any]] = []
    for name, arr0 in _candidate_oriented_maps(mean_pd_raw).items():
        arr = np.power(np.maximum(np.asarray(arr0, dtype=np.float32), 0.0), 1.0 / 3.0)
        arr = _to_gray_unit(arr)
        if arr.shape[0] < template.shape[0] or arr.shape[1] < template.shape[1]:
            continue
        resp = match_template(arr, template, pad_input=False)
        ij = np.unravel_index(np.argmax(resp), resp.shape)
        y, x = int(ij[0]), int(ij[1])
        crop = arr[y : y + template.shape[0], x : x + template.shape[1]]
        corr = float(np.corrcoef(crop.ravel(), template.ravel())[0, 1])
        results.append(
            {
                "orientation": str(name),
                "match_score": float(resp[ij]),
                "corr": corr,
                "crop_yxhw": [y, x, int(template.shape[0]), int(template.shape[1])],
                "oriented_shape": [int(arr.shape[0]), int(arr.shape[1])],
            }
        )
    if not results:
        raise RuntimeError("failed to register PALA PowDop example to synthesized PowDop")
    best = max(results, key=lambda r: (float(r["match_score"]), float(r["corr"])))
    payload = {"best": best, "candidates": results}
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _oriented_to_raw(oriented: np.ndarray, orientation: str) -> np.ndarray:
    arr = np.asarray(oriented, dtype=np.float32)
    if orientation == "transpose":
        return arr.T
    if orientation == "transpose_fliplr":
        return np.fliplr(arr).T
    if orientation == "transpose_flipud":
        return np.flipud(arr).T
    if orientation == "transpose_flipud_fliplr":
        return np.fliplr(np.flipud(arr)).T
    if orientation == "raw":
        return arr
    if orientation == "flipud":
        return np.flipud(arr)
    if orientation == "fliplr":
        return np.fliplr(arr)
    raise ValueError(f"unsupported orientation {orientation!r}")


def _pool_sr_to_coarse(mat_sr: np.ndarray, *, trim_border: int) -> np.ndarray:
    img = np.asarray(mat_sr, dtype=np.float32)
    trim = max(0, int(trim_border))
    if trim > 0:
        img = img[:-trim, :-trim]
    H, W = img.shape
    if (H % 10) != 0 or (W % 10) != 0:
        raise ValueError(f"expected super-res shape divisible by 10 after trim, got {(H, W)}")
    return img.reshape(H // 10, 10, W // 10, 10).mean(axis=(1, 3)).astype(np.float32, copy=False)


def _compute_reference_pala_example_matout(
    *,
    root: Path,
    cache_dir: Path,
    pala_example_root: Path,
    pala_powdop_blocks: list[int],
    svd_cutoff_start: int,
    trim_sr_border: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    map_path = cache_dir / "reference_pala_example_matout.npy"
    support_path = cache_dir / "reference_pala_example_support.npy"
    tele_path = cache_dir / "reference_pala_example_matout.json"
    if map_path.is_file() and support_path.is_file() and tele_path.is_file():
        return (
            np.load(map_path, allow_pickle=False).astype(np.float32, copy=False),
            json.loads(tele_path.read_text()),
        )
    mean_pd_raw = _compute_pala_powdop_mean(
        root=root,
        blocks=[int(b) for b in pala_powdop_blocks],
        cutoff_start=int(svd_cutoff_start),
        cache_dir=cache_dir / "powdop_cache",
    )
    powdop_tif = pala_example_root / "PALA_InVivoRatBrain_example_PowDop.tif"
    matout_tif = pala_example_root / "PALA_InVivoRatBrain_example_MatOut.tif"
    if not powdop_tif.is_file() or not matout_tif.is_file():
        raise FileNotFoundError(
            f"expected PALA example TIFFs under {pala_example_root}, found PowDop={powdop_tif.is_file()} MatOut={matout_tif.is_file()}"
        )
    powdop_template = _to_gray_unit(iio.imread(powdop_tif))
    reg = _register_pala_example_crop(
        mean_pd_raw=mean_pd_raw,
        pala_powdop_tif=powdop_tif,
        cache_dir=cache_dir,
    )
    best = reg["best"]
    matout_gray, scalebar = _mask_pala_scalebar(_to_gray_unit(iio.imread(matout_tif)))
    coarse_crop = _pool_sr_to_coarse(matout_gray, trim_border=int(trim_sr_border))
    y, x, h, w = [int(v) for v in best["crop_yxhw"]]
    if coarse_crop.shape != (h, w):
        raise ValueError(
            f"PALA MatOut coarse crop shape {coarse_crop.shape} does not match registered PowDop crop {(h, w)}"
        )
    orientation = str(best["orientation"])
    oriented_shape = tuple(int(v) for v in best["oriented_shape"])
    oriented = np.full(oriented_shape, np.nan, dtype=np.float32)
    oriented[y : y + h, x : x + w] = coarse_crop.astype(np.float32, copy=False)
    ref_raw = _oriented_to_raw(oriented, orientation).astype(np.float32, copy=False)
    support_crop = _derive_powdop_support_mask(powdop_template)
    oriented_support = np.zeros(oriented_shape, dtype=bool)
    oriented_support[y : y + h, x : x + w] = support_crop.astype(bool, copy=False)
    support_raw = _oriented_to_raw(oriented_support.astype(np.float32), orientation) > 0.5
    np.save(map_path, ref_raw, allow_pickle=False)
    np.save(support_path, support_raw.astype(np.uint8), allow_pickle=False)
    tele_payload = {
        "reference_mode": "pala_example_matout",
        "pala_example_root": str(pala_example_root),
        "pala_powdop_blocks": [int(b) for b in pala_powdop_blocks],
        "svd_cutoff_start": int(svd_cutoff_start),
        "trim_sr_border": int(trim_sr_border),
        "registration": reg,
        "scalebar_removal": scalebar,
        "coarse_crop_shape": [int(v) for v in coarse_crop.shape],
        "raw_shape": [int(v) for v in ref_raw.shape],
        "support_mask_path": str(support_path),
        "support_mask_pixels": int(support_raw.sum()),
    }
    tele_path.write_text(json.dumps(tele_payload, indent=2, sort_keys=True))
    return ref_raw, tele_payload


def _evaluate_window_score(
    score: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    *,
    fprs: list[float],
    tpr_targets: list[float],
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
    for tpr in tpr_targets:
        pct = int(round(100.0 * float(tpr)))
        tag = f"{pct:02d}"
        thr, realized = _threshold_from_pos(pos, tpr)
        if thr is None:
            out[f"fpr_at_tpr{tag}"] = None
            out[f"tpr_realized@{tag}"] = None
            out[f"thr_tpr@{tag}"] = None
        else:
            out[f"fpr_at_tpr{tag}"] = float(np.mean(neg >= thr)) if neg.size else None
            out[f"tpr_realized@{tag}"] = realized
            out[f"thr_tpr@{tag}"] = float(thr)
    return out


def _stap_variant_label(variant: str) -> str:
    raw = str(variant).strip().lower()
    if raw == "unwhitened_ratio":
        return "Matched-subspace detector (fixed head)"
    if raw == "adaptive_guard":
        return "Matched-subspace detector (adaptive head)"
    if raw == "whitened_power":
        return "Whitened-power detector"
    return "Matched-subspace detector (whitened specialist)"


def _score_specs(
    stap_variant: str = "msd_ratio",
    *,
    include_postka: bool = False,
    include_detector_filtered_pd: bool = False,
) -> list[tuple[str, str, str]]:
    specs = [
        ("pd", "score_base.npy", "Baseline (power Doppler)"),
        ("log_pd", "score_base_pdlog.npy", "Baseline (log-power Doppler)"),
        ("kasai", "score_base_kasai.npy", "Baseline (Kasai lag-1 magnitude)"),
        ("matched_subspace", "score_stap_preka.npy", _stap_variant_label(stap_variant)),
    ]
    if include_detector_filtered_pd:
        specs.insert(3, ("stap_pd", "pd_stap.npy", "Detector-filtered power Doppler"))
    if include_postka:
        specs.append(
            (
                "matched_subspace_postka",
                "score_stap.npy",
                f"{_stap_variant_label(stap_variant)} + shrink-only penalty",
            )
        )
    return specs


def _apply_motion_to_cube(
    cube: np.ndarray,
    *,
    motion_kind: str,
    motion_amp_px: float,
    motion_seed: int,
    step_frame: int | None,
    brainlike_rigid_kind: str,
    brainlike_rigid_frac: float,
    brainlike_elastic_frac: float,
    brainlike_elastic_sigma_px: float,
    brainlike_elastic_depth_decay_frac: float,
    brainlike_elastic_rw_step_sigma: float,
    brainlike_micro_jitter_frac: float,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    kind = str(motion_kind or "none").strip().lower()
    amp_px = float(max(0.0, motion_amp_px))
    if amp_px <= 0.0 or kind in {"none", "off"}:
        return np.asarray(cube, dtype=np.complex64), None
    cube0 = np.asarray(cube, dtype=np.complex64)
    T, H, W = cube0.shape
    if kind in {"sine", "sin", "drift_sine", "step", "burst", "rw", "randomwalk"}:
        dy, dx = _motion_shifts(
            kind,
            T=T,
            amp_px=amp_px,
            seed=int(motion_seed),
            step_frame=step_frame,
        )
        moved = _apply_translation_per_frame(cube0, dy, dx)
        return moved.astype(np.complex64, copy=False), {
            "motion_kind": kind,
            "motion_amp_px": amp_px,
            "motion_seed": int(motion_seed),
            "gt_align_kind": "translation",
            "dy_frame_ref": float(dy[T // 2]),
            "dx_frame_ref": float(dx[T // 2]),
        }
    if kind in {"brainlike", "elastic"}:
        if kind == "elastic":
            rigid_kind = "none"
            rigid_frac = 0.0
            elastic_frac = 1.0
            micro_jitter_frac = 0.0
        else:
            rigid_kind = str(brainlike_rigid_kind)
            rigid_frac = float(brainlike_rigid_frac)
            elastic_frac = float(brainlike_elastic_frac)
            micro_jitter_frac = float(brainlike_micro_jitter_frac)
        dy_field, dx_field, tele = _brainlike_displacement_fields(
            T=T,
            H=H,
            W=W,
            amp_px=amp_px,
            seed=int(motion_seed),
            rigid_kind=rigid_kind,
            rigid_frac=float(rigid_frac),
            elastic_frac=float(elastic_frac),
            elastic_sigma_px=float(brainlike_elastic_sigma_px),
            elastic_depth_decay_frac=float(brainlike_elastic_depth_decay_frac),
            elastic_rw_step_sigma=float(brainlike_elastic_rw_step_sigma),
            micro_jitter_frac=float(micro_jitter_frac),
        )
        moved = _apply_warp_per_frame(cube0, dy_field, dx_field)
        return moved.astype(np.complex64, copy=False), {
            "motion_kind": kind,
            "motion_amp_px": amp_px,
            "motion_seed": int(motion_seed),
            "gt_align_kind": "warp",
            **tele,
        }
    raise ValueError(f"unsupported motion_kind={motion_kind!r}")


def _load_bundle_scores(bundle_dir: Path, *, stap_variant: str, include_postka: bool) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, filename, _ in _score_specs(stap_variant, include_postka=include_postka):
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
        "stap_pd": dict(color="#2ca02c", linewidth=1.8),
        "matched_subspace": dict(color="#1f77b4", linewidth=2.0),
        "matched_subspace_postka": dict(color="#d62728", linewidth=2.0, linestyle="-."),
    }
    include_postka = bool((per_window_rows[0] if per_window_rows else {}).get("score_ka_v2_enable"))
    labels = {
        k: lbl
        for k, _, lbl in _score_specs(
            str(((per_window_rows[0] if per_window_rows else {}).get("stap_detector_variant")) or "msd_ratio"),
            include_postka=include_postka,
        )
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    for score_idx, (score_key, _, _) in enumerate(
        _score_specs(
            str(((per_window_rows[0] if per_window_rows else {}).get("stap_detector_variant")) or "msd_ratio"),
            include_postka=include_postka,
        )
    ):
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
        if score_key in {"matched_subspace", "matched_subspace_postka", "pd", "stap_pd"}:
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
    include_postka = bool(((summary.get("frozen_profile") or {}).get("score_ka_v2_enable")) or False)
    include_detector_filtered_pd = bool(
        ((summary.get("frozen_profile") or {}).get("include_detector_filtered_pd")) or False
    )
    score_order = [
        k
        for k, _, _ in _score_specs(
            stap_variant,
            include_postka=include_postka,
            include_detector_filtered_pd=include_detector_filtered_pd,
        )
    ]
    score_labels = {
        k: lbl
        for k, _, lbl in _score_specs(
            stap_variant,
            include_postka=include_postka,
            include_detector_filtered_pd=include_detector_filtered_pd,
        )
    }
    col_tags = [f"{float(fpr):.0e}" for fpr in fprs]
    ref_mode = str(((summary.get("reference_design") or {}).get("reference_mode")) or "local_density")
    window_frames = int(((summary.get("reference_design") or {}).get("window_frames")) or 128)
    motion_kind = str(((summary.get("reference_design") or {}).get("motion_kind")) or "none")
    motion_amp_px = float(((summary.get("reference_design") or {}).get("motion_amp_px")) or 0.0)
    reg_enable = bool(((summary.get("frozen_profile") or {}).get("reg_enable")) or False)
    if ref_mode == "local_density":
        ref_text = (
            "a leave-one-block-out full-acquisition microbubble-density reference obtained by "
            "accumulating sparse local maxima on MC--SVD-filtered long-horizon residuals from the remaining blocks"
        )
    elif ref_mode == "pala_example_matout":
        ref_text = (
            "an external super-resolved vascular reference derived by registering the published PALA rat-brain "
            "ULM example back onto the Zenodo IQ grid and downsampling the registered MatOut rendering to the "
            "short-window evaluation lattice"
        )
    else:
        ref_text = (
            "a leave-one-block-out full-acquisition vascular reference built from MC--SVD-filtered "
            "long-integration power Doppler on the remaining blocks"
        )
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/ulm7883227_structural_roc.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{center}")
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
        "\\captionof{table}{Structural-label ROC on real in vivo rat-brain IQ from ULM Zenodo 7883227. "
        f"For each evaluation block, vessel and background masks are derived once from {ref_text}; "
        f"short {window_frames}-frame windows from the held-out block"
        + (f" after {motion_kind} motion injection at {motion_amp_px:.2f} px amplitude" if motion_kind not in {'none','off'} and motion_amp_px > 0.0 else "")
        + (" are then scored without additional frame-to-frame registration on the "
           if not reg_enable
           else " are then scored on the ")
        + "same MC--SVD residual cube using different downstream detector heads. Entries report window-level "
        "means with 95\\% bootstrap CIs over windows. Reproducibility details are provided in \\SuppOrApp{app:repro}.}"
    )
    lines.append("\\label{tab:ulm7883227_structural_roc}")
    lines.append("\\end{center}")
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
        choices=["msd_ratio", "unwhitened_ratio", "adaptive_guard", "whitened_power"],
    )
    ap.add_argument("--flow-low-hz", type=float, default=10.0)
    ap.add_argument("--flow-high-hz", type=float, default=150.0)
    ap.add_argument("--alias-center-hz", type=float, default=350.0)
    ap.add_argument("--alias-width-hz", type=float, default=150.0)
    ap.add_argument(
        "--fd-span-mode",
        type=str,
        default="psd",
        choices=["psd", "fixed"],
    )
    ap.add_argument("--fd-fixed-span-hz", type=float, default=None)
    ap.add_argument(
        "--reference-mode",
        type=str,
        default="local_density",
        choices=["local_density", "pd", "pala_example_matout"],
    )
    ap.add_argument("--reference-local-density-quantile", type=float, default=0.9995)
    ap.add_argument("--reference-local-density-peak-size", type=int, default=3)
    ap.add_argument("--reference-local-density-sigma", type=float, default=1.0)
    ap.add_argument(
        "--pala-example-root",
        type=Path,
        default=Path("/tmp/PALA_repo_1073521"),
    )
    ap.add_argument("--pala-powdop-blocks", type=str, default="1,3,5,7,9,11,13,15,17,19")
    ap.add_argument("--pala-svd-cutoff-start", type=int, default=5)
    ap.add_argument("--pala-trim-sr-border", type=int, default=1)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument("--cov-estimator", type=str, default="scm")
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--score-ka-v2-enable", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--reg-enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--reg-subpixel", type=int, default=4)
    ap.add_argument(
        "--motion-kind",
        type=str,
        default="none",
        choices=["none", "sine", "step", "randomwalk", "brainlike", "elastic"],
    )
    ap.add_argument("--motion-amp-px", type=float, default=0.0)
    ap.add_argument("--motion-seed", type=int, default=0)
    ap.add_argument("--step-frame", type=int, default=None)
    ap.add_argument("--brainlike-rigid-kind", type=str, default="randomwalk")
    ap.add_argument("--brainlike-rigid-frac", type=float, default=0.7)
    ap.add_argument("--brainlike-elastic-frac", type=float, default=0.3)
    ap.add_argument("--brainlike-elastic-sigma-px", type=float, default=24.0)
    ap.add_argument("--brainlike-elastic-depth-decay-frac", type=float, default=0.8)
    ap.add_argument("--brainlike-elastic-rw-step-sigma", type=float, default=0.25)
    ap.add_argument("--brainlike-micro-jitter-frac", type=float, default=0.08)
    ap.add_argument("--vessel-quantile", type=float, default=0.92)
    ap.add_argument("--background-quantile", type=float, default=0.50)
    ap.add_argument("--mask-erode-iters", type=int, default=1)
    ap.add_argument("--guard-dilate-iters", type=int, default=3)
    ap.add_argument("--edge-margin", type=int, default=4)
    ap.add_argument("--vessel-mask-mode", type=str, default="area", choices=["area", "peaks"])
    ap.add_argument("--peak-size", type=int, default=3)
    ap.add_argument("--peak-dilate-iters", type=int, default=1)
    ap.add_argument("--background-mask-mode", type=str, default="global_low", choices=["global_low", "shell"])
    ap.add_argument("--shell-inner-dilate-iters", type=int, default=4)
    ap.add_argument("--shell-outer-dilate-iters", type=int, default=10)
    ap.add_argument(
        "--mask-transform",
        type=str,
        default="none",
        choices=["none", "translate", "permute"],
    )
    ap.add_argument("--mask-shift-y", type=int, default=32)
    ap.add_argument("--mask-shift-x", type=int, default=24)
    ap.add_argument("--mask-transform-seed", type=int, default=2026)
    ap.add_argument("--fprs", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3])
    ap.add_argument("--tpr-targets", type=float, nargs="+", default=[0.5])
    ap.add_argument(
        "--include-detector-filtered-pd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the derived detector-filtered PD diagnostic in exported tables and summaries.",
    )
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
    lt_eff = min(int(args.lt), int(args.window_frames) - 1)
    if lt_eff < 2:
        raise ValueError(f"window_frames must exceed Lt by at least one sample (got window_frames={args.window_frames}, Lt={args.lt})")
    ref_blocks = _parse_blocks(args.ref_blocks, root=args.data_root)
    eval_blocks = _parse_blocks(args.eval_blocks, root=args.data_root)
    pala_powdop_blocks = _parse_blocks(args.pala_powdop_blocks, root=args.data_root)

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
    support_maps: dict[int, np.ndarray] = {}
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
            pala_example_root=Path(args.pala_example_root),
            pala_powdop_blocks=[int(b) for b in pala_powdop_blocks],
            pala_svd_cutoff_start=int(args.pala_svd_cutoff_start),
            pala_trim_sr_border=int(args.pala_trim_sr_border),
        )
        ref_maps[int(block_id)] = ref_map
        support_path = str(tele.get("support_mask_path", "")).strip()
        if support_path:
            support_maps[int(block_id)] = np.load(support_path, allow_pickle=False).astype(bool)
        else:
            support_maps[int(block_id)] = np.isfinite(ref_map)
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
        support_stack = np.stack([support_maps[int(bid)].astype(np.float32) for bid in loo_ids], axis=0)
        ref_map = np.mean(loo_stack, axis=0).astype(np.float32, copy=False)
        support_mask = np.mean(support_stack, axis=0) >= 0.5
        mask_flow, mask_bg, qc = _derive_structural_masks(
            ref_map,
            support_mask=support_mask,
            vessel_quantile=float(args.vessel_quantile),
            bg_quantile=float(args.background_quantile),
            erode_iters=int(args.mask_erode_iters),
            guard_dilate_iters=int(args.guard_dilate_iters),
            edge_margin=int(args.edge_margin),
            vessel_mask_mode=str(args.vessel_mask_mode),
            peak_size=int(args.peak_size),
            peak_dilate_iters=int(args.peak_dilate_iters),
            background_mask_mode=str(args.background_mask_mode),
            shell_inner_dilate_iters=int(args.shell_inner_dilate_iters),
            shell_outer_dilate_iters=int(args.shell_outer_dilate_iters),
        )
        mask_flow, mask_bg, transform_tele = _transform_structural_masks(
            mask_flow,
            mask_bg,
            support_mask=support_mask,
            mode=str(args.mask_transform),
            shift_y=int(args.mask_shift_y),
            shift_x=int(args.mask_shift_x),
            seed=int(args.mask_transform_seed) + int(block_id),
        )
        if int(mask_bg.sum()) < 1000:
            raise RuntimeError(f"block {block_id}: background mask too small for 1e-3 tail ({int(mask_bg.sum())} pixels)")
        mask_qc[int(block_id)] = qc
        reference_manifest[int(block_id)] = {
            "reference_blocks": [int(b) for b in loo_ids],
            "mask_qc": qc,
            "mask_transform": transform_tele,
            "support_pixels": int(support_mask.sum()),
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
            cube, motion_tele = _apply_motion_to_cube(
                cube,
                motion_kind=str(args.motion_kind),
                motion_amp_px=float(args.motion_amp_px),
                motion_seed=int(args.motion_seed) + 1009 * int(block_id) + 37 * int(win_idx),
                step_frame=args.step_frame,
                brainlike_rigid_kind=str(args.brainlike_rigid_kind),
                brainlike_rigid_frac=float(args.brainlike_rigid_frac),
                brainlike_elastic_frac=float(args.brainlike_elastic_frac),
                brainlike_elastic_sigma_px=float(args.brainlike_elastic_sigma_px),
                brainlike_elastic_depth_decay_frac=float(args.brainlike_elastic_depth_decay_frac),
                brainlike_elastic_rw_step_sigma=float(args.brainlike_elastic_rw_step_sigma),
                brainlike_micro_jitter_frac=float(args.brainlike_micro_jitter_frac),
            )
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
                Lt=int(lt_eff),
                diag_load=float(args.diag_load),
                cov_estimator=str(args.cov_estimator),
                stap_device=str(args.stap_device),
                baseline_type="mc_svd",
                reg_enable=bool(args.reg_enable),
                reg_subpixel=int(args.reg_subpixel),
                svd_energy_frac=float(args.svd_energy_frac),
                run_stap=True,
                stap_detector_variant=str(args.stap_detector_variant),
                score_ka_v2_enable=bool(args.score_ka_v2_enable),
                stap_conditional_enable=False,
                flow_mask_mode="pd_auto",
                mask_flow_override=mask_flow,
                mask_bg_override=mask_bg,
                band_ratio_flow_low_hz=float(args.flow_low_hz),
                band_ratio_flow_high_hz=float(args.flow_high_hz),
                band_ratio_alias_center_hz=float(args.alias_center_hz),
                band_ratio_alias_width_hz=float(args.alias_width_hz),
                fd_span_mode=str(args.fd_span_mode),
                fd_fixed_span_hz=(
                    None if args.fd_fixed_span_hz is None else float(args.fd_fixed_span_hz)
                ),
                meta_extra={
                    "ulm_structural_roc": {
                        "evaluation_block_id": int(block_id),
                        "window_start": int(start),
                        "window_stop": int(stop),
                        "reference_block_ids": [int(b) for b in loo_ids],
                        "motion": motion_tele,
                    }
                },
            )
            bundle_dir = Path(paths["meta"]).parent
            scores = _load_bundle_scores(
                bundle_dir,
                stap_variant=str(args.stap_detector_variant),
                include_postka=bool(args.score_ka_v2_enable),
            )
            for score_key, _, score_label in _score_specs(
                str(args.stap_detector_variant),
                include_postka=bool(args.score_ka_v2_enable),
                include_detector_filtered_pd=bool(args.include_detector_filtered_pd),
            ):
                score = scores.get(score_key)
                if score is None:
                    continue
                metrics = _evaluate_window_score(
                    score,
                    mask_flow,
                    mask_bg,
                    fprs=[float(f) for f in args.fprs_plot],
                    tpr_targets=[float(t) for t in args.tpr_targets],
                )
                row = {
                    "block_id": int(block_id),
                    "window_index": int(win_idx),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "score_key": str(score_key),
                    "score_label": str(score_label),
                    "stap_detector_variant": str(args.stap_detector_variant),
                    "motion_kind": str(args.motion_kind),
                    "motion_amp_px": float(args.motion_amp_px),
                    "score_ka_v2_enable": bool(args.score_ka_v2_enable),
                    "n_vessel": int(mask_flow.sum()),
                    "n_background": int(mask_bg.sum()),
                    **metrics,
                }
                per_window_rows.append(row)

    # Restrict summary table to requested headline FPRs.
    summary_scores: dict[str, Any] = {}
    for score_key, _, score_label in _score_specs(
        str(args.stap_detector_variant),
        include_postka=bool(args.score_ka_v2_enable),
        include_detector_filtered_pd=bool(args.include_detector_filtered_pd),
    ):
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
        for i, tpr in enumerate(args.tpr_targets):
            tag = f"{int(round(100.0 * float(tpr))):02d}"
            sec[f"fpr_at_tpr{tag}"] = _bootstrap_ci(
                [r.get(f"fpr_at_tpr{tag}") for r in rows],
                n_boot=int(args.bootstrap_n),
                seed=int(args.bootstrap_seed) + 500 * (i + 1),
            )
            sec[f"tpr_realized@{tag}"] = _bootstrap_ci(
                [r.get(f"tpr_realized@{tag}") for r in rows],
                n_boot=int(args.bootstrap_n),
                seed=int(args.bootstrap_seed) + 600 * (i + 1),
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
            "tpr_targets": [float(t) for t in args.tpr_targets],
            "motion_kind": str(args.motion_kind),
            "motion_amp_px": float(args.motion_amp_px),
            "vessel_mask_mode": str(args.vessel_mask_mode),
            "background_mask_mode": str(args.background_mask_mode),
            "mask_transform": {
                "mode": str(args.mask_transform),
                "shift_y_px": int(args.mask_shift_y),
                "shift_x_px": int(args.mask_shift_x),
                "seed": int(args.mask_transform_seed),
            },
        },
        "frozen_profile": {
            "baseline": "MC-SVD",
            "svd_energy_frac": float(args.svd_energy_frac),
            "stap_detector_variant": str(args.stap_detector_variant),
            "include_detector_filtered_pd": bool(args.include_detector_filtered_pd),
            "tile_h": int(args.tile_h),
            "tile_w": int(args.tile_w),
            "tile_stride": int(args.tile_stride),
            "Lt_requested": int(args.lt),
            "Lt": int(lt_eff),
            "cov_estimator": str(args.cov_estimator),
            "diag_load": float(args.diag_load),
            "score_ka_v2_enable": bool(args.score_ka_v2_enable),
            "fd_span_mode": str(args.fd_span_mode),
            "fd_fixed_span_hz": (
                None if args.fd_fixed_span_hz is None else float(args.fd_fixed_span_hz)
            ),
            "bands_hz": {
                "flow": [float(args.flow_low_hz), float(args.flow_high_hz)],
                "guard": [float(args.flow_high_hz), float(args.alias_center_hz) - float(args.alias_width_hz)],
                "alias": [
                    float(args.alias_center_hz) - float(args.alias_width_hz),
                    float(args.alias_center_hz) + float(args.alias_width_hz),
                ],
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
