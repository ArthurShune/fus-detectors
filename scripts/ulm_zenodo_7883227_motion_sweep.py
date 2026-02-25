from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from pipeline.realdata.ulm_zenodo_7883227 import (
    load_ulm_block_iq,
    load_ulm_zenodo_7883227_params,
)
from sim.kwave.common import _phasecorr_shift
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> tuple[slice | None, str]:
    spec = (spec or "").strip()
    if spec in {"", "all", ":", "0:"}:
        return None, "all"
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]' or 'all'.")
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    if stop is None:
        raise ValueError("Slice spec must include stop (e.g. 0:128).")
    tag = f"{start:04d}_{stop:04d}" if step == 1 else f"{start:04d}_{stop:04d}_s{step}"
    return slice(start, stop, step), tag


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return out


def _parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return out


def _safe_quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, float(q)))


def _connected_components(binary: np.ndarray, connectivity: int = 4) -> int:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return 0
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    _, n = ndi.label(binary, structure=structure)
    return int(n)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    if a.size == 0:
        return float("nan")
    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0))) + 1e-12
    return float(np.sum(a0 * b0) / denom)


def _nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return float("nan")
    err = a[finite] - b[finite]
    rmse = float(np.sqrt(np.mean(err * err)))
    scale = float(np.std(b[finite])) + 1e-12
    return float(rmse / scale)


def _shannon_entropy(x: np.ndarray, *, bins: int = 128) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = x - float(np.mean(x))
    std = float(np.std(x)) + 1e-12
    x = np.clip(x / std, -6.0, 6.0)
    hist, _ = np.histogram(x, bins=int(max(8, bins)), range=(-6.0, 6.0), density=False)
    p = hist.astype(np.float64)
    s = float(np.sum(p))
    if s <= 0.0:
        return float("nan")
    p /= s
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    inter = int(np.sum(a & b))
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0
    return float(inter / union)


def _topk_mask(x: np.ndarray, frac: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    frac = float(np.clip(frac, 1e-6, 1.0))
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=bool)
    thr = float(np.quantile(x[finite], 1.0 - frac))
    return np.asarray(x >= thr, dtype=bool)


def _std_ratio(x: np.ndarray, ref: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    denom = float(np.std(ref[np.isfinite(ref)])) + 1e-12
    return float(np.std(x[np.isfinite(x)]) / denom)


def _quantile_ratio(x: np.ndarray, ref: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    q = float(np.clip(q, 0.0, 1.0))
    denom = float(np.quantile(ref[np.isfinite(ref)], q)) + 1e-12
    return float(np.quantile(x[np.isfinite(x)], q) / denom)


def _warp2d(img: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    dx = np.asarray(dx, dtype=np.float32)
    if img.shape != dy.shape or img.shape != dx.shape:
        raise ValueError(f"warp2d: img/dy/dx shapes must match, got {img.shape}, {dy.shape}, {dx.shape}")
    H, W = img.shape
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    coords = np.empty((2, H, W), dtype=np.float32)
    coords[0] = yy - dy
    coords[1] = xx - dx
    return ndi.map_coordinates(img, coords, order=1, mode="nearest", prefilter=False).astype(np.float32, copy=False)


def _threshold_at_fpr(score: np.ndarray, neg_mask: np.ndarray, fpr: float) -> float:
    score = np.asarray(score, dtype=np.float64)
    neg = np.asarray(neg_mask, dtype=bool)
    vals = score[neg]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    fpr = float(np.clip(fpr, 1e-9, 1.0 - 1e-9))
    return float(np.quantile(vals, 1.0 - fpr))


def _tpr_fpr_at_threshold(
    score: np.ndarray, *, pos_mask: np.ndarray, neg_mask: np.ndarray, thr: float
) -> tuple[float, float]:
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos_mask, dtype=bool)
    neg = np.asarray(neg_mask, dtype=bool)
    if pos.shape != score.shape or neg.shape != score.shape:
        raise ValueError("Masks must match score shape.")
    pos_vals = score[pos]
    neg_vals = score[neg]
    if pos_vals.size == 0 or neg_vals.size == 0:
        return float("nan"), float("nan")
    tpr = float(np.mean(pos_vals >= float(thr)))
    fpr = float(np.mean(neg_vals >= float(thr)))
    return tpr, fpr


def _apply_translation_per_frame(iq: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    T, H, W = iq.shape
    if dy.shape != (T,) or dx.shape != (T,):
        raise ValueError(f"dy/dx must have shape (T,), got {dy.shape} / {dx.shape} for T={T}")
    out = np.empty_like(iq)
    for t in range(T):
        shift = (float(dy[t]), float(dx[t]))
        re = ndi.shift(iq[t].real, shift=shift, order=1, mode="nearest", prefilter=False)
        im = ndi.shift(iq[t].imag, shift=shift, order=1, mode="nearest", prefilter=False)
        out[t] = re.astype(np.float32, copy=False) + 1j * im.astype(np.float32, copy=False)
    return out


def _apply_warp_per_frame(iq: np.ndarray, dy_field: np.ndarray, dx_field: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq, dtype=np.complex64)
    T, H, W = iq.shape
    if dy_field.shape != (T, H, W) or dx_field.shape != (T, H, W):
        raise ValueError(
            f"dy_field/dx_field must have shape (T,H,W)={(T,H,W)}, got {dy_field.shape} / {dx_field.shape}"
        )
    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    coords = np.empty((2, H, W), dtype=np.float32)
    out = np.empty_like(iq)
    for t in range(T):
        coords[0] = yy - dy_field[t].astype(np.float32, copy=False)
        coords[1] = xx - dx_field[t].astype(np.float32, copy=False)
        re = ndi.map_coordinates(iq[t].real, coords, order=1, mode="nearest", prefilter=False)
        im = ndi.map_coordinates(iq[t].imag, coords, order=1, mode="nearest", prefilter=False)
        out[t] = re.astype(np.float32, copy=False) + 1j * im.astype(np.float32, copy=False)
    return out


def _normalized_random_walk(T: int, *, rng: np.random.Generator, step_sigma: float = 1.0) -> np.ndarray:
    steps = rng.normal(scale=float(step_sigma), size=T).astype(np.float32)
    walk = np.cumsum(steps).astype(np.float32)
    walk -= float(np.mean(walk))
    rms = float(np.sqrt(np.mean(walk * walk))) + 1e-12
    return (walk / rms).astype(np.float32, copy=False)


def _smooth_unit_field(H: int, W: int, *, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    field = rng.standard_normal((H, W)).astype(np.float32)
    field = ndi.gaussian_filter(field, sigma=float(max(sigma_px, 0.5)), mode="nearest")
    field -= float(np.mean(field))
    rms = float(np.sqrt(np.mean(field * field))) + 1e-12
    field /= rms
    return field.astype(np.float32, copy=False)


def _motion_shifts(
    kind: str, T: int, amp_px: float, seed: int, step_frame: int | None
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    kind = (kind or "none").strip().lower()
    if amp_px <= 0.0 or kind in {"none", "off"}:
        return np.zeros(T, dtype=np.float32), np.zeros(T, dtype=np.float32)

    if kind in {"sine", "sin", "drift_sine"}:
        tt = np.arange(T, dtype=np.float32)
        dy = amp_px * np.sin(2.0 * np.pi * tt / max(1.0, float(T)))
        dx = amp_px * np.cos(2.0 * np.pi * tt / max(1.0, float(T)))
        return dy.astype(np.float32), dx.astype(np.float32)

    if kind in {"step", "burst"}:
        if step_frame is None:
            step_frame = T // 2
        dy = np.zeros(T, dtype=np.float32)
        dx = np.zeros(T, dtype=np.float32)
        dy[int(step_frame) :] = float(amp_px)
        dx[int(step_frame) :] = float(-0.5 * amp_px)
        return dy, dx

    if kind in {"rw", "randomwalk"}:
        steps = rng.normal(size=(T, 2)).astype(np.float32)
        steps -= steps.mean(axis=0, keepdims=True)
        walk = np.cumsum(steps, axis=0)
        walk -= walk.mean(axis=0, keepdims=True)
        scale = float(np.sqrt(np.mean(walk[:, 0] ** 2 + walk[:, 1] ** 2))) + 1e-12
        walk = (float(amp_px) / scale) * walk
        return walk[:, 0].astype(np.float32), walk[:, 1].astype(np.float32)

    raise ValueError(f"Unknown motion kind {kind!r}.")


def _brainlike_displacement_fields(
    *,
    T: int,
    H: int,
    W: int,
    amp_px: float,
    seed: int,
    rigid_kind: str,
    rigid_frac: float,
    elastic_frac: float,
    elastic_sigma_px: float,
    elastic_depth_decay_frac: float,
    elastic_rw_step_sigma: float,
    micro_jitter_frac: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    amp_px = float(max(0.0, amp_px))
    rng = np.random.default_rng(int(seed))

    rigid_frac = float(np.clip(rigid_frac, 0.0, 1.0))
    elastic_frac = float(np.clip(elastic_frac, 0.0, 1.0))
    if rigid_frac + elastic_frac <= 1e-6:
        rigid_frac = 0.0
        elastic_frac = 0.0
    else:
        s = rigid_frac + elastic_frac
        rigid_frac /= s
        elastic_frac /= s

    rigid_amp = amp_px * rigid_frac
    dy_rigid_1d, dx_rigid_1d = _motion_shifts(
        rigid_kind, T=T, amp_px=rigid_amp, seed=int(seed) + 1, step_frame=None
    )

    elastic_amp = amp_px * elastic_frac
    if elastic_amp <= 0.0:
        dy_el = np.zeros((T, H, W), dtype=np.float32)
        dx_el = np.zeros((T, H, W), dtype=np.float32)
    else:
        dy0 = _smooth_unit_field(H, W, sigma_px=float(elastic_sigma_px), rng=rng)
        dx0 = _smooth_unit_field(H, W, sigma_px=float(elastic_sigma_px), rng=rng)

        decay = float(np.clip(elastic_depth_decay_frac, 1e-3, 5.0))
        z = (np.arange(H, dtype=np.float32) + 0.5) / max(float(H), 1.0)
        depth_w = np.exp(-z / decay).astype(np.float32)
        depth_w /= float(np.sqrt(np.mean(depth_w * depth_w)) + 1e-12)
        dy0 = dy0 * depth_w[:, None]
        dx0 = dx0 * depth_w[:, None]

        a_t = _normalized_random_walk(T, rng=rng, step_sigma=float(elastic_rw_step_sigma))
        b_t = _normalized_random_walk(T, rng=rng, step_sigma=float(elastic_rw_step_sigma))
        dy_el = (elastic_amp * a_t[:, None, None] * dy0[None, :, :]).astype(np.float32, copy=False)
        dx_el = (elastic_amp * b_t[:, None, None] * dx0[None, :, :]).astype(np.float32, copy=False)

    micro_std = amp_px * float(np.clip(micro_jitter_frac, 0.0, 1.0))
    if micro_std > 0.0:
        dy_j = rng.normal(scale=micro_std, size=T).astype(np.float32)
        dx_j = rng.normal(scale=micro_std, size=T).astype(np.float32)
    else:
        dy_j = np.zeros((T,), dtype=np.float32)
        dx_j = np.zeros((T,), dtype=np.float32)

    dy = dy_el + (dy_rigid_1d + dy_j)[:, None, None]
    dx = dx_el + (dx_rigid_1d + dx_j)[:, None, None]

    disp = np.sqrt(dy * dy + dx * dx)
    tele = {
        "rigid_kind": str(rigid_kind),
        "rigid_frac": float(rigid_frac),
        "elastic_frac": float(elastic_frac),
        "elastic_sigma_px": float(elastic_sigma_px),
        "elastic_depth_decay_frac": float(elastic_depth_decay_frac),
        "elastic_rw_step_sigma": float(elastic_rw_step_sigma),
        "micro_jitter_frac": float(micro_jitter_frac),
        "disp_rms_px": float(np.sqrt(np.mean(disp * disp))),
        "disp_p90_px": float(np.quantile(disp.ravel(), 0.90)),
    }
    return dy.astype(np.float32, copy=False), dx.astype(np.float32, copy=False), tele


def _profile_to_bands(profile: str) -> tuple[float, float, float, float]:
    """Return (flow_low, flow_high, alias_center, alias_half_width)."""
    p = (profile or "").strip().lower()
    if p in {"u", "ulm_u", "shin_u"}:
        return 60.0, 250.0, 400.0, 100.0
    if p in {"b", "brain"}:
        return 30.0, 250.0, 400.0, 100.0
    if p in {"ulm", "ulm_lo", "ulm_low"}:
        # ULM 7883227 empirical peak frequencies (T=128, PRF=1kHz) cluster in the
        # first several off-DC bins (~8–80 Hz) with occasional higher peaks; this
        # band keeps score_stap_preka non-degenerate for motion sweeps.
        return 10.0, 150.0, 350.0, 150.0
    raise ValueError("Profile must be one of: U, B, ULM.")


def _format_amp_tag(amp_px: float) -> str:
    if amp_px == 0.0:
        return "0p00"
    return f"{amp_px:.2f}".replace(".", "p").replace("-", "m")


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic motion-injection sweep for ULM Zenodo 7883227 IQ blocks.\n"
            "Applies controlled per-frame translations/warps to the IQ cube, reruns baseline+STAP,\n"
            "and reports label-free robustness metrics using score_vNext maps (score_base, score_stap_preka)."
        )
    )
    parser.add_argument("--block-ids", type=str, required=True, help="Comma-separated block ids (e.g. 2,3).")
    parser.add_argument("--frames", type=str, default="0:128")
    parser.add_argument("--prf-hz", type=float, default=None, help="Slow-time sampling rate (default: param.json FrameRate).")
    parser.add_argument(
        "--profile",
        type=str,
        default="ULM",
        help="Frozen band profile: U, B, or ULM (default: %(default)s).",
    )

    parser.add_argument("--Lt", type=int, default=64)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)
    parser.add_argument("--diag-load", type=float, default=0.07)
    parser.add_argument("--cov-estimator", type=str, default="scm", help="scm|huber|tyler_pca (default: %(default)s).")
    parser.add_argument("--huber-c", type=float, default=5.0)

    parser.add_argument("--baseline-type", type=str, default="mc_svd", choices=["mc_svd", "svd_bandpass"])
    parser.add_argument("--svd-rank", type=str, default="none", help="For mc_svd: int or 'none' (energy-frac).")
    parser.add_argument("--svd-energy-frac", type=float, default=0.95)
    parser.add_argument("--svd-keep-min", type=int, default=3)
    parser.add_argument("--svd-keep-max", type=int, default=40)
    parser.add_argument(
        "--reg-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rigid phase-correlation registration in baseline (default: %(default)s).",
    )
    parser.add_argument("--reg-subpixel", type=int, default=4)

    parser.add_argument(
        "--motion-kind",
        type=str,
        default="brainlike",
        help="Motion kind: none|sine|step|randomwalk|brainlike|elastic (default: %(default)s).",
    )
    parser.add_argument("--amp-px-list", type=str, default="0,0.5,1,2,3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-frame", type=int, default=None)
    parser.add_argument("--brainlike-rigid-kind", type=str, default="randomwalk")
    parser.add_argument("--brainlike-rigid-frac", type=float, default=0.7)
    parser.add_argument("--brainlike-elastic-frac", type=float, default=0.3)
    parser.add_argument("--brainlike-elastic-sigma-px", type=float, default=24.0)
    parser.add_argument("--brainlike-elastic-depth-decay-frac", type=float, default=0.8)
    parser.add_argument("--brainlike-elastic-rw-step-sigma", type=float, default=0.25)
    parser.add_argument("--brainlike-micro-jitter-frac", type=float, default=0.08)

    parser.add_argument("--snapshot-stride", type=int, default=4, help="STAP_SNAPSHOT_STRIDE (default: %(default)s).")
    parser.add_argument("--max-snapshots", type=int, default=64, help="STAP_MAX_SNAPSHOTS (default: %(default)s).")
    parser.add_argument(
        "--fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable STAP fast path batching (default: %(default)s).",
    )

    parser.add_argument("--vessel-quantile", type=float, default=0.99)
    parser.add_argument("--bg-tail-quantile", type=float, default=0.999)
    parser.add_argument("--fpr-target", type=float, default=1e-3)
    parser.add_argument("--pos-quantile", type=float, default=0.99)
    parser.add_argument("--topk-frac", type=float, default=0.01, help="Top-k fraction for overlap (default: %(default)s).")
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument("--crop-margin", type=int, default=None)
    parser.add_argument("--align-maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--align-psr-min",
        type=float,
        default=3.0,
        help="Minimum phase-correlation PSR to apply a shift (default: %(default)s).",
    )
    parser.add_argument(
        "--gt-align-frame",
        type=int,
        default=None,
        help="Reference frame index to derive GT alignment warp/shift (default: T//2).",
    )

    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/ulm_zenodo_7883227"))
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, default=None)

    args = parser.parse_args()

    block_ids = _parse_int_list(args.block_ids)
    frame_slice, frame_tag = _parse_slice(args.frames)

    params = load_ulm_zenodo_7883227_params()
    prf_hz = float(args.prf_hz) if args.prf_hz is not None else float(params.frame_rate_hz)

    # Fast-path and snapshot subsampling controls (recorded in bundle meta_extra).
    if bool(args.fast_path):
        os.environ["STAP_FAST_PATH"] = "1"
    else:
        os.environ["STAP_FAST_PATH"] = "0"
    os.environ["STAP_SNAPSHOT_STRIDE"] = str(max(1, int(args.snapshot_stride)))
    os.environ["STAP_MAX_SNAPSHOTS"] = str(max(1, int(args.max_snapshots)))

    amp_list = _parse_float_list(args.amp_px_list)
    if 0.0 not in amp_list:
        amp_list = [0.0] + amp_list
    amp_list = sorted(set(float(a) for a in amp_list))

    flow_low_hz, flow_high_hz, alias_center_hz, alias_hw_hz = _profile_to_bands(args.profile)

    # Baseline settings
    baseline_type = str(args.baseline_type).strip().lower()
    svd_rank: int | None
    if baseline_type == "mc_svd":
        r = str(args.svd_rank).strip().lower()
        svd_rank = None if r in {"none", "auto", "energy", "ef"} else int(r)
        svd_keep_min = None
        svd_keep_max = None
    else:
        svd_rank = None
        svd_keep_min = int(args.svd_keep_min)
        svd_keep_max = int(args.svd_keep_max)

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    if args.out_png is not None:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for block_id in block_ids:
        Icube0 = load_ulm_block_iq(int(block_id), frames=frame_slice, cache_dir=args.cache_dir)
        T, H, W = Icube0.shape
        block_tag = f"block{int(block_id):03d}_{frame_tag}"
        t_gt = int(args.gt_align_frame) if args.gt_align_frame is not None else int(T // 2)
        t_gt = int(np.clip(t_gt, 0, max(0, T - 1)))
        ref_bmode = np.mean(np.abs(Icube0), axis=0).astype(np.float32, copy=False)

        meta_common = {
            "orig_data": {
                "dataset": "ULM_Zenodo_7883227",
                "block_id": int(block_id),
                "frames_spec": args.frames,
                "cache_dir": str(args.cache_dir),
                "param_json": asdict(params),
            },
            "ulm_profile": {
                "name": str(args.profile),
                "flow_low_hz": float(flow_low_hz),
                "flow_high_hz": float(flow_high_hz),
                "alias_center_hz": float(alias_center_hz),
                "alias_half_width_hz": float(alias_hw_hz),
                "prf_hz": float(prf_hz),
            },
            "motion_injection": {
                "kind": str(args.motion_kind),
                "seed": int(args.seed),
                "amp_px_list": amp_list,
            },
            "eval_fairness": {
                "topk_frac": float(args.topk_frac),
                "align_psr_min": float(args.align_psr_min),
                "gt_align_frame": int(t_gt),
            },
            "stap_fast_path": {
                "enabled": bool(args.fast_path),
                "snapshot_stride": int(os.environ["STAP_SNAPSHOT_STRIDE"]),
                "max_snapshots": int(os.environ["STAP_MAX_SNAPSHOTS"]),
            },
        }

        # Reference (amp=0) defines per-block proxy masks and thresholds.
        ref_mask_flow = ref_mask_bg = None
        ref_base_score = ref_stap_score = None
        thr_vessel_base = thr_vessel_stap = None
        thr_bg_tail_base = thr_bg_tail_stap = None
        thr_base_fpr_ref = thr_stap_fpr_ref = None
        pos_base_ref = pos_stap_ref = pos_shared_ref = None

        for amp_px in amp_list:
            kind = str(args.motion_kind).strip().lower()
            motion_tele: dict[str, float] = {}
            gt_align_kind = "none"
            gt_shift_dy = gt_shift_dx = 0.0
            gt_dy_field: np.ndarray | None = None
            gt_dx_field: np.ndarray | None = None
            cube = Icube0
            if kind in {"sine", "sin", "drift_sine", "step", "burst", "rw", "randomwalk"}:
                dy, dx = _motion_shifts(
                    kind,
                    T=T,
                    amp_px=float(amp_px),
                    seed=int(args.seed),
                    step_frame=args.step_frame,
                )
                cube = _apply_translation_per_frame(Icube0, dy, dx)
                gt_align_kind = "translation"
                gt_shift_dy = float(dy[t_gt])
                gt_shift_dx = float(dx[t_gt])
            elif kind in {"brainlike", "elastic"}:
                # "elastic" is a nonrigid-only variant of "brainlike" (no rigid, no micro-jitter).
                if kind == "elastic":
                    rigid_kind = "none"
                    rigid_frac = 0.0
                    elastic_frac = 1.0
                    micro_jitter_frac = 0.0
                else:
                    rigid_kind = str(args.brainlike_rigid_kind)
                    rigid_frac = float(args.brainlike_rigid_frac)
                    elastic_frac = float(args.brainlike_elastic_frac)
                    micro_jitter_frac = float(args.brainlike_micro_jitter_frac)
                dy_field, dx_field, motion_tele = _brainlike_displacement_fields(
                    T=T,
                    H=H,
                    W=W,
                    amp_px=float(amp_px),
                    seed=int(args.seed),
                    rigid_kind=rigid_kind,
                    rigid_frac=rigid_frac,
                    elastic_frac=elastic_frac,
                    elastic_sigma_px=float(args.brainlike_elastic_sigma_px),
                    elastic_depth_decay_frac=float(args.brainlike_elastic_depth_decay_frac),
                    elastic_rw_step_sigma=float(args.brainlike_elastic_rw_step_sigma),
                    micro_jitter_frac=micro_jitter_frac,
                )
                cube = _apply_warp_per_frame(Icube0, dy_field, dx_field)
                gt_align_kind = "warp"
                gt_dy_field = dy_field[t_gt].astype(np.float32, copy=False)
                gt_dx_field = dx_field[t_gt].astype(np.float32, copy=False)
            elif kind in {"none", "off"}:
                cube = Icube0
            else:
                raise ValueError(f"Unsupported motion-kind {kind!r}.")

            amp_tag = _format_amp_tag(float(amp_px))
            dataset_name = f"ulm7883227_{block_tag}_{kind}_amp{amp_tag}_seed{int(args.seed):03d}"
            bmode = np.mean(np.abs(cube), axis=0).astype(np.float32, copy=False)

            paths = write_acceptance_bundle_from_icube(
                out_root=Path(args.out_root),
                dataset_name=dataset_name,
                Icube=cube,
                prf_hz=float(prf_hz),
                tile_hw=(int(args.tile_h), int(args.tile_w)),
                tile_stride=int(args.tile_stride),
                Lt=int(args.Lt),
                diag_load=float(args.diag_load),
                cov_estimator=str(args.cov_estimator),
                huber_c=float(args.huber_c),
                fd_span_mode="flow_band",
                baseline_type=str(args.baseline_type),
                reg_enable=bool(args.reg_enable),
                reg_subpixel=max(1, int(args.reg_subpixel)),
                svd_rank=svd_rank,
                svd_energy_frac=float(args.svd_energy_frac),
                svd_keep_min=svd_keep_min,
                svd_keep_max=svd_keep_max,
                # Telemetry bands (also used for fd_span_mode="flow_band" in slow path).
                band_ratio_flow_low_hz=float(flow_low_hz),
                band_ratio_flow_high_hz=float(flow_high_hz),
                band_ratio_alias_center_hz=float(alias_center_hz),
                band_ratio_alias_width_hz=float(alias_hw_hz),
                score_ka_v2_enable=False,
                meta_extra={
                    **meta_common,
                    "motion_injection_run": {
                        "amp_px": float(amp_px),
                        "gt_align_kind": str(gt_align_kind),
                        "gt_align_frame": int(t_gt),
                        "gt_shift_dy": float(gt_shift_dy),
                        "gt_shift_dx": float(gt_shift_dx),
                        "brainlike": motion_tele,
                    },
                },
            )

            bundle_dir = Path(paths["meta"]).parent
            meta = json.loads(Path(paths["meta"]).read_text())
            base_score_map = np.load(bundle_dir / "score_base.npy").astype(np.float32, copy=False)
            stap_score_map = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float32, copy=False)
            mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
            mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)

            if float(amp_px) == 0.0:
                ref_mask_flow = mask_flow
                ref_mask_bg = mask_bg
                ref_base_score = np.asarray(base_score_map, dtype=np.float64)
                ref_stap_score = np.asarray(stap_score_map, dtype=np.float64)

                q_vessel = float(args.vessel_quantile)
                thr_vessel_base = _safe_quantile(ref_base_score, q_vessel)
                thr_vessel_stap = _safe_quantile(ref_stap_score, q_vessel)
                vessel_base_ref = ref_base_score >= float(thr_vessel_base)
                vessel_stap_ref = ref_stap_score >= float(thr_vessel_stap)
                vessel_ref = vessel_base_ref & vessel_stap_ref

                thr_bg_tail_base = _safe_quantile(ref_base_score[ref_mask_bg], float(args.bg_tail_quantile))
                thr_bg_tail_stap = _safe_quantile(ref_stap_score[ref_mask_bg], float(args.bg_tail_quantile))

                thr_base_fpr_ref = _threshold_at_fpr(ref_base_score, ref_mask_bg, float(args.fpr_target))
                thr_stap_fpr_ref = _threshold_at_fpr(ref_stap_score, ref_mask_bg, float(args.fpr_target))

                q_pos = float(np.clip(float(args.pos_quantile), 0.0, 1.0))
                thr_pos_base = _safe_quantile(ref_base_score, q_pos)
                thr_pos_stap = _safe_quantile(ref_stap_score, q_pos)
                pos_base_ref = ref_base_score >= float(thr_pos_base)
                pos_stap_ref = ref_stap_score >= float(thr_pos_stap)
                pos_shared_ref = pos_base_ref & pos_stap_ref

            if (
                ref_mask_flow is None
                or ref_mask_bg is None
                or ref_base_score is None
                or ref_stap_score is None
                or thr_vessel_base is None
                or thr_vessel_stap is None
                or thr_bg_tail_base is None
                or thr_bg_tail_stap is None
                or thr_base_fpr_ref is None
                or thr_stap_fpr_ref is None
                or pos_base_ref is None
                or pos_stap_ref is None
                or pos_shared_ref is None
            ):
                raise RuntimeError("Reference (amp=0) run failed to initialize.")

            # Optional crop for similarity metrics (avoid boundary artifacts).
            if args.crop_margin is None:
                margin = int(math.ceil(max(0.0, float(amp_px)))) + 2
            else:
                margin = int(args.crop_margin)
            if margin * 2 >= ref_base_score.shape[0] or margin * 2 >= ref_base_score.shape[1]:
                margin = 0
            sl = (
                slice(margin, ref_base_score.shape[0] - margin),
                slice(margin, ref_base_score.shape[1] - margin),
            )

            base_map = np.asarray(base_score_map, dtype=np.float32)
            stap_map = np.asarray(stap_score_map, dtype=np.float32)
            ref_base_map = np.asarray(ref_base_score, dtype=np.float32)
            ref_stap_map = np.asarray(ref_stap_score, dtype=np.float32)

            # Precompute reference top-k masks / quantiles for degeneracy checks.
            ref_base_sl = ref_base_map[sl]
            ref_stap_sl = ref_stap_map[sl]
            ref_base_topk = _topk_mask(ref_base_sl, float(args.topk_frac))
            ref_stap_topk = _topk_mask(ref_stap_sl, float(args.topk_frac))

            # Alignment modes:
            #   - none: no post-hoc alignment
            #   - bmode: common rigid alignment estimated from method-independent B-mode proxy
            #   - gt: common alignment from injected motion (rigid shift or single-frame warp at t_gt)
            #   - self: per-method phase-correlation (diagnostic; can be criticized as "oracle per method")
            if bool(args.align_maps):
                dyb, dxb, psrb = _phasecorr_shift(ref_base_map, base_map, upsample=4)
                dys, dxs, psrs = _phasecorr_shift(ref_stap_map, stap_map, upsample=4)
                if float(psrb) < float(args.align_psr_min):
                    dyb = dxb = 0.0
                if float(psrs) < float(args.align_psr_min):
                    dys = dxs = 0.0
            else:
                dyb = dxb = psrb = 0.0
                dys = dxs = psrs = 0.0

            # none
            base_none = base_map
            stap_none = stap_map

            # bmode-common
            dybm, dxbm, psrbm = _phasecorr_shift(ref_bmode, bmode, upsample=4)
            if float(psrbm) < float(args.align_psr_min):
                dybm = dxbm = 0.0
            base_bmode = ndi.shift(base_map, shift=(dybm, dxbm), order=1, mode="nearest", prefilter=False)
            stap_bmode = ndi.shift(stap_map, shift=(dybm, dxbm), order=1, mode="nearest", prefilter=False)

            # gt-common
            if gt_align_kind == "translation":
                base_gt = ndi.shift(
                    base_map,
                    shift=(-float(gt_shift_dy), -float(gt_shift_dx)),
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
                stap_gt = ndi.shift(
                    stap_map,
                    shift=(-float(gt_shift_dy), -float(gt_shift_dx)),
                    order=1,
                    mode="nearest",
                    prefilter=False,
                )
                gt_disp_rms = float(math.sqrt(float(gt_shift_dy) ** 2 + float(gt_shift_dx) ** 2))
            elif gt_align_kind == "warp" and gt_dy_field is not None and gt_dx_field is not None:
                base_gt = _warp2d(base_map, dy=-gt_dy_field, dx=-gt_dx_field)
                stap_gt = _warp2d(stap_map, dy=-gt_dy_field, dx=-gt_dx_field)
                gt_disp_rms = float(np.sqrt(np.mean(gt_dy_field * gt_dy_field + gt_dx_field * gt_dx_field)))
            else:
                base_gt = base_map
                stap_gt = stap_map
                gt_disp_rms = 0.0

            # self (per method)
            base_self = ndi.shift(base_map, shift=(dyb, dxb), order=1, mode="nearest", prefilter=False)
            stap_self = ndi.shift(stap_map, shift=(dys, dxs), order=1, mode="nearest", prefilter=False)

            # Fairness correlations under each alignment.
            corr_base_none = _pearson_corr(base_none[sl], ref_base_sl)
            corr_stap_none = _pearson_corr(stap_none[sl], ref_stap_sl)
            corr_base_bmode = _pearson_corr(base_bmode[sl], ref_base_sl)
            corr_stap_bmode = _pearson_corr(stap_bmode[sl], ref_stap_sl)
            corr_base_gt = _pearson_corr(base_gt[sl], ref_base_sl)
            corr_stap_gt = _pearson_corr(stap_gt[sl], ref_stap_sl)
            corr_base = _pearson_corr(base_self[sl], ref_base_sl)
            corr_stap = _pearson_corr(stap_self[sl], ref_stap_sl)

            # Error metrics (keep on self-aligned maps for continuity with previous runs).
            nrmse_base = _nrmse(base_self[sl], ref_base_sl)
            nrmse_stap = _nrmse(stap_self[sl], ref_stap_sl)

            std_base = float(np.std(base_self[sl]))
            std_stap = float(np.std(stap_self[sl]))
            ent_base = _shannon_entropy(base_self[sl])
            ent_stap = _shannon_entropy(stap_self[sl])

            bg_sl_mask = ref_mask_bg[sl]
            var_ref_base_bg = float(np.var(ref_base_map[sl][bg_sl_mask])) + 1e-12
            var_ref_stap_bg = float(np.var(ref_stap_map[sl][bg_sl_mask])) + 1e-12
            bg_var_ratio_base = float(np.var(base_self[sl][bg_sl_mask]) / var_ref_base_bg)
            bg_var_ratio_stap = float(np.var(stap_self[sl][bg_sl_mask]) / var_ref_stap_bg)

            # Degeneracy checks (alignment-dependent).
            std_ratio_base_none = _std_ratio(base_none[sl], ref_base_sl)
            std_ratio_stap_none = _std_ratio(stap_none[sl], ref_stap_sl)
            std_ratio_base_bmode = _std_ratio(base_bmode[sl], ref_base_sl)
            std_ratio_stap_bmode = _std_ratio(stap_bmode[sl], ref_stap_sl)
            std_ratio_base_gt = _std_ratio(base_gt[sl], ref_base_sl)
            std_ratio_stap_gt = _std_ratio(stap_gt[sl], ref_stap_sl)

            topk_j_base_none = _jaccard(_topk_mask(base_none[sl], float(args.topk_frac)), ref_base_topk)
            topk_j_stap_none = _jaccard(_topk_mask(stap_none[sl], float(args.topk_frac)), ref_stap_topk)
            topk_j_base_bmode = _jaccard(_topk_mask(base_bmode[sl], float(args.topk_frac)), ref_base_topk)
            topk_j_stap_bmode = _jaccard(_topk_mask(stap_bmode[sl], float(args.topk_frac)), ref_stap_topk)
            topk_j_base_gt = _jaccard(_topk_mask(base_gt[sl], float(args.topk_frac)), ref_base_topk)
            topk_j_stap_gt = _jaccard(_topk_mask(stap_gt[sl], float(args.topk_frac)), ref_stap_topk)

            q999_ratio_base_none = _quantile_ratio(base_none[sl], ref_base_sl, 0.999)
            q999_ratio_stap_none = _quantile_ratio(stap_none[sl], ref_stap_sl, 0.999)
            q999_ratio_base_bmode = _quantile_ratio(base_bmode[sl], ref_base_sl, 0.999)
            q999_ratio_stap_bmode = _quantile_ratio(stap_bmode[sl], ref_stap_sl, 0.999)
            q999_ratio_base_gt = _quantile_ratio(base_gt[sl], ref_base_sl, 0.999)
            q999_ratio_stap_gt = _quantile_ratio(stap_gt[sl], ref_stap_sl, 0.999)

            # Shared-structure overlap using fixed no-motion thresholds.
            vessel_base = base_score_map >= float(thr_vessel_base)
            vessel_stap = stap_score_map >= float(thr_vessel_stap)
            j_base = _jaccard(vessel_base, pos_base_ref)
            j_stap = _jaccard(vessel_stap, pos_stap_ref)

            # BG tail clusters/area using per-method no-motion BG thresholds.
            hit_base = ref_mask_bg & (base_score_map >= float(thr_bg_tail_base))
            hit_stap = ref_mask_bg & (stap_score_map >= float(thr_bg_tail_stap))
            area_base = int(np.sum(hit_base))
            area_stap = int(np.sum(hit_stap))
            clust_base = _connected_components(hit_base, connectivity=int(args.connectivity))
            clust_stap = _connected_components(hit_stap, connectivity=int(args.connectivity))

            flow_jacc = _jaccard(mask_flow, ref_mask_flow)

            # Proxy TPR@FPR curves: fixed threshold (ref-calibrated) and per-run threshold.
            pos_flow = ref_mask_flow
            pos_shared = pos_shared_ref
            neg = ref_mask_bg

            thr_base_run = _threshold_at_fpr(base_score_map, neg, float(args.fpr_target))
            thr_stap_run = _threshold_at_fpr(stap_score_map, neg, float(args.fpr_target))

            tpr_base_flow_fixed, fpr_base_flow_fixed = _tpr_fpr_at_threshold(
                base_score_map, pos_mask=pos_flow, neg_mask=neg, thr=float(thr_base_fpr_ref)
            )
            tpr_stap_flow_fixed, fpr_stap_flow_fixed = _tpr_fpr_at_threshold(
                stap_score_map, pos_mask=pos_flow, neg_mask=neg, thr=float(thr_stap_fpr_ref)
            )
            tpr_base_flow_run, fpr_base_flow_run = _tpr_fpr_at_threshold(
                base_score_map, pos_mask=pos_flow, neg_mask=neg, thr=float(thr_base_run)
            )
            tpr_stap_flow_run, fpr_stap_flow_run = _tpr_fpr_at_threshold(
                stap_score_map, pos_mask=pos_flow, neg_mask=neg, thr=float(thr_stap_run)
            )

            tpr_base_shared_run, _ = _tpr_fpr_at_threshold(
                base_score_map, pos_mask=pos_shared, neg_mask=neg, thr=float(thr_base_run)
            )
            tpr_stap_shared_run, _ = _tpr_fpr_at_threshold(
                stap_score_map, pos_mask=pos_shared, neg_mask=neg, thr=float(thr_stap_run)
            )

            # Contract telemetry (label-free, optional).
            ka_v2 = meta.get("ka_contract_v2") or {}
            ka_metrics = (ka_v2.get("metrics") or {}) if isinstance(ka_v2, dict) else {}
            tele = meta.get("stap_fallback_telemetry") or {}

            rows.append(
                {
                    "bundle": str(bundle_dir),
                    "bundle_name": bundle_dir.name,
                    "block_id": int(block_id),
                    "frames": args.frames,
                    "prf_hz": float(prf_hz),
                    "profile": str(args.profile),
                    "amp_px": float(amp_px),
                    "motion_kind": str(args.motion_kind),
                    "seed": int(args.seed),
                    "Lt": int(args.Lt),
                    "tile_hw": f"{int(args.tile_h)}x{int(args.tile_w)}",
                    "tile_stride": int(args.tile_stride),
                    "cov_estimator": str(args.cov_estimator),
                    "diag_load": float(args.diag_load),
                    "baseline_type": baseline_type,
                    "svd_energy_frac": float(args.svd_energy_frac) if baseline_type == "mc_svd" else None,
                    "svd_keep_min": int(svd_keep_min) if svd_keep_min is not None else None,
                    "svd_keep_max": int(svd_keep_max) if svd_keep_max is not None else None,
                    "reg_enable": bool(args.reg_enable),
                    "fast_path": bool(args.fast_path),
                    "snapshot_stride": int(os.environ["STAP_SNAPSHOT_STRIDE"]),
                    "max_snapshots": int(os.environ["STAP_MAX_SNAPSHOTS"]),
                    "flow_low_hz": float(flow_low_hz),
                    "flow_high_hz": float(flow_high_hz),
                    "alias_center_hz": float(alias_center_hz),
                    "alias_half_width_hz": float(alias_hw_hz),
                    # Similarity metrics (aligned + cropped).
                    "align_psr_min": float(args.align_psr_min),
                    "align_shift_base_dy": float(dyb),
                    "align_shift_base_dx": float(dxb),
                    "align_psr_base": float(psrb),
                    "align_applied_base": bool(float(psrb) >= float(args.align_psr_min)),
                    "align_shift_stap_dy": float(dys),
                    "align_shift_stap_dx": float(dxs),
                    "align_psr_stap": float(psrs),
                    "align_applied_stap": bool(float(psrs) >= float(args.align_psr_min)),
                    "align_shift_bmode_dy": float(dybm),
                    "align_shift_bmode_dx": float(dxbm),
                    "align_psr_bmode": float(psrbm),
                    "align_applied_bmode": bool(float(psrbm) >= float(args.align_psr_min)),
                    "gt_align_kind": str(gt_align_kind),
                    "gt_align_frame": int(t_gt),
                    "gt_align_shift_dy": float(gt_shift_dy),
                    "gt_align_shift_dx": float(gt_shift_dx),
                    "gt_align_disp_rms": float(gt_disp_rms),
                    "topk_frac": float(args.topk_frac),
                    # Fairness ablation correlations.
                    "corr_score_base_align_none": float(corr_base_none),
                    "corr_score_stap_align_none": float(corr_stap_none),
                    "corr_score_base_align_bmode": float(corr_base_bmode),
                    "corr_score_stap_align_bmode": float(corr_stap_bmode),
                    "corr_score_base_align_gt": float(corr_base_gt),
                    "corr_score_stap_align_gt": float(corr_stap_gt),
                    "corr_score_base": float(corr_base),
                    "corr_score_stap": float(corr_stap),
                    "nrmse_score_base": float(nrmse_base),
                    "nrmse_score_stap": float(nrmse_stap),
                    "std_score_base": float(std_base),
                    "std_score_stap": float(std_stap),
                    "ent_score_base": float(ent_base),
                    "ent_score_stap": float(ent_stap),
                    "bg_var_ratio_base": float(bg_var_ratio_base),
                    "bg_var_ratio_stap": float(bg_var_ratio_stap),
                    # Degeneracy checks.
                    "std_ratio_score_base_align_none": float(std_ratio_base_none),
                    "std_ratio_score_stap_align_none": float(std_ratio_stap_none),
                    "std_ratio_score_base_align_bmode": float(std_ratio_base_bmode),
                    "std_ratio_score_stap_align_bmode": float(std_ratio_stap_bmode),
                    "std_ratio_score_base_align_gt": float(std_ratio_base_gt),
                    "std_ratio_score_stap_align_gt": float(std_ratio_stap_gt),
                    "topk_jacc_score_base_align_none": float(topk_j_base_none),
                    "topk_jacc_score_stap_align_none": float(topk_j_stap_none),
                    "topk_jacc_score_base_align_bmode": float(topk_j_base_bmode),
                    "topk_jacc_score_stap_align_bmode": float(topk_j_stap_bmode),
                    "topk_jacc_score_base_align_gt": float(topk_j_base_gt),
                    "topk_jacc_score_stap_align_gt": float(topk_j_stap_gt),
                    "q999_ratio_score_base_align_none": float(q999_ratio_base_none),
                    "q999_ratio_score_stap_align_none": float(q999_ratio_stap_none),
                    "q999_ratio_score_base_align_bmode": float(q999_ratio_base_bmode),
                    "q999_ratio_score_stap_align_bmode": float(q999_ratio_stap_bmode),
                    "q999_ratio_score_base_align_gt": float(q999_ratio_base_gt),
                    "q999_ratio_score_stap_align_gt": float(q999_ratio_stap_gt),
                    # Proxy stability.
                    "flow_mask_jaccard": float(flow_jacc),
                    "jaccard_vessel_base": float(j_base),
                    "jaccard_vessel_stap": float(j_stap),
                    # BG tail hygiene (ref-calibrated thresholds).
                    "bg_tail_area_base": int(area_base),
                    "bg_tail_area_stap": int(area_stap),
                    "bg_tail_clusters_base": int(clust_base),
                    "bg_tail_clusters_stap": int(clust_stap),
                    # Proxy TPR@FPR (fixed and matched).
                    "fpr_target": float(args.fpr_target),
                    "tpr_base_flow_fixedthr": float(tpr_base_flow_fixed),
                    "fpr_base_flow_fixedthr": float(fpr_base_flow_fixed),
                    "tpr_stap_flow_fixedthr": float(tpr_stap_flow_fixed),
                    "fpr_stap_flow_fixedthr": float(fpr_stap_flow_fixed),
                    "tpr_base_flow_at_fpr": float(tpr_base_flow_run),
                    "fpr_base_flow_at_fpr": float(fpr_base_flow_run),
                    "tpr_stap_flow_at_fpr": float(tpr_stap_flow_run),
                    "fpr_stap_flow_at_fpr": float(fpr_stap_flow_run),
                    "tpr_base_shared_at_fpr": float(tpr_base_shared_run),
                    "tpr_stap_shared_at_fpr": float(tpr_stap_shared_run),
                    # Contract metrics (telemetry).
                    "ka_state": ka_v2.get("state") if isinstance(ka_v2, dict) else None,
                    "ka_reason": ka_v2.get("reason") if isinstance(ka_v2, dict) else None,
                    "ka_pf_peak_flow": ka_metrics.get("pf_peak_flow"),
                    "ka_guard_q90": ka_metrics.get("guard_q90"),
                    "ka_iqr_alias_bg": ka_metrics.get("iqr_alias_bg"),
                    "ka_delta_tail": ka_metrics.get("delta_tail"),
                    "tiles_skipped_flow0": tele.get("stap_tiles_skipped_flow0") if isinstance(tele, dict) else None,
                    # Brainlike telemetry
                    "brainlike_disp_rms_px": motion_tele.get("disp_rms_px"),
                    "brainlike_disp_p90_px": motion_tele.get("disp_p90_px"),
                }
            )

            print(
                "[ulm-motion]"
                f" block={int(block_id):03d}"
                f" amp={float(amp_px):.2f}px"
                f" corr_none={corr_base_none:.3f}/{corr_stap_none:.3f}"
                f" corr_bmode={corr_base_bmode:.3f}/{corr_stap_bmode:.3f}"
                f" corr_gt={corr_base_gt:.3f}/{corr_stap_gt:.3f}"
                f" tpr_flow@fpr(base/stap)={tpr_base_flow_run:.3f}/{tpr_stap_flow_run:.3f}"
                f" clusters(base/stap)={clust_base}/{clust_stap}"
                f" ka={rows[-1]['ka_state']}({rows[-1]['ka_reason']})"
            )

    # Write CSV
    if not rows:
        raise RuntimeError("No rows produced.")
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[ulm-motion] wrote {len(rows)} rows to {args.out_csv}")

    # Write JSON summary (config + rows)
    summary = {"config": _jsonify(vars(args)), "rows": rows}
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"[ulm-motion] wrote {args.out_json}")

    if args.out_png is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.rcParams.update(
                {
                    "font.family": "serif",
                    "mathtext.fontset": "cm",
                    "font.size": 10,
                    "axes.titlesize": 11,
                    "axes.labelsize": 10,
                    "legend.fontsize": 9,
                    "pdf.fonttype": 42,
                    "ps.fonttype": 42,
                }
            )

            amps_unique = sorted({float(r["amp_px"]) for r in rows})

            def _summarize(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                vals: list[list[float]] = [[] for _ in amps_unique]
                idx = {a: i for i, a in enumerate(amps_unique)}
                for r in rows:
                    vals[idx[float(r["amp_px"])]].append(float(r[key]))
                med = np.array([float(np.median(v)) for v in vals], dtype=float)
                q25 = np.array([float(np.quantile(v, 0.25)) for v in vals], dtype=float)
                q75 = np.array([float(np.quantile(v, 0.75)) for v in vals], dtype=float)
                return med, q25, q75

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

            for ax, mode, title in [
                (axes[0], "none", "No alignment"),
                (axes[1], "bmode", "Common B-mode alignment"),
                (axes[2], "gt", "GT alignment (injected)"),
            ]:
                corr_base, corr_base_q25, corr_base_q75 = _summarize(f"corr_score_base_align_{mode}")
                corr_stap, corr_stap_q25, corr_stap_q75 = _summarize(f"corr_score_stap_align_{mode}")
                ax.plot(amps_unique, corr_base, "o-", color="#666666", label=r"Baseline $S_{\mathrm{base}}$")
                ax.fill_between(amps_unique, corr_base_q25, corr_base_q75, alpha=0.18, color="#666666", linewidth=0)
                ax.plot(amps_unique, corr_stap, "o-", color="#1f77b4", label=r"STAP $S_{\mathrm{stap,pre}}$")
                ax.fill_between(amps_unique, corr_stap_q25, corr_stap_q75, alpha=0.18, color="#1f77b4", linewidth=0)
                ax.set_xlabel("motion amplitude (px)")
                ax.set_ylabel("corr vs no-motion")
                ax.set_title(title)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.3)
                ax.legend()

            fig.savefig(args.out_png, dpi=300, bbox_inches="tight", pad_inches=0.06)
            plt.close(fig)
            print(f"[ulm-motion] wrote {args.out_png}")
        except Exception as exc:
            print(f"[ulm-motion] failed to write plot: {exc}")


if __name__ == "__main__":
    main()
