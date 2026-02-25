from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from sim.kwave.common import _phasecorr_shift
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> tuple[list[int] | None, str]:
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
    frames = list(range(start, stop, step))
    tag = f"f{start}_{stop}"
    return frames, tag


def _parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for part in (spec or "").split(","):
        part = part.strip()
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
        raise ValueError("Cannot compute quantile on empty / non-finite array.")
    return float(np.quantile(x, q))


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
    """Histogram-based Shannon entropy of a real array (higher = more diverse)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = x - float(np.mean(x))
    std = float(np.std(x)) + 1e-12
    x = x / std
    x = np.clip(x, -6.0, 6.0)
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


def _threshold_at_fpr(score: np.ndarray, neg_mask: np.ndarray, fpr: float) -> float:
    score = np.asarray(score, dtype=np.float64)
    neg = np.asarray(neg_mask, dtype=bool)
    vals = score[neg]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    fpr = float(np.clip(fpr, 1e-9, 1.0 - 1e-9))
    return float(np.quantile(vals, 1.0 - fpr))


def _apply_translation_per_frame(iq: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """Subpixel translation of complex IQ per frame using linear interpolation."""
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
    """Spatial warp of complex IQ per frame using a per-frame displacement field.

    Args:
      iq: complex IQ cube (T,H,W)
      dy_field/dx_field: displacement fields in pixels, shape (T,H,W).

    Returns:
      warped iq cube (T,H,W) complex64.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    T, H, W = iq.shape
    if dy_field.shape != (T, H, W) or dx_field.shape != (T, H, W):
        raise ValueError(
            f"dy_field/dx_field must have shape (T,H,W)={(T,H,W)}, got {dy_field.shape} / {dx_field.shape}"
        )
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    coords = np.empty((2, H, W), dtype=np.float32)
    out = np.empty_like(iq)
    for t in range(T):
        coords[0] = yy - dy_field[t].astype(np.float32, copy=False)
        coords[1] = xx - dx_field[t].astype(np.float32, copy=False)
        re = ndi.map_coordinates(iq[t].real, coords, order=1, mode="nearest", prefilter=False)
        im = ndi.map_coordinates(iq[t].imag, coords, order=1, mode="nearest", prefilter=False)
        out[t] = re.astype(np.float32, copy=False) + 1j * im.astype(np.float32, copy=False)
    return out


def _smooth_unit_field(H: int, W: int, *, sigma_px: float, rng: np.random.Generator) -> np.ndarray:
    field = rng.standard_normal((H, W)).astype(np.float32)
    field = ndi.gaussian_filter(field, sigma=float(max(sigma_px, 0.5)), mode="nearest")
    field -= float(np.mean(field))
    rms = float(np.sqrt(np.mean(field * field))) + 1e-12
    field /= rms
    return field.astype(np.float32, copy=False)


def _normalized_random_walk(T: int, *, rng: np.random.Generator, step_sigma: float = 1.0) -> np.ndarray:
    steps = rng.normal(scale=float(step_sigma), size=T).astype(np.float32)
    walk = np.cumsum(steps).astype(np.float32)
    walk -= float(np.mean(walk))
    rms = float(np.sqrt(np.mean(walk * walk))) + 1e-12
    return (walk / rms).astype(np.float32, copy=False)


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
    """Generate Brain-like rigid + elastic displacement fields.

    This mirrors the intent of Brain-SkullOR "residual rigid and elastic motion":
      - a rigid bulk component (translation) which phase-correlation registration can partially correct,
      - plus a smooth, spatially varying elastic component which translation-only registration cannot remove,
      - plus small per-frame micro-jitter.

    The returned dy/dx fields have shape (T,H,W) in pixels.
    """
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

    # ---- Rigid bulk translation ----
    rigid_amp = amp_px * rigid_frac
    dy_rigid_1d, dx_rigid_1d = _motion_shifts(
        rigid_kind,
        T=T,
        amp_px=rigid_amp,
        seed=int(seed) + 1,
        step_frame=None,
    )

    # ---- Elastic warp: smooth spatial fields with slow random-walk time coefficients ----
    elastic_amp = amp_px * elastic_frac
    if elastic_amp <= 0.0:
        dy_el = np.zeros((T, H, W), dtype=np.float32)
        dx_el = np.zeros((T, H, W), dtype=np.float32)
        a_t = np.zeros((T,), dtype=np.float32)
        b_t = np.zeros((T,), dtype=np.float32)
    else:
        dy0 = _smooth_unit_field(H, W, sigma_px=float(elastic_sigma_px), rng=rng)
        dx0 = _smooth_unit_field(H, W, sigma_px=float(elastic_sigma_px), rng=rng)

        # Depth weighting: stronger deformation near the surface, decaying with depth.
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

    # ---- Micro-jitter (residual) ----
    micro_std = amp_px * float(np.clip(micro_jitter_frac, 0.0, 1.0))
    if micro_std > 0.0:
        dy_j = rng.normal(scale=micro_std, size=T).astype(np.float32)
        dx_j = rng.normal(scale=micro_std, size=T).astype(np.float32)
    else:
        dy_j = np.zeros((T,), dtype=np.float32)
        dx_j = np.zeros((T,), dtype=np.float32)

    # Compose displacement fields.
    dy = dy_el + (dy_rigid_1d + dy_j)[:, None, None]
    dx = dx_el + (dx_rigid_1d + dx_j)[:, None, None]

    disp = np.sqrt(dy * dy + dx * dx)
    disp_rms = float(np.sqrt(np.mean(disp * disp)))
    disp_p90 = float(np.quantile(disp.ravel(), 0.90))
    tele = {
        "rigid_kind": str(rigid_kind),
        "rigid_frac": float(rigid_frac),
        "elastic_frac": float(elastic_frac),
        "elastic_sigma_px": float(elastic_sigma_px),
        "elastic_depth_decay_frac": float(elastic_depth_decay_frac),
        "elastic_rw_step_sigma": float(elastic_rw_step_sigma),
        "micro_jitter_frac": float(micro_jitter_frac),
        "disp_rms_px": disp_rms,
        "disp_p90_px": disp_p90,
    }
    return dy.astype(np.float32, copy=False), dx.astype(np.float32, copy=False), tele


def _motion_shifts(kind: str, T: int, amp_px: float, seed: int, step_frame: int | None) -> tuple[np.ndarray, np.ndarray]:
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
        # Random walk with total RMS roughly scaling with amp_px.
        steps = rng.normal(size=(T, 2)).astype(np.float32)
        steps -= steps.mean(axis=0, keepdims=True)
        walk = np.cumsum(steps, axis=0)
        walk -= walk.mean(axis=0, keepdims=True)
        scale = float(np.sqrt(np.mean(walk[:, 0] ** 2 + walk[:, 1] ** 2))) + 1e-12
        walk = (float(amp_px) / scale) * walk
        return walk[:, 0].astype(np.float32), walk[:, 1].astype(np.float32)

    raise ValueError(f"Unknown motion kind {kind!r}. Use one of: none, sine, step, randomwalk.")


def _apply_clutter_tone(
    iq: np.ndarray,
    *,
    prf_hz: float,
    f_hz: float,
    amp_rel: float,
    sigma_px: float,
    seed: int,
) -> np.ndarray:
    """Add a spatially smooth low-frequency complex tone to all pixels."""
    iq = np.asarray(iq, dtype=np.complex64)
    if amp_rel <= 0.0:
        return iq
    T, H, W = iq.shape
    rng = np.random.default_rng(int(seed))
    spatial = rng.normal(size=(H, W)).astype(np.float32)
    spatial = ndi.gaussian_filter(spatial, sigma=float(sigma_px), mode="nearest")
    spatial -= float(np.mean(spatial))
    spatial_rms = float(np.sqrt(np.mean(spatial * spatial))) + 1e-12
    spatial /= spatial_rms
    iq_rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) + 1e-12
    amp = float(amp_rel) * iq_rms
    phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
    tt = np.arange(T, dtype=np.float32) / float(prf_hz)
    tone = np.exp(1j * (2.0 * np.pi * float(f_hz) * tt + phase0)).astype(np.complex64)
    return iq + (amp * tone[:, None, None] * spatial[None, :, :]).astype(np.complex64, copy=False)


def _apply_global_phase_sine(iq: np.ndarray, *, prf_hz: float, f_hz: float, amp_rad: float) -> np.ndarray:
    """Multiply IQ by a global sinusoidal phase modulation (axial motion proxy)."""
    iq = np.asarray(iq, dtype=np.complex64)
    if amp_rad <= 0.0:
        return iq
    T = iq.shape[0]
    tt = np.arange(T, dtype=np.float32) / float(prf_hz)
    phi = float(amp_rad) * np.sin(2.0 * np.pi * float(f_hz) * tt)
    mod = np.exp(1j * phi).astype(np.complex64)
    return (iq * mod[:, None, None]).astype(np.complex64, copy=False)


def _profile_to_bands(profile: str) -> tuple[float, float, float, float]:
    """Return (flow_low, flow_high, alias_center, alias_half_width)."""
    p = (profile or "").strip().lower()
    if p in {"u", "shin_u", "ulm"}:
        return 60.0, 250.0, 400.0, 100.0
    if p in {"s", "shin_s", "strict"}:
        return 20.0, 200.0, 380.0, 120.0
    if p in {"l", "shin_l", "low"}:
        return 10.0, 120.0, 330.0, 170.0
    raise ValueError("Profile must be one of: U, S, L.")


def _format_amp_tag(amp_px: float) -> str:
    if amp_px == 0.0:
        return "0p00"
    s = f"{amp_px:.2f}".replace(".", "p").replace("-", "m")
    return s


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic motion-injection sweep for Shin RatBrain IQ (Fig3).\n"
            "Applies controlled per-frame translations to the IQ cube, reruns baseline+STAP (+ optional KA),\n"
            "and reports label-free robustness metrics (map correlation, vessel-mask overlap, background tail clusters).\n"
            "This supports a motion-robustness story without ground-truth labels."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"),
        help="Directory containing SizeInfo.dat and IQData*.dat (default: %(default)s).",
    )
    parser.add_argument("--iq-file", type=str, required=True, help="One IQData###.dat file name under --data-root.")
    parser.add_argument("--frames", type=str, default="0:128")
    parser.add_argument("--prf-hz", type=float, default=1000.0)
    parser.add_argument("--profile", type=str, default="U", help="Frozen Shin profile: U, S, or L (default: %(default)s).")
    parser.add_argument("--Lt", type=int, default=64)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)

    parser.add_argument("--baseline-type", type=str, default="mc_svd", choices=["mc_svd", "svd_bandpass"])
    parser.add_argument("--svd-rank", type=str, default="none", help="For mc_svd: int or 'none' (energy-frac).")
    parser.add_argument("--svd-energy-frac", type=float, default=0.95)
    parser.add_argument("--svd-keep-min", type=int, default=3)
    parser.add_argument("--svd-keep-max", type=int, default=40)

    parser.add_argument("--flow-mask-pd-quantile", type=float, default=0.99)
    parser.add_argument("--flow-mask-min-pixels", type=int, default=64)
    parser.add_argument("--flow-mask-union-default", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--flow-mask-mode",
        type=str,
        default="pd_auto",
        help="Flow mask mode passed to the bundle writer (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-depth-min-frac",
        type=float,
        default=0.0,
        help="Min depth fraction for pd_auto flow mask (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-depth-max-frac",
        type=float,
        default=1.0,
        help="Max depth fraction for pd_auto flow mask (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-erode-iters",
        type=int,
        default=0,
        help="Morphological erosion iterations for pd_auto flow mask (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-dilate-iters",
        type=int,
        default=1,
        help="Morphological dilation iterations for pd_auto flow mask (default: %(default)s).",
    )

    parser.add_argument(
        "--score-ka-v2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable score-space KA v2 (shrink-only) when contract permits (default: %(default)s).",
    )
    parser.add_argument(
        "--score-ka-v2-mode",
        type=str,
        default="safety",
        help="KA v2 application mode: safety|uplift|auto (default: %(default)s).",
    )

    parser.add_argument(
        "--motion-kind",
        type=str,
        default="sine",
        help=(
            "Motion kind: none|sine|step|randomwalk|brainlike|clutter_tone|phase_sine (default: %(default)s). "
            "Translation kinds (sine/step/randomwalk) use amp in pixels; clutter/phase use amp as a relative/tone magnitude."
        ),
    )
    parser.add_argument(
        "--amp-px-list",
        type=str,
        default="0,0.5,1,2,3",
        help=(
            "Comma-separated amplitudes. For translation motion, units are pixels; "
            "for clutter_tone, amp is relative to IQ RMS; for phase_sine, amp is radians."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-frame", type=int, default=None, help="For step motion: frame index where jump occurs.")
    parser.add_argument("--motion-f-hz", type=float, default=12.0, help="For clutter_tone/phase_sine: tone frequency (Hz).")
    parser.add_argument("--clutter-sigma-px", type=float, default=8.0, help="For clutter_tone: spatial smoothing sigma (px).")
    parser.add_argument(
        "--reg-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rigid phase-correlation registration in MC-SVD/SVD baselines (default: %(default)s).",
    )
    parser.add_argument("--reg-subpixel", type=int, default=4, help="Registration subpixel upsample factor (default: %(default)s).")
    parser.add_argument(
        "--brainlike-rigid-kind",
        type=str,
        default="randomwalk",
        help="For motion-kind=brainlike: rigid component kind (none|sine|step|randomwalk).",
    )
    parser.add_argument(
        "--brainlike-rigid-frac",
        type=float,
        default=0.7,
        help="For motion-kind=brainlike: fraction of amplitude in rigid motion (default: %(default)s).",
    )
    parser.add_argument(
        "--brainlike-elastic-frac",
        type=float,
        default=0.3,
        help="For motion-kind=brainlike: fraction of amplitude in elastic (nonrigid) motion (default: %(default)s).",
    )
    parser.add_argument(
        "--brainlike-elastic-sigma-px",
        type=float,
        default=24.0,
        help="For motion-kind=brainlike: spatial smoothing sigma (px) for elastic fields (default: %(default)s).",
    )
    parser.add_argument(
        "--brainlike-elastic-depth-decay-frac",
        type=float,
        default=0.8,
        help="For motion-kind=brainlike: depth decay fraction for elastic amplitude (default: %(default)s).",
    )
    parser.add_argument(
        "--brainlike-elastic-rw-step-sigma",
        type=float,
        default=0.25,
        help="For motion-kind=brainlike: random-walk step sigma for elastic time coefficients (default: %(default)s).",
    )
    parser.add_argument(
        "--brainlike-micro-jitter-frac",
        type=float,
        default=0.08,
        help="For motion-kind=brainlike: per-frame micro-jitter std as fraction of amp_px (default: %(default)s).",
    )

    parser.add_argument(
        "--vessel-quantile",
        type=float,
        default=0.99,
        help=(
            "Reference vessel/flow-proxy mask quantile on the detector score S=PD (default: %(default)s). "
            "Higher means a smaller, more confident proxy set."
        ),
    )
    parser.add_argument(
        "--bg-tail-quantile",
        type=float,
        default=0.999,
        help="BG tail threshold quantile on the detector score S=PD (default: %(default)s).",
    )
    parser.add_argument(
        "--fpr-target",
        type=float,
        default=1e-3,
        help="Proxy operating point for robustness curves: target FPR on bg proxy (default: %(default)s).",
    )
    parser.add_argument(
        "--pos-quantile",
        type=float,
        default=0.99,
        help=(
            "Quantile used to define a label-free positive proxy set from no-motion score maps "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    parser.add_argument("--crop-margin", type=int, default=None, help="Optional fixed crop margin (pixels) for map similarity.")
    parser.add_argument(
        "--align-maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase-correlate and align PD maps before similarity metrics (default: %(default)s).",
    )

    parser.add_argument("--out-root", type=Path, required=True, help="Output root for bundles.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path for metrics.")
    parser.add_argument("--out-png", type=Path, default=None, help="Optional output plot path.")

    args = parser.parse_args()

    frames, frame_tag = _parse_slice(args.frames)
    iq_path = args.data_root / args.iq_file
    if not iq_path.is_file():
        raise FileNotFoundError(iq_path)

    info = load_shin_metadata(args.data_root)
    Icube0 = load_shin_iq(iq_path, info, frames=frames)

    amp_list = _parse_float_list(args.amp_px_list)
    if 0.0 not in amp_list:
        amp_list = [0.0] + amp_list
    amp_list = sorted(set(float(a) for a in amp_list))

    flow_low_hz, flow_high_hz, alias_center_hz, alias_hw_hz = _profile_to_bands(args.profile)
    meta_extra = {
        "orig_data": {
            "dataset": "ShinRatBrain_Fig3",
            "iq_file": str(iq_path),
            "sizeinfo": asdict(info),
            "frames_spec": args.frames,
        },
        "shin_profile": {
            "name": str(args.profile),
            "flow_low_hz": flow_low_hz,
            "flow_high_hz": flow_high_hz,
            "alias_center_hz": alias_center_hz,
            "alias_half_width_hz": alias_hw_hz,
        },
        "motion_injection": {
            "kind": str(args.motion_kind),
            "seed": int(args.seed),
            "amp_px_list": amp_list,
            "reg_enable": bool(args.reg_enable),
        },
    }

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
    if args.out_png is not None:
        args.out_png.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    # Reference run (amp=0) defines masks and thresholds.
    ref_paths: dict[str, str] | None = None
    ref_pd_base = ref_pd_pre = ref_pd_post = None
    ref_mask_bg = ref_mask_flow = None
    vessel_ref = None
    thr_vessel_base = None
    thr_vessel_stap = None
    thr_bg_tail_base = None
    thr_bg_tail_stap = None
    ref_base_score = ref_stap_score = None
    thr_base_fpr_ref = thr_stap_fpr_ref = None
    pos_base_ref = pos_stap_ref = pos_shared_ref = None

    for amp_px in amp_list:
        kind = str(args.motion_kind).strip().lower()
        dy = dx = np.zeros(Icube0.shape[0], dtype=np.float32)
        motion_tele: dict[str, float] = {}
        if kind in {"sine", "sin", "drift_sine", "step", "burst", "rw", "randomwalk"}:
            dy, dx = _motion_shifts(
                kind,
                T=Icube0.shape[0],
                amp_px=float(amp_px),
                seed=int(args.seed),
                step_frame=args.step_frame,
            )
            Icube = Icube0 if float(amp_px) == 0.0 else _apply_translation_per_frame(Icube0, dy=dy, dx=dx)
            motion_tele = {"shift_rms_px": float(np.sqrt(np.mean(dy * dy + dx * dx)))}
        elif kind in {"brainlike", "brain_like", "brain"}:
            dy_f, dx_f, tele = _brainlike_displacement_fields(
                T=Icube0.shape[0],
                H=Icube0.shape[1],
                W=Icube0.shape[2],
                amp_px=float(amp_px),
                seed=int(args.seed),
                rigid_kind=str(args.brainlike_rigid_kind),
                rigid_frac=float(args.brainlike_rigid_frac),
                elastic_frac=float(args.brainlike_elastic_frac),
                elastic_sigma_px=float(args.brainlike_elastic_sigma_px),
                elastic_depth_decay_frac=float(args.brainlike_elastic_depth_decay_frac),
                elastic_rw_step_sigma=float(args.brainlike_elastic_rw_step_sigma),
                micro_jitter_frac=float(args.brainlike_micro_jitter_frac),
            )
            Icube = Icube0 if float(amp_px) == 0.0 else _apply_warp_per_frame(Icube0, dy_field=dy_f, dx_field=dx_f)
            motion_tele = dict(tele)
        elif kind in {"clutter_tone", "clutter"}:
            Icube = _apply_clutter_tone(
                Icube0,
                prf_hz=float(args.prf_hz),
                f_hz=float(args.motion_f_hz),
                amp_rel=float(amp_px),
                sigma_px=float(args.clutter_sigma_px),
                seed=int(args.seed),
            )
        elif kind in {"phase_sine", "phase"}:
            Icube = _apply_global_phase_sine(
                Icube0,
                prf_hz=float(args.prf_hz),
                f_hz=float(args.motion_f_hz),
                amp_rad=float(amp_px),
            )
        else:
            raise ValueError(f"Unknown motion kind {kind!r}.")

        amp_tag = _format_amp_tag(float(amp_px))
        dataset_name = (
            f"shin_motion_{iq_path.stem}_{frame_tag}"
            f"_p{str(args.profile).strip()}_Lt{int(args.Lt)}"
            f"_{baseline_type}"
        )
        if baseline_type == "mc_svd":
            if svd_rank is None:
                dataset_name += f"_e{float(args.svd_energy_frac):.3f}"
            else:
                dataset_name += f"_r{int(svd_rank)}"
        else:
            dataset_name += f"_k{int(svd_keep_min)}_{int(svd_keep_max)}"
        dataset_name += f"_{str(args.motion_kind)}_a{amp_tag}"

        paths = write_acceptance_bundle_from_icube(
            out_root=args.out_root,
            dataset_name=dataset_name,
            Icube=Icube,
            prf_hz=float(args.prf_hz),
            tile_hw=(int(args.tile_h), int(args.tile_w)),
            tile_stride=int(args.tile_stride),
            Lt=int(args.Lt),
            baseline_type=baseline_type,
            svd_rank=svd_rank,
            svd_energy_frac=float(args.svd_energy_frac),
            svd_keep_min=svd_keep_min,
            svd_keep_max=svd_keep_max,
            flow_mask_mode=str(args.flow_mask_mode),
            flow_mask_pd_quantile=float(args.flow_mask_pd_quantile),
            flow_mask_depth_min_frac=float(args.flow_mask_depth_min_frac),
            flow_mask_depth_max_frac=float(args.flow_mask_depth_max_frac),
            flow_mask_erode_iters=int(args.flow_mask_erode_iters),
            flow_mask_dilate_iters=int(args.flow_mask_dilate_iters),
            flow_mask_min_pixels=int(args.flow_mask_min_pixels),
            flow_mask_union_default=bool(args.flow_mask_union_default),
            band_ratio_flow_low_hz=float(flow_low_hz),
            band_ratio_flow_high_hz=float(flow_high_hz),
            band_ratio_alias_center_hz=float(alias_center_hz),
            band_ratio_alias_width_hz=float(alias_hw_hz),
            run_stap=True,
            reg_enable=bool(args.reg_enable),
            reg_subpixel=int(args.reg_subpixel),
            score_mode="pd",
            cov_estimator="tyler_pca",
            score_ka_v2_enable=bool(args.score_ka_v2),
            score_ka_v2_mode=str(args.score_ka_v2_mode),
            meta_extra=meta_extra,
        )

        bundle_dir = Path(paths["meta"]).parent
        meta = json.loads(Path(paths["meta"]).read_text())
        pd_base = np.load(bundle_dir / "pd_base.npy")
        pd_post = np.load(bundle_dir / "pd_stap.npy")
        pd_pre_path = bundle_dir / "pd_stap_pre_ka.npy"
        if pd_pre_path.is_file():
            pd_pre = np.load(pd_pre_path)
        else:
            # Fallback: undo scale if available.
            scale_path = bundle_dir / "ka_scale_map.npy"
            if scale_path.is_file():
                scale = np.load(scale_path).astype(np.float32)
                # Bundle convention: pd_post = pd_pre / ka_scale (scale>=1).
                pd_pre = pd_post.astype(np.float32) * np.maximum(scale, 1e-12)
            else:
                pd_pre = pd_post

        mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
        mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)

        # Detector score convention: in our PD-mode pipeline, higher scores correspond to
        # more flow-like evidence, and we take s = PD.
        base_score_map = (pd_base).astype(np.float32, copy=False)
        stap_score_map = (pd_pre).astype(np.float32, copy=False)

        # Reference thresholds.
        if float(amp_px) == 0.0:
            ref_paths = paths
            ref_pd_base = np.asarray(pd_base, dtype=np.float32)
            ref_pd_pre = np.asarray(pd_pre, dtype=np.float32)
            ref_pd_post = np.asarray(pd_post, dtype=np.float32)
            ref_mask_flow = mask_flow
            ref_mask_bg = mask_bg
            ref_base_score = np.asarray(base_score_map, dtype=np.float64)
            ref_stap_score = np.asarray(stap_score_map, dtype=np.float64)

            # Shared structure proxy from *both* no-motion maps: intersection of high-score
            # regions from baseline and STAP pre-KA scores, using frozen quantiles.
            q_vessel = float(args.vessel_quantile)
            thr_vessel_base = _safe_quantile(ref_base_score, q_vessel)
            thr_vessel_stap = _safe_quantile(ref_stap_score, q_vessel)
            if thr_vessel_base is None or thr_vessel_stap is None:
                raise RuntimeError("Failed to compute vessel quantiles on reference score maps.")
            vessel_base_ref = ref_base_score >= float(thr_vessel_base)
            vessel_stap_ref = ref_stap_score >= float(thr_vessel_stap)
            vessel_ref = vessel_base_ref & vessel_stap_ref

            # Reference BG tail thresholds per method (score S=PD) on the BG proxy.
            thr_bg_tail_base = _safe_quantile(ref_base_score[ref_mask_bg], float(args.bg_tail_quantile))
            thr_bg_tail_stap = _safe_quantile(ref_stap_score[ref_mask_bg], float(args.bg_tail_quantile))

            # Reference thresholds for proxy TPR@FPR curves on detector score maps.
            thr_base_fpr_ref = _threshold_at_fpr(ref_base_score, ref_mask_bg, float(args.fpr_target))
            thr_stap_fpr_ref = _threshold_at_fpr(ref_stap_score, ref_mask_bg, float(args.fpr_target))
            q_pos = float(np.clip(float(args.pos_quantile), 0.0, 1.0))
            thr_pos_base = _safe_quantile(ref_base_score, q_pos)
            thr_pos_stap = _safe_quantile(ref_stap_score, q_pos)
            pos_base_ref = (ref_base_score >= float(thr_pos_base)) if thr_pos_base is not None else None
            pos_stap_ref = (ref_stap_score >= float(thr_pos_stap)) if thr_pos_stap is not None else None
            pos_shared_ref = (pos_base_ref & pos_stap_ref) if pos_base_ref is not None and pos_stap_ref is not None else None

        if ref_pd_base is None or ref_pd_pre is None or ref_pd_post is None or ref_base_score is None or ref_stap_score is None:
            raise RuntimeError("Reference (amp=0) run failed to initialize.")

        assert vessel_ref is not None
        assert thr_vessel_base is not None and thr_vessel_stap is not None
        assert thr_bg_tail_base is not None and thr_bg_tail_stap is not None
        assert thr_base_fpr_ref is not None and thr_stap_fpr_ref is not None
        assert pos_base_ref is not None and pos_stap_ref is not None and pos_shared_ref is not None
        assert ref_mask_bg is not None and ref_mask_flow is not None

        # Optional crop for similarity metrics (avoid boundary artifacts).
        if args.crop_margin is None:
            margin = int(math.ceil(max(0.0, float(amp_px)))) + 2
        else:
            margin = int(args.crop_margin)
        if margin * 2 >= ref_pd_base.shape[0] or margin * 2 >= ref_pd_base.shape[1]:
            margin = 0
        sl = (slice(margin, ref_pd_base.shape[0] - margin), slice(margin, ref_pd_base.shape[1] - margin))

        # Compare each stage against its own reference stage.
        base_map = np.asarray(pd_base, dtype=np.float32)
        pre_map = np.asarray(pd_pre, dtype=np.float32)
        post_map = np.asarray(pd_post, dtype=np.float32)
        ref_base_map = np.asarray(ref_pd_base, dtype=np.float32)
        ref_pre_map = np.asarray(ref_pd_pre, dtype=np.float32)
        ref_post_map = np.asarray(ref_pd_post, dtype=np.float32)

        if bool(args.align_maps):
            dyb, dxb, psrb = _phasecorr_shift(ref_base_map, base_map, upsample=4)
            dys, dxs, psrs = _phasecorr_shift(ref_pre_map, pre_map, upsample=4)
            dyk, dxk, psrk = _phasecorr_shift(ref_post_map, post_map, upsample=4)
            base_map = ndi.shift(base_map, shift=(dyb, dxb), order=1, mode="nearest", prefilter=False)
            pre_map = ndi.shift(pre_map, shift=(dys, dxs), order=1, mode="nearest", prefilter=False)
            post_map = ndi.shift(post_map, shift=(dyk, dxk), order=1, mode="nearest", prefilter=False)
        else:
            dyb = dxb = psrb = 0.0
            dys = dxs = psrs = 0.0
            dyk = dxk = psrk = 0.0

        corr_base = _pearson_corr(base_map[sl], ref_base_map[sl])
        corr_stap = _pearson_corr(pre_map[sl], ref_pre_map[sl])
        corr_ka = _pearson_corr(post_map[sl], ref_post_map[sl])
        nrmse_base = _nrmse(base_map[sl], ref_base_map[sl])
        nrmse_stap = _nrmse(pre_map[sl], ref_pre_map[sl])
        nrmse_ka = _nrmse(post_map[sl], ref_post_map[sl])

        # Non-collapse diagnostics: map variance / entropy should not collapse toward 0.
        std_pd_base = float(np.std(base_map[sl]))
        std_pd_stap = float(np.std(pre_map[sl]))
        std_pd_ka = float(np.std(post_map[sl]))
        ent_pd_base = _shannon_entropy(base_map[sl])
        ent_pd_stap = _shannon_entropy(pre_map[sl])
        ent_pd_ka = _shannon_entropy(post_map[sl])

        bg_sl_mask = ref_mask_bg[sl]
        var_ref_base_bg = float(np.var(ref_base_map[sl][bg_sl_mask])) + 1e-12
        var_ref_pre_bg = float(np.var(ref_pre_map[sl][bg_sl_mask])) + 1e-12
        var_ref_post_bg = float(np.var(ref_post_map[sl][bg_sl_mask])) + 1e-12
        bg_var_ratio_base = float(np.var(base_map[sl][bg_sl_mask]) / var_ref_base_bg)
        bg_var_ratio_stap = float(np.var(pre_map[sl][bg_sl_mask]) / var_ref_pre_bg)
        bg_var_ratio_ka = float(np.var(post_map[sl][bg_sl_mask]) / var_ref_post_bg)

        # Shared-structure overlap using fixed no-motion thresholds on S=PD.
        s_base = -base_map
        s_pre = -pre_map
        s_post = -post_map
        vessel_base = base_score_map >= float(thr_vessel_base)
        vessel_pre = stap_score_map >= float(thr_vessel_stap)
        vessel_post = s_post >= float(thr_vessel_stap)
        j_base = _jaccard(vessel_base, vessel_ref)
        j_stap = _jaccard(vessel_pre, vessel_ref)
        j_ka = _jaccard(vessel_post, vessel_ref)

        # Background tail clusters/area using per-method no-motion BG thresholds on S=PD.
        hit_base = ref_mask_bg & (base_score_map >= float(thr_bg_tail_base))
        hit_pre = ref_mask_bg & (stap_score_map >= float(thr_bg_tail_stap))
        hit_post = ref_mask_bg & (s_post >= float(thr_bg_tail_stap))
        area_base = int(np.sum(hit_base))
        area_pre = int(np.sum(hit_pre))
        area_post = int(np.sum(hit_post))
        clust_base = _connected_components(hit_base, connectivity=int(args.connectivity))
        clust_pre = _connected_components(hit_pre, connectivity=int(args.connectivity))
        clust_post = _connected_components(hit_post, connectivity=int(args.connectivity))

        # Flow mask stability (optional proxy).
        flow_jacc = _jaccard(mask_flow, ref_mask_flow)

        # Proxy detection robustness: TPR on a fixed reference positive set at a fixed FPR on bg proxy.
        # We report both:
        #   (i) fixed-threshold (calibrated on the no-motion reference) to measure FPR drift under motion, and
        #  (ii) per-run threshold (recalibrated on bg proxy each run) to measure TPR at matched FPR.
        pos_flow = ref_mask_flow
        pos_vessel = vessel_ref
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

        tpr_base_vessel_fixed, fpr_base_vessel_fixed = _tpr_fpr_at_threshold(
            base_score_map, pos_mask=pos_vessel, neg_mask=neg, thr=float(thr_base_fpr_ref)
        )
        tpr_stap_vessel_fixed, fpr_stap_vessel_fixed = _tpr_fpr_at_threshold(
            stap_score_map, pos_mask=pos_vessel, neg_mask=neg, thr=float(thr_stap_fpr_ref)
        )
        tpr_base_vessel_run, fpr_base_vessel_run = _tpr_fpr_at_threshold(
            base_score_map, pos_mask=pos_vessel, neg_mask=neg, thr=float(thr_base_run)
        )
        tpr_stap_vessel_run, fpr_stap_vessel_run = _tpr_fpr_at_threshold(
            stap_score_map, pos_mask=pos_vessel, neg_mask=neg, thr=float(thr_stap_run)
        )

        # Self-consistency: each method evaluated on its own no-motion "positive proxy" set.
        tpr_base_self_fixed, fpr_base_self_fixed = _tpr_fpr_at_threshold(
            base_score_map, pos_mask=pos_base_ref, neg_mask=neg, thr=float(thr_base_fpr_ref)
        )
        tpr_stap_self_fixed, fpr_stap_self_fixed = _tpr_fpr_at_threshold(
            stap_score_map, pos_mask=pos_stap_ref, neg_mask=neg, thr=float(thr_stap_fpr_ref)
        )
        tpr_base_self_run, fpr_base_self_run = _tpr_fpr_at_threshold(
            base_score_map, pos_mask=pos_base_ref, neg_mask=neg, thr=float(thr_base_run)
        )
        tpr_stap_self_run, fpr_stap_self_run = _tpr_fpr_at_threshold(
            stap_score_map, pos_mask=pos_stap_ref, neg_mask=neg, thr=float(thr_stap_run)
        )

        # Cross-method proxy: shared no-motion high-score set (baseline ∩ STAP).
        tpr_base_shared_fixed, _ = _tpr_fpr_at_threshold(
            base_score_map, pos_mask=pos_shared_ref, neg_mask=neg, thr=float(thr_base_fpr_ref)
        )
        tpr_stap_shared_fixed, _ = _tpr_fpr_at_threshold(
            stap_score_map, pos_mask=pos_shared_ref, neg_mask=neg, thr=float(thr_stap_fpr_ref)
        )
        tpr_base_shared_run, _ = _tpr_fpr_at_threshold(
            base_score_map, pos_mask=pos_shared_ref, neg_mask=neg, thr=float(thr_base_run)
        )
        tpr_stap_shared_run, _ = _tpr_fpr_at_threshold(
            stap_score_map, pos_mask=pos_shared_ref, neg_mask=neg, thr=float(thr_stap_run)
        )

        ka_v2 = meta.get("ka_contract_v2") or {}
        ka_metrics = (ka_v2.get("metrics") or {}) if isinstance(ka_v2, dict) else {}
        tele = meta.get("stap_fallback_telemetry") or {}

        rows.append(
            {
                "bundle": str(bundle_dir),
                "bundle_name": bundle_dir.name,
                "iq_file": args.iq_file,
                "frames": args.frames,
                "profile": str(args.profile),
                "Lt": int(args.Lt),
                "baseline_type": baseline_type,
                "svd_rank": svd_rank,
                "svd_energy_frac": float(args.svd_energy_frac) if svd_rank is None and baseline_type == "mc_svd" else None,
                "svd_keep_min": svd_keep_min,
                "svd_keep_max": svd_keep_max,
                "motion_kind": str(args.motion_kind),
                "amp_px": float(amp_px),
                "shift_rms_px": float(motion_tele.get("shift_rms_px", float(np.sqrt(np.mean(dy * dy + dx * dx))))),
                "disp_rms_px": float(motion_tele.get("disp_rms_px", float(np.sqrt(np.mean(dy * dy + dx * dx))))),
                "disp_p90_px": float(motion_tele.get("disp_p90_px", float(np.sqrt(np.mean(dy * dy + dx * dx))))),
                "corr_pd_base": corr_base,
                "corr_pd_stap_pre": corr_stap,
                "corr_pd_ka": corr_ka,
                "nrmse_pd_base": nrmse_base,
                "nrmse_pd_stap_pre": nrmse_stap,
                "nrmse_pd_ka": nrmse_ka,
                "std_pd_base": std_pd_base,
                "std_pd_stap_pre": std_pd_stap,
                "std_pd_ka": std_pd_ka,
                "entropy_pd_base": ent_pd_base,
                "entropy_pd_stap_pre": ent_pd_stap,
                "entropy_pd_ka": ent_pd_ka,
                "bg_var_ratio_pd_base": bg_var_ratio_base,
                "bg_var_ratio_pd_stap_pre": bg_var_ratio_stap,
                "bg_var_ratio_pd_ka": bg_var_ratio_ka,
                "align_maps": bool(args.align_maps),
                "align_shift_base_dy": float(dyb),
                "align_shift_base_dx": float(dxb),
                "align_shift_base_psr": float(psrb),
                "align_shift_stap_dy": float(dys),
                "align_shift_stap_dx": float(dxs),
                "align_shift_stap_psr": float(psrs),
                "align_shift_ka_dy": float(dyk),
                "align_shift_ka_dx": float(dxk),
                "align_shift_ka_psr": float(psrk),
                "jacc_vessel_base": j_base,
                "jacc_vessel_stap_pre": j_stap,
                "jacc_vessel_ka": j_ka,
                "vessel_ref_shared_frac": float(np.mean(vessel_ref)),
                "pos_shared_ref_frac": float(np.mean(pos_shared_ref)),
                "bg_tail_area_base": area_base,
                "bg_tail_area_stap_pre": area_pre,
                "bg_tail_area_ka": area_post,
                "bg_tail_clusters_base": clust_base,
                "bg_tail_clusters_stap_pre": clust_pre,
                "bg_tail_clusters_ka": clust_post,
                "flow_mask_jaccard": flow_jacc,
                "fpr_target": float(args.fpr_target),
                "tpr_base_flow_fixedthr": tpr_base_flow_fixed,
                "fpr_base_flow_fixedthr": fpr_base_flow_fixed,
                "tpr_stap_flow_fixedthr": tpr_stap_flow_fixed,
                "fpr_stap_flow_fixedthr": fpr_stap_flow_fixed,
                "tpr_base_flow_at_fpr": tpr_base_flow_run,
                "tpr_stap_flow_at_fpr": tpr_stap_flow_run,
                "tpr_base_vessel_fixedthr": tpr_base_vessel_fixed,
                "fpr_base_vessel_fixedthr": fpr_base_vessel_fixed,
                "tpr_stap_vessel_fixedthr": tpr_stap_vessel_fixed,
                "fpr_stap_vessel_fixedthr": fpr_stap_vessel_fixed,
                "tpr_base_vessel_at_fpr": tpr_base_vessel_run,
                "tpr_stap_vessel_at_fpr": tpr_stap_vessel_run,
                "tpr_base_self_fixedthr": tpr_base_self_fixed,
                "fpr_base_self_fixedthr": fpr_base_self_fixed,
                "tpr_stap_self_fixedthr": tpr_stap_self_fixed,
                "fpr_stap_self_fixedthr": fpr_stap_self_fixed,
                "tpr_base_self_at_fpr": tpr_base_self_run,
                "tpr_stap_self_at_fpr": tpr_stap_self_run,
                "tpr_base_shared_fixedthr": tpr_base_shared_fixed,
                "tpr_stap_shared_fixedthr": tpr_stap_shared_fixed,
                "tpr_base_shared_at_fpr": tpr_base_shared_run,
                "tpr_stap_shared_at_fpr": tpr_stap_shared_run,
                "ka_state": ka_v2.get("state") if isinstance(ka_v2, dict) else None,
                "ka_reason": ka_v2.get("reason") if isinstance(ka_v2, dict) else None,
                "ka_risk_mode": ka_metrics.get("risk_mode"),
                "ka_p_shrink": ka_metrics.get("p_shrink"),
                "ka_pf_peak_flow": ka_metrics.get("pf_peak_flow"),
                "ka_uplift_veto_pf_peak_reason": ka_metrics.get("uplift_veto_pf_peak_reason"),
                "ka_guard_q90": ka_metrics.get("guard_q90"),
                "score_ka_v2_applied": tele.get("score_ka_v2_applied") if isinstance(tele, dict) else None,
                "score_ka_v2_scaled_px_frac": tele.get("score_ka_v2_scaled_pixel_fraction") if isinstance(tele, dict) else None,
                "score_ka_v2_scale_p90": tele.get("score_ka_v2_scale_p90") if isinstance(tele, dict) else None,
                "score_ka_v2_scale_max": tele.get("score_ka_v2_scale_max") if isinstance(tele, dict) else None,
                # Record motion telemetry details for brainlike mode (NaN otherwise).
                "brainlike_rigid_kind": motion_tele.get("rigid_kind"),
                "brainlike_rigid_frac": motion_tele.get("rigid_frac"),
                "brainlike_elastic_frac": motion_tele.get("elastic_frac"),
                "brainlike_elastic_sigma_px": motion_tele.get("elastic_sigma_px"),
                "brainlike_elastic_depth_decay_frac": motion_tele.get("elastic_depth_decay_frac"),
                "brainlike_elastic_rw_step_sigma": motion_tele.get("elastic_rw_step_sigma"),
                "brainlike_micro_jitter_frac": motion_tele.get("micro_jitter_frac"),
            }
        )

        print(
            "[shin-motion]"
            f" amp={amp_px:.2f}px"
            f" corr(base/stap/ka)={corr_base:.3f}/{corr_stap:.3f}/{corr_ka:.3f}"
            f" tpr_flow@fpr(base/stap)={tpr_base_flow_run:.3f}/{tpr_stap_flow_run:.3f}"
            f" tpr_shared@fpr(base/stap)={tpr_base_shared_run:.3f}/{tpr_stap_shared_run:.3f}"
            f" clusters(base/stap/ka)={clust_base}/{clust_pre}/{clust_post}"
            f" ka={rows[-1]['ka_state']}({rows[-1]['ka_reason']})"
        )

    # Write CSV
    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[shin-motion] wrote {len(rows)} rows to {args.out_csv}")

    if args.out_png is not None:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            amps = np.array([r["amp_px"] for r in rows], dtype=float)
            corr_base = np.array([r["corr_pd_base"] for r in rows], dtype=float)
            corr_stap = np.array([r["corr_pd_stap_pre"] for r in rows], dtype=float)
            cl_base = np.array([r["bg_tail_clusters_base"] for r in rows], dtype=float)
            cl_stap = np.array([r["bg_tail_clusters_stap_pre"] for r in rows], dtype=float)
            tpr_base = np.array([r["tpr_base_flow_at_fpr"] for r in rows], dtype=float)
            tpr_stap = np.array([r["tpr_stap_flow_at_fpr"] for r in rows], dtype=float)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
            ax = axes[0]
            ax.plot(amps, corr_base, "o-", label="baseline PD")
            ax.plot(amps, corr_stap, "o-", label="STAP PD (pre-KA)")
            ax.set_xlabel("motion amplitude (px)")
            ax.set_ylabel("corr vs no-motion")
            ax.set_title("Map Stability")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[1]
            ax.plot(amps, tpr_base, "o-", label="baseline score")
            ax.plot(amps, tpr_stap, "o-", label="STAP score")
            ax.set_xlabel("motion amplitude (px)")
            ax.set_ylabel("TPR on flow-proxy")
            ax.set_title(f"Proxy TPR @ FPR={float(args.fpr_target):.0e} (bg-proxy)")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[2]
            ax.plot(amps, cl_base, "o-", label="baseline PD")
            ax.plot(amps, cl_stap, "o-", label="STAP PD (pre-KA)")
            ax.set_xlabel("motion amplitude (px)")
            ax.set_ylabel("BG tail clusters")
            ax.set_title(f"BG Tail @ q={float(args.bg_tail_quantile):.3f}")
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.suptitle(
                f"Shin motion sweep: {args.iq_file}, profile {args.profile}, Lt={args.Lt}, {args.motion_kind}",
                fontsize=11,
            )
            fig.savefig(args.out_png, dpi=200)
            plt.close(fig)
            print(f"[shin-motion] wrote {args.out_png}")
        except Exception as exc:
            print(f"[shin-motion] failed to write plot: {exc}")


if __name__ == "__main__":
    main()
