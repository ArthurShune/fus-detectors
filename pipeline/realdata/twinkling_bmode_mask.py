from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.ndimage as ndi
import warnings

from pipeline.realdata.twinkling_artifact import (
    RawBCFPar,
    decode_rawbcf_bmode_iq,
    read_rawbcf_frame,
)


@dataclass(frozen=True)
class BModeCfmRoiMapping:
    """
    Map B-mode beamformer output into the CFM ROI grid.

    Decoded B-mode IQ is returned by `decode_rawbcf_bmode_iq` with shape (B, G):
      - B = num_b_beams (lateral beams)
      - G = b_beam_samples (axial samples)

    CFM grid has shape (P, K):
      - P = cfm_beam_samples (axial samples)
      - K = num_cfm_beams (lateral beams)

    The mapping is:
      sample_idx[p] = s0 + p
      beam_idx[k]   = b0 + Q*k
    where invalid (out-of-range) beam indices are marked in `valid_beams`.
    """

    b0: int
    s0: int
    Q: int
    sample_idx: np.ndarray  # (P,), int
    beam_idx: np.ndarray  # (K,), int
    valid_beams: np.ndarray  # (K,), bool


def _candidate_ints(value: object | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, tuple) and value:
        out: list[int] = []
        for v in value:
            try:
                out.append(int(v))
            except Exception:
                pass
        return sorted(set(out))
    try:
        return [int(value)]  # type: ignore[arg-type]
    except Exception:
        return []


def iter_mapping_candidates(par: RawBCFPar) -> list[tuple[int, int, int]]:
    """
    Return candidate (b0, s0, Q) tuples.

    We try both 0-based and 1-based interpretations by including raw and raw-1.
    For fields that appear as "a//b" in .par, we try each value.
    """

    Q = int(par.cfm_density) if par.cfm_density is not None else 1
    Q = max(1, Q)

    b0_raw = _candidate_ints(par.first_scan_cfm_beam)
    if not b0_raw:
        b0_raw = [0]
    s0_raw = _candidate_ints(par.num_first_cfm_sample)
    if not s0_raw:
        s0_raw = [0]

    b0_cands: list[int] = []
    for b in b0_raw:
        b0_cands.extend([b, b - 1])
    s0_cands: list[int] = []
    for s in s0_raw:
        s0_cands.extend([s, s - 1])

    out: list[tuple[int, int, int]] = []
    for b0 in sorted(set(b0_cands)):
        for s0 in sorted(set(s0_cands)):
            out.append((int(b0), int(s0), int(Q)))
    return out


def build_bmode_cfm_roi_mapping(
    par: RawBCFPar,
    *,
    b0: int,
    s0: int,
    Q: int,
) -> BModeCfmRoiMapping:
    Q = max(1, int(Q))
    P = int(par.cfm_beam_samples)
    K = int(par.num_cfm_beams)
    B = int(par.num_b_beams)
    G = int(par.b_beam_samples)

    sample_idx = s0 + np.arange(P, dtype=np.int32)
    if sample_idx.min() < 0 or sample_idx.max() >= G:
        raise ValueError(
            f"Invalid s0={s0}: sample_idx range {int(sample_idx.min())}..{int(sample_idx.max())} "
            f"outside [0,{G-1}]"
        )
    beam_idx = b0 + Q * np.arange(K, dtype=np.int32)
    valid_beams = (beam_idx >= 0) & (beam_idx < B)
    return BModeCfmRoiMapping(
        b0=int(b0),
        s0=int(s0),
        Q=int(Q),
        sample_idx=sample_idx.astype(np.int32, copy=False),
        beam_idx=beam_idx.astype(np.int32, copy=False),
        valid_beams=valid_beams.astype(bool, copy=False),
    )


def extract_bmode_roi(
    bmode_iq_beam_sample: np.ndarray,
    par: RawBCFPar,
    mapping: BModeCfmRoiMapping,
    *,
    fill_value: float = float("nan"),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a (P,K) complex ROI in CFM coordinates from B-mode IQ.

    Returns (roi_complex, valid_mask) where valid_mask indicates which (p,k)
    entries correspond to real B-mode beams (invalid columns occur when b0<0).
    """

    bmode = np.asarray(bmode_iq_beam_sample)
    if bmode.ndim != 2:
        raise ValueError(f"bmode_iq must have shape (B,G), got {bmode.shape}")
    B, G = bmode.shape
    if B != int(par.num_b_beams) or G != int(par.b_beam_samples):
        raise ValueError(
            f"bmode_iq shape mismatch: got (B,G)=({B},{G}), "
            f"expected ({par.num_b_beams},{par.b_beam_samples})"
        )
    P = int(par.cfm_beam_samples)
    K = int(par.num_cfm_beams)
    roi = np.full((P, K), fill_value, dtype=np.complex64)
    valid_cols = np.asarray(mapping.valid_beams, dtype=bool)
    if np.any(valid_cols):
        cols = mapping.beam_idx[valid_cols].astype(np.int64, copy=False)
        samp = mapping.sample_idx.astype(np.int64, copy=False)
        # bmode is (B,G); we want (P, valid_K): depth x beam.
        roi[:, valid_cols] = bmode[cols, :][:, samp].T.astype(np.complex64, copy=False)
    valid_mask = np.zeros((P, K), dtype=bool)
    valid_mask[:, valid_cols] = True
    return roi, valid_mask


def build_bmode_roi_reference_db(
    dat_path: Path,
    par: RawBCFPar,
    mapping: BModeCfmRoiMapping,
    *,
    frame_indices: Iterable[int],
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a median reference B-mode log-envelope image in CFM coordinates.

    Returns (I_ref_db, valid_mask), where I_ref_db has shape (P,K).
    """

    imgs: list[np.ndarray] = []
    valid_mask: np.ndarray | None = None
    for idx in frame_indices:
        frame = read_rawbcf_frame(dat_path, par, int(idx))
        bmode = decode_rawbcf_bmode_iq(frame, par)  # (B,G)
        roi, valid = extract_bmode_roi(bmode, par, mapping)
        env = np.abs(roi).astype(np.float32, copy=False)
        I_db = (20.0 * np.log10(env + float(eps))).astype(np.float32, copy=False)
        # Ensure invalid pixels are NaN so nanmedian ignores them.
        I_db[~valid] = np.nan
        imgs.append(I_db)
        if valid_mask is None:
            valid_mask = valid
    if not imgs:
        raise ValueError("No frames provided for B-mode reference.")
    stack = np.stack(imgs, axis=0)  # (F,P,K)
    # Invalid columns (e.g. when FirstScanCFMBeam < 0 or NumOfCFMBeams > NumOfBBeams)
    # are represented as all-NaN slices; suppress the expected warning.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        I_ref = np.nanmedian(stack, axis=0).astype(np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(I_ref)
    # Fill any NaNs (should only occur in invalid columns) with the minimum
    # finite value so downstream normalization is well-defined.
    finite = np.isfinite(I_ref)
    if np.any(finite):
        I_ref[~finite] = float(np.min(I_ref[finite]))
    else:
        I_ref[:] = 0.0
    return I_ref, valid_mask.astype(bool, copy=False)


@dataclass(frozen=True)
class TubeMaskResult:
    mask_flow: np.ndarray  # (P,K) bool
    mask_bg: np.ndarray  # (P,K) bool
    mask_bright: np.ndarray  # (P,K) bool (excluded)
    mapping: BModeCfmRoiMapping
    qc: dict[str, float | int | str | None]
    params: dict[str, float | int]
    I_ref_norm: np.ndarray  # (P,K) float32 in [0,1]


def segment_tube_masks_from_bmode_ref(
    I_ref_db: np.ndarray,
    valid_mask: np.ndarray,
    mapping: BModeCfmRoiMapping,
    *,
    sigma: float = 1.5,
    tau_percentile: float = 20.0,
    opening_iters: int = 2,
    closing_iters: int = 3,
    flow_erode_iters: int = 3,
    guard_dilate_iters: int = 6,
    wall_percentile: float = 95.0,
    wall_dilate_iters: int = 1,
    area_frac_min: float = 0.005,
    area_frac_max: float = 0.40,
    alpha_center: float = 0.0,
) -> TubeMaskResult:
    I_ref_db = np.asarray(I_ref_db, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if I_ref_db.shape != valid_mask.shape:
        raise ValueError("I_ref_db and valid_mask must have the same shape.")

    # Robust normalization to [0,1].
    finite = valid_mask & np.isfinite(I_ref_db)
    if not np.any(finite):
        raise ValueError("No finite pixels in valid_mask for normalization.")
    p1 = float(np.quantile(I_ref_db[finite], 0.01))
    p99 = float(np.quantile(I_ref_db[finite], 0.99))
    denom = max(1e-6, p99 - p1)
    I_norm = np.clip((I_ref_db - p1) / denom, 0.0, 1.0).astype(np.float32, copy=False)
    I_norm[~finite] = 0.0

    I_smooth = ndi.gaussian_filter(I_norm, sigma=float(sigma)).astype(np.float32, copy=False)

    # Gradient magnitude (wall evidence).
    gx = ndi.sobel(I_smooth, axis=1)
    gy = ndi.sobel(I_smooth, axis=0)
    g = np.sqrt(gx * gx + gy * gy).astype(np.float32, copy=False)
    g_p1 = float(np.quantile(g[finite], 0.01))
    g_p99 = float(np.quantile(g[finite], 0.99))
    g = np.clip((g - g_p1) / max(1e-6, g_p99 - g_p1), 0.0, 1.0).astype(np.float32, copy=False)

    # Dark candidate threshold.
    tau = float(np.quantile(I_smooth[finite], float(tau_percentile) / 100.0))
    cand = (I_smooth < tau) & finite

    struct = ndi.generate_binary_structure(2, 1)
    cand = ndi.binary_opening(cand, structure=struct, iterations=int(opening_iters))
    cand = ndi.binary_closing(cand, structure=struct, iterations=int(closing_iters))
    cand = ndi.binary_fill_holes(cand)

    labels, nlab = ndi.label(cand)
    P, K = I_norm.shape
    total_valid = int(np.count_nonzero(finite))
    if total_valid <= 0:
        raise ValueError("No valid pixels for mask generation.")

    best_score = -1.0
    best_region: np.ndarray | None = None
    best_edge_support = 0.0
    best_area = 0
    best_centroid_dist = 0.0

    # ROI center for weak centrality prior.
    cy = 0.5 * (P - 1)
    cx = 0.5 * (K - 1)
    for lab in range(1, int(nlab) + 1):
        region = labels == lab
        if not np.any(region):
            continue
        # Reject if touches boundary (common shadow/gel artifacts).
        if bool(region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any()):
            continue
        area = int(np.count_nonzero(region))
        frac = area / float(total_valid)
        if frac < float(area_frac_min) or frac > float(area_frac_max):
            continue
        boundary = region & (~ndi.binary_erosion(region, structure=struct, iterations=1))
        if not np.any(boundary):
            continue
        edge_support = float(np.mean(g[boundary]))
        cy_r, cx_r = ndi.center_of_mass(region)
        d2 = float((cy_r - cy) ** 2 + (cx_r - cx) ** 2)
        # S_i = |R| * E_i * exp(-alpha * D^2 / (P*K))
        s = float(area) * float(edge_support) * float(
            np.exp(-float(alpha_center) * d2 / max(1.0, float(P * K)))
        )
        if s > best_score:
            best_score = s
            best_region = region
            best_edge_support = edge_support
            best_area = area
            best_centroid_dist = float(np.sqrt(d2))

    if best_region is None:
        # No component passed the filters; fall back to the largest component.
        if int(nlab) <= 0:
            raise ValueError("No candidate regions found for tube segmentation.")
        areas = np.bincount(labels.ravel())
        areas[0] = 0
        lab = int(np.argmax(areas))
        best_region = labels == lab
        best_area = int(areas[lab])
        boundary = best_region & (~ndi.binary_erosion(best_region, structure=struct, iterations=1))
        best_edge_support = float(np.mean(g[boundary])) if np.any(boundary) else 0.0
        best_score = float(best_area) * best_edge_support

    # Safe flow mask: erode away bright walls.
    mask_flow = ndi.binary_erosion(best_region, structure=struct, iterations=int(flow_erode_iters))
    mask_flow &= finite

    # Bright exclusion (walls / reflectors).
    wall_thr = float(np.quantile(I_smooth[finite], float(wall_percentile) / 100.0))
    mask_bright = (I_smooth > wall_thr) & finite
    if int(wall_dilate_iters) > 0:
        mask_bright = ndi.binary_dilation(mask_bright, structure=struct, iterations=int(wall_dilate_iters))
    mask_bright &= finite

    # Guard band around lumen for background exclusion.
    guard = ndi.binary_dilation(mask_flow, structure=struct, iterations=int(guard_dilate_iters))
    guard &= finite

    mask_bg = finite & (~guard) & (~mask_bright)
    # Ensure disjointness.
    mask_bg &= ~mask_flow

    # QC metrics.
    flow_area = int(np.count_nonzero(mask_flow))
    bg_area = int(np.count_nonzero(mask_bg))
    flow_area_frac = float(flow_area / float(total_valid))
    bg_area_frac = float(bg_area / float(total_valid))
    touches_border = bool(mask_flow[0, :].any() or mask_flow[-1, :].any() or mask_flow[:, 0].any() or mask_flow[:, -1].any())

    # Ring check: lumen should be darker than nearby annulus.
    ring_outer = ndi.binary_dilation(mask_flow, structure=struct, iterations=6)
    ring_inner = ndi.binary_dilation(mask_flow, structure=struct, iterations=3)
    ring = ring_outer & (~ring_inner) & finite
    mean_flow = float(np.mean(I_smooth[mask_flow])) if flow_area else float("nan")
    mean_ring = float(np.mean(I_smooth[ring])) if np.any(ring) else float("nan")
    delta_I = float(mean_flow - mean_ring) if np.isfinite(mean_flow) and np.isfinite(mean_ring) else float("nan")

    qc: dict[str, float | int | str | None] = {
        "total_valid": int(total_valid),
        "valid_beam_fraction": float(np.mean(mapping.valid_beams)),
        "flow_area": int(flow_area),
        "bg_area": int(bg_area),
        "flow_area_frac": float(flow_area_frac),
        "bg_area_frac": float(bg_area_frac),
        "seg_best_score": float(best_score),
        "seg_best_edge_support": float(best_edge_support),
        "seg_best_area": int(best_area),
        "seg_best_centroid_dist": float(best_centroid_dist),
        "touches_border": bool(touches_border),
        "delta_I_lumen_minus_ring": float(delta_I),
    }
    params = {
        "sigma": float(sigma),
        "tau_percentile": float(tau_percentile),
        "opening_iters": int(opening_iters),
        "closing_iters": int(closing_iters),
        "flow_erode_iters": int(flow_erode_iters),
        "guard_dilate_iters": int(guard_dilate_iters),
        "wall_percentile": float(wall_percentile),
        "wall_dilate_iters": int(wall_dilate_iters),
        "area_frac_min": float(area_frac_min),
        "area_frac_max": float(area_frac_max),
        "alpha_center": float(alpha_center),
    }
    return TubeMaskResult(
        mask_flow=mask_flow.astype(bool, copy=False),
        mask_bg=mask_bg.astype(bool, copy=False),
        mask_bright=mask_bright.astype(bool, copy=False),
        mapping=mapping,
        qc=qc,
        params=params,
        I_ref_norm=I_norm.astype(np.float32, copy=False),
    )


def build_tube_masks_from_rawbcf(
    dat_path: Path,
    par: RawBCFPar,
    *,
    ref_frame_indices: Iterable[int],
    sigma: float = 1.5,
    tau_percentile: float = 20.0,
    opening_iters: int = 2,
    closing_iters: int = 3,
    flow_erode_iters: int = 3,
    guard_dilate_iters: int = 6,
    wall_percentile: float = 95.0,
    wall_dilate_iters: int = 1,
    area_frac_min: float = 0.005,
    area_frac_max: float = 0.40,
    alpha_center: float = 0.0,
    min_valid_beam_frac: float = 0.50,
) -> TubeMaskResult:
    """
    Auto-select a B-mode->CFM ROI mapping, build a B-mode reference, and segment tube masks.
    """

    candidates = iter_mapping_candidates(par)
    best: TubeMaskResult | None = None
    best_pass = False
    best_score = -1.0
    best_valid = -1

    ref_frame_indices_list = [int(i) for i in ref_frame_indices]
    if not ref_frame_indices_list:
        raise ValueError("ref_frame_indices must be non-empty.")

    for b0, s0, Q in candidates:
        try:
            mapping = build_bmode_cfm_roi_mapping(par, b0=b0, s0=s0, Q=Q)
        except Exception:
            continue
        valid_beams = int(np.count_nonzero(mapping.valid_beams))
        if valid_beams / float(par.num_cfm_beams) < float(min_valid_beam_frac):
            continue
        try:
            I_ref_db, valid_mask = build_bmode_roi_reference_db(
                dat_path, par, mapping, frame_indices=ref_frame_indices_list
            )
            res = segment_tube_masks_from_bmode_ref(
                I_ref_db,
                valid_mask,
                mapping,
                sigma=sigma,
                tau_percentile=tau_percentile,
                opening_iters=opening_iters,
                closing_iters=closing_iters,
                flow_erode_iters=flow_erode_iters,
                guard_dilate_iters=guard_dilate_iters,
                wall_percentile=wall_percentile,
                wall_dilate_iters=wall_dilate_iters,
                area_frac_min=area_frac_min,
                area_frac_max=area_frac_max,
                alpha_center=alpha_center,
            )
        except Exception:
            continue

        # QC pass heuristic for mapping selection (deterministic, B-mode only).
        qc = res.qc
        pass_basic = True
        if float(qc.get("flow_area_frac") or 0.0) < 0.002 or float(qc.get("flow_area_frac") or 0.0) > 0.30:
            pass_basic = False
        if float(qc.get("bg_area_frac") or 0.0) < 0.20:
            pass_basic = False
        if bool(qc.get("touches_border")):
            pass_basic = False
        dI = float(qc.get("delta_I_lumen_minus_ring") or float("nan"))
        if not np.isfinite(dI) or dI >= 0.0:
            pass_basic = False

        score = float(qc.get("seg_best_score") or 0.0)
        # Prefer candidates that pass QC; otherwise use score as fallback.
        better = False
        if pass_basic and not best_pass:
            better = True
        elif pass_basic == best_pass:
            if score > best_score + 1e-9:
                better = True
            elif abs(score - best_score) <= 1e-9 and valid_beams > best_valid:
                better = True
        if better:
            best = res
            best_pass = pass_basic
            best_score = score
            best_valid = valid_beams

    if best is None:
        raise ValueError("Failed to infer a valid B-mode->CFM ROI mapping and tube mask.")

    # Attach mapping-selection status to qc dict (non-float values are OK; caller can JSON-serialize).
    best.qc.setdefault("mapping_qc_pass", bool(best_pass))
    return best
