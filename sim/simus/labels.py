from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

from sim.simus.config import DopplerBandSpec


@dataclass(frozen=True)
class SimusLabelPack:
    mask_flow: np.ndarray
    mask_bg: np.ndarray
    mask_alias_expected: np.ndarray
    mask_microvascular: np.ndarray
    mask_nuisance_pa: np.ndarray
    mask_guard: np.ndarray
    mask_h1_pf_main: np.ndarray
    mask_h1_alias_qc: np.ndarray
    mask_h0_bg: np.ndarray
    mask_h0_nuisance_pa: np.ndarray
    mask_h0_specular_struct: np.ndarray
    expected_fd_true_hz: np.ndarray
    expected_fd_sampled_hz: np.ndarray


def fold_to_nyquist(fd_hz: np.ndarray, prf_hz: float) -> np.ndarray:
    prf = float(prf_hz)
    nyq = 0.5 * prf
    fd = np.asarray(fd_hz, dtype=np.float32)
    return (((fd + nyq) % prf) - nyq).astype(np.float32, copy=False)


def _in_band(abs_fd_hz: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    return (abs_fd_hz >= float(lo_hz)) & (abs_fd_hz <= float(hi_hz))


def _guard_ring(mask_flow: np.ndarray, guard_px: int) -> np.ndarray:
    flow = np.asarray(mask_flow, dtype=bool)
    if int(guard_px) <= 0 or not np.any(flow):
        return np.zeros_like(flow, dtype=bool)
    structure = np.ones((3, 3), dtype=bool)
    outer = binary_dilation(flow, structure=structure, iterations=int(guard_px))
    inner = binary_erosion(flow, structure=structure, iterations=int(guard_px), border_value=0)
    return (outer & ~inner).astype(bool, copy=False)


def build_label_pack(
    *,
    mask_microvascular: np.ndarray,
    mask_nuisance_pa: np.ndarray,
    mask_specular_struct: np.ndarray | None,
    base_bg_mask: np.ndarray,
    mask_bg_zone: np.ndarray | None = None,
    mask_pf_zone: np.ndarray | None = None,
    mask_nuisance_zone: np.ndarray | None = None,
    mask_alias_source: np.ndarray | None = None,
    expected_fd_true_hz: np.ndarray,
    prf_hz: float,
    bands: DopplerBandSpec,
    guard_px: int,
) -> SimusLabelPack:
    mask_micro = np.asarray(mask_microvascular, dtype=bool)
    mask_nuisance = np.asarray(mask_nuisance_pa, dtype=bool)
    mask_specular = np.asarray(mask_specular_struct, dtype=bool) if mask_specular_struct is not None else np.zeros_like(mask_micro, dtype=bool)
    mask_flow = (mask_micro | mask_nuisance).astype(bool, copy=False)
    bg = np.asarray(base_bg_mask, dtype=bool) & (~mask_flow) & (~mask_specular)
    bg_zone = np.asarray(mask_bg_zone, dtype=bool) if mask_bg_zone is not None else np.ones_like(mask_micro, dtype=bool)
    pf_zone = np.asarray(mask_pf_zone, dtype=bool) if mask_pf_zone is not None else np.ones_like(mask_micro, dtype=bool)
    nuisance_zone = (
        np.asarray(mask_nuisance_zone, dtype=bool)
        if mask_nuisance_zone is not None
        else np.ones_like(mask_micro, dtype=bool)
    )
    alias_source = (
        np.asarray(mask_alias_source, dtype=bool)
        if mask_alias_source is not None
        else mask_flow.astype(bool, copy=False)
    )
    fd_true = np.asarray(expected_fd_true_hz, dtype=np.float32)
    fd_sampled = np.abs(fold_to_nyquist(fd_true, float(prf_hz))).astype(np.float32, copy=False)
    mask_alias_expected = (mask_flow & (np.abs(fd_true) > (0.5 * float(prf_hz)))).astype(bool, copy=False)
    guard = _guard_ring(mask_flow, int(guard_px))

    pf_core = _in_band(fd_sampled, bands.pf_core_low_hz, bands.pf_core_high_hz)
    pa_band = _in_band(fd_sampled, bands.pa_low_hz, bands.pa_high_hz)

    mask_h1_pf_main = mask_micro & pf_core & (~mask_alias_expected) & (~guard) & pf_zone
    mask_h1_alias_qc = alias_source & (mask_alias_expected | pa_band) & (~guard) & pf_zone
    mask_h0_nuisance = mask_nuisance & pa_band & (~guard) & nuisance_zone
    mask_h0_specular = mask_specular & (~guard) & (~mask_flow) & nuisance_zone
    mask_h0_bg = bg & bg_zone & (~guard)
    mask_bg = (bg & bg_zone).astype(bool, copy=False)

    return SimusLabelPack(
        mask_flow=mask_flow.astype(bool, copy=False),
        mask_bg=mask_bg,
        mask_alias_expected=mask_alias_expected.astype(bool, copy=False),
        mask_microvascular=mask_micro.astype(bool, copy=False),
        mask_nuisance_pa=mask_nuisance.astype(bool, copy=False),
        mask_guard=guard.astype(bool, copy=False),
        mask_h1_pf_main=mask_h1_pf_main.astype(bool, copy=False),
        mask_h1_alias_qc=mask_h1_alias_qc.astype(bool, copy=False),
        mask_h0_bg=mask_h0_bg.astype(bool, copy=False),
        mask_h0_nuisance_pa=mask_h0_nuisance.astype(bool, copy=False),
        mask_h0_specular_struct=mask_h0_specular.astype(bool, copy=False),
        expected_fd_true_hz=fd_true.astype(np.float32, copy=False),
        expected_fd_sampled_hz=fd_sampled.astype(np.float32, copy=False),
    )
