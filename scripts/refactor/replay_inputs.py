"""Replay input-preparation helpers (masks, vessels, baseline option parsing)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sim.kwave.common import SimGeom


def maybe_resize_roi_mask(mask: np.ndarray, geom: SimGeom) -> np.ndarray:
    """Resize a 2D ROI mask to match simulation geometry (Ny, Nx)."""
    mask = np.asarray(mask, dtype=bool)
    h_roi, w_roi = mask.shape
    h_geom, w_geom = int(geom.Ny), int(geom.Nx)

    if h_roi == h_geom and w_roi == w_geom:
        return mask

    if w_roi != w_geom:
        raise ValueError(
            f"ROI mask width mismatch: {w_roi} vs geom.Nx={w_geom}; "
            "cannot safely align to simulation grid."
        )

    if h_roi == h_geom - 1:
        resized = np.zeros((h_geom, w_geom), dtype=bool)
        resized[:h_roi, :] = mask
        return resized
    if h_roi == h_geom + 1:
        return mask[:h_geom, :]

    raise ValueError(
        f"ROI mask height mismatch: {h_roi} vs geom.Ny={h_geom}; "
        "only ±1 pixel differences are supported."
    )


def load_default_masks_and_vessels(
    *,
    src_root: Path,
    geom: SimGeom,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    flow_mask_default: np.ndarray | None = None
    bg_mask_default: np.ndarray | None = None
    micro_vessels_arr: np.ndarray | None = None
    alias_vessels_arr: np.ndarray | None = None

    roi_h1_path = src_root / "roi_H1.npy"
    roi_h0_path = src_root / "roi_H0.npy"
    if roi_h1_path.exists():
        roi_h1 = np.load(roi_h1_path)
        roi_h1 = maybe_resize_roi_mask(roi_h1, geom)
        flow_mask_default = roi_h1
        bg_mask_default = ~roi_h1
        if roi_h0_path.exists():
            try:
                roi_h0 = np.load(roi_h0_path)
                _ = maybe_resize_roi_mask(roi_h0, geom)
            except Exception:
                pass
        micro_path = src_root / "micro_vessels.npy"
        alias_path = src_root / "alias_vessels.npy"
        if micro_path.exists():
            micro_vessels_arr = np.load(micro_path)
        if alias_path.exists():
            alias_vessels_arr = np.load(alias_path)

    if flow_mask_default is None or bg_mask_default is None:
        try:
            pw_dirs = sorted(
                p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("pw_")
            )
        except Exception:
            pw_dirs = []
        for pw_dir in pw_dirs:
            mask_flow_path = pw_dir / "mask_flow.npy"
            mask_bg_path = pw_dir / "mask_bg.npy"
            if not (mask_flow_path.exists() and mask_bg_path.exists()):
                continue
            try:
                flow_mask_candidate = np.load(mask_flow_path)
                bg_mask_candidate = np.load(mask_bg_path)
                flow_mask_candidate = maybe_resize_roi_mask(flow_mask_candidate, geom)
                bg_mask_candidate = maybe_resize_roi_mask(bg_mask_candidate, geom)
            except Exception:
                continue
            flow_mask_default = flow_mask_candidate.astype(bool, copy=False)
            bg_mask_default = bg_mask_candidate.astype(bool, copy=False)
            break

    return flow_mask_default, bg_mask_default, micro_vessels_arr, alias_vessels_arr


def parse_hosvd_options(
    *,
    hosvd_ranks_arg: str | None,
    hosvd_energy_fracs_arg: str | None,
) -> tuple[tuple[int, int, int] | None, tuple[float, float, float] | None]:
    hosvd_ranks: tuple[int, int, int] | None = None
    if hosvd_ranks_arg:
        parts = [p.strip() for p in str(hosvd_ranks_arg).split(",") if p.strip()]
        if len(parts) != 3:
            raise SystemExit(
                "Invalid --hosvd-ranks "
                f"'{hosvd_ranks_arg}'; expected 'rT,rH,rW' with three integers."
            )
        try:
            hosvd_ranks = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise SystemExit(
                "Invalid --hosvd-ranks "
                f"'{hosvd_ranks_arg}'; expected 'rT,rH,rW' with three integers."
            ) from exc

    hosvd_energy_fracs: tuple[float, float, float] | None = None
    if hosvd_energy_fracs_arg and hosvd_ranks is None:
        parts_f = [p.strip() for p in str(hosvd_energy_fracs_arg).split(",") if p.strip()]
        if len(parts_f) != 3:
            raise SystemExit(
                "Invalid --hosvd-energy-fracs "
                f"'{hosvd_energy_fracs_arg}'; expected 'fT,fH,fW' with three floats."
            )
        try:
            hosvd_energy_fracs = (float(parts_f[0]), float(parts_f[1]), float(parts_f[2]))
        except ValueError as exc:
            raise SystemExit(
                "Invalid --hosvd-energy-fracs "
                f"'{hosvd_energy_fracs_arg}'; expected 'fT,fH,fW' with three floats."
            ) from exc

    return hosvd_ranks, hosvd_energy_fracs
