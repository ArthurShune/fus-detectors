from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np

from sim.simus.config import (
    StructuredClutterSpec,
    SimusConfig,
    SimusPreset,
    SimusProfile,
    dataset_meta,
    default_config,
    default_profile_config,
    resolve_vessels,
)
from sim.simus.labels import build_label_pack
from sim.simus.motion import (
    apply_phase_screen,
    build_motion_artifacts,
    build_localized_motion_component,
    build_phase_screen_series,
    sample_motion_displacements_m,
)


try:  # optional dependency
    import pymust  # type: ignore
except Exception:  # pragma: no cover - optional
    pymust = None  # type: ignore


SimusSmokeConfig = SimusConfig


def _require_pymust() -> Any:
    if pymust is None:  # pragma: no cover - optional
        raise ImportError(
            "PyMUST is not installed. Install into the stap-fus env via:\n"
            "  conda run -n stap-fus pip install PyMUST==0.1.9\n"
        )
    return pymust


def _vessel_mask_strip(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    cx: float,
    r: float,
    z_min: float | None = None,
    z_max: float | None = None,
) -> np.ndarray:
    mask = (np.abs(X - float(cx)) <= float(r)) & np.isfinite(Z)
    if z_min is not None:
        mask &= Z >= float(z_min)
    if z_max is not None:
        mask &= Z <= float(z_max)
    return mask


def _vz_profile_from_x(
    x: np.ndarray,
    *,
    cx: float,
    radius_m: float,
    vmax_mps: float,
    profile: Literal["plug", "poiseuille"],
) -> np.ndarray:
    r = float(max(1e-9, float(radius_m)))
    dx = np.abs(np.asarray(x, dtype=np.float32) - np.float32(float(cx)))
    if profile == "plug":
        vz = np.full_like(dx, float(vmax_mps), dtype=np.float32)
    else:
        rr = (dx / np.float32(r)) ** 2
        vz = float(vmax_mps) * (1.0 - rr)
        vz = np.maximum(vz, 0.0).astype(np.float32, copy=False)
    return vz


def _distance_to_segment(
    *,
    x: np.ndarray,
    z: np.ndarray,
    x0: float,
    z0: float,
    x1: float,
    z1: float,
) -> np.ndarray:
    px = np.asarray(x, dtype=np.float32)
    pz = np.asarray(z, dtype=np.float32)
    vx = np.float32(float(x1) - float(x0))
    vz = np.float32(float(z1) - float(z0))
    denom = np.float32(vx * vx + vz * vz)
    if float(denom) <= 0.0:
        return np.sqrt((px - np.float32(float(x0))) ** 2 + (pz - np.float32(float(z0))) ** 2).astype(np.float32, copy=False)
    t = (((px - np.float32(float(x0))) * vx) + ((pz - np.float32(float(z0))) * vz)) / denom
    t = np.clip(t, 0.0, 1.0).astype(np.float32, copy=False)
    proj_x = np.float32(float(x0)) + t * vx
    proj_z = np.float32(float(z0)) + t * vz
    return np.sqrt((px - proj_x) ** 2 + (pz - proj_z) ** 2).astype(np.float32, copy=False)


def _structured_clutter_mask(
    *,
    X: np.ndarray,
    Z: np.ndarray,
    spec: StructuredClutterSpec,
) -> np.ndarray:
    dist = _distance_to_segment(
        x=X,
        z=Z,
        x0=float(spec.x0_m),
        z0=float(spec.z0_m),
        x1=float(spec.x1_m),
        z1=float(spec.z1_m),
    )
    return dist <= (0.5 * float(spec.thickness_m))


def _sample_structured_clutter(
    rng: np.random.Generator,
    *,
    spec: StructuredClutterSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = max(1, int(spec.scatterer_count))
    x0 = float(spec.x0_m)
    z0 = float(spec.z0_m)
    x1 = float(spec.x1_m)
    z1 = float(spec.z1_m)
    t = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    cx = np.float32(x0) + t * np.float32(x1 - x0)
    cz = np.float32(z0) + t * np.float32(z1 - z0)
    dx = np.float32(x1 - x0)
    dz = np.float32(z1 - z0)
    seg_len = float(np.hypot(dx, dz))
    if seg_len <= 1e-9:
        nx = np.float32(1.0)
        nz = np.float32(0.0)
    else:
        nx = np.float32(-dz / seg_len)
        nz = np.float32(dx / seg_len)
    jitter = rng.uniform(
        low=-0.5 * float(spec.thickness_m),
        high=0.5 * float(spec.thickness_m),
        size=n,
    ).astype(np.float32)
    xs = (cx + jitter * nx).astype(np.float32, copy=False)
    zs = (cz + jitter * nz).astype(np.float32, copy=False)
    rc = (float(spec.rc_scale) * rng.standard_normal(n)).astype(np.float32)
    return xs, zs, rc


def _sample_uniform_excluding_mask(
    rng: np.random.Generator,
    *,
    n: int,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    exclude_mask_fn,
    max_tries: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Rejection-sample (x,z) points, excluding a region.

    exclude_mask_fn(x,z)->bool returns True for points to exclude.
    """
    xs = np.empty((n,), dtype=np.float32)
    zs = np.empty((n,), dtype=np.float32)
    filled = 0
    for _ in range(max_tries):
        need = n - filled
        if need <= 0:
            break
        x = rng.uniform(float(x_min), float(x_max), size=need).astype(np.float32)
        z = rng.uniform(float(z_min), float(z_max), size=need).astype(np.float32)
        bad = exclude_mask_fn(x, z)
        keep = ~bad
        k = int(np.sum(keep))
        if k <= 0:
            continue
        xs[filled : filled + k] = x[keep][:k]
        zs[filled : filled + k] = z[keep][:k]
        filled += k
    if filled < n:
        # Fall back: accept remaining samples (smoke only).
        need = n - filled
        xs[filled:] = rng.uniform(float(x_min), float(x_max), size=need).astype(np.float32)
        zs[filled:] = rng.uniform(float(z_min), float(z_max), size=need).astype(np.float32)
    return xs, zs


def _sample_gaussian_excluding_mask(
    rng: np.random.Generator,
    *,
    n: int,
    center_x: float,
    center_z: float,
    sigma_x: float,
    sigma_z: float,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    exclude_mask_fn,
    max_tries: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    xs = np.empty((n,), dtype=np.float32)
    zs = np.empty((n,), dtype=np.float32)
    filled = 0
    while filled < n and max_tries > 0:
        max_tries -= 1
        need = n - filled
        x = rng.normal(loc=float(center_x), scale=max(float(sigma_x), 1e-6), size=need).astype(np.float32)
        z = rng.normal(loc=float(center_z), scale=max(float(sigma_z), 1e-6), size=need).astype(np.float32)
        keep = (
            (x >= float(x_min))
            & (x <= float(x_max))
            & (z >= float(z_min))
            & (z <= float(z_max))
            & (~exclude_mask_fn(x, z))
        )
        k = int(np.sum(keep))
        if k <= 0:
            continue
        xs[filled : filled + k] = x[keep][:k]
        zs[filled : filled + k] = z[keep][:k]
        filled += k
    if filled < n:
        x_fill, z_fill = _sample_uniform_excluding_mask(
            rng,
            n=n - filled,
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            exclude_mask_fn=exclude_mask_fn,
        )
        xs[filled:] = x_fill
        zs[filled:] = z_fill
    return xs, zs


def _match_iq_shape(iq_ch: np.ndarray, ref_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(iq_ch)
    if arr.ndim != 2:
        return arr
    if tuple(arr.shape) == tuple(ref_shape):
        return arr
    out = np.zeros(ref_shape, dtype=arr.dtype)
    h = min(int(ref_shape[0]), int(arr.shape[0]))
    w = min(int(ref_shape[1]), int(arr.shape[1]))
    out[:h, :w] = arr[:h, :w]
    return out


def generate_icube(cfg: SimusConfig) -> dict[str, Any]:
    pm = _require_pymust()
    if int(cfg.T) <= 0:
        raise ValueError("T must be positive")
    if float(cfg.prf_hz) <= 0.0:
        raise ValueError("prf_hz must be positive")
    if int(cfg.H) <= 0 or int(cfg.W) <= 0:
        raise ValueError("H and W must be positive")
    if float(cfg.z_min_m) <= 0.0:
        raise ValueError("z_min_m must be > 0")
    if float(cfg.z_max_m) <= float(cfg.z_min_m):
        raise ValueError("z_max_m must be > z_min_m")

    rng = np.random.default_rng(int(cfg.seed))
    dt_prp = 1.0 / float(cfg.prf_hz)

    # ---- PyMUST parameters ----
    param = pm.getparam(str(cfg.probe))
    # Medium / acquisition parameters used by SIMUS/DASMTX.
    param.c = float(cfg.c_mps)
    fs_mult = float(max(4.0, float(cfg.fs_mult)))
    param.fs = fs_mult * float(param.fc)

    tilt_rad = float(np.deg2rad(float(cfg.tilt_deg)))
    txdel = pm.txdelay(param, tilt_rad).astype(np.float32, copy=False)

    vessels = resolve_vessels(cfg)

    # ---- Output grid + masks ----
    xg = np.linspace(float(cfg.x_min_m), float(cfg.x_max_m), int(cfg.W), dtype=np.float32)
    zg = np.linspace(float(cfg.z_min_m), float(cfg.z_max_m), int(cfg.H), dtype=np.float32)
    X, Z = np.meshgrid(xg, zg)  # (H,W)

    expected_vz_mps = np.zeros((int(cfg.H), int(cfg.W)), dtype=np.float32)
    mask_microvascular = np.zeros((int(cfg.H), int(cfg.W)), dtype=bool)
    mask_nuisance_pa = np.zeros((int(cfg.H), int(cfg.W)), dtype=bool)
    mask_structured_clutter = np.zeros((int(cfg.H), int(cfg.W)), dtype=bool)
    vessel_role_map = np.zeros((int(cfg.H), int(cfg.W)), dtype=np.int16)
    for vessel in vessels:
        vessel_mask = _vessel_mask_strip(
            X=X,
            Z=Z,
            cx=float(vessel.center_x_m),
            r=float(vessel.radius_m),
            z_min=vessel.z_min_m,
            z_max=vessel.z_max_m,
        )
        if not np.any(vessel_mask):
            continue
        vz = _vz_profile_from_x(
            X,
            cx=float(vessel.center_x_m),
            radius_m=float(vessel.radius_m),
            vmax_mps=float(vessel.blood_vmax_mps),
            profile=str(vessel.blood_profile),  # type: ignore[arg-type]
        )
        expected_vz_mps[vessel_mask] = np.maximum(expected_vz_mps[vessel_mask], vz[vessel_mask])
        if vessel.role == "nuisance_pa":
            mask_nuisance_pa |= vessel_mask
            vessel_role_map[vessel_mask] = 2
        else:
            mask_microvascular |= vessel_mask
            vessel_role_map[vessel_mask] = 1

    for clutter in tuple(cfg.structured_clutter):
        mask_structured_clutter |= _structured_clutter_mask(X=X, Z=Z, spec=clutter)

    base_bg = (~(mask_microvascular | mask_nuisance_pa | mask_structured_clutter)).copy()
    base_bg[: max(1, int(round(float(cfg.bg_top_exclusion_frac) * int(cfg.H)))), :] = False
    if int(base_bg.sum()) < 16:
        base_bg = ~(mask_microvascular | mask_nuisance_pa | mask_structured_clutter)

    expected_fd_true_hz = (
        (2.0 * float(param.fc) / max(float(param.c), 1e-9)) * expected_vz_mps
    ).astype(np.float32, copy=False)
    labels = build_label_pack(
        mask_microvascular=mask_microvascular,
        mask_nuisance_pa=mask_nuisance_pa,
        mask_specular_struct=mask_structured_clutter,
        base_bg_mask=base_bg,
        expected_fd_true_hz=expected_fd_true_hz,
        prf_hz=float(cfg.prf_hz),
        bands=cfg.bands,
        guard_px=int(cfg.label_guard_px),
    )

    motion_art = build_motion_artifacts(cfg=cfg, seed=int(cfg.seed) + 2001)

    # ---- Scatterers ----
    x_min, x_max = float(cfg.x_min_m), float(cfg.x_max_m)
    z_min, z_max = float(cfg.z_min_m), float(cfg.z_max_m)

    def _in_excluded_scene(x: np.ndarray, z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x, dtype=bool)
        for vessel in vessels:
            keep = np.abs(x - float(vessel.center_x_m)) <= float(vessel.radius_m)
            if vessel.z_min_m is not None:
                keep &= z >= float(vessel.z_min_m)
            if vessel.z_max_m is not None:
                keep &= z <= float(vessel.z_max_m)
            out |= keep
        for clutter in tuple(cfg.structured_clutter):
            out |= _distance_to_segment(
                x=x,
                z=z,
                x0=float(clutter.x0_m),
                z0=float(clutter.z0_m),
                x1=float(clutter.x1_m),
                z1=float(clutter.z1_m),
            ) <= (0.5 * float(clutter.thickness_m))
        return out

    # Tissue: ordinary background pool plus independently driven local parcels.
    xs_tissue, zs_tissue = _sample_uniform_excluding_mask(
        rng,
        n=int(cfg.tissue_count),
        x_min=x_min,
        x_max=x_max,
        z_min=z_min,
        z_max=z_max,
        exclude_mask_fn=_in_excluded_scene,
    )
    rc_tissue = (float(cfg.tissue_rc_scale) * rng.standard_normal(int(cfg.tissue_count))).astype(np.float32)
    scatterers_init: dict[str, np.ndarray] = {
        "xs_tissue": xs_tissue,
        "zs_tissue": zs_tissue,
        "rc_tissue": rc_tissue,
    }

    background_states: list[dict[str, Any]] = []
    for bidx, compartment in enumerate(tuple(cfg.background_compartments)):
        xs_bg, zs_bg = _sample_gaussian_excluding_mask(
            rng,
            n=int(compartment.scatterer_count),
            center_x=float(compartment.center_x_m),
            center_z=float(compartment.center_z_m),
            sigma_x=float(compartment.sigma_x_m),
            sigma_z=float(compartment.sigma_z_m),
            x_min=x_min,
            x_max=x_max,
            z_min=z_min,
            z_max=z_max,
            exclude_mask_fn=_in_excluded_scene,
        )
        rc_bg = (float(compartment.rc_scale) * rng.standard_normal(int(compartment.scatterer_count))).astype(np.float32)
        dx_bg_px, dz_bg_px, tele_bg = build_localized_motion_component(
            cfg=cfg,
            center_x_m=float(compartment.center_x_m),
            center_z_m=float(compartment.center_z_m),
            sigma_x_m=float(compartment.sigma_x_m),
            sigma_z_m=float(compartment.sigma_z_m),
            amp_px=float(compartment.motion_amp_px),
            sigma_px=float(compartment.motion_sigma_px),
            rho=float(compartment.motion_rho),
            jitter_sigma_px=float(compartment.motion_jitter_sigma_px),
            lateral_scale=float(compartment.lateral_scale),
            axial_scale=float(compartment.axial_scale),
            seed=int(cfg.seed) + 9001 + 37 * bidx,
        )
        background_states.append(
            {
                "spec": compartment,
                "xs": xs_bg,
                "zs": zs_bg,
                "rc": rc_bg,
                "dx_px": dx_bg_px,
                "dz_px": dz_bg_px,
                "telemetry": tele_bg,
            }
        )
        scatterers_init[f"xs_bg_compartment_{bidx}"] = xs_bg.copy()
        scatterers_init[f"zs_bg_compartment_{bidx}"] = zs_bg.copy()
        scatterers_init[f"rc_bg_compartment_{bidx}"] = rc_bg.copy()

    clutter_states: list[dict[str, Any]] = []
    for cidx, clutter in enumerate(tuple(cfg.structured_clutter)):
        xs_clutter, zs_clutter, rc_clutter = _sample_structured_clutter(rng, spec=clutter)
        clutter_states.append({"spec": clutter, "xs": xs_clutter, "zs": zs_clutter, "rc": rc_clutter})
        scatterers_init[f"xs_clutter_{cidx}"] = xs_clutter.copy()
        scatterers_init[f"zs_clutter_{cidx}"] = zs_clutter.copy()
        scatterers_init[f"rc_clutter_{cidx}"] = rc_clutter.copy()

    blood_states: list[dict[str, Any]] = []
    res_scale = max(1, int(cfg.reservoir_scale))
    span = float(max(1e-6, float(cfg.reinject_depth_span_m)))
    for vidx, vessel in enumerate(vessels):
        x_lo = max(x_min, float(vessel.center_x_m) - float(vessel.radius_m))
        x_hi = min(x_max, float(vessel.center_x_m) + float(vessel.radius_m))
        z_lo = max(z_min, float(vessel.z_min_m) if vessel.z_min_m is not None else z_min)
        z_hi = min(z_max, float(vessel.z_max_m) if vessel.z_max_m is not None else z_max)
        n_blood = int(vessel.blood_count)
        xs_blood = rng.uniform(x_lo, x_hi, size=n_blood).astype(np.float32)
        zs_blood = rng.uniform(z_lo, z_hi, size=n_blood).astype(np.float32)
        rc_blood = (float(vessel.blood_rc_scale) * rng.standard_normal(n_blood)).astype(np.float32)
        vz_blood = _vz_profile_from_x(
            xs_blood,
            cx=float(vessel.center_x_m),
            radius_m=float(vessel.radius_m),
            vmax_mps=float(vessel.blood_vmax_mps),
            profile=str(vessel.blood_profile),  # type: ignore[arg-type]
        )

        res_n = int(res_scale * max(1, n_blood))
        rng_res = np.random.default_rng(int(cfg.seed) + 1337 + 17 * vidx)
        res_x = rng_res.uniform(x_lo, x_hi, size=res_n).astype(np.float32)
        res_z_hi = min(z_lo + span, z_hi)
        res_z = rng_res.uniform(z_lo, res_z_hi, size=res_n).astype(np.float32)
        res_rc = (float(vessel.blood_rc_scale) * rng_res.standard_normal(res_n)).astype(np.float32)

        blood_states.append(
            {
                "vessel": vessel,
                "xs": xs_blood,
                "zs": zs_blood,
                "rc": rc_blood,
                "vz": vz_blood,
                "res_x": res_x,
                "res_z": res_z,
                "res_rc": res_rc,
                "res_n": res_n,
                "res_ptr": 0,
                "z_hi": np.float32(z_hi),
            }
        )
        scatterers_init[f"xs_blood0_{vidx}"] = xs_blood.copy()
        scatterers_init[f"zs_blood0_{vidx}"] = zs_blood.copy()
        scatterers_init[f"vz_blood0_{vidx}"] = vz_blood.copy()
        scatterers_init[f"rc_blood_{vidx}"] = rc_blood.copy()
        scatterers_init[f"res_x_{vidx}"] = res_x
        scatterers_init[f"res_z_{vidx}"] = res_z
        scatterers_init[f"res_rc_{vidx}"] = res_rc

    # ---- Pulse loop: SIMUS -> RF -> IQch -> DAS -> IQimg ----
    Icube = np.empty((int(cfg.T), int(cfg.H), int(cfg.W)), dtype=np.complex64)
    M = None
    iq_shape_ref: tuple[int, int] | None = None
    phase_series = None
    phase_telemetry: dict[str, Any] = {"enabled": False, "phase_rms_rad": 0.0}
    noise_rng = np.random.default_rng(int(cfg.seed) + 8101)
    noise_sigmas: list[float] = []

    for t in range(int(cfg.T)):
        dx_tissue_m, dz_tissue_m = sample_motion_displacements_m(
            field_dx_px=motion_art.dx_px[t],
            field_dz_px=motion_art.dz_px[t],
            x_m=xs_tissue,
            z_m=zs_tissue,
            cfg=cfg,
        )
        xs_tissue_t = (xs_tissue + dx_tissue_m).astype(np.float32, copy=False)
        zs_tissue_t = (zs_tissue + dz_tissue_m).astype(np.float32, copy=False)

        clutter_xs = []
        clutter_zs = []
        clutter_rc = [np.asarray(state["rc"], dtype=np.float32) for state in clutter_states]
        for state in clutter_states:
            dx_clutter_m, dz_clutter_m = sample_motion_displacements_m(
                field_dx_px=motion_art.dx_px[t],
                field_dz_px=motion_art.dz_px[t],
                x_m=np.asarray(state["xs"], dtype=np.float32),
                z_m=np.asarray(state["zs"], dtype=np.float32),
                cfg=cfg,
            )
            clutter_xs.append((np.asarray(state["xs"], dtype=np.float32) + dx_clutter_m).astype(np.float32, copy=False))
            clutter_zs.append((np.asarray(state["zs"], dtype=np.float32) + dz_clutter_m).astype(np.float32, copy=False))

        bg_xs = []
        bg_zs = []
        bg_rc = [np.asarray(state["rc"], dtype=np.float32) for state in background_states]
        for state in background_states:
            dx_bg_global_m, dz_bg_global_m = sample_motion_displacements_m(
                field_dx_px=motion_art.dx_px[t],
                field_dz_px=motion_art.dz_px[t],
                x_m=np.asarray(state["xs"], dtype=np.float32),
                z_m=np.asarray(state["zs"], dtype=np.float32),
                cfg=cfg,
            )
            dx_bg_local_m, dz_bg_local_m = sample_motion_displacements_m(
                field_dx_px=np.asarray(state["dx_px"], dtype=np.float32)[t],
                field_dz_px=np.asarray(state["dz_px"], dtype=np.float32)[t],
                x_m=np.asarray(state["xs"], dtype=np.float32),
                z_m=np.asarray(state["zs"], dtype=np.float32),
                cfg=cfg,
            )
            bg_xs.append(
                (
                    np.asarray(state["xs"], dtype=np.float32) + dx_bg_global_m + dx_bg_local_m
                ).astype(np.float32, copy=False)
            )
            bg_zs.append(
                (
                    np.asarray(state["zs"], dtype=np.float32) + dz_bg_global_m + dz_bg_local_m
                ).astype(np.float32, copy=False)
            )

        blood_xs = []
        blood_zs = []
        blood_rc = [np.asarray(state["rc"], dtype=np.float32) for state in blood_states]
        for state in blood_states:
            dx_blood_m, dz_blood_m = sample_motion_displacements_m(
                field_dx_px=motion_art.dx_px[t],
                field_dz_px=motion_art.dz_px[t],
                x_m=np.asarray(state["xs"], dtype=np.float32),
                z_m=np.asarray(state["zs"], dtype=np.float32),
                cfg=cfg,
            )
            blood_xs.append((np.asarray(state["xs"], dtype=np.float32) + dx_blood_m).astype(np.float32, copy=False))
            blood_zs.append((np.asarray(state["zs"], dtype=np.float32) + dz_blood_m).astype(np.float32, copy=False))
        xs = np.concatenate([xs_tissue_t] + bg_xs + clutter_xs + blood_xs).astype(np.float32, copy=False)
        zs = np.concatenate([zs_tissue_t] + bg_zs + clutter_zs + blood_zs).astype(np.float32, copy=False)
        rc = np.concatenate([rc_tissue] + bg_rc + clutter_rc + blood_rc).astype(np.float32, copy=False)

        rf, _rf_spec = pm.simus(xs, zs, rc, txdel, param)
        rf = np.asarray(rf, dtype=np.float32)
        iq_ch = pm.rf2iq(rf, param)
        if phase_series is None:
            n_elem = int(min(iq_ch.shape)) if getattr(iq_ch, "ndim", 0) == 2 else 0
            phase_series, phase_telemetry = build_phase_screen_series(
                T=int(cfg.T),
                n_elem=n_elem,
                spec=cfg.phase_screen,
                seed=int(cfg.seed) + 7001,
            )
        if phase_series is not None:
            iq_ch = apply_phase_screen(iq_ch, phase_series[t])
        if iq_shape_ref is None and getattr(iq_ch, "ndim", 0) == 2:
            iq_shape_ref = (int(iq_ch.shape[0]), int(iq_ch.shape[1]))
        if iq_shape_ref is not None:
            iq_ch = _match_iq_shape(iq_ch, iq_shape_ref)

        if M is None:
            # Complex DAS matrix for I/Q data; SIG.shape convention is MATLAB-like.
            M = pm.dasmtx(1j * np.array(iq_ch.shape, dtype=np.int64), X, Z, txdel, param)

        img = (M @ iq_ch.flatten(order="F")).reshape(X.shape, order="F")
        if cfg.noise.enabled and float(cfg.noise.iq_rms_frac) > 0.0:
            img_arr = np.asarray(img, dtype=np.complex64)
            frame_rms = float(np.sqrt(np.mean(np.abs(img_arr) ** 2))) if img_arr.size else 0.0
            sigma = float(cfg.noise.iq_rms_frac) * frame_rms / float(np.sqrt(2.0))
            noise = sigma * (
                noise_rng.standard_normal(img_arr.shape).astype(np.float32)
                + 1j * noise_rng.standard_normal(img_arr.shape).astype(np.float32)
            )
            img = (img_arr + noise.astype(np.complex64, copy=False)).astype(np.complex64, copy=False)
            noise_sigmas.append(float(sigma))
        Icube[t] = np.asarray(img, dtype=np.complex64)

        for state in blood_states:
            vessel = state["vessel"]
            zs_state = np.asarray(state["zs"], dtype=np.float32)
            vz_state = np.asarray(state["vz"], dtype=np.float32)
            zs_state = (zs_state + vz_state * np.float32(dt_prp)).astype(np.float32, copy=False)

            out = zs_state > np.float32(state["z_hi"])
            n_out = int(np.sum(out))
            if n_out > 0:
                ptr = int(state["res_ptr"])
                res_n = int(state["res_n"])
                idx = (np.arange(n_out, dtype=np.int64) + ptr) % res_n
                state["res_ptr"] = int((ptr + n_out) % res_n)
                xs_state = np.asarray(state["xs"], dtype=np.float32)
                rc_state = np.asarray(state["rc"], dtype=np.float32)
                xs_state[out] = np.asarray(state["res_x"], dtype=np.float32)[idx]
                zs_state[out] = np.asarray(state["res_z"], dtype=np.float32)[idx]
                rc_state[out] = np.asarray(state["res_rc"], dtype=np.float32)[idx]
                vz_state[out] = _vz_profile_from_x(
                    xs_state[out],
                    cx=float(vessel.center_x_m),
                    radius_m=float(vessel.radius_m),
                    vmax_mps=float(vessel.blood_vmax_mps),
                    profile=str(vessel.blood_profile),  # type: ignore[arg-type]
                )
                state["xs"] = xs_state
                state["rc"] = rc_state
                state["vz"] = vz_state
            state["zs"] = zs_state

    # Debug artifacts: initial scatterers + reservoir (small enough for smoke).
    debug = {
        "expected_fd_hz": labels.expected_fd_true_hz,
        "expected_fd_true_hz": labels.expected_fd_true_hz,
        "expected_fd_sampled_hz": labels.expected_fd_sampled_hz,
        "expected_vz_mps": expected_vz_mps,
        "vessel_role_map": vessel_role_map,
        "mask_h0_specular_struct": labels.mask_h0_specular_struct,
        "scatterers_init": scatterers_init,
        "motion_dx_px": motion_art.dx_px,
        "motion_dz_px": motion_art.dz_px,
        "motion_rigid_dx_px": motion_art.rigid_dx_px,
        "motion_rigid_dz_px": motion_art.rigid_dz_px,
        "motion_elastic_base_dx_px": motion_art.elastic_base_dx_px,
        "motion_elastic_base_dz_px": motion_art.elastic_base_dz_px,
        "motion_elastic_coef_x": motion_art.elastic_coef_x,
        "motion_elastic_coef_z": motion_art.elastic_coef_z,
        "motion_telemetry": motion_art.telemetry,
        "phase_screen_rad": np.asarray(phase_series, dtype=np.float32) if phase_series is not None else np.zeros((int(cfg.T), 0), dtype=np.float32),
        "phase_screen_telemetry": phase_telemetry,
        "noise_telemetry": {
            "enabled": bool(cfg.noise.enabled),
            "iq_rms_frac": float(cfg.noise.iq_rms_frac),
            "sigma_q50": float(np.quantile(np.asarray(noise_sigmas, dtype=np.float32), 0.50)) if noise_sigmas else 0.0,
            "sigma_q90": float(np.quantile(np.asarray(noise_sigmas, dtype=np.float32), 0.90)) if noise_sigmas else 0.0,
        },
        "txdel_s": txdel,
        "grid_x_m": xg,
        "grid_z_m": zg,
        "fd_vmax_hz": float(np.max(labels.expected_fd_true_hz)) if labels.expected_fd_true_hz.size else 0.0,
        "scene_telemetry": {
            "n_microvascular_vessels": int(sum(1 for v in vessels if v.role == "microvascular")),
            "n_nuisance_vessels": int(sum(1 for v in vessels if v.role == "nuisance_pa")),
            "n_structured_clutter": int(len(tuple(cfg.structured_clutter))),
            "n_background_compartments": int(len(tuple(cfg.background_compartments))),
            "microvascular_fraction": float(np.mean(mask_microvascular)),
            "nuisance_fraction": float(np.mean(mask_nuisance_pa)),
            "specular_struct_fraction": float(np.mean(labels.mask_h0_specular_struct)),
            "h1_pf_main_fraction": float(np.mean(labels.mask_h1_pf_main)),
            "h1_alias_qc_fraction": float(np.mean(labels.mask_h1_alias_qc)),
            "h0_nuisance_fraction": float(np.mean(labels.mask_h0_nuisance_pa)),
            "background_compartment_scatterers": int(sum(int(state["xs"].size) for state in background_states)),
            "expected_fd_true_q50_hz": float(np.quantile(labels.expected_fd_true_hz[labels.mask_flow], 0.50)) if np.any(labels.mask_flow) else 0.0,
            "expected_fd_sampled_q50_hz": float(np.quantile(labels.expected_fd_sampled_hz[labels.mask_flow], 0.50)) if np.any(labels.mask_flow) else 0.0,
        },
        "background_compartment_telemetry": [dict(state["telemetry"]) for state in background_states],
    }

    return {
        "Icube": Icube,
        "mask_flow": labels.mask_flow.astype(bool, copy=False),
        "mask_bg": labels.mask_bg.astype(bool, copy=False),
        "mask_alias_expected": labels.mask_alias_expected.astype(bool, copy=False),
        "mask_microvascular": labels.mask_microvascular.astype(bool, copy=False),
        "mask_nuisance_pa": labels.mask_nuisance_pa.astype(bool, copy=False),
        "mask_guard": labels.mask_guard.astype(bool, copy=False),
        "mask_h1_pf_main": labels.mask_h1_pf_main.astype(bool, copy=False),
        "mask_h1_alias_qc": labels.mask_h1_alias_qc.astype(bool, copy=False),
        "mask_h0_bg": labels.mask_h0_bg.astype(bool, copy=False),
        "mask_h0_nuisance_pa": labels.mask_h0_nuisance_pa.astype(bool, copy=False),
        "mask_h0_specular_struct": labels.mask_h0_specular_struct.astype(bool, copy=False),
        "debug": debug,
        "param": {"probe": str(cfg.probe), "fc_hz": float(param.fc), "fs_hz": float(param.fs), "c_mps": float(param.c)},
    }
