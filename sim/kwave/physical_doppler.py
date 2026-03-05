from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

from sim.kwave.common import SimGeom


ReinjectionMode = Literal["reservoir", "wrap"]


@dataclass(frozen=True)
class VesselSpec:
    """Simple 2D tube model in the (x,z) plane.

    angle_deg is relative to the +z axis (0 => axial flow along +z).
    """

    center_x_m: float
    center_z_m: float
    radius_m: float
    vmax_mps: float
    angle_deg: float = 0.0
    profile: Literal["poiseuille", "plug"] = "poiseuille"


@dataclass(frozen=True)
class PsfSpec:
    """Phase 1 PSF: (optionally) depth-dependent separable Gaussian blur.

    When alpha_*_per_m == 0, this reduces to a constant sigma blur.
    """

    sigma_x0_px: float = 1.25
    sigma_z0_px: float = 1.75
    alpha_x_per_m: float = 0.0
    alpha_z_per_m: float = 0.0
    calib_path: str | None = None
    mode: str = "nearest"


@dataclass(frozen=True)
class NoiseSpec:
    """Complex AWGN added after PSF."""

    snr_db: float | None = 25.0


@dataclass(frozen=True)
class PhysicalDopplerConfig:
    sim_geom: SimGeom
    prf_hz: float
    pulses_per_set: int
    ensembles: int
    seed: int
    vessels: tuple[VesselSpec, ...]

    # Blood reflectivity amplitude relative to tissue, in power dB.
    blood_to_tissue_power_db: float = -15.0
    tissue_static: bool = True

    # Blood advection settings
    reinjection_mode: ReinjectionMode = "reservoir"
    pad_x_px: int = 0
    pad_z_px: int = 0
    reservoir_scale: int = 4
    reservoir_seed: int | None = None
    reservoir_step_x_px: int = 137
    reservoir_step_z_px: int = 251
    reservoir_offset_init: tuple[int, int] | None = None
    vessel_offset_kx: int = 193
    vessel_offset_kz: int = 97

    psf: PsfSpec = PsfSpec()
    noise: NoiseSpec = NoiseSpec()

    @property
    def T(self) -> int:
        return int(self.pulses_per_set) * int(self.ensembles)


def _complex_white(rng: np.random.Generator, shape: Sequence[int]) -> np.ndarray:
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    out = (real + 1j * imag).astype(np.complex64)
    out *= np.complex64(1.0 / math.sqrt(2.0))
    return out


def _compute_vessel_fields(
    *,
    Nx: int,
    Ny: int,
    dx_m: float,
    dz_m: float,
    vessels: Sequence[VesselSpec],
    x_center_idx: float,
    z0_idx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Mesh in (z,x) order to match Icube[:, Ny, Nx].
    x = (np.arange(Nx, dtype=np.float32) - float(x_center_idx)) * float(dx_m)
    z = (np.arange(Ny, dtype=np.float32) - float(z0_idx)) * float(dz_m)
    XX, ZZ = np.meshgrid(x, z, indexing="xy")  # (Ny,Nx)

    mask = np.zeros((Ny, Nx), dtype=bool)
    vx = np.zeros((Ny, Nx), dtype=np.float32)
    vz = np.zeros((Ny, Nx), dtype=np.float32)
    speed2 = np.zeros((Ny, Nx), dtype=np.float32)
    vessel_id = -np.ones((Ny, Nx), dtype=np.int16)

    for vidx, v in enumerate(vessels):
        theta = float(np.deg2rad(v.angle_deg))
        ux = float(math.sin(theta))
        uz = float(math.cos(theta))
        nx = float(math.cos(theta))
        nz = float(-math.sin(theta))

        dx = XX - float(v.center_x_m)
        dz = ZZ - float(v.center_z_m)
        # Signed perpendicular distance to tube centerline (in 2D).
        dist = nx * dx + nz * dz
        r = np.abs(dist)
        inside = r <= float(v.radius_m)
        if not np.any(inside):
            continue

        rr = np.clip((r / float(v.radius_m)) ** 2, 0.0, 1.0)
        if v.profile == "plug":
            vm = float(v.vmax_mps) * inside.astype(np.float32)
        else:
            vm = float(v.vmax_mps) * (1.0 - rr)
            vm *= inside.astype(np.float32)
        vxi = vm * ux
        vzi = vm * uz

        s2 = vxi * vxi + vzi * vzi
        take = s2 > speed2
        if np.any(take):
            speed2[take] = s2[take]
            vx[take] = vxi[take]
            vz[take] = vzi[take]
            vessel_id[take] = np.int16(vidx)
        mask |= inside

    return vx, vz, mask, vessel_id


def _sample_complex_bilinear(
    field: np.ndarray,
    src_z: np.ndarray,
    src_x: np.ndarray,
    *,
    mode: str,
    cval: float = 0.0,
) -> np.ndarray:
    real = map_coordinates(
        np.asarray(field.real, dtype=np.float32),
        [src_z, src_x],
        order=1,
        mode=mode,
        cval=float(cval),
        prefilter=False,
    )
    imag = map_coordinates(
        np.asarray(field.imag, dtype=np.float32),
        [src_z, src_x],
        order=1,
        mode=mode,
        cval=float(cval),
        prefilter=False,
    )
    return (real + 1j * imag).astype(np.complex64, copy=False)


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def _load_psf_calib(psf: PsfSpec) -> PsfSpec:
    if not psf.calib_path:
        return psf
    try:
        import json

        with open(psf.calib_path, "r", encoding="utf-8") as f:
            calib = json.load(f)
        return dataclasses.replace(
            psf,
            sigma_x0_px=float(calib.get("sigma_x0_px", psf.sigma_x0_px)),
            sigma_z0_px=float(calib.get("sigma_z0_px", psf.sigma_z0_px)),
            alpha_x_per_m=float(calib.get("alpha_x_per_m", psf.alpha_x_per_m)),
            alpha_z_per_m=float(calib.get("alpha_z_per_m", psf.alpha_z_per_m)),
        )
    except Exception:
        return psf


def _apply_psf(frame: np.ndarray, g: SimGeom, psf: PsfSpec) -> np.ndarray:
    sig_x0 = float(max(0.0, psf.sigma_x0_px))
    sig_z0 = float(max(0.0, psf.sigma_z0_px))
    ax = float(psf.alpha_x_per_m or 0.0)
    az = float(psf.alpha_z_per_m or 0.0)
    if sig_x0 <= 0.0 and sig_z0 <= 0.0:
        return frame.astype(np.complex64, copy=False)
    if abs(ax) < 1e-12 and abs(az) < 1e-12:
        real = gaussian_filter(
            np.asarray(frame.real, dtype=np.float32),
            sigma=(sig_z0, sig_x0),
            mode=str(psf.mode),
        )
        imag = gaussian_filter(
            np.asarray(frame.imag, dtype=np.float32),
            sigma=(sig_z0, sig_x0),
            mode=str(psf.mode),
        )
        return (real + 1j * imag).astype(np.complex64, copy=False)

    # Depth-dependent approximation: blur along x per row, then apply a
    # space-variant blur along z using per-row kernels. This is slower than
    # constant-sigma filtering but is deterministic and auditable.
    Ny, Nx = frame.shape
    z_m = (np.arange(Ny, dtype=np.float32) * float(g.dy)).astype(np.float32, copy=False)
    sig_x = np.maximum(0.0, sig_x0 + ax * z_m).astype(np.float32, copy=False)
    sig_z = np.maximum(0.0, sig_z0 + az * z_m).astype(np.float32, copy=False)

    real_x = np.empty((Ny, Nx), dtype=np.float32)
    imag_x = np.empty((Ny, Nx), dtype=np.float32)
    for y in range(Ny):
        sx = float(sig_x[y])
        if sx <= 0.0:
            real_x[y] = np.asarray(frame.real[y], dtype=np.float32)
            imag_x[y] = np.asarray(frame.imag[y], dtype=np.float32)
        else:
            real_x[y] = gaussian_filter1d(
                np.asarray(frame.real[y], dtype=np.float32),
                sigma=sx,
                axis=0,
                mode=str(psf.mode),
            )
            imag_x[y] = gaussian_filter1d(
                np.asarray(frame.imag[y], dtype=np.float32),
                sigma=sx,
                axis=0,
                mode=str(psf.mode),
            )

    def _blur_z_var(src: np.ndarray) -> np.ndarray:
        out = np.empty_like(src)
        for y in range(Ny):
            sz = float(sig_z[y])
            if sz <= 0.0:
                out[y] = src[y]
                continue
            rad = int(max(1, math.ceil(3.0 * sz)))
            idx0 = max(0, y - rad)
            idx1 = min(Ny - 1, y + rad)
            offs = np.arange(idx0, idx1 + 1, dtype=np.float32) - float(y)
            w = np.exp(-0.5 * (offs / sz) ** 2).astype(np.float32)
            w /= float(np.sum(w) + 1e-12)
            out[y] = (src[idx0 : idx1 + 1] * w[:, None]).sum(axis=0)
        return out

    real = _blur_z_var(real_x)
    imag = _blur_z_var(imag_x)
    return (real + 1j * imag).astype(np.complex64, copy=False)


def build_expected_fd_hz(cfg: PhysicalDopplerConfig, vz_mps: np.ndarray) -> np.ndarray:
    f0 = float(cfg.sim_geom.f0)
    c0 = float(cfg.sim_geom.c0)
    return (2.0 * f0 / max(c0, 1e-9) * vz_mps).astype(np.float32, copy=False)


def generate_icube(
    cfg: PhysicalDopplerConfig,
) -> dict[str, Any]:
    """
    Generate a canonical physical-Doppler dataset.

    Returns a dict with:
      - Icube: complex64 (T,Ny,Nx)
      - mask_flow: bool (Ny,Nx)
      - mask_bg: bool (Ny,Nx)
      - mask_alias_expected: bool (Ny,Nx)
      - debug: dict with vx_mps/vz_mps/fd_expected_hz
    """
    g = cfg.sim_geom
    Nx, Ny = int(g.Nx), int(g.Ny)
    if Nx <= 0 or Ny <= 0:
        raise ValueError("SimGeom Nx/Ny must be positive")
    if cfg.prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive")
    if cfg.T <= 0:
        raise ValueError("Need at least one slow-time frame")
    if not cfg.vessels:
        raise ValueError("At least one VesselSpec is required")

    dt = 1.0 / float(cfg.prf_hz)

    pad_x = max(0, int(cfg.pad_x_px))
    pad_z = max(0, int(cfg.pad_z_px))
    Nx_pad = Nx + 2 * pad_x
    Ny_pad = Ny + 2 * pad_z

    vx_pad, vz_pad, mask_flow_pad, vessel_id_pad = _compute_vessel_fields(
        Nx=Nx_pad,
        Ny=Ny_pad,
        dx_m=float(g.dx),
        dz_m=float(g.dy),
        vessels=cfg.vessels,
        x_center_idx=(Nx / 2.0 + float(pad_x)),
        z0_idx=float(pad_z),
    )
    if not np.any(mask_flow_pad):
        raise ValueError("Vessel configuration produced an empty flow mask")

    y0 = pad_z
    x0 = pad_x
    ys = slice(y0, y0 + Ny)
    xs = slice(x0, x0 + Nx)

    vx_mps = vx_pad[ys, xs]
    vz_mps = vz_pad[ys, xs]
    mask_flow = mask_flow_pad[ys, xs]
    mask_bg = ~mask_flow

    fd_expected_hz = build_expected_fd_hz(cfg, vz_mps)
    mask_alias_expected = np.abs(fd_expected_hz) > (0.5 * float(cfg.prf_hz))

    rng = np.random.default_rng(int(cfg.seed))
    tissue = _complex_white(rng, (Ny, Nx))
    tissue[mask_flow] = np.complex64(0.0)
    tissue_rms = float(np.sqrt(np.mean(np.abs(tissue[mask_bg]) ** 2)) + 1e-12)
    tissue = (tissue / np.complex64(tissue_rms)).astype(np.complex64, copy=False)

    blood_amp_ratio = float(10 ** (float(cfg.blood_to_tissue_power_db) / 20.0))
    blood_pad = np.zeros((Ny_pad, Nx_pad), dtype=np.complex64)

    res_scale = max(1, int(cfg.reservoir_scale))
    target = int(res_scale * max(Nx_pad, Ny_pad))
    Hres = _next_pow2(target)
    Wres = _next_pow2(target)
    reservoir_seed = int(cfg.reservoir_seed) if cfg.reservoir_seed is not None else int(cfg.seed + 1337)
    rng_res = np.random.default_rng(reservoir_seed)
    reservoir = _complex_white(rng_res, (Hres, Wres))
    if cfg.reservoir_offset_init is not None:
        ox0, oz0 = int(cfg.reservoir_offset_init[0]), int(cfg.reservoir_offset_init[1])
    else:
        ox0 = int(rng_res.integers(0, Wres))
        oz0 = int(rng_res.integers(0, Hres))

    # Initialize blood texture inside the padded vessel region from the reservoir.
    z_idx_pad, x_idx_pad = np.meshgrid(
        np.arange(Ny_pad, dtype=np.int64),
        np.arange(Nx_pad, dtype=np.int64),
        indexing="ij",
    )
    init_rz = (z_idx_pad + oz0) % Hres
    init_rx = (x_idx_pad + ox0) % Wres
    blood_pad[mask_flow_pad] = reservoir[init_rz[mask_flow_pad], init_rx[mask_flow_pad]]
    blood_rms = float(np.sqrt(np.mean(np.abs(blood_pad[mask_flow_pad]) ** 2)) + 1e-12)
    blood_pad[mask_flow_pad] = (blood_pad[mask_flow_pad] / np.complex64(blood_rms)).astype(
        np.complex64, copy=False
    )

    x_idx, z_idx = np.meshgrid(
        np.arange(Nx_pad, dtype=np.float32),
        np.arange(Ny_pad, dtype=np.float32),
        indexing="xy",
    )

    dx_pix = (vx_pad * dt / float(g.dx)).astype(np.float32, copy=False)
    dz_pix = (vz_pad * dt / float(g.dy)).astype(np.float32, copy=False)
    delta_phi = (4.0 * math.pi * float(g.f0) / max(float(g.c0), 1e-9) * vz_pad * dt).astype(
        np.float32, copy=False
    )
    phase_step = np.exp(1j * delta_phi).astype(np.complex64, copy=False)

    Icube = np.empty((cfg.T, Ny, Nx), dtype=np.complex64)

    psf = _load_psf_calib(cfg.psf)

    step_x = int(cfg.reservoir_step_x_px)
    step_z = int(cfg.reservoir_step_z_px)
    if step_x == 0:
        step_x = 137
    if step_z == 0:
        step_z = 251
    for t in range(cfg.T):
        src_x = x_idx - dx_pix
        src_z = z_idx - dz_pix

        adv = _sample_complex_bilinear(
            blood_pad,
            src_z,
            src_x,
            mode="constant",
            cval=0.0,
        )

        if cfg.reinjection_mode == "wrap":
            # Simple wrap-around: periodic in the main domain (kept for debugging only).
            src_xi = np.mod(np.rint(src_x).astype(np.int64), Nx_pad).astype(np.int64, copy=False)
            src_zi = np.mod(np.rint(src_z).astype(np.int64), Ny_pad).astype(np.int64, copy=False)
            inside_src = mask_flow_pad[src_zi, src_xi]
            reinject = mask_flow_pad & (~inside_src)
            if np.any(reinject):
                adv[reinject] = blood_pad[src_zi[reinject], src_xi[reinject]]
        else:
            # Deterministic reinjection from a larger reservoir texture.
            src_xi = np.rint(src_x).astype(np.int64)
            src_zi = np.rint(src_z).astype(np.int64)
            in_bounds = (src_xi >= 0) & (src_xi < Nx_pad) & (src_zi >= 0) & (src_zi < Ny_pad)
            src_xi_clip = np.clip(src_xi, 0, Nx_pad - 1)
            src_zi_clip = np.clip(src_zi, 0, Ny_pad - 1)
            inside_src = np.zeros_like(mask_flow_pad)
            inside_src[in_bounds] = mask_flow_pad[src_zi_clip[in_bounds], src_xi_clip[in_bounds]]
            reinject = mask_flow_pad & (~inside_src)
            if np.any(reinject):
                ox = (ox0 + t * step_x) % Wres
                oz = (oz0 + t * step_z) % Hres
                vid = vessel_id_pad[reinject].astype(np.int64, copy=False)
                rx = (x_idx[reinject].astype(np.int64) + int(ox) + vid * int(cfg.vessel_offset_kx)) % Wres
                rz = (z_idx[reinject].astype(np.int64) + int(oz) + vid * int(cfg.vessel_offset_kz)) % Hres
                adv[reinject] = reservoir[rz, rx]

        adv[~mask_flow_pad] = np.complex64(0.0)
        adv[mask_flow_pad] *= phase_step[mask_flow_pad]
        blood_pad = adv

        blood = blood_pad[ys, xs]
        reflectivity = tissue + np.complex64(blood_amp_ratio) * blood
        blurred = _apply_psf(reflectivity, g, psf)

        if cfg.noise.snr_db is not None and float(cfg.noise.snr_db) > 0.0:
            sig_rms = float(np.sqrt(np.mean(np.abs(blurred) ** 2)) + 1e-12)
            noise_rms = sig_rms / float(10 ** (float(cfg.noise.snr_db) / 20.0))
            noise = _complex_white(rng, (Ny, Nx))
            noise *= np.complex64(noise_rms)
            blurred = blurred + noise

        Icube[t] = blurred.astype(np.complex64, copy=False)

    return {
        "Icube": Icube,
        "mask_flow": mask_flow.astype(bool, copy=False),
        "mask_bg": mask_bg.astype(bool, copy=False),
        "mask_alias_expected": mask_alias_expected.astype(bool, copy=False),
        "debug": {
            "vx_mps": vx_mps,
            "vz_mps": vz_mps,
            "fd_expected_hz": fd_expected_hz,
        },
    }


def default_brainlike_config(
    *,
    preset: Literal["microvascular_like", "alias_stress"],
    seed: int,
    Nx: int = 240,
    Ny: int = 240,
    dx: float = 90e-6,
    dy: float = 90e-6,
    prf_hz: float = 1500.0,
    pulses_per_set: int = 64,
    ensembles: int = 5,
    f0_hz: float = 7.5e6,
    c0: float = 1540.0,
) -> PhysicalDopplerConfig:
    g = SimGeom(
        Nx=int(Nx),
        Ny=int(Ny),
        dx=float(dx),
        dy=float(dy),
        c0=float(c0),
        rho0=1000.0,
        pml_size=16,
        f0=float(f0_hz),
        ncycles=3,
    )

    # Place a vessel at mid-depth with a small radius, centered laterally.
    # These values are intentionally conservative; downstream validation will
    # adjust presets if needed, without altering detector profiles.
    z0 = 0.55 * Ny * float(dy)
    x0 = 0.0
    base = VesselSpec(
        center_x_m=float(x0),
        center_z_m=float(z0),
        radius_m=float(0.00018),
        vmax_mps=float(0.015),
        angle_deg=float(25.0),
        profile="poiseuille",
    )
    vessels: tuple[VesselSpec, ...]
    if preset == "alias_stress":
        vessels = (
            dataclasses.replace(base, radius_m=float(0.00028), vmax_mps=float(0.090), angle_deg=10.0),
        )
    else:
        vessels = (base,)

    return PhysicalDopplerConfig(
        sim_geom=g,
        prf_hz=float(prf_hz),
        pulses_per_set=int(pulses_per_set),
        ensembles=int(ensembles),
        seed=int(seed),
        vessels=vessels,
        pad_x_px=32 if (Nx <= 96 and Ny <= 96) else (192 if preset == "alias_stress" else 96),
        pad_z_px=32 if (Nx <= 96 and Ny <= 96) else (192 if preset == "alias_stress" else 96),
    )


def dataset_meta(cfg: PhysicalDopplerConfig) -> Dict[str, Any]:
    """JSON-friendly config snapshot for dataset provenance."""
    d = dataclasses.asdict(cfg)
    # Convert tuples to lists for JSON stability.
    d["vessels"] = [dataclasses.asdict(v) for v in cfg.vessels]
    return d
