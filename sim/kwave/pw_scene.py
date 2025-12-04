# sim/kwave/pw_scene.py
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional
import numpy as np

# k-Wave Python 0.4.0 API (names mirror the MATLAB toolbox)
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.utils.signals import tone_burst

# -------------------- Config & helpers -------------------- #


@dataclass
class PWConfig:
    # Grid
    c_brain: float = 1540.0  # m/s
    rho_brain: float = 1000.0  # kg/m^3
    c_skull: float = 2800.0  # m/s
    rho_skull: float = 1900.0  # kg/m^3
    skull_thickness_mm: float = 2.5
    alpha_brain_db_mhz_cm: float = 0.6
    alpha_skull_db_mhz_cm: float = 10.0
    alpha_power: float = 1.5

    # Imaging geometry
    height_mm: float = 30.0  # z depth
    width_mm: float = 32.0  # x span
    f0_mhz: float = 8.0  # center
    cfl: float = 0.2
    pml_size: int = 20

    # Array
    n_elements: int = 128
    pitch_mm: float = 0.25
    kerf_mm: float = 0.0
    aperture_mm: Optional[float] = None  # if None, infer from n_elements*pitch
    elev_mm: float = 0.3

    # Plane-wave TX
    angles_deg: Tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0, 6.0)
    cycles: int = 3
    prf_hz: float = 3000.0
    pulses: int = 64  # slow-time length (keep moderate for runtime)

    # Motion & flow (synthetic vessel moves nonrigidly)
    vessel_radius_mm: float = 1.0
    vessel_center_mm: Tuple[float, float] = (0.0, 18.0)  # (x,z)
    flow_speed_mm_s: float = 8.0
    motion_um: float = 120.0
    motion_freq_hz: float = 0.5
    micro_jitter_um: float = 10.0

    # Macroscopic heterogeneity / dropout metadata (used downstream for logging/telemetry)
    heterogeneity: str = "medium"
    sensor_dropout: float = 0.0

    # Randomness
    seed: int = 0


def _mm(v: float) -> float:
    return v / 1000.0


def _hanning(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1))


# -------------------- Scene construction -------------------- #


def build_grid(cfg: PWConfig) -> Tuple[kWaveGrid, float, float, int, int]:
    lam = cfg.c_brain / (cfg.f0_mhz * 1e6)
    dx = lam / 6.0  # ~λ/6 for accuracy
    dy = lam / 6.0
    Nx = int(np.ceil(_mm(cfg.width_mm) / dx))
    Ny = int(np.ceil(_mm(cfg.height_mm) / dy))
    # Ensure even sizes for PML symmetry, add PML margin inside solver
    Nx = int(max(128, 2 * (Nx // 2)))
    Ny = int(max(128, 2 * (Ny // 2)))

    kgrid = kWaveGrid((Nx, Ny), (dx, dy))
    # Time array: simulate long enough for echoes from max depth to return
    c0 = cfg.c_brain
    t_end = 2.0 * _mm(cfg.height_mm) / c0 + (cfg.cycles / (cfg.f0_mhz * 1e6))
    kgrid.makeTime(c0, cfl=cfg.cfl, t_end=t_end)
    return kgrid, dx, dy, Nx, Ny


def build_medium(cfg: PWConfig, kgrid: kWaveGrid) -> kWaveMedium:
    Nx, Ny = kgrid.Nx, kgrid.Ny
    c_map = np.full((Nx, Ny), cfg.c_brain, dtype=np.float32)
    rho_map = np.full((Nx, Ny), cfg.rho_brain, dtype=np.float32)
    alpha_coeff = np.full((Nx, Ny), cfg.alpha_brain_db_mhz_cm, dtype=np.float32)

    skull_pix = int(np.ceil(_mm(cfg.skull_thickness_mm) / kgrid.dy))
    skull_pix = max(1, min(skull_pix, Ny // 6))
    # Top skull layer
    c_map[:, :skull_pix] = cfg.c_skull
    rho_map[:, :skull_pix] = cfg.rho_skull
    alpha_coeff[:, :skull_pix] = cfg.alpha_skull_db_mhz_cm

    medium = kWaveMedium(
        sound_speed=c_map, density=rho_map, alpha_coeff=alpha_coeff, alpha_power=cfg.alpha_power
    )
    return medium


def build_tx_rx_masks(
    cfg: PWConfig, kgrid: kWaveGrid
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (tx_mask, rx_mask, x_coords_m) where masks are boolean [Nx,Ny].
    Transducer line sits just below the skull layer to avoid PML interaction.
    """
    Nx, Ny = kgrid.Nx, kgrid.Ny
    # aperture (x extent) centered
    if cfg.aperture_mm is None:
        aperture = cfg.n_elements * _mm(cfg.pitch_mm)
    else:
        aperture = _mm(cfg.aperture_mm)
    x_span = aperture
    x0 = -x_span / 2
    x1 = x_span / 2
    # grid coordinates
    x = (np.arange(Nx) - Nx / 2) * kgrid.dx
    z = np.arange(Ny) * kgrid.dy

    # transducer depth just below skull (and PML region handled internally)
    skull_pix = int(np.ceil(_mm(cfg.skull_thickness_mm) / kgrid.dy))
    z_tx_idx = max(skull_pix + 2, cfg.pml_size + 2)
    z_rx_idx = z_tx_idx

    # TX: a continuous line over aperture -> we will time delay along x to steer
    tx_mask = np.zeros((Nx, Ny), dtype=bool)
    # RX: discrete receiving elements (n_elements) across same aperture
    rx_mask = np.zeros((Nx, Ny), dtype=bool)

    # compute x positions of the n_elements centers
    if cfg.n_elements > 1:
        x_elem = np.linspace(x0, x1, cfg.n_elements)
    else:
        x_elem = np.array([0.0], dtype=float)

    # nearest grid x indices for element centers
    xi = np.round(x_elem / kgrid.dx + Nx / 2).astype(int)
    xi = np.clip(xi, 1, Nx - 2)

    tx_mask[np.min(xi) : np.max(xi) + 1, z_tx_idx] = True
    rx_mask[xi, z_rx_idx] = True
    return tx_mask, rx_mask, x_elem


# -------------------- Motion model (vessel) -------------------- #


def vessel_mask(cfg: PWConfig, kgrid: kWaveGrid, t_sec: float) -> np.ndarray:
    """Boolean [Nx,Ny] vessel region that moves/deforms over time (nonrigid)."""
    rng = np.random.default_rng(cfg.seed)
    # Nonrigid oscillation along x, slow drift in z
    amp_x = _mm(cfg.motion_um / 1000.0) * (1.0 + 0.05 * np.sin(2 * np.pi * 0.37 * t_sec))
    amp_z = _mm(0.5 * cfg.motion_um / 1000.0)
    dx = amp_x * math.sin(2 * np.pi * cfg.motion_freq_hz * t_sec + 0.3)
    dz = amp_z * math.sin(2 * np.pi * 0.25 * cfg.motion_freq_hz * t_sec + 1.1)

    # Micro‑jitter per frame
    dx += _mm(cfg.micro_jitter_um / 1000.0) * rng.normal()
    dz += _mm(0.5 * cfg.micro_jitter_um / 1000.0) * rng.normal()

    # Map vessel center into grid coords
    x0_mm, z0_mm = cfg.vessel_center_mm
    x0_m = _mm(x0_mm) + dx
    z0_m = _mm(z0_mm) + dz
    r_m = _mm(cfg.vessel_radius_mm)

    Nx, Ny = kgrid.Nx, kgrid.Ny
    X = (np.arange(Nx) - Nx / 2) * kgrid.dx
    Z = np.arange(Ny) * kgrid.dy
    XX, ZZ = np.meshgrid(X, Z, indexing="ij")
    return (XX - x0_m) ** 2 + (ZZ - z0_m) ** 2 <= r_m**2


def apply_vessel_scatter(
    medium: kWaveMedium,
    vessel: np.ndarray,
    c_delta: float = -20.0,
    rho_delta: float = -30.0,
    alpha_delta: float = 0.8,
) -> None:
    """
    Apply small contrast in vessel to create slow‑flow scatterers.
    (Negative deltas -> slower sound speed & lower density inside.)
    """
    c = medium.sound_speed.copy()
    rho = medium.density.copy()
    alpha = medium.alpha_coeff.copy()
    c[vessel] = np.clip(c[vessel] + c_delta, 1200.0, 3000.0)
    rho[vessel] = np.clip(rho[vessel] + rho_delta, 900.0, 2200.0)
    alpha[vessel] = np.clip(alpha[vessel] + alpha_delta, 0.1, 20.0)
    medium.sound_speed = c
    medium.density = rho
    medium.alpha_coeff = alpha
