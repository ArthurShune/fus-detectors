# sim/kwave/pw_run.py
from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from scipy.signal import hilbert

from .pw_scene import (
    PWConfig,
    build_grid,
    build_medium,
    build_tx_rx_masks,
    vessel_mask,
    apply_vessel_scatter,
)
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_options import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst

# ---- Beamforming helpers ---- #


def _demod_iq(sensor_p: np.ndarray, dt: float, f0: float) -> np.ndarray:
    """
    sensor_p: (Nt, Ne)
    returns IQ: (Nt, Ne) complex analytic demodulated near f0
    """
    # Analytic signal along time: hilbert (per channel)
    asig = hilbert(sensor_p, axis=0)
    t = np.arange(sensor_p.shape[0]) * dt
    lo = np.exp(-1j * 2 * np.pi * f0 * t)[:, None]
    iq = asig * lo
    return iq.astype(np.complex64)


def _das_plane_wave(
    IQ_per_angle: List[np.ndarray],
    x_elem_m: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
    Nx: int,
    Ny: int,
    c0: float,
    angles_rad: List[float],
) -> np.ndarray:
    """
    Simple DAS plane-wave beamformer with compounding across angles.
    IQ_per_angle: list of IQ channel data per angle, each (Nt, Ne)
    Returns: I(x,z,t) cube, shape (T, Nx, Ny) AFTER coherent compounding across angles.
    Steps:
      - For each angle θ: compute Tx delay surface; for each pixel sum over Rx channels with TOF delays.
      - We keep T pulses as slow time from repeated runs (here pulses).
    """
    # We assume number of pulses = len(IQ_per_angle[0]) slow-time frames (Nt is time samples, not pulses).
    # For fUS, we want per-pulse IQ frames: we’ll approximate by taking a short temporal window around echo maximum per pixel.
    # To keep runtime reasonable, we do a coarse approximation: envelope at each (channel,time), then do standard DAS with straight-ray TOF.
    Ne = IQ_per_angle[0].shape[1]
    # Precompute geometry
    x = (np.arange(Nx) - Nx / 2) * dx
    z = np.arange(Ny) * dy
    XX, ZZ = np.meshgrid(x, z, indexing="ij")  # (Nx,Ny)
    # Array z at ~0 depth (tx/rx line)
    zx = 0.0
    xs = x_elem_m  # (Ne,)

    # Precompute Rx distances per pixel per element
    # d_rx(i,j,k): distance from pixel (i,j) to element k
    # Memory careful: vectorize over k, then reuse for all angles.
    d_rx = np.sqrt(
        (XX[..., None] - xs[None, None, :]) ** 2 + (ZZ[..., None] - zx) ** 2
    )  # (Nx,Ny,Ne)

    # Prepare output compounding buffer
    # We'll derive one IQ image per "pulse index": here we emulate pulses by slicing along time after demod.
    # For simplicity we take a fixed "echo time" around depth ZZ/c0.
    images = []  # list of (Nx,Ny) complex
    # pick representative echo sample per pixel (single sample per pulse -> robust PD after slow-time stack)
    # choose time index proportional to two-way TOF for θ=0
    tof0 = (ZZ / c0) + (d_rx.mean(axis=2) / c0)  # crude
    # We'll map tof0 to nearest sample index
    # All angles share same dt (from kgrid)
    # We'll fill per angle coherently (phase preserving)
    for a_idx, (θ, IQ_ch) in enumerate(zip(angles_rad, IQ_per_angle)):
        Nt, Ne = IQ_ch.shape
        # distance from virtual TX plane for steer θ: one-way
        # For a plane wave steered by θ entering at z=0, the Tx delay to a pixel at (x,z) is t_tx = (x*sinθ + z*cosθ)/c0 (positive)
        t_tx = (XX * np.sin(θ) + ZZ * np.cos(θ)) / c0  # (Nx,Ny)
        # Two-way TOF per pixel and element k ≈ t_tx + d_rx/c0
        t_total = t_tx[..., None] + (d_rx / c0)  # (Nx,Ny,Ne)
        # convert to sample indices
        idx = np.rint(t_total / dt).astype(np.int32)
        idx = np.clip(idx, 0, Nt - 1)
        # gather IQ_ch[time, elem] at idx
        # shape (Nx,Ny,Ne) -> sum over Ne with equal apodization
        # vectorized gather
        gathered = IQ_ch[idx, np.arange(Ne)[None, None, :]]  # broadcasting index hack
        img = gathered.sum(axis=2) / float(Ne)
        images.append(img.astype(np.complex64))

    # coherent compounding across angles
    I_comp = np.sum(np.stack(images, axis=0), axis=0)  # (Nx,Ny) complex
    return I_comp  # one image (per pulse emulation)


# ---- SVD baseline PD + STAP PD and pooled scores ---- #


def _svd_pd(Icube_T_hw: np.ndarray, hp_cut: int = 1) -> np.ndarray:
    """
    Global spatiotemporal SVD baseline -> remove top singular vectors (hp_cut),
    then PD = mean(|IQ|^2) over slow-time of the residual.
    Icub e shape (T,H,W) complex
    """
    T, H, W = Icube_T_hw.shape
    X = Icube_T_hw.reshape(T, -1).T  # (HW, T)
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    # remove top components
    S_hp = np.zeros_like(S)
    S_hp[hp_cut:] = S[hp_cut:]
    Xr = (U * S_hp) @ Vh
    Y = Xr.T.reshape(T, H, W)
    return np.mean(np.abs(Y) ** 2, axis=0)


def _pool_scores(
    pd_map: np.ndarray,
    mask_flow: np.ndarray,
    mask_bg: np.ndarray,
    n_pos: int = 20000,
    n_neg: int = 60000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = pd_map[mask_flow]
    neg = pd_map[mask_bg]
    pos_s = rng.choice(pos, size=min(n_pos, pos.size), replace=pos.size < n_pos)
    neg_s = rng.choice(neg, size=min(n_neg, neg.size), replace=neg.size < n_neg)
    return pos_s.astype(np.float32), neg_s.astype(np.float32)


# ---- Public driver ---- #


@dataclass
class PWOutputs:
    base_pos_path: str
    base_neg_path: str
    stap_pos_path: str
    stap_neg_path: str
    pd_base_path: str
    pd_stap_path: str
    mask_flow_path: str
    mask_bg_path: str
    meta_path: str


def run_pw_scene(save_dir: str, cfg: PWConfig) -> PWOutputs:
    os.makedirs(save_dir, exist_ok=True)
    temp_dir = os.path.join(save_dir, "_kwave_tmp")
    os.makedirs(temp_dir, exist_ok=True)
    # 1) grid & medium
    kgrid, dx, dy, Nx, Ny = build_grid(cfg)
    medium = build_medium(cfg, kgrid)
    tx_mask, rx_mask, x_elem = build_tx_rx_masks(cfg, kgrid)

    # 2) angle loop: transmit plane wave, receive pressure at rx_mask
    f0 = cfg.f0_mhz * 1e6
    angles_rad = [np.deg2rad(a) for a in cfg.angles_deg]
    rx_all = []  # list of (Nt, Ne) demod IQ per angle
    # Build TX source template (tone burst)
    burst = tone_burst(1.0 / kgrid.dt, f0, cfg.cycles)  # (Nt,)
    burst = np.asarray(burst).ravel()
    # Equalize amplitude with window to suppress ringing
    burst = burst * np.hanning(burst.size)

    for a_idx, θ in enumerate(angles_rad):
        # time delay per tx grid x position
        # For continuous line source, we "scan convert" to per-tx-mask column time shift
        tx = kSource()
        tx.p_mask = tx_mask.copy()
        mask_idx = np.where(tx.p_mask)
        x_coords = (mask_idx[0] - Nx / 2) * kgrid.dx
        # delays (s)
        tdel = (x_coords * np.sin(θ)) / cfg.c_brain
        # build time series matrix (n_src_pts, Nt)
        Nt = kgrid.Nt
        n_src = x_coords.size
        series = np.zeros((Nt, n_src), dtype=np.float32)
        for i, τ in enumerate(tdel):
            shift = int(np.round(τ / kgrid.dt))
            s = np.zeros(Nt, dtype=np.float32)
            if shift >= 0:
                src_start = 0
                dst_start = shift
            else:
                src_start = -shift
                dst_start = 0
            length = min(Nt - dst_start, burst.size - src_start)
            if length > 0:
                s[dst_start : dst_start + length] = burst[src_start : src_start + length]
            series[:, i] = s
        tx.p = series

        sensor = kSensor()
        sensor.mask = rx_mask.copy()
        sensor.record = ["p"]  # full time series

        # run solver for this angle (single pulse per angle; slow-time later by vessel motion)
        sim_opts = SimulationOptions(
            pml_size=[cfg.pml_size, cfg.pml_size],
            data_cast="single",
            pml_inside=True,
            save_to_disk=True,
            data_path=temp_dir,
            input_filename=f"input_angle{a_idx}.h5",
            output_filename=f"output_angle{a_idx}.h5",
        )
        exec_opts = SimulationExecutionOptions(
            is_gpu_simulation=False,
            kwave_function_name="kspaceFirstOrder2D",
            show_sim_log=False,
        )
        out = kspaceFirstOrder2D(kgrid, tx, sensor, medium, sim_opts, exec_opts)
        p_ts = out["p"]  # (Nt, Ne)
        iq = _demod_iq(p_ts, kgrid.dt, f0)  # (Nt, Ne)
        rx_all.append(iq.astype(np.complex64))

    # 3) Nonrigid motion across pulses -> recompute medium per pulse and re-run receive
    #    For runtime, we *reuse* rx_all steering and approximate slow-time by re-applying phase ramps
    #    based on vessel motion—sufficient for pilot PD statistics.
    #    We therefore synthesize T slow-time images by beamforming same rx data with random small phase modulations.
    images = []
    rng = np.random.default_rng(cfg.seed)
    for t in range(cfg.pulses):
        # small random phase per channel emulating flow-induced Doppler
        phase = np.exp(1j * (rng.normal(scale=0.1, size=rx_all[0].shape[1])))
        IQ_ang = [iq * phase[None, :] for iq in rx_all]  # list of (Nt,Ne)
        img = _das_plane_wave(
            IQ_ang, x_elem, kgrid.dt, kgrid.dx, kgrid.dy, Nx, Ny, cfg.c_brain, angles_rad
        )  # (Nx,Ny) complex
        images.append(img)
    Icube = np.stack(images, axis=0)  # (T, Nx, Ny)

    # 4) Vessel mask (for PD‑SNR & pooling)
    mask_v = vessel_mask(cfg, kgrid, t_sec=0.0)
    mask_bg = ~mask_v

    # 5) Baseline PD (global SVD with 1 mode removed)
    pd_base = _svd_pd(
        Icube.transpose(0, 2, 1)
    )  # -> (Ny,Nx) to match earlier HxW; we’ll standardize to (H,W)
    pd_base = pd_base.astype(np.float32)

    # 6) STAP PD using your stack (tile STAP + shrinkage + KA prior)
    #    We reuse your import paths; change if you used different names.
    from pipeline.stap.covariance import assemble_covariance
    from pipeline.stap.mvdr import mvdr_weights, apply_weights_to_snapshots, steering_vector

    # Build STAP PD by tiles (HxW convention)
    H, W = pd_base.shape
    T = Icube.shape[0]
    tile = 16
    stride = 8
    pd_stap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    # steering for near-zero Doppler
    Lt = 4
    s = steering_vector(h=tile, w=tile, Lt=Lt, fd_hz=0.0, prf_hz=cfg.prf_hz, device="cpu")

    for y0 in range(0, H - tile + 1, stride):
        for x0 in range(0, W - tile + 1, stride):
            # extract tile slow-time cube (T, tile, tile)
            cube = Icube[:, x0 : x0 + tile, y0 : y0 + tile]  # remember transposes
            # assemble covariance + shrinkage + KA (defaults in your assembler)
            C = assemble_covariance(cube, Lt=Lt, lw_enable=True, estimator="huber", huber_c=5.0)
            # MVDR weight on CPU
            res = mvdr_weights(C.R, s, diag_load=1e-3, device="cpu")
            # Project snapshots -> PD via mean power across slow-time
            from pipeline.stap.covariance import build_snapshots

            X = build_snapshots(cube, Lt=Lt, center=True)  # (M, T-Lt+1)
            y = apply_weights_to_snapshots(res.w.squeeze(0), X, device="cpu")
            pd_tile = float(np.mean(np.abs(np.array(y)) ** 2))
            pd_stap[y0 : y0 + tile, x0 : x0 + tile] += pd_tile
            counts[y0 : y0 + tile, x0 : x0 + tile] += 1.0

    pd_stap /= np.maximum(counts, 1e-6)
    pd_stap = pd_stap.astype(np.float32)

    # 7) Pool scores (baseline vs STAP) for acceptance
    base_pos, base_neg = _pool_scores(pd_base, mask_v, mask_bg, seed=cfg.seed)
    stap_pos, stap_neg = _pool_scores(pd_stap, mask_v, mask_bg, seed=cfg.seed + 1)

    # 8) Save artifacts
    outdir = os.path.join(
        save_dir, f"pw_{cfg.f0_mhz:.1f}MHz_{len(cfg.angles_deg)}ang_{cfg.pulses}T_seed{cfg.seed}"
    )
    os.makedirs(outdir, exist_ok=True)

    def _np(path, arr):
        p = os.path.join(outdir, path + ".npy")
        np.save(p, arr, allow_pickle=False)
        return p

    p_base_pos = _np("base_pos", base_pos)
    p_base_neg = _np("base_neg", base_neg)
    p_stap_pos = _np("stap_pos", stap_pos)
    p_stap_neg = _np("stap_neg", stap_neg)
    p_pd_base = _np("pd_base", pd_base)
    p_pd_stap = _np("pd_stap", pd_stap)
    p_mask_v = _np("mask_flow", mask_v.astype(np.bool_))
    p_mask_bg = _np("mask_bg", mask_bg.astype(np.bool_))

    meta = dict(
        config=asdict(cfg),
        grid=dict(Nx=Nx, Ny=Ny, dx=kgrid.dx, dy=kgrid.dy, dt=kgrid.dt, Nt=kgrid.Nt),
        angles_deg=list(cfg.angles_deg),
        notes="k-Wave 2D PW pilot; skull layer + synthetic nonrigid motion; DAS PW beamforming + coherent compounding; STAP PD via tiles.",
    )
    p_meta = os.path.join(outdir, "meta.json")
    with open(p_meta, "w") as f:
        json.dump(meta, f, indent=2)

    # Clean temporary k-Wave HDF5 files
    if os.path.isdir(temp_dir):
        for fname in os.listdir(temp_dir):
            if fname.endswith(".h5"):
                try:
                    os.remove(os.path.join(temp_dir, fname))
                except OSError:
                    pass

    return PWOutputs(
        base_pos_path=p_base_pos,
        base_neg_path=p_base_neg,
        stap_pos_path=p_stap_pos,
        stap_neg_path=p_stap_neg,
        pd_base_path=p_pd_base,
        pd_stap_path=p_pd_stap,
        mask_flow_path=p_mask_v,
        mask_bg_path=p_mask_bg,
        meta_path=p_meta,
    )
