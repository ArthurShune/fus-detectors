from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import torch

from pipeline.stap.temporal import projector_from_tones
from pipeline.stap.temporal_shared import (
    build_temporal_hankels_batch,
    conditioned_lambda_batch,
    robust_temporal_cov_batch,
)


@dataclass
class HemoStapConfig:
    """
    Configuration for hemodynamic STAP on PD time-series.
    """

    dt: float  # seconds between PD frames (e.g. 0.1 for Macé)
    L_t: int  # Hankel slow-time length (e.g. 8 or 10)
    pf_band: Tuple[float, float]  # flow band (Hz), e.g. (0.05, 0.3)
    pa_band: Tuple[float, float]  # alias/noise band (Hz), e.g. (0.6, 1.5)
    pg_band: Tuple[float, float]  # guard band (Hz), e.g. (0.3, 0.6)
    cov_mode: Literal["scm", "huber", "tyler"] = "scm"
    diag_load: float = 1e-3
    kappa_target: float = 200.0
    use_glrt: bool = True
    glrt_rho: float = 1e-2
    # Optional background covariance blending: R_eff = (1-bg_beta)*R_i + bg_beta*R_bg.
    bg_beta: float = 0.0


def _preprocess_pd_series(p_tiles: np.ndarray) -> np.ndarray:
    """
    Center tile-averaged PD per tile over time.

    Parameters
    ----------
    p_tiles : (N_tiles, T_pd) array

    Returns
    -------
    p_centered : (N_tiles, T_pd) array
    """

    mean = p_tiles.mean(axis=1, keepdims=True)
    return p_tiles - mean


def _build_hemo_projectors(
    cfg: HemoStapConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build Pf, Pg, Pa, Po projectors for hemodynamic bands using DFT tones.

    Pf and Pa are constructed as projectors onto the span of complex tones
    whose continuous-time frequencies fall into the specified Pf/Pa bands.
    Po is defined as I - Pf - Pa - Pg, where Pg is a guard band.
    """

    L_t = int(cfg.L_t)
    dt = float(cfg.dt)
    if L_t < 2:
        raise ValueError(f"HemoStapConfig.L_t must be >= 2 (got {L_t})")

    # Discrete-time indexes and continuous-time sampling grid.
    n = torch.arange(L_t, dtype=torch.float64, device=device)
    t_grid = n * dt

    # Full DFT frequency grid for this L_t and dt.
    freqs = torch.fft.rfftfreq(L_t, d=dt).to(device=device)  # (L_r,)

    pf_lo, pf_hi = cfg.pf_band
    pa_lo, pa_hi = cfg.pa_band
    pg_lo, pg_hi = cfg.pg_band

    # NOTE: band masks must be disjoint. We treat the guard as an "open edge"
    # separator by removing any frequencies that lie in Pf or Pa. This avoids
    # double-counting bins when band edges fall exactly on the DFT grid.
    pf_mask = (freqs >= pf_lo) & (freqs <= pf_hi)
    pa_mask = (freqs >= pa_lo) & (freqs <= pa_hi)
    if bool(torch.any(pf_mask & pa_mask)):
        raise ValueError(
            "HemoStapConfig bands overlap: pf_band and pa_band must be disjoint "
            f"(pf_band={cfg.pf_band}, pa_band={cfg.pa_band})"
        )
    pg_mask = (freqs >= pg_lo) & (freqs <= pg_hi) & (~pf_mask) & (~pa_mask)

    def _tones_for_mask(mask: torch.Tensor) -> torch.Tensor:
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return torch.empty((L_t, 0), dtype=torch.complex128, device=device)
        f_sel = freqs[idx]  # (M,)
        # Complex exponentials e^{j 2π f t}.
        A = torch.exp(2j * torch.pi * torch.outer(t_grid, f_sel))
        return A.to(dtype=torch.complex128, device=device)

    Cf = _tones_for_mask(pf_mask)
    Ca = _tones_for_mask(pa_mask)
    Cg = _tones_for_mask(pg_mask)

    def _proj_from_tones(C: torch.Tensor) -> torch.Tensor:
        if C.numel() == 0:
            return torch.zeros((L_t, L_t), dtype=torch.complex128, device=device)
        P = projector_from_tones(C)
        return P.to(dtype=torch.complex128, device=device)

    Pf_c = _proj_from_tones(Cf)
    Pa_c = _proj_from_tones(Ca)
    Pg_c = _proj_from_tones(Cg)

    I_c = torch.eye(L_t, dtype=torch.complex128, device=device)
    Po_c = I_c - Pf_c - Pa_c - Pg_c

    # Symmetrize and cast to real float tensors for use with real z.
    Pf = 0.5 * (Pf_c + Pf_c.conj().transpose(-2, -1))
    Pa = 0.5 * (Pa_c + Pa_c.conj().transpose(-2, -1))
    Pg = 0.5 * (Pg_c + Pg_c.conj().transpose(-2, -1))
    Po = 0.5 * (Po_c + Po_c.conj().transpose(-2, -1))

    return (
        Pf.real.to(dtype=dtype),
        Pg.real.to(dtype=dtype),
        Pa.real.to(dtype=dtype),
        Po.real.to(dtype=dtype),
    )


def hemo_stap_scores_for_tiles(
    p_tiles: np.ndarray,
    cfg: HemoStapConfig,
    bg_mask: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    Compute hemodynamic STAP scores for a batch of tile-averaged PD series.

    Parameters
    ----------
    p_tiles : np.ndarray
        Tile-averaged PD time-series, shape (N_tiles, T_pd).
    cfg : HemoStapConfig
        Configuration specifying dt, L_t, Pf/Pa/guard bands, and covariance options.

    Returns
    -------
    scores : dict
        A dictionary with fields:
          - 'Ef', 'Ea', 'Eg', 'Eo' : band energies per tile (Pf, Pa, Pg, Po).
          - 'hemo_br'        : Pf/Pa band-ratio per tile.
          - 'hemo_glrt'      : GLRT-like Pf/(Pa+rho*||z||^2) per tile (or None).
    """

    if p_tiles.ndim != 2:
        raise ValueError(f"Expected (N_tiles, T_pd) input, got shape {p_tiles.shape}")
    N_tiles, T_pd = p_tiles.shape
    if T_pd <= cfg.L_t:
        raise ValueError(f"T_pd must be > L_t (got T_pd={T_pd}, L_t={cfg.L_t})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # 1. Preprocess: center each series over time.
    p_centered = _preprocess_pd_series(np.asarray(p_tiles, dtype=np.float32))
    x = torch.as_tensor(p_centered, dtype=dtype, device=device)  # (N_tiles, T_pd)

    # 2. Build Hankel stacks using existing temporal helpers.
    # Shape: (B,T,H,W) with H=W=1 for each tile.
    cube_B_T_hw = x.unsqueeze(-1).unsqueeze(-1)  # (B,T,1,1)
    S, R_scm = build_temporal_hankels_batch(
        cube_B_T_hw,
        cfg.L_t,
        center=True,
        device=str(device),
        dtype=dtype,
    )
    # S : (B,L_t,N,H,W) ; R_scm : (B,L_t,L_t)

    # 3. Covariance estimation per tile.
    if cfg.cov_mode == "scm":
        R_hat = R_scm
    else:
        # Flatten snapshots over spatial dims: (B,L_t,K)
        B, L_t, N, H, W = S.shape
        X = S.permute(0, 1, 3, 4, 2).contiguous().view(B, L_t, -1)
        R_hat, _ = robust_temporal_cov_batch(
            X,
            estimator="huber" if cfg.cov_mode == "huber" else "tyler",
        )

    # Optional background covariance blending using tiles flagged by bg_mask.
    if bg_mask is not None and cfg.bg_beta > 0.0:
        if bg_mask.shape[0] != N_tiles:
            raise ValueError(
                f"bg_mask length {bg_mask.shape[0]} does not match N_tiles={N_tiles}"
            )
        bg_mask_bool = np.asarray(bg_mask, dtype=bool)
        if bg_mask_bool.any():
            R_bg = R_hat[bg_mask_bool].mean(dim=0, keepdim=True)  # (1,L_t,L_t)
            beta = float(cfg.bg_beta)
            R_hat = (1.0 - beta) * R_hat + beta * R_bg

    # 4. Condition covariance (diagonal loading + κ control).
    lam_out, _, _ = conditioned_lambda_batch(
        R_hat,
        base_lambda=cfg.diag_load,
        kappa_target=cfg.kappa_target,
    )
    eye = torch.eye(cfg.L_t, dtype=dtype, device=device)
    R_loaded = R_hat + lam_out.view(-1, 1, 1) * eye

    # 5. Whitening and pooled snapshot per tile.
    B, L_t, _ = R_loaded.shape
    _, _, N_snap, H, W = S.shape
    z = torch.empty((B, L_t), dtype=dtype, device=device)
    S_flat = S.view(B, L_t, -1)  # (B,L_t,K)
    for b in range(B):
        Rb = 0.5 * (R_loaded[b] + R_loaded[b].T)
        evals, U = torch.linalg.eigh(Rb.to(dtype=torch.float64))
        evals_clamped = torch.clamp(evals, min=1e-6)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(evals_clamped))
        Wb = (U @ D_inv_sqrt @ U.T).to(dtype=dtype)
        Yb = Wb @ S_flat[b]  # (L_t,K)
        z[b] = Yb.mean(dim=1)

    # 6. Band projectors in hemodynamic space.
    Pf, Pg, Pa, Po = _build_hemo_projectors(cfg, device=device, dtype=dtype)

    # 7. Band energies and scores per tile.
    Ef = torch.empty((B,), dtype=dtype, device=device)
    Ea = torch.empty((B,), dtype=dtype, device=device)
    Eg = torch.empty((B,), dtype=dtype, device=device)
    Eo = torch.empty((B,), dtype=dtype, device=device)

    for b in range(B):
        zb = z[b]
        Ef[b] = torch.dot(zb, Pf @ zb)
        Ea[b] = torch.dot(zb, Pa @ zb)
        Eg[b] = torch.dot(zb, Pg @ zb)
        Eo[b] = torch.dot(zb, Po @ zb)

    eps = 1e-6
    hemo_br = Ef / (Ea + eps)

    if cfg.use_glrt:
        z_norm2 = (z * z).sum(dim=1)
        denom = Ea + cfg.glrt_rho * z_norm2
        hemo_glrt = Ef / (denom + eps)
    else:
        hemo_glrt = torch.zeros_like(hemo_br)

    return {
        "Ef": Ef.detach().cpu().numpy(),
        "Ea": Ea.detach().cpu().numpy(),
        "Eg": Eg.detach().cpu().numpy(),
        "Eo": Eo.detach().cpu().numpy(),
        "hemo_br": hemo_br.detach().cpu().numpy(),
        "hemo_glrt": hemo_glrt.detach().cpu().numpy(),
    }


__all__ = ["HemoStapConfig", "hemo_stap_scores_for_tiles"]
