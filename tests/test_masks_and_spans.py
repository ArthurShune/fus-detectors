from pathlib import Path

import numpy as np
import pytest

from sim.kwave.common import (
    SimGeom,
    build_grid_and_time,
    _default_masks,
    _stap_pd_tile_lcmv,
    run_angle_once,
)

try:
    # If k-Wave Python is installed and CPU backend available, run the smoke by default
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC  # noqa: F401

    _KWAVE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _KWAVE_AVAILABLE = False


def test_default_masks_nonempty_and_disjoint():
    # Sweep a few geometries to ensure masks are usable
    for Nx, Ny in [(24, 24), (48, 32), (64, 40)]:
        g = SimGeom(Nx=Nx, Ny=Ny, dx=90e-6, dy=90e-6, c0=1540.0, rho0=1000.0)
        mask_flow, mask_bg = _default_masks(g, Ny, Nx)
        assert mask_flow.shape == (Ny, Nx)
        assert mask_bg.shape == (Ny, Nx)
        assert mask_flow.any()
        assert mask_bg.any()
        assert not np.any(mask_flow & mask_bg)
        # If background has enough support, it should avoid a top PML band
        if mask_bg.sum() >= 64:
            assert not mask_bg[: min(g.pml_size + 4, Ny // 4), :].any()


def test_fixed_span_grid_clamped_and_odd():
    # Build a tiny synthetic slow-time tile and request a fixed span
    T, h, w = 28, 6, 6
    prf = 2800.0
    rng = np.random.default_rng(0)
    cube = (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )
    # Ask for a large span; grid must still be bounded and odd-length
    band_frac_tile, score_tile, info, _ = _stap_pd_tile_lcmv(
        cube,
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.1,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        fd_span_mode="fixed",
        fd_fixed_span_hz=0.8 * prf,  # overly large request
        grid_step_rel=0.1,
        min_pts=5,
        max_pts=9,
        capture_debug=False,
        device="cpu",
    )
    fd_grid = np.asarray(info["fd_grid"]) if info.get("fd_grid") is not None else np.array([])
    assert fd_grid.size % 2 == 1 and fd_grid.size >= 1
    # Frequencies stay within a safe fraction of PRF
    if fd_grid.size:
        assert np.all(np.abs(fd_grid) <= 0.49 * prf + 1e-6)


@pytest.mark.skipif(not _KWAVE_AVAILABLE, reason="k-Wave not available in environment")
def test_run_angle_once_cpu_smoke(tmp_path):
    # Very small geometry + short t_end for a fast CPU-only k-Wave smoke
    g = SimGeom(Nx=16, Ny=16, dx=120e-6, dy=120e-6, c0=1540.0, rho0=1000.0, ncycles=1, t_end=2e-6)
    out_dir = Path(tmp_path) / "angle0"
    out_dir.mkdir(parents=True, exist_ok=True)
    ad = run_angle_once(out_dir, angle_deg=0.0, g=g, use_gpu=False)
    assert isinstance(ad.dt, float) and ad.dt > 0.0
    assert ad.rf.ndim == 2 and ad.rf.shape[1] == g.Nx
    # Ensure reasonable energy level (non-degenerate)
    assert float(np.abs(ad.rf).mean()) > 0.0
