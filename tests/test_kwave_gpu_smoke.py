import numpy as np
import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from sim.kwave.common import (
    SimGeom,
    _baseline_pd,
    _stap_pd,
    _stap_pd_tile_lcmv,
    build_grid_and_time,
    build_linear_masks,
    build_medium,
    build_source_p,
)

try:
    from kwave.kspaceFirstOrder2D import kspaceFirstOrder2DC
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.options.simulation_options import SimulationOptions
    from kwave.options.simulation_execution_options import SimulationExecutionOptions

    _KWAVE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _KWAVE_AVAILABLE = False


def _gpu_available() -> bool:
    if torch is not None and torch.cuda.is_available():
        return True
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(
    not (_KWAVE_AVAILABLE and _gpu_available()), reason="k-Wave GPU backend unavailable"
)
def test_kwave_gpu_smoke(tmp_path):
    # Tiny domain to keep GPU run fast
    g = SimGeom(
        Nx=32,
        Ny=24,
        dx=120e-6,
        dy=120e-6,
        c0=1540.0,
        rho0=1000.0,
        ncycles=1,
        t_end=2.5e-6,
        f0=7.0e6,
    )

    kgrid = build_grid_and_time(g)
    medium = build_medium(g)
    tx_mask, rx_mask = build_linear_masks(g)

    source = kSource()
    source.p_mask = tx_mask
    source.p = build_source_p(g, kgrid, tx_mask, theta_rad=0.0)

    sensor = kSensor(mask=rx_mask, record=["p"])

    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_x_size=g.pml_size,
        pml_y_size=g.pml_size,
        save_to_disk=True,
        data_path=str(tmp_path),
        input_filename="gpu_input.h5",
        output_filename="gpu_output.h5",
    )
    exec_opts = SimulationExecutionOptions(
        is_gpu_simulation=True,
        show_sim_log=False,
    )

    sensor_data = kspaceFirstOrder2DC(
        kgrid,
        source,
        sensor,
        medium,
        sim_opts,
        exec_opts,
    )

    if isinstance(sensor_data, dict):
        sensor_data = sensor_data.get("p")
    assert isinstance(sensor_data, np.ndarray)
    assert sensor_data.shape[1] == g.Nx
    assert float(np.abs(sensor_data).mean()) > 0.0


@pytest.mark.skipif(torch is None or not torch.cuda.is_available(), reason="CUDA not available")
def test_stap_gpu_parity():
    rng = np.random.default_rng(0)
    T, h, w = 24, 4, 4
    prf = 2800.0
    cube = (rng.standard_normal((T, h, w)) + 1j * rng.standard_normal((T, h, w))).astype(
        np.complex64
    )
    tile = cube[:, :4, :4]
    kwargs = dict(
        prf_hz=prf,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        constraint_ridge=0.12,
        msd_lambda=5e-2,
        msd_ridge=0.12,
        msd_agg_mode="trim10",
        fd_span_mode="psd",
        fd_span_rel=(0.2, 0.8),
        grid_step_rel=0.1,
        min_pts=3,
        max_pts=5,
        capture_debug=False,
    )

    band_cpu, score_cpu, info_cpu, _ = _stap_pd_tile_lcmv(tile, device="cpu", **kwargs)
    band_gpu, score_gpu, info_gpu, _ = _stap_pd_tile_lcmv(tile, device="cuda", **kwargs)

    assert not info_cpu.get("fallback", False)
    assert not info_gpu.get("fallback", False), info_gpu

    np.testing.assert_allclose(band_gpu, band_cpu, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(score_gpu, score_cpu, rtol=1e-4, atol=1e-6)
