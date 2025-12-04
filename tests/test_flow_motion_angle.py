import numpy as np

from sim.kwave import common


def test_tile_flow_motion_angle_recorded():
    rng = np.random.default_rng(0)
    # Small synthetic tile (time, height, width)
    cube = rng.standard_normal((12, 4, 4)).astype(np.float32)

    if common.build_motion_basis_temporal is None or common.torch is None:
        raise AssertionError("motion basis utilities unavailable for test")

    basis = common.build_motion_basis_temporal(  # type: ignore[call-arg]
        Lt=4,
        prf_hz=2000.0,
        width_bins=1,
        include_dc=True,
        device="cpu",
        dtype=common.torch.complex64,
    )
    motion_basis_geom = basis.detach().cpu().numpy()

    _, _, info, _ = common._stap_pd_tile_lcmv(
        cube,
        prf_hz=2000.0,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        fd_span_mode="fixed",
        fd_fixed_span_hz=200.0,
        Lt_fixed=4,
        motion_basis_geom=motion_basis_geom,
    )

    angle = info.get("flow_motion_angle_deg")
    assert angle is not None
    assert 0.0 <= angle <= 90.0
