import numpy as np
import torch

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    ka_blend_covariance_temporal,
    ka_prior_temporal_from_psd,
)


def test_library_prior_reduces_mismatch_and_conditioning() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    Lt = 4
    prf = 3000.0
    Cf = bandpass_constraints_temporal(
        Lt,
        prf,
        np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64),
        device="cpu",
    )

    Rt = torch.tensor(
        [
            [4.5 + 0j, 0.6 - 0.1j, 0.2 + 0.05j, 0.0],
            [0.6 + 0.1j, 3.2 + 0j, 0.1 + 0j, 0.05j],
            [0.2 - 0.05j, 0.1 + 0j, 1.2 + 0j, 0.08 - 0.02j],
            [0.0, -0.05j, 0.08 + 0.02j, 0.7 + 0j],
        ],
        dtype=torch.complex64,
    )

    R0_analytic = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=False,
        device="cpu",
    )

    R0_lib = Rt + 0.05 * torch.eye(Lt, dtype=torch.complex64)

    Rlam_an, tel_an = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0_analytic,
        Cf_flow=Cf,
        beta=0.4,
        kappa_target=40.0,
        device="cpu",
    )
    Rlam_lib, tel_lib = ka_blend_covariance_temporal(
        R_sample=Rt,
        R0_prior=R0_lib,
        Cf_flow=Cf,
        beta=0.4,
        kappa_target=40.0,
        device="cpu",
    )

    assert tel_lib["mismatch_full"] <= tel_an["mismatch_full"] * 0.9
    assert tel_lib["mismatch"] <= tel_an["mismatch"] * 1.05
    assert tel_lib["lambda_used"] <= tel_an["lambda_used"] + 1e-6
