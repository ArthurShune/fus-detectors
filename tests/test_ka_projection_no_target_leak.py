import numpy as np
import torch

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    ka_prior_temporal_from_psd,
    _project_out_flow_from_prior,
)


def test_flow_projection_removes_target_energy_from_prior() -> None:
    Lt = 6
    prf = 3000.0
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=prf,
        f_peaks_hz=(0.0,),
        width_bins=2,
        add_deriv=True,
        device="cpu",
    )
    fd = np.linspace(-600.0, 600.0, 5)
    Cf = bandpass_constraints_temporal(Lt, prf, fd, device="cpu")
    energy_raw = torch.linalg.norm(Cf.conj().T @ R0 @ Cf, ord="fro").item()
    R_proj = _project_out_flow_from_prior(R0, Cf)
    energy_proj = torch.linalg.norm(Cf.conj().T @ R_proj @ Cf, ord="fro").item()
    assert energy_proj <= energy_raw * 0.2 + 1e-9
