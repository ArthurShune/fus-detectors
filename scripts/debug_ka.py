import numpy as np
import torch

from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    projector_from_tones,
    ka_blend_covariance_temporal,
)


def _rand_psd(Lt: int, scale: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Lt, Lt)) + 1j * rng.standard_normal((Lt, Lt))
    H = (A + A.conj().T) * 0.5
    H += (scale * Lt) * np.eye(Lt, dtype=np.complex64)
    return torch.as_tensor(H.astype(np.complex64))


def main() -> None:
    Lt = 6
    prf = 3000.0
    Cf = bandpass_constraints_temporal(
        Lt=Lt, prf_hz=prf, fd_grid_hz=np.array([-450.0, 0.0, 450.0]), device="cpu"
    )
    Pf = projector_from_tones(Cf.to(torch.complex128)).to(torch.complex64)
    eye = torch.eye(Lt, dtype=torch.complex64)
    Pperp = eye - Pf

    R_hat = _rand_psd(Lt, 1e-3, 2)
    R0 = 0.9 * R_hat + 0.1 * torch.eye(Lt, dtype=torch.complex64)

    R_loaded, info = ka_blend_covariance_temporal(
        R_sample=R_hat,
        R0_prior=R0,
        Cf_flow=Cf.detach().cpu().numpy(),
        alpha=0.0,
        beta=None,
        beta_bounds=(0.0, 0.3),
        mismatch_tau=1.0,
        kappa_target=40.0,
        add_noise_floor_frac=0.0,
        lambda_override=None,
        beta_directional=True,
        target_retain_f=0.95,
        target_shrink_perp=0.90,
        ridge_split=True,
        lambda_override_split=0.01,
        device="cpu",
        dtype=torch.complex64,
    )

    R_loaded = 0.5 * (R_loaded + R_loaded.conj().transpose(-2, -1))

    lam_used = float(info.get("lambda_used", 0.01))
    R_base = 0.5 * (R_hat + lam_used * Pperp + (R_hat + lam_used * Pperp).conj().transpose(-2, -1))

    tr_f_base = torch.real(torch.trace(Pf @ R_base @ Pf.conj().transpose(-2, -1))).item()
    tr_p_base = torch.real(torch.trace(Pperp @ R_base @ Pperp.conj().transpose(-2, -1))).item()
    tr_f_loaded = torch.real(torch.trace(Pf @ R_loaded @ Pf.conj().transpose(-2, -1))).item()
    tr_p_loaded = torch.real(torch.trace(Pperp @ R_loaded @ Pperp.conj().transpose(-2, -1))).item()
    print("lambda_used", lam_used)
    print(
        "beta",
        info.get("beta"),
        "beta_dir",
        info.get("beta_directional"),
        "retain_f_beta",
        info.get("retain_f_beta"),
        "shrink_perp_beta",
        info.get("shrink_perp_beta"),
    )
    print(
        "retain_f_total",
        info.get("retain_f_total"),
        "shrink_perp_total",
        info.get("shrink_perp_total"),
    )
    print("tr_f_base", tr_f_base, "tr_f_loaded", tr_f_loaded)
    print("tr_p_base", tr_p_base, "tr_p_loaded", tr_p_loaded)


if __name__ == "__main__":
    main()
