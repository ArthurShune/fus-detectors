import numpy as np
import torch
from pipeline.gpu.linalg import cholesky_solve_hermitian
from pipeline.stap.temporal import (
    bandpass_constraints_temporal,
    build_temporal_hankels_and_cov,
    ka_prior_temporal_from_psd,
    ka_blend_covariance_temporal,
)


def ratio_map(R_loaded, S, Cf):
    Lt = R_loaded.shape[0]
    L = torch.linalg.cholesky(R_loaded)
    Sw = torch.linalg.solve_triangular(L, S.reshape(Lt, -1), upper=False)
    Cw = torch.linalg.solve_triangular(L, Cf, upper=False)
    z = Cw.conj().transpose(-2, -1) @ Sw
    gram = Cw.conj().transpose(-2, -1) @ Cw + 0.20 * torch.eye(Cw.shape[-1], dtype=R_loaded.dtype)
    proj, _ = cholesky_solve_hermitian(gram, z, jitter_init=1e-10, max_tries=3)
    t_band = torch.sum(z.conj() * proj, dim=0).real
    sw_pow = torch.sum(Sw.conj() * Sw, dim=0).real
    return (t_band / torch.clamp(sw_pow - t_band, min=1e-10)).reshape(-1)


torch.manual_seed(0)
np.random.seed(0)
T, h, w, Lt = 48, 20, 20, 4
prf = 3000.0
t = torch.arange(T, dtype=torch.float32) / prf
noise = 0.1 * (
    torch.randn(T, h, w, dtype=torch.float32) + 1j * torch.randn(T, h, w, dtype=torch.float32)
)
drift = 0.05 * torch.exp(1j * 2 * np.pi * 75.0 * t)[:, None, None]
cube = (noise + drift).to(torch.complex64)
y0, x0 = 7, 8
cube[:, y0, x0] += 0.12 * torch.exp(1j * 2 * np.pi * 600.0 * t)
S, Rt, _ = build_temporal_hankels_and_cov(
    cube, Lt=Lt, center=True, estimator="huber", huber_c=5.0, device="cpu", dtype=torch.complex64
)
Cf = bandpass_constraints_temporal(
    Lt, prf, np.array([-600.0, -300.0, 0.0, 300.0, 600.0], dtype=np.float64), device="cpu"
)
R0 = ka_prior_temporal_from_psd(
    Lt=Lt, prf_hz=prf, f_peaks_hz=(0.0,), width_bins=1, add_deriv=True, device="cpu"
)
R_lam, info = ka_blend_covariance_temporal(
    R_sample=Rt, R0_prior=R0, Cf_flow=Cf, beta=None, kappa_target=40.0, device="cpu"
)
ratio_no = ratio_map(Rt + 0.02 * torch.eye(Lt, dtype=Rt.dtype), S, Cf)
ratio_ka = ratio_map(R_lam, S, Cf)
print("lambda_used", info.get("lambda_used"))
print("q99_no", float(torch.quantile(ratio_no, 0.99).item()))
print("q99_ka", float(torch.quantile(ratio_ka, 0.99).item()))
