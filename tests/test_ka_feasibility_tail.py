import math

import numpy as np
import torch

from pipeline.stap.temporal import (
    aggregate_over_snapshots,
    bandpass_constraints_temporal,
    ka_blend_covariance_temporal,
    msd_snapshot_energies_batched,
)


def _student_t_complex(
    n_samples: int, cov: np.ndarray, nu: float, *, rng: np.random.Generator
) -> np.ndarray:
    """Generate complex Student-t samples with covariance `cov`."""
    Lt = cov.shape[0]
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(Lt))
    real = L @ rng.standard_normal((Lt, n_samples))
    imag = L @ rng.standard_normal((Lt, n_samples))
    chi = rng.chisquare(df=nu, size=n_samples)
    scale = np.sqrt(chi / nu)
    samples = (real + 1j * imag) / scale
    return samples.astype(np.complex64)


def _build_scenario():
    Lt = 6
    prf = 3000.0
    nu = 3.0  # heavy-tailed
    n_train = 32
    n_neg = 80_000
    n_pos = 600
    flow_amp = 0.08
    flow_freq = 700.0
    ridge = 0.10
    ratio_rho = 0.12
    lam_floor = 1.5e-2
    beta = 0.28

    rng = np.random.default_rng(31415)
    base_cov = np.geomspace(1.0, 0.08, Lt)
    Q, _ = np.linalg.qr(rng.standard_normal((Lt, Lt)))
    R_true = (Q * base_cov) @ Q.T + 1.0e-6 * np.eye(Lt)

    drift_vec = rng.standard_normal(Lt)
    drift_vec /= np.linalg.norm(drift_vec)
    R_drift = 0.08 * np.outer(drift_vec, drift_vec)
    R_train = R_true + R_drift
    R_test = R_true - R_drift

    S_train = _student_t_complex(n_train, R_train, nu, rng=rng)
    R_hat = (S_train @ S_train.conj().T).real / n_train

    S_lib = _student_t_complex(4000, R_true, nu, rng=rng)
    R_lib = (S_lib @ S_lib.conj().T).real / S_lib.shape[1]

    S_neg = _student_t_complex(n_neg, R_test, nu, rng=rng)
    S_pos_bg = _student_t_complex(n_pos, R_test, nu, rng=rng)
    t = np.arange(Lt, dtype=np.float64)
    tone = np.exp(1j * 2.0 * np.pi * flow_freq * t / prf).astype(np.complex64)
    S_pos = (S_pos_bg + flow_amp * tone[:, None]).astype(np.complex64)

    Cf = bandpass_constraints_temporal(
        Lt,
        prf,
        [280.0, 460.0, flow_freq, 900.0, 1100.0],
        device="cpu",
    )

    return {
        "Lt": Lt,
        "prf": prf,
        "R_hat": torch.as_tensor(R_hat, dtype=torch.complex64),
        "R_lib": torch.as_tensor(R_lib, dtype=torch.complex64),
        "neg": torch.as_tensor(S_neg, dtype=torch.complex64).reshape(Lt, -1, 1, 1),
        "pos": torch.as_tensor(S_pos, dtype=torch.complex64).reshape(Lt, -1, 1, 1),
        "Cf": Cf,
        "ridge": ridge,
        "ratio_rho": ratio_rho,
        "lam_floor": lam_floor,
        "beta": beta,
    }


def _aggregate_ratio(R_t, samples, Cf, *, lam_abs, ridge, ratio_rho, ka_kwargs=None):
    T_band, sw_pow = msd_snapshot_energies_batched(
        R_t,
        samples,
        Cf,
        lam_abs=lam_abs,
        ridge=ridge,
        ratio_rho=ratio_rho,
        R0_prior=ka_kwargs.get("R0_prior") if ka_kwargs else None,
        Cf_flow=ka_kwargs.get("Cf_flow") if ka_kwargs else None,
        ka_opts=ka_kwargs.get("ka_opts") if ka_kwargs else None,
        device="cpu",
    )
    ratio = T_band / torch.clamp(sw_pow - T_band + ratio_rho * sw_pow, min=1e-10)
    return aggregate_over_snapshots(ratio, mode="mean").reshape(-1)


def test_ka_library_tightens_tail_and_improves_tpr():
    data = _build_scenario()
    R_hat = data["R_hat"]
    R_lib = data["R_lib"]
    Cf = data["Cf"]
    lam_floor = data["lam_floor"]
    ridge = data["ridge"]
    ratio_rho = data["ratio_rho"]
    beta = data["beta"]

    neg_base = _aggregate_ratio(
        R_hat + lam_floor * torch.eye(R_hat.shape[0], dtype=R_hat.dtype),
        data["neg"],
        Cf,
        lam_abs=0.0,
        ridge=ridge,
        ratio_rho=ratio_rho,
    )
    pos_base = _aggregate_ratio(
        R_hat + lam_floor * torch.eye(R_hat.shape[0], dtype=R_hat.dtype),
        data["pos"],
        Cf,
        lam_abs=0.0,
        ridge=ridge,
        ratio_rho=ratio_rho,
    )

    R_blend, details = ka_blend_covariance_temporal(
        R_sample=R_hat,
        R0_prior=R_lib,
        Cf_flow=None,
        beta=beta,
        lambda_override=lam_floor,
        kappa_target=40.0,
        device="cpu",
    )

    neg_ka = _aggregate_ratio(
        R_blend,
        data["neg"],
        Cf,
        lam_abs=0.0,
        ridge=ridge,
        ratio_rho=ratio_rho,
        ka_kwargs={
            "R0_prior": R_lib,
            "Cf_flow": None,
            "ka_opts": {"beta": beta, "kappa_target": 40.0, "lambda_override": lam_floor},
        },
    )
    pos_ka = _aggregate_ratio(
        R_blend,
        data["pos"],
        Cf,
        lam_abs=0.0,
        ridge=ridge,
        ratio_rho=ratio_rho,
        ka_kwargs={
            "R0_prior": R_lib,
            "Cf_flow": None,
            "ka_opts": {"beta": beta, "kappa_target": 40.0, "lambda_override": lam_floor},
        },
    )

    q999_base = torch.quantile(neg_base, 0.999).item()
    q999_ka = torch.quantile(neg_ka, 0.999).item()
    tail_drop = q999_base - q999_ka
    assert (
        tail_drop >= 0.015
    ), f"KA should tighten null tail, but drop was {tail_drop:.4f} (base={q999_base:.3f}, ka={q999_ka:.3f})"

    thr1e3_base = torch.quantile(neg_base, 1 - 1e-3).item()
    thr1e3_ka = torch.quantile(neg_ka, 1 - 1e-3).item()
    tpr1e3_base = float((pos_base >= thr1e3_base).float().mean().item())
    tpr1e3_ka = float((pos_ka >= thr1e3_ka).float().mean().item())
    assert (
        tpr1e3_ka >= tpr1e3_base - 1e-3
    ), f"KA should not regress tail TPR at 1e-3: base={tpr1e3_base:.4f}, ka={tpr1e3_ka:.4f}"

    thr1e4_base = torch.quantile(neg_base, 1 - 1e-4).item()
    thr1e4_ka = torch.quantile(neg_ka, 1 - 1e-4).item()
    thr_drop = thr1e4_base - thr1e4_ka
    assert (
        thr_drop >= 0.01
    ), f"KA should reduce 1e-4 threshold: drop={thr_drop:.4f}, base={thr1e4_base:.4f}, ka={thr1e4_ka:.4f}"
    tpr1e4_base = float((pos_base >= thr1e4_base).float().mean().item())
    tpr1e4_ka = float((pos_ka >= thr1e4_ka).float().mean().item())
    assert (
        tpr1e4_ka >= tpr1e4_base - 1e-3
    ), f"KA should not regress tail TPR at 1e-4: base={tpr1e4_base:.4f}, ka={tpr1e4_ka:.4f}"

    # Flow retention: median score should not collapse
    flow_ratio = torch.median(pos_ka).item() / max(torch.median(pos_base).item(), 1e-12)
    assert flow_ratio >= 0.8, f"Flow median dropped too much under KA (ratio={flow_ratio:.3f})"

    # Diagnostic sanity: condition number reduced
    cond_base = torch.linalg.cond(
        R_hat + lam_floor * torch.eye(R_hat.shape[0], dtype=R_hat.dtype)
    ).item()
    cond_ka = torch.linalg.cond(R_blend).item()
    assert cond_ka <= cond_base + 1e-6, "KA whitening should not worsen conditioning"
