import numpy as np
import torch

from pipeline.stap.temporal import bandpass_constraints_temporal


def _rand_spd(Lt: int, scale: float = 1e-3, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((Lt, Lt)) + 1j * rng.standard_normal((Lt, Lt))
    A = A.astype(np.complex64)
    At = torch.as_tensor(A)
    R = At @ At.conj().transpose(-2, -1)
    R = R + scale * torch.eye(Lt, dtype=R.dtype)
    return R


def _joint_lcmv_weights(
    R: torch.Tensor,
    Cf: torch.Tensor,
    Cm: torch.Tensor,
    *,
    lam_abs: float = 2e-2,
    max_cond: float = 1e8,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute joint LCMV weights enforcing Cf^H w = 1 and (I-Pf)Cm nulls.
    Reduces motion columns to fit Lt - rank(Cf) if necessary and conditions Gram.
    """
    R = R.to(device)
    Cf = Cf.to(device)
    Cm = Cm.to(device)
    Lt = R.shape[0]
    # Condition R
    eye = torch.eye(Lt, dtype=R.dtype, device=device)
    R_loaded = R + lam_abs * eye
    # Flow projector Pf
    F = Cf
    Gf = F.conj().transpose(-2, -1) @ F
    Gf = Gf + 1e-6 * torch.eye(Gf.shape[-1], dtype=Gf.dtype, device=device)
    F_pinv = torch.linalg.solve(Gf, F.conj().transpose(-2, -1))
    Pf = F @ F_pinv
    Pperp = eye - Pf
    M_eff = Pperp @ Cm
    # Orthonormalize motion columns and trim rank
    if M_eff.numel() == 0:
        K_m_eff = 0
        Qm = M_eff
    else:
        # QR with column pivoting via SVD for robustness
        U, S, Vh = torch.linalg.svd(M_eff, full_matrices=False)
        tol = float(torch.max(S).item()) * 1e-5
        r = int((S > tol).sum().item())
        Qm = U[:, :r]
        K_m_eff = int(Qm.shape[1])
    # Limit to available DOF
    r_f = int(torch.linalg.matrix_rank(Cf).item())
    max_m = max(Lt - r_f, 0)
    if K_m_eff > max_m:
        Qm = Qm[:, :max_m]
        K_m_eff = max_m

    # Assemble constraints
    if K_m_eff > 0:
        C = torch.cat([Cf, Qm], dim=1)
        g = torch.cat(
            [
                torch.ones((Cf.shape[1],), dtype=R.dtype, device=device),
                torch.zeros((K_m_eff,), dtype=R.dtype, device=device),
            ]
        )
    else:
        C = Cf
        g = torch.ones((Cf.shape[1],), dtype=R.dtype, device=device)

    # Solve w = R^{-1} C (C^H R^{-1} C)^{-1} g
    RiC = torch.linalg.solve(R_loaded, C)
    Gram = C.conj().transpose(-2, -1) @ RiC
    # Condition Gram if needed
    evals = torch.linalg.eigvalsh(Gram).real
    cond = float((evals.max().item()) / max(evals.min().item(), 1e-12))
    if not np.isfinite(cond) or cond > max_cond:
        Gram = Gram + (1e-3 * torch.trace(Gram).real / Gram.shape[-1]) * torch.eye(
            Gram.shape[-1], dtype=Gram.dtype, device=device
        )
    tmp = torch.linalg.solve(Gram, g)
    w = RiC @ tmp
    # Enforce exact equality for Cf block: Cf^H (w - Cf*delta) = 1
    c = Cf.conj().transpose(-2, -1) @ w
    Aeq = Cf.conj().transpose(-2, -1) @ Cf + 1e-6 * torch.eye(
        Cf.shape[1], dtype=R.dtype, device=device
    )
    delta = torch.linalg.solve(Aeq, c - torch.ones_like(c))
    w = w - Cf @ delta
    return w


def test_joint_lcmv_preserves_flow_and_nulls_motion():
    torch.manual_seed(0)
    Lt = 8
    prf = 3000.0
    R = _rand_spd(Lt, scale=1e-3, seed=1)
    # Flow: ±600 Hz (keep small K to leave DOF)
    fd_flow = np.array([-600.0, 600.0], dtype=np.float32)
    Cf = bandpass_constraints_temporal(
        Lt, prf_hz=prf, fd_grid_hz=fd_flow, mode="exp", device="cpu"
    )
    # Motion: near-DC ±150 Hz
    fd_motion = np.array([-150.0, 0.0, 150.0], dtype=np.float32)
    Cm = bandpass_constraints_temporal(
        Lt, prf_hz=prf, fd_grid_hz=fd_motion, mode="exp", device="cpu"
    )

    w = _joint_lcmv_weights(R, Cf, Cm, lam_abs=0.03, device="cpu")

    # Equality on flow: Cf^H w == 1 (per column)
    eq = (Cf.conj().transpose(-2, -1) @ w).detach().cpu().numpy()
    assert np.allclose(eq, np.ones_like(eq), atol=1e-2, rtol=1e-2)

    # Null on motion in the orthogonal complement of flow
    eye = torch.eye(Lt, dtype=R.dtype)
    Gf = Cf.conj().transpose(-2, -1) @ Cf
    Gf = Gf + 1e-6 * torch.eye(Gf.shape[-1], dtype=Gf.dtype)
    Pf = Cf @ torch.linalg.solve(Gf, Cf.conj().transpose(-2, -1))
    Pperp = eye - Pf
    Cm_eff = Pperp @ Cm
    nul = (Cm_eff.conj().transpose(-2, -1) @ w).detach().cpu().numpy()
    assert np.allclose(nul, 0.0, atol=5e-2, rtol=1e-2)

    # Compare against a flow-only LCMV (no motion projector) to prove the new
    # weights actually attenuate the motion subspace instead of just matching pf.
    Cm_empty = torch.zeros((Lt, 0), dtype=R.dtype)
    w_flow_only = _joint_lcmv_weights(R, Cf, Cm_empty, lam_abs=0.03, device="cpu")
    nul_flow = (Cm_eff.conj().transpose(-2, -1) @ w_flow_only).detach().cpu().numpy()
    assert np.linalg.norm(nul) < np.linalg.norm(nul_flow)


def test_joint_lcmv_reduces_motion_rank_when_dof_insufficient():
    torch.manual_seed(1)
    Lt = 6
    prf = 3000.0
    R = _rand_spd(Lt, scale=1e-3, seed=3)
    # Flow: 3 tones -> rank 3
    Cf = bandpass_constraints_temporal(
        Lt, prf_hz=prf, fd_grid_hz=[-600.0, 0.0, 600.0], mode="exp", device="cpu"
    )
    # Motion: too many tones -> force reduction
    Cm = bandpass_constraints_temporal(
        Lt, prf_hz=prf, fd_grid_hz=[-300.0, -150.0, 0.0, 150.0, 300.0], mode="exp", device="cpu"
    )

    w = _joint_lcmv_weights(R, Cf, Cm, lam_abs=0.05, device="cpu")

    # Flow equalities still hold
    eq = (Cf.conj().transpose(-2, -1) @ w).detach().cpu().numpy()
    assert np.allclose(eq, np.ones_like(eq), atol=2e-2, rtol=2e-2)

    # Motion residual should be reduced compared to raw Cm null check
    eye = torch.eye(Lt, dtype=R.dtype)
    Gf = Cf.conj().transpose(-2, -1) @ Cf
    Gf = Gf + 1e-6 * torch.eye(Gf.shape[-1], dtype=Gf.dtype)
    Pf = Cf @ torch.linalg.solve(Gf, Cf.conj().transpose(-2, -1))
    Pperp = eye - Pf
    Cm_eff = Pperp @ Cm
    nul = (Cm_eff.conj().transpose(-2, -1) @ w).detach().cpu().numpy()
    # Tolerance is looser because rank was truncated
    assert np.all(np.linalg.norm(nul, axis=0) < 0.15)

    # The truncated solution should still suppress motion energy relative to the
    # unconstrained flow-only design.
    Cm_empty = torch.zeros((Lt, 0), dtype=R.dtype)
    w_flow_only = _joint_lcmv_weights(R, Cf, Cm_empty, lam_abs=0.05, device="cpu")
    nul_flow = (Cm_eff.conj().transpose(-2, -1) @ w_flow_only).detach().cpu().numpy()
    if np.linalg.norm(nul_flow) > 1e-6:
        assert np.linalg.norm(nul) < np.linalg.norm(nul_flow)
