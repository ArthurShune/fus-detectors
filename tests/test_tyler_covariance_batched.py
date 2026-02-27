import torch


def _rand_complex(shape, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    rr = torch.randn(shape, device=device, dtype=torch.float32)
    ri = torch.randn(shape, device=device, dtype=torch.float32)
    return (rr + 1j * ri).to(dtype=dtype)


def test_tyler_covariance_batched_matches_single_iteration(monkeypatch) -> None:
    from pipeline.stap.temporal import _tyler_covariance_batched

    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.complex64

    B, Lt, P = 3, 6, 32
    X = _rand_complex((B, Lt, P), device=device, dtype=dtype)

    monkeypatch.setenv("STAP_TYLER_MAX_ITER", "1")
    monkeypatch.setenv("STAP_TYLER_EARLY_STOP", "0")
    R1 = _tyler_covariance_batched(X, max_iter=25, tol=1e-6, eps=1e-10)

    # Manual first iteration from R0=I.
    q = torch.sum(X.conj() * X, dim=1).real.clamp_min(1e-10)  # (B,P)
    w = (float(Lt) / q).clamp_max(1e6)
    Xw = X * w.unsqueeze(1)
    R_manual = torch.matmul(Xw, Xw.conj().transpose(-2, -1)) / float(P)
    tr = torch.real(torch.diagonal(R_manual, dim1=1, dim2=2).sum(dim=1)).clamp_min(1e-10)
    R_manual = R_manual * (float(Lt) / tr).view(B, 1, 1)

    torch.testing.assert_close(R1, R_manual, rtol=1e-4, atol=1e-6)
