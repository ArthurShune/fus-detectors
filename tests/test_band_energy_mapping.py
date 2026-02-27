import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_band_energy_whitened_batched_preserves_sample_mapping_cpu():
    """
    Regression guard:

    `_band_energy_whitened_batched` returns (B, N, h, w). For paper-baseline
    compatibility we preserve the legacy flattening layout used inside the
    batched band-energy kernel:

      S(B,Lt,N,h,w) -> permute(B,Lt,h,w,N) -> view(B,Lt,P) -> ... -> view(B,N,h,w)

    This implies a deterministic permutation between output indices (n,y,x) and
    the original per-snapshot samples (n_src,y_src,x_src). This test encodes that
    mapping so latency refactors cannot silently change manuscript outputs.
    """
    from pipeline.stap.temporal import _band_energy_whitened_batched

    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.complex64

    B, Lt, N, h, w, K = 2, 4, 5, 2, 3, 7

    A = (torch.randn(B, Lt, Lt) + 1j * torch.randn(B, Lt, Lt)).to(dtype=dtype)
    R = A @ A.conj().transpose(-2, -1)
    R = 0.5 * (R + R.conj().transpose(-2, -1))
    R = R + 0.1 * torch.eye(Lt, dtype=dtype).unsqueeze(0)

    S = (torch.randn(B, Lt, N, h, w) + 1j * torch.randn(B, Lt, N, h, w)).to(dtype=dtype)
    Ct = (torch.randn(Lt, K) + 1j * torch.randn(Lt, K)).to(dtype=dtype)
    lam = torch.full((B,), 0.07, dtype=torch.float32)

    T_band, sw_pow = _band_energy_whitened_batched(
        R,
        S,
        Ct,
        lam,
        ridge=0.10,
        device=device,
        dtype=dtype,
        eps=1e-10,
    )

    for b in range(B):
        R_loaded = R[b] + lam[b] * torch.eye(Lt, dtype=dtype)
        L = torch.linalg.cholesky(R_loaded)
        Cw = torch.linalg.solve_triangular(L, Ct, upper=False)
        Gram0 = Cw.conj().transpose(-2, -1) @ Cw
        evals = torch.linalg.eigvalsh(Gram0).real.clamp_min(1e-12)
        ridge_vec = 0.10 * (evals.max() if float(evals.max()) < 1.0 else 1.0)
        Gram = Gram0 + ridge_vec * torch.eye(K, dtype=dtype)
        Lg = torch.linalg.cholesky(Gram)

        for n in range(N):
            for yi in range(h):
                for xi in range(w):
                    # Legacy mapping between output (n,yi,xi) and source sample
                    # indices used in the flattened (h,w,N) order.
                    p = (n * h + yi) * w + xi
                    n_src = int(p % N)
                    tmp = int(p // N)
                    yi_src = int(tmp // w)
                    xi_src = int(tmp % w)
                    s_col = S[b, :, n_src, yi_src, xi_src]
                    sw_col = torch.linalg.solve_triangular(
                        L, s_col[:, None], upper=False
                    ).squeeze(-1)
                    z = Cw.conj().transpose(-2, -1) @ sw_col
                    tmp = torch.linalg.solve_triangular(
                        Lg, z[:, None], upper=False
                    ).squeeze(-1)
                    energy = torch.sum(tmp.conj() * tmp).real
                    pow_tot = torch.sum(sw_col.conj() * sw_col).real

                    torch.testing.assert_close(
                        T_band[b, n, yi, xi], energy.to(dtype=T_band.dtype), rtol=2e-4, atol=2e-6
                    )
                    torch.testing.assert_close(
                        sw_pow[b, n, yi, xi], pow_tot.to(dtype=sw_pow.dtype), rtol=2e-4, atol=2e-6
                    )
