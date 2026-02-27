import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_build_temporal_hankels_batch_matches_explicit_stack_cpu() -> None:
    from pipeline.stap.temporal_shared import build_temporal_hankels_batch

    torch.manual_seed(0)
    B, T, h, w = 2, 7, 3, 4
    Lt = 4
    N = T - Lt + 1
    # Use a deterministic, unique-valued cube so any axis mixup is obvious.
    base = torch.arange(B * T * h * w, dtype=torch.float32).reshape(B, T, h, w)
    cube = (base + 1j * (base + 0.25)).to(dtype=torch.complex64)

    S, R = build_temporal_hankels_batch(cube, Lt, center=False, device="cpu", dtype=torch.complex64)
    assert tuple(int(x) for x in S.shape) == (B, Lt, N, h, w)
    assert tuple(int(x) for x in R.shape) == (B, Lt, Lt)

    rows = [cube[:, k : k + N] for k in range(Lt)]
    S_ref = torch.stack(rows, dim=1).contiguous()
    torch.testing.assert_close(S, S_ref, rtol=0, atol=0)

