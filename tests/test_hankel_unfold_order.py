import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_build_temporal_hankels_batch_matches_explicit_stack_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    from pipeline.stap.temporal_shared import build_temporal_hankels_batch

    monkeypatch.delenv("STAP_SNAPSHOT_STRIDE", raising=False)
    monkeypatch.delenv("STAP_MAX_SNAPSHOTS", raising=False)

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


@pytest.mark.skipif(torch is None, reason="torch not available")
def test_build_temporal_hankels_batch_snapshot_stride_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Snapshot subsampling should never collapse very short Hankel supports down to N=1.

    Example: T=17, Lt=16 -> N=2 (Twinkling/Gammex regime). Even if a large
    STAP_SNAPSHOT_STRIDE is set, we keep N=2 to avoid destabilizing cov estimates.
    """
    from pipeline.stap.temporal_shared import build_temporal_hankels_batch

    monkeypatch.setenv("STAP_SNAPSHOT_STRIDE", "6")
    monkeypatch.delenv("STAP_MAX_SNAPSHOTS", raising=False)

    cube = torch.randn(1, 17, 2, 2, dtype=torch.complex64)
    S, _ = build_temporal_hankels_batch(cube, 16, center=False, device="cpu", dtype=torch.complex64)
    assert int(S.shape[2]) == 2
