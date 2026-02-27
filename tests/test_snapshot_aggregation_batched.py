import pytest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


@pytest.mark.skipif(torch is None, reason="torch not available")
@pytest.mark.parametrize("mode", ["mean", "median", "trim10"])
def test_aggregate_over_snapshots_batched_matches_loop(mode: str) -> None:
    from pipeline.stap.temporal import aggregate_over_snapshots, aggregate_over_snapshots_batched

    torch.manual_seed(0)
    B, N, h, w = 7, 23, 5, 6
    x = torch.randn((B, N, h, w), dtype=torch.float32)

    ref = torch.stack([aggregate_over_snapshots(x[b], mode=mode) for b in range(B)], dim=0)
    out = aggregate_over_snapshots_batched(x, mode=mode)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)

