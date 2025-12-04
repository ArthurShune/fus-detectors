import torch

from pipeline.stap.temporal import ka_prior_temporal_from_psd


def test_psd_prior_is_hermitian_psd_and_trace_normalized() -> None:
    Lt = 6
    R0 = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=3000.0,
        f_peaks_hz=(0.0,),
        width_bins=2,
        add_deriv=True,
        device="cpu",
    )
    hermitian_residual = (R0 - R0.conj().T).abs().max().item()
    assert hermitian_residual < 1e-5
    herm = (R0 + R0.conj().T) * 0.5
    ev = torch.linalg.eigvalsh(herm).real
    assert torch.all(ev >= -1e-7), f"Negative eigenvalue min={ev.min().item():.3e}"
    trace_val = torch.real(torch.trace(R0)).item()
    assert abs(trace_val - Lt) < 1e-3


def test_derivative_column_increases_rank() -> None:
    Lt = 6
    R0_no = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=3000.0,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=False,
        device="cpu",
    )
    R0_yes = ka_prior_temporal_from_psd(
        Lt=Lt,
        prf_hz=3000.0,
        f_peaks_hz=(0.0,),
        width_bins=1,
        add_deriv=True,
        device="cpu",
    )
    herm_no = (R0_no + R0_no.conj().T) * 0.5
    herm_yes = (R0_yes + R0_yes.conj().T) * 0.5
    ev_no = torch.linalg.eigvalsh(herm_no).real
    ev_yes = torch.linalg.eigvalsh(herm_yes).real
    rank_no = int((ev_no > 1e-6).sum().item())
    rank_yes = int((ev_yes > 1e-6).sum().item())
    assert rank_yes >= rank_no
