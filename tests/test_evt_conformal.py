"""Tests for EVT and conformal stubs."""

import numpy as np

from pipeline.calib.conformal import conformal_threshold_from_scores, empirical_pfa
from pipeline.calib.evt_pot import fit_gpd


def test_evt_stub() -> None:
    rng = np.random.default_rng(0)
    samples = rng.weibull(2.0, size=2000)
    thresh = np.quantile(samples, 0.95)
    exceed = samples[samples > thresh] - thresh
    shape, scale = fit_gpd(exceed)
    assert scale > 0.0
    assert -1.0 < shape < 1.0


def test_conformal_coverage_relaxed():
    rng = np.random.default_rng(1)
    # Null scores ~ Exp(1) => heavy-ish tail; independent samples
    s_calib = rng.exponential(scale=1.0, size=50_000)
    thr = conformal_threshold_from_scores(
        s_calib, alpha1=1e-2, q0=0.95, min_exceedances=300, split_ratio=0.6, seed=42
    )
    s_holdout = rng.exponential(scale=1.0, size=20_000)
    p_emp = empirical_pfa(s_holdout, thr)
    # With finite-sample conformal, miscoverage should be near alpha1 up to randomness.
    # Use loose bounds to keep test stable and fast.
    assert 0.004 <= p_emp <= 0.02
