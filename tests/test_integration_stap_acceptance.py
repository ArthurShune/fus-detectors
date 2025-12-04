import json
from pathlib import Path

import numpy as np

from eval.acceptance import AcceptanceTargets, DetectorDataset, Masks, evaluate_performance
from sim.kwave.common import (
    AngleData,
    SimGeom,
    write_acceptance_bundle,
)


def _subsample(path: str, n: int, seed: int = 0) -> str:
    arr = np.load(path, allow_pickle=False)
    rng = np.random.default_rng(seed)
    if arr.ndim == 1:
        idx = rng.choice(arr.size, size=min(n, arr.size), replace=arr.size < n)
        out = arr[idx]
    else:
        out = arr
    p = Path(path)
    out_path = p.with_name(f"sub_{p.name}")
    np.save(out_path, out.astype(np.float32), allow_pickle=False)
    return str(out_path)


def test_end_to_end_acceptance_on_bundle(tmp_path):
    # 1) Create a small synthetic bundle (fast path)
    g = SimGeom(Nx=32, Ny=32, dx=90e-6, dy=90e-6, c0=1540.0, rho0=1000.0, ncycles=1, f0=7.0e6)
    prf = 3000.0
    rng = np.random.default_rng(0)
    Nt = 96
    dt = g.cfl * min(g.dx, g.dy) / g.c0
    angles = [0.0, 10.0]
    angle_sets = [
        [
            AngleData(
                angle_deg=ang, rf=rng.standard_normal((Nt, g.Nx)).astype(np.float32), dt=float(dt)
            )
            for ang in angles
        ]
    ]

    paths = write_acceptance_bundle(
        out_root=Path(tmp_path),
        g=g,
        angle_sets=angle_sets,
        pulses_per_set=4,
        prf_hz=prf,
        seed=123,
        tile_hw=(8, 8),
        tile_stride=4,
        Lt=3,
        diag_load=1e-2,
        cov_estimator="scm",
        huber_c=5.0,
        mvdr_load_mode="absolute",
        mvdr_auto_kappa=40.0,
        constraint_ridge=0.12,
        fd_span_mode="psd",
        fd_span_rel=(0.30, 1.10),
        grid_step_rel=0.08,
        fd_min_pts=5,
        fd_max_pts=7,
        msd_lambda=2e-2,
        msd_ridge=0.10,
        msd_agg_mode="trim10",
        stap_debug_samples=0,
        stap_device="cpu",
        dataset_name="bundle_small",
        meta_extra={"source": "integration_test"},
    )

    # 2) Subsample to keep acceptance quick and resolvable at 1e-4
    base_pos = _subsample(paths["base_pos"], n=4000, seed=1)
    base_neg = _subsample(paths["base_neg"], n=10000, seed=2)
    stap_pos = _subsample(paths["stap_pos"], n=4000, seed=3)
    # Use baseline nulls to ensure enough exceedances for POT calibration in this fast test
    stap_neg = base_neg
    s1 = _subsample(paths["confirm2_scores1"], n=10000, seed=5)
    s2 = _subsample(paths["confirm2_scores2"], n=10000, seed=6)

    # 3) Evaluate performance integration (fast path; skip calibration)
    base = DetectorDataset(
        scores_pos=np.load(base_pos),
        scores_null=np.load(base_neg),
        pd_map=np.load(paths["pd_base"]),
    )
    stap = DetectorDataset(
        scores_pos=np.load(stap_pos),
        scores_null=np.load(stap_neg),
        pd_map=np.load(paths["pd_stap"]),
    )
    masks = Masks(
        mask_flow=np.load(paths["mask_flow"]).astype(bool),
        mask_bg=np.load(paths["mask_bg"]).astype(bool),
    )
    targets = AcceptanceTargets(fpr_target=1e-4, delta_pdsnrdB_min=0.0, delta_tpr_at_fpr_min=0.0)
    perf = evaluate_performance(base, stap, masks, targets)
    # Values should be finite and within [0,1] where applicable
    assert 0.0 <= perf["tpr_at_fpr_baseline"] <= 1.0
    assert 0.0 <= perf["tpr_at_fpr_stap"] <= 1.0
    assert perf["pauc_baseline"] >= 0.0 and perf["pauc_stap"] >= 0.0
    # PD-SNR should be finite given masks and maps
    assert np.isfinite(perf["pd_snr_baseline_db"]) and np.isfinite(perf["pd_snr_stap_db"])
