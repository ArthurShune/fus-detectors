#!/usr/bin/env python3
"""
Replay the STAP/acceptance stage using pre-generated angle data from an existing pilot run.

Usage:
    PYTHONPATH=. conda run -n stap-fus python scripts/replay_stap_from_run.py \
        --src runs/pilot/r1_real_psd_bg_guard095_inspect \
        --out runs/pilot/r1_real_psd_bg_guard095_coords \
        --stap-debug-coord 126,120 --stap-debug-coord 132,108 ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from sim.kwave.common import AngleData, SimGeom, write_acceptance_bundle


def _maybe_resize_roi_mask(mask: np.ndarray, geom: SimGeom) -> np.ndarray:
    """
    Resize a 2D ROI mask to match the simulation geometry (Ny, Nx).

    For now we support only the common MaceBridge case where the mask
    height differs from Ny by at most one pixel (due to a small upward
    adjustment of Ny for k-Wave prime-factor friendliness). In that case
    we pad or crop along the depth dimension and leave the lateral size
    unchanged. Any other mismatch is treated as an error.
    """
    mask = np.asarray(mask, dtype=bool)
    H_roi, W_roi = mask.shape
    H_geom, W_geom = int(geom.Ny), int(geom.Nx)

    if H_roi == H_geom and W_roi == W_geom:
        return mask

    if W_roi != W_geom:
        raise ValueError(
            f"ROI mask width mismatch: {W_roi} vs geom.Nx={W_geom}; "
            "cannot safely align to simulation grid."
        )

    # Allow a single-pixel height adjustment (typical for MaceBridge where
    # Ny may be incremented to avoid large prime factors); treat the extra
    # row as background and pad at the far depth edge.
    if H_roi == H_geom - 1:
        resized = np.zeros((H_geom, W_geom), dtype=bool)
        resized[:H_roi, :] = mask
        return resized
    if H_roi == H_geom + 1:
        # Rare case: ROI one row taller than the PD grid; drop the last row.
        return mask[:H_geom, :]

    raise ValueError(
        f"ROI mask height mismatch: {H_roi} vs geom.Ny={H_geom}; "
        "only ±1 pixel differences are supported."
    )


def _load_angle_data(src_root: Path, angles: Sequence[float]) -> List[List[AngleData]]:
    """Load one or more ensembles of angle data from the source directory."""

    def _load_dir(path: Path, ang: float) -> AngleData:
        if not path.exists():
            raise FileNotFoundError(f"Expected {path} with rf.npy/dt.npy")
        rf = np.load(path / "rf.npy")
        dt = float(np.load(path / "dt.npy"))
        return AngleData(angle_deg=float(ang), rf=rf, dt=dt)

    # First try the simple layout angle_{deg}
    angle_data: List[AngleData] = []
    simple_ok = True
    for ang in angles:
        name = f"angle_{int(round(ang))}"
        d = src_root / name
        if not d.exists():
            simple_ok = False
            break
        angle_data.append(_load_dir(d, ang))
    if simple_ok:
        return [angle_data]

    # Otherwise look for ensemble-prefixed directories (e.g., ens0_angle_-6)
    ensemble_dirs = sorted(
        {
            p.name.split("_")[0]
            for p in src_root.iterdir()
            if p.is_dir() and p.name.startswith("ens")
        }
    )
    if not ensemble_dirs:
        missing = ", ".join(f"angle_{int(round(a))}" for a in angles)
        raise FileNotFoundError(
            f"Expected angle directories {missing} under {src_root}, none found."
        )

    angle_sets: List[List[AngleData]] = []
    for ens in ensemble_dirs:
        ens_set: List[AngleData] = []
        for ang in angles:
            name = f"{ens}_angle_{int(round(ang))}"
            d = src_root / name
            ens_set.append(_load_dir(d, ang))
        angle_sets.append(ens_set)
    return angle_sets


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay STAP on existing angle data.")
    ap.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Source run directory containing angle_* dirs",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination run directory",
    )
    ap.add_argument(
        "--stap-profile",
        type=str,
        default="lab",
        choices=["lab", "clinical"],
        help=(
            "Preset STAP configuration. 'lab' uses raw CLI defaults; "
            "'clinical' applies a fixed, conservative configuration intended "
            "to mimic a deployable intra-op fUS setting."
        ),
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
        help=(
            "High-level operating profile matching the methodology (Brain-*). "
            "When set, overrides a small set of baseline/STAP/mask defaults "
            "to match the corresponding brain fUS profile."
        ),
    )
    ap.add_argument(
        "--stap-debug-coord",
        action="append",
        default=[],
        help="Tile coordinate y,x (repeatable)",
    )
    ap.add_argument("--stap-debug-samples", type=int, default=32)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument(
        "--stap-detector-variant",
        type=str,
        default="msd_ratio",
        choices=["msd_ratio", "whitened_power", "unwhitened_ratio"],
        help=(
            "Detector statistic exported as score_stap_preka.npy. "
            "'msd_ratio' (default) is the whitened matched-subspace ratio; "
            "'whitened_power' is total whitened slow-time power (no Doppler band partition); "
            "'unwhitened_ratio' disables covariance whitening (R=I) while keeping the same band partition."
        ),
    )
    ap.add_argument(
        "--cuda-warmup",
        dest="cuda_warmup",
        action="store_true",
        help="Warm up CUDA (cuBLAS/cuSOLVER/cuFFT) before timing-sensitive replay.",
    )
    ap.add_argument(
        "--no-cuda-warmup",
        dest="cuda_warmup",
        action="store_false",
        help="Disable CUDA warmup step.",
    )
    ap.set_defaults(cuda_warmup=True)
    ap.add_argument(
        "--baseline",
        type=str,
        default="svd",
        choices=["svd", "mc_svd", "svd_similarity", "local_svd", "rpca", "hosvd"],
        help=(
            "Baseline PD computation: plain SVD, motion-compensated SVD, adaptive SVD "
            "(spatial-singular similarity cutoff), block-wise local SVD, RPCA, or tensor HOSVD."
        ),
    )
    ap.add_argument(
        "--baseline-support",
        type=str,
        default="window",
        choices=["window", "full"],
        help=(
            "Slow-time support used to fit the baseline filter. 'window' fits the "
            "baseline on the requested time window (default). 'full' fits MC--SVD "
            "on the full slow-time stack and applies that projector to the requested "
            "window (other baselines fall back to 'window')."
        ),
    )
    ap.add_argument("--reg-enable", dest="reg_enable", action="store_true")
    ap.add_argument("--reg-disable", dest="reg_enable", action="store_false")
    ap.set_defaults(reg_enable=False)
    ap.add_argument("--reg-method", type=str, default="phasecorr", choices=["phasecorr"])
    ap.add_argument("--reg-subpixel", type=int, default=4, choices=[1, 2, 4])
    ap.add_argument("--reg-reference", type=str, default="median", choices=["first", "median"])
    svd_group = ap.add_mutually_exclusive_group()
    svd_group.add_argument("--svd-rank", type=int, default=None)
    svd_group.add_argument("--svd-energy-frac", type=float, default=None)
    ap.add_argument(
        "--svd-profile",
        type=str,
        default="default",
        choices=["default", "literature"],
        help=(
            "Preset for SVD-based baselines when --baseline=mc_svd. "
            "'default' uses explicit --svd-rank/--svd-energy-frac (or rank=3 "
            "when both are omitted); 'literature' applies a data-driven "
            "energy-fraction rule (95%% of slow-time energy) when no explicit "
            "SVD hyperparameters are provided, to better match common "
            "spatiotemporal SVD clutter filters in the fUS literature."
        ),
    )
    ap.add_argument("--rpca-enable", action="store_true")
    ap.add_argument("--rpca-lambda", type=float, default=None)
    ap.add_argument("--rpca-max-iters", type=int, default=250)
    ap.add_argument(
        "--hosvd-spatial-downsample",
        type=int,
        default=1,
        help="Spatial downsample factor (complex avg pooling) for HOSVD baseline.",
    )
    ap.add_argument(
        "--hosvd-t-sub",
        type=int,
        default=None,
        help="Optional temporal sub-window length for HOSVD baseline.",
    )
    ap.add_argument(
        "--hosvd-ranks",
        type=str,
        default=None,
        help="Optional HOSVD multilinear ranks as 'rT,rH,rW'.",
    )
    ap.add_argument(
        "--hosvd-energy-fracs",
        type=str,
        default=None,
        help=(
            "Optional per-mode HOSVD energy fractions as 'fT,fH,fW'; ignored if --hosvd-ranks"
            " is set."
        ),
    )
    ap.add_argument("--tile-h", type=int, default=12)
    ap.add_argument("--tile-w", type=int, default=12)
    ap.add_argument("--tile-stride", type=int, default=6)
    ap.add_argument("--lt", type=int, default=4)
    ap.add_argument(
        "--tile-debug-limit",
        type=int,
        default=None,
        help="Debug mode: process only the first N tiles to quickly inspect telemetry.",
    )
    ap.add_argument(
        "--time-window-length",
        type=int,
        default=None,
        help="Number of slow-time samples per replay window (defaults to full length).",
    )
    ap.add_argument(
        "--time-window-offset",
        type=int,
        action="append",
        default=[],
        help="Slow-time offset for a replay window; repeat to export multiple windows.",
    )
    ap.add_argument("--diag-load", type=float, default=1e-2)
    ap.add_argument(
        "--stap-cov-trim-q",
        type=float,
        default=0.0,
        help=(
            "Optional covariance-training trim fraction in [0,1): exclude the top-q "
            "highest-energy Hankel snapshots from covariance estimation (ablation for "
            "self-training/contamination sensitivity)."
        ),
    )
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--huber-c", type=float, default=5.0)
    ap.add_argument("--fd-span-mode", type=str, default="psd")
    ap.add_argument("--fd-span-rel", type=str, default="0.30,1.10")
    ap.add_argument("--fd-fixed-span-hz", type=float, default=None)
    ap.add_argument("--grid-step-rel", type=float, default=0.12)
    ap.add_argument("--max-pts", type=int, default=5)
    ap.add_argument("--fd-min-pts", type=int, default=3)
    ap.add_argument("--fd-min-abs-hz", type=float, default=0.0)
    ap.add_argument("--msd-lambda", type=float, default=0.05)
    ap.add_argument("--msd-ridge", type=float, default=0.12)
    ap.add_argument("--msd-agg", type=str, default="median")
    ap.add_argument("--msd-ratio-rho", type=float, default=0.05)
    ap.add_argument("--msd-contrast-alpha", type=float, default=None)
    ap.add_argument("--motion-half-span-rel", type=float, default=None)
    ap.add_argument("--constraint-mode", type=str, default="exp+deriv")
    ap.add_argument("--constraint-ridge", type=float, default=0.10)
    ap.add_argument("--mvdr-load-mode", type=str, default="auto")
    ap.add_argument("--mvdr-auto-kappa", type=float, default=50.0)
    ap.add_argument("--ka-mode", type=str, default="none")
    ap.add_argument(
        "--ka-prior-path",
        type=Path,
        default=None,
        help="Path to KA prior .npy when --ka-mode=library.",
    )
    ap.add_argument(
        "--ka-directional-beta",
        action="store_true",
        help="Enable directional beta with passband/complement split shrinkage.",
    )
    ap.add_argument("--ka-kappa", type=float, default=40.0)
    ap.add_argument("--ka-beta-bounds", type=str, default="0.05,0.5")
    ap.add_argument("--ka-alpha", type=float, default=None)
    ap.add_argument("--alias-cap-enable", action="store_true")
    ap.add_argument("--alias-cap-alias-thresh", type=float, default=2.0)
    ap.add_argument("--alias-cap-band-med-thresh", type=float, default=0.98)
    ap.add_argument("--alias-cap-smin", type=float, default=0.4)
    ap.add_argument("--alias-cap-c0", type=float, default=1.0)
    ap.add_argument("--alias-cap-exp", type=float, default=1.0)
    ap.add_argument(
        "--alias-psd-select",
        action="store_true",
        help="Enable alias-aware PSD bin selection before KA guards.",
    )
    ap.add_argument(
        "--alias-psd-select-ratio",
        type=float,
        default=1.2,
        help="Alias ratio threshold (flow/f0) for triggering PSD down-selection.",
    )
    ap.add_argument(
        "--alias-psd-select-bins",
        type=int,
        default=1,
        help="Number of positive bins (per sign) to retain when alias PSD selection fires.",
    )
    ap.add_argument(
        "--psd-telemetry",
        action="store_true",
        help="Capture multi-taper PSD telemetry for each tile.",
    )
    ap.add_argument(
        "--psd-tapers",
        type=int,
        default=3,
        help="Number of DPSS tapers to use when PSD telemetry is enabled.",
    )
    ap.add_argument(
        "--psd-bandwidth",
        type=float,
        default=2.0,
        help="DPSS time-half-bandwidth parameter (only used with --psd-telemetry).",
    )
    ap.add_argument(
        "--feasibility-mode",
        type=str,
        default="legacy",
        choices=["legacy", "updated", "blend"],
        help="Feasibility configuration applied to STAP/KA processing.",
    )
    ap.add_argument(
        "--ka-target-retain-f",
        type=float,
        default=None,
        help="Optional KA passband retain target (>=1 keeps or boosts flow energy).",
    )
    ap.add_argument(
        "--ka-target-shrink-perp",
        type=float,
        default=None,
        help=(
            "Optional KA orthogonal shrink target "
            "(<=1 enforces stronger background suppression)."
        ),
    )
    ap.add_argument(
        "--ka-equalize-pf-trace",
        action="store_true",
        help="Enable Pf-trace equalization in KA to preserve flow mean",
    )
    ap.add_argument(
        "--ka-beta-fixed",
        type=float,
        default=None,
        help="Optional fixed beta for blend mode; when set with feasibility_mode=blend, "
        "this value is passed as beta into the KA blend.",
    )
    ap.add_argument(
        "--ka-gate-enable",
        action="store_true",
        help="Enable flow-aware KA gating (alias/flow/depth/PD/registration checks).",
    )
    ap.add_argument(
        "--ka-score-model-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON path for score-space KA risk model. When set together "
            "with PD scoring and a positive --ka-score-alpha, a shrink-only "
            "transform is applied to STAP PD scores based on telemetry features."
        ),
    )
    ap.add_argument(
        "--ka-score-alpha",
        type=float,
        default=0.0,
        help=(
            "Score-space KA shrink strength in [0,1]. When >0 and a score model "
            "is provided, PD scores on high-risk tiles are shrunk so that "
            "S = PD is shrunk."
        ),
    )
    ap.add_argument(
        "--ka-score-contract-v2",
        action="store_true",
        help=(
            "Enable KA Contract v2 score-space shrink-only mode (Phase 2). "
            "This uses alias/guard telemetry + flow coverage to localize a "
            "bounded shrink on PD scores (S=PD). By default this applies in "
            "C1_SAFETY; use --ka-score-contract-v2-mode to also allow C2_UPLIFT "
            "(research-only)."
        ),
    )
    ap.add_argument(
        "--ka-score-contract-v2-force",
        action="store_true",
        help=(
            "Ablation-only: force-apply the KA Contract v2 shrink mapping even "
            "when the contract would disable it (quantifies worst-case harm)."
        ),
    )
    ap.add_argument(
        "--ka-score-contract-v2-mode",
        type=str,
        default="safety",
        choices=["safety", "uplift", "auto"],
        help=(
            "When --ka-score-contract-v2 is enabled, choose which contract states "
            "permit applying the shrink-only mapping: "
            "'safety' applies only in C1_SAFETY, "
            "'uplift' applies only in C2_UPLIFT, and "
            "'auto' applies in both (safety mapping in C1, uplift mapping in C2)."
        ),
    )
    ap.add_argument(
        "--ka-contract-v2-proxy-source",
        type=str,
        default="pd",
        choices=["pd", "eval"],
        help=(
            "Choose the flow-coverage proxy used by the v2 contract. "
            "'pd' uses the PD-derived proxy mask (label-free; default). "
            "'eval' uses the evaluation flow mask (simulator-truth in Brain-* sims; "
            "ablation/oracle mode)."
        ),
    )
    ap.add_argument(
        "--ka-contract-v2-alias-source",
        type=str,
        default="peak",
        choices=["peak", "sum", "whitened", "auto"],
        help=(
            "Choose the alias-risk metric source for the v2 contract: "
            "'peak' uses log(peak Pa / peak Pf) (default), "
            "'sum' uses log(sum Pa / sum Pf), "
            "'whitened' uses the whitened band-ratio tile score (-log Ef/Ea), "
            "'auto' prefers peak then falls back to sum."
        ),
    )
    ap.add_argument(
        "--ka-gate-alias-rmin",
        type=float,
        default=1.30,
        help="Minimum PSD alias ratio required for KA gating to trigger.",
    )
    ap.add_argument(
        "--ka-gate-flow-cov-min",
        type=float,
        default=0.20,
        help="Minimum flow-mask coverage fraction within a tile for gating.",
    )
    ap.add_argument(
        "--ka-gate-depth-min-frac",
        type=float,
        default=0.30,
        help="Lower depth fraction bound for KA gating (0-1).",
    )
    ap.add_argument(
        "--ka-gate-depth-max-frac",
        type=float,
        default=0.70,
        help="Upper depth fraction bound for KA gating (0-1).",
    )
    ap.add_argument(
        "--ka-gate-pd-min",
        type=float,
        default=None,
        help="Minimum normalized PD metric for a tile to allow KA action.",
    )
    ap.add_argument(
        "--ka-gate-reg-psr-max",
        type=float,
        default=6.0,
        help="Upper bound on registration PSR for KA gating (None disables this check).",
    )
    ap.add_argument(
        "--flow-mask-mode",
        type=str,
        default="default",
        choices=["default", "pd_auto"],
        help="Flow mask strategy (default uses geometry, pd_auto thresholds PD map).",
    )
    ap.add_argument(
        "--flow-mask-pd-quantile",
        type=float,
        default=0.995,
        help="Quantile for PD-based flow mask (only used when --flow-mask-mode=pd_auto).",
    )
    ap.add_argument(
        "--flow-mask-depth-min-frac",
        type=float,
        default=0.25,
        help="Minimum depth fraction for PD-based flow mask.",
    )
    ap.add_argument(
        "--flow-mask-depth-max-frac",
        type=float,
        default=0.9,
        help="Maximum depth fraction for PD-based flow mask.",
    )
    ap.add_argument(
        "--flow-mask-erode-iters",
        type=int,
        default=0,
        help="Binary erosion iterations for PD-based flow mask.",
    )
    ap.add_argument(
        "--flow-mask-dilate-iters",
        type=int,
        default=2,
        help="Binary dilation iterations for PD-based flow mask.",
    )
    ap.add_argument(
        "--flow-mask-min-pixels",
        type=int,
        default=64,
        help="Minimum number of pixels required for PD-based flow mask to take effect.",
    )
    ap.add_argument(
        "--flow-mask-min-coverage-frac",
        type=float,
        default=0.0,
        help="Minimum fractional area needed before accepting PD-based mask.",
    )
    ap.add_argument(
        "--flow-mask-union-default",
        dest="flow_mask_union_default",
        action="store_true",
        help="Union the PD-based flow mask with the default geometric ROI.",
    )
    ap.add_argument(
        "--flow-mask-no-union-default",
        dest="flow_mask_union_default",
        action="store_false",
        help="Use the PD-based mask alone (no union with default ROI).",
    )
    ap.set_defaults(flow_mask_union_default=True)
    ap.add_argument(
        "--flow-mask-suppress-alias-depth",
        action="store_true",
        help=(
            "Treat the bg_alias depth band as background in the flow/bg masks "
            "(used for pial-alias regimes so pial alias tiles are H0)."
        ),
    )
    ap.add_argument(
        "--stap-conditional-enable",
        dest="stap_conditional_enable",
        action="store_true",
        help=(
            "Enable conditional STAP execution (default): tiles with zero overlap "
            "with the conditional flow mask are skipped and fall back to baseline PD."
        ),
    )
    ap.add_argument(
        "--stap-conditional-disable",
        dest="stap_conditional_enable",
        action="store_false",
        help="Disable conditional STAP execution (run full STAP on all tiles).",
    )
    ap.set_defaults(stap_conditional_enable=True)
    ap.add_argument(
        "--stap-conditional-mask",
        type=Path,
        default=None,
        help=(
            "Optional .npy flow mask used for conditional STAP gating. "
            "When set, this overrides the bundle's mask_flow.npy for conditional execution, "
            "while evaluation masks remain fixed."
        ),
    )
    ap.add_argument(
        "--stap-conditional-mask-tag",
        type=str,
        default=None,
        help="Optional tag recorded in meta for the conditional mask source.",
    )
    ap.add_argument(
        "--stap-disable",
        action="store_true",
        help=(
            "Skip the STAP core and write a baseline-only bundle "
            "(pd_stap := pd_base; score_pd_stap := score_pd_base)."
        ),
    )
    ap.add_argument("--bg-guard-target-med", type=float, default=0.45)
    ap.add_argument("--bg-guard-target-low", type=float, default=0.16)
    ap.add_argument("--bg-guard-percentile-low", type=float, default=0.10)
    ap.add_argument("--bg-guard-coverage-min", type=float, default=0.10)
    ap.add_argument("--bg-guard-max-scale", type=float, default=1.30)
    ap.add_argument("--bg-guard-enabled", action="store_true")
    ap.add_argument("--bg-guard-target-p90", type=float, default=0.95)
    ap.add_argument("--bg-guard-min-alpha", type=float, default=0.4)
    ap.add_argument("--bg-guard-metric", type=str, default="global")
    ap.add_argument(
        "--flow-alias-hz",
        type=float,
        default=None,
        help="Inject alias tone at this Doppler frequency (Hz); None disables injection.",
    )
    ap.add_argument(
        "--flow-alias-fraction",
        type=float,
        default=0.4,
        help="Fraction of flow mask to modulate when aliasing is enabled.",
    )
    ap.add_argument(
        "--flow-alias-depth-min-frac",
        type=float,
        default=None,
        help="Minimum depth fraction allowed for aliasing (None keeps entire mask).",
    )
    ap.add_argument(
        "--flow-alias-depth-max-frac",
        type=float,
        default=None,
        help="Maximum depth fraction allowed for aliasing (None keeps entire mask).",
    )
    ap.add_argument(
        "--flow-alias-jitter-hz",
        type=float,
        default=0.0,
        help="Uniform ±jitter applied to the alias tone per seed (Hz).",
    )
    ap.add_argument(
        "--bg-alias-hz",
        type=float,
        default=None,
        help=(
            "Inject alias tone at this Doppler frequency (Hz) on the "
            "background mask; None disables."
        ),
    )
    ap.add_argument(
        "--bg-alias-fraction",
        type=float,
        default=0.3,
        help="Fraction of background mask to modulate when background aliasing is enabled.",
    )
    ap.add_argument(
        "--bg-alias-depth-min-frac",
        type=float,
        default=None,
        help="Minimum depth fraction allowed for background aliasing (None keeps entire mask).",
    )
    ap.add_argument(
        "--bg-alias-depth-max-frac",
        type=float,
        default=None,
        help="Maximum depth fraction allowed for background aliasing (None keeps entire mask).",
    )
    ap.add_argument(
        "--bg-alias-jitter-hz",
        type=float,
        default=0.0,
        help="Uniform ±jitter applied to the background alias tone per seed (Hz).",
    )
    ap.add_argument(
        "--bg-alias-highrank-mode",
        type=str,
        default="none",
        choices=["none", "gw_reverb", "gw_reverb_add", "gw_reverb_muladd"],
        help=(
            "Optional high-rank background alias overlay. 'gw_reverb' injects a "
            "spatially varying multi-mode Pa artifact with reverberation-like deep "
            "ghost patches. 'gw_reverb_add' applies the same construction additively "
            "(more aggressive; intended to survive MC--SVD and impact band telemetry). "
            "'gw_reverb_muladd' applies both multiplicative and additive forms."
        ),
    )
    ap.add_argument(
        "--bg-alias-highrank-coverage",
        type=float,
        default=0.0,
        help="Target deep ghost patch coverage fraction in [0,1] for bg-alias-highrank-mode.",
    )
    ap.add_argument(
        "--bg-alias-highrank-shallow-coverage",
        type=float,
        default=1.0,
        help="Fraction of shallow-layer bg pixels included in the high-rank alias mask (1.0 uses the full shallow layer).",
    )
    ap.add_argument(
        "--bg-alias-highrank-margin-px",
        type=int,
        default=0,
        help=(
            "Optional erosion margin (pixels) applied to the eligible deep bg region "
            "before placing high-rank alias ghost patches. This keeps deep patches away "
            "from flow/bg boundaries (reduces mixed-tile overlap)."
        ),
    )
    ap.add_argument(
        "--bg-alias-highrank-freq-jitter-hz",
        type=float,
        default=25.0,
        help="Std dev (Hz) of the spatial frequency-offset field for high-rank alias modes.",
    )
    ap.add_argument(
        "--bg-alias-highrank-drift-step-hz",
        type=float,
        default=12.0,
        help="Std dev (Hz) of the per-ensemble alias frequency random-walk step (0 disables drift).",
    )
    ap.add_argument(
        "--bg-alias-highrank-drift-block-len",
        type=int,
        default=None,
        help=(
            "Optional drift block length (frames) for high-rank alias modes. "
            "When set to a value smaller than pulses_per_set, the alias frequency "
            "random walk is applied across blocks within a replay window (helps "
            "create within-window high-rank behavior for 64-frame windows)."
        ),
    )
    ap.add_argument(
        "--bg-alias-highrank-pf-leak-eta",
        type=float,
        default=0.0,
        help="Pf leakage coupling factor for high-rank alias modes (relative to Pa mode amplitudes).",
    )
    ap.add_argument(
        "--bg-alias-highrank-amp",
        type=float,
        default=0.20,
        help="Overall complex amplitude scale for high-rank alias modes (0 disables).",
    )
    ap.add_argument(
        "--synth-amp-jitter",
        type=float,
        default=0.05,
        help="Per-pulse amplitude jitter used when synthesizing the compounded slow-time cube (pre-injection).",
    )
    ap.add_argument(
        "--synth-phase-jitter",
        type=float,
        default=0.25,
        help="Per-pulse phase jitter (radians) used when synthesizing the compounded slow-time cube (pre-injection).",
    )
    ap.add_argument(
        "--synth-noise-level",
        type=float,
        default=0.01,
        help="Additive complex noise level used during slow-time cube synthesis (pre-injection).",
    )
    ap.add_argument(
        "--synth-shift-max-px",
        type=int,
        default=1,
        help="Max absolute integer per-pulse spatial shift (px) during cube synthesis; 0 disables shifts.",
    )
    ap.add_argument(
        "--flow-amp-scale",
        type=float,
        default=1.0,
        help="Optional global amplitude scale for flow-mask pixels (1.0 leaves them unchanged).",
    )
    ap.add_argument(
        "--alias-amp-scale",
        type=float,
        default=1.0,
        help="Optional amplitude scale for synthetic alias components (1.0 leaves them unchanged).",
    )
    ap.add_argument(
        "--flow-doppler-min-hz",
        type=float,
        default=None,
        help="Minimum synthetic flow Doppler frequency (Hz) applied on the flow mask.",
    )
    ap.add_argument(
        "--flow-doppler-max-hz",
        type=float,
        default=None,
        help="Maximum synthetic flow Doppler frequency (Hz) applied on the flow mask.",
    )
    ap.add_argument(
        "--flow-doppler-tone-amp",
        type=float,
        default=0.0,
        help=(
            "Optional additive narrowband flow tone amplitude (relative to flow-mask RMS). "
            "0 disables; when >0, adds a per-pixel tone at the synthetic Doppler frequency."
        ),
    )
    ap.add_argument(
        "--flow-doppler-noise-amp",
        type=float,
        default=0.0,
        help=(
            "Optional additive high-rank flow noise amplitude (relative to flow-mask RMS). "
            "0 disables; when >0, adds a per-pixel temporally correlated complex noise "
            "process centered at the synthetic Doppler frequency."
        ),
    )
    ap.add_argument(
        "--flow-doppler-noise-rho",
        type=float,
        default=0.97,
        help=(
            "AR(1) correlation coefficient for --flow-doppler-noise-amp (0..1). "
            "Higher values yield a narrowerband Doppler-like spectrum around the "
            "per-pixel Doppler frequency."
        ),
    )
    ap.add_argument(
        "--flow-doppler-noise-mode",
        type=str,
        default="fft_band",
        choices=["fft_band", "ar1_shift"],
        help=(
            "Noise model used when --flow-doppler-noise-amp>0. "
            "'fft_band' injects band-limited complex noise in [flow_doppler_min_hz,flow_doppler_max_hz]. "
            "'ar1_shift' injects AR(1) noise shifted by the per-pixel Doppler frequency."
        ),
    )
    ap.add_argument(
        "--aperture-phase-std",
        type=float,
        default=0.0,
        help="RMS phase screen (rad) applied across aperture (0 disables).",
    )
    ap.add_argument(
        "--aperture-phase-corr-len",
        type=float,
        default=12.0,
        help="Correlation length (elements) for aperture phase screen.",
    )
    ap.add_argument(
        "--aperture-phase-seed",
        type=int,
        default=111,
        help="Additional RNG seed offset for aperture phase screen.",
    )
    ap.add_argument(
        "--clutter-beta",
        type=float,
        default=0.0,
        help="Temporal clutter 1/f^beta slope (<=0 disables).",
    )
    ap.add_argument(
        "--clutter-snr-db",
        type=float,
        default=-6.0,
        help="Temporal clutter SNR (dB) relative to background when beta>0.",
    )
    ap.add_argument(
        "--clutter-depth-min-frac",
        type=float,
        default=0.20,
        help="Minimum depth fraction for clutter injection.",
    )
    ap.add_argument(
        "--clutter-depth-max-frac",
        type=float,
        default=0.95,
        help="Maximum depth fraction for clutter injection.",
    )
    ap.add_argument(
        "--clutter-mode",
        type=str,
        default="fullrank",
        choices=["fullrank", "lowrank"],
        help=(
            "Temporal clutter model used when --clutter-beta>0. "
            "'fullrank' injects independent colored noise per pixel; "
            "'lowrank' injects a small number of global colored temporal modes mixed spatially."
        ),
    )
    ap.add_argument(
        "--clutter-rank",
        type=int,
        default=3,
        help="Rank used when --clutter-mode=lowrank (number of global temporal modes).",
    )
    ap.add_argument(
        "--vibration-hz",
        type=float,
        default=None,
        help="Global vibration tone Doppler frequency (Hz); None disables.",
    )
    ap.add_argument(
        "--vibration-amp",
        type=float,
        default=0.0,
        help="Relative complex amplitude of the global vibration tone (0 disables).",
    )
    ap.add_argument(
        "--vibration-depth-min-frac",
        type=float,
        default=0.15,
        help="Depth fraction where the global vibration amplitude starts decaying.",
    )
    ap.add_argument(
        "--vibration-depth-decay-frac",
        type=float,
        default=0.25,
        help="Depth decay scale (in depth fraction units) for the global vibration tone.",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default="msd",
        help=(
            "Score pooling mode: 'msd' (default), 'pd', 'band_ratio', or 'band_ratio_whitened' "
            "(whitened PSD log-ratio stored in band_ratio pools)"
        ),
    )
    ap.add_argument(
        "--band-ratio-mode",
        type=str,
        default="legacy",
        help="Band-ratio flavor: 'legacy' PD ratio or 'whitened' PSD log-ratio.",
    )
    ap.add_argument(
        "--psd-br-tile-mode",
        type=str,
        default="mean",
        help=(
            "Whitened band-ratio tile PSD aggregation mode: "
            "'mean' (coherent tile-mean series), "
            "'incoherent' (avg per-pixel PSD), "
            "'incoherent_max' (max per-pixel PSD), "
            "or 'incoherent_qNN' (e.g. incoherent_q90)."
        ),
    )
    ap.add_argument(
        "--psd-br-flow-low",
        type=float,
        default=120.0,
        help="Lower (Hz) bound of the flow band for whitened band-ratio scoring.",
    )
    ap.add_argument(
        "--psd-br-flow-high",
        type=float,
        default=400.0,
        help="Upper (Hz) bound of the flow band for whitened band-ratio scoring.",
    )
    ap.add_argument(
        "--psd-br-alias-center",
        type=float,
        default=900.0,
        help="Center (Hz) of the alias band for whitened band-ratio scoring.",
    )
    ap.add_argument(
        "--psd-br-alias-width",
        type=float,
        default=15.625,
        help="Half-width (Hz) of the alias band for whitened band-ratio scoring.",
    )
    return ap.parse_args()


def parse_coords(coord_list: Sequence[str]) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for entry in coord_list:
        clean = entry.replace(":", ",")
        parts = [p.strip() for p in clean.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid coord '{entry}'")
        coords.append((int(parts[0]), int(parts[1])))
    return coords


def _apply_brain_profile_defaults(args: argparse.Namespace) -> None:
    """Apply high-level Brain-* profile defaults.

    The Brain-* profiles mirror the operating regimes described in the
    methodology (Brain-OpenSkull / Brain-AliasContract / Brain-SkullOR /
    Brain-Pial128). When a Brain-* profile is requested we:
      - use a motion-compensated SVD baseline with registration and a
        literature-style energy-fraction rule,
      - enable the clinical STAP preset with PD-based scoring, and
      - configure PD-based flow masks consistent with the brain fUS methods.
    """
    profile = getattr(args, "profile", None)
    if profile is None:
        return

    # Common defaults for all Brain-* profiles: MC-SVD baseline with
    # registration, clinical STAP PD preset, and PD-based flow masks.
    #
    # IMPORTANT: allow callers to override the baseline (e.g. RPCA/HOSVD) for
    # fair baseline comparisons. We only force MC-SVD when the caller left the
    # baseline at its parser default ("svd").
    if getattr(args, "baseline", "svd") == "svd":
        args.baseline = "mc_svd"
    args.reg_enable = True
    args.reg_method = "phasecorr"
    args.reg_subpixel = 4
    args.reg_reference = "median"
    args.svd_profile = "literature"
    # Phase 3 (baseline fairness): tune-once-then-freeze MC--SVD energy-fraction
    # baseline for Brain-* on a pre-committed calibration configuration.
    # See: reports/brain_mcsvd_energy_sweep.csv (summary of the calibration sweep).
    if args.svd_rank is None and args.svd_energy_frac is None:
        args.svd_energy_frac = 0.90

    args.stap_profile = "clinical"
    args.score_mode = "pd"

    # Slow-time cube synthesis (pre-injection): for Brain-* pilots we prefer a
    # largely coherent, low-rank tissue stack so that MC--SVD removes a small
    # clutter subspace and the explicit Doppler/alias/clutter injections drive
    # spectral structure. The legacy synth jitter defaults can make the stack
    # unrealistically high-rank for short windows (e.g. 64 frames), causing
    # MC--SVD to remove dozens of modes and wiping out flow evidence.
    if getattr(args, "synth_amp_jitter", 0.05) == 0.05:
        args.synth_amp_jitter = 0.0
    if getattr(args, "synth_phase_jitter", 0.25) == 0.25:
        args.synth_phase_jitter = 0.0
    if getattr(args, "synth_noise_level", 0.01) == 0.01:
        args.synth_noise_level = 0.0
    if getattr(args, "synth_shift_max_px", 1) == 1:
        args.synth_shift_max_px = 0

    # Temporal clutter injection: model clutter as low-rank global modes to
    # match the MC--SVD "low-rank clutter" assumption used throughout the
    # Brain-* methodology. Users can still override to fullrank explicitly.
    if getattr(args, "clutter_mode", "fullrank") == "fullrank":
        args.clutter_mode = "lowrank"
    if getattr(args, "clutter_rank", 3) == 3:
        args.clutter_rank = 3

    # Evaluation masks for Brain-* simulations should reflect simulator truth
    # (default geometric/injection masks). A separate PD-derived proxy mask is
    # still exported and used for conditional execution / contract telemetry.
    args.flow_mask_mode = "default"
    # Allow explicit CLI overrides for the proxy-mask extraction knobs by only
    # applying these defaults when the user did not provide a different value.
    if getattr(args, "flow_mask_pd_quantile", 0.995) == 0.995:
        args.flow_mask_pd_quantile = 0.995
    if getattr(args, "flow_mask_depth_min_frac", 0.25) == 0.25:
        args.flow_mask_depth_min_frac = 0.25
    if getattr(args, "flow_mask_depth_max_frac", 0.85) == 0.85:
        args.flow_mask_depth_max_frac = 0.85
    if getattr(args, "flow_mask_dilate_iters", 2) == 2:
        args.flow_mask_dilate_iters = 2
    if getattr(args, "flow_mask_union_default", True) is True:
        args.flow_mask_union_default = False

    # Pial profile additionally suppresses the shallow alias depth band when
    # building the flow mask so the pial band is always treated as H0.
    if profile == "Brain-Pial128":
        args.flow_mask_suppress_alias_depth = True


def main() -> None:
    args = parse_args()

    # Apply high-level Brain-* operating profile presets (if any) before
    # expanding the lower-level STAP presets.
    _apply_brain_profile_defaults(args)

    if getattr(args, "stap_profile", "lab") == "clinical":
        # Clinical STAP profile: fixed, conservative configuration intended to
        # reflect deployable intra-op fUS constraints. This preset is applied
        # before downstream configuration is built.
        # Allow explicit CLI overrides by only applying these defaults when the
        # user did not provide a different value. We detect this by checking
        # whether the argument is still at the parser default.
        if getattr(args, "tile_h", 12) == 12:
            args.tile_h = 8
        if getattr(args, "tile_w", 12) == 12:
            args.tile_w = 8
        if getattr(args, "tile_stride", 6) == 6:
            args.tile_stride = 3
        if getattr(args, "lt", 4) == 4:
            args.lt = 8

        # Covariance / MVDR configuration
        if getattr(args, "diag_load", 1e-2) == 1e-2:
            args.diag_load = 0.07
        args.cov_estimator = "tyler_pca"
        args.huber_c = 5.0
        # Primary STAP score uses a *fixed* Doppler grid (frozen per profile)
        # rather than a per-tile PSD-adaptive grid. This avoids silent regime
        # shifts where alias-dominant tiles drive the "flow" subspace selection.
        #
        # Use a fixed span covering Pf (up to ~250 Hz for PRF~1.5 kHz brain fUS).
        # The underlying tile kernel will cap the effective grid to fit Lt.
        args.fd_span_mode = "fixed"
        args.fd_span_rel = "0.30,1.10"
        args.fd_fixed_span_hz = 250.0
        args.grid_step_rel = 0.20
        # Keep a small but nontrivial grid. A tiny grid (e.g. 3 points) can
        # under-represent heterogeneous flow Dopplers; a very dense grid is
        # unnecessary for the conservative profile.
        args.max_pts = 15
        args.fd_min_pts = 9
        args.constraint_mode = "exp+deriv"
        args.constraint_ridge = 0.18
        args.mvdr_load_mode = "auto"
        args.mvdr_auto_kappa = 120.0

        # MSD scoring configuration
        args.msd_lambda = 0.05
        args.msd_ridge = 0.10
        args.msd_agg = "median"
        args.msd_ratio_rho = 0.05
        args.msd_contrast_alpha = 0.6

        # Whitened band-ratio telemetry configuration (only when using an
        # MC-SVD/SVD baseline, since the whitened ratio path assumes an
        # MC-SVD-style clutter prior). For other baselines (e.g. RPCA, HOSVD)
        # we leave the default legacy band-ratio so that PD scoring still
        # works without triggering whitened-ratio constraints.
        if getattr(args, "baseline", "mc_svd") in {"mc_svd", "svd"}:
            args.band_ratio_mode = "whitened"
            args.psd_br_flow_low = 30.0
            # Match the brain-fUS band geometry described in the methodology:
            # Pf ~ [30,250] Hz, guard ~ [250,400] Hz, Pa ~ [400,Nyquist] Hz.
            # The band-ratio recorder uses (flow_low, flow_high) and an
            # (alias_center, alias_half_width) parameterization; center=575 and
            # half-width=175 yields an alias band starting at ~400 Hz.
            args.psd_br_flow_high = 250.0
            args.psd_br_alias_center = 575.0
            args.psd_br_alias_width = 175.0

        # Limit effective slow-time support per window if not explicitly set.
        if args.time_window_length is None:
            args.time_window_length = 32

        # Limit training snapshots per tile via environment controls if not
        # already set by the user.
        os.environ.setdefault("STAP_SNAPSHOT_STRIDE", "4")
        os.environ.setdefault("STAP_MAX_SNAPSHOTS", "64")

        # Enable the batched fast-path by default for the clinical preset.
        # The core will fall back to the slow path automatically if it is not
        # eligible (e.g., torch unavailable, KA enabled, debug requested).
        os.environ.setdefault("STAP_FAST_PATH", "1")

        # For PD-based clinical runs, default to the PD-only fast path so that
        # the GPU batched core uses the lighter band-energy computation instead
        # of the full MSD contrast kernel. This preserves PD ROC while reducing
        # latency, and can be disabled manually by clearing STAP_FAST_PD_ONLY.
        if getattr(args, "score_mode", "pd") == "pd":
            os.environ.setdefault("STAP_FAST_PD_ONLY", "1")

    if getattr(args, "cuda_warmup", True):
        dev = str(getattr(args, "stap_device", "cuda") or "")
        if dev.lower().startswith("cuda"):
            try:
                import torch  # local import (torch optional)

                if torch.cuda.is_available():
                    # Initialize CUDA context + common libraries so baseline/STAP
                    # telemetry does not include one-time setup overhead.
                    _ = torch.empty((1,), device=dev)
                    a = torch.randn((32, 32), device=dev, dtype=torch.float32)
                    b = torch.randn((32, 32), device=dev, dtype=torch.float32)
                    ac = (a + 1j * b).to(dtype=torch.complex64)
                    bc = (b + 1j * a).to(dtype=torch.complex64)
                    _ = ac @ bc
                    C = ac @ ac.conj().T
                    _ = torch.linalg.eigh(C)
                    # Warm up Cholesky + triangular solves (used heavily by STAP).
                    C_pd = C + (1e-2 * torch.eye(C.shape[0], device=dev, dtype=C.dtype))
                    L = torch.linalg.cholesky(C_pd)
                    rr = torch.randn((C.shape[0], 64), device=dev, dtype=torch.float32)
                    ri = torch.randn((C.shape[0], 64), device=dev, dtype=torch.float32)
                    rhs = (rr + 1j * ri).to(dtype=C.dtype)
                    _ = torch.linalg.solve_triangular(L, rhs, upper=False)
                    _ = torch.cholesky_solve(rhs, L)
                    x = torch.randn((32, 32), device=dev, dtype=torch.float32)
                    xc = (x + 1j * x).to(dtype=torch.complex64)
                    _ = torch.fft.fft2(xc)
                    # Warm up common reduction/select kernels used by registration
                    # and telemetry (sort/topk); this avoids one-time overhead
                    # polluting latency measurements.
                    v = torch.rand((1024,), device=dev, dtype=torch.float32)
                    _ = torch.sort(v)
                    _ = torch.topk(v, k=32)
                    torch.cuda.synchronize()
            except Exception as exc:  # pragma: no cover - optional warmup
                print(f"[replay_stap_from_run] CUDA warmup skipped: {exc}")

    # Optional preset for motion-compensated SVD baseline. When the user
    # selects the 'literature' profile and has not provided an explicit SVD
    # rank or energy fraction, fall back to an energy-fraction rule that
    # removes the smallest number of singular components accounting for ~95%
    # of the slow-time energy (default). Brain-* profiles override this to a
    # separately frozen value (currently e=0.90; see _apply_brain_profile_defaults).
    # This mirrors common practice in spatiotemporal SVD clutter filtering for
    # ultrafast Doppler / fUS, while leaving manual --svd-rank/--svd-energy-frac
    # overrides untouched.
    if getattr(args, "svd_profile", "default") == "literature":
        if args.svd_rank is None and args.svd_energy_frac is None:
            args.svd_energy_frac = 0.95

    src_root = args.src
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src_root / "meta.json").read_text())
    geom_meta = meta.get("geometry") or meta.get("sim_geom")
    if geom_meta is None:
        raise KeyError("meta.json missing 'geometry' or 'sim_geom'")
    geom = SimGeom(
        Nx=int(geom_meta["Nx"]),
        Ny=int(geom_meta["Ny"]),
        dx=float(geom_meta["dx"]),
        dy=float(geom_meta["dy"]),
        c0=float(geom_meta["c0"]),
        rho0=float(geom_meta["rho0"]),
        pml_size=int(geom_meta.get("pml", 20)),
        cfl=float(geom_meta.get("cfl", 0.3)),
        f0=float(meta.get("f0_hz", 7.5e6)),
        ncycles=int(meta.get("ncycles", 3)),
    )
    if "angles_deg" in meta:
        angles = [float(a) for a in meta["angles_deg"]]
    elif "angles_deg_sets" in meta:
        angle_sets_meta = meta["angles_deg_sets"]
        if not angle_sets_meta:
            raise ValueError("meta['angles_deg_sets'] is empty")
        angles = [float(a) for a in angle_sets_meta[0]]
    elif "base_angles_deg" in meta:
        # Pilot generators (e.g., pilot_motion) record base angles and
        # per-ensemble jittered 'angles_used_deg'. Replay directories are named
        # by base angles (ensX_angle_{deg}), so prefer base_angles_deg here.
        angles = [float(a) for a in meta["base_angles_deg"]]
    elif "angles_used_deg" in meta:
        # Fallback: take the first ensemble's used angles. Directory names remain
        # based on rounded base angles, but in some pilots they may coincide. If
        # not, this will raise downstream when directories are missing.
        used = meta["angles_used_deg"]
        if not used:
            raise ValueError("meta['angles_used_deg'] is empty")
        angles = [float(a) for a in used[0]]
    else:
        raise KeyError(
            "meta.json missing 'angles_deg'/'angles_deg_sets'/'base_angles_deg'/'angles_used_deg'"
        )
    angle_sets = _load_angle_data(src_root, angles)

    # Optional heavy CUDA warmup for steady-state latency measurements.
    #
    # The default warmup above initializes CUDA + small representative kernels,
    # but cuFFT (and some reductions) can have substantial one-time overhead
    # that depends on the actual H×W problem size. For latency profiling we want
    # timings representative of steady state, so allow an opt-in warmup that
    # runs a full MC--SVD baseline pass on a dummy cube of the same shape.
    if os.getenv("CUDA_WARMUP_HEAVY", "").strip().lower() in {"1", "true", "yes", "on"}:
        dev = str(getattr(args, "stap_device", "cuda") or "")
        if dev.lower().startswith("cuda"):
            try:
                import torch  # local import (torch optional)

                if torch.cuda.is_available():
                    env_mcsvd_torch = os.getenv("MC_SVD_TORCH", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    env_reg_torch = os.getenv("MC_SVD_REG_TORCH", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    baseline_type = str(getattr(args, "baseline", "mc_svd") or "").strip().lower()
                    if (
                        env_mcsvd_torch
                        and env_reg_torch
                        and baseline_type in {"mc_svd", "svd"}
                        and bool(getattr(args, "reg_enable", False))
                    ):
                        from sim.kwave.common import _baseline_pd_mcsvd

                        T_warm = int(getattr(args, "time_window_length", 0) or 32)
                        H_warm = int(getattr(geom, "Ny", 0) or 0)
                        W_warm = int(getattr(geom, "Nx", 0) or 0)
                        if T_warm > 1 and H_warm > 0 and W_warm > 0:
                            print(
                                f"[replay_stap_from_run] Heavy CUDA warmup (MC-SVD torch reg): "
                                f"T={T_warm}, H={H_warm}, W={W_warm}",
                                flush=True,
                            )
                            rng = np.random.default_rng(0)
                            dummy = (
                                rng.standard_normal((T_warm, H_warm, W_warm), dtype=np.float32)
                                + 1j
                                * rng.standard_normal((T_warm, H_warm, W_warm), dtype=np.float32)
                            ).astype(np.complex64, copy=False)
                            _baseline_pd_mcsvd(
                                dummy,
                                reg_enable=True,
                                reg_method=str(getattr(args, "reg_method", "phasecorr")),
                                reg_subpixel=int(getattr(args, "reg_subpixel", 4)),
                                reg_reference=str(getattr(args, "reg_reference", "median")),
                                svd_rank=getattr(args, "svd_rank", None),
                                svd_energy_frac=getattr(args, "svd_energy_frac", None),
                                device=dev,
                                return_filtered_cube=False,
                            )

                    # Warm up key STAP kernels (batched eigvalsh/cholesky/triangular solves)
                    # at realistic Lt and snapshot sizes, so STAP timings are closer to
                    # steady state even when the baseline stage is CPU-heavy (e.g. NumPy
                    # registration).
                    env_fast = os.getenv("STAP_FAST_PATH", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                    if env_fast:
                        Lt_warm = int(getattr(args, "lt", 0) or 0)
                        T_warm = int(getattr(args, "time_window_length", 0) or 0)
                        th_warm = int(getattr(args, "tile_h", 0) or 0)
                        tw_warm = int(getattr(args, "tile_w", 0) or 0)
                        if T_warm <= 0:
                            T_warm = 32
                        if Lt_warm <= 0:
                            Lt_warm = min(8, max(2, T_warm - 1))
                        if th_warm <= 0:
                            th_warm = 8
                        if tw_warm <= 0:
                            tw_warm = 8
                        if 2 <= Lt_warm < T_warm and th_warm > 0 and tw_warm > 0:
                            stride_env = os.getenv("STAP_SNAPSHOT_STRIDE", "").strip()
                            max_env = os.getenv("STAP_MAX_SNAPSHOTS", "").strip()
                            try:
                                stride = int(stride_env) if stride_env else 1
                            except ValueError:
                                stride = 1
                            stride = max(1, stride)
                            try:
                                max_snaps = int(max_env) if max_env else None
                            except ValueError:
                                max_snaps = None

                            N_full = T_warm - Lt_warm + 1
                            N_eff = (N_full + stride - 1) // stride
                            if max_snaps is not None and max_snaps > 0 and N_eff > max_snaps:
                                N_eff = int(max_snaps)
                            P = int(max(1, N_eff) * th_warm * tw_warm)

                            # Choose a warm batch size matching the default CUDA tiler batch,
                            # but clamp to avoid OOM for larger Lt/P regimes.
                            B_target = 192
                            P_target = int(max(1, P))
                            # Approx bytes per element for rr+ri+complex S ~ 16 bytes.
                            # Keep this warmup lightweight: it is only to initialize kernels.
                            max_bytes = 64 * 1024 * 1024
                            denom = int(max(1, Lt_warm) * max(1, P_target))
                            max_B = int(max(1, (max_bytes // 16) // denom))
                            B_warm = int(min(B_target, max_B))
                            P_warm = P_target
                            if B_warm < 8:
                                B_warm = 8
                                max_P = int(max(1, (max_bytes // 16) // (B_warm * max(1, Lt_warm))))
                                P_warm = int(min(P_target, max_P))
                            if P_warm <= 0:
                                P_warm = 1
                            try:
                                K_warm = int(getattr(args, "max_pts", 15) or 15)
                            except Exception:
                                K_warm = 15
                            K_warm = max(3, min(K_warm, 21))
                            print(
                                f"[replay_stap_from_run] Heavy CUDA warmup (STAP core): "
                                f"B={B_warm}, Lt={Lt_warm}, P={P_warm}, K={K_warm}",
                                flush=True,
                            )

                            rr = torch.randn(
                                (B_warm, Lt_warm, P_warm), device=dev, dtype=torch.float32
                            )
                            ri = torch.randn(
                                (B_warm, Lt_warm, P_warm), device=dev, dtype=torch.float32
                            )
                            S = (rr + 1j * ri).to(dtype=torch.complex64)
                            R = torch.matmul(S, S.conj().transpose(-2, -1)) / float(P_warm)
                            herm = 0.5 * (R + R.conj().transpose(-2, -1))
                            _ = torch.linalg.eigvalsh(herm).real

                            eye = torch.eye(Lt_warm, device=dev, dtype=herm.dtype).unsqueeze(0)
                            R_lam = herm + (1e-2 * eye)
                            L = torch.linalg.cholesky(R_lam)
                            Y = torch.linalg.solve_triangular(L, S, upper=False)

                            # Optional warmup: Triton-fused Tyler weights kernel (if enabled).
                            # Without this, the first real STAP window can include Triton JIT
                            # compilation overhead in latency telemetry.
                            triton_env = os.getenv("STAP_TYLER_TRITON_WEIGHTS", "").strip().lower()
                            if triton_env in {"1", "true", "yes", "on"}:
                                warm_triton = True
                            elif triton_env in {"0", "false", "no", "off"}:
                                warm_triton = False
                            else:
                                # Mirror the runtime default: enable Triton
                                # Tyler weights fusion when available on CUDA.
                                warm_triton = True
                            if warm_triton:
                                try:
                                    from pipeline.stap.triton_ops import (
                                        TylerWeightsConfig,
                                        triton_available as _triton_available,
                                        tyler_weights_scale_triton,
                                    )

                                    if _triton_available():
                                        out = torch.empty_like(S)
                                        tyler_weights_scale_triton(
                                            Y,
                                            S,
                                            out,
                                            Lt=int(Lt_warm),
                                            eps=1e-8,
                                            cfg=TylerWeightsConfig(),
                                        )
                                except Exception:
                                    pass

                            cr = torch.randn(
                                (B_warm, Lt_warm, K_warm), device=dev, dtype=torch.float32
                            )
                            ci = torch.randn(
                                (B_warm, Lt_warm, K_warm), device=dev, dtype=torch.float32
                            )
                            Ct_exp = (cr + 1j * ci).to(dtype=torch.complex64)
                            Cw = torch.linalg.solve_triangular(L, Ct_exp, upper=False)
                            Gram = torch.bmm(Cw.conj().transpose(1, 2), Cw)
                            eye_k = torch.eye(K_warm, device=dev, dtype=Gram.dtype).unsqueeze(0)
                            Gram = Gram + (1e-2 * eye_k)
                            Lg = torch.linalg.cholesky(Gram)

                            ar = torch.randn((B_warm, K_warm, K_warm), device=dev, dtype=torch.float32)
                            ai = torch.randn((B_warm, K_warm, K_warm), device=dev, dtype=torch.float32)
                            A = (ar + 1j * ai).to(dtype=torch.complex64)
                            _ = torch.cholesky_solve(A, Lg)
                            torch.cuda.synchronize()
            except Exception as exc:  # pragma: no cover - optional warmup
                print(f"[replay_stap_from_run] Heavy CUDA warmup skipped: {exc}", flush=True)

    # Optional Macé/MaceBridge ROI defaults: when a replay run directory
    # contains roi_H1.npy / roi_H0.npy, treat roi_H1 as the default flow
    # mask and everything else as background. These defaults are passed
    # into write_acceptance_bundle so that synthetic flow/alias/clutter
    # injections and PD-based flow mask refinement respect Macé-derived
    # anatomy. For non-MaceBridge runs these files are absent and the
    # internal circular defaults are used instead.
    flow_mask_default: np.ndarray | None = None
    bg_mask_default: np.ndarray | None = None
    micro_vessels_arr: np.ndarray | None = None
    alias_vessels_arr: np.ndarray | None = None
    roi_h1_path = src_root / "roi_H1.npy"
    roi_h0_path = src_root / "roi_H0.npy"
    if roi_h1_path.exists():
        roi_H1 = np.load(roi_h1_path)
        roi_H1 = _maybe_resize_roi_mask(roi_H1, geom)
        # Background is simply the complement of H1; any explicit H0 ROI is
        # a subset of this and does not need to be treated separately for
        # purposes of flow/background masking.
        flow_mask_default = roi_H1
        bg_mask_default = ~roi_H1
        # If an explicit H0 mask is present and shape-compatible, we keep it
        # only to sanity-check alignment; otherwise it is ignored.
        if roi_h0_path.exists():
            try:
                roi_H0 = np.load(roi_h0_path)
                _ = _maybe_resize_roi_mask(roi_H0, geom)
            except Exception:
                # Do not fail replay if the auxiliary H0 mask is misaligned;
                # flow_mask_default/bg_mask_default already carry a safe prior.
                pass
        # Optional Macé vessel fields for MaceBridge: when present, these
        # arrays encode per-slice microvascular and alias vessel centerlines
        # on the Macé grid. We treat them as defined on the PD grid up to
        # clamping and pass them into write_acceptance_bundle so that Pf/Pa
        # modulations can be injected at replay time.
        micro_path = src_root / "micro_vessels.npy"
        alias_path = src_root / "alias_vessels.npy"
        if micro_path.exists():
            micro_vessels_arr = np.load(micro_path)
        if alias_path.exists():
            alias_vessels_arr = np.load(alias_path)

    # For k-Wave Brain-* pilots, prefer simulator-truth masks from an existing
    # acceptance bundle under the source directory when present. This ensures
    # synthetic injections (flow/alias/clutter) and evaluation masks share the
    # same geometry, avoiding accidental injection on a fallback circular mask.
    if flow_mask_default is None or bg_mask_default is None:
        try:
            pw_dirs = sorted(p for p in src_root.iterdir() if p.is_dir() and p.name.startswith("pw_"))
        except Exception:
            pw_dirs = []
        for pw_dir in pw_dirs:
            mask_flow_path = pw_dir / "mask_flow.npy"
            mask_bg_path = pw_dir / "mask_bg.npy"
            if not (mask_flow_path.exists() and mask_bg_path.exists()):
                continue
            try:
                flow_mask_candidate = np.load(mask_flow_path)
                bg_mask_candidate = np.load(mask_bg_path)
                flow_mask_candidate = _maybe_resize_roi_mask(flow_mask_candidate, geom)
                bg_mask_candidate = _maybe_resize_roi_mask(bg_mask_candidate, geom)
            except Exception:
                continue
            flow_mask_default = flow_mask_candidate.astype(bool, copy=False)
            bg_mask_default = bg_mask_candidate.astype(bool, copy=False)
            break

    hosvd_ranks: tuple[int, int, int] | None = None
    if args.hosvd_ranks:
        parts = [p.strip() for p in str(args.hosvd_ranks).split(",") if p.strip()]
        if len(parts) != 3:
            raise SystemExit(
                "Invalid --hosvd-ranks "
                f"'{args.hosvd_ranks}'; expected 'rT,rH,rW' with three integers."
            )
        try:
            hosvd_ranks = (int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError as exc:
            raise SystemExit(
                "Invalid --hosvd-ranks "
                f"'{args.hosvd_ranks}'; expected 'rT,rH,rW' with three integers."
            ) from exc

    hosvd_energy_fracs: tuple[float, float, float] | None = None
    if args.hosvd_energy_fracs and hosvd_ranks is None:
        parts_f = [p.strip() for p in str(args.hosvd_energy_fracs).split(",") if p.strip()]
        if len(parts_f) != 3:
            raise SystemExit(
                "Invalid --hosvd-energy-fracs "
                f"'{args.hosvd_energy_fracs}'; expected 'fT,fH,fW' with three floats."
            )
        try:
            hosvd_energy_fracs = (float(parts_f[0]), float(parts_f[1]), float(parts_f[2]))
        except ValueError as exc:
            raise SystemExit(
                "Invalid --hosvd-energy-fracs "
                f"'{args.hosvd_energy_fracs}'; expected 'fT,fH,fW' with three floats."
            ) from exc
    pulses_per_set = int(
        meta.get(
            "pulses_per_set",
            meta.get("pulses_per_ensemble", meta.get("pulses", 64)),
        )
    )
    prf = float(meta.get("prf_hz", 3000.0))
    seed = int(meta.get("seed", 0))
    ensembles = len(angle_sets)
    total_frames_full = int(pulses_per_set) * int(ensembles)

    span_bounds = tuple(float(x.strip()) for x in args.fd_span_rel.split(",") if x.strip())
    if len(span_bounds) != 2:
        raise ValueError("--fd-span-rel must be 'min,max'")
    ka_beta_bounds = tuple(float(x.strip()) for x in args.ka_beta_bounds.split(",") if x.strip())
    if len(ka_beta_bounds) != 2:
        raise ValueError("--ka-beta-bounds must be 'min,max'")

    guard_opts = {
        "guard_target_med": float(args.bg_guard_target_med),
        "guard_target_low": float(args.bg_guard_target_low),
        "guard_percentile_low": float(args.bg_guard_percentile_low),
        "guard_tile_coverage_min": float(args.bg_guard_coverage_min),
        "guard_max_scale": float(args.bg_guard_max_scale),
        "bg_guard_enabled": bool(args.bg_guard_enabled),
        "bg_guard_target_p90": float(args.bg_guard_target_p90),
        "bg_guard_min_alpha": float(args.bg_guard_min_alpha),
        "bg_guard_metric": str(args.bg_guard_metric),
    }
    if args.alias_cap_enable:
        guard_opts.update(
            {
                "alias_cap_enable": True,
                "alias_cap_alias_thresh": float(args.alias_cap_alias_thresh),
                "alias_cap_band_med_thresh": float(args.alias_cap_band_med_thresh),
                "alias_cap_smin": float(args.alias_cap_smin),
                "alias_cap_c0": float(args.alias_cap_c0),
                "alias_cap_exp": float(args.alias_cap_exp),
            }
        )
    ka_gate_enabled = bool(args.ka_gate_enable or args.feasibility_mode == "updated")
    if ka_gate_enabled:
        gate_opts: dict[str, float | bool] = {
            "ka_gate_enable": True,
            "ka_gate_alias_rmin": float(args.ka_gate_alias_rmin),
            "ka_gate_flow_cov_min": float(args.ka_gate_flow_cov_min),
            "ka_gate_depth_min_frac": float(args.ka_gate_depth_min_frac),
            "ka_gate_depth_max_frac": float(args.ka_gate_depth_max_frac),
        }
        if args.ka_gate_pd_min is not None:
            gate_opts["ka_gate_pd_min"] = float(args.ka_gate_pd_min)
        if args.ka_gate_reg_psr_max is not None:
            gate_opts["ka_gate_reg_psr_max"] = float(args.ka_gate_reg_psr_max)
        guard_opts.update(gate_opts)

    debug_coords = parse_coords(args.stap_debug_coord)

    motion_half_span = (
        float(args.motion_half_span_rel)
        if args.motion_half_span_rel and args.motion_half_span_rel > 0
        else None
    )

    base_meta_extra = {
        "source": "replay",
        "orig_run": str(src_root),
        "profile": getattr(args, "profile", None),
        "aperture_phase_std": float(args.aperture_phase_std),
        "aperture_phase_corr_len": float(args.aperture_phase_corr_len),
        "clutter_beta": float(args.clutter_beta),
        "clutter_snr_db": float(args.clutter_snr_db),
        "clutter_mode": str(args.clutter_mode),
        "clutter_rank": int(args.clutter_rank),
        "clutter_depth_min_frac": float(args.clutter_depth_min_frac),
        "clutter_depth_max_frac": float(args.clutter_depth_max_frac),
        "flow_alias_hz": (float(args.flow_alias_hz) if args.flow_alias_hz is not None else None),
        "flow_alias_fraction": float(args.flow_alias_fraction),
        "flow_alias_depth_min_frac": args.flow_alias_depth_min_frac,
        "flow_alias_depth_max_frac": args.flow_alias_depth_max_frac,
        "flow_alias_jitter_hz": float(args.flow_alias_jitter_hz),
        "bg_alias_hz": float(args.bg_alias_hz) if args.bg_alias_hz is not None else None,
        "bg_alias_fraction": float(args.bg_alias_fraction),
        "bg_alias_depth_min_frac": args.bg_alias_depth_min_frac,
        "bg_alias_depth_max_frac": args.bg_alias_depth_max_frac,
        "bg_alias_jitter_hz": float(args.bg_alias_jitter_hz),
        "flow_doppler_min_hz": (
            float(args.flow_doppler_min_hz) if args.flow_doppler_min_hz is not None else None
        ),
        "flow_doppler_max_hz": (
            float(args.flow_doppler_max_hz) if args.flow_doppler_max_hz is not None else None
        ),
        "vibration_hz": float(args.vibration_hz) if args.vibration_hz is not None else None,
        "vibration_amp": float(args.vibration_amp),
        "vibration_depth_min_frac": float(args.vibration_depth_min_frac),
        "vibration_depth_decay_frac": float(args.vibration_depth_decay_frac),
    }

    window_length = args.time_window_length
    window_offsets = args.time_window_offset or []
    window_specs: list[dict[str, int | None | str]] = []
    if window_length is None:
        if window_offsets:
            raise ValueError("--time-window-offset requires --time-window-length")
        window_specs.append({"offset": None, "length": None, "suffix": None})
    else:
        if window_length <= 0:
            raise ValueError("--time-window-length must be positive")
        offsets = window_offsets if window_offsets else [0]
        for idx, offset in enumerate(offsets):
            if offset < 0:
                raise ValueError("time-window offsets must be non-negative")
            if offset + window_length > total_frames_full:
                raise ValueError(
                    "time window offset "
                    f"{offset} + length {window_length} exceeds total frames {total_frames_full}"
                )
            suffix = None if len(offsets) == 1 else f"win{idx}_off{offset}"
            window_specs.append({"offset": offset, "length": window_length, "suffix": suffix})

    total_windows = len(window_specs)
    overall_start = time.time()
    for idx, spec in enumerate(window_specs):
        window_idx = idx + 1
        offset = spec["offset"]
        length = spec["length"]
        suffix = spec["suffix"]
        window_label = suffix or (f"offset{offset}_len{length}" if length is not None else "full")
        slow_time_offset = None
        slow_time_length = None
        if length is None or offset is None:
            angle_sets_window = angle_sets
            pulses_window = pulses_per_set
            meta_extra = dict(base_meta_extra)
        else:
            angle_sets_window = angle_sets
            pulses_window = pulses_per_set
            slow_time_offset = int(offset)
            slow_time_length = int(length)
            meta_extra = dict(base_meta_extra)
            meta_extra["time_window"] = {
                "offset": int(offset),
                "length": int(length),
                "total_length": int(total_frames_full),
            }

        print(
            f"[replay_stap_from_run] ({window_idx}/{total_windows}) "
            f"Processing window {window_label} (frames={slow_time_length or total_frames_full})...",
            flush=True,
        )
        win_start = time.time()

        # KA extra options (including optional fixed beta for blend mode and
        # optional score-space KA v1 risk model parameters).
        ka_opts_extra = dict(guard_opts)
        if args.ka_beta_fixed is not None:
            ka_opts_extra["beta"] = float(args.ka_beta_fixed)
        if args.ka_score_model_json is not None:
            ka_opts_extra["score_model_json"] = str(args.ka_score_model_json)
        if args.ka_score_alpha is not None and args.ka_score_alpha > 0.0:
            ka_opts_extra["score_alpha"] = float(args.ka_score_alpha)
        if args.ka_score_contract_v2 or args.ka_score_contract_v2_force:
            ka_opts_extra["score_contract_v2"] = 1.0
            ka_opts_extra["score_contract_v2_mode"] = str(
                getattr(args, "ka_score_contract_v2_mode", "safety")
            )
            ka_opts_extra["score_contract_v2_proxy_source"] = str(
                getattr(args, "ka_contract_v2_proxy_source", "pd")
            )
            ka_opts_extra["score_contract_v2_alias_source"] = str(
                getattr(args, "ka_contract_v2_alias_source", "peak")
            )
        if args.ka_score_contract_v2_force:
            ka_opts_extra["score_contract_v2_force"] = 1.0

        stap_conditional_mask = None
        if args.stap_conditional_mask is not None:
            stap_conditional_mask = np.load(args.stap_conditional_mask)

        write_acceptance_bundle(
            out_root=out_root,
            g=geom,
            angle_sets=angle_sets_window,
            pulses_per_set=pulses_window,
            prf_hz=prf,
            seed=seed,
            synth_amp_jitter=float(args.synth_amp_jitter),
            synth_phase_jitter=float(args.synth_phase_jitter),
            synth_noise_level=float(args.synth_noise_level),
            synth_shift_max_px=int(args.synth_shift_max_px),
            tile_hw=(int(args.tile_h), int(args.tile_w)),
            tile_stride=int(args.tile_stride),
            Lt=int(args.lt),
            diag_load=float(args.diag_load),
            stap_cov_train_trim_q=float(args.stap_cov_trim_q),
            cov_estimator=str(args.cov_estimator).lower(),
            huber_c=float(args.huber_c),
            stap_debug_samples=int(args.stap_debug_samples),
            fd_span_mode=str(args.fd_span_mode).lower(),
            fd_span_rel=span_bounds,
            fd_fixed_span_hz=args.fd_fixed_span_hz,
            constraint_mode=str(args.constraint_mode).lower(),
            grid_step_rel=float(args.grid_step_rel),
            fd_min_pts=int(args.fd_min_pts),
            fd_max_pts=int(args.max_pts),
            fd_min_abs_hz=float(args.fd_min_abs_hz),
            msd_lambda=args.msd_lambda,
            msd_ridge=float(args.msd_ridge),
            msd_agg_mode=str(args.msd_agg).lower(),
            msd_ratio_rho=float(args.msd_ratio_rho),
            motion_half_span_rel=motion_half_span,
            msd_contrast_alpha=args.msd_contrast_alpha,
            stap_detector_variant=str(getattr(args, "stap_detector_variant", "msd_ratio")),
            tile_debug_limit=args.tile_debug_limit,
            alias_psd_select_enable=bool(args.alias_psd_select),
            alias_psd_select_ratio_thresh=float(args.alias_psd_select_ratio),
            alias_psd_select_bins=max(1, int(args.alias_psd_select_bins)),
            stap_debug_tile_coords=debug_coords,
            ka_mode=str(args.ka_mode).lower(),
            ka_prior_path=str(args.ka_prior_path) if args.ka_prior_path else None,
            ka_beta_bounds=ka_beta_bounds,
            ka_kappa=float(args.ka_kappa),
            ka_alpha=args.ka_alpha,
            ka_target_retain_f=args.ka_target_retain_f,
            ka_target_shrink_perp=args.ka_target_shrink_perp,
            ka_equalize_pf_trace=args.ka_equalize_pf_trace,
            ka_opts_extra=ka_opts_extra,
            mvdr_load_mode=str(args.mvdr_load_mode).lower(),
            mvdr_auto_kappa=float(args.mvdr_auto_kappa),
            constraint_ridge=float(args.constraint_ridge),
            stap_device=args.stap_device,
            meta_extra=meta_extra,
            dataset_suffix=suffix,
            slow_time_offset=slow_time_offset,
            slow_time_length=slow_time_length,
            score_mode=str(args.score_mode),
            flow_mask_mode=str(args.flow_mask_mode).lower(),
            flow_mask_pd_quantile=float(args.flow_mask_pd_quantile),
            flow_mask_depth_min_frac=float(args.flow_mask_depth_min_frac),
            flow_mask_depth_max_frac=float(args.flow_mask_depth_max_frac),
            flow_mask_erode_iters=int(args.flow_mask_erode_iters),
            flow_mask_dilate_iters=int(args.flow_mask_dilate_iters),
            flow_mask_min_pixels=int(args.flow_mask_min_pixels),
            flow_mask_min_coverage_fraction=float(args.flow_mask_min_coverage_frac),
            flow_mask_union_default=bool(args.flow_mask_union_default),
            flow_mask_suppress_alias_depth=bool(args.flow_mask_suppress_alias_depth),
            stap_enable=not bool(getattr(args, "stap_disable", False)),
            stap_conditional_enable=bool(args.stap_conditional_enable),
            stap_conditional_flow_mask=stap_conditional_mask,
            stap_conditional_mask_tag=args.stap_conditional_mask_tag,
            baseline_type=str(args.baseline).lower(),
            baseline_support=str(args.baseline_support),
            reg_enable=bool(args.reg_enable),
            reg_method=str(args.reg_method),
            reg_subpixel=int(args.reg_subpixel),
            reg_reference=str(args.reg_reference),
            svd_rank=args.svd_rank,
            svd_energy_frac=args.svd_energy_frac,
            rpca_enable=bool(args.rpca_enable),
            rpca_lambda=args.rpca_lambda,
            rpca_max_iters=int(args.rpca_max_iters),
            flow_alias_hz=args.flow_alias_hz,
            flow_alias_fraction=float(args.flow_alias_fraction),
            flow_alias_depth_min_frac=args.flow_alias_depth_min_frac,
            flow_alias_depth_max_frac=args.flow_alias_depth_max_frac,
            flow_alias_jitter_hz=float(args.flow_alias_jitter_hz),
            flow_doppler_min_hz=args.flow_doppler_min_hz,
            flow_doppler_max_hz=args.flow_doppler_max_hz,
            flow_doppler_tone_amp=float(args.flow_doppler_tone_amp),
            flow_doppler_noise_amp=float(args.flow_doppler_noise_amp),
            flow_doppler_noise_rho=float(args.flow_doppler_noise_rho),
            flow_doppler_noise_mode=str(args.flow_doppler_noise_mode),
            bg_alias_hz=args.bg_alias_hz,
            bg_alias_fraction=float(args.bg_alias_fraction),
            bg_alias_depth_min_frac=args.bg_alias_depth_min_frac,
            bg_alias_depth_max_frac=args.bg_alias_depth_max_frac,
            bg_alias_jitter_hz=float(args.bg_alias_jitter_hz),
            bg_alias_highrank_mode=None
            if str(args.bg_alias_highrank_mode).lower() == "none"
            else str(args.bg_alias_highrank_mode),
            bg_alias_highrank_deep_patch_coverage=float(args.bg_alias_highrank_coverage),
            bg_alias_highrank_shallow_patch_coverage=float(args.bg_alias_highrank_shallow_coverage),
            bg_alias_highrank_margin_px=int(args.bg_alias_highrank_margin_px),
            bg_alias_highrank_freq_jitter_hz=float(args.bg_alias_highrank_freq_jitter_hz),
            bg_alias_highrank_drift_step_hz=float(args.bg_alias_highrank_drift_step_hz),
            bg_alias_highrank_drift_block_len=(
                int(args.bg_alias_highrank_drift_block_len)
                if args.bg_alias_highrank_drift_block_len is not None
                else None
            ),
            bg_alias_highrank_pf_leak_eta=float(args.bg_alias_highrank_pf_leak_eta),
            bg_alias_highrank_amp=float(args.bg_alias_highrank_amp),
            flow_amp_scale=float(args.flow_amp_scale),
            alias_amp_scale=float(args.alias_amp_scale),
            vibration_hz=args.vibration_hz,
            vibration_amp=float(args.vibration_amp),
            vibration_depth_min_frac=float(args.vibration_depth_min_frac),
            vibration_depth_decay_frac=float(args.vibration_depth_decay_frac),
            aperture_phase_std=float(args.aperture_phase_std),
            aperture_phase_corr_len=float(args.aperture_phase_corr_len),
            aperture_phase_seed=int(args.aperture_phase_seed),
            clutter_beta=float(args.clutter_beta),
            clutter_snr_db=float(args.clutter_snr_db),
            clutter_mode=str(args.clutter_mode),
            clutter_rank=int(args.clutter_rank),
            clutter_depth_min_frac=float(args.clutter_depth_min_frac),
            clutter_depth_max_frac=float(args.clutter_depth_max_frac),
            psd_telemetry=bool(args.psd_telemetry),
            psd_tapers=int(args.psd_tapers),
            psd_bandwidth=float(args.psd_bandwidth),
            band_ratio_mode=str(args.band_ratio_mode),
            band_ratio_tile_mode=str(args.psd_br_tile_mode),
            band_ratio_flow_low_hz=float(args.psd_br_flow_low),
            band_ratio_flow_high_hz=float(args.psd_br_flow_high),
            band_ratio_alias_center_hz=float(args.psd_br_alias_center),
            band_ratio_alias_width_hz=float(args.psd_br_alias_width),
            feasibility_mode=str(args.feasibility_mode),
            hosvd_spatial_downsample=int(args.hosvd_spatial_downsample),
            hosvd_t_sub=args.hosvd_t_sub,
            hosvd_ranks=hosvd_ranks,
            hosvd_energy_fracs=hosvd_energy_fracs,
            flow_mask_default=flow_mask_default,
            bg_mask_default=bg_mask_default,
            micro_vessels=micro_vessels_arr,
            alias_vessels=alias_vessels_arr,
        )
        win_elapsed = time.time() - win_start
        total_elapsed = time.time() - overall_start
        remaining = total_windows - window_idx
        eta = (total_elapsed / window_idx) * remaining if window_idx else 0.0
        print(
            f"[replay_stap_from_run] Completed window {window_label} "
            f"in {win_elapsed/60:.1f} min (ETA {eta/60:.1f} min)",
            flush=True,
        )


if __name__ == "__main__":
    main()
