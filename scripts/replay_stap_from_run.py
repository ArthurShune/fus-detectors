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
import sys
import time
from pathlib import Path

import numpy as np

from scripts.refactor.replay_bundle_io import build_window_specs, load_replay_source
from scripts.refactor.replay_inputs import (
    load_default_masks_and_vessels,
    parse_hosvd_options,
)
from scripts.refactor.replay_telemetry import (
    build_base_meta_extra,
    build_guard_opts,
    build_ka_opts_extra,
    compose_window_meta_extra,
)
from scripts.refactor.replay_profiles import (
    apply_brain_profile_defaults,
    apply_stap_profile_defaults,
    apply_svd_profile_defaults,
    parse_coords,
)
from scripts.refactor.replay_warmup import run_cuda_warmup, run_heavy_cuda_warmup
from scripts.refactor.replay_write_args import build_write_acceptance_kwargs
from sim.kwave.common import write_acceptance_bundle


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
        "--allow-custom-stap-hyperparams",
        action="store_true",
        help=(
            "When used with --stap-profile clinical, preserve explicit CLI covariance/load "
            "hyperparameters instead of forcing the clinical preset values."
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
        choices=["msd_ratio", "whitened_power", "unwhitened_ratio", "hybrid_rescue", "adaptive_guard"],
        help=(
            "Detector statistic exported as score_stap_preka.npy. "
            "'msd_ratio' (default) is the whitened matched-subspace ratio; "
            "'whitened_power' is total whitened slow-time power (no Doppler band partition); "
            "'unwhitened_ratio' disables covariance whitening (R=I) while keeping the same band partition; "
            "'hybrid_rescue' runs the advanced whitened score with an unwhitened rescue branch on selected pixels; "
            "'adaptive_guard' uses the unwhitened score by default and promotes clutter-heavy tiles onto the advanced "
            "whitened branch when baseline guard energy indicates structured clutter."
        ),
    )
    ap.add_argument(
        "--hybrid-rescue-rule",
        type=str,
        default="guard_frac_v1",
        choices=[
            "guard_frac_v1",
            "alias_rescue_v1",
            "band_ratio_v1",
            "guard_promote_v1",
            "guard_promote_tile_v1",
        ],
        help=(
            "Pixelwise routing rule for --stap-detector-variant=hybrid_rescue. "
            "The rule uses baseline telemetry to decide where to keep the whitened score "
            "and where to rescue with the unwhitened matched-subspace score."
        ),
    )
    ap.add_argument(
        "--stap-whiten-gamma",
        type=float,
        default=1.0,
        help=(
            "Fractional whitening exponent for the STAP-family detector. "
            "gamma=0 gives the unwhitened ratio, gamma=1 the current full STAP score."
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


def main() -> None:
    args = parse_args()
    reg_override_disable = "--reg-disable" in sys.argv
    reg_override_enable = "--reg-enable" in sys.argv
    custom_stap_overrides = {
        "diag_load": args.diag_load,
        "stap_cov_trim_q": args.stap_cov_trim_q,
        "cov_estimator": args.cov_estimator,
        "huber_c": args.huber_c,
        "constraint_ridge": args.constraint_ridge,
        "mvdr_auto_kappa": args.mvdr_auto_kappa,
        "mvdr_load_mode": args.mvdr_load_mode,
    }

    apply_brain_profile_defaults(args)
    if reg_override_disable:
        args.reg_enable = False
    elif reg_override_enable:
        args.reg_enable = True
    apply_stap_profile_defaults(args)
    if bool(args.allow_custom_stap_hyperparams):
        for key, value in custom_stap_overrides.items():
            setattr(args, key, value)

    run_cuda_warmup(args)

    apply_svd_profile_defaults(args)

    src_root = args.src
    out_root = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    meta, geom, angle_sets = load_replay_source(src_root)

    run_heavy_cuda_warmup(args, geom=geom)

    flow_mask_default, bg_mask_default, micro_vessels_arr, alias_vessels_arr = (
        load_default_masks_and_vessels(src_root=src_root, geom=geom)
    )
    hosvd_ranks, hosvd_energy_fracs = parse_hosvd_options(
        hosvd_ranks_arg=args.hosvd_ranks,
        hosvd_energy_fracs_arg=args.hosvd_energy_fracs,
    )
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

    guard_opts = build_guard_opts(args)

    debug_coords = parse_coords(args.stap_debug_coord)

    motion_half_span = (
        float(args.motion_half_span_rel)
        if args.motion_half_span_rel and args.motion_half_span_rel > 0
        else None
    )

    base_meta_extra = build_base_meta_extra(args, src_root=src_root)

    window_length = args.time_window_length
    window_offsets = args.time_window_offset or []
    window_specs = build_window_specs(
        window_length=window_length,
        window_offsets=window_offsets,
        total_frames_full=total_frames_full,
    )

    total_windows = len(window_specs)
    overall_start = time.time()
    for idx, spec in enumerate(window_specs):
        window_idx = idx + 1
        offset = spec.offset
        length = spec.length
        suffix = spec.suffix
        window_label = spec.label
        slow_time_offset = int(offset) if offset is not None else None
        slow_time_length = int(length) if length is not None else None
        angle_sets_window = angle_sets
        pulses_window = pulses_per_set
        meta_extra = compose_window_meta_extra(
            base_meta_extra=base_meta_extra,
            offset=offset,
            length=length,
            total_frames_full=total_frames_full,
        )

        print(
            f"[replay_stap_from_run] ({window_idx}/{total_windows}) "
            f"Processing window {window_label} (frames={slow_time_length or total_frames_full})...",
            flush=True,
        )
        win_start = time.time()

        ka_opts_extra = build_ka_opts_extra(args, guard_opts=guard_opts)

        stap_conditional_mask = None
        if args.stap_conditional_mask is not None:
            stap_conditional_mask = np.load(args.stap_conditional_mask)

        write_kwargs = build_write_acceptance_kwargs(
            args,
            out_root=out_root,
            geom=geom,
            angle_sets_window=angle_sets_window,
            pulses_window=pulses_window,
            prf=prf,
            seed=seed,
            span_bounds=span_bounds,
            motion_half_span=motion_half_span,
            debug_coords=debug_coords,
            ka_beta_bounds=ka_beta_bounds,
            ka_opts_extra=ka_opts_extra,
            meta_extra=meta_extra,
            suffix=suffix,
            slow_time_offset=slow_time_offset,
            slow_time_length=slow_time_length,
            flow_mask_default=flow_mask_default,
            bg_mask_default=bg_mask_default,
            micro_vessels_arr=micro_vessels_arr,
            alias_vessels_arr=alias_vessels_arr,
            hosvd_ranks=hosvd_ranks,
            hosvd_energy_fracs=hosvd_energy_fracs,
            stap_conditional_mask=stap_conditional_mask,
        )
        write_acceptance_bundle(**write_kwargs)
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
