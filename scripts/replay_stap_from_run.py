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
import os
import time
from pathlib import Path
from typing import List, Sequence, Tuple

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

    meta, geom, angle_sets = load_replay_source(src_root)

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
