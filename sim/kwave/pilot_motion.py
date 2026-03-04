# sim/kwave/pilot_motion.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from sim.kwave.common import (  # noqa: E402
    AngleData,
    SimGeom,
    run_angle_once,
    write_acceptance_bundle,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="k-Wave motion-stress pilot: multiple ensembles with jittered steering."
    )
    ap.add_argument("--Nx", type=int, default=256)
    ap.add_argument("--Ny", type=int, default=256)
    ap.add_argument("--dx", type=float, default=90e-6)
    ap.add_argument("--dy", type=float, default=90e-6)
    ap.add_argument("--c0", type=float, default=1540.0)
    ap.add_argument("--rho0", type=float, default=1000.0)
    ap.add_argument(
        "--pml-size",
        type=int,
        default=20,
        help="PML thickness (grid points) for k-Wave simulation (default 20).",
    )
    ap.add_argument(
        "--angles",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Steering angles in degrees; accepts a comma-separated string or a "
            "space-separated list."
        ),
    )
    ap.add_argument("--f0", type=float, default=7.5e6)
    ap.add_argument("--ncycles", type=int, default=3)
    ap.add_argument("--ensembles", type=int, default=5)
    ap.add_argument(
        "--jitter_um",
        type=float,
        default=15.0,
        help="Std dev of steering jitter (microns converted to angle).",
    )
    ap.add_argument(
        "--pulses",
        type=int,
        default=48,
        help="Synthetic slow-time pulses per ensemble.",
    )
    ap.add_argument("--prf", type=float, default=3000.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip k-Wave and generate synthetic RF ensembles.",
    )
    ap.add_argument("--tile-h", type=int, default=12)
    ap.add_argument("--tile-w", type=int, default=12)
    ap.add_argument("--tile-stride", type=int, default=6)
    ap.add_argument("--lt", type=int, default=4)
    ap.add_argument(
        "--regime",
        type=str,
        default="generic",
        choices=["generic", "hab_v2", "hab_v3_skull", "hab_contract"],
        help=(
            "Optional preset for regime-specific defaults. "
            "Use 'hab_v2' for the original Hybrid Alias Brain configuration, "
            "'hab_contract' for a HAB-style contract regime with alias separation "
            "and STAP headroom, or 'hab_v3_skull' for a skull/OR-inspired HAB "
            "variant with additional alias, clutter, and motion realism."
        ),
    )
    ap.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["Brain-OpenSkull", "Brain-AliasContract", "Brain-SkullOR", "Brain-Pial128"],
        help=(
            "Optional high-level brain fUS profile tag. When set to Brain-AliasContract "
            "or Brain-SkullOR and --regime is left at 'generic', the corresponding HAB "
            "regime preset (hab_contract or hab_v3_skull) is selected; the value is "
            "also recorded in the acceptance bundle metadata."
        ),
    )
    ap.add_argument("--diag-load", type=float, default=1e-2)
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--huber-c", type=float, default=5.0)
    ap.add_argument("--stap-debug-samples", type=int, default=0)
    ap.add_argument(
        "--fd-span-mode",
        type=str,
        default="psd",
        choices=["psd", "auto", "fixed"],
    )
    ap.add_argument(
        "--fd-span-rel",
        type=str,
        default="0.30,1.10",
        help="Min,max relative span to PRF/Lt (psd/auto).",
    )
    ap.add_argument(
        "--fd-fixed-span-hz",
        type=float,
        default=None,
        help="Override absolute span (Hz) when fd-span-mode=fixed.",
    )
    ap.add_argument(
        "--constraint-mode",
        type=str,
        default="exp+deriv",
        choices=["exp", "exp+deriv"],
        help="Constraint mode for temporal band-pass projector.",
    )
    ap.add_argument("--grid-step-rel", type=float, default=0.12)
    ap.add_argument(
        "--max-pts",
        type=int,
        default=5,
        help="Maximum Doppler tones (odd) for the LCMV constraint grid.",
    )
    ap.add_argument(
        "--mvdr-load-mode",
        type=str,
        default="auto",
        choices=["auto", "absolute"],
        help="Diagonal loading mode for the LCMV solver.",
    )
    ap.add_argument(
        "--mvdr-auto-kappa",
        type=float,
        default=30.0,
        help="Target condition number when mvdr-load-mode=auto.",
    )
    ap.add_argument(
        "--constraint-ridge",
        type=float,
        default=0.15,
        help="Constraint ridge (δ) added to the Gram matrix.",
    )
    ap.add_argument(
        "--msd-agg",
        type=str,
        default="median",
        choices=["mean", "median", "trim10"],
        help="Aggregation mode for temporal MSD ratio (default median).",
    )
    ap.add_argument(
        "--msd-lambda",
        type=float,
        default=6e-2,
        help="Absolute diagonal loading (λ) for MSD ratio scoring (None => match PD λ).",
    )
    ap.add_argument(
        "--msd-ridge",
        type=float,
        default=0.15,
        help="Ridge term added to the MSD constraint Gram matrix.",
    )
    ap.add_argument(
        "--msd-ratio-rho",
        type=float,
        default=0.05,
        help="Shrinkage factor ρ applied to the MSD ratio denominator to tame heavy tails.",
    )
    ap.add_argument(
        "--motion-half-span-rel",
        type=float,
        default=0.20,
        help=(
            "Relative half-span (w.r.t. PRF/Lt) treated as motion band; set <=0 to disable"
            " motion contrast."
        ),
    )
    ap.add_argument(
        "--msd-contrast-alpha",
        type=float,
        default=0.8,
        help="Contrast weight between flow and motion bands (<=0 disables motion contrast).",
    )
    ap.add_argument(
        "--ka-mode",
        type=str,
        default="none",
        choices=["none", "analytic", "library"],
        help="KA prior mode for temporal whitening (default none).",
    )
    ap.add_argument(
        "--ka-prior-path",
        type=str,
        default=None,
        help="Path to a saved KA prior matrix (required when --ka-mode=library).",
    )
    ap.add_argument(
        "--ka-beta-bounds",
        type=str,
        default="0.05,0.50",
        help="Comma-separated min,max bounds for the KA beta blend weight.",
    )
    ap.add_argument(
        "--ka-kappa",
        type=float,
        default=40.0,
        help="Target condition number for KA-conditioned whitening.",
    )
    ap.add_argument(
        "--ka-alpha",
        type=float,
        default=None,
        help="Optional LW-style alpha toward identity before KA prior blend.",
    )
    ap.add_argument(
        "--ka-directional-beta",
        action="store_true",
        help="Enable directional beta with passband retention/shrink constraints.",
    )
    ap.add_argument(
        "--ka-target-retain-f",
        type=float,
        default=None,
        help="Target fraction of passband energy to retain (e.g., 0.9).",
    )
    ap.add_argument(
        "--ka-target-shrink-perp",
        type=float,
        default=None,
        help="Target upper bound on off-band energy fraction (e.g., 0.95).",
    )
    ap.add_argument(
        "--ka-ridge-split",
        action="store_true",
        help="Apply absolute ridge on the complement of the passband only.",
    )
    ap.add_argument(
        "--ka-lambda-override-split",
        type=float,
        default=None,
        help="Optional λ to apply on complement (overrides auto-conditioned value)",
    )
    ap.add_argument(
        "--alias-psd-select-enable",
        action="store_true",
        help="Enable Doppler-grid alias pruning based on PSD alias ratio (KA mode only).",
    )
    ap.add_argument(
        "--alias-psd-select-ratio",
        type=float,
        default=1.15,
        help="Minimum flow/alias PSD ratio required before keeping only narrow tone bins.",
    )
    ap.add_argument(
        "--alias-psd-select-bins",
        type=int,
        default=1,
        help="Number of fundamental-multiple bins to retain when alias selection fires.",
    )
    ap.add_argument(
        "--band-ratio-mode",
        type=str,
        default="whitened",
        choices=["legacy", "whitened"],
        help="Band-ratio flavor to record (legacy PD ratio vs. whitened multi-taper).",
    )
    ap.add_argument(
        "--band-ratio-tile-mode",
        type=str,
        default="mean",
        choices=[
            "mean",
            "incoherent",
            "incoherent_max",
            "incoherent_q90",
            "incoherent_q95",
        ],
        help=(
            "Tile PSD for band-ratio telemetry: 'mean' (coherent tile-mean series), "
            "'incoherent' (avg per-pixel PSD), 'incoherent_max' (max per-pixel PSD), "
            "or 'incoherent_qNN' (e.g. incoherent_q90)."
        ),
    )
    ap.add_argument(
        "--band-ratio-flow-low-hz",
        type=float,
        default=120.0,
        help="Lower bound of the flow band for band-ratio scoring (Hz).",
    )
    ap.add_argument(
        "--band-ratio-flow-high-hz",
        type=float,
        default=400.0,
        help="Upper bound of the flow band for band-ratio scoring (Hz).",
    )
    ap.add_argument(
        "--band-ratio-alias-center-hz",
        type=float,
        default=900.0,
        help="Center frequency of the alias band for band-ratio scoring (Hz).",
    )
    ap.add_argument(
        "--band-ratio-alias-width-hz",
        type=float,
        default=15.625,
        help="Half-width of the alias band for band-ratio scoring (Hz).",
    )
    ap.add_argument(
        "--psd-telemetry",
        action="store_true",
        help="Emit multi-taper PSD telemetry and band-ratio tile maps.",
    )
    ap.add_argument(
        "--psd-tapers",
        type=int,
        default=3,
        help="Number of DPSS tapers for the PSD estimator when telemetry is enabled.",
    )
    ap.add_argument(
        "--psd-bandwidth",
        type=float,
        default=2.0,
        help="DPSS time-half-bandwidth product for PSD telemetry (>=1).",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default="band_ratio_whitened",
        choices=["msd", "pd", "band_ratio", "band_ratio_whitened"],
        help="Which score map to treat as default in the acceptance bundle.",
    )
    ap.add_argument(
        "--ka-gate-enable",
        action="store_true",
        help="Enable flow-aware KA gating (alias/flow/depth/PD/registration checks).",
    )
    ap.add_argument(
        "--ka-gate-alias-rmin",
        type=float,
        default=1.10,
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
        default=0.20,
        help="Lower depth fraction bound for KA gating (0-1).",
    )
    ap.add_argument(
        "--ka-gate-depth-max-frac",
        type=float,
        default=0.95,
        help="Upper depth fraction bound for KA gating (0-1).",
    )
    ap.add_argument(
        "--ka-gate-pd-min",
        type=float,
        default=0.90,
        help="Minimum normalized PD metric for a tile to allow KA action.",
    )
    ap.add_argument(
        "--ka-gate-reg-psr-max",
        type=float,
        default=6.0,
        help="Upper bound on registration PSR for KA gating (None disables this check).",
    )
    ap.add_argument(
        "--flow-alias-hz",
        type=float,
        default=None,
        help=(
            "Inject a synthetic alias component at the given Doppler frequency (Hz) "
            "within the flow mask."
        ),
    )
    ap.add_argument(
        "--flow-alias-fraction",
        type=float,
        default=0.4,
        help="Fraction of the default flow mask to modulate when --flow-alias-hz is set (0-1).",
    )
    ap.add_argument(
        "--flow-alias-depth-min-frac",
        type=float,
        default=0.25,
        help="Limit alias injection to rows deeper than this fraction of depth (None disables).",
    )
    ap.add_argument(
        "--flow-alias-depth-max-frac",
        type=float,
        default=0.4,
        help=(
            "Limit alias injection to rows shallower than this fraction of depth "
            "(None disables)."
        ),
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
        help="Inject a synthetic alias component at this frequency (Hz) on the background mask.",
    )
    ap.add_argument(
        "--bg-alias-fraction",
        type=float,
        default=0.3,
        help="Fraction of the background mask to modulate when --bg-alias-hz is set (0-1).",
    )
    ap.add_argument(
        "--bg-alias-depth-min-frac",
        type=float,
        default=0.30,
        help="Lower depth fraction bound for background alias injection (0-1).",
    )
    ap.add_argument(
        "--bg-alias-depth-max-frac",
        type=float,
        default=0.70,
        help="Upper depth fraction bound for background alias injection (0-1).",
    )
    ap.add_argument(
        "--bg-alias-jitter-hz",
        type=float,
        default=50.0,
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
            "random walk is applied across blocks within a replay window."
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
        "--aperture-phase-std",
        type=float,
        default=0.0,
        help="RMS phase screen (rad) applied per angle across the aperture (0 disables).",
    )
    ap.add_argument(
        "--aperture-phase-corr-len",
        type=float,
        default=12.0,
        help="Correlation length (elements) for aperture phase screen (ignored if std=0).",
    )
    ap.add_argument(
        "--aperture-phase-seed",
        type=int,
        default=111,
        help="Additional RNG seed offset for aperture phase screen synthesis.",
    )
    ap.add_argument(
        "--skull-phase-std",
        type=float,
        default=0.0,
        help=(
            "RMS (rad) of a 2D skull slab phase screen applied on the beamformed cube "
            "(0 disables)."
        ),
    )
    ap.add_argument(
        "--skull-phase-corr-lat",
        type=float,
        default=6.0,
        help="Lateral correlation length (pixels/elements) for the skull slab phase screen.",
    )
    ap.add_argument(
        "--skull-phase-corr-depth-frac",
        type=float,
        default=0.08,
        help="Depth correlation length (fraction of depth) for the skull slab phase screen.",
    )
    ap.add_argument(
        "--skull-depth-max-frac",
        type=float,
        default=0.35,
        help="Maximum depth fraction for the skull slab phase screen.",
    )
    ap.add_argument(
        "--skull-guided-hz",
        type=float,
        default=0.0,
        help="Frequency (Hz) of a shallow guided-wave contaminant (0 disables).",
    )
    ap.add_argument(
        "--skull-guided-amp",
        type=float,
        default=0.0,
        help="Relative amplitude of the shallow guided-wave contaminant (0 disables).",
    )
    ap.add_argument(
        "--skull-guided-depth-max-frac",
        type=float,
        default=0.20,
        help="Maximum depth fraction for the shallow guided-wave contaminant.",
    )
    ap.add_argument(
        "--skull-guided-lat-corr",
        type=float,
        default=15.0,
        help="Lateral correlation length (pixels/elements) for the guided-wave contaminant.",
    )
    ap.add_argument(
        "--resid-shift-std-px",
        type=float,
        default=0.0,
        help=(
            "Std dev (pixels) of residual rigid motion applied frame-wise on the "
            "beamformed cube (models incomplete motion correction)."
        ),
    )
    ap.add_argument(
        "--clutter-beta",
        type=float,
        default=0.0,
        help="Temporal clutter 1/f^beta slope (<=0 disables colored clutter injection).",
    )
    ap.add_argument(
        "--clutter-snr-db",
        type=float,
        default=-6.0,
        help="Temporal clutter SNR (dB) relative to background (ignored if beta<=0).",
    )
    ap.add_argument(
        "--clutter-depth-min-frac",
        type=float,
        default=0.20,
        help="Minimum depth fraction for injected clutter (0 disables lower clamp).",
    )
    ap.add_argument(
        "--clutter-depth-max-frac",
        type=float,
        default=0.95,
        help="Maximum depth fraction for injected clutter (>=1 disables upper clamp).",
    )
    ap.add_argument(
        "--hemo-mod-amp",
        type=float,
        default=0.0,
        help=(
            "Relative amplitude of slow hemodynamic modulation applied on the flow mask "
            "(0 disables; typical skull/OR values 0.2-0.4)."
        ),
    )
    ap.add_argument(
        "--hemo-mod-breath-period",
        type=float,
        default=3.0,
        help="Period (seconds) of the respiratory component in hemodynamic modulation.",
    )
    ap.add_argument(
        "--hemo-mod-card-period",
        type=float,
        default=0.8,
        help="Period (seconds) of the cardiac component in hemodynamic modulation.",
    )
    ap.add_argument(
        "--depth-amp-profile",
        type=str,
        default=None,
        help=(
            "Optional depth-dependent amplitude profile applied after beamforming "
            "(e.g. 'skull_or_v1' for skull/OR-like attenuation and SNR)."
        ),
    )
    ap.add_argument(
        "--depth-amp-min-factor",
        type=float,
        default=0.3,
        help=(
            "Minimum relative amplitude at the deepest depth when --depth-amp-profile "
            "is set (skull/OR regimes typically use 0.3-0.5)."
        ),
    )
    ap.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution even if GPU binaries are available.",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run STAP processing on (default auto).",
    )
    ap.add_argument(
        "--feasibility-mode",
        type=str,
        default="legacy",
        choices=["legacy", "updated"],
        help="Feasibility configuration applied to STAP/KA processing.",
    )
    ap.add_argument(
        "--baseline-type",
        type=str,
        default="mc_svd",
        choices=["svd", "mc_svd"],
        help="Baseline type used for acceptance bundle (must be mc_svd for whitened scores).",
    )
    ap.add_argument(
        "--flow-amp-scale",
        type=float,
        default=1.0,
        help=(
            "Optional multiplicative scale applied to the complex flow-mask "
            "amplitude before synthetic Doppler/alias/clutter injection."
        ),
    )
    ap.add_argument(
        "--flow-amp-shallow-scale",
        type=float,
        default=1.0,
        help=(
            "Optional depth-varying flow amplitude scale in the shallow band "
            "(used with --flow-depth-mid-frac/--flow-depth-deep-frac to mimic "
            "skull/OR SNR variation)."
        ),
    )
    ap.add_argument(
        "--flow-amp-mid-scale",
        type=float,
        default=1.0,
        help="Optional depth-varying flow amplitude scale in the mid-depth band.",
    )
    ap.add_argument(
        "--flow-amp-deep-scale",
        type=float,
        default=1.0,
        help="Optional depth-varying flow amplitude scale in the deep band.",
    )
    ap.add_argument(
        "--flow-depth-mid-frac",
        type=float,
        default=0.45,
        help="Depth fraction separating shallow/mid flow amplitude bands.",
    )
    ap.add_argument(
        "--flow-depth-deep-frac",
        type=float,
        default=0.75,
        help="Depth fraction separating mid/deep flow amplitude bands.",
    )
    ap.add_argument(
        "--alias-amp-scale",
        type=float,
        default=1.0,
        help=(
            "Optional multiplicative scale applied to the aliased component "
            "on the flow mask when alias injection is enabled."
        ),
    )
    ap.add_argument("--out", type=str, default="runs/pilot/r2")
    args = ap.parse_args()

    # Optional Brain-* profile tag: when used together with the default
    # regime='generic', map Brain-AliasContract and Brain-SkullOR to their
    # corresponding HAB regime presets so that the simulator and methodology
    # share the same high-level vocabulary.
    if args.profile == "Brain-AliasContract" and args.regime == "generic":
        args.regime = "hab_contract"
    if args.profile == "Brain-SkullOR" and args.regime == "generic":
        args.regime = "hab_v3_skull"

    # Normalize angles: accept comma-separated or space-separated lists.
    raw_angles = args.angles or ["-12,-6,0,6,12"]
    angles_list: List[float] = []
    for token in raw_angles:
        if token is None:
            continue
        parts = str(token).split(",")
        for p in parts:
            p_clean = p.strip()
            if not p_clean:
                continue
            try:
                angles_list.append(float(p_clean))
            except ValueError as exc:
                raise ValueError(f"Invalid angle value '{p_clean}'") from exc
    if not angles_list:
        raise ValueError("No steering angles provided after parsing --angles.")
    args.angles = ",".join(str(a) for a in angles_list)

    # Optional HAB presets: override subsets of synthesis knobs to construct
    # hybrid alias brain regimes while keeping the core geometry (grid, f0, PRF,
    # angles, T) unchanged. These affect only the synthetic flow/alias/clutter/
    # vibration layers and aperture phase screen.
    if args.regime in {"hab_v2", "hab_v3_skull", "hab_contract"}:
        # Microvascular flow Doppler range (Hz) and relative flow amplitude.
        if args.flow_doppler_min_hz is None:
            args.flow_doppler_min_hz = 40.0
        if args.flow_doppler_max_hz is None:
            args.flow_doppler_max_hz = 220.0
        if args.flow_amp_scale == 1.0:
            args.flow_amp_scale = 2.0

        # Pial-like shallow alias layer on background (Zone A).
        if args.bg_alias_hz is None:
            args.bg_alias_hz = 650.0
        if args.bg_alias_fraction == 0.3:
            args.bg_alias_fraction = 0.75
        if args.bg_alias_depth_min_frac == 0.30:
            args.bg_alias_depth_min_frac = 0.12
        if args.bg_alias_depth_max_frac == 0.70:
            args.bg_alias_depth_max_frac = 0.28
        if args.bg_alias_jitter_hz == 50.0:
            args.bg_alias_jitter_hz = 120.0

        # Deep alias component on flow support (Zones B/C). For the original
        # HAB-v2 and skull/OR HAB-v3 regimes we allow a small fraction of flow
        # support to carry alias energy. For the KA-contract HAB regime
        # ('hab_contract') we explicitly keep flow tiles Pf-dominant by
        # disabling flow-alias injection and relying only on background alias.
        if args.regime in {"hab_v2", "hab_v3_skull"}:
            if args.flow_alias_hz is None:
                args.flow_alias_hz = 650.0
            if args.flow_alias_fraction == 0.4:
                args.flow_alias_fraction = 0.15
            if args.flow_alias_depth_min_frac == 0.25:
                args.flow_alias_depth_min_frac = 0.28
            if args.flow_alias_depth_max_frac == 0.4:
                args.flow_alias_depth_max_frac = 0.85
            if args.flow_alias_jitter_hz == 0.0:
                args.flow_alias_jitter_hz = 80.0
        elif args.regime == "hab_contract":
            # Keep flow alias disabled unless explicitly overridden by the user.
            if args.flow_alias_fraction == 0.4:
                args.flow_alias_fraction = 0.0
            # If the user did not override flow_alias_hz, leave it as None so
            # that only background alias is injected.
            if args.flow_alias_hz is None:
                args.flow_alias_hz = None

        # Global vibration in a Pa-adjacent band.
        if args.vibration_hz is None:
            args.vibration_hz = 450.0
        if args.vibration_amp == 0.0:
            args.vibration_amp = 0.30
        if args.vibration_depth_min_frac == 0.15:
            args.vibration_depth_min_frac = 0.12
        if args.vibration_depth_decay_frac == 0.25:
            args.vibration_depth_decay_frac = 0.30

        # Temporal clutter: 1/f^beta with depth-varying support and higher SNR.
        if args.clutter_beta == 0.0:
            args.clutter_beta = 1.0
        if args.clutter_snr_db == -6.0:
            args.clutter_snr_db = 25.0
        # For the HAB-contract regime, slightly emphasize alias relative to clutter
        # by increasing the background alias amplitude and fraction while keeping
        # clutter SNR in a realistic range.
        if args.regime == "hab_contract":
            if args.bg_alias_fraction == 0.75:
                args.bg_alias_fraction = 0.9
            if args.alias_amp_scale == 1.0:
                args.alias_amp_scale = 1.5
        # Keep existing depth range defaults (0.20, 0.95) unless explicitly overridden.

        # Skull-like aperture phase aberration (per-angle phase screen).
        if args.aperture_phase_std == 0.0:
            args.aperture_phase_std = 1.2
        if args.aperture_phase_corr_len == 12.0:
            args.aperture_phase_corr_len = 10.0

        # Whitened band-ratio design centered on the HAB alias band.
        if args.band_ratio_mode == "legacy":
            args.band_ratio_mode = "whitened"
        if args.band_ratio_flow_low_hz == 120.0:
            args.band_ratio_flow_low_hz = 60.0
        if args.band_ratio_flow_high_hz == 400.0:
            args.band_ratio_flow_high_hz = 200.0
        if args.band_ratio_alias_center_hz == 900.0:
            args.band_ratio_alias_center_hz = 650.0
        if args.band_ratio_alias_width_hz == 15.625:
            args.band_ratio_alias_width_hz = 140.0

        # Skull/OR HAB variant: enable depth-dependent attenuation by default.
        if args.regime == "hab_v3_skull" and args.depth_amp_profile is None:
            args.depth_amp_profile = "skull_or_v1"
        # Skull/OR HAB variant: modest defaults for skull slab and guided-wave
        # components if the user did not override them.
        if args.regime == "hab_v3_skull":
            if args.skull_phase_std == 0.0:
                args.skull_phase_std = 1.0
            if args.skull_guided_hz == 0.0:
                args.skull_guided_hz = 900.0
            if args.skull_guided_amp == 0.0:
                args.skull_guided_amp = 0.25

        # HAB-v3 skull/OR variant will add further realism patches (depth-varying
        # attenuation, residual motion, PRF jitter, broadened spectra) in
        # subsequent steps. For now, it shares the HAB-v2 spectral/alias defaults
        # but is tagged separately via --regime so that downstream scripts can
        # treat it as a distinct scenario.

    try:
        ka_beta_bounds = tuple(float(v) for v in args.ka_beta_bounds.split(","))
    except ValueError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(f"Invalid --ka-beta-bounds value {args.ka_beta_bounds!r}") from exc
    if len(ka_beta_bounds) != 2:
        raise SystemExit("--ka-beta-bounds must provide two comma-separated floats (min,max).")
    if ka_beta_bounds[0] < 0 or ka_beta_bounds[1] <= ka_beta_bounds[0]:
        raise SystemExit("Require 0 <= min_beta < max_beta for --ka-beta-bounds.")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    ka_opts_extra: dict[str, float | bool] = {}
    if args.ka_gate_enable:
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
        ka_opts_extra.update(gate_opts)

    geom = SimGeom(
        Nx=args.Nx,
        Ny=args.Ny,
        dx=args.dx,
        dy=args.dy,
        c0=args.c0,
        rho0=args.rho0,
        f0=args.f0,
        ncycles=args.ncycles,
    )
    geom.pml_size = int(args.pml_size)

    base_angles: List[float] = [float(s) for s in args.angles.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    # Convert a lateral micro-jitter scale (microns) into an angular steering jitter.
    # Small-angle approx: dtheta ~= dx / depth. Use the canonical flow ROI depth
    # (~60% of imaging depth) so the jitter magnitude is physically plausible.
    depth_m = max(1e-6, 0.6 * float(geom.Ny) * float(geom.dy))
    jitter_rad_std = (float(args.jitter_um) * 1e-6) / depth_m

    angle_sets = []
    tensor_list = []
    angles_used_meta = []

    total_ensembles = max(args.ensembles, 1)
    angles_per_set = len(base_angles)
    total_angle_shots = total_ensembles * max(angles_per_set, 1)
    completed_shots = 0

    for ens in range(total_ensembles):
        current_set: List[AngleData] = []
        rf_list = []
        used_angles = []
        for idx, base_angle in enumerate(base_angles):
            dtheta = rng.normal(scale=jitter_rad_std)
            angle_used = float(base_angle + np.rad2deg(dtheta))
            print(
                f"[pilot_motion] ensemble {ens + 1}/{total_ensembles} "
                f"angle {idx + 1}/{max(angles_per_set,1)} -> {angle_used:.2f}°",
                flush=True,
            )
            angle_dir = out_root / f"ens{ens}_angle_{int(round(base_angle))}"
            angle_dir.mkdir(parents=True, exist_ok=True)
            if args.synthetic:
                Nt = max(128, geom.Ny * 2)
                dt = geom.cfl * min(geom.dx, geom.dy) / max(geom.c0, 1.0)
                rf = rng.standard_normal((Nt, geom.Nx)).astype(np.float32)
                rf += 0.25 * rng.standard_normal((Nt, geom.Nx)).astype(np.float32)
                t = np.arange(Nt, dtype=np.float32) * dt
                mod = 0.04 * (ens + 1) + 0.01 * (idx + 1)
                rf += 0.08 * np.sin(2.0 * np.pi * mod * t)[:, None]
                res = AngleData(angle_deg=angle_used, rf=rf, dt=float(dt))
            else:
                res = run_angle_once(angle_dir, angle_used, geom, use_gpu=not args.force_cpu)
            np.save(angle_dir / "rf.npy", res.rf.astype(np.float32), allow_pickle=False)
            np.save(angle_dir / "dt.npy", np.array(res.dt, dtype=np.float32), allow_pickle=False)
            current_set.append(res)
            rf_list.append(res.rf)
            used_angles.append(res.angle_deg)
            completed_shots += 1
            print(
                f"[pilot_motion]   captured {completed_shots}/{total_angle_shots} angles",
                flush=True,
            )
        if current_set:
            angle_sets.append(current_set)
            tensor_list.append(np.stack(rf_list, axis=0).astype(np.float32))
            angles_used_meta.append(used_angles)

    if tensor_list:
        big = np.stack(tensor_list, axis=0)
    else:
        big = np.empty((0,), dtype=np.float32)
    np.save(out_root / "rf_tensor.npy", big, allow_pickle=False)

    meta = {
        "geometry": {
            "Nx": geom.Nx,
            "Ny": geom.Ny,
            "dx": geom.dx,
            "dy": geom.dy,
            "c0": geom.c0,
            "rho0": geom.rho0,
            "pml": geom.pml_size,
            "cfl": geom.cfl,
        },
        "base_angles_deg": base_angles,
        "angles_used_deg": angles_used_meta,
        "f0_hz": geom.f0,
        "ncycles": geom.ncycles,
        "ensembles": args.ensembles,
        "jitter_um": args.jitter_um,
        "seed": args.seed,
        "pulses_per_ensemble": args.pulses,
        "prf_hz": args.prf,
        "synthetic": bool(args.synthetic),
    }
    with open(out_root / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    span_bounds = tuple(float(x.strip()) for x in args.fd_span_rel.split(",") if x.strip())
    if len(span_bounds) != 2:
        raise ValueError("--fd-span-rel must be 'min,max'")
    ratio_rho = max(0.0, float(args.msd_ratio_rho))
    motion_half_span_rel = None
    if args.motion_half_span_rel is not None and args.motion_half_span_rel > 0.0:
        motion_half_span_rel = float(args.motion_half_span_rel)
    contrast_alpha = None
    if args.msd_contrast_alpha is not None and args.msd_contrast_alpha > 0.0:
        contrast_alpha = float(args.msd_contrast_alpha)

    bundle_paths = write_acceptance_bundle(
        out_root=out_root,
        g=geom,
        angle_sets=angle_sets,
        pulses_per_set=args.pulses,
        prf_hz=args.prf,
        seed=args.seed,
        tile_hw=(int(args.tile_h), int(args.tile_w)),
        tile_stride=int(args.tile_stride),
        Lt=int(args.lt),
        diag_load=float(args.diag_load),
        cov_estimator=str(args.cov_estimator).lower(),
        huber_c=float(args.huber_c),
        msd_lambda=args.msd_lambda,
        msd_ridge=float(args.msd_ridge),
        msd_agg_mode=str(args.msd_agg).lower(),
        msd_ratio_rho=ratio_rho,
        motion_half_span_rel=motion_half_span_rel,
        msd_contrast_alpha=contrast_alpha,
        ka_mode=str(args.ka_mode).lower(),
        ka_prior_path=args.ka_prior_path,
        ka_beta_bounds=ka_beta_bounds,
        ka_kappa=float(args.ka_kappa),
        ka_alpha=args.ka_alpha,
        ka_directional_beta=bool(args.ka_directional_beta),
        ka_target_retain_f=args.ka_target_retain_f,
        ka_target_shrink_perp=args.ka_target_shrink_perp,
        ka_ridge_split=bool(args.ka_ridge_split),
        ka_lambda_override_split=args.ka_lambda_override_split,
        ka_opts_extra=ka_opts_extra if ka_opts_extra else None,
        stap_debug_samples=int(args.stap_debug_samples),
        fd_span_mode=str(args.fd_span_mode).lower(),
        fd_span_rel=span_bounds,
        fd_fixed_span_hz=args.fd_fixed_span_hz,
        constraint_mode=str(args.constraint_mode).lower(),
        grid_step_rel=float(args.grid_step_rel),
        fd_min_pts=3,
        fd_max_pts=int(args.max_pts),
        mvdr_load_mode=str(args.mvdr_load_mode).lower(),
        mvdr_auto_kappa=float(args.mvdr_auto_kappa),
        constraint_ridge=float(args.constraint_ridge),
        stap_device=args.stap_device,
        baseline_type=str(args.baseline_type).lower(),
        alias_psd_select_enable=bool(args.alias_psd_select_enable),
        alias_psd_select_ratio_thresh=float(args.alias_psd_select_ratio),
        alias_psd_select_bins=int(args.alias_psd_select_bins),
        band_ratio_mode=str(args.band_ratio_mode).lower(),
        band_ratio_tile_mode=str(args.band_ratio_tile_mode).lower(),
        band_ratio_flow_low_hz=float(args.band_ratio_flow_low_hz),
        band_ratio_flow_high_hz=float(args.band_ratio_flow_high_hz),
        band_ratio_alias_center_hz=float(args.band_ratio_alias_center_hz),
        band_ratio_alias_width_hz=float(args.band_ratio_alias_width_hz),
        psd_telemetry=bool(args.psd_telemetry),
        psd_tapers=int(args.psd_tapers),
        psd_bandwidth=float(args.psd_bandwidth),
        score_mode=str(args.score_mode).lower(),
        flow_alias_hz=args.flow_alias_hz,
        flow_alias_fraction=float(args.flow_alias_fraction),
        flow_alias_depth_min_frac=args.flow_alias_depth_min_frac,
        flow_alias_depth_max_frac=args.flow_alias_depth_max_frac,
        flow_alias_jitter_hz=float(args.flow_alias_jitter_hz),
        flow_doppler_min_hz=args.flow_doppler_min_hz,
        flow_doppler_max_hz=args.flow_doppler_max_hz,
        flow_doppler_tone_amp=float(args.flow_doppler_tone_amp),
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
        flow_amp_shallow_scale=float(args.flow_amp_shallow_scale),
        flow_amp_mid_scale=float(args.flow_amp_mid_scale),
        flow_amp_deep_scale=float(args.flow_amp_deep_scale),
        flow_depth_mid_frac=float(args.flow_depth_mid_frac),
        flow_depth_deep_frac=float(args.flow_depth_deep_frac),
        alias_amp_scale=float(args.alias_amp_scale),
        vibration_hz=args.vibration_hz,
        vibration_amp=float(args.vibration_amp),
        vibration_depth_min_frac=float(args.vibration_depth_min_frac),
        vibration_depth_decay_frac=float(args.vibration_depth_decay_frac),
        aperture_phase_std=float(args.aperture_phase_std),
        aperture_phase_corr_len=float(args.aperture_phase_corr_len),
        aperture_phase_seed=int(args.aperture_phase_seed),
        skull_phase_std=float(args.skull_phase_std),
        skull_phase_corr_lat=float(args.skull_phase_corr_lat),
        skull_phase_corr_depth_frac=float(args.skull_phase_corr_depth_frac),
        skull_depth_max_frac=float(args.skull_depth_max_frac),
        skull_guided_hz=float(args.skull_guided_hz),
        skull_guided_amp=float(args.skull_guided_amp),
        skull_guided_depth_max_frac=float(args.skull_guided_depth_max_frac),
        skull_guided_lat_corr=float(args.skull_guided_lat_corr),
        resid_shift_std_px=float(args.resid_shift_std_px),
        clutter_beta=float(args.clutter_beta),
        clutter_snr_db=float(args.clutter_snr_db),
        clutter_depth_min_frac=float(args.clutter_depth_min_frac),
        clutter_depth_max_frac=float(args.clutter_depth_max_frac),
        depth_amp_profile=args.depth_amp_profile,
        depth_amp_min_factor=float(args.depth_amp_min_factor),
        hemo_mod_amp=float(args.hemo_mod_amp),
        hemo_mod_breath_period=float(args.hemo_mod_breath_period),
        hemo_mod_card_period=float(args.hemo_mod_card_period),
        meta_extra={
            "source": "pilot_motion",
            "profile": args.profile,
            "ensembles": args.ensembles,
            "jitter_um": args.jitter_um,
            "synthetic": bool(args.synthetic),
            "force_cpu": bool(args.force_cpu),
            "stap_device": args.stap_device,
            "msd_ratio_rho": ratio_rho,
            "motion_half_span_rel": motion_half_span_rel,
            "msd_contrast_alpha": contrast_alpha,
            "msd_agg": str(args.msd_agg).lower(),
            "aperture_phase_std": float(args.aperture_phase_std),
            "aperture_phase_corr_len": float(args.aperture_phase_corr_len),
            "clutter_beta": float(args.clutter_beta),
            "clutter_snr_db": float(args.clutter_snr_db),
            "clutter_depth_min_frac": float(args.clutter_depth_min_frac),
            "clutter_depth_max_frac": float(args.clutter_depth_max_frac),
            "flow_alias_hz": (
                float(args.flow_alias_hz) if args.flow_alias_hz is not None else None
            ),
            "flow_alias_fraction": float(args.flow_alias_fraction),
            "flow_alias_depth_min_frac": float(args.flow_alias_depth_min_frac),
            "flow_alias_depth_max_frac": float(args.flow_alias_depth_max_frac),
            "flow_alias_jitter_hz": float(args.flow_alias_jitter_hz),
            "bg_alias_hz": float(args.bg_alias_hz) if args.bg_alias_hz is not None else None,
            "bg_alias_fraction": float(args.bg_alias_fraction),
            "bg_alias_depth_min_frac": float(args.bg_alias_depth_min_frac),
            "bg_alias_depth_max_frac": float(args.bg_alias_depth_max_frac),
            "bg_alias_jitter_hz": float(args.bg_alias_jitter_hz),
            "bg_alias_highrank_mode": (
                None
                if str(args.bg_alias_highrank_mode).lower() == "none"
                else str(args.bg_alias_highrank_mode)
            ),
            "bg_alias_highrank_coverage": float(args.bg_alias_highrank_coverage),
            "bg_alias_highrank_freq_jitter_hz": float(args.bg_alias_highrank_freq_jitter_hz),
            "bg_alias_highrank_drift_step_hz": float(args.bg_alias_highrank_drift_step_hz),
            "bg_alias_highrank_drift_block_len": (
                int(args.bg_alias_highrank_drift_block_len)
                if args.bg_alias_highrank_drift_block_len is not None
                else None
            ),
            "bg_alias_highrank_pf_leak_eta": float(args.bg_alias_highrank_pf_leak_eta),
            "bg_alias_highrank_amp": float(args.bg_alias_highrank_amp),
            "flow_doppler_min_hz": (
                float(args.flow_doppler_min_hz) if args.flow_doppler_min_hz is not None else None
            ),
            "flow_doppler_max_hz": (
                float(args.flow_doppler_max_hz) if args.flow_doppler_max_hz is not None else None
            ),
        },
        feasibility_mode=str(args.feasibility_mode),
    )

    print(
        f"[pilot_motion] wrote {out_root} with {len(angle_sets)} ensembles, "
        f"{len(base_angles)} base angles. Bundle at {bundle_paths['meta']}"
    )


if __name__ == "__main__":
    main()
