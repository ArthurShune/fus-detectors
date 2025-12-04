#!/usr/bin/env python3
"""Track A pipeline helper for running replay + coverage analysis with canonical settings."""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable or "python"


@dataclass(frozen=True)
class FlowMaskConfig:
    mode: str = "pd_auto"
    pd_quantile: float = 0.95
    depth_min_frac: float = 0.15
    depth_max_frac: float = 0.95
    erode_iters: int = 0
    dilate_iters: int = 7
    min_pixels: int = 800
    min_coverage_frac: float = 0.01
    union_default: bool = True


@dataclass(frozen=True)
class TrackAConfig:
    tile_h: int = 12
    tile_w: int = 12
    tile_stride: int = 4
    Lt: int = 8
    diag_load: float = 1e-2
    cov_estimator: str = "tyler_pca"
    huber_c: float = 4.5
    grid_step_rel: float = 0.12
    fd_span_rel: tuple[float, float] = (0.30, 1.10)
    fd_min_pts: int = 3
    fd_max_pts: int = 7
    fd_min_abs_hz: float = 30.0
    msd_lambda: float = 0.05
    msd_ridge: float = 0.12
    msd_agg: str = "median"
    msd_ratio_rho: float = 0.05
    msd_contrast_alpha: float | None = None
    alias_psd_ratio: float = 1.2
    alias_psd_bins: int = 2
    score_mode: str = "msd"
    stap_device: str = "cuda"
    ka_kappa: float = 40.0
    motion_half_span_rel: float | None = None
    flow_mask: FlowMaskConfig = FlowMaskConfig()


def _default_replay_args(config: TrackAConfig, src: Path, out_dir: Path) -> list[str]:
    fm = config.flow_mask
    args = [
        PYTHON,
        str(REPO / "scripts" / "replay_stap_from_run.py"),
        "--src",
        str(src),
        "--out",
        str(out_dir),
        "--tile-h",
        str(config.tile_h),
        "--tile-w",
        str(config.tile_w),
        "--tile-stride",
        str(config.tile_stride),
        "--lt",
        str(config.Lt),
        "--diag-load",
        str(config.diag_load),
        "--cov-estimator",
        config.cov_estimator,
        "--huber-c",
        str(config.huber_c),
        "--grid-step-rel",
        str(config.grid_step_rel),
        "--fd-span-rel",
        f"{config.fd_span_rel[0]},{config.fd_span_rel[1]}",
        "--fd-min-pts",
        str(config.fd_min_pts),
        "--max-pts",
        str(config.fd_max_pts),
        "--fd-min-abs-hz",
        str(config.fd_min_abs_hz),
        "--msd-lambda",
        str(config.msd_lambda),
        "--msd-ridge",
        str(config.msd_ridge),
        "--msd-agg",
        config.msd_agg,
        "--msd-ratio-rho",
        str(config.msd_ratio_rho),
        "--stap-device",
        config.stap_device,
        "--score-mode",
        config.score_mode,
        "--ka-mode",
        "analytic",
        "--ka-kappa",
        str(config.ka_kappa),
        "--alias-psd-select",
        "--alias-psd-select-ratio",
        str(config.alias_psd_ratio),
        "--alias-psd-select-bins",
        str(config.alias_psd_bins),
        "--flow-mask-mode",
        fm.mode,
        "--flow-mask-pd-quantile",
        str(fm.pd_quantile),
        "--flow-mask-depth-min-frac",
        str(fm.depth_min_frac),
        "--flow-mask-depth-max-frac",
        str(fm.depth_max_frac),
        "--flow-mask-erode-iters",
        str(fm.erode_iters),
        "--flow-mask-dilate-iters",
        str(fm.dilate_iters),
        "--flow-mask-min-pixels",
        str(fm.min_pixels),
        "--flow-mask-min-coverage-frac",
        str(fm.min_coverage_frac),
    ]
    if fm.union_default:
        args.append("--flow-mask-union-default")
    else:
        args.append("--flow-mask-no-union-default")
    if config.msd_contrast_alpha is not None:
        args.extend(["--msd-contrast-alpha", str(config.msd_contrast_alpha)])
    if config.motion_half_span_rel is not None:
        args.extend(["--motion-half-span-rel", str(config.motion_half_span_rel)])
    return args


@dataclass
class PipelineStep:
    name: str
    command: list[str]


def _run(cmd: list[str], dry_run: bool) -> None:
    print("\n$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def _latest_bundle(out_dir: Path) -> Path:
    bundles = sorted(out_dir.glob("pw_*"), key=lambda p: p.stat().st_mtime)
    if not bundles:
        raise FileNotFoundError(f"No bundles found under {out_dir}")
    return bundles[-1]


def _coverage_cmd(bundles: Sequence[Path], thresholds: Sequence[float]) -> list[str]:
    cmd = [
        PYTHON,
        str(REPO / "scripts" / "analyze_coverage_roc.py"),
    ]
    for bundle in bundles:
        cmd.extend(["--bundle", str(bundle)])
    if thresholds:
        cmd.append("--thresholds")
        cmd.extend(str(thr) for thr in thresholds)
    return cmd


def build_pipeline(
    src_root: Path,
    out_root: Path,
    config: TrackAConfig,
    run_alias: bool,
    run_pftrace: bool,
    run_analysis: bool,
) -> tuple[list[PipelineStep], list[Path]]:
    out_root.mkdir(parents=True, exist_ok=True)
    steps: list[PipelineStep] = []
    bundles: list[Path] = []
    alias_out = out_root / "alias"
    pf_out = out_root / "alias_pftrace"
    if run_alias:
        alias_out.mkdir(parents=True, exist_ok=True)
        steps.append(
            PipelineStep(
                name="alias_only",
                command=_default_replay_args(config, src_root, alias_out),
            )
        )
        bundles.append(alias_out)
    if run_pftrace:
        pf_out.mkdir(parents=True, exist_ok=True)
        cmd = _default_replay_args(config, src_root, pf_out) + ["--ka-equalize-pf-trace"]
        steps.append(PipelineStep(name="alias_pftrace", command=cmd))
        bundles.append(pf_out)
    if run_analysis:
        steps.append(PipelineStep(name="coverage_analysis", command=["__ANALYZE__"]))
    return steps, bundles


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Track A replay + coverage pipeline")
    ap.add_argument(
        "--src", type=Path, required=True, help="Source run root (containing angle_* dirs)"
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Destination root for pipeline outputs (will create alias/alias_pftrace subdirs)",
    )
    ap.add_argument("--skip-alias", action="store_true", help="Skip alias-only replay")
    ap.add_argument("--skip-pftrace", action="store_true", help="Skip Pf-trace replay")
    ap.add_argument("--skip-analysis", action="store_true", help="Skip coverage analysis step")
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help="Coverage thresholds for the analysis report",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config = TrackAConfig()
    steps, bundle_roots = build_pipeline(
        src_root=args.src.expanduser().resolve(),
        out_root=args.out_root.expanduser().resolve(),
        config=config,
        run_alias=not args.skip_alias,
        run_pftrace=not args.skip_pftrace,
        run_analysis=not args.skip_analysis,
    )
    dry_run = bool(args.dry_run)
    produced_bundles: list[Path] = []
    for step in steps:
        if step.command == ["__ANALYZE__"]:
            if not bundle_roots:
                print("\n[warn] No replay outputs available for analysis; skipping coverage step.")
                continue
            if dry_run:
                placeholder = [path / "pw_*" for path in bundle_roots]
                analyze_cmd = _coverage_cmd(placeholder, args.thresholds)
            else:
                produced_bundles = [_latest_bundle(path) for path in bundle_roots]
                analyze_cmd = _coverage_cmd(produced_bundles, args.thresholds)
            _run(analyze_cmd, dry_run=dry_run)
            continue
        _run(step.command, dry_run=dry_run)


if __name__ == "__main__":
    main()
