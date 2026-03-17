#!/usr/bin/env python3
"""
Run KA-STAP ablations (Pf off, alias PSD off, Pf-trace off, guards off) and
produce a compact report with multi-FPR ROC metrics plus amplitude telemetry.

Example:
    PYTHONPATH=. conda run -n fus-detectors python scripts/run_ka_ablation.py \
        --src runs/pilot/r3_kwave \
        --dataset pw_7.5MHz_5ang_3ens_192T_seed1 \
        --out-root runs/motion/ka_ablation \
        --report-json reports/ka_ablation_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

COMMON_ARGS = [
    "--tile-h",
    "10",
    "--tile-w",
    "10",
    "--tile-stride",
    "5",
    "--lt",
    "4",
    "--diag-load",
    "1e-2",
    "--cov-estimator",
    "tyler_pca",
    "--huber-c",
    "5.0",
    "--fd-span-mode",
    "psd",
    "--fd-span-rel",
    "0.30,1.10",
    "--grid-step-rel",
    "0.12",
    "--max-pts",
    "5",
    "--fd-min-pts",
    "3",
    "--msd-lambda",
    "5e-2",
    "--msd-ridge",
    "0.06",
    "--msd-agg",
    "median",
    "--msd-ratio-rho",
    "0.05",
    "--motion-half-span-rel",
    "0.20",
    "--msd-contrast-alpha",
    "0.8",
    "--constraint-mode",
    "exp+deriv",
    "--constraint-ridge",
    "0.10",
    "--mvdr-load-mode",
    "auto",
    "--mvdr-auto-kappa",
    "50",
    "--flow-mask-mode",
    "default",
]

FIXED_GROUPS: Dict[str, List[str]] = {
    "ka": [
        "--ka-mode",
        "library",
        "--ka-prior-path",
        "runs/motion/priors/ka_prior_lt4_prf3k.npy",
        "--ka-directional-beta",
        "--ka-kappa",
        "30",
        "--ka-beta-bounds",
        "0.05,0.5",
        "--ka-alpha",
        "0.0",
        "--ka-target-retain-f",
        "1.0",
        "--ka-target-shrink-perp",
        "0.95",
    ],
    "pftrace": ["--ka-equalize-pf-trace"],
}


def resolve_group_args(group: str, args: argparse.Namespace) -> List[str]:
    if group in FIXED_GROUPS:
        return FIXED_GROUPS[group]
    if group == "alias":
        return [
            "--alias-psd-select",
            "--alias-psd-select-ratio",
            str(args.alias_psd_ratio),
            "--alias-psd-select-bins",
            str(args.alias_psd_bins),
        ]
    if group == "guard":
        return [
            "--bg-guard-enabled",
            "--bg-guard-target-med",
            "0.45",
            "--bg-guard-target-low",
            "0.16",
            "--bg-guard-percentile-low",
            "0.10",
            "--bg-guard-coverage-min",
            "0.10",
            "--bg-guard-max-scale",
            "1.30",
            "--bg-guard-target-p90",
            str(args.bg_guard_target_p90),
            "--bg-guard-min-alpha",
            str(args.bg_guard_min_alpha),
            "--bg-guard-metric",
            "global",
        ]
    raise ValueError(f"Unknown group '{group}'")


ABLATIONS = [
    {
        "name": "baseline",
        "description": "KA-STAP baseline (Pf + alias PSD + Pf-trace + guards)",
        "groups": ["ka", "pftrace", "alias", "guard"],
        "extra": [],
    },
    {
        "name": "pf_off",
        "description": "Pf projector disabled (KA off)",
        "groups": ["alias", "guard"],
        "extra": [],
    },
    {
        "name": "alias_psd_off",
        "description": "Alias PSD selection disabled",
        "groups": ["ka", "pftrace", "guard"],
        "extra": [],
    },
    {
        "name": "pftrace_off",
        "description": "Pf-trace equalization disabled",
        "groups": ["ka", "alias", "guard"],
        "extra": [],
    },
    {
        "name": "guards_off",
        "description": "Flow/background guards disabled",
        "groups": ["ka", "pftrace", "alias"],
        "extra": [],
    },
]


def run_cmd(cmd: List[str], dry_run: bool) -> None:
    if dry_run:
        print("[dry-run]", " ".join(cmd))
        return
    subprocess.run(cmd, check=True)


def build_replay_cmd(
    args: argparse.Namespace,
    out_dir: Path,
    ablation: Dict[str, object],
) -> List[str]:
    cmd = [
        "python",
        "scripts/replay_stap_from_run.py",
        "--src",
        str(args.src),
        "--out",
        str(out_dir),
    ]
    cmd += COMMON_ARGS
    cmd += ["--stap-device", args.stap_device]
    for group in ablation["groups"]:
        cmd += resolve_group_args(group, args)
    cmd += ablation.get("extra", [])
    return cmd


def summarize_bundle(bundle_path: Path) -> Dict[str, float]:
    meta = json.loads((bundle_path / "meta.json").read_text())
    tele = meta.get("stap_fallback_telemetry", {})

    def maybe_float(val):
        if val is None:
            return float("nan")
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "flow_pdmask_ratio": maybe_float(tele.get("flow_pdmask_ratio_median")),
        "bg_var_ratio": maybe_float(tele.get("bg_var_ratio_actual")),
        "stap_ms": maybe_float(tele.get("stap_ms")),
        "alias_select_count": maybe_float(tele.get("psd_alias_select_count")),
        "alias_select_fraction": maybe_float(tele.get("psd_alias_select_fraction")),
        "bg_guard_applied": bool(tele.get("bg_guard_applied", False)),
        "bg_guard_alpha": maybe_float(tele.get("bg_guard_alpha")),
    }


def extract_roc_metrics(roc_json: Path, threshold: float) -> Dict[str, Dict[str, float]]:
    data = json.loads(roc_json.read_text())
    metrics: Dict[str, Dict[str, float]] = {}
    for entry in data:
        path = Path(entry["path"])
        name = path.parent.name
        for row in entry["coverage_results"]:
            if abs(row["threshold"] - threshold) < 1e-6:
                tpr_1e5 = float(row["tpr_stap"])  # use detector (stap) TPRs
                extras = {float(e["fpr_target"]): e for e in row.get("tpr_extra", [])}
                metrics[name] = {
                    "tpr@1e-5": tpr_1e5,
                    "tpr@1e-4": float(extras.get(1e-4, {}).get("tpr_stap", float("nan"))),
                    "tpr@1e-3": float(extras.get(1e-3, {}).get("tpr_stap", float("nan"))),
                }
                break
    return metrics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run KA-STAP ablations and summarize results.")
    ap.add_argument("--src", type=Path, required=True, help="Source run directory (pilot bundle).")
    ap.add_argument(
        "--dataset",
        type=str,
        default="pw_7.5MHz_5ang_3ens_192T_seed1",
        help="Dataset directory name inside each output run.",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/motion/ka_ablation"),
        help="Root directory to place ablation outputs.",
    )
    ap.add_argument(
        "--stap-device",
        type=str,
        default="cuda",
        help="Device for replay_stap_from_run.py (cuda/cpu).",
    )
    ap.add_argument(
        "--alias-psd-ratio",
        type=float,
        default=1.2,
        help="Alias PSD selection ratio threshold (used when alias group enabled).",
    )
    ap.add_argument(
        "--alias-psd-bins",
        type=int,
        default=1,
        help="Alias PSD selection positive bins (per sign).",
    )
    ap.add_argument(
        "--bg-guard-target-p90",
        type=float,
        default=0.95,
        help="Background guard target p90 (used when guard group enabled).",
    )
    ap.add_argument(
        "--bg-guard-min-alpha",
        type=float,
        default=0.4,
        help="Minimum alpha shrink for background guard.",
    )
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.5, 0.8],
        help="Coverage thresholds passed to analyze_coverage_roc.py.",
    )
    ap.add_argument(
        "--extra-fpr-targets",
        nargs="+",
        type=float,
        default=[1e-4, 1e-3],
        help="Additional FPR targets for TPR deltas.",
    )
    ap.add_argument(
        "--pauc-max",
        type=float,
        default=1e-3,
        help="Upper limit for partial AUC.",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default=None,
        help="Override score mode for ROC analysis (msd/pd/band_ratio).",
    )
    ap.add_argument(
        "--flow-mask-kind",
        type=str,
        choices=("default", "pd"),
        default="default",
        help="Flow mask for coverage gating during ROC analysis (default or pd-derived).",
    )
    ap.add_argument(
        "--report-json",
        type=Path,
        default=Path("reports/ka_ablation_summary.json"),
        help="Path to write the summary JSON.",
    )
    ap.add_argument(
        "--roc-json",
        type=Path,
        default=Path("reports/ka_ablation_roc.json"),
        help="Path for analyze_coverage_roc output.",
    )
    ap.add_argument(
        "--roc-threshold",
        type=float,
        default=0.5,
        help="Coverage threshold from which to report TPR metrics.",
    )
    ap.add_argument("--skip-replay", action="store_true", help="Skip rerunning ablation replays.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    bundle_paths: Dict[str, Path] = {}

    for ablation in ABLATIONS:
        name = ablation["name"]
        out_dir = args.out_root / name
        bundle_dir = out_dir / args.dataset
        bundle_paths[name] = bundle_dir
        if not args.skip_replay:
            cmd = build_replay_cmd(args, out_dir, ablation)
            print(f"[+] Running ablation '{name}'")
            run_cmd(cmd, args.dry_run)
        else:
            print(f"[skip] Replay for '{name}'")

    bundles_for_roc = []
    for name, path in bundle_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Bundle path missing for {name}: {path}")
        bundles_for_roc.append(str(path))

    roc_cmd = [
        "python",
        "scripts/analyze_coverage_roc.py",
        *sum([["--bundle", b] for b in bundles_for_roc], []),
        "--thresholds",
        *[str(x) for x in args.thresholds],
        "--fpr-target",
        "1e-5",
        "--pauc-max",
        str(args.pauc_max),
        "--json",
        str(args.roc_json),
    ]
    if args.extra_fpr_targets:
        roc_cmd += ["--extra-fpr-targets", *[str(x) for x in args.extra_fpr_targets]]
    if args.score_mode:
        roc_cmd += ["--score-mode", args.score_mode]
    if args.flow_mask_kind and args.flow_mask_kind != "default":
        roc_cmd += ["--flow-mask-kind", args.flow_mask_kind]
    print("[+] Running analyze_coverage_roc")
    run_cmd(roc_cmd, args.dry_run)

    if args.dry_run:
        return

    roc_metrics = extract_roc_metrics(args.roc_json, args.roc_threshold)
    summary = []
    for ablation in ABLATIONS:
        name = ablation["name"]
        bundle_dir = bundle_paths[name]
        tele = summarize_bundle(bundle_dir)
        metrics = roc_metrics.get(name, {})
        entry = {
            "name": name,
            "description": ablation["description"],
            "bundle": str(bundle_dir),
            "tpr_at_1e-5": metrics.get("tpr@1e-5"),
            "tpr_at_1e-4": metrics.get("tpr@1e-4"),
            "tpr_at_1e-3": metrics.get("tpr@1e-3"),
            "flow_pdmask_ratio": tele["flow_pdmask_ratio"],
            "bg_var_ratio": tele["bg_var_ratio"],
            "stap_ms": tele["stap_ms"],
            "alias_select_count": tele["alias_select_count"],
            "alias_select_fraction": tele["alias_select_fraction"],
            "bg_guard_applied": tele["bg_guard_applied"],
            "bg_guard_alpha": tele["bg_guard_alpha"],
        }
        summary.append(entry)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {args.report_json}\n")
    print("name\tTPR@1e-4\tPD-mask\tbg_var\tstap_ms\talias_frac\tbg_guard")
    for entry in summary:
        print(
            f"{entry['name']}\t{entry['tpr_at_1e-4']:.4f}\t"
            f"{entry['flow_pdmask_ratio']:.3f}\t{entry['bg_var_ratio']:.3f}\t{entry['stap_ms']:.0f}\t"
            f"{entry['alias_select_fraction']:.3f}\t"
            f"{('Y' if entry['bg_guard_applied'] else 'N')}({entry['bg_guard_alpha'] if entry['bg_guard_alpha']==entry['bg_guard_alpha'] else 'NA'})"
        )


if __name__ == "__main__":
    main()
