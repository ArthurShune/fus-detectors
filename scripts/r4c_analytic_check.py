#!/usr/bin/env python3
"""
Quick analytical sanity checks for k-Wave brain fUS bundles.

For each requested bundle type (e.g., mcsvd_k8_reg4, ka_seed) and seed, the script
loads `meta.json`, pulls key telemetry fields, and prints out alias engagement and
registration/coverage stats. This helps confirm that the replay configuration
actually stresses MC-SVD (alias inside the ≥50% slice, degraded PSR) while keeping
KA guardrails within spec before investing in long ROC runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_bundle_meta(path: Path) -> dict | None:
    meta_path = path / "meta.json"
    if not meta_path.exists():
        print(f"[warn] missing meta.json at {meta_path}")
        return None
    return json.loads(meta_path.read_text())


def gather_stats(path: Path) -> dict | None:
    meta = load_bundle_meta(path)
    if meta is None:
        return None
    tel = meta.get("stap_fallback_telemetry") or {}
    stats = {
        "flow_cov_ge_50": _to_float(tel.get("flow_cov_ge_50_fraction")),
        "alias_flag_ge50": _to_float(tel.get("alias_cov_ge_50_alias_flag_fraction")),
        "alias_ratio_ge50": _to_float(tel.get("alias_cov_ge_50_alias_ratio_median")),
        "psd_alias_ratio_median": _to_float(tel.get("psd_alias_ratio_median")),
        "psd_alias_ratio_p90": _to_float(tel.get("psd_alias_ratio_p90")),
        "reg_psr_median": _to_float(tel.get("reg_psr_median")),
        "reg_psr_p10": _to_float(tel.get("reg_psr_p10")),
        "reg_psr_p90": _to_float(tel.get("reg_psr_p90")),
        "reg_shift_rms": _to_float(tel.get("reg_shift_rms")),
        "reg_shift_p90": _to_float(tel.get("reg_shift_p90")),
        "bg_var_ratio": _to_float(tel.get("bg_var_ratio_actual")),
        "pd_mask_median": _to_float(tel.get("flow_pdmask_ratio_median")),
    }
    return stats


def format_float(val: float | None) -> str:
    if val is None or not isinstance(val, (float, int)):
        return "  - "
    if abs(val) < 0.005:
        return f"{val:.3e}"
    return f"{val:5.3f}"


def summarize(label: str, stats_list: list[dict], seeds: Iterable[int]) -> None:
    if not stats_list:
        print(f"[warn] no stats collected for {label}")
        return
    print(f"\n[{label}] alias & registration summary")
    header = (
        "seed | flow≥50 | alias_flag≥50 | alias_ratio≥50 | reg_psr_med | reg_psr_p10 | "
        "reg_shift_rms | bg_var_ratio | pd_mask_med"
    )
    print(header)
    print("-" * len(header))
    for seed, stats in zip(seeds, stats_list, strict=False):
        if stats is None:
            print(f"{seed:4d} | (missing)")
            continue
        row = " | ".join(
            [
                f"{seed:4d}",
                format_float(stats.get("flow_cov_ge_50")),
                format_float(stats.get("alias_flag_ge50")),
                format_float(stats.get("alias_ratio_ge50")),
                format_float(stats.get("reg_psr_median")),
                format_float(stats.get("reg_psr_p10")),
                format_float(stats.get("reg_shift_rms")),
                format_float(stats.get("bg_var_ratio")),
                format_float(stats.get("pd_mask_median")),
            ]
        )
        print(row)

    def mean_of(key: str) -> float | None:
        vals = [s.get(key) for s in stats_list if s and s.get(key) is not None]
        return statistics.mean(vals) if vals else None

    print("-----")
    print(
        "means: flow≥50={:s}, alias_flag≥50={:s}, alias_ratio≥50={:s}, reg_psr_med={:s}, "
        "reg_shift_rms={:s}".format(
            format_float(mean_of("flow_cov_ge_50")),
            format_float(mean_of("alias_flag_ge50")),
            format_float(mean_of("alias_ratio_ge50")),
            format_float(mean_of("reg_psr_median")),
            format_float(mean_of("reg_shift_rms")),
        )
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Analytical telemetry check for k-Wave brain fUS bundles."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("runs/motion/r4c"),
        help="Root directory containing per-bundle run folders.",
    )
    ap.add_argument(
        "--dataset-prefix",
        type=str,
        default="pw_7.5MHz_5ang_5ens_320T_seed",
        help="Dataset prefix inside each bundle directory.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="1 2 3 4 5",
        help='Space-separated list of seeds to inspect (e.g., "1 2 3").',
    )
    ap.add_argument(
        "--bundles",
        type=str,
        default="mcsvd_k8_reg4,ka_seed",
        help=(
            "Comma-separated bundle names under base-dir (script appends `_seed{seed}`); "
            "e.g., 'mcsvd_k8_reg4,ka_seed,ka_alias_forced_seed'."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split() if s.strip()]
    bundle_names = [b.strip() for b in args.bundles.split(",") if b.strip()]

    for bundle in bundle_names:
        stats_per_seed: list[dict | None] = []
        for seed in seeds:
            if "{seed}" in bundle:
                rel_dir = bundle.format(seed=seed)
            elif bundle.endswith("_seed"):
                rel_dir = f"{bundle}{seed}"
            else:
                rel_dir = f"{bundle}_seed{seed}"
            bundle_dir = args.base_dir / rel_dir / f"{args.dataset_prefix}{seed}"
            stats_per_seed.append(gather_stats(bundle_dir))
        summarize(bundle, stats_per_seed, seeds)


if __name__ == "__main__":
    main()
