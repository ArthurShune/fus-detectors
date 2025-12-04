#!/usr/bin/env python3
"""
Evaluate KA-STAP feasibility conditions against bundle telemetry.

Reads meta.json files (or directories containing them) and checks the metrics that correspond
to the conditions defined in feasibility_document.txt. Outputs a pass/warn/fail summary.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np

Status = str


@dataclass
class ConditionResult:
    name: str
    status: Status
    detail: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class BundleInfo:
    path: Path
    meta: Dict[str, Any]
    telemetry: Dict[str, Any]
    score_stats: Dict[str, Any]
    seed: int | None


def load_bundle(path_str: str) -> BundleInfo:
    path = Path(path_str)
    meta_path = path
    if path.is_dir():
        meta_path = path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found at {meta_path}")
    meta = json.loads(meta_path.read_text())
    telemetry = meta.get("stap_fallback_telemetry", {})
    return BundleInfo(
        path=path,
        meta=meta,
        telemetry=telemetry,
        score_stats=meta.get("score_stats", {}),
        seed=meta.get("seed"),
    )


def collect_metric(
    bundles: Iterable[BundleInfo],
    extractor: Callable[[BundleInfo], Any],
) -> Tuple[List[float], List[Tuple[str, float]]]:
    values: List[float] = []
    samples: List[Tuple[str, float]] = []
    for bundle in bundles:
        raw = extractor(bundle)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isnan(val):
            continue
        values.append(val)
        samples.append((bundle.path.name, val))
    return values, samples


def median_or_none(values: Iterable[float]) -> float | None:
    vals = list(values)
    if not vals:
        return None
    return float(statistics.median(vals))


def percentile(values: Iterable[float], q: float) -> float | None:
    vals = list(values)
    if not vals:
        return None
    if len(vals) == 1:
        return float(vals[0])
    arr = np.asarray(vals, dtype=float)
    return float(np.quantile(arr, q))


def check_flow_alignment(bundles: List[BundleInfo]) -> ConditionResult:
    vals, samples = collect_metric(
        bundles,
        lambda b: b.telemetry.get("flow_band_alignment_stats", {}).get("median"),
    )
    if not vals:
        return ConditionResult(
            name="G1 Flow-band alignment",
            status="pending",
            detail="Missing flow_band_alignment_stats in telemetry.",
            metrics={},
        )
    median_val = float(np.median(vals))
    p10 = float(np.quantile(vals, 0.10)) if len(vals) > 1 else median_val
    status: Status
    if median_val >= 0.75:
        status = "pass"
    elif median_val >= 0.65:
        status = "warn"
    else:
        status = "fail"
    detail = (
        f"Median cosθ={median_val:.3f} (10th percentile {p10:.3f}); "
        "doc target ≥0.7 for majority of tiles."
    )
    return ConditionResult(
        name="G1 Flow-band alignment",
        status=status,
        detail=detail,
        metrics={"samples": samples},
    )


def check_flow_motion_angle(bundles: List[BundleInfo]) -> ConditionResult:
    def _extract_angle_stat(bundle: BundleInfo) -> float | None:
        stats = bundle.telemetry.get("flow_motion_angle_stats", {})
        if not stats:
            return None
        # Telemetry stores "median" (degrees) plus p10/p90. Older bundles used "median_deg".
        if "median" in stats and stats["median"] is not None:
            return float(stats["median"])
        if "median_deg" in stats and stats["median_deg"] is not None:
            return float(stats["median_deg"])
        return None

    vals, samples = collect_metric(bundles, _extract_angle_stat)
    if not vals:
        return ConditionResult(
            name="G2 Flow-motion angle",
            status="pending",
            detail="Missing flow_motion_angle_stats in telemetry.",
            metrics={},
        )
    median_val = float(np.median(vals))
    p10 = float(np.quantile(vals, 0.10)) if len(vals) > 1 else median_val
    if median_val >= 25.0:
        status = "pass"
    elif median_val >= 18.0:
        status = "warn"
    else:
        status = "fail"
    detail = (
        f"Median angle={median_val:.1f}° (10th percentile {p10:.1f}°); " "doc target ≥20°–40°."
    )
    return ConditionResult(
        name="G2 Flow-motion angle",
        status=status,
        detail=detail,
        metrics={"samples": samples},
    )


def check_psd_alignment(bundles: List[BundleInfo]) -> ConditionResult:
    flow_fracs, flow_samples = collect_metric(
        bundles,
        lambda b: b.telemetry.get("psd_peak_alignment", {}).get("flow_fraction_in_band"),
    )
    # Prefer alias fraction conditioned on alias-candidate tiles if available.
    alias_fracs, alias_samples = collect_metric(
        bundles,
        lambda b: b.telemetry.get("psd_peak_alignment", {}).get(
            "alias_fraction_in_band_on_alias_tiles"
        ),
    )
    alias_fracs_global, alias_samples_global = collect_metric(
        bundles,
        lambda b: b.telemetry.get("psd_peak_alignment", {}).get("alias_fraction_in_band"),
    )
    alias_cand_counts, _ = collect_metric(
        bundles,
        lambda b: b.telemetry.get("psd_peak_alignment", {}).get("alias_candidate_count"),
    )
    if not flow_fracs and not alias_fracs:
        return ConditionResult(
            name="S1 Band alignment via PSD",
            status="pending",
            detail="Missing psd_peak_alignment telemetry.",
            metrics={},
        )
    band_info = bundles[0].telemetry.get("psd_peak_alignment", {}) if bundles else {}
    metrics = {
        "flow_samples": flow_samples,
        "alias_samples": alias_samples,
        "alias_samples_global": alias_samples_global,
        "alias_candidate_counts": alias_cand_counts,
    }
    status = "pass"
    detail_parts = []
    if flow_fracs:
        median_flow = float(np.median(flow_fracs))
        detail_parts.append(f"flow fraction median {median_flow:.2f}")
        if median_flow < 0.7:
            status = "fail"
        elif status != "fail" and median_flow < 0.85:
            status = "warn"
    else:
        status = "pending"
        detail_parts.append("flow fraction missing")
    if alias_fracs:
        median_alias = float(np.median(alias_fracs))
        cand_total = int(float(np.median(alias_cand_counts))) if alias_cand_counts else 0
        detail_parts.append(
            f"alias fraction (alias tiles) median {median_alias:.2f} "
            f"(alias_candidate_count≈{cand_total})"
        )
        # If we have very few alias candidates, treat alias alignment as a weak signal.
        if cand_total < 10:
            if status == "pass":
                status = "warn"
        else:
            if median_alias < 0.1:
                status = "fail"
            elif status != "fail" and median_alias < 0.3:
                status = "warn"
    elif alias_fracs_global:
        median_alias = float(np.median(alias_fracs_global))
        detail_parts.append(f"alias fraction (global) median {median_alias:.2f}")
        if median_alias < 0.5:
            status = "fail"
        elif status != "fail" and median_alias < 0.7:
            status = "warn"
    else:
        detail_parts.append("alias fraction missing")
        status = "pending" if status == "pass" else status
    flow_band = band_info.get("flow_band_hz")
    alias_band = band_info.get("alias_band_hz")
    if flow_band:
        detail_parts.append(f"flow band Hz [{flow_band[0]:.1f},{flow_band[1]:.1f}]")
    if alias_band:
        detail_parts.append(f"alias band Hz [{alias_band[0]:.1f},{alias_band[1]:.1f}]")
    detail = " / ".join(detail_parts) + " (doc sections G3/S1)."
    return ConditionResult(
        name="S1 Band alignment via PSD",
        status=status,
        detail=detail,
        metrics=metrics,
    )


def check_ka_snr(bundles: List[BundleInfo]) -> ConditionResult:
    vals, samples = collect_metric(
        bundles,
        lambda b: b.telemetry.get("ka_median_snr_flow_ratio"),
    )
    if not vals:
        return ConditionResult(
            name="C1 Flow SNR retention",
            status="pending",
            detail="ka_median_snr_flow_ratio not logged.",
            metrics={},
        )
    median_val = float(np.median(vals))
    if median_val >= 0.95:
        status = "pass"
    elif median_val >= 0.85:
        status = "warn"
    else:
        status = "fail"
    detail = f"Median KA/base SNR ratio {median_val:.3f}; doc target ≥0.9."
    return ConditionResult(
        name="C1 Flow SNR retention",
        status=status,
        detail=detail,
        metrics={"samples": samples},
    )


def check_ka_noise_ratio(bundles: List[BundleInfo]) -> ConditionResult:
    vals, samples = collect_metric(
        bundles,
        lambda b: b.telemetry.get("ka_median_noise_perp_ratio"),
    )
    if not vals:
        return ConditionResult(
            name="C2 Noise subspace energy",
            status="pending",
            detail="ka_median_noise_perp_ratio not logged.",
            metrics={},
        )
    median_val = float(np.median(vals))
    if 0.5 <= median_val <= 1.2:
        status = "pass"
    elif 0.4 <= median_val <= 1.4:
        status = "warn"
    else:
        status = "fail"
    detail = f"Median noise ratio {median_val:.3f}; doc target within [0.5, 1.2]."
    return ConditionResult(
        name="C2 Noise-subspace control",
        status=status,
        detail=detail,
        metrics={"samples": samples},
    )


def check_gating(bundles: List[BundleInfo]) -> ConditionResult:
    def _alias_ratio_summary(bundles: List[BundleInfo]) -> dict:
        ratios: List[float] = []
        for b in bundles:
            for tile in b.telemetry.get("tile_infos_sample", []):
                r = tile.get("psd_flow_alias_ratio")
                if r is None:
                    continue
                try:
                    ratios.append(float(r))
                except (TypeError, ValueError):
                    continue
        if not ratios:
            return {}
        arr = np.array(ratios, dtype=float)
        summary = {
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": float(np.max(arr)),
            "frac_gt_1_1": float((arr > 1.1).mean()),
            "frac_gt_1_3": float((arr > 1.3).mean()),
            "frac_gt_1_5": float((arr > 1.5).mean()),
            "frac_gt_2_0": float((arr > 2.0).mean()),
            "count": int(arr.size),
        }
        return summary

    alias_summary = _alias_ratio_summary(bundles)
    flow_vals, _ = collect_metric(
        bundles,
        lambda b: b.telemetry.get("ka_gate_fraction_on_flow"),
    )
    bg_vals, _ = collect_metric(
        bundles,
        lambda b: b.telemetry.get("ka_gate_fraction_on_bg"),
    )
    alias_medians, _ = collect_metric(
        bundles,
        lambda b: b.telemetry.get("ka_gate_alias_stats", {}).get("flow", {}).get("median"),
    )
    if not flow_vals:
        detail = "No ka_gate_fraction telemetry found (ensure KA replay bundles)."
        if alias_summary:
            detail += f" Alias ratio sample: med={alias_summary['median']:.2f}, p90={alias_summary['p90']:.2f}, max={alias_summary['max']:.2f}."
        return ConditionResult(
            name="G2 Class-differential gating",
            status="pending",
            detail=detail,
            metrics={"alias_summary": alias_summary} if alias_summary else {},
        )
    median_flow = float(np.median(flow_vals))
    median_bg = float(np.median(bg_vals)) if bg_vals else 0.0
    status = "pass"
    if median_bg > 0 and median_flow < 5 * median_bg:
        status = "fail"
    elif median_bg > 0 and median_flow < 10 * median_bg:
        status = "warn"
    elif median_flow <= 0:
        status = "na"
    alias_detail = ""
    if alias_medians:
        median_alias = float(np.median(alias_medians))
        alias_detail = f", gated alias median {median_alias:.2f}"
        if median_alias < 1.05:
            status = "warn"
    if status == "na":
        detail = f"Gating telemetry present but gate never fired (median flow fraction {median_flow:.4f}). "
    else:
        detail = (
            f"Median gate fraction on flow {median_flow:.4f} vs bg {median_bg:.4f}{alias_detail}; "
            "doc expects p_pos ≫ p_neg and alias > threshold."
        )
    if alias_summary:
        detail += (
            f" Alias ratio sample: med={alias_summary['median']:.2f}, "
            f"p90={alias_summary['p90']:.2f}, max={alias_summary['max']:.2f}; "
            f">1.1={alias_summary['frac_gt_1_1']:.2f}, >1.3={alias_summary['frac_gt_1_3']:.2f}, "
            f">1.5={alias_summary['frac_gt_1_5']:.2f}, >2.0={alias_summary['frac_gt_2_0']:.2f}."
        )
    return ConditionResult(
        name="G2 Class-differential gating",
        status=status,
        detail=detail,
        metrics={
            "flow_fraction_samples": flow_vals,
            "bg_fraction_samples": bg_vals,
            "alias_median_samples": alias_medians,
            "alias_summary": alias_summary,
        },
    )


def check_rank_consistency(bundles: List[BundleInfo]) -> ConditionResult:
    tau_vals, _ = collect_metric(
        bundles,
        lambda b: (b.meta.get("rank_consistency_gated", {}).get("flow", {}).get("kendall_tau")),
    )
    if not tau_vals:
        return ConditionResult(
            name="S2 Rank consistency (gated subset)",
            status="pending",
            detail="rank_consistency_gated missing; rerun with recent code.",
            metrics={},
        )
    median_tau = float(np.median(tau_vals))
    if median_tau >= 0.7:
        status = "pass"
    elif median_tau >= 0.5:
        status = "warn"
    else:
        status = "fail"
    detail = f"Gated Kendall τ median {median_tau:.3f}; doc target ≥0.7."
    return ConditionResult(
        name="S2 Rank consistency (gated subset)",
        status=status,
        detail=detail,
        metrics={"tau_samples": tau_vals},
    )


def check_score_transform(bundles: List[BundleInfo]) -> ConditionResult:
    spreads: List[float] = []
    for bundle in bundles:
        stats = bundle.meta.get("score_transform_gated", {}).get("flow", {})
        p10 = stats.get("p10_ratio")
        p90 = stats.get("p90_ratio")
        if p10 is None or p90 is None:
            continue
        try:
            spread = float(p90) - float(p10)
        except (TypeError, ValueError):
            continue
        spreads.append(spread)
    if not spreads:
        return ConditionResult(
            name="S3 Score transform diversity",
            status="pending",
            detail="score_transform_gated missing; rerun with recent code.",
            metrics={},
        )
    median_spread = float(np.median(spreads))
    if median_spread >= 0.05:
        status = "pass"
    elif median_spread >= 0.02:
        status = "warn"
    else:
        status = "fail"
    detail = (
        f"Gated score ratio spread (p90-p10) median {median_spread:.3f}; "
        "doc notes constant transforms imply ROC invariance."
    )
    return ConditionResult(
        name="S3 Score transform diversity",
        status=status,
        detail=detail,
        metrics={"spread_samples": spreads},
    )


def check_operator_metrics(bundles: List[BundleInfo]) -> ConditionResult:
    pf_mins, pf_maxs, perp_maxs, mix_vals = [], [], [], []
    alias_means, noise_means = [], []
    samples = []
    for b in bundles:
        tele = b.telemetry
        pf_min = tele.get("ka_pf_lambda_min")
        pf_max = tele.get("ka_pf_lambda_max")
        perp_max = tele.get("ka_perp_lambda_max")
        mix = tele.get("ka_operator_mixing_epsilon")
        alias_mean = tele.get("ka_alias_lambda_mean")
        noise_mean = tele.get("ka_noise_lambda_mean")

        samples.append(
            (b.path.name, pf_min, pf_max, perp_max, mix, alias_mean, noise_mean),
        )
        if pf_min is not None:
            pf_mins.append(float(pf_min))
        if pf_max is not None:
            pf_maxs.append(float(pf_max))
        if perp_max is not None:
            perp_maxs.append(float(perp_max))
        if mix is not None:
            mix_vals.append(float(mix))
        if alias_mean is not None:
            alias_means.append(float(alias_mean))
        if noise_mean is not None:
            noise_means.append(float(noise_mean))

    if not samples:
        return ConditionResult(
            name="OC2 Operator constraints",
            status="pending",
            detail="Operator metrics missing.",
            metrics={},
        )
    pf_min_med = float(np.median(pf_mins)) if pf_mins else None
    pf_max_med = float(np.median(pf_maxs)) if pf_maxs else None
    perp_max_med = float(np.median(perp_maxs)) if perp_maxs else None
    mix_med = float(np.median(mix_vals)) if mix_vals else None
    alias_mean_med = float(np.median(alias_means)) if alias_means else None
    noise_mean_med = float(np.median(noise_means)) if noise_means else None

    status = "pass"
    detail_parts = []

    # OC-2: Flow band isotropy
    if pf_min_med is None or pf_min_med < 0.95:
        status = "fail"
        detail_parts.append(f"pf_min median {pf_min_med} (<0.95)")
    if pf_max_med is not None and pf_max_med > 1.10:
        status = "fail"
        detail_parts.append(f"pf_max median {pf_max_med} (>1.10)")

    # OC-1: Mixing
    if mix_med is None or mix_med > 0.05:
        status = "fail"
        detail_parts.append(f"mix median {mix_med} (>0.05)")

    # OC-2: Differential Gain (s_a / s_f >= Lambda)
    # Metric is eigenvalue of R_beta^{-1} R_sample.
    # If s_a / s_f > 1, then R_beta < R_sample in Alias.
    # So alias_mean should be < 1.0.
    # Specifically, if s_a/s_f >= 1.25, then alias_mean <= (1/1.25)^2 = 0.64?
    # Wait, s_a are singular values of Q. Q ~ R_beta^{-1/2}.
    # So s_a^2 ~ R_beta^{-1}.
    # alias_mean ~ s_a^2.
    # So if s_a >= 1.25, alias_mean >= 1.56.
    #
    # Let's re-verify the test case.
    # In test: R_beta = 2.0 (Inflated). Metric = 0.5.
    # If R_beta is inflated, we are SUPPRESSING alias.
    # If we want to INFLATE alias (s_a > 1), we need R_beta < R_sample.
    # Then Metric > 1.0.
    #
    # So if target is s_a / s_f >= 1.25.
    # And s_f ~ 1.0.
    # Then s_a >= 1.25.
    # Then alias_mean (eigenvalue of Q^H Q) >= 1.25^2 = 1.56.
    #
    # So we expect alias_mean >= 1.56.

    if alias_mean_med is not None:
        if alias_mean_med < 1.5:
            status = "fail"
            detail_parts.append(f"alias_mean median {alias_mean_med} (<1.5)")

        # Check Noise floor
        if noise_mean_med is not None and noise_mean_med > 1.10:
            status = "fail"
            detail_parts.append(f"noise_mean median {noise_mean_med} (>1.10)")

        # If we have alias/noise separation, we ignore perp_max check
        # because perp_max will be alias_max (>= 1.5).
    else:
        # Legacy check
        if perp_max_med is None or perp_max_med > 1.10:
            status = "fail"
            detail_parts.append(f"perp_max median {perp_max_med} (>1.10)")

    if not detail_parts:
        detail_parts.append("All band/mixing metrics within targets.")

    return ConditionResult(
        name="OC2 Operator constraints",
        status=status,
        detail="; ".join(detail_parts),
        metrics={"samples": samples},
    )


def check_seed_count(bundles: List[BundleInfo], min_seeds: int) -> ConditionResult:
    seeds = {b.seed for b in bundles if b.seed is not None}
    count = len(seeds)
    if count >= min_seeds:
        status = "pass"
    else:
        status = "pending"
    detail = f"{count} unique seeds found; doc requires {min_seeds} for low-FPR ROC/EVT."
    return ConditionResult(
        name="E4 Effective negatives",
        status=status,
        detail=detail,
        metrics={"unique_seeds": count},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check KA-STAP feasibility telemetry.")
    parser.add_argument(
        "--bundles",
        nargs="+",
        required=True,
        help="Paths to bundle directories (each containing meta.json).",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=12,
        help="Minimum number of unique seeds required for full low-FPR evaluation.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path to write JSON report.",
    )
    args = parser.parse_args()

    try:
        bundles = [load_bundle(b) for b in args.bundles]
    except Exception as exc:
        print(f"Error loading bundles: {exc}", file=sys.stderr)
        return 2

    if not bundles:
        print("No bundles provided.", file=sys.stderr)
        return 2

    results: List[ConditionResult] = []
    results.append(check_seed_count(bundles, args.min_seeds))
    results.append(check_flow_alignment(bundles))
    results.append(check_flow_motion_angle(bundles))
    results.append(check_psd_alignment(bundles))
    results.append(check_ka_snr(bundles))
    results.append(check_ka_noise_ratio(bundles))
    results.append(check_gating(bundles))
    results.append(check_rank_consistency(bundles))
    results.append(check_score_transform(bundles))
    results.append(check_operator_metrics(bundles))

    summary = {
        "bundle_count": len(bundles),
        "conditions": [r.to_dict() for r in results],
    }
    fail_present = any(r.status == "fail" for r in results)

    for res in results:
        print(f"[{res.status.upper():7}] {res.name}: {res.detail}")

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"Wrote JSON report to {args.json_out}")

    if fail_present:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
