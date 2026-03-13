#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.shin_map_routing_analysis import _evaluate_fixed, _evaluate_rule


def _paired_frame_dirs(msd_root: Path, unw_root: Path) -> list[tuple[str, Path, Path]]:
    msd = {p.name: p for p in msd_root.iterdir() if p.is_dir()}
    unw = {p.name: p for p in unw_root.iterdir() if p.is_dir()}
    stems = sorted(set(msd) & set(unw))
    return [(stem, msd[stem], unw[stem]) for stem in stems]


def _rule_payload(
    pairs: list[tuple[str, Path, Path]],
    *,
    name: str,
    feature_name: str,
    direction: str,
    threshold: float,
    alpha: float,
    connectivity: int,
) -> dict[str, Any]:
    out = _evaluate_rule(
        pairs,
        feature_name=feature_name,
        direction=direction,
        threshold=threshold,
        alpha=alpha,
        connectivity=connectivity,
    )
    out["name"] = name
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate fixed hybrid-rescue rules on paired Gammex/Twinkling bundle roots "
            "without rerunning bundle generation."
        )
    )
    ap.add_argument(
        "--along-msd-root",
        type=Path,
        default=Path(
            "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast/"
            "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine"
        ),
    )
    ap.add_argument(
        "--along-unwhitened-root",
        type=Path,
        default=Path(
            "runs/real/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio/"
            "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__along_-_linear_probe___RawBCFCine"
        ),
    )
    ap.add_argument(
        "--across-msd-root",
        type=Path,
        default=Path(
            "runs/real/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast/"
            "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17"
        ),
    )
    ap.add_argument(
        "--across-unwhitened-root",
        type=Path,
        default=Path(
            "runs/real/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio/"
            "data_twinkling_artifact_Flow_in_Gammex_phantom_Flow_in_Gammex_phantom__across_-_linear_probe___RawBCFCine_08062017_145434_17"
        ),
    )
    ap.add_argument("--alpha", type=float, default=1e-3)
    ap.add_argument("--connectivity", type=int, default=4, choices=[4, 8])
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/twinkling_gammex_hybrid_pair_eval.json"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scenario_specs = {
        "along": (args.along_msd_root, args.along_unwhitened_root),
        "across": (args.across_msd_root, args.across_unwhitened_root),
    }
    rules = [
        ("guard_frac_v1", "base_guard_frac_map", ">=", 0.0037712273420765995),
        ("alias_rescue_v1", "base_m_alias_map", ">=", -2.006748914718628),
        ("band_ratio_v1", "base_band_ratio_map", "<=", 2.006748914718628),
    ]

    payload: dict[str, Any] = {
        "alpha": float(args.alpha),
        "connectivity": int(args.connectivity),
        "rules": [r[0] for r in rules],
        "scenarios": {},
    }
    for name, (msd_root, unw_root) in scenario_specs.items():
        pairs = _paired_frame_dirs(msd_root, unw_root)
        if not pairs:
            raise SystemExit(f"No paired frame dirs found for {name}: {msd_root} vs {unw_root}")
        _rows_w, fixed_w = _evaluate_fixed(
            pairs, head="whitened", alpha=float(args.alpha), connectivity=int(args.connectivity)
        )
        _rows_u, fixed_u = _evaluate_fixed(
            pairs, head="unwhitened", alpha=float(args.alpha), connectivity=int(args.connectivity)
        )
        scenario_out = {
            "pair_count": len(pairs),
            "fixed_whitened": fixed_w,
            "fixed_unwhitened": fixed_u,
            "rules": [],
        }
        for rule_name, feature_name, direction, threshold in rules:
            scenario_out["rules"].append(
                _rule_payload(
                    pairs,
                    name=rule_name,
                    feature_name=feature_name,
                    direction=direction,
                    threshold=float(threshold),
                    alpha=float(args.alpha),
                    connectivity=int(args.connectivity),
                )
            )
        payload["scenarios"][name] = scenario_out

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out_json)


if __name__ == "__main__":
    main()
