#!/usr/bin/env python3
"""
Dump and visualize alias/PSD telemetry for MC-SVD vs KA bundles.

Example:
    PYTHONPATH=. python scripts/r4c_psd_metrics.py \
        --bundles mcsvd=mcsvd_k8_reg4,ka=ka_seed \
        --seeds "1 2 3 4 5"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


TELEMETRY_FIELDS = [
    "psd_alias_ratio_median",
    "psd_alias_ratio_p90",
    "alias_cap_alias_ratio_median",
    "alias_cap_alias_ratio_p90",
    "psd_alias_select_fraction",
    "alias_cap_applied_fraction",
    "median_alias_cap_scale",
    "psd_alias_fraction",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot per-bundle PSD/alias telemetry.")
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("runs/motion/r4c"),
        help="Root directory containing bundle folders.",
    )
    ap.add_argument(
        "--dataset-prefix",
        type=str,
        default="pw_7.5MHz_5ang_5ens_320T_seed",
        help="Dataset prefix inside each bundle folder.",
    )
    ap.add_argument(
        "--bundles",
        type=str,
        default="mcsvd=mcsvd_k8_reg4,ka=ka_seed",
        help="Comma-separated label=prefix entries (prefix + _seed{n}).",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default="1 2 3 4 5",
        help='Space-separated list of seeds (e.g., "1 2 3").',
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("reports/r4c_psd_metrics.png"),
        help="Path for the saved plot.",
    )
    return ap.parse_args()


def load_meta(bundle_dir: Path) -> Dict | None:
    meta_path = bundle_dir / "meta.json"
    if not meta_path.exists():
        print(f"[warn] missing {meta_path}")
        return None
    return json.loads(meta_path.read_text())


def gather_records(
    base_dir: Path,
    dataset_prefix: str,
    bundle_specs: Dict[str, str],
    seeds: Iterable[int],
) -> List[Dict]:
    records: List[Dict] = []
    for label, prefix in bundle_specs.items():
        for seed in seeds:
            if "{seed}" in prefix:
                bundle_name = prefix.format(seed=seed)
            elif prefix.endswith("_seed"):
                bundle_name = f"{prefix}{seed}"
            else:
                bundle_name = f"{prefix}_seed{seed}"
            bundle_dir = base_dir / bundle_name / f"{dataset_prefix}{seed}"
            meta = load_meta(bundle_dir)
            if meta is None:
                continue
            tel = meta.get("stap_fallback_telemetry") or {}
            rec = {"label": label, "seed": seed}
            for field in TELEMETRY_FIELDS:
                val = tel.get(field)
                rec[field] = float(val) if isinstance(val, (int, float)) else None
            records.append(rec)
    return records


def print_table(records: List[Dict]) -> None:
    if not records:
        print("[info] no telemetry records loaded")
        return
    fields = ["label", "seed"] + TELEMETRY_FIELDS
    header = " | ".join(f"{f:>10}" for f in fields)
    print(header)
    print("-" * len(header))
    for rec in records:
        row = []
        for field in fields:
            val = rec.get(field)
            if isinstance(val, float):
                row.append(f"{val:10.3f}")
            elif val is None:
                row.append(f"{'--':>10}")
            else:
                row.append(f"{val:>10}")
        print(" | ".join(row))


def plot_records(records: List[Dict], out_path: Path) -> None:
    if not records:
        return
    labels = sorted({rec["label"] for rec in records})
    seeds = sorted({rec["seed"] for rec in records})
    metrics = [
        ("psd_alias_ratio_median", "Alias ratio median"),
        ("alias_cap_alias_ratio_median", "Alias ratio after cap"),
        ("psd_alias_select_fraction", "PSD alias-select frac"),
        ("alias_cap_applied_fraction", "Alias-cap applied frac"),
    ]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), sharex=True)
    for ax, (field, title) in zip(axes, metrics, strict=False):
        for label in labels:
            y = []
            for seed in seeds:
                rec = next((r for r in records if r["label"] == label and r["seed"] == seed), None)
                y.append(rec.get(field) if rec else None)
            ax.plot(seeds, y, marker="o", label=label)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("Seed")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[info] wrote plot -> {out_path}")


def main() -> None:
    args = parse_args()
    bundle_specs = {}
    for entry in args.bundles.split(","):
        if not entry.strip():
            continue
        label, _, prefix = entry.partition("=")
        bundle_specs[label.strip()] = prefix.strip()
    seed_list = [int(s) for s in args.seeds.split() if s.strip()]
    records = gather_records(args.base_dir, args.dataset_prefix, bundle_specs, seed_list)
    print_table(records)
    plot_records(records, args.out)


if __name__ == "__main__":
    main()
