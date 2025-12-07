#!/usr/bin/env python3
"""
Aggregate robustness-sweep outputs (Phase 1.1) into compact tables.

This script is intentionally simple: it reads the JSON emitted by
scripts/run_brain_sweep.py (--summary-json) and produces:
  - A pivot-style CSV summarizing ΔTPR at each FPR plus latency/coverage.

Example:
    PYTHONPATH=. python scripts/aggregate_brain_sweep.py \
        --summary-json reports/brain_sweep_summary.json \
        --csv reports/brain_sweep_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate brain STAP PD robustness sweeps.")
    ap.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to JSON produced by run_brain_sweep.py.",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Output CSV path for matrix-style summary.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rows: List[Dict[str, object]] = json.loads(args.summary_json.read_text())
    if not rows:
        raise SystemExit(f"No rows found in {args.summary_json}")

    # Collect all keys to keep a stable column order.
    key_set: set[str] = set()
    for row in rows:
        key_set.update(row.keys())

    # Basic, interpretable column ordering.
    preferred_order = [
        "tag",
        "regime",
        "prf_hz",
        "tile",
        "stride",
        "lt",
        "clutter_offset_db",
        "alias_scale",
        "baseline_ms",
        "stap_ms",
        "coverage",
    ]
    # Place ΔTPR columns next, then raw TPRs, then anything else.
    delta_cols = sorted([k for k in key_set if k.startswith("delta_tpr@")])
    tpr_cols = sorted([k for k in key_set if k.startswith("tpr_base@") or k.startswith("tpr_stap@")])
    other_cols = sorted(k for k in key_set if k not in preferred_order + delta_cols + tpr_cols)

    header = preferred_order + delta_cols + tpr_cols + other_cols

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[aggregate] wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()

