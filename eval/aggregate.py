"""Aggregate CSV/JSON acceptance results into summary tables."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


def _read_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value  # type: ignore
            rows.append(parsed)
    return rows


def aggregate_tables(
    csv_files: Sequence[str],
    group_cols: Sequence[str],
    out_csv: str,
    out_md: Optional[str] = None,
) -> None:
    records: List[Dict[str, float]] = []
    for csv_path in csv_files:
        records.extend(_read_csv(Path(csv_path)))

    grouped: Dict[tuple, List[Dict[str, float]]] = defaultdict(list)
    for rec in records:
        key = tuple(rec.get(col) for col in group_cols)
        grouped[key].append(rec)

    numeric_keys = {
        key
        for rec in records
        for key, value in rec.items()
        if isinstance(value, (int, float)) and key not in group_cols
    }

    summary_rows: List[Dict[str, float]] = []
    for key, group in grouped.items():
        stats: Dict[str, float] = {col: val for col, val in zip(group_cols, key, strict=False)}
        count = len(group)
        stats["count"] = count
        for metric in numeric_keys:
            values = [float(g[metric]) for g in group if metric in g]
            if not values:
                continue
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / max(len(values) - 1, 1)
            std_val = math.sqrt(variance)
            ci = 1.96 * std_val / math.sqrt(len(values))
            stats[f"{metric}_mean"] = mean_val
            stats[f"{metric}_std"] = std_val
            stats[f"{metric}_ci95"] = ci
        summary_rows.append(stats)

    if not summary_rows:
        Path(out_csv).write_text("")
        if out_md:
            Path(out_md).write_text("")
        return

    fieldnames = sorted(summary_rows[0].keys())
    with Path(out_csv).open("w", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if out_md:
        lines = [
            "| " + " | ".join(fieldnames) + " |",
            "|" + "|".join([" --- "] * len(fieldnames)) + "|",
        ]
        for row in summary_rows:
            values = [str(row.get(col, "")) for col in fieldnames]
            lines.append("| " + " | ".join(values) + " |")
        Path(out_md).write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Aggregate Monte-Carlo/stress CSV logs")
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--group-cols", nargs="+", default=["fpr_target"])
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", default=None)
    return ap


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    aggregate_tables(args.csv, args.group_cols, args.out_csv, args.out_md)


if __name__ == "__main__":
    main()
