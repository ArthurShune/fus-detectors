#!/usr/bin/env python3
"""
Summarize a same-residual Gammex detector-component sweep into a compact paper table.

Inputs are pooled structural-summary JSON files produced by
`scripts/twinkling_eval_structural.py` for:
  - the default STAP detector (`msd_ratio`)
  - the whitened-power detector ablation
  - the unwhitened-ratio detector ablation

The baseline PD / Kasai rows are read from the default summary because they are
identical across detector-variant reruns on the same residual cube.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Cell:
    view: str
    method: str
    fpr: float
    tpr: float | None
    tpr_ci_lo: float | None
    tpr_ci_hi: float | None
    fpr_realized: float | None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--along-default",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_msd_ratio_fast_structural_summary.json"),
        help="Along-view default STAP summary JSON.",
    )
    ap.add_argument(
        "--along-whitened-power",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_whitened_power_structural_summary.json"),
        help="Along-view whitened-power detector summary JSON.",
    )
    ap.add_argument(
        "--along-unwhitened-ratio",
        type=Path,
        default=Path("reports/twinkling_gammex_alonglinear17_prf2500_str6_unwhitened_ratio_structural_summary.json"),
        help="Along-view unwhitened-ratio detector summary JSON.",
    )
    ap.add_argument(
        "--across-default",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_msd_ratio_fast_structural_summary.json"),
        help="Across-view default STAP summary JSON.",
    )
    ap.add_argument(
        "--across-whitened-power",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_whitened_power_structural_summary.json"),
        help="Across-view whitened-power detector summary JSON.",
    )
    ap.add_argument(
        "--across-unwhitened-ratio",
        type=Path,
        default=Path("reports/twinkling_gammex_across17_prf2500_str4_unwhitened_ratio_structural_summary.json"),
        help="Across-view unwhitened-ratio detector summary JSON.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/twinkling_gammex_detector_ablation.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/twinkling_gammex_detector_ablation.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=Path("reports/twinkling_gammex_detector_ablation_table.tex"),
        help="Output LaTeX table path.",
    )
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _method_block(summary: dict[str, Any], method_key: str) -> dict[str, Any]:
    pooled = summary.get("pooled_roc")
    if not isinstance(pooled, dict):
        raise ValueError("Missing pooled_roc")
    methods = pooled.get("methods")
    if not isinstance(methods, dict):
        raise ValueError("Missing pooled_roc.methods")
    block = methods.get(method_key)
    if not isinstance(block, dict):
        raise ValueError(f"Missing pooled_roc.methods[{method_key!r}]")
    return block


def _point_for_fpr(block: dict[str, Any], fpr_target: float) -> dict[str, Any]:
    roc = block.get("roc")
    if not isinstance(roc, list):
        raise ValueError("Method block missing roc list")
    best: dict[str, Any] | None = None
    for pt in roc:
        if not isinstance(pt, dict):
            continue
        try:
            fpr = float(pt.get("fpr_target"))
        except (TypeError, ValueError):
            continue
        if abs(fpr - float(fpr_target)) <= 1e-12:
            best = pt
            break
    if best is None:
        raise ValueError(f"Missing ROC point for fpr_target={fpr_target}")
    return best


def _num(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out


def _extract_cell(summary: dict[str, Any], method_key: str, view: str, fpr: float, label: str) -> Cell:
    block = _method_block(summary, method_key)
    pt = _point_for_fpr(block, fpr)
    ci = pt.get("tpr_ci95")
    ci_lo = None
    ci_hi = None
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        ci_lo = _num(ci[0])
        ci_hi = _num(ci[1])
    return Cell(
        view=view,
        method=label,
        fpr=float(fpr),
        tpr=_num(pt.get("tpr")),
        tpr_ci_lo=ci_lo,
        tpr_ci_hi=ci_hi,
        fpr_realized=_num(pt.get("fpr_realized")),
    )


def _fmt_num(v: float | None) -> str:
    if v is None:
        return "n/a"
    a = abs(float(v))
    if a < 5e-7:
        return "0.00000"
    if a < 1e-3:
        return f"{float(v):.5f}"
    return f"{float(v):.4f}"


def _fmt_sci_tex(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "n/a"
    if v == 0.0:
        return "0"
    s = f"{float(v):.{decimals}e}"
    mantissa, exp = s.split("e")
    exp_i = int(exp)
    mantissa_f = float(mantissa)
    if abs(mantissa_f - 1.0) <= 1e-12:
        return f"10^{{{exp_i}}}"
    mantissa_str = f"{mantissa_f:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{mantissa_str}\\times 10^{{{exp_i}}}"


def _fmt_cell(cell: Cell | None) -> str:
    if cell is None or cell.tpr is None or cell.tpr_ci_lo is None or cell.tpr_ci_hi is None:
        return "n/a"
    return f"{_fmt_num(cell.tpr)} [{_fmt_num(cell.tpr_ci_lo)},{_fmt_num(cell.tpr_ci_hi)}]"


def _alpha_tex(a: float) -> str:
    if abs(a - 1e-4) <= 1e-12:
        return "10^{-4}"
    if abs(a - 3e-4) <= 1e-12:
        return "3\\!\\times\\!10^{-4}"
    if abs(a - 1e-3) <= 1e-12:
        return "10^{-3}"
    return f"{a:g}"


def _render_table(
    out_tex: Path,
    cells: dict[tuple[str, str, float], Cell],
    fprs: list[float],
    along_default: dict[str, Any],
    across_default: dict[str, Any],
) -> None:
    def _cell(view: str, method: str, fpr: float) -> str:
        return _fmt_cell(cells.get((view, method, float(fpr))))

    along_block = _method_block(along_default, "base")
    across_block = _method_block(across_default, "base")
    n_bg_along = _num(along_block.get("n_bg"))
    n_bg_across = _num(across_block.get("n_bg"))
    fpr_min_along = _num(along_block.get("fpr_min"))
    fpr_min_across = _num(across_block.get("fpr_min"))

    method_order = [
        "Baseline (power Doppler)",
        "Baseline (Kasai lag-1 power)",
        "Detector ablation (whitened power)",
        "Detector ablation (unwhitened ratio)",
        "STAP (whitened matched-subspace)",
    ]

    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/twinkling_gammex_detector_ablation_table.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l ccc ccc}")
    lines.append("\\hline")
    lines.append(
        " & \\multicolumn{3}{c}{Along (linear17 @ PRF 2500)} & "
        "\\multicolumn{3}{c}{Across (linear17 @ PRF 2500)} \\\\"
    )
    lines.append(
        "Method / score & "
        + " & ".join([f"TPR@$\\alpha={_alpha_tex(a)}$" for a in fprs])
        + " & "
        + " & ".join([f"TPR@$\\alpha={_alpha_tex(a)}$" for a in fprs])
        + " \\\\"
    )
    lines.append("\\hline")
    for method in method_order:
        row = [method]
        for fpr in fprs:
            row.append(_cell("along", method, fpr))
        for fpr in fprs:
            row.append(_cell("across", method, fpr))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append(
        "\\caption{Gammex flow phantom structural-label operating points using B-mode-derived structural masks. "
        "All scores are right-tail (higher indicates more flow evidence). This is a same-residual detector-head "
        "comparison: every row is evaluated on the identical baseline SVD band-pass residual cube, changing only "
        "the detector head. The baseline rows are power Doppler and Kasai lag-1 power on that residual; the detector "
        "ablations replace the STAP matched-subspace score with either total whitened slow-time power (no Doppler "
        "band partition) or the same flow-band matched-subspace ratio without covariance whitening ($R=I$). "
        "Thresholds are chosen per method on pooled background pixels to match each target FPR and evaluated on "
        "pooled lumen pixels. Brackets show 95\\% frame-bootstrap CIs ($n=2000$ resamples over cine frames) at the "
        "fixed pooled threshold. Pooled background counts are "
        f"$n_{{\\mathrm{{bg}}}}={_fmt_sci_tex(n_bg_along)}$ for the along view "
        f"($fpr_{{\\min}}\\approx {_fmt_sci_tex(fpr_min_along, decimals=1)}$) and "
        f"$n_{{\\mathrm{{bg}}}}={_fmt_sci_tex(n_bg_across)}$ for the across view "
        f"($fpr_{{\\min}}\\approx {_fmt_sci_tex(fpr_min_across, decimals=1)}$). "
        "Reproducibility details are provided in \\SuppOrApp{app:repro}.}"
    )
    lines.append("\\label{tab:twinkling_gammex_structural_roc}")
    lines.append("\\end{table}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    fprs = [1e-4, 3e-4, 1e-3]

    along_default = _load_json(args.along_default)
    along_power = _load_json(args.along_whitened_power)
    along_unwhite = _load_json(args.along_unwhitened_ratio)
    across_default = _load_json(args.across_default)
    across_power = _load_json(args.across_whitened_power)
    across_unwhite = _load_json(args.across_unwhitened_ratio)

    method_specs = [
        ("Baseline (power Doppler)", "base", along_default, across_default),
        ("Baseline (Kasai lag-1 power)", "base_kasai", along_default, across_default),
        ("Detector ablation (whitened power)", "stap_preka", along_power, across_power),
        ("Detector ablation (unwhitened ratio)", "stap_preka", along_unwhite, across_unwhite),
        ("STAP (whitened matched-subspace)", "stap_preka", along_default, across_default),
    ]

    rows: list[Cell] = []
    cell_map: dict[tuple[str, str, float], Cell] = {}
    for label, method_key, along_summary, across_summary in method_specs:
        for fpr in fprs:
            along_cell = _extract_cell(along_summary, method_key, "along", fpr, label)
            across_cell = _extract_cell(across_summary, method_key, "across", fpr, label)
            rows.extend([along_cell, across_cell])
            cell_map[("along", label, float(fpr))] = along_cell
            cell_map[("across", label, float(fpr))] = across_cell

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["view", "method", "fpr", "tpr", "tpr_ci_lo", "tpr_ci_hi", "fpr_realized"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "fprs": fprs,
                "rows": [asdict(r) for r in rows],
                "sources": {
                    "along_default": str(args.along_default),
                    "along_whitened_power": str(args.along_whitened_power),
                    "along_unwhitened_ratio": str(args.along_unwhitened_ratio),
                    "across_default": str(args.across_default),
                    "across_whitened_power": str(args.across_whitened_power),
                    "across_unwhitened_ratio": str(args.across_unwhitened_ratio),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _render_table(args.out_tex, cell_map, fprs, along_default, across_default)


if __name__ == "__main__":
    main()
