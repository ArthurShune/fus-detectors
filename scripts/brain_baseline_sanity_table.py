#!/usr/bin/env python3
"""
Brain-* baseline sanity table (strict vs relaxed operating points).

Purpose
-------
Reviewer-facing sanity check: demonstrate that PD-after-filter baselines are not
degenerate (they can achieve nontrivial TPR at relaxed operating points) while
still collapsing to TPR≈0 at strict-tail operating points when the background
right tail is enormous.

This script reads the fair-filter matrix bundles (reports/fair_matrix_*.json)
and recomputes per-window tail operating points directly from the saved score
maps and masks, then summarizes across the five disjoint 64-frame windows as
median and IQR.

Outputs (tracked):
  - reports/brain_baseline_sanity_relaxed.csv
  - reports/brain_baseline_sanity_relaxed.json
  - reports/brain_baseline_sanity_relaxed_table.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]


def _finite_flat(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _right_tail_threshold(bg_scores: np.ndarray, alpha: float) -> tuple[float, float]:
    bg = _finite_flat(bg_scores)
    n = int(bg.size)
    if n <= 0:
        raise ValueError("Empty background score pool.")
    a = float(alpha)
    if not np.isfinite(a) or a <= 0.0:
        return float("inf"), 0.0
    if a >= 1.0:
        tau = float(np.min(bg))
        return tau, 1.0
    k = int(math.ceil(a * n))
    k = max(1, min(k, n))
    tau = float(np.partition(bg, n - k)[n - k])
    realized = float(np.mean(bg >= tau))
    return tau, realized


def _quantile_summary(vals: Iterable[float]) -> tuple[float, float, float]:
    v = np.asarray(list(vals), dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size <= 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.quantile(v, 0.5))
    q25 = float(np.quantile(v, 0.25))
    q75 = float(np.quantile(v, 0.75))
    return med, q25, q75


def _fmt_cell(med: float, q25: float, q75: float) -> str:
    if not (np.isfinite(med) and np.isfinite(q25) and np.isfinite(q75)):
        return "nan"
    return f"{med:.4f} ({q25:.4f},{q75:.4f})"


@dataclass(frozen=True)
class WindowPoint:
    regime: str
    method: str
    scenario: str
    bundle_dir: str
    alpha: float
    thr: float
    fpr_realized: float
    tpr: float


def _score_path_for_method(bundle_dir: Path, method: str) -> Path:
    # In the fair-matrix JSON, STAP rows point at the MC--SVD bundle directory.
    if method.strip().lower().startswith("stap"):
        return bundle_dir / "score_stap_preka.npy"
    return bundle_dir / "score_base.npy"


def _load_bool(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr.astype(bool, copy=False)


def _load_score(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr.astype(np.float64, copy=False)


def _compute_window_points(
    *,
    bundle_dir: Path,
    method: str,
    regime: str,
    scenario: str,
    alphas: list[float],
) -> list[WindowPoint]:
    mb = _load_bool(bundle_dir / "mask_bg.npy")
    mf = _load_bool(bundle_dir / "mask_flow.npy")
    score_path = _score_path_for_method(bundle_dir, method)
    if not score_path.is_file():
        raise FileNotFoundError(f"Missing score file for {method}: {score_path}")
    score = _load_score(score_path)
    if score.shape != mb.shape or score.shape != mf.shape:
        raise ValueError(f"Shape mismatch in {bundle_dir}: score={score.shape} bg={mb.shape} flow={mf.shape}")

    bg = score[mb]
    flow = score[mf]
    flow_finite = _finite_flat(flow)
    out: list[WindowPoint] = []
    for a in alphas:
        thr, fpr_realized = _right_tail_threshold(bg, a)
        tpr = float(np.mean(flow_finite >= float(thr))) if np.isfinite(thr) else 0.0
        out.append(
            WindowPoint(
                regime=str(regime),
                method=str(method),
                scenario=str(scenario),
                bundle_dir=str(bundle_dir),
                alpha=float(a),
                thr=float(thr),
                fpr_realized=float(fpr_realized),
                tpr=float(tpr),
            )
        )
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fair-matrix-json",
        type=Path,
        default=Path("reports/fair_matrix_vnext_r3_localbaselines.json"),
        help="Input fair-matrix JSON (list of per-window rows).",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="0.001,0.01,0.1",
        help="Comma-separated alpha targets to report (default: %(default)s).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/brain_baseline_sanity_relaxed.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/brain_baseline_sanity_relaxed.json"),
        help="Output JSON path (default: %(default)s).",
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=Path("reports/brain_baseline_sanity_relaxed_table.tex"),
        help="Output LaTeX table path (default: %(default)s).",
    )
    return ap.parse_args()


def _parse_alpha_list(spec: str) -> list[float]:
    parts = [p.strip() for p in str(spec or "").split(",") if p.strip()]
    return [float(p) for p in parts]


def _render_table_tex(
    *,
    out_path: Path,
    cell_map: dict[tuple[str, str, float], tuple[float, float, float]],
    methods: list[str],
    method_display: dict[str, str],
    regimes: list[tuple[str, str]],
    alphas: list[float],
) -> None:
    # Expect exactly three alphas for the table layout: strict + two relaxed anchors.
    if len(alphas) != 3:
        raise ValueError("Table renderer expects exactly three alphas (strict, relaxed1, relaxed2).")
    a_strict, a_relaxed1, a_relaxed2 = float(alphas[0]), float(alphas[1]), float(alphas[2])

    def _alpha_tex(a: float) -> str:
        a = float(a)
        if a > 0.0 and np.isfinite(a):
            k = round(-math.log10(a))
            if abs(a - (10.0 ** (-k))) <= 1e-12:
                return f"10^{{-{int(k)}}}"
        return f"{a:g}"

    def _cell(regime_key: str, method: str, alpha: float) -> str:
        stats = cell_map.get((regime_key, method, float(alpha)))
        if stats is None:
            return "n/a"
        return _fmt_cell(*stats)

    # Table environment (standalone; included via \\input in appendix).
    cols = "l" + "ccc " * len(regimes)
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/brain_baseline_sanity_table.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{cols}}}")
    lines.append("\\hline")
    # Header row.
    header = " & " + " & ".join(
        f"\\multicolumn{{3}}{{c}}{{{title}}}" for title, _key in regimes
    ) + " \\\\"
    lines.append(header)
    sub = "Method / score" + " & " + " & ".join(
        [f"TPR@${_alpha_tex(a_strict)}$", f"TPR@${_alpha_tex(a_relaxed1)}$", f"TPR@${_alpha_tex(a_relaxed2)}$"]
        * len(regimes)
    ) + " \\\\"
    lines.append(sub)
    lines.append("\\hline")
    for method in methods:
        row = [str(method_display.get(method, method))]
        for _title, key in regimes:
            row.append(_cell(key, method, a_strict))
            row.append(_cell(key, method, a_relaxed1))
            row.append(_cell(key, method, a_relaxed2))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Baseline sanity check on labeled Brain-* simulations at a strict-tail operating point "
        f"($\\alpha={_alpha_tex(a_strict)}$) versus relaxed operating points "
        f"($\\alpha\\in\\{{{_alpha_tex(a_relaxed1)},{_alpha_tex(a_relaxed2)}\\}}$). "
        "Entries report median (IQR) TPR over the same five disjoint 64-frame windows used in the main tables, "
        "with thresholds calibrated per window on that window's negative set. The relaxed columns are included only "
        "to verify that PD-after-filter baselines are not degenerate; strict-tail reporting remains the focus.}"
    )
    lines.append("\\label{tab:brain_baseline_sanity_relaxed}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    alphas = _parse_alpha_list(args.alphas)
    if len(alphas) != 3:
        raise SystemExit("--alphas must contain exactly three values (e.g. 0.001,0.01,0.1).")

    fair_path = Path(args.fair_matrix_json)
    rows = json.loads(fair_path.read_text())
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit(f"Expected a list[dict] JSON at {fair_path}")

    # Restrict to the canonical Brain-* fair-matrix methods.
    method_order = [
        "MC-SVD",
        "Adaptive SVD (similarity cutoff)",
        "Local SVD (block-wise)",
        "RPCA",
        "HOSVD",
        "STAP (MC-SVD+STAP, full)",
    ]
    method_display = {
        "MC-SVD": "MC--SVD",
        "Adaptive SVD (similarity cutoff)": "Adaptive SVD (SV similarity cutoff)",
        "Local SVD (block-wise)": "Block-wise local SVD (overlap-add)",
        "RPCA": "RPCA",
        "HOSVD": "HOSVD",
        "STAP (MC-SVD+STAP, full)": "Matched-subspace detector on MC--SVD",
    }
    regimes = [
        ("Brain-OpenSkull", "open"),
        ("Brain-AliasContract", "aliascontract"),
        ("Brain-SkullOR", "skullor"),
    ]
    regime_keys = {k for _t, k in regimes}

    points: list[WindowPoint] = []
    skipped = 0
    for r in rows:
        regime = str(r.get("regime") or "")
        method = str(r.get("method") or "")
        scenario = str(r.get("scenario") or "")
        bundle_dir = r.get("bundle_dir")
        if regime not in regime_keys or method not in method_order or not bundle_dir:
            continue
        bdir = Path(str(bundle_dir))
        try:
            points.extend(
                _compute_window_points(
                    bundle_dir=bdir,
                    method=method,
                    regime=regime,
                    scenario=scenario,
                    alphas=alphas,
                )
            )
        except Exception:
            skipped += 1
            continue

    if not points:
        raise SystemExit("No points computed; check --fair-matrix-json path and expected methods/regimes.")

    # Write per-window CSV.
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(points[0]).keys()))
        w.writeheader()
        for p in points:
            w.writerow(asdict(p))

    # Aggregate (regime, method, alpha) -> TPR stats.
    cell_map: dict[tuple[str, str, float], tuple[float, float, float]] = {}
    summary_rows: list[dict[str, Any]] = []
    for _title, reg_key in regimes:
        for method in method_order:
            for a in alphas:
                vals = [p.tpr for p in points if p.regime == reg_key and p.method == method and p.alpha == float(a)]
                med, q25, q75 = _quantile_summary(vals)
                cell_map[(reg_key, method, float(a))] = (med, q25, q75)
                summary_rows.append(
                    {
                        "regime": reg_key,
                        "method": method,
                        "alpha": float(a),
                        "tpr_median": med,
                        "tpr_q25": q25,
                        "tpr_q75": q75,
                        "n_windows": int(len(vals)),
                    }
                )

    out_json = Path(args.out_json)
    out_json.write_text(
        json.dumps(
            {
                "fair_matrix_json": str(fair_path),
                "alphas": alphas,
                "methods": method_order,
                "regimes": [{"title": t, "key": k} for t, k in regimes],
                "skipped_rows": int(skipped),
                "summary": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    out_tex = Path(args.out_tex)
    _render_table_tex(
        out_path=out_tex,
        cell_map=cell_map,
        methods=method_order,
        method_display=method_display,
        regimes=regimes,
        alphas=alphas,
    )

    print(f"[brain-baseline-sanity] wrote {out_csv}")
    print(f"[brain-baseline-sanity] wrote {out_json}")
    print(f"[brain-baseline-sanity] wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
