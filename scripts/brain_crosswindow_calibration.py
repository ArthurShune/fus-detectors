#!/usr/bin/env python3
"""
Cross-window threshold calibration audit for labeled Brain-* k-Wave bundles.

Motivation: the main paper reports strict low-FPR operating points by selecting a
threshold on each window's negative set (H0) and evaluating TPR on that same
window. This script quantifies how those operating points transfer across
disjoint windows by calibrating a threshold on window A negatives and applying
it to window B.

Inputs:
  - runs/pilot/fair_filter_matrix_pd_r3_localbaselines/*_mcsvd_full/pw_*_win*_off*
    score_stap_preka.npy (+ mask_flow.npy / mask_bg.npy)
  - corresponding *_rpca_full and *_hosvd_full bundles for baseline-only methods

Outputs:
  - CSV/JSON summaries (median + IQR over windows/pairs) for within-window,
    pairwise cross-window, and leave-one-window-out fixed-calibration transfer
    at specified FPR targets.
  - LaTeX tabular fragment for the STAP rows used in the paper/appendix.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _glob_windows(root: Path) -> list[Path]:
    wins = [p for p in root.glob("pw_*_win*_off*") if p.is_dir()]
    wins.sort()
    return wins


def _load_scores(bundle_dir: Path, kind: str) -> tuple[np.ndarray, np.ndarray]:
    kind = kind.lower().strip()
    if kind == "base":
        pos = np.load(bundle_dir / "base_pos.npy").astype(np.float64, copy=False).ravel()
        neg = np.load(bundle_dir / "base_neg.npy").astype(np.float64, copy=False).ravel()
        return pos, neg
    if kind == "stap":
        score = np.load(bundle_dir / "score_stap_preka.npy").astype(np.float64, copy=False)
        mf = np.load(bundle_dir / "mask_flow.npy").astype(bool, copy=False)
        mb = np.load(bundle_dir / "mask_bg.npy").astype(bool, copy=False)
        pos = score[mf].ravel()
        neg = score[mb].ravel()
        return pos, neg
    raise ValueError(f"Unsupported kind: {kind!r}")


def _tau_for_fpr(neg: np.ndarray, alpha: float) -> float:
    """Right-tail threshold tau so that P(neg >= tau) ~= alpha."""
    n = int(neg.size)
    if n <= 0:
        return float("inf")
    # Order-statistic threshold; ties can make realized FPR exceed target slightly.
    neg_sorted = np.sort(neg)  # ascending
    q = 1.0 - float(alpha)
    k = int(np.ceil(q * n)) - 1
    k = max(0, min(n - 1, k))
    return float(neg_sorted[k])


def _eval_at_tau(pos: np.ndarray, neg: np.ndarray, tau: float) -> tuple[float, float]:
    if not np.isfinite(tau):
        return 0.0, 0.0
    tpr = float(np.mean(pos >= tau)) if pos.size else 0.0
    fpr = float(np.mean(neg >= tau)) if neg.size else 0.0
    return tpr, fpr


def _summarize(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan")}
    return {
        "median": float(np.quantile(arr, 0.5)),
        "q25": float(np.quantile(arr, 0.25)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Brain-* cross-window threshold calibration audit.")
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs/pilot/fair_filter_matrix_pd_r3_localbaselines"),
        help="Root containing per-regime run folders (default: %(default)s).",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default="1e-4,3e-4,1e-3",
        help="Comma-separated FPR targets (default: %(default)s).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/brain_crosswindow_calibration.csv"),
        help="Output CSV path.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/brain_crosswindow_calibration_summary.json"),
        help="Output JSON path.",
    )
    ap.add_argument(
        "--out-tex",
        type=Path,
        default=Path("reports/brain_crosswindow_calibration_table.tex"),
        help="Output LaTeX tabular fragment path.",
    )
    args = ap.parse_args()

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    runs_root = Path(args.runs_root)
    regimes = [
        ("Brain-OpenSkull", runs_root / "open_seed1_mcsvd_full", runs_root / "open_seed1_rpca_full", runs_root / "open_seed1_hosvd_full"),
        ("Brain-AliasContract", runs_root / "aliascontract_seed2_mcsvd_full", runs_root / "aliascontract_seed2_rpca_full", runs_root / "aliascontract_seed2_hosvd_full"),
        ("Brain-SkullOR", runs_root / "skullor_seed2_mcsvd_full", runs_root / "skullor_seed2_rpca_full", runs_root / "skullor_seed2_hosvd_full"),
    ]

    methods = [
        ("MC--SVD (PD)", "mcsvd", "base", 0),
        ("RPCA (PD)", "rpca", "base", 1),
        ("HOSVD (PD)", "hosvd", "base", 2),
        ("STAP (pre-KA) on MC--SVD", "stap", "stap", 0),
    ]

    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"runs_root": str(args.runs_root), "alphas": alphas, "regimes": []}

    for regime_name, d_mcsvd, d_rpca, d_hosvd in regimes:
        roots = [d_mcsvd, d_rpca, d_hosvd]
        win_dirs = [_glob_windows(r) for r in roots]
        if not (win_dirs[0] and win_dirs[1] and win_dirs[2]):
            raise SystemExit(f"Missing windows for {regime_name}: {[len(w) for w in win_dirs]}")
        # Use the MC--SVD window list as the canonical ordering.
        n_win = len(win_dirs[0])
        if any(len(w) != n_win for w in win_dirs):
            raise SystemExit(f"Window count mismatch for {regime_name}: {[len(w) for w in win_dirs]}")

        reg_out: dict[str, object] = {"name": regime_name, "methods": {}}

        for method_label, method_key, kind, root_idx in methods:
            wins = win_dirs[root_idx]

            within_tpr = np.zeros((n_win, len(alphas)), dtype=np.float64)
            within_fpr = np.zeros((n_win, len(alphas)), dtype=np.float64)
            for i, d in enumerate(wins):
                pos_i, neg_i = _load_scores(d, kind)
                for a_idx, alpha in enumerate(alphas):
                    tau = _tau_for_fpr(neg_i, alpha)
                    tpr, fpr = _eval_at_tau(pos_i, neg_i, tau)
                    within_tpr[i, a_idx] = tpr
                    within_fpr[i, a_idx] = fpr

            # Cross-window: calibrate on i, evaluate on j != i.
            cross_tpr: list[list[float]] = [[] for _ in alphas]
            cross_fpr: list[list[float]] = [[] for _ in alphas]
            for i, d_i in enumerate(wins):
                pos_i, neg_i = _load_scores(d_i, kind)
                taus = [ _tau_for_fpr(neg_i, alpha) for alpha in alphas ]
                for j, d_j in enumerate(wins):
                    if j == i:
                        continue
                    pos_j, neg_j = _load_scores(d_j, kind)
                    for a_idx, tau in enumerate(taus):
                        tpr, fpr = _eval_at_tau(pos_j, neg_j, tau)
                        cross_tpr[a_idx].append(tpr)
                        cross_fpr[a_idx].append(fpr)

            # Fixed calibration set: leave one window out, calibrate on pooled
            # negatives from the remaining windows, evaluate on the held-out window.
            fixed_tpr: list[list[float]] = [[] for _ in alphas]
            fixed_fpr: list[list[float]] = [[] for _ in alphas]
            posneg_by_win = [_load_scores(d, kind) for d in wins]
            for i, _ in enumerate(wins):
                pos_i, neg_i = posneg_by_win[i]
                neg_pool = np.concatenate([posneg_by_win[j][1] for j in range(n_win) if j != i])
                taus = [_tau_for_fpr(neg_pool, alpha) for alpha in alphas]
                for a_idx, tau in enumerate(taus):
                    tpr, fpr = _eval_at_tau(pos_i, neg_i, tau)
                    fixed_tpr[a_idx].append(tpr)
                    fixed_fpr[a_idx].append(fpr)

            # Summaries for CSV/JSON.
            meth_out: dict[str, object] = {"label": method_label, "within": {}, "cross": {}}
            meth_out["fixed_cal"] = {}
            for a_idx, alpha in enumerate(alphas):
                w_tpr = within_tpr[:, a_idx]
                w_fpr = within_fpr[:, a_idx]
                c_tpr = np.asarray(cross_tpr[a_idx], dtype=np.float64)
                c_fpr = np.asarray(cross_fpr[a_idx], dtype=np.float64)
                f_tpr = np.asarray(fixed_tpr[a_idx], dtype=np.float64)
                f_fpr = np.asarray(fixed_fpr[a_idx], dtype=np.float64)

                meth_out["within"][str(alpha)] = {
                    "tpr": _summarize(w_tpr),
                    "fpr": _summarize(w_fpr),
                    "n": int(w_tpr.size),
                }
                meth_out["cross"][str(alpha)] = {
                    "tpr": _summarize(c_tpr),
                    "fpr": _summarize(c_fpr),
                    "n": int(c_tpr.size),
                }
                meth_out["fixed_cal"][str(alpha)] = {
                    "tpr": _summarize(f_tpr),
                    "fpr": _summarize(f_fpr),
                    "n": int(f_tpr.size),
                }

                rows.append(
                    {
                        "regime": regime_name,
                        "method": method_key,
                        "method_label": method_label,
                        "split": "within",
                        "alpha": alpha,
                        "tpr_median": float(np.quantile(w_tpr, 0.5)),
                        "tpr_q25": float(np.quantile(w_tpr, 0.25)),
                        "tpr_q75": float(np.quantile(w_tpr, 0.75)),
                        "fpr_median": float(np.quantile(w_fpr, 0.5)),
                        "fpr_q25": float(np.quantile(w_fpr, 0.25)),
                        "fpr_q75": float(np.quantile(w_fpr, 0.75)),
                        "n": int(w_tpr.size),
                    }
                )
                rows.append(
                    {
                        "regime": regime_name,
                        "method": method_key,
                        "method_label": method_label,
                        "split": "cross",
                        "alpha": alpha,
                        "tpr_median": float(np.quantile(c_tpr, 0.5)) if c_tpr.size else float("nan"),
                        "tpr_q25": float(np.quantile(c_tpr, 0.25)) if c_tpr.size else float("nan"),
                        "tpr_q75": float(np.quantile(c_tpr, 0.75)) if c_tpr.size else float("nan"),
                        "fpr_median": float(np.quantile(c_fpr, 0.5)) if c_fpr.size else float("nan"),
                        "fpr_q25": float(np.quantile(c_fpr, 0.25)) if c_fpr.size else float("nan"),
                        "fpr_q75": float(np.quantile(c_fpr, 0.75)) if c_fpr.size else float("nan"),
                        "n": int(c_tpr.size),
                    }
                )
                rows.append(
                    {
                        "regime": regime_name,
                        "method": method_key,
                        "method_label": method_label,
                        "split": "fixed_cal",
                        "alpha": alpha,
                        "tpr_median": float(np.quantile(f_tpr, 0.5)) if f_tpr.size else float("nan"),
                        "tpr_q25": float(np.quantile(f_tpr, 0.25)) if f_tpr.size else float("nan"),
                        "tpr_q75": float(np.quantile(f_tpr, 0.75)) if f_tpr.size else float("nan"),
                        "fpr_median": float(np.quantile(f_fpr, 0.5)) if f_fpr.size else float("nan"),
                        "fpr_q25": float(np.quantile(f_fpr, 0.25)) if f_fpr.size else float("nan"),
                        "fpr_q75": float(np.quantile(f_fpr, 0.75)) if f_fpr.size else float("nan"),
                        "n": int(f_tpr.size),
                    }
                )

            reg_out["methods"][method_key] = meth_out

        summary["regimes"].append(reg_out)

    # Write CSV.
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv

    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write JSON.
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")

    # Write LaTeX table fragment for the STAP rows (used in the paper appendix).
    def _fmt_alpha_tex(a: float) -> str:
        a = float(a)
        if abs(a - 1e-4) <= 1e-12:
            return "$10^{-4}$"
        if abs(a - 3e-4) <= 1e-12:
            return "$3\\times 10^{-4}$"
        if abs(a - 1e-3) <= 1e-12:
            return "$10^{-3}$"
        if a > 0.0 and np.isfinite(a):
            e = int(round(math.log10(a)))
            if abs(a - (10.0**e)) <= 1e-12:
                return f"$10^{{{e}}}$"
        return f"${a:g}$"

    def _fmt_tpr_cell(d: dict[str, float]) -> str:
        med, q25, q75 = float(d["median"]), float(d["q25"]), float(d["q75"])
        if not (np.isfinite(med) and np.isfinite(q25) and np.isfinite(q75)):
            return "nan"
        return f"{med:.4f} ({q25:.4f},{q75:.4f})"

    def _fmt_sci(x: float) -> str:
        x = float(x)
        if not np.isfinite(x):
            return "nan"
        if x == 0.0:
            return "0"
        return f"{x:.2e}"

    def _fmt_fpr_cell(d: dict[str, float]) -> str:
        med, q25, q75 = float(d["median"]), float(d["q25"]), float(d["q75"])
        return f"{_fmt_sci(med)} ({_fmt_sci(q25)},{_fmt_sci(q75)})"

    regimes_in_order = [r[0] for r in regimes]
    stap_by_regime: dict[str, dict[str, object]] = {}
    for reg in summary["regimes"]:
        name = str(reg.get("name"))
        methods_dict = reg.get("methods", {})
        stap = methods_dict.get("stap")
        if stap is not None:
            stap_by_regime[name] = stap

    tex_lines: list[str] = []
    tex_lines.append("% AUTO-GENERATED by scripts/brain_crosswindow_calibration.py; DO NOT EDIT BY HAND.")
    regime_display = {
        "Brain-OpenSkull": "Open-skull",
        "Brain-AliasContract": "Alias-dominant",
        "Brain-SkullOR": "Structured clutter",
    }

    tex_lines.append("\\resizebox{\\linewidth}{!}{%")
    tex_lines.append("\\begin{tabular}{lcccccc}")
    tex_lines.append("\\hline")
    tex_lines.append(
        "Regime & $\\alpha$ & \\shortstack{within-window\\\\TPR med (IQR)} & \\shortstack{pairwise cross\\\\TPR med (IQR)} & \\shortstack{pairwise cross\\\\FPR med (IQR)} & \\shortstack{fixed-cal-set\\\\TPR med (IQR)} & \\shortstack{fixed-cal-set\\\\FPR med (IQR)} \\\\"
    )
    tex_lines.append("\\hline")
    for regime_name in regimes_in_order:
        stap = stap_by_regime.get(regime_name)
        if stap is None:
            raise SystemExit(f"Missing STAP method for table generation: {regime_name}")
        for alpha in alphas:
            k = str(alpha)
            within = stap["within"][k]
            cross = stap["cross"][k]
            fixed_cal = stap["fixed_cal"][k]
            within_tpr = within["tpr"]
            cross_tpr = cross["tpr"]
            cross_fpr = cross["fpr"]
            fixed_tpr = fixed_cal["tpr"]
            fixed_fpr = fixed_cal["fpr"]
            tex_lines.append(
                f"{regime_display.get(regime_name, regime_name)} & {_fmt_alpha_tex(alpha)} & {_fmt_tpr_cell(within_tpr)} & {_fmt_tpr_cell(cross_tpr)} & {_fmt_fpr_cell(cross_fpr)} & {_fmt_tpr_cell(fixed_tpr)} & {_fmt_fpr_cell(fixed_fpr)} \\\\"
            )
    tex_lines.append("\\hline")
    tex_lines.append("\\end{tabular}%")
    tex_lines.append("}")

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(tex_lines) + "\n")

    print(f"[brain-crosswindow] wrote {args.out_csv}, {args.out_json}, {args.out_tex}")


if __name__ == "__main__":
    main()
