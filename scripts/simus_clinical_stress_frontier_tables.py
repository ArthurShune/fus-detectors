#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

AXIS_META = {
    "cardiac_pulsation": {
        "label": r"Cardiac-like pulsation",
        "provenance": (
            "Stress axis motivated by cardiac-like tissue pulsation broadening the low-frequency "
            "tissue spectrum in neonatal and open-skull fUS, with Demen\\'e et al. identifying "
            "roughly 1--10 mm/s tissue motion as the critical operating regime for SVD clutter filtering."
        ),
        "levels": {
            "reference": "1.0x cardiac amplitude at 1.30 Hz",
            "moderate": "2.0x cardiac amplitude at 1.30 Hz",
            "hard": "3.0x cardiac amplitude at 1.30 Hz",
        },
    },
    "short_ensemble": {
        "label": r"Short ensemble",
        "provenance": (
            "Stress axis motivated by clinically constrained acquisitions in which shorter "
            "ensembles are used to limit motion corruption or workflow burden in intra-operative and mobile settings."
        ),
        "levels": {
            "reference": "64 frames (42.7 ms)",
            "moderate": "48 frames (32.0 ms)",
            "hard": "32 frames (21.3 ms)",
        },
    },
}

LEVEL_ORDER = {"reference": 0, "moderate": 1, "hard": 2}
AXIS_ORDER = {"cardiac_pulsation": 0, "short_ensemble": 1}
LEVEL_DISPLAY = {"reference": "Ref.", "moderate": "Mod.", "hard": "Hard"}
PIPELINE_DISPLAY = {
    "RPCA -> PD": r"\shortstack[l]{RPCA\\$\rightarrow$ PD}",
    "RPCA -> Matched-subspace default": r"\shortstack[l]{RPCA\\$\rightarrow$ Fixed matched-subspace}",
    "RPCA -> Adaptive guard": r"\shortstack[l]{RPCA\\$\rightarrow$ Adaptive head}",
    "RPCA -> Whitened specialist": r"\shortstack[l]{RPCA\\$\rightarrow$ Whitened specialist}",
    "Adaptive Global SVD -> Matched-subspace default": r"\shortstack[l]{Adaptive-global SVD\\$\rightarrow$ Fixed matched-subspace}",
    "Adaptive Global SVD -> Adaptive guard": r"\shortstack[l]{Adaptive-global SVD\\$\rightarrow$ Adaptive head}",
    "Adaptive Global SVD -> Whitened specialist": r"\shortstack[l]{Adaptive-global SVD\\$\rightarrow$ Whitened specialist}",
}
DETECTOR_DISPLAY = {
    "unwhitened_ref": "Fixed head",
    "adaptive_guard_huber": "Adaptive head",
    "huber_trim8": "Whitened head",
}
DETECTOR_ORDER = ["unwhitened_ref", "adaptive_guard_huber", "huber_trim8"]
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260316


def _fmt(value: float | str | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _best_by(rows: list[dict[str, str]], *, split: str, role: str, axis_key: str, level: str) -> dict[str, str]:
    subset = [
        row
        for row in rows
        if row["split"] == split and row["role"] == role and row["axis_key"] == axis_key and row["level"] == level
    ]
    if not subset:
        raise ValueError(f"missing rows for split={split} role={role} axis={axis_key} level={level}")
    return max(subset, key=lambda row: float(row["selection_score"]))


def _headline_rank(row: dict[str, str]) -> float:
    return float(row["auc_main_vs_nuisance"]) - float(row["fpr_nuisance_match@0p5"])


def _best_headline_detector(
    rows: list[dict[str, str]], *, split: str, axis_key: str, level: str
) -> dict[str, str]:
    subset = [
        row
        for row in rows
        if row["split"] == split and row["role"] == "detector" and row["axis_key"] == axis_key and row["level"] == level
    ]
    if not subset:
        raise ValueError(f"missing detector rows for split={split} axis={axis_key} level={level}")
    return max(subset, key=_headline_rank)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pipeline_display(label: str) -> str:
    return PIPELINE_DISPLAY.get(label, label.replace(" -> ", r"$\rightarrow$"))


def _pipeline_caption(label: str) -> str:
    return label.replace(" -> ", r" $\rightarrow$ ")


def _score_file_for_row(row: dict[str, str]) -> str:
    if row["role"] == "detector":
        return "score_stap_preka.npy"
    label = str(row.get("pipeline_label", "")).lower()
    detector_head = str(row.get("detector_head", "")).lower()
    baseline_type = str(row.get("baseline_type", "")).lower()
    if "kasai" in label or detector_head == "kasai":
        return "score_base_kasai.npy"
    if detector_head == "pd" or baseline_type in {"rpca", "svd_similarity", "hosvd", "mc_svd", "local_svd"}:
        return "score_base.npy"
    return "score_base.npy"


def _finite(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("AUC requires non-empty positive and negative samples.")
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    equal = right - less
    return float((float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos.size * neg.size))


def _fpr_nuisance_at_matched_tpr(pos: np.ndarray, neg: np.ndarray, tpr: float = 0.5) -> float:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Matched-TPR FPR requires non-empty positive and negative samples.")
    thr = float(np.quantile(pos, float(np.clip(1.0 - tpr, 0.0, 1.0))))
    return float(np.mean(neg >= thr))


def _bootstrap_interval(
    pos: np.ndarray,
    neg: np.ndarray,
    stat_fn,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int,
) -> tuple[float, float]:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("Bootstrap interval requires non-empty samples.")
    rng = np.random.default_rng(seed)
    out = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        pos_s = pos[rng.integers(0, pos.size, size=pos.size)]
        neg_s = neg[rng.integers(0, neg.size, size=neg.size)]
        out[i] = float(stat_fn(pos_s, neg_s))
    lo, hi = np.quantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def _masked_scores(row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    bundle_dir = Path(row["bundle_dir"])
    run_dir = Path(row["run_dir"])
    score = np.load(bundle_dir / _score_file_for_row(row))
    pos_mask = np.load(run_dir / "dataset" / "mask_h1_pf_main.npy").astype(bool, copy=False)
    neg_mask = np.load(run_dir / "dataset" / "mask_h0_nuisance_pa.npy").astype(bool, copy=False)
    return score[pos_mask], score[neg_mask]


def _row_intervals(
    row: dict[str, str],
    cache: dict[tuple[str, str], tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    key_auc = (row["bundle_dir"], "auc_main_vs_nuisance")
    key_fpr = (row["bundle_dir"], "fpr_nuisance_match@0p5")
    if key_auc not in cache or key_fpr not in cache:
        pos, neg = _masked_scores(row)
        seed_base = BOOTSTRAP_SEED + sum(ord(ch) for ch in row["bundle_dir"])
        if key_auc not in cache:
            cache[key_auc] = _bootstrap_interval(pos, neg, _auc_pos_vs_neg, seed=seed_base)
        if key_fpr not in cache:
            cache[key_fpr] = _bootstrap_interval(
                pos,
                neg,
                lambda p, n: _fpr_nuisance_at_matched_tpr(p, n, 0.5),
                seed=seed_base + 1,
            )
    return {
        "auc_main_vs_nuisance": cache[key_auc],
        "fpr_nuisance_match@0p5": cache[key_fpr],
    }


def _fmt_ci(point: float | str | None, ci: tuple[float, float] | None = None) -> str:
    if point is None:
        return "--"
    p = float(point)
    if ci is None:
        return rf"${p:.3f}$"
    lo, hi = ci
    return rf"${p:.3f}_{{{lo:.3f}}}^{{{hi:.3f}}}$"


def _arrow_metric_cell(
    public: dict[str, str],
    detector: dict[str, str],
    *,
    cache: dict[tuple[str, str], tuple[float, float]],
) -> str:
    pub_ci = _row_intervals(public, cache)
    det_ci = _row_intervals(detector, cache)
    return (
        rf"\shortstack[l]{{AUC$_{{\mathrm{{main/nuis}}}}$: "
        rf"{_fmt_ci(public['auc_main_vs_nuisance'], pub_ci['auc_main_vs_nuisance'])} "
        rf"$\rightarrow$ {_fmt_ci(detector['auc_main_vs_nuisance'], det_ci['auc_main_vs_nuisance'])}\\"
        rf"FPR$_{{\mathrm{{nuis}}}}$@$0.5$: "
        rf"{_fmt_ci(public['fpr_nuisance_match@0p5'], pub_ci['fpr_nuisance_match@0p5'])} "
        rf"$\rightarrow$ {_fmt_ci(detector['fpr_nuisance_match@0p5'], det_ci['fpr_nuisance_match@0p5'])}}}"
    )


def _best_detector_by_name(
    rows: list[dict[str, str]],
    *,
    split: str,
    axis_key: str,
    level: str,
    method_family: str,
    detector_name: str,
) -> dict[str, str]:
    subset = [
        row
        for row in rows
        if row["split"] == split
        and row["role"] == "detector"
        and row["axis_key"] == axis_key
        and row["level"] == level
        and row["method_family"] == method_family
        and row["detector_name"] == detector_name
    ]
    if not subset:
        raise ValueError(
            f"missing rows for split={split} axis={axis_key} level={level} "
            f"method_family={method_family} detector_name={detector_name}"
        )
    return max(subset, key=lambda row: float(row["selection_score"]))


def _provenance_table() -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}P{2.8cm} P{4.8cm} P{5.0cm}@{}}",
        r"\hline",
        r"Stress axis & Supplementary perturbation used here & Literature grounding \\",
        r"\hline",
    ]
    for axis_key in ("cardiac_pulsation", "short_ensemble"):
        meta = AXIS_META[axis_key]
        levels = meta["levels"]
        perturb = "; ".join(
            [
                f"Reference: {levels['reference']}",
                f"Moderate: {levels['moderate']}",
                f"Hard: {levels['hard']}",
            ]
        )
        lines.append(
            f"{meta['label']} & {perturb} & {meta['provenance']} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\caption{Literature-grounded stress axes used in the held-out SIMUS stress regime. This analysis fixes attention to the residualizer families that dominated the preliminary reduced-grid sweep (RPCA and adaptive-global SVD), then reruns those clinically motivated perturbations on the held-out mobile setting. These perturbations are intended to approximate clinically relevant nuisance pressure rather than replay any single in vivo acquisition one-to-one.}",
            r"\label{tab:simus_clinical_stress_frontier_provenance}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _headline_table(rows: list[dict[str, str]]) -> str:
    public_labels: list[str] = []
    for axis_key in sorted(AXIS_META, key=lambda k: AXIS_ORDER[k]):
        meta = AXIS_META[axis_key]
        for level in sorted(meta["levels"], key=lambda l: LEVEL_ORDER[l]):
            public = _best_by(rows, split="eval", role="public", axis_key=axis_key, level=level)
            public_labels.append(public["pipeline_label"])
    common_public = public_labels[0] if public_labels and len(set(public_labels)) == 1 else None

    interval_cache: dict[tuple[str, str], tuple[float, float]] = {}

    lines = [
        r"\begin{center}",
        r"\captionsetup{type=table}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
    ]
    for axis_key in sorted(AXIS_META, key=lambda k: AXIS_ORDER[k]):
        meta = AXIS_META[axis_key]
        lines.extend(
            [
                rf"\par\medskip\noindent\textbf{{{meta['label']}}}\par\medskip",
                r"\begin{tabular}{@{}P{4.4cm} P{3.7cm} P{5.1cm}@{}}",
                r"\hline",
                r"Level / perturbation & Best detector-family stack & Metrics (Baseline $\rightarrow$ Detector) \\",
                r"\hline",
            ]
        )
        for level in sorted(meta["levels"], key=lambda l: LEVEL_ORDER[l]):
            detector = _best_headline_detector(rows, split="eval", axis_key=axis_key, level=level)
            public = _best_by(rows, split="eval", role="public", axis_key=axis_key, level=level)
            condition_cell = rf"\shortstack[l]{{{LEVEL_DISPLAY[level]}\\{meta['levels'][level]}}}"
            metrics_cell = _arrow_metric_cell(public, detector, cache=interval_cache)
            lines.append(" & ".join([condition_cell, _pipeline_display(detector["pipeline_label"]), metrics_cell]) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}"])
        if axis_key != "short_ensemble":
            lines.append(r"\medskip")
    lines.extend(
        [
            rf"\caption{{Held-out SIMUS mobile stress regime. Each row compares the detector-family stack with the strongest nuisance-control summary on the displayed endpoints (higher AUC$_{{\mathrm{{main/nuis}}}}$, lower FPR$_{{\mathrm{{nuis}}}}$@$0.5$) against the best conventional baseline among evaluated methods on the same held-out evaluation seed.{(' On every row in this reduced table, that baseline was ' + _pipeline_caption(common_public) + '.') if common_public else ''} Point estimates are shown as $x_{{\mathrm{{lo}}}}^{{\mathrm{{hi}}}}$ with 95\% nonparametric bootstrap intervals over the held-out masked score samples. Because this table uses one held-out evaluation seed per axis, these intervals describe within-seed score uncertainty rather than between-seed variation. The held-out reference regime is shown separately to identify the fixed default detector head.}}",
            r"\label{tab:simus_stress_frontier_headline}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def _frozen_rpca_heads_table(rows: list[dict[str, str]]) -> str:
    public_rows = [
        _best_by(rows, split="eval", role="public", axis_key=axis_key, level=level)
        for axis_key in sorted(AXIS_META, key=lambda k: AXIS_ORDER[k])
        for level in sorted(AXIS_META[axis_key]["levels"], key=lambda l: LEVEL_ORDER[l])
    ]
    if not public_rows or any(row["pipeline_label"] != "RPCA -> PD" for row in public_rows):
        raise ValueError("Frozen RPCA stress-head table expects RPCA -> PD to be the public reference on every row.")

    interval_cache: dict[tuple[str, str], tuple[float, float]] = {}

    lines = [
        r"\begin{center}",
        r"\captionsetup{type=table}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
    ]
    for axis_key in sorted(AXIS_META, key=lambda k: AXIS_ORDER[k]):
        meta = AXIS_META[axis_key]
        lines.extend(
            [
                rf"\par\medskip\noindent\textbf{{{meta['label']}}}\par\medskip",
                r"\begin{tabular}{@{}P{3.9cm} P{2.3cm} P{2.3cm} P{2.3cm} P{2.3cm}@{}}",
                r"\hline",
                r"Level / perturbation & Baseline (RPCA$\rightarrow$PD) & Fixed head & Adaptive head & Whitened head \\",
                r"\hline",
            ]
        )
        for level in sorted(meta["levels"], key=lambda l: LEVEL_ORDER[l]):
            condition_cell = rf"\shortstack[l]{{{LEVEL_DISPLAY[level]}\\{meta['levels'][level]}}}"
            public = _best_by(rows, split="eval", role="public", axis_key=axis_key, level=level)
            row_cells = [condition_cell]
            public_ci = _row_intervals(public, interval_cache)
            row_cells.append(
                rf"\shortstack[l]{{{_fmt_ci(public['auc_main_vs_nuisance'], public_ci['auc_main_vs_nuisance'])}\\{_fmt_ci(public['fpr_nuisance_match@0p5'], public_ci['fpr_nuisance_match@0p5'])}}}"
            )
            for detector_name in DETECTOR_ORDER:
                detector = _best_detector_by_name(
                    rows,
                    split="eval",
                    axis_key=axis_key,
                    level=level,
                    method_family="rpca",
                    detector_name=detector_name,
                )
                det_ci = _row_intervals(detector, interval_cache)
                row_cells.append(
                    rf"\shortstack[l]{{{_fmt_ci(detector['auc_main_vs_nuisance'], det_ci['auc_main_vs_nuisance'])}\\{_fmt_ci(detector['fpr_nuisance_match@0p5'], det_ci['fpr_nuisance_match@0p5'])}}}"
                )
            lines.append(" & ".join(row_cells) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}"])
        if axis_key != "short_ensemble":
            lines.append(r"\medskip")
    lines.extend(
        [
            r"\caption{Concrete detector-head view of the held-out SIMUS mobile stress regime on the same rows as Table~\ref{tab:simus_stress_frontier_headline}, now fixing the residualizer to RPCA on every row. Entries report AUC$_{\mathrm{main/nuis}}$ (top) and FPR$_{\mathrm{nuis}}$@$0.5$ (bottom), with point estimates shown as $x_{\mathrm{lo}}^{\mathrm{hi}}$ using 95\% nonparametric bootstrap intervals over the held-out masked score samples. RPCA is used because RPCA $\rightarrow$ PD is the best conventional baseline among evaluated methods on every stress row in Table~\ref{tab:simus_stress_frontier_headline}. This table separates the fixed, adaptive, and fully whitened heads directly instead of reporting only the best family member on each row.}",
            r"\label{tab:simus_stress_frontier_rpca_heads}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate journal-style tables for the SIMUS stress frontier audit.")
    ap.add_argument(
        "--cardiac-json",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_cardiac_eval_paper_targeted.json"),
    )
    ap.add_argument(
        "--cardiac-csv",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_cardiac_eval_paper_targeted.csv"),
    )
    ap.add_argument(
        "--short-json",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_short_eval_paper_targeted.json"),
    )
    ap.add_argument(
        "--short-csv",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_short_eval_paper_targeted.csv"),
    )
    ap.add_argument(
        "--out-provenance",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_provenance_table.tex"),
    )
    ap.add_argument(
        "--out-headline",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_headline_table.tex"),
    )
    ap.add_argument(
        "--out-rpca-heads",
        type=Path,
        default=Path("reports/simus_clinical_stress_frontier_rpca_heads_table.tex"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cardiac_rows = _load_rows(Path(args.cardiac_csv))
    short_rows = _load_rows(Path(args.short_csv))
    rows = cardiac_rows + short_rows

    # Sanity check that the requested JSON/CSV pairs are aligned.
    for json_name in (Path(args.cardiac_json), Path(args.short_json)):
        payload = json.loads(json_name.read_text(encoding="utf-8"))
        axes = tuple(payload.get("axes") or ())
        if len(axes) != 1:
            raise ValueError(f"{json_name} should contain a single stress axis, found {axes!r}")

    _write(Path(args.out_provenance), _provenance_table())
    _write(Path(args.out_headline), _headline_table(rows))
    _write(Path(args.out_rpca_heads), _frozen_rpca_heads_table(rows))


if __name__ == "__main__":
    main()
