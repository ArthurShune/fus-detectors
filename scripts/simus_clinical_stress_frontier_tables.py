#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


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
    "Adaptive Global SVD -> Whitened specialist": r"\shortstack[l]{Adaptive-global SVD\\$\rightarrow$ Whitened specialist}",
}


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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _pipeline_display(label: str) -> str:
    return PIPELINE_DISPLAY.get(label, label.replace(" -> ", r"$\rightarrow$"))


def _pipeline_caption(label: str) -> str:
    return label.replace(" -> ", r" $\rightarrow$ ")


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
            r"\caption{Literature-grounded stress axes used in the held-out paper-tier SIMUS stress confirmation. The paper-tier confirmation fixes attention to the residualizer families that dominated the preliminary reduced-grid sweep (RPCA and adaptive-global SVD), then reruns those clinically motivated perturbations on the held-out paper-tier mobile setting. These perturbations are intended to approximate clinically relevant nuisance pressure, not to claim one-to-one replay of any single in vivo acquisition.}",
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
                r"Level / perturbation & Best detector-family stack & Metrics (Public $\rightarrow$ Detector) \\",
                r"\hline",
            ]
        )
        for level in sorted(meta["levels"], key=lambda l: LEVEL_ORDER[l]):
            detector = _best_by(rows, split="eval", role="detector", axis_key=axis_key, level=level)
            public = _best_by(rows, split="eval", role="public", axis_key=axis_key, level=level)
            condition_cell = rf"\shortstack[l]{{{LEVEL_DISPLAY[level]}\\{meta['levels'][level]}}}"
            metrics_cell = (
                rf"\shortstack[l]{{AUC$_{{\mathrm{{main/nuis}}}}$: "
                rf"{_fmt(public['auc_main_vs_nuisance'])} $\rightarrow$ {_fmt(detector['auc_main_vs_nuisance'])}\\"
                rf"FPR$_{{\mathrm{{nuis}}}}$@$0.5$: "
                rf"{_fmt(public['fpr_nuisance_match@0p5'])} $\rightarrow$ {_fmt(detector['fpr_nuisance_match@0p5'])}}}"
            )
            lines.append(" & ".join([condition_cell, _pipeline_display(detector["pipeline_label"]), metrics_cell]) + r" \\")
        lines.extend([r"\hline", r"\end{tabular}"])
        if axis_key != "short_ensemble":
            lines.append(r"\medskip")
    lines.extend(
        [
            rf"\caption{{Held-out SIMUS mobile stress frontier. Each row compares the best detector-family stack with the strongest public comparator on the same held-out evaluation seed.{(' On every row in this reduced table, the strongest public comparator was ' + _pipeline_caption(common_public) + '.') if common_public else ''} The cleaner structural checkpoint is shown separately to identify the fixed default detector head.}}",
            r"\label{tab:simus_stress_frontier_headline}",
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


if __name__ == "__main__":
    main()
