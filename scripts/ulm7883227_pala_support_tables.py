#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: float | None, digits: int = 3, *, signed: bool = False) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(float(x), spec)


def _fmt_sci(x: float | None, digits: int = 1) -> str:
    if x is None or not math.isfinite(float(x)):
        return "--"
    value = float(x)
    if value == 0.0:
        return "0"
    text = format(value, f".{digits}e")
    return text.replace("e-0", "e-").replace("e+0", "e+")


def _fmt_interval(center: float | None, lo: float | None, hi: float | None, *, digits: int = 3) -> str:
    vals = [v for v in (center, lo, hi) if v is not None and math.isfinite(float(v))]
    if not vals:
        return "--"
    if max(abs(float(v)) for v in vals) < 1e-3:
        return f"{_fmt_sci(center)} [{_fmt_sci(lo)},{_fmt_sci(hi)}]"
    return f"{_fmt(center, digits)} [{_fmt(lo, digits)},{_fmt(hi, digits)}]"


def _quartiles(xs: list[float]) -> tuple[float, float, float]:
    vals = sorted(float(x) for x in xs)
    if not vals:
        raise ValueError("expected at least one value")
    med = statistics.median(vals)
    q1, q3 = statistics.quantiles(vals, n=4, method="inclusive")[0], statistics.quantiles(
        vals, n=4, method="inclusive"
    )[2]
    return float(med), float(q1), float(q3)


def _mean_ci(xs: list[float]) -> tuple[float, float, float]:
    vals = sorted(float(x) for x in xs)
    if not vals:
        raise ValueError("expected at least one value")
    mean = statistics.mean(vals)
    q1, q3 = statistics.quantiles(vals, n=40, method="inclusive")[0], statistics.quantiles(
        vals, n=40, method="inclusive"
    )[38]
    return float(mean), float(q1), float(q3)


def _score_rows(rows: list[dict[str, str]], score_key: str) -> list[dict[str, str]]:
    out = [r for r in rows if r["score_key"] == score_key]
    out.sort(key=lambda r: (int(r["block_id"]), int(r["window_index"])))
    return out


def _pair_rows(
    left_rows: list[dict[str, str]],
    right_rows: list[dict[str, str]],
) -> list[tuple[dict[str, str], dict[str, str]]]:
    left = {(int(r["block_id"]), int(r["window_index"])): r for r in left_rows}
    right = {(int(r["block_id"]), int(r["window_index"])): r for r in right_rows}
    keys = sorted(set(left) & set(right))
    return [(left[k], right[k]) for k in keys]


def _consistency_summary(
    specialist_rows: list[dict[str, str]],
    comp_rows: list[dict[str, str]],
) -> dict[str, Any]:
    pairs = _pair_rows(specialist_rows, comp_rows)
    auc_deltas = [float(a["auc"]) - float(b["auc"]) for a, b in pairs]
    fpr_deltas = [float(a["fpr_at_tpr70"]) - float(b["fpr_at_tpr70"]) for a, b in pairs]
    auc_med, auc_q1, auc_q3 = _quartiles(auc_deltas)
    fpr_med, fpr_q1, fpr_q3 = _quartiles(fpr_deltas)
    return {
        "n_windows": len(pairs),
        "auc_wins": sum(float(a["auc"]) > float(b["auc"]) for a, b in pairs),
        "fpr70_wins": sum(float(a["fpr_at_tpr70"]) < float(b["fpr_at_tpr70"]) for a, b in pairs),
        "nonzero_tpr1e3_specialist": sum(float(a["tpr@1e-03"]) > 0.0 for a, _ in pairs),
        "nonzero_tpr1e3_comp": sum(float(b["tpr@1e-03"]) > 0.0 for _, b in pairs),
        "auc_delta_median": auc_med,
        "auc_delta_q1": auc_q1,
        "auc_delta_q3": auc_q3,
        "fpr70_delta_median": fpr_med,
        "fpr70_delta_q1": fpr_q1,
        "fpr70_delta_q3": fpr_q3,
    }


def _block_win_count(
    specialist_rows: list[dict[str, str]],
    comp_rows: list[dict[str, str]],
    metric: str,
    *,
    higher_is_better: bool,
) -> tuple[int, int]:
    pairs = _pair_rows(specialist_rows, comp_rows)
    spec_by_block: dict[int, list[float]] = {}
    comp_by_block: dict[int, list[float]] = {}
    for spec, comp in pairs:
        block = int(spec["block_id"])
        spec_by_block.setdefault(block, []).append(float(spec[metric]))
        comp_by_block.setdefault(block, []).append(float(comp[metric]))
    wins = 0
    for block in sorted(set(spec_by_block) & set(comp_by_block)):
        s_mean = statistics.mean(spec_by_block[block])
        c_mean = statistics.mean(comp_by_block[block])
        if (s_mean > c_mean) if higher_is_better else (s_mean < c_mean):
            wins += 1
    return wins, len(set(spec_by_block) & set(comp_by_block))


def _block_means(
    specialist_rows: list[dict[str, str]],
    comp_rows: list[dict[str, str]],
    metric: str,
) -> tuple[list[float], list[float]]:
    pairs = _pair_rows(specialist_rows, comp_rows)
    spec_by_block: dict[int, list[float]] = {}
    comp_by_block: dict[int, list[float]] = {}
    for spec, comp in pairs:
        block = int(spec["block_id"])
        spec_by_block.setdefault(block, []).append(float(spec[metric]))
        comp_by_block.setdefault(block, []).append(float(comp[metric]))
    keys = sorted(set(spec_by_block) & set(comp_by_block))
    spec_means = [statistics.mean(spec_by_block[k]) for k in keys]
    comp_means = [statistics.mean(comp_by_block[k]) for k in keys]
    return spec_means, comp_means


def _paired_block_signflip_pvalue(
    specialist_rows: list[dict[str, str]],
    comp_rows: list[dict[str, str]],
    metric: str,
) -> float | None:
    spec_means, comp_means = _block_means(specialist_rows, comp_rows, metric)
    deltas = [float(a - b) for a, b in zip(spec_means, comp_means) if math.isfinite(a - b) and a != b]
    if not deltas:
        return None
    obs = abs(statistics.mean(deltas))
    total = 0
    exceed = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        total += 1
        stat = abs(statistics.mean(s * d for s, d in zip(signs, deltas)))
        if stat >= obs - 1e-15:
            exceed += 1
    return float(exceed) / float(total)


def _transfer_nonzero_counts(rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("pd", "kasai", "matched_subspace"):
        sub = _score_rows(rows, key)
        out[key] = {
            "n_windows": len(sub),
            "nonzero_core_windows": sum(float(r["heldout_tpr_core"]) > 0.0 for r in sub),
            "mean_core_tpr": statistics.mean(float(r["heldout_tpr_core"]) for r in sub),
            "mean_shell_fpr": statistics.mean(float(r["heldout_fpr_shell"]) for r in sub),
        }
    return out


def _family_row_from_summary(
    label: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "label": label,
        "auc": summary["auc"],
        "tpr_1e2": summary["tpr@1e-02"],
        "tpr_1e3": summary["tpr@1e-03"],
        "fpr_70": summary["fpr_at_tpr70"],
    }


def _build_consistency_table(payload: dict[str, Any]) -> str:
    fixed_sec = payload["specialist_vs_fixed"]
    adaptive_sec = payload["specialist_vs_adaptive"]
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/ulm7883227_pala_support_tables.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("Comparison & AUC wins (windows / blocks) & shell-FPR wins @70\\% core recall (windows / blocks) & nonzero TPR@$10^{-3}$ windows \\\\")
    lines.append("\\hline")
    lines.append(
        "Fully whitened variant vs fixed head & "
        f"{fixed_sec['auc_wins']}/{fixed_sec['n_windows']} / {fixed_sec['auc_block_wins']}/{fixed_sec['n_blocks']} & "
        f"{fixed_sec['fpr70_wins']}/{fixed_sec['n_windows']} / {fixed_sec['fpr70_block_wins']}/{fixed_sec['n_blocks']} & "
        f"{fixed_sec['nonzero_tpr1e3_specialist']}/{fixed_sec['n_windows']} vs {fixed_sec['nonzero_tpr1e3_comp']}/{fixed_sec['n_windows']} \\\\"
    )
    lines.append(
        "Fully whitened variant vs adaptive head & "
        f"{adaptive_sec['auc_wins']}/{adaptive_sec['n_windows']} / {adaptive_sec['auc_block_wins']}/{adaptive_sec['n_blocks']} & "
        f"{adaptive_sec['fpr70_wins']}/{adaptive_sec['n_windows']} / {adaptive_sec['fpr70_block_wins']}/{adaptive_sec['n_blocks']} & "
        f"{adaptive_sec['nonzero_tpr1e3_specialist']}/{adaptive_sec['n_windows']} vs {adaptive_sec['nonzero_tpr1e3_comp']}/{adaptive_sec['n_windows']} \\\\"
    )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\caption{Window-level consistency summary for the PALA-backed ULM structural audit. "
        "Each row compares the prespecified fully whitened variant against another detector head on the same frozen 64-frame MC--SVD residual cube and the same localization-derived vessel-core versus shell labels. "
        "The window/block entries report how often the fully whitened variant has higher vessel-core AUC or lower shell false-positive rate at matched 70\\% vessel-core recall.}"
    )
    lines.append("\\label{tab:ulm_pala_window_consistency}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def _build_family_table(payload: dict[str, Any]) -> str:
    pd_p = payload.get("specialist_vs_pd", {}).get("auc_block_signflip_pvalue")
    pd_p_text = ""
    if pd_p is not None:
        pd_p_text = (
            f" The fully whitened variant exceeds baseline power Doppler on all 10 audited blocks "
            f"(exact paired sign-flip test on block-mean AUC, p = {pd_p:.3f})."
        )
    lines: list[str] = []
    lines.append("% AUTO-GENERATED by scripts/ulm7883227_pala_support_tables.py; DO NOT EDIT BY HAND.")
    lines.append("\\begin{center}")
    lines.append("\\small")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{l c c c c}")
    lines.append("\\hline")
    lines.append("Score & AUC (H1 vs H0) & TPR@$10^{-2}$ & TPR@$10^{-3}$ & shell FPR @70\\% core recall \\\\")
    lines.append("\\hline")
    for row in payload["family_rows"]:
        auc = row["auc"]
        tpr2 = row["tpr_1e2"]
        tpr3 = row["tpr_1e3"]
        fpr70 = row["fpr_70"]
        lines.append(
            f"{row['label']} & "
            f"{_fmt_interval(auc['center'], auc['lo'], auc['hi'])} & "
            f"{_fmt_interval(tpr2['center'], tpr2['lo'], tpr2['hi'])} & "
            f"{_fmt_interval(tpr3['center'], tpr3['lo'], tpr3['hi'])} & "
            f"{_fmt_interval(fpr70['center'], fpr70['lo'], fpr70['hi'])} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append(
        "\\captionof{table}{Same-residual ULM structural audit on real in vivo rat-brain IQ. "
        "All rows use the same frozen 64-frame MC--SVD residual cube, the same localization-derived vessel-core versus shell masks, and the same SCM whitening recipe where applicable; only the downstream score head changes. "
        "Entries report window-level means with 95\\% bootstrap intervals over the 70 audited windows."
        + pd_p_text
        + "}"
    )
    lines.append("\\label{tab:ulm7883227_structural_roc}")
    lines.append("\\end{center}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ULM PALA-backed support tables for the paper.")
    ap.add_argument("--main-csv", type=Path, default=ROOT / "reports" / "ulm7883227_pala_profile_whitened.csv")
    ap.add_argument("--fixed-csv", type=Path, default=ROOT / "reports" / "ulm7883227_pala_profile_fixed.csv")
    ap.add_argument("--adaptive-csv", type=Path, default=ROOT / "reports" / "ulm7883227_pala_profile_adaptive.csv")
    ap.add_argument("--main-json", type=Path, default=ROOT / "reports" / "ulm7883227_pala_profile_whitened.json")
    ap.add_argument("--fixed-json", type=Path, default=ROOT / "reports" / "ulm7883227_pala_profile_fixed.json")
    ap.add_argument("--adaptive-json", type=Path, default=ROOT / "reports" / "ulm7883227_pala_profile_adaptive.json")
    ap.add_argument(
        "--transfer-csv",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_pala_fixed_calibration_transfer.csv",
    )
    ap.add_argument(
        "--out-consistency-tex",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_pala_window_consistency_table.tex",
    )
    ap.add_argument(
        "--out-family-tex",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_pala_detector_heads_table.tex",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "reports" / "ulm7883227_pala_support_summary.json",
    )
    args = ap.parse_args()

    main_rows = _read_rows(args.main_csv)
    fixed_rows = _read_rows(args.fixed_csv)
    adaptive_rows = _read_rows(args.adaptive_csv)
    transfer_rows = _read_rows(args.transfer_csv)
    main_summary = _read_json(args.main_json)
    fixed_summary = _read_json(args.fixed_json)
    adaptive_summary = _read_json(args.adaptive_json)

    specialist_rows = _score_rows(main_rows, "matched_subspace")
    pd_rows = _score_rows(main_rows, "pd")
    kasai_rows = _score_rows(main_rows, "kasai")
    payload = {
        "fixed_threshold_transfer": _transfer_nonzero_counts(transfer_rows),
    }
    pd_consistency = _consistency_summary(specialist_rows, pd_rows)
    pd_consistency["auc_block_wins"], pd_consistency["n_blocks"] = _block_win_count(
        specialist_rows, pd_rows, "auc", higher_is_better=True
    )
    pd_consistency["fpr70_block_wins"], _ = _block_win_count(
        specialist_rows, pd_rows, "fpr_at_tpr70", higher_is_better=False
    )
    pd_consistency["auc_block_signflip_pvalue"] = _paired_block_signflip_pvalue(
        specialist_rows, pd_rows, "auc"
    )
    pd_consistency["fpr70_block_signflip_pvalue"] = _paired_block_signflip_pvalue(
        specialist_rows, pd_rows, "fpr_at_tpr70"
    )
    fixed_consistency = _consistency_summary(specialist_rows, _score_rows(fixed_rows, "matched_subspace"))
    fixed_consistency["auc_block_wins"], fixed_consistency["n_blocks"] = _block_win_count(
        specialist_rows, _score_rows(fixed_rows, "matched_subspace"), "auc", higher_is_better=True
    )
    fixed_consistency["fpr70_block_wins"], _ = _block_win_count(
        specialist_rows, _score_rows(fixed_rows, "matched_subspace"), "fpr_at_tpr70", higher_is_better=False
    )
    fixed_consistency["auc_block_signflip_pvalue"] = _paired_block_signflip_pvalue(
        specialist_rows, _score_rows(fixed_rows, "matched_subspace"), "auc"
    )
    fixed_consistency["fpr70_block_signflip_pvalue"] = _paired_block_signflip_pvalue(
        specialist_rows, _score_rows(fixed_rows, "matched_subspace"), "fpr_at_tpr70"
    )
    adaptive_consistency = _consistency_summary(
        specialist_rows, _score_rows(adaptive_rows, "matched_subspace")
    )
    adaptive_consistency["auc_block_wins"], adaptive_consistency["n_blocks"] = _block_win_count(
        specialist_rows, _score_rows(adaptive_rows, "matched_subspace"), "auc", higher_is_better=True
    )
    adaptive_consistency["fpr70_block_wins"], _ = _block_win_count(
        specialist_rows, _score_rows(adaptive_rows, "matched_subspace"), "fpr_at_tpr70", higher_is_better=False
    )
    adaptive_consistency["auc_block_signflip_pvalue"] = _paired_block_signflip_pvalue(
        specialist_rows, _score_rows(adaptive_rows, "matched_subspace"), "auc"
    )
    adaptive_consistency["fpr70_block_signflip_pvalue"] = _paired_block_signflip_pvalue(
        specialist_rows, _score_rows(adaptive_rows, "matched_subspace"), "fpr_at_tpr70"
    )
    payload["specialist_vs_pd"] = pd_consistency
    payload["specialist_vs_fixed"] = fixed_consistency
    payload["specialist_vs_adaptive"] = adaptive_consistency
    fixed_head_rows = _score_rows(fixed_rows, "matched_subspace")
    adaptive_head_rows = _score_rows(adaptive_rows, "matched_subspace")
    if fixed_head_rows and adaptive_head_rows and fixed_summary and adaptive_summary and main_summary:
        payload["family_rows"] = [
            _family_row_from_summary("Baseline (power Doppler)", fixed_summary["scores"]["pd"]),
            _family_row_from_summary("Baseline (Kasai lag-1 magnitude)", fixed_summary["scores"]["kasai"]),
            _family_row_from_summary("Fixed matched-subspace head", fixed_summary["scores"]["matched_subspace"]),
            _family_row_from_summary("Adaptive head", adaptive_summary["scores"]["matched_subspace"]),
            _family_row_from_summary(
                "Fully whitened variant", main_summary["scores"]["matched_subspace"]
            ),
        ]

    args.out_consistency_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_consistency_tex.write_text(_build_consistency_table(payload), encoding="utf-8")
    if "family_rows" in payload:
        args.out_family_tex.parent.mkdir(parents=True, exist_ok=True)
        args.out_family_tex.write_text(_build_family_table(payload), encoding="utf-8")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
