from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _finite(vals: list[float | None]) -> np.ndarray:
    arr = np.array([v for v in vals if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return arr


def _median(vals: list[float | None]) -> float | None:
    arr = _finite(vals)
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _summary(vals: list[float | None]) -> dict[str, Any]:
    arr = _finite(vals)
    out: dict[str, Any] = {
        "count_total": int(len(vals)),
        "count_finite": int(arr.size),
        "median": None,
        "p25": None,
        "p75": None,
        "min": None,
        "max": None,
    }
    if arr.size:
        out.update(
            {
                "median": float(np.median(arr)),
                "p25": float(np.quantile(arr, 0.25)),
                "p75": float(np.quantile(arr, 0.75)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
    return out


def _safe_get(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Twinkling RawBCF decode sanity reports written by scripts/twinkling_decode_sanity.py.\n"
            "Produces a per-report CSV/JSON and an aggregate-by-sequence summary."
        )
    )
    parser.add_argument(
        "--in-root",
        type=Path,
        default=Path("reports/twinkling_decode_sanity"),
        help="Input directory root containing decode_report.json files (default: %(default)s).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/twinkling_decode_sanity_summary.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/twinkling_decode_sanity_summary.json"),
        help="Output JSON path (default: %(default)s).",
    )
    args = parser.parse_args()

    reports = sorted(Path(args.in_root).rglob("decode_report.json"))
    if not reports:
        raise SystemExit(f"No decode_report.json files found under {args.in_root}")

    rows: list[dict[str, Any]] = []
    for p in reports:
        rep = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(rep, dict):
            continue
        seq_dir = str(rep.get("seq_dir") or "")
        par_path = str(rep.get("par_path") or "")
        dat_path = str(rep.get("dat_path") or "")
        picture_path = rep.get("picture_path")
        frame_idx = rep.get("frame_idx")

        corr_direct = _safe_get(rep, "bmode_picture_corr", "direct")
        corr_flip = _safe_get(rep, "bmode_picture_corr", "flip_lr")
        try:
            corr_direct_f = float(corr_direct) if corr_direct is not None else None
        except Exception:
            corr_direct_f = None
        try:
            corr_flip_f = float(corr_flip) if corr_flip is not None else None
        except Exception:
            corr_flip_f = None
        corr_best = None
        if corr_direct_f is not None or corr_flip_f is not None:
            corr_best = max([v for v in [corr_direct_f, corr_flip_f] if v is not None], default=None)

        coh_beam = _safe_get(rep, "cfm_order_sanity", "mean_temporal_coherence", "beam_major")
        coh_shot = _safe_get(rep, "cfm_order_sanity", "mean_temporal_coherence", "shot_major")
        try:
            coh_beam_f = float(coh_beam) if coh_beam is not None else None
        except Exception:
            coh_beam_f = None
        try:
            coh_shot_f = float(coh_shot) if coh_shot is not None else None
        except Exception:
            coh_shot_f = None
        coh_ratio = None
        if coh_beam_f is not None and coh_shot_f is not None and coh_shot_f > 0:
            coh_ratio = float(coh_beam_f / coh_shot_f)

        rows.append(
            {
                "decode_report": str(p),
                "seq_dir": seq_dir,
                "par_path": par_path,
                "dat_path": dat_path,
                "picture_path": str(picture_path) if picture_path else None,
                "frame_idx": int(frame_idx) if frame_idx is not None else None,
                "num_cfm_shots": _safe_get(rep, "par", "num_cfm_shots"),
                "num_cfm_beams": _safe_get(rep, "par", "num_cfm_beams"),
                "cfm_beam_samples": _safe_get(rep, "par", "cfm_beam_samples"),
                "cfm_order_selected": _safe_get(rep, "cfm_order_sanity", "selected"),
                "coh_beam_major": coh_beam_f,
                "coh_shot_major": coh_shot_f,
                "coh_ratio_beam_over_shot": coh_ratio,
                "bmode_picture_corr_direct": corr_direct_f,
                "bmode_picture_corr_flip_lr": corr_flip_f,
                "bmode_picture_corr_best": corr_best,
            }
        )

    # Aggregate by (seq_dir, par_path) so multiple frames/reports per sequence fold together.
    by_seq: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        by_seq.setdefault((str(r.get("seq_dir") or ""), str(r.get("par_path") or "")), []).append(r)

    agg_rows: list[dict[str, Any]] = []
    for (seq_dir, par_path), rs in sorted(by_seq.items(), key=lambda t: (t[0][0], t[0][1])):
        agg_rows.append(
            {
                "seq_dir": seq_dir,
                "par_path": par_path,
                "report_count": int(len(rs)),
                "num_cfm_shots": rs[0].get("num_cfm_shots"),
                "num_cfm_beams": rs[0].get("num_cfm_beams"),
                "cfm_beam_samples": rs[0].get("cfm_beam_samples"),
                "median_coh_ratio_beam_over_shot": _median([r.get("coh_ratio_beam_over_shot") for r in rs]),
                "median_bmode_picture_corr_best": _median([r.get("bmode_picture_corr_best") for r in rs]),
                "corr_best_summary": _summary([r.get("bmode_picture_corr_best") for r in rs]),
                "coh_ratio_summary": _summary([r.get("coh_ratio_beam_over_shot") for r in rs]),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "in_root": str(args.in_root),
        "report_count": int(len(rows)),
        "rows": rows,
        "aggregate_by_sequence": agg_rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

