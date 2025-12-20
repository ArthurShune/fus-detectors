from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _as_float(val: Any) -> float | None:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _as_int(val: Any) -> int | None:
    try:
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _load_meta(meta_path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(meta_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _iter_bundle_metas(root: Path) -> list[Path]:
    metas: list[Path] = []
    for meta_path in root.rglob("meta.json"):
        if not meta_path.is_file():
            continue
        metas.append(meta_path)
    metas.sort()
    return metas


def _extract_row(meta_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = meta_path.parent
    tele = meta.get("stap_fallback_telemetry", {}) or {}
    if not isinstance(tele, dict):
        tele = {}

    ka_v2 = meta.get("ka_contract_v2", {}) or {}
    if not isinstance(ka_v2, dict):
        ka_v2 = {}
    ka_metrics = ka_v2.get("metrics", {}) or {}
    if not isinstance(ka_metrics, dict):
        ka_metrics = {}

    sim_geom = meta.get("sim_geom", {}) or {}
    if not isinstance(sim_geom, dict):
        sim_geom = {}

    row: dict[str, Any] = {
        "bundle_dir": str(bundle_dir),
        "bundle_name": bundle_dir.name,
        "orig_run": meta.get("orig_run"),
        "seed": _as_int(meta.get("seed")),
        "prf_hz": _as_float(meta.get("prf_hz")),
        "total_frames": _as_int(meta.get("total_frames")),
        "Lt": _as_int(meta.get("Lt")),
        "tile_h": _as_int(sim_geom.get("tile_h")),
        "tile_w": _as_int(sim_geom.get("tile_w")),
        "tile_stride": _as_int(sim_geom.get("tile_stride")),
        "score_mode": (meta.get("score_stats", {}) or {}).get("mode"),
        "ka_contract_v2_state": ka_v2.get("state"),
        "ka_contract_v2_reason": ka_v2.get("reason"),
        "ka_contract_v2_risk_mode": ka_metrics.get("risk_mode"),
        "ka_contract_v2_uplift_vetoed_by_guard": ka_metrics.get("uplift_vetoed_by_guard"),
        "ka_contract_v2_uplift_vetoed_by_pf_peak": ka_metrics.get("uplift_vetoed_by_pf_peak"),
        "ka_contract_v2_uplift_veto_pf_peak_reason": ka_metrics.get("uplift_veto_pf_peak_reason"),
        "ka_contract_v2_pf_peak_nonbg": _as_float(ka_metrics.get("pf_peak_nonbg")),
        "ka_contract_v2_pf_peak_flow": _as_float(ka_metrics.get("pf_peak_flow")),
        "ka_contract_v2_n_flow_proxy": _as_int(ka_metrics.get("n_flow_proxy")),
        "ka_contract_v2_n_nonbg_proxy": _as_int(ka_metrics.get("n_nonbg_proxy")),
        "ka_contract_v2_pf_peak_min_c2": _as_float(ka_metrics.get("pf_peak_min_c2")),
        "ka_contract_v2_n_flow_min_c2": _as_int(ka_metrics.get("n_flow_min_c2")),
        "ka_contract_v2_iqr_alias_bg": _as_float(ka_metrics.get("iqr_alias_bg")),
        "ka_contract_v2_guard_q90": _as_float(ka_metrics.get("guard_q90")),
        "ka_contract_v2_guard_iqr_bg": _as_float(ka_metrics.get("guard_iqr_bg")),
        "ka_contract_v2_delta_tail_guard": _as_float(ka_metrics.get("delta_tail_guard")),
        "ka_contract_v2_delta_bg_flow_median": _as_float(ka_metrics.get("delta_bg_flow_median")),
        "ka_contract_v2_delta_tail": _as_float(ka_metrics.get("delta_tail")),
        "ka_contract_v2_uplift_eligible_raw": ka_metrics.get("uplift_eligible_raw"),
        "ka_contract_v2_p_shrink": _as_float(ka_metrics.get("p_shrink")),
        "ka_contract_v2_uplift_eligible": ka_metrics.get("uplift_eligible"),
        "score_ka_v2_state": tele.get("score_ka_v2_state"),
        "score_ka_v2_contract_reason": tele.get("score_ka_v2_contract_reason"),
        "score_ka_v2_disabled_reason": tele.get("score_ka_v2_disabled_reason"),
        "score_ka_v2_risk_mode": tele.get("score_ka_v2_risk_mode"),
        "score_ka_v2_applied": tele.get("score_ka_v2_applied"),
        "score_ka_v2_scaled_pixel_fraction": _as_float(tele.get("score_ka_v2_scaled_pixel_fraction")),
        "score_ka_v2_scale_p50": _as_float(tele.get("score_ka_v2_scale_p50")),
        "score_ka_v2_scale_p90": _as_float(tele.get("score_ka_v2_scale_p90")),
        "score_ka_v2_scale_max": _as_float(tele.get("score_ka_v2_scale_max")),
        "score_pool_refresh_error": tele.get("score_pool_refresh_error"),
    }
    return row


def _print_summary(rows: list[dict[str, Any]]) -> None:
    contract_counts = Counter((row.get("ka_contract_v2_state"), row.get("ka_contract_v2_reason")) for row in rows)
    score_counts = Counter((row.get("score_ka_v2_state"), row.get("score_ka_v2_disabled_reason")) for row in rows)
    applied = sum(bool(row.get("score_ka_v2_applied")) for row in rows)

    print(f"[ka-v2] bundles={len(rows)} score_ka_v2_applied={applied}")
    print("[ka-v2] Contract v2 (state, reason) top:")
    for (state, reason), count in contract_counts.most_common(12):
        print(f"  {count:4d}  {state}/{reason}")
    print("[ka-v2] Score KA v2 (state, disabled_reason) top:")
    for (state, reason), count in score_counts.most_common(12):
        print(f"  {count:4d}  {state}/{reason}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate KA Contract v2 / score-space KA v2 telemetry from acceptance bundles."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to scan recursively for bundle meta.json files.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    meta_paths = _iter_bundle_metas(root)
    rows: list[dict[str, Any]] = []
    for meta_path in meta_paths:
        meta = _load_meta(meta_path)
        if meta is None:
            continue
        if "stap_fallback_telemetry" not in meta and "ka_contract_v2" not in meta:
            continue
        rows.append(_extract_row(meta_path, meta))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    if rows:
        fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    _print_summary(rows)
    print(f"[ka-v2] wrote {out_path}")


if __name__ == "__main__":
    main()
