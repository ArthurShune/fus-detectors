#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.simus_v2_anchor_envelopes import DEFAULT_ACCEPTANCE_METRICS, flatten_summary_row
from scripts.physical_doppler_sanity_link import BandEdges, TileSpec, summarize_icube
from sim.simus.bundle import estimate_simus_policy_features, load_canonical_run

ANCHOR_PRESETS: dict[str, list[str]] = {
    "pooled_iq": ["shin", "gammex_along", "gammex_across", "ulm_7883227"],
    "intraop_brainlike": ["shin", "ulm_7883227"],
    "phantom_nuisance": ["gammex_along", "gammex_across"],
}

PROFILE_GATES: dict[str, dict[str, Any]] = {
    "ClinIntraOp-Pf-v2": {
        "hard_sections": (
            {
                "name": "brain_background",
                "anchor_preset": "intraop_brainlike",
                "metrics": ("bg_fpeak_q50", "bg_coh1_q50"),
            },
            {
                "name": "pooled_background_subspace",
                "anchor_preset": "pooled_iq",
                "metrics": ("svd_bg_cum_r1", "svd_bg_cum_r2"),
            },
        ),
        "hard_rules": (
            {
                "name": "scene.expected_fd_sampled_q50_hz",
                "path": ("scene", "expected_fd_sampled_q50_hz"),
                "lo": 60.0,
                "hi": 180.0,
            },
            {
                "name": "scene.h1_alias_qc_fraction",
                "path": ("scene", "h1_alias_qc_fraction"),
                "lo": 0.0,
                "hi": 0.20,
            },
            {
                "name": "scene.h0_nuisance_fraction",
                "path": ("scene", "h0_nuisance_fraction"),
                "lo": 0.01,
                "hi": 0.08,
            },
        ),
        "soft_sections": (
            {
                "name": "phantom_nuisance",
                "anchor_preset": "phantom_nuisance",
                "metrics": ("bg_malias_q50",),
            },
            {
                "name": "brain_flow_motion",
                "anchor_preset": "intraop_brainlike",
                "metrics": (
                    "flow_fpeak_q50",
                    "flow_coh1_q50",
                    "flow_malias_q50",
                    "svd_flow_cum_r1",
                    "svd_flow_cum_r2",
                    "reg_shift_rms",
                    "reg_shift_p90",
                    "reg_psr_median",
                ),
            },
        ),
    },
    "ClinMobile-Pf-v2": {
        "hard_sections": (
            {
                "name": "mobile_background",
                "anchor_preset": "pooled_iq",
                "metrics": ("bg_fpeak_q50",),
            },
            {
                "name": "pooled_background_subspace",
                "anchor_preset": "pooled_iq",
                "metrics": ("svd_bg_cum_r1", "svd_bg_cum_r2"),
            },
        ),
        "hard_rules": (
            {
                "name": "scene.expected_fd_sampled_q50_hz",
                "path": ("scene", "expected_fd_sampled_q50_hz"),
                "lo": 60.0,
                "hi": 180.0,
            },
            {
                "name": "scene.h1_alias_qc_fraction",
                "path": ("scene", "h1_alias_qc_fraction"),
                "lo": 0.0,
                "hi": 0.25,
            },
            {
                "name": "scene.h0_nuisance_fraction",
                "path": ("scene", "h0_nuisance_fraction"),
                "lo": 0.03,
                "hi": 0.15,
            },
        ),
        "soft_sections": (
            {
                "name": "mobile_background_soft",
                "anchor_preset": "pooled_iq",
                "metrics": ("bg_coh1_q50",),
            },
            {
                "name": "phantom_nuisance",
                "anchor_preset": "phantom_nuisance",
                "metrics": ("bg_malias_q50",),
            },
            {
                "name": "mobile_flow_motion",
                "anchor_preset": "pooled_iq",
                "metrics": (
                    "flow_fpeak_q50",
                    "flow_coh1_q50",
                    "flow_malias_q50",
                    "svd_flow_cum_r1",
                    "svd_flow_cum_r2",
                    "reg_shift_rms",
                    "reg_shift_p90",
                    "reg_psr_median",
                ),
            },
        ),
    },
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _discover_runs(root: Path) -> list[Path]:
    root = Path(root)
    if (root / "dataset" / "meta.json").is_file():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "dataset" / "meta.json").is_file())


def _parse_kinds(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for part in str(value).split(","):
            text = part.strip()
            if text:
                out.append(text)
    return out


def _combine_anchor_envelope(payload: dict[str, Any], anchor_kinds: list[str], metrics: list[str], *, lower_q: float, upper_q: float) -> dict[str, Any]:
    rows = [row for row in (payload.get("rows") or []) if str(row.get("anchor_kind")) in set(anchor_kinds)]
    if not rows:
        raise ValueError(f"no anchor rows matched {anchor_kinds!r}")
    combined: dict[str, Any] = {}
    for metric in metrics:
        vals = np.asarray(
            [float(row[metric]) for row in rows if row.get(metric) is not None],
            dtype=np.float64,
        )
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            combined[metric] = {"n": 0, "lo": None, "hi": None, "q50": None}
            continue
        combined[metric] = {
            "n": int(vals.size),
            "lo": float(np.quantile(vals, float(lower_q))),
            "hi": float(np.quantile(vals, float(upper_q))),
            "q50": float(np.quantile(vals, 0.50)),
        }
    return combined


def summarize_run(
    *,
    run_dir: Path,
    bands: BandEdges,
    tile: TileSpec,
) -> dict[str, Any]:
    icube, masks, meta = load_canonical_run(Path(run_dir))
    prf_hz = float(meta.get("acquisition", {}).get("prf_hz", meta.get("config", {}).get("prf_hz", 0.0)))
    report = summarize_icube(
        name=Path(run_dir).name,
        Icube=icube,
        prf_hz=prf_hz,
        tile=tile,
        bands=bands,
        mask_flow=masks.get("mask_flow"),
        mask_bg=masks.get("mask_bg"),
        derive_masks=False,
        derive_vessel_q=0.99,
        derive_bg_q=0.20,
    )
    motion = {
        "reg_shift_rms": meta.get("bundle_policy_features", {}).get("reg_shift_rms"),
        "reg_shift_p90": meta.get("bundle_policy_features", {}).get("reg_shift_p90"),
        "reg_psr_median": meta.get("bundle_policy_features", {}).get("reg_psr_median"),
    }
    if any(motion.get(key) is None for key in ("reg_shift_rms", "reg_shift_p90", "reg_psr_median")):
        motion = estimate_simus_policy_features(icube, reg_subpixel=4, reg_reference="median")
    row = flatten_summary_row(
        anchor_kind="simus_candidate",
        case_key=str(Path(run_dir).name),
        report=report,
        motion_features=motion,
    )
    return {
        "run_dir": str(Path(run_dir)),
        "case_key": str(Path(run_dir).name),
        "row": row,
        "scene": dict(meta.get("scene") or {}),
    }


def evaluate_run(
    *,
    run_dir: Path,
    bands: BandEdges,
    tile: TileSpec,
    acceptance_env: dict[str, Any],
    metrics: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_summary = summarize_run(run_dir=run_dir, bands=bands, tile=tile)
    row = dict(run_summary["row"])
    metric_rows: list[dict[str, Any]] = []
    required = 0
    passed = 0
    for metric in metrics:
        env = acceptance_env.get(metric) or {}
        value = row.get(metric)
        lo = env.get("lo")
        hi = env.get("hi")
        if value is None or lo is None or hi is None:
            status = "skipped"
            is_pass = None
        else:
            required += 1
            is_pass = bool(float(lo) <= float(value) <= float(hi))
            passed += int(is_pass)
            status = "pass" if is_pass else "fail"
        metric_rows.append(
            {
                "run_dir": str(Path(run_dir)),
                "case_key": str(Path(run_dir).name),
                "metric": metric,
                "value": value,
                "lo": lo,
                "hi": hi,
                "q50": env.get("q50"),
                "status": status,
                "delta_to_q50": None if value is None or env.get("q50") is None else float(value) - float(env["q50"]),
            }
        )
    summary = {
        "run_dir": str(run_summary["run_dir"]),
        "case_key": str(run_summary["case_key"]),
        "required_metrics": int(required),
        "passed_metrics": int(passed),
        "failed_metrics": int(max(required - passed, 0)),
        "pass_fraction": float(passed / required) if required else None,
        "overall_pass": bool(required > 0 and passed == required),
        "row": row,
        "scene": dict(run_summary["scene"]),
    }
    return summary, metric_rows


def _lookup_nested(obj: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _evaluate_profile_gate(
    *,
    run_summary: dict[str, Any],
    anchor_payload: dict[str, Any],
    gate_name: str,
    lower_q: float,
    upper_q: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gate = PROFILE_GATES[str(gate_name)]
    row = dict(run_summary["row"])
    scene = dict(run_summary.get("scene") or {})
    metric_rows: list[dict[str, Any]] = []
    hard_required = 0
    hard_passed = 0
    soft_required = 0
    soft_passed = 0

    def add_metric_row(section_name: str, metric: str, value: Any, lo: Any, hi: Any, source: str, required_kind: str) -> None:
        nonlocal hard_required, hard_passed, soft_required, soft_passed
        if value is None or lo is None or hi is None:
            status = "skipped"
            is_pass = None
        else:
            is_pass = bool(float(lo) <= float(value) <= float(hi))
            status = "pass" if is_pass else "fail"
            if required_kind == "hard":
                hard_required += 1
                hard_passed += int(is_pass)
            else:
                soft_required += 1
                soft_passed += int(is_pass)
        metric_rows.append(
            {
                "run_dir": str(run_summary["run_dir"]),
                "case_key": str(run_summary["case_key"]),
                "section": section_name,
                "metric": metric,
                "value": value,
                "lo": lo,
                "hi": hi,
                "status": status,
                "source": source,
                "required_kind": required_kind,
            }
        )

    for section_kind, required_kind in (("hard_sections", "hard"), ("soft_sections", "soft")):
        for section in gate.get(section_kind, ()):
            acceptance_env = _combine_anchor_envelope(
                anchor_payload,
                list(ANCHOR_PRESETS[str(section["anchor_preset"])]),
                list(section["metrics"]),
                lower_q=float(lower_q),
                upper_q=float(upper_q),
            )
            for metric in section["metrics"]:
                env = acceptance_env.get(str(metric), {})
                add_metric_row(
                    str(section["name"]),
                    str(metric),
                    row.get(str(metric)),
                    env.get("lo"),
                    env.get("hi"),
                    f"anchors:{section['anchor_preset']}",
                    required_kind,
                )

    for rule in gate.get("hard_rules", ()):
        value = _lookup_nested({"scene": scene, "row": row}, tuple(rule["path"]))
        add_metric_row(
            "design_rules",
            str(rule["name"]),
            value,
            rule.get("lo"),
            rule.get("hi"),
            "design",
            "hard",
        )

    summary = {
        "run_dir": str(run_summary["run_dir"]),
        "case_key": str(run_summary["case_key"]),
        "required_metrics": int(hard_required),
        "passed_metrics": int(hard_passed),
        "failed_metrics": int(max(hard_required - hard_passed, 0)),
        "pass_fraction": float(hard_passed / hard_required) if hard_required else None,
        "overall_pass": bool(hard_required > 0 and hard_passed == hard_required),
        "soft_required_metrics": int(soft_required),
        "soft_passed_metrics": int(soft_passed),
        "soft_failed_metrics": int(max(soft_required - soft_passed, 0)),
        "soft_pass_fraction": float(soft_passed / soft_required) if soft_required else None,
        "row": row,
        "scene": scene,
        "profile_gate": str(gate_name),
    }
    return summary, metric_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate SIMUS candidate runs against frozen SIMUS v2 anchor envelopes.")
    ap.add_argument("--run", type=Path, action="append", default=None)
    ap.add_argument("--sim-root", type=Path, action="append", default=None)
    ap.add_argument("--anchor-json", type=Path, default=Path("reports/simus_v2/anchors/simus_v2_anchor_envelopes.json"))
    ap.add_argument("--profile-gate", type=str, choices=sorted(PROFILE_GATES.keys()), default=None)
    ap.add_argument("--anchor-preset", type=str, choices=sorted(ANCHOR_PRESETS.keys()), default=None)
    ap.add_argument("--anchor-kind", type=str, action="append", default=None)
    ap.add_argument("--metrics", type=str, default=",".join(DEFAULT_ACCEPTANCE_METRICS))
    ap.add_argument("--lower-q", type=float, default=0.10)
    ap.add_argument("--upper-q", type=float, default=0.90)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/simus_v2/acceptance"))
    ap.add_argument("--out-csv", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--pf", type=float, nargs=2, default=(30.0, 250.0))
    ap.add_argument("--pg", type=float, nargs=2, default=(250.0, 400.0))
    ap.add_argument("--pa-lo", type=float, default=400.0)
    ap.add_argument("--tile-hw", type=int, nargs=2, default=(8, 8))
    ap.add_argument("--tile-stride", type=int, default=3)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv) if args.out_csv is not None else out_dir / "simus_v2_acceptance_metrics.csv"
    out_json = Path(args.out_json) if args.out_json is not None else out_dir / "simus_v2_acceptance_summary.json"

    payload = json.loads(Path(args.anchor_json).read_text(encoding="utf-8"))
    anchor_kinds = _parse_kinds(args.anchor_kind)
    if args.profile_gate and (args.anchor_preset or anchor_kinds):
        raise SystemExit("--profile-gate is mutually exclusive with --anchor-preset/--anchor-kind")
    if args.anchor_preset:
        if anchor_kinds:
            raise SystemExit("--anchor-preset and --anchor-kind are mutually exclusive")
        anchor_kinds = list(ANCHOR_PRESETS[str(args.anchor_preset)])
    if not anchor_kinds and not args.profile_gate:
        anchor_kinds = list(ANCHOR_PRESETS["pooled_iq"])
    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    acceptance_env = None
    if not args.profile_gate:
        acceptance_env = _combine_anchor_envelope(
            payload,
            anchor_kinds,
            metrics,
            lower_q=float(args.lower_q),
            upper_q=float(args.upper_q),
        )

    bands = BandEdges(
        pf_lo_hz=float(args.pf[0]),
        pf_hi_hz=float(args.pf[1]),
        pg_lo_hz=float(args.pg[0]),
        pg_hi_hz=float(args.pg[1]),
        pa_lo_hz=float(args.pa_lo),
    )
    tile = TileSpec(h=int(args.tile_hw[0]), w=int(args.tile_hw[1]), stride=int(args.tile_stride))

    run_dirs: list[Path] = []
    for root in args.sim_root or []:
        run_dirs.extend(_discover_runs(Path(root)))
    for run in args.run or []:
        run_dirs.append(Path(run))
    if not run_dirs:
        raise SystemExit("no SIMUS runs provided")

    summaries: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run_dir in run_dirs:
        key = str(Path(run_dir).resolve())
        if key in seen:
            continue
        seen.add(key)
        if args.profile_gate:
            run_summary = summarize_run(run_dir=Path(run_dir), bands=bands, tile=tile)
            summary, rows = _evaluate_profile_gate(
                run_summary=run_summary,
                anchor_payload=payload,
                gate_name=str(args.profile_gate),
                lower_q=float(args.lower_q),
                upper_q=float(args.upper_q),
            )
        else:
            summary, rows = evaluate_run(
                run_dir=Path(run_dir),
                bands=bands,
                tile=tile,
                acceptance_env=dict(acceptance_env or {}),
                metrics=metrics,
            )
        summaries.append(summary)
        metric_rows.extend(rows)

    out_payload = {
        "schema_version": "simus_v2_acceptance.v1",
        "anchor_json": str(Path(args.anchor_json)),
        "profile_gate": None if args.profile_gate is None else str(args.profile_gate),
        "anchor_preset": None if args.anchor_preset is None else str(args.anchor_preset),
        "anchor_kinds": anchor_kinds,
        "metrics": metrics,
        "lower_q": float(args.lower_q),
        "upper_q": float(args.upper_q),
        "acceptance_envelope": acceptance_env,
        "runs": summaries,
    }
    _write_csv(out_csv, metric_rows)
    _write_json(out_json, out_payload)
    print(f"[simus-v2-acceptance] wrote {out_csv}")
    print(f"[simus-v2-acceptance] wrote {out_json}")


if __name__ == "__main__":
    main()
