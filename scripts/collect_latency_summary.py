#!/usr/bin/env python3
"""
Collect steady-state latency + parity summaries from existing benchmark run folders.

Outputs:
  - reports/latency/4080super_latency_summary.csv
  - reports/latency/4080super_latency_summary.json

Policy:
  - "steady-state" = mean/median over windows/frames 2..N (exclude index 0).
  - Parity compares legacy vs optimized maps saved in each bundle.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).replace(microsecond=0).isoformat()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: Sequence[str]) -> str:
    return subprocess.check_output(list(cmd), cwd=_repo_root()).decode("utf-8").strip()


def _git_info() -> Dict[str, Any]:
    try:
        return {
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "commit_short": _run(["git", "rev-parse", "--short", "HEAD"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
        }
    except Exception as exc:
        return {"error": str(exc)}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / float(len(xs))) if xs else float("nan")


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _quantiles_abs(diff: np.ndarray, qs: Sequence[float]) -> Dict[str, float]:
    flat = np.asarray(diff, dtype=np.float64).ravel()
    out: Dict[str, float] = {}
    for q in qs:
        out[f"p{int(round(q*1000)):04d}"] = float(np.quantile(flat, float(q)))
    out["max"] = float(np.max(flat))
    return out


def _bundle_sort_key_brain(name: str) -> Tuple[int, int]:
    m = re.search(r"_win(\d+)_off(\d+)$", name)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def _bundle_sort_key_shin(name: str) -> Tuple[int, int]:
    m = re.search(r"_f(\d+)_(\d+)$", name)
    if not m:
        return (10**9, 10**9)
    return (int(m.group(1)), int(m.group(2)))


def _list_bundle_dirs(parent: Path, *, sort_kind: str) -> List[Path]:
    if not parent.exists():
        return []
    bundles = [p for p in parent.iterdir() if p.is_dir() and (p / "meta.json").exists()]
    if sort_kind == "brain":
        bundles.sort(key=lambda p: _bundle_sort_key_brain(p.name))
    elif sort_kind == "shin":
        bundles.sort(key=lambda p: _bundle_sort_key_shin(p.name))
    else:
        bundles.sort(key=lambda p: p.name)
    return bundles


def _extract_ms_list(bundle_dirs: Sequence[Path], key: str) -> List[float]:
    vals: List[float] = []
    for b in bundle_dirs:
        meta = _load_json(b / "meta.json")
        tele = meta.get("stap_fallback_telemetry") or {}
        v = _safe_float(tele.get(key))
        vals.append(float("nan") if v is None else float(v))
    return vals


def _compute_latency_stats(per_window_ms: List[float]) -> Dict[str, Any]:
    if not per_window_ms:
        return {"cold_ms": None, "steady_mean_ms": None, "steady_median_ms": None, "per_window_ms": []}
    cold = per_window_ms[0]
    steady = per_window_ms[1:] if len(per_window_ms) > 1 else []
    return {
        "cold_ms": float(cold),
        "steady_mean_ms": float(_mean(steady)) if steady else None,
        "steady_median_ms": float(_median(steady)) if steady else None,
        "per_window_ms": [float(x) for x in per_window_ms],
        "steady_windows": list(range(1, len(per_window_ms))),
    }


def _compare_bundle_pairs(
    legacy_bundles: Sequence[Path],
    opt_bundles: Sequence[Path],
    *,
    files: Sequence[str],
    qs: Sequence[float],
) -> Dict[str, Any]:
    if len(legacy_bundles) != len(opt_bundles):
        return {"status": "error", "error": f"bundle count mismatch legacy={len(legacy_bundles)} opt={len(opt_bundles)}"}
    per_file: Dict[str, Any] = {}
    overall_max = 0.0
    for fn in files:
        diffs: List[np.ndarray] = []
        for lb, ob in zip(legacy_bundles, opt_bundles):
            a_path = lb / fn
            b_path = ob / fn
            if not a_path.exists() or not b_path.exists():
                return {"status": "error", "error": f"missing comparison file {fn} in {lb} or {ob}"}
            a = np.load(a_path)
            b = np.load(b_path)
            if a.shape != b.shape:
                return {"status": "error", "error": f"shape mismatch {fn}: legacy={a.shape} opt={b.shape}"}
            diffs.append(np.abs(b.astype(np.float64, copy=False) - a.astype(np.float64, copy=False)))
        all_d = np.concatenate([d.ravel() for d in diffs], axis=0) if diffs else np.zeros((0,), dtype=np.float64)
        qd = _quantiles_abs(all_d, qs) if all_d.size else {f"p{int(round(q*1000)):04d}": float("nan") for q in qs}
        qd.setdefault("max", float("nan"))
        overall_max = max(overall_max, float(qd["max"]) if np.isfinite(qd["max"]) else overall_max)
        per_file[fn] = qd
    return {"status": "ok", "files": per_file, "overall_max_abs": float(overall_max)}


@dataclass(frozen=True)
class RunSpec:
    key: str
    dataset: str
    scenario: str
    decision_rule: str
    run_root: Path
    sort_kind: str
    bundle_rel: Path
    tile_batch: int
    window_length: int
    windows: List[int] | List[str]
    baseline_key: str
    stap_key: str


def _collect_one(spec: RunSpec, *, compare_files: Sequence[str], qs: Sequence[float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "key": spec.key,
        "dataset": spec.dataset,
        "scenario": spec.scenario,
        "decision_rule": spec.decision_rule,
        "run_root": str(spec.run_root),
        "tile_batch": int(spec.tile_batch),
        "window_length": int(spec.window_length),
        "windows": spec.windows,
    }
    if not spec.run_root.exists():
        out["status"] = "skipped"
        out["notes"] = "missing run_root"
        return out

    legacy_parent = spec.run_root / "legacy" / spec.bundle_rel
    opt_parent = spec.run_root / "optimized" / spec.bundle_rel
    legacy_bundles = _list_bundle_dirs(legacy_parent, sort_kind=spec.sort_kind)
    opt_bundles = _list_bundle_dirs(opt_parent, sort_kind=spec.sort_kind)
    if not legacy_bundles or not opt_bundles:
        out["status"] = "skipped"
        out["notes"] = "missing legacy/optimized bundles"
        out["legacy_parent"] = str(legacy_parent)
        out["optimized_parent"] = str(opt_parent)
        return out

    legacy_baseline_ms = _extract_ms_list(legacy_bundles, spec.baseline_key)
    legacy_stap_ms = _extract_ms_list(legacy_bundles, spec.stap_key)
    legacy_total_ms = [float(a) + float(b) for a, b in zip(legacy_baseline_ms, legacy_stap_ms)]

    opt_baseline_ms = _extract_ms_list(opt_bundles, spec.baseline_key)
    opt_stap_ms = _extract_ms_list(opt_bundles, spec.stap_key)
    opt_total_ms = [float(a) + float(b) for a, b in zip(opt_baseline_ms, opt_stap_ms)]

    out["legacy"] = {
        "bundle_parent": str(legacy_parent),
        "bundles": [b.name for b in legacy_bundles],
        "baseline_ms": _compute_latency_stats(legacy_baseline_ms),
        "stap_ms": _compute_latency_stats(legacy_stap_ms),
        "total_ms": _compute_latency_stats(legacy_total_ms),
    }
    out["optimized"] = {
        "bundle_parent": str(opt_parent),
        "bundles": [b.name for b in opt_bundles],
        "baseline_ms": _compute_latency_stats(opt_baseline_ms),
        "stap_ms": _compute_latency_stats(opt_stap_ms),
        "total_ms": _compute_latency_stats(opt_total_ms),
    }

    legacy_stap_mean = out["legacy"]["stap_ms"]["steady_mean_ms"]
    opt_stap_mean = out["optimized"]["stap_ms"]["steady_mean_ms"]
    legacy_total_mean = out["legacy"]["total_ms"]["steady_mean_ms"]
    opt_total_mean = out["optimized"]["total_ms"]["steady_mean_ms"]
    out["speedup"] = {
        "stap_mean": (float(legacy_stap_mean) / float(opt_stap_mean)) if (legacy_stap_mean and opt_stap_mean) else None,
        "total_mean": (float(legacy_total_mean) / float(opt_total_mean)) if (legacy_total_mean and opt_total_mean) else None,
    }

    out["parity"] = _compare_bundle_pairs(legacy_bundles, opt_bundles, files=compare_files, qs=qs)
    out["status"] = "ok" if out["parity"].get("status") == "ok" else "error"
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Collect latency summaries into CSV/JSON.")
    ap.add_argument("--out-csv", type=Path, default=Path("reports/latency/4080super_latency_summary.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/latency/4080super_latency_summary.json"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    compare_files = ["pd_base.npy", "pd_stap.npy", "score_stap.npy", "stap_score_map.npy"]
    qs = [0.50, 0.90, 0.99, 0.999]

    # Capture repo state before writing outputs so "dirty" isn't caused by this script.
    git_info = _git_info()

    runs: List[RunSpec] = [
        RunSpec(
            key="brain_openskull_full",
            dataset="brain_kwave",
            scenario="Brain-OpenSkull",
            decision_rule="full",
            run_root=Path("runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256"),
            sort_kind="brain",
            bundle_rel=Path("."),
            tile_batch=192,
            window_length=64,
            windows=[0, 64, 128, 192, 256],
            baseline_key="baseline_ms",
            stap_key="stap_total_ms",
        ),
        RunSpec(
            key="brain_aliascontract_full",
            dataset="brain_kwave",
            scenario="Brain-AliasContract",
            decision_rule="full",
            run_root=Path("runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256"),
            sort_kind="brain",
            bundle_rel=Path("."),
            tile_batch=192,
            window_length=64,
            windows=[0, 64, 128, 192, 256],
            baseline_key="baseline_ms",
            stap_key="stap_total_ms",
        ),
        RunSpec(
            key="brain_skullor_full",
            dataset="brain_kwave",
            scenario="Brain-SkullOR",
            decision_rule="full",
            run_root=Path("runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256"),
            sort_kind="brain",
            bundle_rel=Path("."),
            tile_batch=192,
            window_length=64,
            windows=[0, 64, 128, 192, 256],
            baseline_key="baseline_ms",
            stap_key="stap_total_ms",
        ),
        RunSpec(
            key="brain_openskull_conditional",
            dataset="brain_kwave",
            scenario="Brain-OpenSkull",
            decision_rule="conditional",
            run_root=Path("runs/latency_4080super_brain_openskull_w64_tb192_off0_64_128_192_256_cond"),
            sort_kind="brain",
            bundle_rel=Path("."),
            tile_batch=192,
            window_length=64,
            windows=[0, 64, 128, 192, 256],
            baseline_key="baseline_ms",
            stap_key="stap_total_ms",
        ),
        RunSpec(
            key="brain_aliascontract_conditional",
            dataset="brain_kwave",
            scenario="Brain-AliasContract",
            decision_rule="conditional",
            run_root=Path("runs/latency_4080super_brain_aliascontract_w64_tb192_off0_64_128_192_256_cond"),
            sort_kind="brain",
            bundle_rel=Path("."),
            tile_batch=192,
            window_length=64,
            windows=[0, 64, 128, 192, 256],
            baseline_key="baseline_ms",
            stap_key="stap_total_ms",
        ),
        RunSpec(
            key="brain_skullor_conditional",
            dataset="brain_kwave",
            scenario="Brain-SkullOR",
            decision_rule="conditional",
            run_root=Path("runs/latency_4080super_brain_skullor_w64_tb192_off0_64_128_192_256_cond"),
            sort_kind="brain",
            bundle_rel=Path("."),
            tile_batch=192,
            window_length=64,
            windows=[0, 64, 128, 192, 256],
            baseline_key="baseline_ms",
            stap_key="stap_total_ms",
        ),
        RunSpec(
            key="shin_fig3_full",
            dataset="shin_ratbrain",
            scenario="ShinRatBrain_Fig3",
            decision_rule="full",
            run_root=Path("runs/latency_4080super_shin_fig3_w128_tb192"),
            sort_kind="shin",
            bundle_rel=Path("shin/stap_full"),
            tile_batch=192,
            window_length=128,
            windows=["f0_128", "f64_192", "f122_250"],
            baseline_key="baseline_ms",
            stap_key="stap_ms",
        ),
        RunSpec(
            key="shin_fig3_conditional",
            dataset="shin_ratbrain",
            scenario="ShinRatBrain_Fig3",
            decision_rule="conditional",
            run_root=Path("runs/latency_4080super_shin_fig3_w128_tb192"),
            sort_kind="shin",
            bundle_rel=Path("shin/stap_conditional"),
            tile_batch=192,
            window_length=128,
            windows=["f0_128", "f64_192", "f122_250"],
            baseline_key="baseline_ms",
            stap_key="stap_ms",
        ),
        RunSpec(
            key="gammex_along_linear17_full",
            dataset="twinkling_gammex",
            scenario="Gammex along (linear17 @ PRF 2500; per cine frame)",
            decision_rule="full",
            run_root=Path("runs/latency_4080super_gammex_linear17_f0_6_tb512"),
            sort_kind="alpha",
            bundle_rel=Path("gammex/along_linear17__RawBCFCine/stap_full/along_linear17__RawBCFCine"),
            tile_batch=512,
            window_length=1,
            windows=[0, 1, 2, 3, 4, 5],
            baseline_key="baseline_ms",
            stap_key="stap_ms",
        ),
        RunSpec(
            key="gammex_across_linear17_full",
            dataset="twinkling_gammex",
            scenario="Gammex across (linear17 @ PRF 2500; per cine frame)",
            decision_rule="full",
            run_root=Path("runs/latency_4080super_gammex_linear17_f0_6_tb512"),
            sort_kind="alpha",
            bundle_rel=Path(
                "gammex/across_linear17__RawBCFCine_08062017_145434_17/stap_full/"
                "across_linear17__RawBCFCine_08062017_145434_17"
            ),
            tile_batch=512,
            window_length=1,
            windows=[0, 1, 2, 3, 4, 5],
            baseline_key="baseline_ms",
            stap_key="stap_ms",
        ),
    ]

    records: List[Dict[str, Any]] = []
    for spec in runs:
        records.append(_collect_one(spec, compare_files=compare_files, qs=qs))

    out_json = {
        "generated_at_utc": _utc_now_iso(),
        "git": git_info,
        "latency_policy": {
            "steady_state": "mean/median over windows/frames 2..N (exclude index 0)",
            "parity_files": compare_files,
            "parity_quantiles": qs,
        },
        "records": records,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_json, indent=2) + "\n")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "key",
        "dataset",
        "scenario",
        "decision_rule",
        "window_length",
        "windows",
        "tile_batch",
        "legacy_stap_cold_ms",
        "legacy_stap_steady_mean_ms",
        "legacy_stap_steady_median_ms",
        "optimized_stap_cold_ms",
        "optimized_stap_steady_mean_ms",
        "optimized_stap_steady_median_ms",
        "stap_speedup_mean",
        "legacy_total_steady_mean_ms",
        "optimized_total_steady_mean_ms",
        "total_speedup_mean",
        "parity_overall_max_abs",
        "parity_score_stap_max_abs",
        "status",
        "notes",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            row: Dict[str, Any] = {k: "" for k in fieldnames}
            row["key"] = rec.get("key", "")
            row["dataset"] = rec.get("dataset", "")
            row["scenario"] = rec.get("scenario", "")
            row["decision_rule"] = rec.get("decision_rule", "")
            row["window_length"] = rec.get("window_length", "")
            row["windows"] = json.dumps(rec.get("windows", []))
            row["tile_batch"] = rec.get("tile_batch", "")
            row["status"] = rec.get("status", "")
            row["notes"] = rec.get("notes", "")

            if rec.get("status") == "ok":
                leg = rec.get("legacy", {})
                opt = rec.get("optimized", {})
                spd = rec.get("speedup", {})
                parity = rec.get("parity", {})

                row["legacy_stap_cold_ms"] = (leg.get("stap_ms") or {}).get("cold_ms", "")
                row["legacy_stap_steady_mean_ms"] = (leg.get("stap_ms") or {}).get("steady_mean_ms", "")
                row["legacy_stap_steady_median_ms"] = (leg.get("stap_ms") or {}).get("steady_median_ms", "")

                row["optimized_stap_cold_ms"] = (opt.get("stap_ms") or {}).get("cold_ms", "")
                row["optimized_stap_steady_mean_ms"] = (opt.get("stap_ms") or {}).get("steady_mean_ms", "")
                row["optimized_stap_steady_median_ms"] = (opt.get("stap_ms") or {}).get("steady_median_ms", "")

                row["stap_speedup_mean"] = spd.get("stap_mean", "")
                row["legacy_total_steady_mean_ms"] = (leg.get("total_ms") or {}).get("steady_mean_ms", "")
                row["optimized_total_steady_mean_ms"] = (opt.get("total_ms") or {}).get("steady_mean_ms", "")
                row["total_speedup_mean"] = spd.get("total_mean", "")

                row["parity_overall_max_abs"] = parity.get("overall_max_abs", "")
                score_stats = ((parity.get("files") or {}).get("score_stap.npy") or {})
                row["parity_score_stap_max_abs"] = score_stats.get("max", "")

            w.writerow(row)


if __name__ == "__main__":
    main()
