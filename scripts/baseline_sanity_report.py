#!/usr/bin/env python3
"""
Baseline sanity report: quantiles + fixed-FPR operating points.

This is a lightweight diagnostic intended to answer:
  "Is the baseline score implementation functioning, but failing in the strict tail?"

It loads precomputed score maps + masks from acceptance bundles and reports:
  - background/flow quantiles
  - thresholds at fixed background tail rates (FPR targets alpha)
  - realized FPR + flow hit rate (TPR) at those thresholds

Input modes:
  (1) --bundle <dir> ... (direct bundle dirs; prints score_base + stap scores if present)
  (2) --fair-matrix-json <reports/fair_matrix_*.json> (select scenario/method bundle dirs; prints
      score_base for baselines and score_stap_preka for STAP rows)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_ALPHAS = (1e-3, 1e-2, 1e-1)
DEFAULT_QUANTILES = (0.5, 0.9, 0.99, 0.999)
DEFAULT_FAIR_SCENARIOS = (
    "open_seed1_off0",
    "aliascontract_seed2_off0",
    "skullor_seed2_off0",
)


def _parse_float_list(spec: str) -> list[float]:
    parts = [p.strip() for p in str(spec or "").split(",") if p.strip()]
    return [float(p) for p in parts]


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


def _quantiles(x: np.ndarray, qs: Iterable[float]) -> dict[float, float]:
    x = _finite_flat(x)
    if x.size <= 0:
        return {float(q): float("nan") for q in qs}
    return {float(q): float(np.quantile(x, float(q))) for q in qs}


def _load_bool(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr.astype(bool, copy=False)


def _load_score(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    return arr.astype(np.float64, copy=False)


@dataclass(frozen=True)
class TailPoint:
    alpha: float
    thr: float
    fpr_realized: float
    tpr: float


@dataclass(frozen=True)
class ScoreReport:
    score_path: str
    n_bg: int
    n_flow: int
    bg_quantiles: dict[float, float]
    flow_quantiles: dict[float, float]
    points: list[TailPoint]


def _score_report(
    *,
    bundle_dir: Path,
    score_path: Path,
    mask_bg: np.ndarray,
    mask_flow: np.ndarray,
    alphas: list[float],
    quantiles: list[float],
) -> ScoreReport:
    score = _load_score(score_path)
    if score.shape != mask_bg.shape or score.shape != mask_flow.shape:
        raise ValueError(
            f"Shape mismatch in {bundle_dir}: score={score.shape} bg={mask_bg.shape} flow={mask_flow.shape}"
        )

    bg = score[mask_bg]
    flow = score[mask_flow]
    bg_q = _quantiles(bg, quantiles)
    flow_q = _quantiles(flow, quantiles)

    points: list[TailPoint] = []
    flow_finite = _finite_flat(flow)
    for a in alphas:
        thr, fpr_realized = _right_tail_threshold(bg, a)
        tpr = float(np.mean(flow_finite >= float(thr))) if np.isfinite(thr) else 0.0
        points.append(TailPoint(alpha=float(a), thr=float(thr), fpr_realized=float(fpr_realized), tpr=float(tpr)))

    return ScoreReport(
        score_path=str(score_path),
        n_bg=int(np.sum(mask_bg)),
        n_flow=int(np.sum(mask_flow)),
        bg_quantiles=bg_q,
        flow_quantiles=flow_q,
        points=points,
    )


def _fmt_q(qs: dict[float, float]) -> str:
    parts: list[str] = []
    for q in sorted(qs.keys()):
        v = qs[q]
        parts.append(f"q{q:g}={v:.4g}" if np.isfinite(v) else f"q{q:g}=nan")
    return ", ".join(parts)


def _fmt_points(points: list[TailPoint]) -> str:
    return " | ".join(f"a={p.alpha:g}: fpr={p.fpr_realized:.3g} tpr={p.tpr:.3g}" for p in points)


def _infer_score_candidates(bundle_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in ("score_base.npy", "score_stap_preka.npy", "score_stap.npy"):
        p = bundle_dir / name
        if p.is_file():
            candidates.append(p)
    return candidates


def _summarize_bundle(
    *,
    bundle_dir: Path,
    score_paths: list[Path] | None,
    alphas: list[float],
    quantiles: list[float],
) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    mask_bg_path = bundle_dir / "mask_bg.npy"
    mask_flow_path = bundle_dir / "mask_flow.npy"
    if not mask_bg_path.is_file() or not mask_flow_path.is_file():
        raise FileNotFoundError(f"Missing mask(s) in bundle dir: {bundle_dir}")

    mask_bg = _load_bool(mask_bg_path)
    mask_flow = _load_bool(mask_flow_path)
    if mask_bg.shape != mask_flow.shape:
        raise ValueError(f"Mask shape mismatch in {bundle_dir}: bg={mask_bg.shape} flow={mask_flow.shape}")

    if score_paths is None:
        score_paths = _infer_score_candidates(bundle_dir)
    if not score_paths:
        raise FileNotFoundError(f"No score_*.npy found in {bundle_dir}")

    reports: list[ScoreReport] = []
    for sp in score_paths:
        reports.append(
            _score_report(
                bundle_dir=bundle_dir,
                score_path=sp,
                mask_bg=mask_bg,
                mask_flow=mask_flow,
                alphas=alphas,
                quantiles=quantiles,
            )
        )

    max_abs_delta_base_stap = None
    base_path = bundle_dir / "score_base.npy"
    stap_path = bundle_dir / "score_stap_preka.npy"
    if base_path.is_file() and stap_path.is_file():
        base = _load_score(base_path)
        stap = _load_score(stap_path)
        if base.shape == stap.shape:
            max_abs_delta_base_stap = float(np.max(np.abs(base - stap)))

    return {
        "bundle_dir": str(bundle_dir),
        "max_abs_delta_base_vs_stap_preka": max_abs_delta_base_stap,
        "scores": [asdict(r) for r in reports],
    }


def _load_fair_matrix_rows(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, list) or not all(isinstance(r, dict) for r in obj):
        raise ValueError(f"Expected a list[dict] JSON at {path}")
    return obj  # type: ignore[return-value]


def _fair_matrix_bundle_map(
    rows: list[dict[str, Any]],
    *,
    scenarios: list[str],
) -> dict[str, dict[str, Path]]:
    out: dict[str, dict[str, Path]] = {s: {} for s in scenarios}
    for r in rows:
        scenario = str(r.get("scenario") or "")
        method = str(r.get("method") or "")
        bdir = r.get("bundle_dir")
        if scenario not in out or not method or not bdir:
            continue
        out[scenario][method] = Path(str(bdir))
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", type=Path, action="append", default=[], help="Acceptance bundle dir (repeatable).")
    ap.add_argument(
        "--fair-matrix-json",
        type=Path,
        default=None,
        help="Optional reports/fair_matrix_*.json to select scenario/method bundle dirs.",
    )
    ap.add_argument(
        "--fair-scenarios",
        type=str,
        default=",".join(DEFAULT_FAIR_SCENARIOS),
        help="Comma-separated fair-matrix scenario keys (default: %(default)s).",
    )
    ap.add_argument(
        "--alphas",
        type=str,
        default=",".join(f"{a:g}" for a in DEFAULT_ALPHAS),
        help="Comma-separated FPR targets alpha (default: %(default)s).",
    )
    ap.add_argument(
        "--quantiles",
        type=str,
        default=",".join(f"{q:g}" for q in DEFAULT_QUANTILES),
        help="Comma-separated quantiles to report (default: %(default)s).",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="Optional output JSON path.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    alphas = _parse_float_list(args.alphas)
    quantiles = _parse_float_list(args.quantiles)

    out: dict[str, Any] = {
        "alphas": alphas,
        "quantiles": quantiles,
        "bundles": [],
        "fair_matrix": None,
    }

    if args.fair_matrix_json is not None:
        scenarios = [s.strip() for s in str(args.fair_scenarios).split(",") if s.strip()]
        rows = _load_fair_matrix_rows(Path(args.fair_matrix_json))
        bundle_map = _fair_matrix_bundle_map(rows, scenarios=scenarios)
        out["fair_matrix"] = {"path": str(args.fair_matrix_json), "scenarios": scenarios}
        print(f"[sanity] fair_matrix_json={args.fair_matrix_json} scenarios={scenarios}")

        for scenario in scenarios:
            methods = bundle_map.get(scenario) or {}
            if not methods:
                print(f"\n=== Scenario {scenario}: (no rows found)\n")
                continue

            print(f"\n=== Scenario {scenario} ===")
            for method in sorted(methods.keys()):
                bdir = methods[method]
                score_paths = (
                    [bdir / "score_stap_preka.npy"]
                    if method.strip().lower().startswith("stap")
                    else [bdir / "score_base.npy"]
                )
                try:
                    rep = _summarize_bundle(bundle_dir=bdir, score_paths=score_paths, alphas=alphas, quantiles=quantiles)
                except Exception as exc:
                    print(f"  - {method}: ERROR {exc}")
                    continue

                score = rep["scores"][0]
                print(f"  - {method}")
                print(f"    bundle={bdir}")
                print(f"    score={Path(score['score_path']).name} n_bg={score['n_bg']} n_flow={score['n_flow']}")
                print(f"    bg:   {_fmt_q(score['bg_quantiles'])}")
                print(f"    flow: {_fmt_q(score['flow_quantiles'])}")
                pts = [TailPoint(**p) for p in score["points"]]
                print(f"    tail: {_fmt_points(pts)}")

                out["bundles"].append({"scenario": scenario, "method": method, **rep})

    for b in args.bundle:
        try:
            rep = _summarize_bundle(bundle_dir=b, score_paths=None, alphas=alphas, quantiles=quantiles)
        except Exception as exc:
            print(f"[sanity] bundle={b}: ERROR {exc}")
            continue

        print(f"\n=== Bundle {b} ===")
        if rep.get("max_abs_delta_base_vs_stap_preka") is not None:
            d = float(rep["max_abs_delta_base_vs_stap_preka"])
            print(f"  max_abs_delta(score_base, score_stap_preka)={d:.6g}")
        for s in rep["scores"]:
            print(f"  - score={Path(s['score_path']).name} n_bg={s['n_bg']} n_flow={s['n_flow']}")
            print(f"    bg:   {_fmt_q(s['bg_quantiles'])}")
            print(f"    flow: {_fmt_q(s['flow_quantiles'])}")
            pts = [TailPoint(**p) for p in s["points"]]
            print(f"    tail: {_fmt_points(pts)}")
        out["bundles"].append({"scenario": None, "method": None, **rep})

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"[sanity] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

