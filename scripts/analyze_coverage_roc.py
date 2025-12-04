#!/usr/bin/env python3
"""Compute coverage-conditioned ROC/AUC summaries from STAP bundle outputs."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np

from eval.metrics import partial_auc, roc_curve, tpr_at_fpr_target

REQUIRED_COVERAGE_KEYS = (
    "tile_flow_coverage_p50",
    "tile_flow_coverage_p90",
    "flow_cov_ge_20_fraction",
    "flow_cov_ge_50_fraction",
    "flow_cov_ge_80_fraction",
)


@dataclass
class BundleData:
    name: str
    path: Path
    meta: dict
    mask_flow: np.ndarray
    mask_bg: np.ndarray
    stap_map: np.ndarray
    base_map: np.ndarray
    tile_hw: tuple[int, int]
    stride: int
    telemetry: dict | None = None


@dataclass
class CoverageResult:
    threshold: float
    retained_flow_fraction: float
    retained_area_fraction: float
    tile_fraction: float
    n_flow: int
    n_bg: int
    fpr_floor: float
    auc_base: float
    auc_stap: float
    delta_auc: float
    tpr_base: float
    tpr_stap: float
    delta_tpr: float
    pauc_base: float
    pauc_stap: float
    delta_pauc: float
    tpr_extra: list[dict[str, float]]
    stap_pos: np.ndarray
    stap_neg: np.ndarray
    base_pos: np.ndarray
    base_neg: np.ndarray
    selected_tiles: int
    total_tiles: int
    total_flow: int
    total_area: int


@dataclass
class BundleSummary:
    bundle: BundleData
    telemetry: dict
    coverage_results: list[CoverageResult]
    latency_ms: float | None = None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _ensure_coverage_telemetry(meta: dict) -> dict:
    telemetry = meta.get("stap_fallback_telemetry") or {}
    missing = [key for key in REQUIRED_COVERAGE_KEYS if key not in telemetry]
    if missing:
        raise KeyError(
            f"Bundle meta missing coverage telemetry fields: {', '.join(sorted(missing))}"
        )
    return telemetry


def _select_maps(bundle_dir: Path, score_mode: str) -> tuple[np.ndarray, np.ndarray]:
    score_mode = score_mode.lower()
    if score_mode == "msd":
        stap_map = np.load(bundle_dir / "stap_score_pool_map.npy")
        base_map = np.load(bundle_dir / "pd_base.npy")
    elif score_mode == "pd":
        stap_map = np.load(bundle_dir / "pd_stap.npy")
        base_map = np.load(bundle_dir / "pd_base.npy")
    elif score_mode == "band_ratio":
        stap_map = np.load(bundle_dir / "stap_band_ratio_map.npy")
        base_map = np.load(bundle_dir / "base_band_ratio_map.npy")
    else:
        raise ValueError(f"Unsupported score_mode '{score_mode}'. Expected msd/pd/band_ratio.")
    return stap_map.astype(np.float32, copy=False), base_map.astype(np.float32, copy=False)


def _tile_iter(
    shape: tuple[int, int], tile_hw: tuple[int, int], stride: int
) -> Iterable[tuple[int, int]]:
    H, W = shape
    th, tw = tile_hw
    if H < th or W < tw:
        return []
    for y in range(0, H - th + 1, stride):
        for x in range(0, W - tw + 1, stride):
            yield y, x


def _compute_tile_coverages(
    mask_flow: np.ndarray, tile_hw: tuple[int, int], stride: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    coords: list[tuple[int, int]] = []
    covs: list[float] = []
    th, tw = tile_hw
    for y0, x0 in _tile_iter(mask_flow.shape, tile_hw, stride):
        tile = mask_flow[y0 : y0 + th, x0 : x0 + tw]
        cov = float(tile.mean())
        coords.append((y0, x0))
        covs.append(cov)
    return np.asarray(covs, dtype=np.float32), coords


def _tile_gate(
    tile_covs: np.ndarray,
    tile_coords: Sequence[tuple[int, int]],
    threshold: float,
    shape: tuple[int, int],
    tile_hw: tuple[int, int],
) -> tuple[np.ndarray, int]:
    th, tw = tile_hw
    gate = np.zeros(shape, dtype=bool)
    selected = 0
    for cov, (y0, x0) in zip(tile_covs, tile_coords, strict=False):
        if cov >= threshold:
            selected += 1
            gate[y0 : y0 + th, x0 : x0 + tw] = True
    return gate, selected


def _roc_metrics(
    stap_pos: np.ndarray,
    stap_neg: np.ndarray,
    base_pos: np.ndarray,
    base_neg: np.ndarray,
    fpr_target: float,
    pauc_max: float,
    extra_targets: Sequence[float],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    list[dict[str, float]],
]:
    if stap_pos.size == 0 or stap_neg.size == 0 or base_pos.size == 0 or base_neg.size == 0:
        raise ValueError("Insufficient samples after coverage gating")
    fpr_b, tpr_b, _ = roc_curve(base_pos, base_neg)
    fpr_s, tpr_s, _ = roc_curve(stap_pos, stap_neg)
    auc_b = partial_auc(fpr_b, tpr_b, fpr_max=fpr_target)
    auc_s = partial_auc(fpr_s, tpr_s, fpr_max=fpr_target)
    pauc_b = partial_auc(fpr_b, tpr_b, fpr_max=pauc_max)
    pauc_s = partial_auc(fpr_s, tpr_s, fpr_max=pauc_max)
    tpr_b_target = tpr_at_fpr_target(fpr_b, tpr_b, target_fpr=fpr_target)
    tpr_s_target = tpr_at_fpr_target(fpr_s, tpr_s, target_fpr=fpr_target)
    extras: list[dict[str, float]] = []
    for target in extra_targets:
        val_b = tpr_at_fpr_target(fpr_b, tpr_b, target_fpr=target)
        val_s = tpr_at_fpr_target(fpr_s, tpr_s, target_fpr=target)
        extras.append(
            {
                "fpr_target": float(target),
                "tpr_base": float(val_b),
                "tpr_stap": float(val_s),
                "delta_tpr": float(val_s - val_b),
            }
        )
    return (
        float(auc_b),
        float(auc_s),
        float(auc_s - auc_b),
        float(tpr_b_target),
        float(tpr_s_target),
        float(tpr_s_target - tpr_b_target),
        float(pauc_b),
        float(pauc_s),
        float(pauc_s - pauc_b),
        extras,
    )


def load_bundle(
    bundle_dir: Path,
    *,
    score_mode: str | None = None,
    flow_mask_kind: str = "default",
) -> BundleData:
    meta = _load_json(bundle_dir / "meta.json")
    telemetry = _ensure_coverage_telemetry(meta)
    tile_hw = tuple(int(v) for v in meta.get("tile_hw", (0, 0)))
    if len(tile_hw) != 2 or tile_hw[0] <= 0 or tile_hw[1] <= 0:
        raise ValueError(f"Invalid tile_hw in meta.json for bundle {bundle_dir}")
    stride = int(meta.get("tile_stride", meta.get("tile_step", 4)))
    if stride <= 0:
        raise ValueError(f"Invalid tile stride ({stride}) in bundle {bundle_dir}")
    mode = score_mode or meta.get("score_pool_default", "msd")
    stap_map, base_map = _select_maps(bundle_dir, mode)
    flow_mask_path = bundle_dir / "mask_flow.npy"
    if flow_mask_kind and flow_mask_kind.lower() != "default":
        if flow_mask_kind.lower() == "pd":
            candidate = bundle_dir / "mask_flow_pd.npy"
            if not candidate.exists():
                raise FileNotFoundError(
                    f"mask_flow_pd.npy missing in bundle {bundle_dir} (flow-mask-kind=pd)"
                )
            flow_mask_path = candidate
        else:
            raise ValueError(
                f"Unsupported flow mask kind '{flow_mask_kind}'. Expected 'default' or 'pd'."
            )
    mask_flow = np.load(flow_mask_path).astype(bool)
    mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)
    name = meta.get("dataset_name") or bundle_dir.name
    # ensure telemetry keys exist even if not used downstream
    _ = telemetry
    return BundleData(
        name=name,
        path=bundle_dir,
        meta=meta,
        mask_flow=mask_flow,
        mask_bg=mask_bg,
        stap_map=stap_map,
        base_map=base_map,
        tile_hw=tile_hw,
        stride=stride,
        telemetry=meta.get("stap_fallback_telemetry"),
    )


def summarize_bundle(
    bundle: BundleData,
    thresholds: Sequence[float],
    fpr_target: float,
    pauc_max: float,
    extra_targets: Sequence[float],
) -> BundleSummary:
    telemetry = _ensure_coverage_telemetry(bundle.meta)
    tile_covs, coords = _compute_tile_coverages(
        bundle.mask_flow.astype(float), bundle.tile_hw, bundle.stride
    )
    results: list[CoverageResult] = []
    H, W = bundle.mask_flow.shape
    total_tiles = len(coords)
    total_flow = int(bundle.mask_flow.sum())
    total_area = H * W
    for thr in thresholds:
        gate, selected_tiles = _tile_gate(tile_covs, coords, thr, (H, W), bundle.tile_hw)
        if selected_tiles == 0:
            continue
        flow_mask = bundle.mask_flow & gate
        bg_mask = bundle.mask_bg & gate
        n_flow = int(flow_mask.sum())
        n_bg = int(bg_mask.sum())
        if n_flow == 0 or n_bg == 0:
            continue
        stap_pos = bundle.stap_map[flow_mask]
        stap_neg = bundle.stap_map[bg_mask]
        base_pos = bundle.base_map[flow_mask]
        base_neg = bundle.base_map[bg_mask]
        retained_flow_fraction = (n_flow / total_flow) if total_flow > 0 else 0.0
        retained_area_fraction = float(gate.mean()) if total_area > 0 else 0.0
        tile_fraction = selected_tiles / total_tiles if total_tiles else 0.0
        (
            auc_b,
            auc_s,
            delta_auc,
            tpr_b,
            tpr_s,
            delta_tpr,
            pauc_b,
            pauc_s,
            delta_pauc,
            tpr_extra,
        ) = _roc_metrics(
            stap_pos, stap_neg, base_pos, base_neg, fpr_target, pauc_max, extra_targets
        )
        results.append(
            CoverageResult(
                threshold=float(thr),
                retained_flow_fraction=float(retained_flow_fraction),
                retained_area_fraction=float(retained_area_fraction),
                tile_fraction=float(tile_fraction),
                n_flow=n_flow,
                n_bg=n_bg,
                fpr_floor=(1.0 / n_bg) if n_bg > 0 else 0.0,
                auc_base=auc_b,
                auc_stap=auc_s,
                delta_auc=delta_auc,
                tpr_base=tpr_b,
                tpr_stap=tpr_s,
                delta_tpr=delta_tpr,
                pauc_base=pauc_b,
                pauc_stap=pauc_s,
                delta_pauc=delta_pauc,
                tpr_extra=tpr_extra,
                stap_pos=stap_pos,
                stap_neg=stap_neg,
                base_pos=base_pos,
                base_neg=base_neg,
                selected_tiles=selected_tiles,
                total_tiles=total_tiles,
                total_flow=total_flow,
                total_area=total_area,
            )
        )
    latency_ms: float | None = None
    if bundle.telemetry:
        val = bundle.telemetry.get("stap_ms")
        if val is not None:
            try:
                latency_ms = float(val)
            except (TypeError, ValueError):
                latency_ms = None
    return BundleSummary(
        bundle=bundle,
        telemetry=telemetry,
        coverage_results=results,
        latency_ms=latency_ms,
    )


def aggregate_label_summaries(
    label: str,
    summaries: Sequence[BundleSummary],
    thresholds: Sequence[float],
    fpr_target: float,
    pauc_max: float,
    extra_targets: Sequence[float],
) -> BundleSummary:
    agg_results: list[CoverageResult] = []
    for thr in thresholds:
        rows: list[CoverageResult] = []
        for summary in summaries:
            for row in summary.coverage_results:
                if abs(row.threshold - thr) < 1e-6:
                    rows.append(row)
        if not rows:
            continue
        stap_pos = np.concatenate([row.stap_pos for row in rows], axis=0)
        stap_neg = np.concatenate([row.stap_neg for row in rows], axis=0)
        base_pos = np.concatenate([row.base_pos for row in rows], axis=0)
        base_neg = np.concatenate([row.base_neg for row in rows], axis=0)
        n_flow = sum(row.n_flow for row in rows)
        n_bg = sum(row.n_bg for row in rows)
        total_flow = sum(row.total_flow for row in rows)
        total_area = sum(row.total_area for row in rows)
        selected_tiles = sum(row.selected_tiles for row in rows)
        total_tiles = sum(row.total_tiles for row in rows)
        retained_flow_fraction = (n_flow / total_flow) if total_flow else 0.0
        retained_area_px = sum(row.retained_area_fraction * row.total_area for row in rows)
        retained_area_fraction = (retained_area_px / total_area) if total_area else 0.0
        tile_fraction = (selected_tiles / total_tiles) if total_tiles else 0.0
        (
            auc_b,
            auc_s,
            delta_auc,
            tpr_b,
            tpr_s,
            delta_tpr,
            pauc_b,
            pauc_s,
            delta_pauc,
            tpr_extra,
        ) = _roc_metrics(
            stap_pos, stap_neg, base_pos, base_neg, fpr_target, pauc_max, extra_targets
        )
        agg_results.append(
            CoverageResult(
                threshold=float(thr),
                retained_flow_fraction=float(retained_flow_fraction),
                retained_area_fraction=float(retained_area_fraction),
                tile_fraction=float(tile_fraction),
                n_flow=int(n_flow),
                n_bg=int(n_bg),
                fpr_floor=(1.0 / n_bg) if n_bg > 0 else 0.0,
                auc_base=auc_b,
                auc_stap=auc_s,
                delta_auc=delta_auc,
                tpr_base=tpr_b,
                tpr_stap=tpr_s,
                delta_tpr=delta_tpr,
                pauc_base=pauc_b,
                pauc_stap=pauc_s,
                delta_pauc=delta_pauc,
                tpr_extra=tpr_extra,
                stap_pos=stap_pos,
                stap_neg=stap_neg,
                base_pos=base_pos,
                base_neg=base_neg,
                selected_tiles=selected_tiles,
                total_tiles=total_tiles,
                total_flow=total_flow,
                total_area=total_area,
            )
        )
    dummy_bundle = BundleData(
        name=f"aggregate:{label}",
        path=Path(f"aggregate:{label}"),
        meta={},
        mask_flow=np.zeros((1, 1), dtype=bool),
        mask_bg=np.zeros((1, 1), dtype=bool),
        stap_map=np.zeros((1, 1), dtype=np.float32),
        base_map=np.zeros((1, 1), dtype=np.float32),
        tile_hw=(0, 0),
        stride=0,
        telemetry=None,
    )
    latencies = [s.latency_ms for s in summaries if s.latency_ms is not None]
    latency_ms = float(np.mean(latencies)) if latencies else None
    return BundleSummary(
        bundle=dummy_bundle,
        telemetry={},
        coverage_results=agg_results,
        latency_ms=latency_ms,
    )


def _format_pct(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def _print_summary(
    summary: BundleSummary, fpr_target: float, pauc_max: float, extra_targets: Sequence[float]
) -> None:
    print(f"Bundle: {summary.bundle.name}")
    stap_ms = None
    if summary.bundle.telemetry:
        stap_ms = summary.bundle.telemetry.get("stap_ms")
    print(
        "  Telemetry: tile_flow_cov50="
        f"{summary.telemetry.get('tile_flow_coverage_p50')} | flow_cov_ge_50="
        f"{summary.telemetry.get('flow_cov_ge_50_fraction')}"
    )
    if stap_ms is not None:
        print(f"  stap_ms: {stap_ms:.0f} ms")
    if not summary.coverage_results:
        print("  No coverage slice produced usable samples.")
        return
    header = (
        "  thr  | tiles  | flow   | area   | N_neg | FPR_floor | auc_b  | auc_s  | dAUC  |"
        " TPRb  | TPRs  | dTPR  | pAUCb | pAUCs | dPAUC"
    )
    print(header)
    row_fmt = (
        "  {thr:4.2f} | {tiles:>5} | {flow} | {area} | {n_bg:5d} | {fpr_floor:9.2e} |"
        " {auc_b:7.4f} | {auc_s:7.4f} | {dauc:6.4f} | {tpr_b:6.4f} | {tpr_s:6.4f} |"
        " {dtpr:6.4f} | {pauc_b:6.4f} | {pauc_s:6.4f} | {dpauc:6.4f}"
    )
    for row in summary.coverage_results:
        print(
            row_fmt.format(
                thr=row.threshold,
                tiles=_format_pct(row.tile_fraction),
                flow=_format_pct(row.retained_flow_fraction),
                area=_format_pct(row.retained_area_fraction),
                n_bg=row.n_bg,
                fpr_floor=row.fpr_floor,
                auc_b=row.auc_base,
                auc_s=row.auc_stap,
                dauc=row.delta_auc,
                tpr_b=row.tpr_base,
                tpr_s=row.tpr_stap,
                dtpr=row.delta_tpr,
                pauc_b=row.pauc_base,
                pauc_s=row.pauc_stap,
                dpauc=row.delta_pauc,
            )
        )
        for entry in row.tpr_extra:
            print(
                "       └ TPR@{fpr:6.1e}: base={tpr_b:6.4f} | stap={tpr_s:6.4f} | "
                "Δ={dtpr:6.4f}".format(
                    fpr=entry["fpr_target"],
                    tpr_b=entry["tpr_base"],
                    tpr_s=entry["tpr_stap"],
                    dtpr=entry["delta_tpr"],
                )
            )
    print(
        f"  Metrics computed at FPR target = {fpr_target:.1e}; pAUC integrates up to {pauc_max:.1e}\n"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Coverage-conditioned ROC/AUC analyzer")
    ap.add_argument(
        "--bundle",
        dest="bundles",
        action="append",
        required=True,
        help="Path to a bundle directory (pw_*). Repeat for multiple bundles.",
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5, 0.8],
        help="Coverage thresholds (tile flow fraction) to evaluate.",
    )
    ap.add_argument(
        "--fpr-target", type=float, default=1e-5, help="Target FPR for TPR/partial AUC"
    )
    ap.add_argument(
        "--extra-fpr-targets",
        type=float,
        nargs="+",
        default=None,
        help="Additional FPR targets to report TPR deltas for (e.g., 1e-4 1e-3).",
    )
    ap.add_argument(
        "--pauc-max",
        type=float,
        default=1e-3,
        help="FPR limit for reporting low-FPR partial AUC (pAUC_lowFPR).",
    )
    ap.add_argument(
        "--score-mode",
        type=str,
        default=None,
        help="Override score mode (msd/pd/band_ratio). Defaults to bundle meta setting.",
    )
    ap.add_argument(
        "--flow-mask-kind",
        type=str,
        choices=("default", "pd"),
        default="default",
        help="Flow mask used for coverage gating (default=mask_flow.npy, pd=mask_flow_pd.npy).",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate bundles by label (specify bundles as label=path).",
    )
    ap.add_argument("--json", type=Path, default=None, help="Optional path to dump JSON summary.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = sorted(set(float(t) for t in args.thresholds))
    summaries: list[BundleSummary] = []
    bundle_specs: list[tuple[str | None, Path]] = []
    for spec in args.bundles:
        if "=" in spec:
            label, _, path_str = spec.partition("=")
            label = label.strip() or None
            bundle_specs.append((label, Path(path_str).expanduser().resolve()))
        else:
            bundle_specs.append((None, Path(spec).expanduser().resolve()))
    if args.aggregate:
        unlabeled = [str(path) for label, path in bundle_specs if not label]
        if unlabeled:
            raise ValueError("--aggregate requires bundles specified as label=path")
    extra_targets = []
    if args.extra_fpr_targets:
        for t in args.extra_fpr_targets:
            if t <= 0:
                continue
            if abs(t - args.fpr_target) < 1e-12:
                continue
            extra_targets.append(float(t))
    extra_targets = sorted(set(extra_targets))
    for _, bundle_dir in bundle_specs:
        bundle = load_bundle(
            bundle_dir,
            score_mode=args.score_mode,
            flow_mask_kind=args.flow_mask_kind,
        )
        summaries.append(
            summarize_bundle(bundle, thresholds, args.fpr_target, args.pauc_max, extra_targets)
        )
    for summary in summaries:
        _print_summary(summary, args.fpr_target, args.pauc_max, extra_targets)
    aggregate_summaries: list[BundleSummary] = []
    if args.aggregate:
        label_map: dict[str, list[BundleSummary]] = {}
        for (label, _), summary in zip(bundle_specs, summaries):
            assert label is not None
            label_map.setdefault(label, []).append(summary)
        for label, group in label_map.items():
            agg_summary = aggregate_label_summaries(
                label, group, thresholds, args.fpr_target, args.pauc_max, extra_targets
            )
            aggregate_summaries.append(agg_summary)
            _print_summary(agg_summary, args.fpr_target, args.pauc_max, extra_targets)
    if args.json:

        def row_to_dict(row: CoverageResult) -> dict:
            return {
                "threshold": row.threshold,
                "retained_flow_fraction": row.retained_flow_fraction,
                "retained_area_fraction": row.retained_area_fraction,
                "tile_fraction": row.tile_fraction,
                "n_flow": row.n_flow,
                "n_bg": row.n_bg,
                "n_neg": row.n_bg,
                "fpr_floor": row.fpr_floor,
                "auc_base": row.auc_base,
                "auc_stap": row.auc_stap,
                "delta_auc": row.delta_auc,
                "tpr_base": row.tpr_base,
                "tpr_stap": row.tpr_stap,
                "delta_tpr": row.delta_tpr,
                "pauc_lowfpr_base": row.pauc_base,
                "pauc_lowfpr_stap": row.pauc_stap,
                "pauc_lowfpr_delta": row.delta_pauc,
                "tpr_extra": row.tpr_extra,
            }

        payload = []
        for summary in summaries:
            payload.append(
                {
                    "bundle": summary.bundle.name,
                    "path": str(summary.bundle.path),
                    "coverage_results": [row_to_dict(row) for row in summary.coverage_results],
                    "telemetry": {
                        key: summary.telemetry.get(key) for key in REQUIRED_COVERAGE_KEYS
                    },
                    "latency_ms": summary.latency_ms,
                }
            )
        for summary in aggregate_summaries:
            payload.append(
                {
                    "bundle": summary.bundle.name,
                    "path": str(summary.bundle.path),
                    "coverage_results": [row_to_dict(row) for row in summary.coverage_results],
                    "telemetry": {},
                    "latency_ms": summary.latency_ms,
                }
            )
        args.json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
