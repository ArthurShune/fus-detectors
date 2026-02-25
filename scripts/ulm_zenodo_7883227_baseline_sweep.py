from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from pipeline.realdata.ulm_zenodo_7883227 import (
    load_ulm_block_iq,
    load_ulm_zenodo_7883227_params,
)
from sim.kwave.common import _phasecorr_shift
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> tuple[slice, str]:
    spec = (spec or "").strip()
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]'.")
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    if stop is None:
        raise ValueError("Slice spec must include stop (e.g. 0:128).")
    tag = f"{start:04d}_{stop:04d}" if step == 1 else f"{start:04d}_{stop:04d}_s{step}"
    return slice(start, stop, step), tag


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated integer list.")
    return out


def _parse_float_list(spec: str) -> list[float]:
    out: list[float] = []
    for part in (spec or "").replace(" ", "").split(","):
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated float list.")
    return out


def _shannon_entropy(x: np.ndarray, *, bins: int = 128) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = x - float(np.mean(x))
    std = float(np.std(x)) + 1e-12
    x = np.clip(x / std, -6.0, 6.0)
    hist, _ = np.histogram(x, bins=int(max(8, bins)), range=(-6.0, 6.0), density=False)
    p = hist.astype(np.float64)
    s = float(np.sum(p))
    if s <= 0.0:
        return float("nan")
    p /= s
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def _connected_components(binary: np.ndarray, connectivity: int = 4) -> int:
    binary = np.asarray(binary, dtype=bool)
    if binary.size == 0:
        return 0
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=int)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    _, n = ndi.label(binary, structure=structure)
    return int(n)


def _threshold_at_fpr(score: np.ndarray, neg_mask: np.ndarray, fpr: float) -> float:
    score = np.asarray(score, dtype=np.float64)
    neg = np.asarray(neg_mask, dtype=bool)
    vals = score[neg]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    fpr = float(np.clip(fpr, 1e-9, 1.0 - 1e-9))
    return float(np.quantile(vals, 1.0 - fpr))


def _safe_get(d: dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _profile_to_bands(profile: str) -> tuple[float, float, float, float]:
    p = (profile or "").strip().lower()
    if p in {"ulm", "u", "shin_u", "ulm_u"}:
        # Reuse the Shin-U-like telemetry bands at 1 kHz slow-time.
        return 60.0, 250.0, 400.0, 100.0
    if p in {"brain", "brain_default"}:
        return 30.0, 220.0, 400.0, 120.0
    raise ValueError(f"Unknown profile {profile!r}. Use ULM or brain.")


def _format_energy_tag(e: float) -> str:
    return f"e{int(round(1000.0 * float(e))):03d}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-time, label-free baseline calibration sweep for ULM Zenodo 7883227.\n"
            "Sweeps MC-SVD energy fraction on a small calibration subset, writes bundles,\n"
            "and reports contract-v2 telemetry + hygiene/stability proxies (no ROC tuning)."
        )
    )
    parser.add_argument("--block-ids", type=str, default="1,2,3")
    parser.add_argument("--frames", type=str, default="0:128")
    parser.add_argument(
        "--svd-energy-frac-list",
        type=str,
        default="0.90,0.95,0.97,0.975,0.98,0.99",
        help="Comma-separated MC-SVD energy fractions to try (default: %(default)s).",
    )
    parser.add_argument("--prf-hz", type=float, default=None, help="Slow-time rate (default: param.json FrameRate).")
    parser.add_argument("--profile", type=str, default="ULM", help="Telemetry band profile (default: %(default)s).")

    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)
    parser.add_argument("--Lt", type=int, default=64, help="Used for bundle metadata + band-ratio tiling.")
    parser.add_argument(
        "--reg-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable rigid phase-correlation registration in baseline (default: %(default)s).",
    )
    parser.add_argument("--reg-subpixel", type=int, default=4)

    parser.add_argument(
        "--run-stap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run STAP in the sweep bundles (default: %(default)s).",
    )
    parser.add_argument(
        "--bg-tail-fpr",
        type=float,
        default=1e-3,
        help="Right-tail rate for background-tail hygiene (default: %(default)s).",
    )
    parser.add_argument(
        "--stability-split",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also compute stability proxies by splitting frames into two halves (default: %(default)s).",
    )

    parser.add_argument("--cache-dir", type=Path, default=Path("tmp/ulm_zenodo_7883227"))
    parser.add_argument("--out-root", type=Path, default=Path("runs/real/ulm7883227_baseline_mcsvd_energy_sweep"))
    parser.add_argument("--out-csv", type=Path, default=Path("reports/ulm7883227_baseline_mcsvd_energy_sweep.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/ulm7883227_baseline_mcsvd_energy_sweep.json"))
    args = parser.parse_args()

    block_ids = _parse_int_list(args.block_ids)
    frame_slice, frame_tag = _parse_slice(args.frames)
    energy_fracs = _parse_float_list(args.svd_energy_frac_list)
    energy_fracs = [float(np.clip(e, 0.5, 0.9999)) for e in energy_fracs]

    params = load_ulm_zenodo_7883227_params()
    prf_hz = float(args.prf_hz) if args.prf_hz is not None else float(params.frame_rate_hz)

    flow_low_hz, flow_high_hz, alias_center_hz, alias_hw_hz = _profile_to_bands(args.profile)

    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for block_id in block_ids:
        Icube_full = load_ulm_block_iq(int(block_id), frames=frame_slice, cache_dir=args.cache_dir)
        T, H, W = Icube_full.shape
        if T < 8:
            print(f"[ulm-baseline-sweep] block {block_id}: too few frames ({T}), skipping")
            continue
        t_mid = int(T // 2)
        Icube_a = Icube_full[:t_mid]
        Icube_b = Icube_full[t_mid:]

        # Method-independent structural proxy for optional stability alignment.
        bmode_a = np.mean(np.abs(Icube_a), axis=0).astype(np.float32, copy=False)
        bmode_b = np.mean(np.abs(Icube_b), axis=0).astype(np.float32, copy=False)
        dybm, dxbm, psrbm = _phasecorr_shift(bmode_a, bmode_b, upsample=4)
        if float(psrbm) < 3.0:
            dybm = dxbm = 0.0

        for e in energy_fracs:
            e_tag = _format_energy_tag(float(e))
            dataset_name = f"ulm7883227_block{int(block_id):03d}_{frame_tag}_mcsvd_{e_tag}"
            meta_extra = {
                "orig_data": {
                    "dataset": "ULM_Zenodo_7883227",
                    "block_id": int(block_id),
                    "frames_spec": args.frames,
                    "cache_dir": str(args.cache_dir),
                    "param_json": asdict(params),
                },
                "baseline_sweep": {
                    "svd_energy_frac": float(e),
                    "run_stap": bool(args.run_stap),
                    "stability_split": bool(args.stability_split),
                },
            }

            paths = write_acceptance_bundle_from_icube(
                out_root=Path(args.out_root),
                dataset_name=dataset_name,
                Icube=Icube_full,
                prf_hz=float(prf_hz),
                tile_hw=(int(args.tile_h), int(args.tile_w)),
                tile_stride=int(args.tile_stride),
                Lt=int(args.Lt),
                baseline_type="mc_svd",
                reg_enable=bool(args.reg_enable),
                reg_subpixel=max(1, int(args.reg_subpixel)),
                svd_rank=None,
                svd_energy_frac=float(e),
                run_stap=bool(args.run_stap),
                # Telemetry bands (also used to build risk telemetry in contract).
                band_ratio_flow_low_hz=float(flow_low_hz),
                band_ratio_flow_high_hz=float(flow_high_hz),
                band_ratio_alias_center_hz=float(alias_center_hz),
                band_ratio_alias_width_hz=float(alias_hw_hz),
                score_ka_v2_enable=False,
                meta_extra=meta_extra,
            )
            bundle_dir = Path(paths["meta"]).parent
            meta = json.loads(Path(paths["meta"]).read_text())

            mask_flow = np.load(bundle_dir / "mask_flow.npy").astype(bool)
            mask_bg = np.load(bundle_dir / "mask_bg.npy").astype(bool)
            score_base = np.load(bundle_dir / "score_base.npy").astype(np.float32, copy=False)
            score_base_pdlog = None
            score_base_kasai = None
            try:
                score_base_pdlog = np.load(bundle_dir / "score_base_pdlog.npy").astype(np.float32, copy=False)
            except Exception:
                pass
            try:
                score_base_kasai = np.load(bundle_dir / "score_base_kasai.npy").astype(np.float32, copy=False)
            except Exception:
                pass

            # Contract telemetry (label-free).
            ka_v2 = meta.get("ka_contract_v2") or {}
            ka_metrics = (ka_v2.get("metrics") or {}) if isinstance(ka_v2, dict) else {}

            def _hygiene(stats_score: np.ndarray) -> tuple[int, int, float]:
                thr = _threshold_at_fpr(stats_score, mask_bg, float(args.bg_tail_fpr))
                hit = mask_bg & (np.asarray(stats_score, dtype=np.float64) >= float(thr))
                area = int(np.sum(hit))
                clust = _connected_components(hit, connectivity=4)
                return area, clust, float(thr)

            base_area, base_clust, base_thr = _hygiene(score_base)
            pdlog_area = pdlog_clust = None
            pdlog_thr = None
            if score_base_pdlog is not None:
                pdlog_area, pdlog_clust, pdlog_thr = _hygiene(score_base_pdlog)
            kasai_area = kasai_clust = None
            kasai_thr = None
            if score_base_kasai is not None:
                kasai_area, kasai_clust, kasai_thr = _hygiene(score_base_kasai)

            # Optional stability proxy: split the window into two halves and correlate
            # baseline score maps under common B-mode alignment.
            corr_split_base = corr_split_pdlog = corr_split_kasai = None
            if bool(args.stability_split):
                # Write two additional "baseline-only" bundles for auditability.
                # Keep them small and do not run STAP regardless of args.run_stap.
                for part, cube_part in [("a", Icube_a), ("b", Icube_b)]:
                    ds_part = f"{dataset_name}_split{part}"
                    write_acceptance_bundle_from_icube(
                        out_root=Path(args.out_root),
                        dataset_name=ds_part,
                        Icube=cube_part,
                        prf_hz=float(prf_hz),
                        tile_hw=(int(args.tile_h), int(args.tile_w)),
                        tile_stride=int(args.tile_stride),
                        Lt=int(args.Lt),
                        baseline_type="mc_svd",
                        reg_enable=bool(args.reg_enable),
                        reg_subpixel=max(1, int(args.reg_subpixel)),
                        svd_rank=None,
                        svd_energy_frac=float(e),
                        run_stap=False,
                        band_ratio_flow_low_hz=float(flow_low_hz),
                        band_ratio_flow_high_hz=float(flow_high_hz),
                        band_ratio_alias_center_hz=float(alias_center_hz),
                        band_ratio_alias_width_hz=float(alias_hw_hz),
                        score_ka_v2_enable=False,
                        meta_extra={
                            **meta_extra,
                            "baseline_sweep": {**meta_extra["baseline_sweep"], "split_part": part},
                        },
                    )

                a_dir = Path(args.out_root) / f"{dataset_name}_splita"
                b_dir = Path(args.out_root) / f"{dataset_name}_splitb"
                base_a = np.load(a_dir / "score_base.npy").astype(np.float32, copy=False)
                base_b = np.load(b_dir / "score_base.npy").astype(np.float32, copy=False)
                base_b_al = ndi.shift(base_b, shift=(float(dybm), float(dxbm)), order=1, mode="nearest", prefilter=False)

                def _corr(x: np.ndarray, y: np.ndarray) -> float:
                    x = np.asarray(x, dtype=np.float64).ravel()
                    y = np.asarray(y, dtype=np.float64).ravel()
                    finite = np.isfinite(x) & np.isfinite(y)
                    x = x[finite]
                    y = y[finite]
                    if x.size == 0:
                        return float("nan")
                    x0 = x - float(np.mean(x))
                    y0 = y - float(np.mean(y))
                    denom = float(np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))) + 1e-12
                    return float(np.sum(x0 * y0) / denom)

                corr_split_base = _corr(base_a, base_b_al)

                try:
                    pdlog_a = np.load(a_dir / "score_base_pdlog.npy").astype(np.float32, copy=False)
                    pdlog_b = np.load(b_dir / "score_base_pdlog.npy").astype(np.float32, copy=False)
                    pdlog_b_al = ndi.shift(
                        pdlog_b, shift=(float(dybm), float(dxbm)), order=1, mode="nearest", prefilter=False
                    )
                    corr_split_pdlog = _corr(pdlog_a, pdlog_b_al)
                except Exception:
                    corr_split_pdlog = None
                try:
                    kasai_a = np.load(a_dir / "score_base_kasai.npy").astype(np.float32, copy=False)
                    kasai_b = np.load(b_dir / "score_base_kasai.npy").astype(np.float32, copy=False)
                    kasai_b_al = ndi.shift(
                        kasai_b, shift=(float(dybm), float(dxbm)), order=1, mode="nearest", prefilter=False
                    )
                    corr_split_kasai = _corr(kasai_a, kasai_b_al)
                except Exception:
                    corr_split_kasai = None

            rows.append(
                {
                    "bundle": str(bundle_dir),
                    "block_id": int(block_id),
                    "frames": args.frames,
                    "prf_hz": float(prf_hz),
                    "profile": str(args.profile),
                    "svd_energy_frac": float(e),
                    "run_stap": bool(args.run_stap),
                    "reg_enable": bool(args.reg_enable),
                    "reg_subpixel": int(args.reg_subpixel),
                    "Lt": int(args.Lt),
                    "tile_hw": f"{int(args.tile_h)}x{int(args.tile_w)}",
                    "tile_stride": int(args.tile_stride),
                    "n_flow_px": int(np.sum(mask_flow)),
                    "n_bg_px": int(np.sum(mask_bg)),
                    "flow_area_frac": float(np.mean(mask_flow)),
                    "bg_area_frac": float(np.mean(mask_bg)),
                    "ka_state": ka_v2.get("state") if isinstance(ka_v2, dict) else None,
                    "ka_reason": ka_v2.get("reason") if isinstance(ka_v2, dict) else None,
                    "ka_pf_peak_flow": ka_metrics.get("pf_peak_flow"),
                    "ka_guard_q90": ka_metrics.get("guard_q90"),
                    "ka_iqr_alias_bg": ka_metrics.get("iqr_alias_bg"),
                    "ka_delta_bg_flow_median": ka_metrics.get("delta_bg_flow_median"),
                    "ka_delta_tail": ka_metrics.get("delta_tail"),
                    "ka_p_shrink": ka_metrics.get("p_shrink"),
                    "ka_uplift_eligible": ka_metrics.get("uplift_eligible"),
                    "hyg_base_bg_tail_area": int(base_area),
                    "hyg_base_bg_tail_clusters": int(base_clust),
                    "hyg_base_bg_tail_thr": float(base_thr),
                    "hyg_pdlog_bg_tail_area": int(pdlog_area) if pdlog_area is not None else None,
                    "hyg_pdlog_bg_tail_clusters": int(pdlog_clust) if pdlog_clust is not None else None,
                    "hyg_pdlog_bg_tail_thr": float(pdlog_thr) if pdlog_thr is not None else None,
                    "hyg_kasai_bg_tail_area": int(kasai_area) if kasai_area is not None else None,
                    "hyg_kasai_bg_tail_clusters": int(kasai_clust) if kasai_clust is not None else None,
                    "hyg_kasai_bg_tail_thr": float(kasai_thr) if kasai_thr is not None else None,
                    "std_base": float(np.std(score_base[np.isfinite(score_base)])),
                    "ent_base": float(_shannon_entropy(score_base)),
                    "std_pdlog": float(np.std(score_base_pdlog[np.isfinite(score_base_pdlog)]))
                    if score_base_pdlog is not None
                    else None,
                    "ent_pdlog": float(_shannon_entropy(score_base_pdlog)) if score_base_pdlog is not None else None,
                    "std_kasai": float(np.std(score_base_kasai[np.isfinite(score_base_kasai)]))
                    if score_base_kasai is not None
                    else None,
                    "ent_kasai": float(_shannon_entropy(score_base_kasai)) if score_base_kasai is not None else None,
                    "stability_split_corr_base": float(corr_split_base) if corr_split_base is not None else None,
                    "stability_split_corr_pdlog": float(corr_split_pdlog) if corr_split_pdlog is not None else None,
                    "stability_split_corr_kasai": float(corr_split_kasai) if corr_split_kasai is not None else None,
                    "stability_split_bmode_psr": float(psrbm),
                    "stability_split_bmode_dy": float(dybm),
                    "stability_split_bmode_dx": float(dxbm),
                }
            )
            print(
                "[ulm-baseline-sweep]"
                f" block={int(block_id):03d}"
                f" e={float(e):.3f}"
                f" ka={rows[-1]['ka_state']}({rows[-1]['ka_reason']})"
                f" pf_peak_flow={rows[-1]['ka_pf_peak_flow']}"
                f" delta_tail={rows[-1]['ka_delta_tail']}"
                f" corr_split(kasai)={rows[-1]['stability_split_corr_kasai']}"
            )

    if not rows:
        raise RuntimeError("No rows produced.")

    # Write CSV
    fieldnames = list(rows[0].keys())
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[ulm-baseline-sweep] wrote {len(rows)} rows to {args.out_csv}")

    # Aggregate by energy fraction and pick a deterministic recommendation.
    # Primary objective: maximize median delta_tail (actionability); tie-break on
    # delta_bg_flow_median and pf_peak_flow; require finite iqr_alias_bg.
    by_e: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_e.setdefault(float(r["svd_energy_frac"]), []).append(r)

    def _median(vals: list[float]) -> float:
        vals = [v for v in vals if math.isfinite(float(v))]
        if not vals:
            return float("nan")
        return float(np.median(np.asarray(vals, dtype=np.float64)))

    agg: list[dict[str, Any]] = []
    for e, rs in sorted(by_e.items(), key=lambda t: t[0]):
        agg.append(
            {
                "svd_energy_frac": float(e),
                "median_delta_tail": _median([float(r["ka_delta_tail"]) for r in rs if r["ka_delta_tail"] is not None]),
                "median_delta_bg_flow_median": _median(
                    [float(r["ka_delta_bg_flow_median"]) for r in rs if r["ka_delta_bg_flow_median"] is not None]
                ),
                "median_pf_peak_flow": _median([float(r["ka_pf_peak_flow"]) for r in rs if r["ka_pf_peak_flow"] is not None]),
                "median_iqr_alias_bg": _median([float(r["ka_iqr_alias_bg"]) for r in rs if r["ka_iqr_alias_bg"] is not None]),
                "median_guard_q90": _median([float(r["ka_guard_q90"]) for r in rs if r["ka_guard_q90"] is not None]),
                "median_corr_split_kasai": _median(
                    [float(r["stability_split_corr_kasai"]) for r in rs if r["stability_split_corr_kasai"] is not None]
                ),
                "median_corr_split_pdlog": _median(
                    [float(r["stability_split_corr_pdlog"]) for r in rs if r["stability_split_corr_pdlog"] is not None]
                ),
            }
        )

    # Pick best by lexicographic objective under a minimal integrity floor.
    candidates = [
        a for a in agg if math.isfinite(float(a["median_iqr_alias_bg"])) and float(a["median_iqr_alias_bg"]) > 0.05
    ]
    if not candidates:
        candidates = agg
    candidates_sorted = sorted(
        candidates,
        key=lambda a: (
            float(a["median_delta_tail"]) if math.isfinite(float(a["median_delta_tail"])) else -1e9,
            float(a["median_delta_bg_flow_median"]) if math.isfinite(float(a["median_delta_bg_flow_median"])) else -1e9,
            float(a["median_pf_peak_flow"]) if math.isfinite(float(a["median_pf_peak_flow"])) else -1e9,
        ),
        reverse=True,
    )
    recommended = candidates_sorted[0] if candidates_sorted else None

    summary = {
        "config": {
            **{k: v for k, v in vars(args).items() if k not in {"out_root", "out_csv", "out_json", "cache_dir"}},
            "out_root": str(args.out_root),
            "out_csv": str(args.out_csv),
            "out_json": str(args.out_json),
            "cache_dir": str(args.cache_dir),
            "bands": {
                "flow_low_hz": float(flow_low_hz),
                "flow_high_hz": float(flow_high_hz),
                "alias_center_hz": float(alias_center_hz),
                "alias_half_width_hz": float(alias_hw_hz),
            },
        },
        "rows": rows,
        "aggregate_by_energy": agg,
        "recommended": recommended,
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"[ulm-baseline-sweep] wrote {args.out_json}")
    if recommended is not None:
        print(f"[ulm-baseline-sweep] recommended svd_energy_frac={recommended['svd_energy_frac']}")


if __name__ == "__main__":
    main()

