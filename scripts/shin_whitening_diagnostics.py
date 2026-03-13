#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> tuple[list[int] | None, str]:
    spec = (spec or "").strip()
    if spec in {"", "all", ":", "0:"}:
        return None, "all"
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]' or 'all'.")
    start = int(parts[0]) if parts[0] else 0
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
    if stop is None:
        raise ValueError("Slice spec must include stop (e.g. 0:128).")
    frames = list(range(start, stop, step))
    tag = f"f{start}_{stop}"
    return frames, tag


def _parse_list(spec: str) -> list[str]:
    vals = [s.strip() for s in (spec or "").split(",") if s.strip()]
    if not vals:
        raise ValueError("Expected at least one value.")
    return vals


def _parse_int_list(spec: str) -> list[int]:
    return [int(v) for v in _parse_list(spec)]


def _profile_to_bands(profile: str) -> tuple[float, float, float, float]:
    p_raw = (profile or "").strip()
    p = p_raw.lower()
    if p.startswith("band:"):
        parts = p_raw.split(":")
        if len(parts) != 5:
            raise ValueError(
                f"Custom band spec {profile!r} must be 'band:flow_low:flow_high:alias_center:alias_hw'."
            )
        try:
            return tuple(float(v) for v in parts[1:5])  # type: ignore[return-value]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric custom band spec {profile!r}.") from exc
    if p in {"u", "shin_u", "ulm"}:
        return 60.0, 250.0, 400.0, 100.0
    if p in {"s", "shin_s", "strict"}:
        return 20.0, 200.0, 380.0, 120.0
    if p in {"l", "shin_l", "low"}:
        return 10.0, 120.0, 330.0, 170.0
    if p in {"vl", "verylow"}:
        return 5.0, 80.0, 260.0, 220.0
    if p in {"m10_80", "10_80", "midlow"}:
        return 10.0, 80.0, 260.0, 220.0
    if p in {"m15_80", "15_80"}:
        return 15.0, 80.0, 260.0, 220.0
    if p in {"m5_60", "5_60"}:
        return 5.0, 60.0, 240.0, 240.0
    raise ValueError(
        f"Unknown profile {profile!r}. Expected one of U,S,L,VL,M10_80,M15_80,M5_60 or "
        "'band:flow_low:flow_high:alias_center:alias_hw'."
    )


def _default_subset5() -> list[str]:
    return [
        "IQData001.dat",
        "IQData002.dat",
        "IQData003.dat",
        "IQData004.dat",
        "IQData005.dat",
    ]


def _all_iq_files(data_root: Path) -> list[str]:
    files = sorted(p.name for p in data_root.glob("IQData*.dat") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No IQData*.dat files found under {data_root}")
    return files


def _safe_get(d: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    return x[np.isfinite(x)]


def _auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float | None:
    pos = _finite(pos)
    neg = _finite(neg)
    if pos.size == 0 or neg.size == 0:
        return None
    neg_sorted = np.sort(neg)
    less = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    equal = right - less
    return float((float(np.sum(less)) + 0.5 * float(np.sum(equal))) / float(pos.size * neg.size))


def _threshold_at_fpr(score: np.ndarray, neg_mask: np.ndarray, fpr: float = 1e-3) -> float | None:
    neg = _finite(np.asarray(score, dtype=np.float64)[np.asarray(neg_mask, dtype=bool)])
    if neg.size == 0:
        return None
    return float(np.quantile(neg, 1.0 - float(fpr)))


def _score_diag(score: np.ndarray, mask_flow: np.ndarray, mask_bg: np.ndarray) -> dict[str, float | None]:
    score = np.asarray(score, dtype=np.float64)
    flow = np.asarray(mask_flow, dtype=bool)
    bg = np.asarray(mask_bg, dtype=bool)
    flow_vals = _finite(score[flow])
    bg_vals = _finite(score[bg])
    out: dict[str, float | None] = {
        "auc_flow_bg": _auc_pos_vs_neg(flow_vals, bg_vals),
        "flow_q50": float(np.quantile(flow_vals, 0.50)) if flow_vals.size else None,
        "flow_q90": float(np.quantile(flow_vals, 0.90)) if flow_vals.size else None,
        "bg_q999": float(np.quantile(bg_vals, 0.999)) if bg_vals.size else None,
    }
    thr = _threshold_at_fpr(score, bg, 1e-3)
    out["thr_fpr1e3"] = thr
    if thr is not None and flow_vals.size:
        out["flow_hit_fpr1e3"] = float(np.mean(flow_vals >= thr))
        out["flow_margin_q50_fpr1e3"] = (
            None if out["flow_q50"] is None else float(out["flow_q50"]) - float(thr)
        )
        out["flow_margin_q90_fpr1e3"] = (
            None if out["flow_q90"] is None else float(out["flow_q90"]) - float(thr)
        )
    else:
        out["flow_hit_fpr1e3"] = None
        out["flow_margin_q50_fpr1e3"] = None
        out["flow_margin_q90_fpr1e3"] = None
    return out


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['profile']}_Lt{row['Lt']}_{row['detector_variant']}"
        groups.setdefault(key, []).append(row)

    def med(key: str, arr: list[dict[str, Any]]) -> float | None:
        vals = [float(r[key]) for r in arr if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    out: dict[str, Any] = {}
    for key, arr in groups.items():
        out[key] = {
            "n_bundles": len(arr),
            "median_auc_stap": med("auc_flow_bg_stap", arr),
            "median_delta_auc_stap_minus_pd": med("delta_auc_stap_minus_pd", arr),
            "median_delta_auc_stap_minus_kasai": med("delta_auc_stap_minus_kasai", arr),
            "median_flow_hit_stap_1e3": med("flow_hit_fpr1e3_stap", arr),
            "median_flow_margin_stap_q50_1e3": med("flow_margin_q50_fpr1e3_stap", arr),
            "median_peak_p50_hz": med("br_peak_p50_hz", arr),
            "median_pf_peak_nonbg": med("br_pf_peak_nonbg", arr),
            "median_alias_peak_bg": med("br_pa_peak_bg", arr),
            "median_flow_alignment": med("flow_alignment_median", arr),
            "median_condR": med("median_condR", arr),
            "median_cond_loaded": med("median_cond_loaded", arr),
            "median_cov_eff_rank": med("median_cov_eff_rank", arr),
            "median_cov_rank_proxy": med("median_cov_rank_proxy", arr),
            "median_cov_trace": med("median_cov_trace", arr),
            "median_lambda_needed": med("median_lambda_condition_needed", arr),
            "median_lambda_conditioned": med("median_lambda_conditioned", arr),
            "median_flow_mu_ratio": med("median_flow_mu_ratio", arr),
            "median_bg_var_inflation": med("median_bg_var_inflation", arr),
            "median_psd_alias_ratio": med("median_psd_alias_ratio", arr),
            "median_band_fraction_q50": med("median_band_fraction_q50", arr),
            "median_score_q50": med("median_score_q50", arr),
            "median_score_q90": med("median_score_q90", arr),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run a focused Shin whitening diagnostics sweep. "
            "Compares detector variants on the same residual and exports the "
            "band/covariance telemetry needed to explain when whitening helps or hurts."
        )
    )
    ap.add_argument("--data-root", type=Path, default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"))
    ap.add_argument("--iq-files", type=str, default="subset5")
    ap.add_argument("--frames", type=str, default="0:128")
    ap.add_argument("--prf-hz", type=float, default=1000.0)
    ap.add_argument("--profiles", type=str, default="U,S,L")
    ap.add_argument("--lts", type=str, default="32,64")
    ap.add_argument("--detector-variants", type=str, default="msd_ratio,unwhitened_ratio")
    ap.add_argument("--baseline-type", type=str, default="mc_svd")
    ap.add_argument("--svd-energy-frac", type=float, default=0.97)
    ap.add_argument("--fd-span-mode", type=str, default="flow_band")
    ap.add_argument("--tile-h", type=int, default=8)
    ap.add_argument("--tile-w", type=int, default=8)
    ap.add_argument("--tile-stride", type=int, default=3)
    ap.add_argument("--diag-load", type=float, default=0.07)
    ap.add_argument("--cov-estimator", type=str, default="tyler_pca")
    ap.add_argument("--huber-c", type=float, default=5.0)
    ap.add_argument("--stap-cov-trim-q", type=float, default=0.0)
    ap.add_argument("--mvdr-auto-kappa", type=float, default=120.0)
    ap.add_argument("--constraint-ridge", type=float, default=0.18)
    ap.add_argument("--snapshot-stride", type=int, default=4)
    ap.add_argument("--max-snapshots", type=int, default=64)
    ap.add_argument("--kappa-shrink", type=float, default=80.0)
    ap.add_argument("--kappa-msd", type=float, default=80.0)
    ap.add_argument("--stap-device", type=str, default="cuda")
    ap.add_argument("--out-root", type=Path, default=Path("runs/shin_whitening_diagnostics"))
    ap.add_argument("--out-csv", type=Path, default=Path("reports/shin_whitening_diagnostics.csv"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/shin_whitening_diagnostics.json"))
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    os.environ["STAP_FAST_PATH"] = "1"
    os.environ["STAP_SNAPSHOT_STRIDE"] = str(max(1, int(args.snapshot_stride)))
    os.environ["STAP_MAX_SNAPSHOTS"] = str(max(1, int(args.max_snapshots)))
    os.environ["STAP_KAPPA_SHRINK"] = str(float(args.kappa_shrink))
    os.environ["STAP_KAPPA_MSD"] = str(float(args.kappa_msd))

    iq_spec = str(args.iq_files).strip().lower()
    if iq_spec == "subset5":
        iq_files = _default_subset5()
    elif iq_spec == "all":
        iq_files = _all_iq_files(args.data_root)
    else:
        iq_files = _parse_list(args.iq_files)
    frames, frame_tag = _parse_slice(args.frames)
    profiles = _parse_list(args.profiles)
    lts = _parse_int_list(args.lts)
    variants = _parse_list(args.detector_variants)

    info = load_shin_metadata(args.data_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for iq_file in iq_files:
        iq_path = args.data_root / iq_file
        if not iq_path.is_file():
            print(f"[shin-diag] missing {iq_path}, skipping")
            continue
        icube = load_shin_iq(iq_path, info, frames=frames)
        for profile in profiles:
            flow_low, flow_high, alias_center, alias_hw = _profile_to_bands(profile)
            for Lt in lts:
                for variant in variants:
                    variant_norm = variant.strip().lower()
                    whiten_gamma = 0.0 if variant_norm == "unwhitened_ratio" else 1.0
                    dataset_name = (
                        f"shin_{iq_path.stem}_{frame_tag}_p{profile}_Lt{Lt}_{args.baseline_type}"
                        f"_e{float(args.svd_energy_frac):.3f}_{variant_norm}"
                    ).replace(".", "p")
                    bundle_dir = args.out_root / dataset_name
                    meta_path = bundle_dir / "meta.json"
                    if bool(args.resume) and meta_path.is_file():
                        meta = json.loads(meta_path.read_text())
                    else:
                        meta_extra = {
                            "orig_data": {
                                "dataset": "ShinRatBrain_Fig3",
                                "iq_file": str(iq_path),
                                "sizeinfo": asdict(info),
                                "frames_spec": args.frames,
                            },
                            "shin_profile": {
                                "name": profile,
                                "flow_low_hz": flow_low,
                                "flow_high_hz": flow_high,
                                "alias_center_hz": alias_center,
                                "alias_half_width_hz": alias_hw,
                            },
                        }
                        paths = write_acceptance_bundle_from_icube(
                            out_root=args.out_root,
                            dataset_name=dataset_name,
                            Icube=icube,
                            prf_hz=float(args.prf_hz),
                            tile_hw=(int(args.tile_h), int(args.tile_w)),
                            tile_stride=int(args.tile_stride),
                            Lt=int(Lt),
                            diag_load=float(args.diag_load),
                            cov_estimator=str(args.cov_estimator),
                            huber_c=float(args.huber_c),
                            stap_cov_train_trim_q=float(args.stap_cov_trim_q),
                            mvdr_auto_kappa=float(args.mvdr_auto_kappa),
                            constraint_ridge=float(args.constraint_ridge),
                            baseline_type=str(args.baseline_type),
                            svd_energy_frac=float(args.svd_energy_frac),
                            flow_mask_mode="pd_auto",
                            flow_mask_pd_quantile=0.99,
                            flow_mask_min_pixels=64,
                            flow_mask_union_default=False,
                            flow_mask_depth_min_frac=0.0,
                            flow_mask_depth_max_frac=1.0,
                            flow_mask_erode_iters=0,
                            flow_mask_dilate_iters=1,
                            band_ratio_flow_low_hz=float(flow_low),
                            band_ratio_flow_high_hz=float(flow_high),
                            band_ratio_alias_center_hz=float(alias_center),
                            band_ratio_alias_width_hz=float(alias_hw),
                            fd_span_mode=str(args.fd_span_mode),
                            stap_conditional_enable=False,
                            stap_detector_variant=str(variant_norm),
                            stap_whiten_gamma=float(whiten_gamma),
                            msd_ratio_rho=0.05,
                            reg_enable=True,
                            reg_subpixel=4,
                            reg_reference="median",
                            stap_device=str(args.stap_device),
                            score_mode="pd",
                            meta_extra=meta_extra,
                        )
                        meta = json.loads(Path(paths["meta"]).read_text())

                    score_base = np.load(bundle_dir / "score_base.npy", allow_pickle=False)
                    score_stap = np.load(bundle_dir / "score_stap_preka.npy", allow_pickle=False)
                    mask_flow = np.load(bundle_dir / "mask_flow.npy", allow_pickle=False)
                    mask_bg = np.load(bundle_dir / "mask_bg.npy", allow_pickle=False)
                    score_kasai_path = bundle_dir / "score_base_kasai.npy"
                    score_kasai = (
                        np.load(score_kasai_path, allow_pickle=False) if score_kasai_path.exists() else None
                    )

                    base_diag = _score_diag(score_base, mask_flow, mask_bg)
                    stap_diag = _score_diag(score_stap, mask_flow, mask_bg)
                    kasai_diag = (
                        _score_diag(score_kasai, mask_flow, mask_bg) if score_kasai is not None else {}
                    )

                    tele = meta.get("stap_fallback_telemetry") or {}
                    br = tele.get("band_ratio_stats") or {}
                    align = tele.get("flow_band_alignment_stats") or {}

                    row = {
                        "bundle": dataset_name,
                        "iq_file": iq_file,
                        "frames": args.frames,
                        "profile": profile,
                        "Lt": int(Lt),
                        "detector_variant": variant_norm,
                        "whiten_gamma": float(whiten_gamma),
                        "baseline_type": str(args.baseline_type),
                        "stap_device": meta.get("stap_device"),
                        "reg_shift_p90": _safe_get(meta, "baseline_stats", "reg_shift_p90"),
                        "br_pf_peak_nonbg": br.get("br_flow_peak_fraction_nonbg"),
                        "br_pa_peak_bg": br.get("br_alias_peak_fraction_bg"),
                        "br_peak_p50_hz": br.get("br_peak_freq_hz_p50"),
                        "br_peak_p90_hz": br.get("br_peak_freq_hz_p90"),
                        "flow_alignment_median": align.get("median"),
                        "flow_alignment_p90": align.get("p90"),
                        "median_f_peak_hz": tele.get("median_f_peak_hz"),
                        "median_condR": tele.get("median_condR"),
                        "median_cond_loaded": tele.get("median_cond_loaded"),
                        "median_cov_eff_rank": tele.get("median_cov_eff_rank"),
                        "median_cov_rank_proxy": tele.get("median_cov_rank_proxy"),
                        "median_cov_trace": tele.get("median_cov_trace"),
                        "median_lambda_condition_needed": tele.get("median_lambda_condition_needed"),
                        "median_lambda_conditioned": tele.get("median_lambda_conditioned"),
                        "median_band_fraction": tele.get("median_band_fraction"),
                        "median_band_fraction_q50": tele.get("median_band_fraction_q50"),
                        "median_flow_mu_ratio": tele.get("median_flow_mu_ratio"),
                        "median_bg_var_inflation": tele.get("median_bg_var_inflation"),
                        "median_psd_peak_hz": tele.get("median_psd_peak_hz"),
                        "median_psd_alias_ratio": tele.get("psd_alias_ratio_median"),
                        "psd_alias_fraction": tele.get("psd_alias_fraction"),
                        "median_score_q50": tele.get("median_score_q50"),
                        "median_score_q90": tele.get("median_score_q90"),
                        "auc_flow_bg_base": base_diag.get("auc_flow_bg"),
                        "auc_flow_bg_stap": stap_diag.get("auc_flow_bg"),
                        "delta_auc_stap_minus_pd": (
                            None
                            if base_diag.get("auc_flow_bg") is None or stap_diag.get("auc_flow_bg") is None
                            else float(stap_diag["auc_flow_bg"]) - float(base_diag["auc_flow_bg"])
                        ),
                        "thr_fpr1e3_base": base_diag.get("thr_fpr1e3"),
                        "thr_fpr1e3_stap": stap_diag.get("thr_fpr1e3"),
                        "flow_hit_fpr1e3_base": base_diag.get("flow_hit_fpr1e3"),
                        "flow_hit_fpr1e3_stap": stap_diag.get("flow_hit_fpr1e3"),
                        "flow_margin_q50_fpr1e3_base": base_diag.get("flow_margin_q50_fpr1e3"),
                        "flow_margin_q50_fpr1e3_stap": stap_diag.get("flow_margin_q50_fpr1e3"),
                        "flow_margin_q90_fpr1e3_base": base_diag.get("flow_margin_q90_fpr1e3"),
                        "flow_margin_q90_fpr1e3_stap": stap_diag.get("flow_margin_q90_fpr1e3"),
                    }
                    if kasai_diag:
                        row["auc_flow_bg_kasai"] = kasai_diag.get("auc_flow_bg")
                        row["delta_auc_stap_minus_kasai"] = (
                            None
                            if kasai_diag.get("auc_flow_bg") is None or stap_diag.get("auc_flow_bg") is None
                            else float(stap_diag["auc_flow_bg"]) - float(kasai_diag["auc_flow_bg"])
                        )
                    rows.append(row)
                    print(
                        "[shin-diag]"
                        f" {iq_file} p={profile} Lt={Lt} variant={variant_norm}"
                        f" aucΔ(pd)={row['delta_auc_stap_minus_pd']}"
                        f" peak50={row['br_peak_p50_hz']} condR={row['median_condR']}"
                        f" rank~={row['median_cov_rank_proxy']}"
                    )

    if not rows:
        raise RuntimeError("No diagnostics rows produced.")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "config": {
            "iq_files": iq_files,
            "frames": args.frames,
            "profiles": profiles,
            "lts": lts,
            "detector_variants": variants,
            "baseline_type": args.baseline_type,
            "stap_device": args.stap_device,
        },
        "summary": _summarize_rows(rows),
    }
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote", args.out_csv)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
