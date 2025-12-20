from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

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


def _parse_int_list(spec: str) -> list[int]:
    vals: list[int] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def _parse_rank_list(spec: str) -> list[int | None]:
    """Parse comma-separated SVD ranks; supports 'none' to use energy-frac auto."""
    vals: list[int | None] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() in {"none", "auto", "energy", "ef"}:
            vals.append(None)
        else:
            vals.append(int(part))
    if not vals:
        raise ValueError("Expected at least one svd rank (int) or 'none'.")
    return vals


def _profile_to_bands(profile: str) -> tuple[float, float, float, float]:
    """Return (flow_low, flow_high, alias_center, alias_half_width)."""
    p = (profile or "").strip().lower()
    if p in {"brain", "brain_default"}:
        return 30.0, 220.0, 400.0, 120.0
    if p in {"u", "shin_u", "ulm", "shin_ulm"}:
        # Shin RatBrain Fig3 (post-compounded ~1 kHz slow-time): treat Pf as a
        # higher-frequency "bubble/flow" band and Pa as a high-frequency
        # nuisance/artifact band.
        #
        # Po: [0,60], Pf: [60,250], guard: [250,300], Pa: [300,500]
        return 60.0, 250.0, 400.0, 100.0
    if p in {"s", "shin_s", "strict"}:
        # Po: [0,20], Pf: [20,200], guard: [200,260], Pa: [260,500]
        # Alias band is specified via center +/- half-width.
        return 20.0, 200.0, 380.0, 120.0
    if p in {"l", "shin_l", "low"}:
        # Po: [0,10], Pf: [10,120], guard: [120,160], Pa: [160,500]
        return 10.0, 120.0, 330.0, 170.0
    raise ValueError(f"Unknown profile {profile!r}. Use one of: brain, U, S, L.")


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep frozen Shin RatBrain IQ profiles (bands + Lt) and baseline SVD settings.\n"
            "Writes acceptance bundles and a summary CSV of telemetry + contract v2 outputs.\n"
            "This is intended for profile calibration (choose one frozen profile) rather than\n"
            "per-window retuning."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"),
        help="Directory containing SizeInfo.dat and IQData*.dat (default: %(default)s).",
    )
    parser.add_argument(
        "--iq-files",
        type=str,
        default="IQData001.dat,IQData002.dat,IQData003.dat,IQData004.dat,IQData005.dat",
        help="Comma-separated IQData file names under --data-root (default: first 5).",
    )
    parser.add_argument("--frames", type=str, default="0:128")
    parser.add_argument("--prf-hz", type=float, default=1000.0)
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)
    parser.add_argument(
        "--profiles",
        type=str,
        default="S,L",
        help="Comma-separated profile names: brain,S,L (default: %(default)s).",
    )
    parser.add_argument(
        "--lts",
        type=str,
        default="8,64",
        help="Comma-separated Lt values to try (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="mc_svd",
        help="Baseline type: mc_svd or svd_bandpass (default: %(default)s).",
    )
    parser.add_argument(
        "--svd-ranks",
        type=str,
        default="8,16,32,48",
        help="For mc_svd: comma-separated ranks to remove (default: %(default)s).",
    )
    parser.add_argument(
        "--svd-energy-frac",
        type=float,
        default=0.95,
        help="For mc_svd: energy fraction removed when svd_rank is None (default: %(default)s).",
    )
    parser.add_argument("--svd-keep-min", type=int, default=3, help="For svd_bandpass: 1-based keep min.")
    parser.add_argument("--svd-keep-max", type=int, default=40, help="For svd_bandpass: 1-based keep max.")
    parser.add_argument("--flow-mask-pd-quantile", type=float, default=0.99)
    parser.add_argument("--flow-mask-min-pixels", type=int, default=64)
    parser.add_argument(
        "--flow-mask-union-default",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--flow-mask-depth-min-frac", type=float, default=0.0)
    parser.add_argument("--flow-mask-depth-max-frac", type=float, default=1.0)
    parser.add_argument("--flow-mask-erode-iters", type=int, default=0)
    parser.add_argument("--flow-mask-dilate-iters", type=int, default=1)
    parser.add_argument(
        "--run-stap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run STAP when writing bundles (default: %(default)s).",
    )
    parser.add_argument(
        "--score-ka-v2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable score-space KA v2 (shrink-only) when contract permits (default: %(default)s).",
    )
    parser.add_argument(
        "--score-ka-v2-mode",
        type=str,
        default="safety",
        help="KA v2 application mode: safety|uplift|auto (default: %(default)s).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/shin_ratbrain_sweep"),
        help="Output root for acceptance bundles (default: %(default)s).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("reports/shin_ratbrain_sweep.csv"),
        help="Output CSV path (default: %(default)s).",
    )
    args = parser.parse_args()

    iq_files = [f.strip() for f in args.iq_files.split(",") if f.strip()]
    if not iq_files:
        raise ValueError("No IQ files provided.")

    frames, frame_tag = _parse_slice(args.frames)
    info = load_shin_metadata(args.data_root)
    lts = _parse_int_list(args.lts)
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    svd_ranks = _parse_rank_list(args.svd_ranks)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for iq_file in iq_files:
        iq_path = args.data_root / iq_file
        if not iq_path.is_file():
            print(f"[shin-sweep] missing {iq_path}, skipping")
            continue
        Icube = load_shin_iq(iq_path, info, frames=frames)
        for profile in profiles:
            flow_low, flow_high, alias_center, alias_hw = _profile_to_bands(profile)
            for Lt in lts:
                if args.baseline_type.strip().lower() == "mc_svd":
                    rank_list = svd_ranks
                    keep_min = None
                    keep_max = None
                else:
                    rank_list = [None]
                    keep_min = int(args.svd_keep_min)
                    keep_max = int(args.svd_keep_max)
                for svd_rank in rank_list:
                    dataset_name = (
                        f"shin_{iq_path.stem}_{frame_tag}"
                        f"_p{profile}_Lt{Lt}"
                        f"_{args.baseline_type}"
                    )
                    if svd_rank is not None:
                        dataset_name += f"_r{int(svd_rank)}"
                    elif args.baseline_type.strip().lower() == "mc_svd":
                        dataset_name += f"_e{float(args.svd_energy_frac):.3f}"
                    else:
                        dataset_name += f"_k{keep_min}_{keep_max}"

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
                        Icube=Icube,
                        prf_hz=float(args.prf_hz),
                        tile_hw=(int(args.tile_h), int(args.tile_w)),
                        tile_stride=int(args.tile_stride),
                        Lt=int(Lt),
                        baseline_type=str(args.baseline_type),
                        svd_rank=int(svd_rank) if svd_rank is not None else None,
                        svd_energy_frac=float(args.svd_energy_frac),
                        svd_keep_min=keep_min,
                        svd_keep_max=keep_max,
                        flow_mask_pd_quantile=float(args.flow_mask_pd_quantile),
                        flow_mask_min_pixels=int(args.flow_mask_min_pixels),
                        flow_mask_union_default=bool(args.flow_mask_union_default),
                        flow_mask_depth_min_frac=float(args.flow_mask_depth_min_frac),
                        flow_mask_depth_max_frac=float(args.flow_mask_depth_max_frac),
                        flow_mask_erode_iters=int(args.flow_mask_erode_iters),
                        flow_mask_dilate_iters=int(args.flow_mask_dilate_iters),
                        band_ratio_flow_low_hz=float(flow_low),
                        band_ratio_flow_high_hz=float(flow_high),
                        band_ratio_alias_center_hz=float(alias_center),
                        band_ratio_alias_width_hz=float(alias_hw),
                        score_ka_v2_enable=bool(args.score_ka_v2),
                        score_ka_v2_mode=str(args.score_ka_v2_mode),
                        run_stap=bool(args.run_stap),
                        cov_estimator="tyler_pca",
                        score_mode="pd",
                        reg_enable=False,
                        meta_extra=meta_extra,
                    )

                    meta = json.loads(Path(paths["meta"]).read_text())
                    tele = meta.get("stap_fallback_telemetry") or {}
                    br = tele.get("band_ratio_stats") or {}
                    ka = meta.get("ka_contract_v2") or {}
                    ka_metrics = ka.get("metrics") or {}
                    flow_stats = tele.get("flow_mask_stats") or {}
                    try:
                        mask_flow_pixels = int(np.load(paths["mask_flow"]).sum())
                        mask_bg_pixels = int(np.load(paths["mask_bg"]).sum())
                    except Exception:
                        mask_flow_pixels = None
                        mask_bg_pixels = None
                    row = {
                        "bundle": dataset_name,
                        "iq_file": iq_file,
                        "frames": args.frames,
                        "prf_hz": float(args.prf_hz),
                        "profile": profile,
                        "Lt": int(Lt),
                        "baseline_type": str(args.baseline_type),
                        "svd_rank": int(svd_rank) if svd_rank is not None else None,
                        "svd_keep_min": keep_min,
                        "svd_keep_max": keep_max,
                        "flow_low_hz": flow_low,
                        "flow_high_hz": flow_high,
                        "alias_center_hz": alias_center,
                        "alias_hw_hz": alias_hw,
                        "mask_flow_pixels": mask_flow_pixels,
                        "mask_bg_pixels": mask_bg_pixels,
                        "flow_mask_coverage_post": flow_stats.get("coverage_post_union"),
                        "flow_mask_pd_auto_used": flow_stats.get("pd_auto_used"),
                        "flow_mask_pd_auto_reason": flow_stats.get("pd_auto_reason"),
                        "flow_mask_pd_quantile": float(args.flow_mask_pd_quantile),
                        "br_pf_peak_nonbg": br.get("br_flow_peak_fraction_nonbg"),
                        "br_pa_peak_bg": br.get("br_alias_peak_fraction_bg"),
                        "br_peak_p50_hz": br.get("br_peak_freq_hz_p50"),
                        "br_peak_p90_hz": br.get("br_peak_freq_hz_p90"),
                        "ka_state": ka.get("state"),
                        "ka_reason": ka.get("reason"),
                        "ka_guard_q90": ka_metrics.get("guard_q90"),
                        "ka_iqr_alias_bg": ka_metrics.get("iqr_alias_bg"),
                        "ka_delta_bg_flow": ka_metrics.get("delta_bg_flow_median"),
                        "ka_delta_tail": ka_metrics.get("delta_tail"),
                        "ka_p_shrink": ka_metrics.get("p_shrink"),
                        "ka_uplift_eligible": ka_metrics.get("uplift_eligible"),
                        "ka_pf_peak_nonbg": ka_metrics.get("pf_peak_nonbg"),
                        "ka_pf_peak_flow": ka_metrics.get("pf_peak_flow"),
                        "ka_n_flow_proxy": ka_metrics.get("n_flow_proxy"),
                        "ka_uplift_vetoed_by_pf_peak": ka_metrics.get("uplift_vetoed_by_pf_peak"),
                        "ka_uplift_veto_pf_peak_reason": ka_metrics.get("uplift_veto_pf_peak_reason"),
                        "score_ka_v2_applied": tele.get("score_ka_v2_applied"),
                        "score_ka_v2_mode_applied": tele.get("score_ka_v2_mode_applied"),
                        "score_ka_v2_risk_mode": tele.get("score_ka_v2_risk_mode"),
                        "score_ka_v2_disabled_reason": tele.get("score_ka_v2_disabled_reason"),
                        "score_ka_v2_scaled_px_frac": tele.get("score_ka_v2_scaled_pixel_fraction"),
                        "score_ka_v2_scale_p90": tele.get("score_ka_v2_scale_p90"),
                        "score_ka_v2_scale_max": tele.get("score_ka_v2_scale_max"),
                        "stap_ms": tele.get("stap_ms"),
                    }
                    rows.append(row)
                    print(
                        "[shin-sweep]"
                        f" {iq_file} p={profile} Lt={Lt}"
                        f" baseline={args.baseline_type}"
                        f" svd={row['svd_rank'] or f'{keep_min}-{keep_max}'}"
                        f" ka={row['ka_state']}({row['ka_reason']})"
                        f" pf_peak={row['br_pf_peak_nonbg']} peak_p50={row['br_peak_p50_hz']}"
                    )

    if not rows:
        raise RuntimeError("No sweep results produced.")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[shin-sweep] wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
