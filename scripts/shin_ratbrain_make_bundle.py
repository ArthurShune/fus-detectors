from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pipeline.realdata.shin_ratbrain import load_shin_iq, load_shin_metadata
from sim.kwave.icube_bundle import write_acceptance_bundle_from_icube


def _parse_slice(spec: str) -> slice | None:
    spec = (spec or "").strip()
    if spec in {"", "all", ":", "0:"}:
        return None
    parts = spec.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid slice spec {spec!r}; expected 'start:stop[:step]' or 'all'.")
    start = int(parts[0]) if parts[0] else None
    stop = int(parts[1]) if parts[1] else None
    step = int(parts[2]) if len(parts) == 3 and parts[2] else None
    return slice(start, stop, step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an acceptance bundle from the Shin RatBrain Fig3 IQData*.dat files.\n"
            "This converts one IQData file (beamformed complex IQ) into the same bundle\n"
            "format used by k-Wave runs so you can run scripts/hab_contract_check.py."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/shin_zenodo_10711806/ratbrain_fig3_raw"),
        help="Directory containing SizeInfo.dat and IQData*.dat (default: %(default)s).",
    )
    parser.add_argument(
        "--iq-file",
        type=str,
        default="IQData001.dat",
        help="IQData file name under --data-root (default: %(default)s).",
    )
    parser.add_argument(
        "--frames",
        type=str,
        default="all",
        help="Frame slice as start:stop[:step] (default: all). Example: 0:128",
    )
    parser.add_argument(
        "--prf-hz",
        type=float,
        default=1000.0,
        help="Slow-time sampling rate in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/shin_ratbrain"),
        help="Output root directory for acceptance bundles (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="",
        help="Optional explicit bundle directory name (default: derived from iq-file and frames).",
    )
    parser.add_argument("--tile-h", type=int, default=8)
    parser.add_argument("--tile-w", type=int, default=8)
    parser.add_argument("--tile-stride", type=int, default=3)
    parser.add_argument("--lt", type=int, default=8, help="STAP aperture Lt (default: %(default)s).")
    parser.add_argument(
        "--run-stap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to run STAP (default: %(default)s). Use --no-run-stap for telemetry-only bundles.",
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
    parser.add_argument("--diag-load", type=float, default=0.07)
    parser.add_argument(
        "--cov-estimator",
        type=str,
        default="tyler_pca",
        help="Robust covariance estimator (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline-type",
        type=str,
        default="mc_svd",
        help="Baseline clutter suppression type (default: %(default)s).",
    )
    parser.add_argument(
        "--svd-rank",
        type=int,
        default=None,
        help="Fixed number of slow-time SVD components to remove (default: auto via --svd-energy-frac).",
    )
    parser.add_argument(
        "--svd-energy-frac",
        type=float,
        default=0.95,
        help="MC-SVD baseline energy fraction removed (default: %(default)s).",
    )
    parser.add_argument(
        "--svd-keep-min",
        type=int,
        default=None,
        help="For baseline-type svd_bandpass: 1-based min singular component index to keep.",
    )
    parser.add_argument(
        "--svd-keep-max",
        type=int,
        default=None,
        help="For baseline-type svd_bandpass: 1-based max singular component index to keep (default: keep to T).",
    )
    parser.add_argument(
        "--flow-mask-mode",
        type=str,
        default="pd_auto",
        help="Flow mask mode (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-pd-quantile",
        type=float,
        default=0.99,
        help="PD quantile used to derive flow mask in pd_auto mode (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-min-pixels",
        type=int,
        default=64,
        help="Minimum pixels required for pd_auto mask (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-union-default",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to union pd_auto mask with default mask (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-depth-min-frac",
        type=float,
        default=0.0,
        help="Min depth fraction for pd_auto mask (default: %(default)s).",
    )
    parser.add_argument(
        "--flow-mask-depth-max-frac",
        type=float,
        default=1.0,
        help="Max depth fraction for pd_auto mask (default: %(default)s).",
    )
    parser.add_argument("--flow-mask-erode-iters", type=int, default=0)
    parser.add_argument("--flow-mask-dilate-iters", type=int, default=1)
    parser.add_argument(
        "--br-flow-low",
        type=float,
        default=30.0,
        help="Band-ratio telemetry flow low cutoff in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--br-flow-high",
        type=float,
        default=220.0,
        help="Band-ratio telemetry flow high cutoff in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--br-alias-center",
        type=float,
        default=400.0,
        help="Band-ratio telemetry alias band center in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--br-alias-width",
        type=float,
        default=120.0,
        help="Band-ratio telemetry alias half-width in Hz (default: %(default)s).",
    )
    parser.add_argument(
        "--stap-device",
        type=str,
        default=None,
        help="STAP device: cpu, cuda, cuda:0, ... (default: auto).",
    )
    args = parser.parse_args()

    data_root = args.data_root
    iq_path = data_root / args.iq_file
    if not iq_path.is_file():
        raise FileNotFoundError(f"IQ file not found: {iq_path}")

    info = load_shin_metadata(data_root)
    frame_slice = _parse_slice(args.frames)
    if frame_slice is None:
        frames = None
        frame_tag = f"T{info.frames}"
    else:
        start = frame_slice.start or 0
        stop = frame_slice.stop or info.frames
        frame_tag = f"f{start}_{stop}"
        frames = list(range(start, stop, frame_slice.step or 1))

    Icube = load_shin_iq(iq_path, info, frames=frames)
    th, tw = int(args.tile_h), int(args.tile_w)
    dataset_name = args.dataset_name.strip()
    if not dataset_name:
        dataset_name = f"shin_fig3_{iq_path.stem}_{frame_tag}"

    meta_extra = {
        "orig_data": {
            "dataset": "ShinRatBrain_Fig3",
            "iq_file": str(iq_path),
            "sizeinfo": asdict(info),
            "frames_spec": args.frames,
        }
    }

    paths = write_acceptance_bundle_from_icube(
        out_root=args.out_root,
        dataset_name=dataset_name,
        Icube=Icube,
        prf_hz=float(args.prf_hz),
        tile_hw=(th, tw),
        tile_stride=int(args.tile_stride),
        Lt=int(args.lt),
        diag_load=float(args.diag_load),
        cov_estimator=str(args.cov_estimator),
        baseline_type=str(args.baseline_type),
        svd_rank=args.svd_rank,
        svd_energy_frac=float(args.svd_energy_frac),
        svd_keep_min=args.svd_keep_min,
        svd_keep_max=args.svd_keep_max,
        flow_mask_mode=str(args.flow_mask_mode),
        flow_mask_pd_quantile=float(args.flow_mask_pd_quantile),
        flow_mask_min_pixels=int(args.flow_mask_min_pixels),
        flow_mask_union_default=bool(args.flow_mask_union_default),
        flow_mask_depth_min_frac=float(args.flow_mask_depth_min_frac),
        flow_mask_depth_max_frac=float(args.flow_mask_depth_max_frac),
        flow_mask_erode_iters=int(args.flow_mask_erode_iters),
        flow_mask_dilate_iters=int(args.flow_mask_dilate_iters),
        band_ratio_flow_low_hz=float(args.br_flow_low),
        band_ratio_flow_high_hz=float(args.br_flow_high),
        band_ratio_alias_center_hz=float(args.br_alias_center),
        band_ratio_alias_width_hz=float(args.br_alias_width),
        score_ka_v2_enable=bool(args.score_ka_v2),
        score_ka_v2_mode=str(args.score_ka_v2_mode),
        run_stap=bool(args.run_stap),
        stap_device=args.stap_device,
        meta_extra=meta_extra,
    )

    meta_path = Path(paths["meta"])
    print(f"[shin-bundle] wrote bundle: {meta_path.parent}")
    print(f"[shin-bundle] meta: {meta_path}")
    print("[shin-bundle] files:")
    for key in sorted(paths.keys()):
        print(f"  {key}: {paths[key]}")

    # Also print a compact one-line summary for convenience.
    meta = json.loads(meta_path.read_text())
    tele = meta.get("stap_fallback_telemetry") or {}
    print(
        "[shin-bundle] summary:"
        f" prf_hz={meta.get('prf_hz')}"
        f" Lt={meta.get('Lt')}"
        f" tiles={tele.get('tile_count')}"
        f" ka_state={tele.get('ka_contract_v2_state')}"
        f" ka_reason={tele.get('ka_contract_v2_reason')}"
    )


if __name__ == "__main__":
    main()
